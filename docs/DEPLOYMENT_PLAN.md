# Deployment plan: AWS Lambda (the API) and SageMaker (the encoder endpoint)

**Status: a plan. Nothing in it has been executed.** Every file it refers to has been written and the parts that can run on a laptop have been tested (see "What is already verified"), but no Docker image has been built from these Dockerfiles, nothing has been deployed to AWS, and some details (marked **unverified**) may need a small fix when you first run them. Do the phases in order and stop whenever you run out of time: each phase leaves you in a safe, honest state.

How to use this file: work from top to bottom, tick the boxes, fill in the progress log at the end, and when something fails copy the **complete error text** (never any keys) to whoever is helping you.

---

## 0. The shape of the work

| Phase | What you get | Needs AWS? | Costs money? | Stop here and you can honestly say |
|---|---|---|---|---|
| 0. Setup | Tools, account, budget alarm, safe credentials | account only | no | nothing new |
| 1. Local containers | Both Docker images built and tested on your Mac | no | no | "containerised with Docker, verified locally" |
| 2. Lambda | The API running at a public URL, protected by a key | yes | cents | "deployed to AWS Lambda" |
| 3. SageMaker | Embedding and reranking served from a serverless endpoint | yes | cents if torn down | "served models from a SageMaker serverless endpoint" |
| 4. Wrap-up | Numbers recorded, docs and resume wording updated, everything deleted | yes | no | the exact claims in section 6 |

Phase 2 and Phase 3 are independent: do either one, or both. If you have only a couple of hours in total, do Phase 0 and Phase 1 and push to GitHub so the CI workflow runs.

### Costs: how to stay safe
- I do not know current AWS prices; use the AWS pricing pages and your own Budget alert instead of any number written here.
- Both designs are **pay per request**: Lambda and SageMaker *serverless* inference cost almost nothing while idle. **Never** create a SageMaker *real-time* endpoint for this project: it bills for instance hours until you delete it.
- What can still cost money while idle: the container image in ECR (storage), CloudWatch logs (small), and anything you forget to delete. Section 7 lists everything to delete.
- The Anthropic API also costs money per request. Each `/research` call costs roughly a cent or two with the default system; the API has a spending limit in the Anthropic console that you control.
- The public Lambda URL spends your Anthropic credits if someone else calls it. The app requires an `x-api-key` header, the template adds a traffic alarm and a budget alert, and you can delete the stack at any time. Do not share the URL or the key.

### What is already verified (on a laptop, no AWS)
- 18 automated tests pass: API auth and validation, citation handling, the SageMaker container contract (`/ping`, `/invocations`) and the client.
- Run locally with the real models, the encoder service returned bit-identical embeddings and reranker scores compared with the local models, and identical top-5 results for 12 of 12 queries.
- The Linux dependency lists (`requirements-runtime.txt`, `sagemaker/requirements.txt`) resolve for both `aarch64` and `x86_64` Linux with CPU-only PyTorch (checked with `uv pip compile`, nothing installed).

### What is NOT verified (expect to fix some of these)
- Neither Dockerfile has been built. Whether GitHub's CI builds the main image is unknown until you push.
- The Lambda Web Adapter version in the Dockerfile (`0.8.4`) was not checked online. **Unverified.**
- The SAM template has never been deployed. The IAM policy expression for reading SSM parameters in `template.yaml` is **unverified**.
- A brand-new AWS account may have a lower Lambda memory limit than the 4096 MB the template asks for. **Unverified**; the fix is in the troubleshooting table.
- Whether your chosen region offers SageMaker serverless inference. **Unverified**; check before starting Phase 3.
- Real cold-start and latency numbers. Nobody has measured them.

---

## Phase 0: setup (do this once)

### 0.1 Install the tools
```bash
brew install awscli aws-sam-cli
brew install --cask docker
```
If `brew` cannot find `aws-sam-cli`, install the SAM CLI from the AWS documentation instead. Open **Docker Desktop** once from Applications, accept the terms, and wait until it says it is running.

Check:
```bash
aws --version
sam --version
docker --version
docker run --rm hello-world        # must print "Hello from Docker!"
```
- [ ] all four commands work

### 0.2 AWS account
Create one at aws.amazon.com if you do not have it (phone verification and a card are required; creating the account is free). Then, **before creating any resources**:

1. **Turn on MFA** for the root user (Console -> your name -> Security credentials).
2. **Create a budget alert** (Console -> Billing -> Budgets -> Create budget -> "Monthly cost budget" template). Use a small amount and your email.
- [ ] MFA on root user
- [ ] budget alert created

### 0.3 A non-root user for the command line
Do not use the root user for the CLI.
1. Console -> IAM -> Users -> Create user (name it `paper-assistant-deployer`).
2. Permissions: attach `AdministratorAccess`. This is broader than ideal; it is acceptable for a personal learning account that holds nothing else, because the alternative (hand-writing a minimal policy for CloudFormation, Lambda, IAM, ECR, S3, SSM, SNS, CloudWatch and Budgets) takes longer than the whole exercise. **Delete the access key when you finish** (section 7).
3. Open the user -> Security credentials -> Create access key -> use case "Command Line Interface". Copy the two values into a password manager. **Never paste them into a chat, a file in the repository, or a screenshot.**
4. Configure a named profile and use it:
```bash
aws configure --profile paper-assistant      # paste key id, secret, region, output format "json"
export AWS_PROFILE=paper-assistant
export AWS_REGION=us-east-1                  # pick one region and use it for everything
aws sts get-caller-identity                  # must show your account id and the deployer user
```
- [ ] `aws sts get-caller-identity` works with the deployer user

---

## Phase 1: build and test both containers locally (no AWS needed)

Do this before touching AWS: it finds most problems for free.

### 1.1 The API image (used by Lambda)
From the repository root:
```bash
docker build -t paper-assistant .
```
Expect this to be slow the first time. It installs CPU-only PyTorch, then downloads the 59 papers from arXiv politely (about 3 seconds apart) and builds the index, and warms both models. A failure while downloading papers means arXiv refused or the network dropped; run the build again. **If arXiv keeps refusing the downloads from inside the build, fallback:** run the download once on your Mac (`uv run python -c "from src.copilot.papers import load_papers, download; download(list(load_papers()))"`), then change the Dockerfile's download line to `COPY data/raw ./data/raw` and remove `data/raw` from `.dockerignore`.

Run it (the Anthropic key comes from your local `.env` file at run time and is not baked into the image):
```bash
docker run --rm -p 8000:8000 --env-file .env -e APP_API_KEY=local-test paper-assistant
```
In another terminal:
```bash
curl localhost:8000/health                       # {"status":"ok","chunks":4645} once the models are loaded
curl -s -o /dev/null -w "%{http_code}\n" -X POST localhost:8000/research \
  -H 'content-type: application/json' -d '{"task":"Compare LoRA and QLoRA memory use, please"}'   # 401 (no key)
curl -s -X POST localhost:8000/research -H 'x-api-key: local-test' -H 'content-type: application/json' \
  -d '{"task":"Compare LoRA and QLoRA: how do they relate and what memory does QLoRA save for a 65B model?","system":"single_shot"}' | head -c 400
```
- [ ] image builds
- [ ] `/health` returns 4645 chunks
- [ ] 401 without the key, a cited brief with the key

### 1.2 The SageMaker image
SageMaker serverless inference should get an `amd64` image (on Apple Silicon this is emulated, so the build is slow; **I have not verified whether arm64 is accepted**, amd64 is the safe choice):
```bash
docker build --platform linux/amd64 -f sagemaker/Dockerfile -t paper-assistant-encoders .
docker run --rm -p 8080:8080 paper-assistant-encoders serve
```
In another terminal: `curl -i localhost:8080/ping` returns 503 for a few seconds while the models load, then 200. Then:
```bash
curl -s -X POST localhost:8080/invocations -H 'content-type: application/json' -d '{"op":"embed","texts":["hello"]}' | head -c 200
```
- [ ] image builds
- [ ] `/ping` turns 200, embed returns a vector

### 1.3 Push to GitHub and read the CI result
`git push`, then open the repository's **Actions** tab. The `test` job runs pytest; the `docker-build` job builds the main image from scratch. Green = the Dockerfile works on Linux/amd64. Red = open the failing step and read the last 30 lines.
- [ ] CI is green (or the error is recorded in the progress log)

**You can stop here honestly saying:** containerised with Docker, images verified locally, tests and CI in place.

---

## Phase 2: deploy the API to AWS Lambda

What gets created (all from `template.yaml`): a Lambda function from the container image (arm64, 4096 MB, 300 s timeout) with a public Function URL, a log group with 14-day retention, an SNS topic with your email, an alarm for more than 200 invocations in a day, and a monthly budget with an 80% alert. Secrets stay in SSM Parameter Store, not in the image or the template.

### 2.1 Store the two secrets
```bash
export APP_KEY=$(openssl rand -hex 24)
aws ssm put-parameter --name /paper-assistant/anthropic_api_key --type SecureString \
  --value "$(grep '^ANTHROPIC_API_KEY=' .env | cut -d= -f2-)"
aws ssm put-parameter --name /paper-assistant/app_api_key --type SecureString --value "$APP_KEY"
echo "$APP_KEY"      # save it in your password manager; this is the x-api-key for your API
```
- [ ] both parameters created (`aws ssm describe-parameters` lists them)

### 2.2 Build and deploy
```bash
sam build
sam deploy --guided --parameter-overrides AlertEmail=you@example.com
```
Answers for the guided prompts: stack name `paper-assistant`; your region; "Allow SAM CLI IAM role creation": yes; "Disable rollback": no; for the Function URL with no IAM auth, SAM warns that it is public: yes (the app checks `x-api-key` itself); "Save arguments to configuration file": yes; "Create managed ECR repositories": yes. `sam build` builds the image (slow the first time) and `sam deploy` pushes it to ECR and creates the stack.

Open the email AWS sends and **confirm the SNS subscription**, otherwise alerts are not delivered.
- [ ] stack reaches `CREATE_COMPLETE`
- [ ] SNS subscription confirmed

### 2.3 Try it
```bash
export URL=$(aws cloudformation describe-stacks --stack-name paper-assistant \
  --query "Stacks[0].Outputs[?OutputKey=='FunctionUrl'].OutputValue" --output text)
time curl -s $URL/health                    # the first call after idle includes a cold start: note the time
time curl -s $URL/health                    # the second one is warm
curl -s -X POST ${URL}research -H "x-api-key: $APP_KEY" -H 'content-type: application/json' \
  -d '{"task":"Compare LoRA and QLoRA: how do they relate and what memory does QLoRA save for a 65B model?","system":"staged_r1"}' | python3 -m json.tool | head -40
```
(The URL ends with a slash, so `${URL}research` is correct.) Check the response: the brief text, `sources`, `cost_usd`, `seconds`.
- [ ] `/health` works, cold and warm times noted
- [ ] `/research` returns a cited brief
- [ ] a request without the key gets 401

### 2.4 See the monitoring
Console -> CloudWatch -> Logs -> the function's log group (see the request lines). CloudWatch -> Metrics -> namespace `PaperAssistant` (CostUSD, LatencySeconds, LLMCalls, tokens; may take a few minutes to appear).
- [ ] logs visible
- [ ] metrics visible

### 2.5 If you are done for now: tear it down
See section 7. Deleting the stack removes the function, URL, log group, alarm, topic and budget.

---

## Phase 3: the SageMaker serverless endpoint for the encoders

What it is: the embedding model and the reranker moved behind a SageMaker endpoint, so the API can call them instead of loading them. Setting `SAGEMAKER_ENDPOINT` switches the project to the remote models; leaving it unset keeps the local ones. The honest expectation is that it **removes model loading from the service but adds a network hop**, so latency will not drop. Recording that trade-off with real numbers is the point.

Before starting, confirm your region supports SageMaker **serverless** inference (check the SageMaker documentation for regional availability; `us-east-1` is the safe choice). Never create a real-time endpoint.

### 3.1 Push the image to ECR
```bash
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export IMAGE=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/paper-assistant-encoders:latest
aws ecr create-repository --repository-name paper-assistant-encoders
aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag paper-assistant-encoders $IMAGE
docker push $IMAGE
```
- [ ] image visible in the ECR console

### 3.2 Create the execution role
```bash
cat > /tmp/trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name paper-assistant-sagemaker --assume-role-policy-document file:///tmp/trust.json
aws iam attach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
export ROLE_ARN=arn:aws:iam::$ACCOUNT:role/paper-assistant-sagemaker
```
IAM changes can take a few seconds to take effect; wait a moment before the next step.
- [ ] role exists

### 3.3 Deploy
```bash
uv run python sagemaker/deploy.py --role-arn $ROLE_ARN --image-uri $IMAGE --yes
```
The script creates the model, the serverless endpoint configuration (3072 MB, max concurrency 2) and the endpoint, waits until it is in service (several minutes), and prints the latency of a cold and a warm first call. If it fails, see the troubleshooting table; the CloudWatch log group `/aws/sagemaker/Endpoints/paper-assistant-encoders` holds the container's own output.
- [ ] endpoint `InService`
- [ ] cold and warm latencies noted

### 3.4 Measure
```bash
SAGEMAKER_ENDPOINT=paper-assistant-encoders uv run python -m eval.measure_sagemaker
```
It prints how many of 20 queries return the same top-5 as the local models, and the latency per retrieval (each retrieval makes two endpoint calls: embed, then rerank). Copy the output into the progress log.
- [ ] output recorded

### 3.5 Tear it down
```bash
uv run python sagemaker/teardown.py
```
Then delete the ECR repository and the role (section 7).

---

## 4. Optional extension (only if everything above worked and you want more)

Have the Lambda function call the SageMaker endpoint for embeddings and reranking: give the function's role permission `sagemaker:InvokeEndpoint` on the endpoint and set `SAGEMAKER_ENDPOINT` in the template's environment variables. Be aware of the limits: SageMaker real-time and serverless endpoints must respond within 60 seconds per call (each call here is a fraction of a second), the Lambda image still imports `sentence_transformers` at module level so it keeps its dependencies unless you refactor the imports, and you would pay for both services. This is a refactor and an experiment, not a requirement.

---

## 5. Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| `docker build` fails while downloading papers | arXiv rate limiting or a dropped connection | Run again; if it persists use the fallback in 1.1 (copy the PDFs in) |
| `docker build` fails with "no matching distribution" for torch | CPU index not used or version mismatch | Compare the failing version against `requirements-runtime.txt`; the file pins the versions from `uv.lock` |
| `pull access denied ... aws-lambda-adapter` | The pinned adapter tag does not exist (**unverified** version) | Look up the current tag of the Lambda Web Adapter image and edit the `COPY --from=` line in `Dockerfile` |
| `sam build` very slow or out of disk | Large image layers | Free Docker disk space (Docker Desktop -> Troubleshoot -> Clean), retry |
| CloudFormation error about Lambda memory (e.g. memory size limit) | New accounts can have a lower memory limit (**unverified**) | In `template.yaml` lower `MemorySize` to 3008, or request a quota increase; redeploy |
| Deploy fails on the SSM policy | The policy expression that builds the parameter path (**unverified**) | Replace it with the literal ARN: `arn:aws:ssm:<region>:<account>:parameter/paper-assistant/*` |
| Function times out on the first call | Cold start while loading torch and two models | Retry once; if it always fails, raise `Timeout` or memory, and read the log group |
| `/research` returns 502 | An upstream error (Anthropic limit, network) | Read the log line; check the Anthropic usage limit |
| `/research` returns 401 | Wrong or missing `x-api-key` | Use the value you stored as `app_api_key`; the header name is exactly `x-api-key` |
| SageMaker endpoint ends `Failed` | Role lacks ECR/log permission; wrong image architecture; `/ping` not 200 in 8 minutes | Read the endpoint's CloudWatch logs; re-check 3.2; rebuild with `--platform linux/amd64` |
| `ValidationException ... serverless` | Region or configuration not supported | Pick a region that supports serverless inference |
| `AccessDenied` while deploying | The CLI user lacks a permission | The message names it; for this exercise use the admin user from 0.3 |
| Budget email never arrived | Subscription not confirmed | Confirm the SNS/Budgets email, check spam |

When asking for help, include: which step, the full command you ran, and the full error text. Do not include keys.

---

## 6. Definition of done and what you may say

Update the claims only for phases you actually completed and verified yourself.

| You finished | You may say |
|---|---|
| Phase 1 | "Containerised the API and the model service with Docker, verified locally; CI builds the image." |
| Phase 2 | "Deployed the research-agent API to AWS Lambda (container image, Function URL, secrets in SSM Parameter Store, CloudWatch metrics and alarms); measured cold and warm latency and per-request cost." |
| Phase 3 | "Served the embedding and reranking models from a SageMaker serverless endpoint (custom container); verified identical retrieval results against the local models and measured cold and warm latency." |

Do **not** write "production", "users", "scalable to N requests" or any number you did not measure. If a phase failed, "prepared deployment files for X" is true; "deployed X" is not.

After finishing, update these places so the repository and the resume agree:
1. Fill in the progress log below.
2. In `README.md`, section "Limitations", change the bullet "Deployment is untested" to say exactly what was deployed and verified, with the measured numbers.
3. In `docs/PROJECT_FACTS.md`, move the deployed items from the "NOT done" list into the done list, with the numbers, and update the mapping table (AWS / SageMaker rows).
4. In `docs/DEPLOY.md` and `docs/SAGEMAKER.md`, change the "Status" line, and fill the Results table in `docs/SAGEMAKER.md`.
5. In `ROADMAP.md`, step 8.

## 7. Cleanup checklist (do this when you stop, even if only for a while)

- [ ] `sam delete --stack-name paper-assistant` (removes the function, URL, log group, alarm, topic and budget)
- [ ] `uv run python sagemaker/teardown.py` (endpoint, endpoint config, model)
- [ ] ECR: delete the repositories (`aws ecr delete-repository --repository-name paper-assistant-encoders --force`; the SAM-managed repository is listed in the ECR console, delete it too)
- [ ] S3: SAM's managed artifact bucket (stack `aws-sam-cli-managed-default`); empty and delete it from the console if you will not redeploy
- [ ] SSM: `aws ssm delete-parameter --name /paper-assistant/anthropic_api_key` and `.../app_api_key`
- [ ] IAM: the SageMaker role (detach its two policies, then delete it)
- [ ] IAM: **deactivate and delete the deployer user's access key** (Console -> IAM -> Users -> Security credentials); delete the user too if you are finished
- [ ] Billing: check the Billing dashboard the next day for anything unexpected
- [ ] Anthropic console: lower or keep the spending limit as you prefer

## 8. Progress log (fill in as you go)

| Date | Phase / step | Result (OK / failed + error) | Numbers noted |
|---|---|---|---|
| | 0 setup | | |
| | 1.1 API image | | |
| | 1.2 encoder image | | |
| | 1.3 CI | | |
| | 2.2 SAM deploy | | |
| | 2.3 cold / warm / brief | | cold: / warm: / cost_usd: |
| | 3.3 SageMaker deploy | | cold: / warm: |
| | 3.4 measure | | identical top-5: /20, median ms: |
| | 7 cleanup | | |

## 9. Rough effort (my guess, not a measurement)

Phase 0 and 1 together: one sitting of a few hours, mostly waiting for builds. Phase 2: another sitting, most of it the first `sam build`/`deploy` plus fixing one or two issues. Phase 3: a shorter sitting once Docker and AWS are set up. It is normal for the first deployment to the cloud to take several tries; each failure above is listed because it is a plausible one, not because it is certain.
