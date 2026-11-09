# SageMaker experiment: serve the embedding and reranking models from an endpoint

**Status: written and tested locally, not yet run on AWS.** The container, the client and the deploy scripts exist and pass tests; with the real models running in the container on a laptop, the endpoint returned bit-identical embeddings and reranker scores and identical top-5 results for 12 of 12 queries. Nothing has been built with Docker or deployed to SageMaker. Fill in the "Results" section below only with numbers you measured yourself.

## What this is, and why this shape

- The Claude models cannot be hosted on SageMaker, so this experiment moves the two small local models (MiniLM embedder, cross-encoder reranker) behind a SageMaker endpoint. The rest of the project does not change: setting `SAGEMAKER_ENDPOINT` switches `DenseIndex` and `rerank()` to the endpoint, leaving it unset keeps the local models.
- It is a **serverless inference** endpoint: no instance runs while idle, and you pay for the requests. A real-time endpoint bills for instance hours as long as it exists, which is the most common way to burn a budget on SageMaker; avoid it.
- Limits that matter (from the SageMaker documentation): the container must listen on port 8080 and answer `POST /invocations` and `GET /ping`; it must answer within 60 seconds; the new container must start passing `/ping` within 8 minutes or the endpoint fails; `/ping` must answer within 2 seconds. The service in `sagemaker/app.py` loads the models in a background thread and answers 503 on `/ping` until they are ready.
- Why not put the whole research agent there: a staged brief takes about 38-60 s, too close to the 60 s response limit; Lambda is the better home for the service (see `docs/DEPLOY.md`).

## Steps (you run these)

Replace the region if you prefer another one. Check that your region supports serverless inference; `us-east-1` and `us-west-2` are the safest. Never paste access keys into chat or commit them.

**0. Tools and a billing guard.** AWS account, an IAM user (not the root user) with CLI credentials (`aws configure`), Docker Desktop. In the AWS console create a **Budget** (Billing -> Budgets) with a small monthly amount and an email alert before creating anything else.

**1. Build and test the container locally** (build for `linux/amd64`; on Apple Silicon this is emulated, so the build is slower):
```bash
docker build --platform linux/amd64 -f sagemaker/Dockerfile -t paper-assistant-encoders .
docker run --rm -p 8080:8080 paper-assistant-encoders serve
# in another terminal; /ping answers 503 for a few seconds while the models load, then 200
curl -i localhost:8080/ping
curl -s -X POST localhost:8080/invocations -H 'content-type: application/json' -d '{"op":"embed","texts":["hello"]}' | head -c 200
```

**2. Push the image to ECR:**
```bash
export AWS_REGION=us-east-1
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export IMAGE=$ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com/paper-assistant-encoders:latest
aws ecr create-repository --repository-name paper-assistant-encoders --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com
docker tag paper-assistant-encoders $IMAGE
docker push $IMAGE
```

**3. Create the execution role** SageMaker uses to pull the image and write logs:
```bash
cat > trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON
aws iam create-role --role-name paper-assistant-sagemaker --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam attach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
export ROLE_ARN=arn:aws:iam::$ACCOUNT:role/paper-assistant-sagemaker
```
Your own IAM user also needs permission to use SageMaker and to pass this role (`iam:PassRole`). If a later step fails with an access error, the message names the missing permission.

**4. Deploy the serverless endpoint** (this creates billable resources, hence the `--yes`):
```bash
uv run python sagemaker/deploy.py --role-arn $ROLE_ARN --image-uri $IMAGE --yes
```
It prints the cold-start and warm latency of a first call. If the endpoint ends in a failed state, read the logs in CloudWatch under `/aws/sagemaker/Endpoints/paper-assistant-encoders`; the usual causes are a missing role permission, an image built for the wrong architecture, or `/ping` not turning 200 in time.

**5. Measure** (this produces the numbers you may quote):
```bash
SAGEMAKER_ENDPOINT=paper-assistant-encoders uv run python -m eval.measure_sagemaker
```
To run the whole system on the endpoint: `export SAGEMAKER_ENDPOINT=paper-assistant-encoders` before starting the API or any evaluation script.

**6. Tear down when you are done** (serverless costs almost nothing idle, but remove everything you do not need):
```bash
uv run python sagemaker/teardown.py
aws ecr delete-repository --repository-name paper-assistant-encoders --force --region $AWS_REGION
aws iam detach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam detach-role-policy --role-name paper-assistant-sagemaker --policy-arn arn:aws:iam::aws:policy/CloudWatchLogsFullAccess
aws iam delete-role --role-name paper-assistant-sagemaker
```

## What to expect and what it costs

- Cost: per-request compute time on the memory you chose, plus ECR storage for the image. I do not know current prices; check the SageMaker and ECR pricing pages and your Budget alert instead of trusting a number here.
- Latency: each retrieval makes two endpoint calls (embed, then rerank). Locally each call is about the model time; on AWS add network time, and the first request after idle includes a cold start. That is the honest trade-off of this experiment: it removes the model load from the service but adds a network hop.
- Expect the first deploy to need a fix or two (permissions, architecture). That is normal.

## Results (fill in from your own run)

| Measurement | Value |
|---|---|
| Region, memory, max concurrency | |
| Cold-start first call | |
| Warm retrieval median / max (embed + rerank) | |
| Identical top-5 vs local models | /20 |
| Observed cost for the experiment (from Billing) | |

## What you can truthfully say afterwards

Only after step 5 has run: "Served the embedding and reranking models from a SageMaker serverless inference endpoint (custom container following the SageMaker contract), verified identical retrieval results against the local models and measured cold/warm latency." Before that, say it was designed, implemented and tested locally.
