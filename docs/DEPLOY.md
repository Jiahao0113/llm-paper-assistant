# Deploying to AWS

**Status: written but not deployed.** The image and template have not been built or deployed from this repository's author environment (no Docker or AWS credentials there). The CI workflow builds the image on every push to `main`; treat the first `sam deploy` as the real test and expect to fix small things.

## What gets created

```
curl --x-api-key--> Lambda Function URL --> Lambda (container image, 4 GB, arm64)
                                              |-- FastAPI + Lambda Web Adapter
                                              |-- index and models baked into the image (no network needed at runtime)
                                              |-- secrets read from SSM Parameter Store at cold start
                                              `-- logs + metrics (cost, latency, tokens per request) in CloudWatch
```

Also: a CloudWatch log group (14-day retention), an SNS topic with an email subscription, an alarm on > 200 invocations/day, and a monthly cost budget with an 80% alert.

Why Lambda and not ECS/Fargate or EC2: traffic is a handful of requests, and Lambda costs nothing while idle. A Fargate task running all month would cost more than the whole budget for this project. The trade-off is a cold start of several seconds (loading torch and two models); the first request after idle time is slow.

## Cost (order of magnitude, check current AWS pricing)

| Item | Estimate |
|---|---|
| Lambda: 4 GB x ~60 s per staged brief | well under a cent per request |
| Model API (Haiku 5.5): staged brief | about 2 cents per request |
| ECR image storage (~2-3 GB) | cents per month |
| SSM standard parameters, SNS email, first budgets | free tier / negligible |
| CloudWatch logs, 14-day retention | cents |

An idle deployment costs roughly a dollar a month or less. The real risk is someone else calling a public URL, which is why the API requires a key, there is a traffic alarm and a budget, and `sam delete` removes everything.

## Steps (you run these; never paste keys into a chat or commit them)

1. **Tools**: an AWS account, AWS CLI (`aws configure` with a least-privilege IAM user, not the root user), SAM CLI, Docker Desktop.
2. **Billing guard first**: in the AWS console enable billing alerts / confirm the budget in the template will be created (it is part of the stack).
3. **Secrets** (SecureString in Parameter Store; the values come from your shell, not from this repo):
   ```bash
   aws ssm put-parameter --name /paper-assistant/anthropic_api_key --type SecureString --value "$ANTHROPIC_API_KEY"
   aws ssm put-parameter --name /paper-assistant/app_api_key --type SecureString --value "$(openssl rand -hex 24)"
   ```
   Read the generated `app_api_key` back once with `--with-decryption` to use as the `x-api-key` header.
4. **Build and deploy** (the build downloads 59 PDFs politely, so allow ~5-10 minutes the first time):
   ```bash
   sam build
   sam deploy --guided --parameter-overrides AlertEmail=you@example.com
   ```
   Confirm the SNS email subscription when AWS sends the message.
5. **Try it**:
   ```bash
   curl -X POST "$FUNCTION_URL/research" -H "x-api-key: $APP_KEY" -H 'content-type: application/json' \
     -d '{"task": "Compare LoRA and QLoRA: how do they relate and what memory does QLoRA save for a 65B model?"}'
   ```
   `/health` should answer first (and is how you see a cold start).
6. **Tear down** when you are not demoing: `sam delete`. Delete the SSM parameters too if you are done.

## What is deliberately simple, and what a real deployment would add

- Auth is a single shared key. A real service would use IAM auth or an identity provider and per-user rate limits.
- No VPC, WAF, or custom domain.
- No request tracing beyond CloudWatch metrics and logs.
- One function per environment; no staging or canary.
- Model access is by API key to the Anthropic API. Inside a company on AWS the usual choice is Amazon Bedrock, which keeps traffic and billing in the AWS account; the code change is the client class, not the logic.
