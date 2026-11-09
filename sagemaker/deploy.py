"""Create a SageMaker SERVERLESS endpoint for the encoder container (no instance runs while idle).
Needs: the image already pushed to ECR, an execution role, and AWS credentials in your shell.
Usage: python sagemaker/deploy.py --role-arn ARN --image-uri URI --yes [--name NAME] [--memory 3072] [--max-concurrency 2]
Remove everything afterwards with sagemaker/teardown.py."""
import argparse
import json
import time

import boto3

ap = argparse.ArgumentParser()
ap.add_argument("--role-arn", required=True)
ap.add_argument("--image-uri", required=True)
ap.add_argument("--name", default="paper-assistant-encoders")
ap.add_argument("--memory", type=int, default=3072, choices=range(1024, 6145, 1024), help="MB, serverless allows 1024-6144 in 1024 steps")
ap.add_argument("--max-concurrency", type=int, default=2)
ap.add_argument("--yes", action="store_true", help="required: confirms you understand this creates billable AWS resources")
args = ap.parse_args()
if not args.yes:
    raise SystemExit("Refusing to create AWS resources without --yes. Serverless endpoints bill per request and per GB-second of compute; delete with sagemaker/teardown.py when done.")

sm = boto3.client("sagemaker")
print("creating model ...")
sm.create_model(ModelName=args.name, ExecutionRoleArn=args.role_arn, PrimaryContainer={"Image": args.image_uri})
print("creating endpoint config (serverless) ...")
sm.create_endpoint_config(EndpointConfigName=args.name, ProductionVariants=[{
    "VariantName": "AllTraffic", "ModelName": args.name,
    "ServerlessConfig": {"MemorySizeInMB": args.memory, "MaxConcurrency": args.max_concurrency}}])
print("creating endpoint (takes a few minutes) ...")
sm.create_endpoint(EndpointName=args.name, EndpointConfigName=args.name)
sm.get_waiter("endpoint_in_service").wait(EndpointName=args.name, WaiterConfig={"Delay": 30, "MaxAttempts": 40})

rt = boto3.client("sagemaker-runtime")
for label in ("first call (cold start)", "second call (warm)"):
    t = time.time()
    out = json.loads(rt.invoke_endpoint(EndpointName=args.name, ContentType="application/json",
                                        Body=json.dumps({"op": "embed", "texts": ["hello world"]}))["Body"].read())
    print(f"{label}: {time.time() - t:.2f}s, embedding dim {len(out['embeddings'][0])}")
print(f"\nready. Use it with:  export SAGEMAKER_ENDPOINT={args.name}\nWhen finished:        python sagemaker/teardown.py --name {args.name}")
