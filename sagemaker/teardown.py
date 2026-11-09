"""Delete the endpoint, endpoint config and model created by deploy.py. Safe to run twice."""
import argparse

import boto3
from botocore.exceptions import ClientError

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="paper-assistant-encoders")
name = ap.parse_args().name
sm = boto3.client("sagemaker")
for what, call in (("endpoint", lambda: sm.delete_endpoint(EndpointName=name)),
                   ("endpoint config", lambda: sm.delete_endpoint_config(EndpointConfigName=name)),
                   ("model", lambda: sm.delete_model(ModelName=name))):
    try:
        call()
        print("deleted", what)
    except ClientError as e:
        print(f"{what}: {e.response['Error']['Message']}")
