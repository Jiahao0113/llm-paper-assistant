"""Secrets for deployment. Locally the key comes from .env. On AWS, set SSM_PREFIX and the secrets are read from
SSM Parameter Store (SecureString) at cold start, so they never sit in the image or in the function's environment."""
import os


def load_secrets() -> None:
    prefix = os.environ.get("SSM_PREFIX")
    if not prefix:
        return
    import boto3  # only needed on AWS

    ssm = boto3.client("ssm")
    for env_name, param in (("ANTHROPIC_API_KEY", "anthropic_api_key"), ("APP_API_KEY", "app_api_key")):
        if env_name not in os.environ:
            os.environ[env_name] = ssm.get_parameter(Name=f"{prefix}/{param}", WithDecryption=True)["Parameter"]["Value"]
