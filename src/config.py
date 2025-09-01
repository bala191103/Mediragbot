import os
from dotenv import load_dotenv
import boto3
import json 
from botocore.exceptions import ClientError

# Load local .env
load_dotenv()

def get_secret(name, default=None):
    """Check .env first, then AWS Secrets Manager (for EC2/Fargate)."""
    # First try env vars
    val = os.getenv(name)
    if val:
        return val

    # If running on AWS and env not set, fall back to Secrets Manager
    try:
        client = boto3.client("secretsmanager", region_name="ap-south-1")
        response = client.get_secret_value(SecretId="myapp/medical-bot-keys")
        secrets = json.loads(response["SecretString"])
        return secrets.get(name, default)
    except ClientError:
        return default

# Keys & settings
LLM_API_KEY = get_secret("LLM_API_KEY")
PINECONE_API_KEY = get_secret("PINECONE_API_KEY")
PINECONE_INDEX_NAME = get_secret("PINECONE_INDEX_NAME", "default-index")
HF_TOKEN = get_secret("HF_TOKEN")

GEN_MODEL = get_secret("GEN_MODEL", "llama-3.3-70b-versatile")
JUDGE_MODEL = get_secret("JUDGE_MODEL", "llama-3.3-70b-versatile")
