# Embedding + reranking inference container for SageMaker. Build from the repository root:
#   docker build -f sagemaker/Dockerfile -t paper-assistant-encoders .
FROM python:3.13-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/opt/hf
WORKDIR /opt/program
COPY sagemaker/requirements.txt .
RUN pip install -r requirements.txt
# Bake both models into the image so a cold start needs no download.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
COPY sagemaker/app.py sagemaker/serve.py ./
ENV HF_HUB_OFFLINE=1
EXPOSE 8080
# exec form (SageMaker sends SIGTERM); runs as root, which SageMaker expects
ENTRYPOINT ["python", "serve.py"]
