# Multi-stage image for AWS Lambda (via the Lambda Web Adapter) or any container host.
# The PDFs are downloaded and the index is built at image build time, so the running container starts with no network and no setup.
FROM python:3.13-slim AS base
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

FROM base AS indexer
ENV HF_HOME=/opt/hf
COPY requirements-build.txt .
RUN pip install -r requirements-build.txt
# Download layer first: it only changes when the paper list changes (arXiv asks for ~1 request per 3 s, so this is slow).
COPY data/papers.json data/papers.json
COPY src/copilot/papers.py src/copilot/papers.py
RUN python -c "from src.copilot.papers import load_papers, download; download(list(load_papers()))"
COPY src ./src
RUN python -c "from src.copilot.retrieval import DenseIndex; DenseIndex().build()" \
 && python -c "from src.copilot.rerank import rerank; rerank('warm up', [0], ['text'])"

FROM base AS runtime
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter
COPY src ./src
COPY data/papers.json data/papers.json
COPY --from=indexer /app/data/index ./data/index
COPY --from=indexer /opt/hf /opt/hf
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/health
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "src.copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
