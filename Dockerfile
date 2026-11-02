FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
# The index (data/index) and the API key are NOT baked into the image: mount the data folder, pass the key at run time.
EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "src.copilot.api:app", "--host", "0.0.0.0", "--port", "8000"]
