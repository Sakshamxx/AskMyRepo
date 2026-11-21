# AskMyRepo web app (Streamlit) on port 8501.
#
# Not baked in, on purpose:
#   - the keys: pass them at run time with --env-file .env
#   - the two Qwen3 models (about 2.4 GB): mount your Hugging Face cache at /root/.cache/huggingface
#   - ShopLite's database: mount the file at /data/shoplite.db (opened read-only)
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 \
    ASKMYREPO_DATABASE=/data/shoplite.db \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Libraries first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY askmyrepo ./askmyrepo
COPY app.py corpus.toml ./
COPY index ./index
RUN uv sync --frozen --no-dev

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
