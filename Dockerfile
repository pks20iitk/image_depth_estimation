# ─────────────────────────────────────────────────────────────────
# Dockerfile — Depth Estimation API (CPU default)
# ─────────────────────────────────────────────────────────────────
# CPU build:
#   docker build -t depth-api:latest .
#
# GPU build (CUDA 12.1):
#   docker build --build-arg BASE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 \
#                -t depth-api:gpu .
# ─────────────────────────────────────────────────────────────────

ARG BASE=python:3.11-slim
FROM ${BASE}

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        curl wget git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps — separate layer so code changes don't bust the cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

# HuggingFace cache inside the container (mount a volume in production)
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
