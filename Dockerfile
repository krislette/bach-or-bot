# Use smaller base image since Render free tier doesn't have GPU
FROM python:3.11-slim

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml poetry.lock* ./
RUN python3.11 -m pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main


# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create cache directories
RUN mkdir -p /app/.cache/huggingface /app/.cache/torch /tmp/numba_cache && \
    chmod -R 777 /app/.cache /tmp/numba_cache

# Set environment
ENV PYTHONPATH="/app"
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface"
ENV TORCH_HOME="/app/.cache/torch"
ENV NUMBA_CACHE_DIR="/tmp/numba_cache"
ENV NUMBA_DISABLE_JIT=0
ENV MUSICLIME_NUM_SAMPLES=1000
ENV MUSICLIME_NUM_FEATURES=10

# Render uses PORT environment variable
EXPOSE 8000

# Use PORT env var that Render provides
CMD uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8000} --timeout-keep-alive 600
