# Use CUDA base for GPU support
FROM nvidia/cuda:13.0.1-runtime-ubuntu22.04

# Set timezone non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install Python and basic dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    git \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# Copy essential files first
COPY pyproject.toml poetry.lock* ./

# Install poetry and dependencies
RUN python3.11 -m pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main

# Copy application code
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY models/ ./models/
COPY scripts/ ./scripts/
COPY .env ./

# Set environment
ENV PYTHONPATH="/app"
ENV HF_HOME="/app/.cache/huggingface"

EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
