# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Avoid interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive

# Set a predictable HuggingFace cache location inside the image.
ENV HF_HOME=/app/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    tesseract-ocr \
    tesseract-ocr-deu \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ensure the German spaCy model is available.
RUN python -m spacy download de_core_news_lg

# Warm the HuggingFace cache with the small baseline model. The full 7B model is
# intentionally NOT downloaded here; it is fetched at first run when selected.
COPY scripts/warm_hf_cache.py ./scripts/warm_hf_cache.py
RUN python scripts/warm_hf_cache.py

# Copy the rest of the application.
COPY . .

# Create runtime directories that can also be mounted as volumes.
RUN mkdir -p data/raw data/processed data/evaluation checkpoints outputs logs

EXPOSE 8000

CMD ["python", "run_api.py"]
