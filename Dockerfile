FROM python:3.11-slim

ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV HF_HOME=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/hf_cache && chmod -R 777 /tmp/hf_cache

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "app.py"]

