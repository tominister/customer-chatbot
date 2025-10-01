FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps if needed (adjust if you don't need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for caching
COPY requirements.txt .
RUN python -m pip install --upgrade pip
# Use --no-cache-dir to avoid pip cache in image layers and reduce final size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (model/ is excluded by .dockerignore)
COPY . .

ENV FLASK_ENV=production

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app", "--workers", "2", "--timeout", "120"]
