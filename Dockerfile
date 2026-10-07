FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY README.md .

# Create data dirs
RUN mkdir -p data/cache data/logs data/metrics

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["python", "-m", "src.gateway.api"]