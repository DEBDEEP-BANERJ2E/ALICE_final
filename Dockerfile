FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install only server-required dependencies (skip torch/trl/transformers for the env server)
RUN uv pip install --system --no-cache \
    "openenv-core[core]>=0.2.2" \
    "numpy>=1.26.0" \
    "openai>=1.0.0" \
    "httpx>=0.24.0" \
    "psutil>=5.9.0" \
    "python-dotenv>=1.0.0" \
    "RestrictedPython>=7.0" \
    "sentence-transformers>=2.0.0" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.20.0" \
    "gradio>=4.0.0" \
    "matplotlib>=3.7.0"

# Copy application code
COPY . .

# Persistent storage directories
RUN mkdir -p /app/data/failure_bank /app/data/trajectories /app/logs

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health', timeout=5)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
