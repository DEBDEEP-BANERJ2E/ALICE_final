FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies
RUN uv sync --no-dev 2>/dev/null || uv sync --no-dev --inexact

# Copy application code
COPY . .

# Persistent storage directories
RUN mkdir -p /app/data/failure_bank /app/data/trajectories /app/logs

ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health', timeout=5)"

CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
