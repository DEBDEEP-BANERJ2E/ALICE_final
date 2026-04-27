# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency manifest and install into an isolated virtualenv
COPY pyproject.toml uv.lock* ./
RUN uv venv /app/.venv && \
    VIRTUAL_ENV=/app/.venv uv pip install --no-cache \
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
        "matplotlib>=3.7.0" \
        "python-dateutil>=2.9.0"

# Install GPU training stack (torch + unsloth); skip silently on CPU-only builds
RUN VIRTUAL_ENV=/app/.venv uv pip install --no-cache \
        "torch>=2.1.0" \
        "transformers>=4.40.0" \
        "trl>=0.11.0" \
        "accelerate>=0.26.0" \
    && VIRTUAL_ENV=/app/.venv uv pip install --no-cache \
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    || echo "Unsloth install skipped (no CUDA at build time — will fall back to standard transformers)"

# ── Stage 2: minimal runtime image ────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy uv (needed at runtime for `uv run` commands if used)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only the pre-built virtualenv from builder — no build tools needed
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Persistent storage for failure bank, trajectory history, and logs
RUN mkdir -p /app/data/failure_bank /app/data/trajectories /app/logs

ENV PATH="/app/.venv/bin:$PATH" \
    VIRTUAL_ENV=/app/.venv \
    PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health', timeout=5)"

CMD ["python", "alice_server.py"]
