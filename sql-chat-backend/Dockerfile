# ============================================================================
# OptimaX Backend - Production Dockerfile
# ============================================================================
# This Dockerfile creates a fully self-contained image that:
# - Does NOT depend on any local .venv or host Python
# - Fails fast if LlamaIndex NL-SQL is not properly installed
# - Runs as non-root user for security
# - Injects environment variables at runtime (not build time)
# ============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Build dependencies
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment in builder
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Production image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="OptimaX Team"
LABEL version="6.1"
LABEL description="OptimaX NL-SQL Backend - LlamaIndex + FastAPI + Groq"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd --gid 1000 optimax \
    && useradd --uid 1000 --gid optimax --shell /bin/bash --create-home optimax

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code (excluding dev artifacts via .dockerignore)
COPY --chown=optimax:optimax . .

# -----------------------------------------------------------------------------
# Build-time verification: FAIL FAST if LlamaIndex is broken
# -----------------------------------------------------------------------------
RUN echo "Running build-time LlamaIndex verification..." \
    && python docker-verify.py \
    && echo "Build verification passed!"

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
# Environment variables are injected at runtime, NOT baked into image
# Required at runtime:
#   - GROQ_API_KEY: API key for Groq LLM
#   - DATABASE_URL: PostgreSQL connection string

# Switch to non-root user
USER optimax

# Expose port
EXPOSE 8000

# Health check using curl (more reliable than Python in minimal images)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# Start the application
# Using exec form to ensure signals are properly handled
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
