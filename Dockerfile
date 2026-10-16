# ==============================================================================
# Production Dockerfile for AI Career Copilot
# Optimized for Hugging Face Spaces (UID 1000, Port 7860) & Cloud Containers
# ==============================================================================

FROM python:3.11-slim

# Set environment variables for Python & Server
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0 \
    APP_ENV=production \
    DEBUG=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    DEBIAN_FRONTEND=noninteractive

# Install minimal OS dependencies for numerical C-extensions & health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Render and Hugging Face Spaces compatibility with UID 1000
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, clean datasets, and static frontend
COPY --chown=appuser:appuser . .

# Ensure appuser owns everything in /app and has write permissions to logs
RUN chown -R appuser:appuser /app && chmod -R 775 /app/logs

# Switch to non-root user
USER appuser

# Standard Cloud Run is 8080, Render is 10000, local is 8000, HF Spaces is 7860
EXPOSE 8080 8000 10000 7860

# Health check to ensure the service is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI ASGI server via main.py
CMD ["python", "main.py"]
