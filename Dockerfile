# =============================================================================
# Multi-stage Dockerfile for MinerU DataAgent (MIDC26-Track2-Evo-Eval)
# =============================================================================
# Stage 1 — Builder: install Python deps with build tools
# Stage 2 — Runtime: slim image with only runtime dependencies
# =============================================================================

# --------------- Stage 1: Builder ---------------
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System build dependencies (needed to compile some Python wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        poppler-utils \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install Python dependencies (heavy ones like paddle may take time)
RUN pip install --no-cache-dir -r requirements.txt

# --------------- Stage 2: Runtime ---------------
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime system libraries required by OpenCV, PaddleOCR, pdf2image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        poppler-utils \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy project source
COPY src/ ./src/
COPY main.py .
COPY requirements.txt .

# Create data directories
RUN mkdir -p /app/data/output /app/data/temp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
