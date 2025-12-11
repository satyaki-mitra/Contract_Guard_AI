FROM python:3.11-slim-bookworm

WORKDIR /app

# Install ONLY minimal runtime dependencies (no build tools!)
# libgomp1 is needed by OpenMP in NumPy/SciPy; git/wget/curl for huggingface-hub
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (wheels only — no compilation!)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy app code
COPY . .

# Create directories
RUN mkdir -p uploads cache logs /data/models

# Environment: enforce CPU mode
ENV CUDA_VISIBLE_DEVICES=""
ENV LLAMA_CPP_N_GPU_LAYERS=0
ENV OMP_NUM_THREADS=2
ENV NUMEXPR_MAX_THREADS=2

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/api/v1/health || exit 1

# Run app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "30"]