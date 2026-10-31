# ── StreetSynth Production Backend Container ──────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    LIGHTWEIGHT=1 \
    PORT=8000

# Install system dependencies for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend codebase
COPY api/ ./api/
COPY cv/ ./cv/
COPY gan/ ./gan/
COPY pipeline.py .
COPY checkpoints/ ./checkpoints/
COPY outputs/ ./outputs/

EXPOSE 8000

# Start FastAPI production server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

