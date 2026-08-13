# Backend inference server. The fitted memory bank is large and gitignored, so
# it is NOT baked in -- mount checkpoints/ at runtime and pick one via FD_CATEGORY:
#
#   docker build -t fabric-defect .
#   docker run --rm -p 8000:8000 \
#       -v "$PWD/checkpoints:/app/checkpoints" \
#       -e FD_CATEGORY=carpet fabric-defect
#
# For GPU inference, use an appropriate CUDA base image and the CUDA torch wheels.
FROM python:3.11-slim

# OpenCV needs these shared libs even in headless use.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the server needs at runtime.
COPY src ./src
COPY backend ./backend
COPY config ./config

EXPOSE 8000
ENV FD_CATEGORY=carpet
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
