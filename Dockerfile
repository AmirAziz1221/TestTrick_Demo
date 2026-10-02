FROM python:3.11-slim

# Libraries MediaPipe / OpenCV need on a bare Linux server
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libegl1 libgles2 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code, frontend and models/face_landmarker.task
COPY . .

# Recordings and reports are written here. Mount a Railway Volume at /app/data
RUN mkdir -p /app/data/sessions

ENV PYTHONUNBUFFERED=1

# ONE worker on purpose: sessions are kept in memory (no Redis yet).
# Railway provides $PORT automatically.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-800