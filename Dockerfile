# Onion Quality Grading - Streamlit app, CPU only (Render.com)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    ONION_DEVICE=cpu \
    ONION_DISABLE_LIVE=1 \
    ONION_DEMO=1 \
    PORT=10000

# libGL/glib for OpenCV, libgomp for torch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default wheels pull in ~3 GB of CUDA)
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.14.1 torchvision==0.29.1 \
    && pip install -r requirements.txt

COPY . .

EXPOSE 10000
CMD streamlit run onion_grader/app.py \
        --server.port=${PORT} --server.address=0.0.0.0 \
        --server.headless=true --server.fileWatcherType=none \
        --browser.gatherUsageStats=false
