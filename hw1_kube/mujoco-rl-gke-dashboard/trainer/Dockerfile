FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV MUJOCO_GL=osmesa

RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6 libosmesa6-dev libgl1 libglfw3 libglew2.2 \
    patchelf ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    "gymnasium[mujoco]==1.2.2" \
    "stable-baselines3==2.7.1" \
    "numpy<2.0" \    
    prometheus-client \
    google-cloud-storage \
    imageio imageio-ffmpeg numpy matplotlib

WORKDIR /app
COPY train.py /app/train.py
CMD ["python", "/app/train.py"]
