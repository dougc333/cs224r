FROM python:3.11-slim

WORKDIR /app
COPY train.py /app/train.py

RUN pip install --no-cache-dir minigrid gymnasium

CMD ["python", "train.py"]
