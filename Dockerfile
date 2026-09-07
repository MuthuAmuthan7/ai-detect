FROM python:3.12-slim

WORKDIR /app

# Install system audio dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose server port for reverse proxy / Nginx / public IP
EXPOSE 8000

ENV MODEL_PATH="models/detector/model.onnx"
ENV VAD_MODEL_PATH="models/vad/silero_vad.onnx"
ENV ONNX_PROVIDER="cpu"
ENV ORT_INTRA_OP_THREADS=4

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
