# Real-Time AI Voice Detection System

Production-ready server to detect whether a live telephone caller is using an AI-generated/synthetic voice or a human voice using ONNX Runtime, Silero VAD, sliding window speech buffers, rolling EMA aggregation, and WebSockets.

Primary ML Model: [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)

---

## High-Level Architecture

```
                    TELEPHONE CALL (WebSocket)
                               |
                               v
                     Audio Queue (Unblocked)
                               |
                               v
                     Voice Activity Detection (VAD)
                               |
                     +---------+---------+
                     |                   |
                  SILENCE              SPEECH
                     |                   |
                  discard                v
                               Speech Sliding Buffer
                               (3s window, 1s stride)
                                         |
                                         v
                                Resample (8k -> 16k)
                                         |
                                         v
                              ONNX Runtime Inference
                                         |
                                         v
                             Exponential Moving Average
                                 Rolling Aggregator
                                         |
                            +------------+------------+
                            |                         |
                       AI confidence             Human confidence
                            |                         |
                            v                         v
                           AI                       HUMAN
```

---

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Export Hugging Face Model to ONNX FP32
Exports `garystafford/wav2vec2-deepfake-voice-detector` to `models/detector/model.onnx` and downloads Silero VAD to `models/vad/silero_vad.onnx`.
```bash
python scripts/export_onnx.py
```

### 3. Validate ONNX vs PyTorch Equivalence
Validates logit/probability closeness between PyTorch model and exported ONNX model.
```bash
python scripts/validate_onnx.py
```

### 4. Quantize ONNX to INT8 (Optional)
Quantizes ONNX FP32 model to `models/detector/model_int8.onnx` for CPU speedup.
```bash
python scripts/quantize.py
```

### 5. Run System Benchmark
Measures latency breakdown (VAD, Preprocessing, Inference, Postprocessing) and percentiles (P50, P95, P99).
```bash
python scripts/benchmark.py
```

### 6. Run Test Suite
```bash
pytest
```

---

## Running the Server

Start the FastAPI server listening on `0.0.0.0:8000` (accessible publicly via Nginx / ngrok):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### API Endpoints
- `GET /health`: Server liveness check.
- `GET /ready`: Readiness check (verifies ONNX model initialized & warmed up).
- `GET /metrics`: Continuous performance and detection statistics.
- `WS /ws/call/{call_id}`: Real-time telephone audio WebSocket endpoint.

---

## Nginx Public IP Reverse Proxy Setup

To expose the server to your teammates on a public IP or domain:

```nginx
server {
    listen 80;
    server_name your-public-ip-or-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400s;
    }
}
```

---

## Python WebSocket Client Example

```python
import asyncio
import websockets
import json

async def stream_audio_call():
    uri = "ws://localhost:8000/ws/call/call-abc123"
    async with websockets.connect(uri) as websocket:
        print("Connected to AI Voice Detector Server!")
        
        # Read PCM16 mono 8000Hz audio file or live microphone stream
        with open("sample_audio.pcm", "rb") as f:
            while chunk := f.read(1600):  # 100ms chunks at 8kHz
                await websocket.send(chunk)
                await asyncio.sleep(0.1)

                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    print("Detection Event:", json.loads(response))
                except asyncio.TimeoutError:
                    pass

asyncio.run(stream_audio_call())
```

---

## Configuration Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/detector/model.onnx` | Path to exported ONNX detector model |
| `VAD_MODEL_PATH` | `models/vad/silero_vad.onnx` | Path to Silero VAD ONNX model |
| `INPUT_SAMPLE_RATE` | `8000` | Input telephone audio sample rate (Hz) |
| `AUDIO_SAMPLE_RATE` | `16000` | Target model input sample rate (Hz) |
| `WINDOW_SECONDS` | `3.0` | Audio speech window duration (seconds) |
| `STRIDE_SECONDS` | `1.0` | Sliding window stride duration (seconds) |
| `AI_THRESHOLD` | `0.90` | Probability threshold for AI prediction |
| `HUMAN_THRESHOLD` | `0.10` | Probability threshold for Human prediction |
| `EMA_ALPHA` | `0.4` | Exponential Moving Average smoothing factor |
| `AI_CONFIRMATIONS` | `2` | Consecutive windows required for AI status |
| `HUMAN_CONFIRMATIONS` | `3` | Consecutive windows required for Human status |
| `ONNX_PROVIDER` | `cpu` | `cpu` or `cuda` for ONNX execution |
