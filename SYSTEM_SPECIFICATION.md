# Real-Time AI Voice Detection System - Complete Specification

Production-ready architecture for inferring whether a telephone caller is using an AI-generated/synthetic voice clone or a real human voice in real time.

---

## 1. High-Level Architecture

```
                    TELEPHONE CALL (WebSocket)
                               |
                               v
                     Audio Queue (Async / Decoupled)
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
                       AI threshold             Human threshold
                            |                         |
                            v                         v
                           AI                       HUMAN
```

---

## 2. Primary ML Model & Constraints

- **Hugging Face Model**: [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)
- **Model Architecture**: Wav2Vec2 Audio Classification
- **Input Requirements**:
  - Sample Rate: **16,000 Hz** (16 kHz mono)
  - Format: **Float32 array** [-1.0, 1.0] normalized with mean 0, variance 1.
  - Window Length: **3.0 seconds** (48,000 samples)
  - Stride: **1.0 second** (16,000 samples)

---

## 3. Technology Stack

- **Python 3.11+ / 3.12**
- **FastAPI & Uvicorn**: Async WebSockets & HTTP server framework
- **ONNX Runtime (`onnxruntime` / `onnxruntime-gpu`)**: Optimized execution engine
- **Silero VAD**: Voice Activity Detection in ONNX format
- **NumPy & SciPy**: Polyphase audio resampling and array processing
- **Transformers & Optimum**: Model export and normalization semantics
- **Pytest**: Automated unit and integration test suite

---

## 4. API Endpoints Specification

### 4.1 Endpoint: `GET /health`
* **Protocol & Method**: `HTTP GET`
* **URL**: `http://localhost:8000/health`
* **Description**: Server liveness check endpoint indicating if the process is running.
* **Curl Command**:
  ```bash
  curl http://localhost:8000/health
  ```
* **Response Schema**:
  ```json
  {
    "status": "ok",
    "active_calls": 0
  }
  ```

### 4.2 Endpoint: `GET /ready`
* **Protocol & Method**: `HTTP GET`
* **URL**: `http://localhost:8000/ready`
* **Description**: Readiness check endpoint confirming ONNX model is loaded, warmed up, and ready to accept live inference.
* **Curl Command**:
  ```bash
  curl http://localhost:8000/ready
  ```
* **Response Schema**:
  ```json
  {
    "ready": true,
    "onnx_provider": "cpu",
    "model_loaded": true
  }
  ```

### 4.3 Endpoint: `GET /metrics`
* **Protocol & Method**: `HTTP GET`
* **URL**: `http://localhost:8000/metrics`
* **Description**: Real-time system telemetry reporting active call count, total inferences processed, average ONNX latency, and classification counts.
* **Curl Command**:
  ```bash
  curl http://localhost:8000/metrics
  ```
* **Response Schema**:
  ```json
  {
    "active_calls": 0,
    "total_calls": 12,
    "inferences_total": 45,
    "avg_inference_latency_ms": 94.75,
    "ai_predictions": 2,
    "human_predictions": 40,
    "unknown_predictions": 3
  }
  ```

### 4.4 Endpoint: `WS /ws/call/{call_id}`
* **Protocol**: `WebSocket`
* **URL**: `ws://localhost:8000/ws/call/{call_id}`
* **Description**: Streaming WebSocket endpoint for telephone calls. Receives binary 16-bit PCM audio frames, processes speech through unblocked VAD and sliding window buffer, and returns JSON detection events.
* **Path Parameter**: `call_id` — Unique string identifier for the call session (e.g. `call-abc123`).
* **Input Data**: Binary 16-bit PCM mono audio bytes (e.g. 8000Hz or 16000Hz streamed in 100ms–500ms chunks).
* **Output Event JSON**:
  ```json
  {
    "type": "detection",
    "call_id": "call-abc123",
    "status": "ai",
    "confidence": 0.9412,
    "raw_ai_score": 0.9520,
    "timestamp_ms": 3000,
    "inference_latency_ms": 94.75,
    "speech_detected": true
  }
  ```

#### JSON Response Field Definitions:
- **`status`**: `"ai"` (AI voice confirmed), `"human"` (Human voice confirmed), or `"unknown"` (Ambiguous score or unconfirmed).
- **`confidence`**: Exponential Moving Average (EMA) smoothed score between `0.0` (100% Human) and `1.0` (100% AI).
- **`raw_ai_score`**: Instantaneous raw ONNX probability output for current 3-second audio window.
- **`timestamp_ms`**: Current position in call audio stream in milliseconds.
- **`inference_latency_ms`**: Time spent in ONNX Runtime execution (in milliseconds).
- **`speech_detected`**: Boolean flag confirming valid speech presence.

---

## 5. Component Breakdown

### 5.1 Audio Resampler ([app/audio/resampler.py](file:///d:/muthu/app/audio/resampler.py))
- Converts 16-bit PCM integer byte buffers or float32 arrays from input sample rate (e.g. 8000 Hz telephone rate) to 16,000 Hz float32 array.
- Uses `scipy.signal.resample_poly` for fast polyphase resampling.
- **Zero-disk I/O**: Performs all audio operations in RAM memory.

### 5.2 Audio Preprocessor ([app/audio/preprocessing.py](file:///d:/muthu/app/audio/preprocessing.py))
- Formats 16kHz float32 audio arrays into `(1, sequence_length)` tensor shape.
- Applies Hugging Face `Wav2Vec2FeatureExtractor` zero-mean unit-variance normalization semantics.

### 5.3 Voice Activity Detector ([app/audio/vad.py](file:///d:/muthu/app/audio/vad.py))
- Filters non-speech frames and background silence using Silero VAD ONNX model.
- Includes energy-based RMS fallback if VAD ONNX is absent.

### 5.4 Sliding Speech Buffer ([app/audio/buffer.py](file:///d:/muthu/app/audio/buffer.py))
- Accumulates active speech frames into a sliding window.
- Yields 3.0s speech audio arrays every 1.0s stride once initial 3.0s speech is reached.
- Maintains speech context across short pauses (<1.5s).
- Clears buffer context on long silence pauses (≥1.5s).

### 5.5 ONNX Model Session ([app/inference/onnx_model.py](file:///d:/muthu/app/inference/onnx_model.py))
- Singleton `VoiceDetector` class loading `models/detector/model.onnx` ONNX session once on startup.
- Supports auto-provider switching (`CPUExecutionProvider` / `CUDAExecutionProvider`).
- Configurable intra/inter op thread counts (`ORT_INTRA_OP_THREADS`, `ORT_INTER_OP_THREADS`).
- Executes warmup dummy inference requests before marking readiness.

### 5.6 Decision Engine ([app/inference/decision.py](file:///d:/muthu/app/inference/decision.py))
- Applies Exponential Moving Average (EMA) smoothing:
  $$\text{EMA} = \alpha \cdot \text{Score}_{\text{raw}} + (1 - \alpha) \cdot \text{EMA}_{\text{prev}}$$
- Hysteresis thresholds: `AI_THRESHOLD = 0.90`, `HUMAN_THRESHOLD = 0.10`.
- Requires `AI_CONFIRMATIONS = 2` consecutive windows ≥ 0.90 to transition status to `"ai"`.
- Requires `HUMAN_CONFIRMATIONS = 3` consecutive windows ≤ 0.10 to transition status to `"human"`.
- Emits status `"unknown"` for ambiguous or unconfirmed predictions.

### 5.7 Per-Call Pipeline & WebSocket Receiver ([app/api/websocket.py](file:///d:/muthu/app/api/websocket.py))
- WebSocket endpoint `/ws/call/{call_id}`.
- Decouples network frame receipt loop from call worker loop using an `asyncio.Queue`.
- Each call maintains isolated state (`CallDetectorPipeline`) so call streams cannot leak data between callers.

---

## 6. Directory Structure

```
d:/muthu/
├── app/
│   ├── main.py                   # FastAPI server entrypoint & endpoints
│   ├── config.py                 # Application settings & environment configuration
│   ├── api/
│   │   └── websocket.py          # WebSocket route handler & call metrics
│   ├── audio/
│   │   ├── resampler.py          # PCM16 & sample rate converter
│   │   ├── preprocessing.py      # Wav2Vec2 feature normalization
│   │   ├── vad.py                # Voice activity detector (Silero ONNX)
│   │   └── buffer.py             # 3s window / 1s stride sliding speech buffer
│   ├── inference/
│   │   ├── onnx_model.py         # Singleton ONNX model session & warmup
│   │   └── decision.py           # EMA aggregator & hysteresis state machine
│   ├── pipeline/
│   │   └── call_detector.py      # Per-call processing pipeline state
│   └── schemas/
│       └── responses.py          # Pydantic response data schemas
├── scripts/
│   ├── export_onnx.py            # Model exporter (HF -> ONNX FP32)
│   ├── validate_onnx.py          # PyTorch vs ONNX numerical validation
│   ├── quantize.py               # Dynamic INT8 ONNX quantizer
│   ├── benchmark.py              # System latency benchmark script
│   ├── infer_audio.py            # Standalone audio file inference script
│   └── test_client.py            # WebSocket call simulator script
├── tests/                        # Pytest suite (9 tests)
├── models/
│   ├── detector/                 # Stores model.onnx & model_int8.onnx
│   └── vad/                      # Stores silero_vad.onnx
├── Dockerfile                    # Docker build configuration
├── docker-compose.yml            # Docker container compose
├── requirements.txt              # Project dependencies
├── .env.example                  # Environment settings template
├── README.md                     # Technical overview & deployment guide
└── SYSTEM_SPECIFICATION.md       # Complete system specification
```

---

## 7. Performance Benchmarks & Validation Results

### 7.1 ONNX vs PyTorch Validation
Tested on identical 3-second audio arrays via [scripts/validate_onnx.py](file:///d:/muthu/scripts/validate_onnx.py):
- **Max Absolute Logit Difference**: `0.000013`
- **PyTorch AI Probability**: `0.0321`
- **ONNX AI Probability**: `0.0321`
- **Equivalence Status**: **PASSED**

### 7.2 Latency Benchmark Results ([scripts/benchmark.py](file:///d:/muthu/scripts/benchmark.py))
Measured over 3.0-second audio window inputs on CPU:

| Stage / Precision | Mean Latency | P50 Latency | P95 Latency | P99 Latency |
|---|---|---|---|---|
| **VAD Latency** | 0.05 ms | - | - | - |
| **Preprocessing Latency** | 13.43 ms | - | - | - |
| **ONNX FP32 (CPU)** | 167.33 ms | 164.81 ms | 188.08 ms | 194.57 ms |
| **ONNX INT8 (CPU)** | **94.75 ms** | **93.76 ms** | **104.99 ms** | **109.11 ms** |

---

## 8. How to Infer the Model (Code Examples)

### 8.1 Standalone Python Audio File Inference
```bash
python scripts/infer_audio.py your_audio_file.wav
```

### 8.2 Python Direct ONNX Code Example
```python
import soundfile as sf
from app.audio.resampler import AudioResampler
from app.audio.preprocessing import AudioPreprocessor
from app.inference.onnx_model import VoiceDetector

# Load model session once
detector = VoiceDetector(model_path="models/detector/model.onnx")
detector.load_model()

# Load & resample audio to 16kHz float32
audio_data, sample_rate = sf.read("my_audio.wav", dtype="float32")
resampler = AudioResampler(target_sample_rate=16000)
audio_16khz = resampler.resample(audio_data, source_sample_rate=sample_rate)

# Take 3.0s window (48,000 samples)
chunk_3s = audio_16khz[0:48000]

# Preprocess & run ONNX inference
preprocessor = AudioPreprocessor()
input_tensor = preprocessor.preprocess(chunk_3s)
ai_prob, latency_ms = detector.predict(input_tensor)

print(f"AI Probability: {ai_prob:.4f} (Inference time: {latency_ms:.2f} ms)")
```

### 8.3 Python WebSocket Client Example
```python
import asyncio
import json
import websockets

async def stream_live_call():
    uri = "ws://localhost:8000/ws/call/call-abc123"
    async with websockets.connect(uri) as websocket:
        print("WebSocket connected!")
        while call_is_active:
            pcm_bytes = get_100ms_pcm16_audio_chunk()
            await websocket.send(pcm_bytes)

            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                event = json.loads(msg)
                print(f"Status: {event['status'].upper()}, Confidence: {event['confidence']}")
            except asyncio.TimeoutError:
                pass

asyncio.run(stream_live_call())
```

---

## 9. Terminal Commands Quick Reference

```bash
# 1. Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 2. Run WebSocket stream test client
python scripts/test_client.py

# 3. Infer audio file directly
python scripts/infer_audio.py sample_audio.wav

# 4. Run unit test suite (9 tests)
python -m pytest

# 5. Validate PyTorch vs ONNX equivalence
python scripts/validate_onnx.py

# 6. Quantize ONNX model to INT8
python scripts/quantize.py

# 7. Run system latency benchmark
python scripts/benchmark.py

# 8. Export ONNX model
python scripts/export_onnx.py
```

---

## 10. Testing Datasets for Real vs AI Voices

To evaluate real human speakers vs synthetic AI voice clones:
- **AI Synthetic Voice Datasets**: [ASVspoof 2019/2021](https://www.asvspoof.org/), [WaveFake Dataset](https://github.com/dineshratnani/WaveFake), ElevenLabs API samples.
- **Human Voice Datasets**: [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), [LibriSpeech](https://www.openslr.org/12), [Mozilla Common Voice](https://commonvoice.mozilla.org/).
