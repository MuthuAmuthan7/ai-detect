import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.inference.onnx_model import voice_detector
from app.api.websocket import router as websocket_router, active_calls, metrics_counters
from app.schemas.responses import HealthResponse, ReadyResponse, MetricsResponse

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ai_voice_detector")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing AI Voice Detector Server...")
    # Attempt ONNX model loading and warmup
    success = voice_detector.load_model()
    if not success:
        logger.warning("ONNX model not found. Run 'python scripts/export_onnx.py' to generate model.onnx")

    yield

    logger.info("Shutting down AI Voice Detector Server...")


app = FastAPI(
    title="Real-Time AI Voice Detection API",
    description="High-performance low-latency voice deepfake classifier using ONNX Runtime, VAD, and WebSockets.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for public IP and proxy connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include WebSocket router
app.include_router(websocket_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Liveness check endpoint."""
    return HealthResponse(
        status="ok",
        active_calls=len(active_calls)
    )


@app.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Readiness check endpoint - verifies ONNX model initialization & warmup."""
    return ReadyResponse(
        ready=voice_detector.is_ready,
        onnx_provider=voice_detector.provider,
        model_loaded=voice_detector.session is not None
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics_endpoint():
    """Prometheus / system metrics summary endpoint."""
    total_inferences = metrics_counters["total_inferences"]
    avg_latency = (
        metrics_counters["total_latency_ms"] / total_inferences
        if total_inferences > 0 else 0.0
    )
    return MetricsResponse(
        active_calls=len(active_calls),
        total_calls=metrics_counters["total_calls"],
        inferences_total=total_inferences,
        avg_inference_latency_ms=round(avg_latency, 2),
        ai_predictions=metrics_counters["ai_predictions"],
        human_predictions=metrics_counters["human_predictions"],
        unknown_predictions=metrics_counters["unknown_predictions"]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
