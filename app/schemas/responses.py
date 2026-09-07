from typing import Optional, Literal
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    status: Literal["ai", "human", "unknown"] = Field(..., description="Detection status: ai, human, or unknown")
    confidence: float = Field(..., description="Smoother score / confidence level between 0.0 and 1.0")
    raw_ai_score: float = Field(..., description="Instantaneous model output score for AI class")
    timestamp_ms: int = Field(..., description="Audio position timestamp in milliseconds")
    window_duration_ms: int = Field(..., description="Audio window length evaluated in milliseconds")
    inference_latency_ms: float = Field(..., description="ONNX model inference duration in milliseconds")
    speech_detected: bool = Field(..., description="Whether valid speech was processed in this frame")


class WebSocketDetectionEvent(BaseModel):
    type: str = "detection"
    call_id: str
    status: Literal["ai", "human", "unknown"]
    confidence: float
    raw_ai_score: float
    timestamp_ms: int
    inference_latency_ms: float
    speech_detected: bool = True


class HealthResponse(BaseModel):
    status: str = "ok"
    active_calls: int


class ReadyResponse(BaseModel):
    ready: bool
    onnx_provider: str
    model_loaded: bool


class MetricsResponse(BaseModel):
    active_calls: int
    total_calls: int
    inferences_total: int
    avg_inference_latency_ms: float
    ai_predictions: int
    human_predictions: int
    unknown_predictions: int
