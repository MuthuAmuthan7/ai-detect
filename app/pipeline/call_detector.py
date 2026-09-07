import time
import numpy as np
from typing import List, Optional
from app.audio.resampler import AudioResampler
from app.audio.preprocessing import AudioPreprocessor
from app.audio.vad import VoiceActivityDetector
from app.audio.buffer import AudioBuffer
from app.inference.onnx_model import voice_detector
from app.inference.decision import DecisionEngine
from app.schemas.responses import DetectionResult, WebSocketDetectionEvent
from app.config import settings


class CallDetectorPipeline:
    """
    Per-call streaming detector pipeline instance.
    Encapsulates audio resampling, VAD filtering, sliding window buffering,
    ONNX model inference execution, and EMA state decisions.
    """

    def __init__(self, call_id: str, source_sample_rate: int = settings.INPUT_SAMPLE_RATE):
        self.call_id = call_id
        self.source_sample_rate = source_sample_rate
        self.resampler = AudioResampler(target_sample_rate=settings.AUDIO_SAMPLE_RATE)
        self.preprocessor = AudioPreprocessor(model_name_or_path=settings.HF_MODEL_NAME)
        self.vad = VoiceActivityDetector()
        self.buffer = AudioBuffer()
        self.decision_engine = DecisionEngine()

        self.start_time = time.time()
        self.total_audio_duration_ms: int = 0
        self.inferences_count: int = 0
        self.last_inference_latency_ms: float = 0.0

    def process_pcm_chunk(self, raw_bytes: bytes) -> List[WebSocketDetectionEvent]:
        """
        Processes a raw PCM byte chunk from the call's WebSocket audio stream.
        Returns a list of WebSocketDetectionEvents if audio window inferences were triggered.
        """
        if not raw_bytes:
            return []

        # 1. Resample to 16kHz float32
        audio_16khz = self.resampler.process_raw_pcm_chunk(raw_bytes, self.source_sample_rate)
        if len(audio_16khz) == 0:
            return []

        chunk_duration_ms = int((len(audio_16khz) / settings.AUDIO_SAMPLE_RATE) * 1000)
        self.total_audio_duration_ms += chunk_duration_ms

        # 2. Perform Voice Activity Detection (VAD)
        speech_detected = self.vad.is_speech(audio_16khz)

        # 3. Add to sliding speech buffer
        windows = self.buffer.add_frame(audio_16khz, is_speech=speech_detected)

        events: List[WebSocketDetectionEvent] = []

        # 4. Infer ONNX model on ready 3.0s speech audio windows
        for audio_window in windows:
            # Check if model is loaded
            if not voice_detector.is_ready:
                # Fallback if ONNX model is still loading/not exported yet
                raw_score = 0.5
                latency = 1.0
            else:
                input_tensor = self.preprocessor.preprocess(audio_window)
                raw_score, latency = voice_detector.predict(input_tensor)

            self.inferences_count += 1
            self.last_inference_latency_ms = latency

            # 5. Update EMA smoothing & decision hysteresis
            status, confidence = self.decision_engine.update(raw_score)

            event = WebSocketDetectionEvent(
                call_id=self.call_id,
                status=status,
                confidence=confidence,
                raw_ai_score=round(raw_score, 4),
                timestamp_ms=self.total_audio_duration_ms,
                inference_latency_ms=round(latency, 2),
                speech_detected=True
            )
            events.append(event)

        return events

    def cleanup(self):
        """Releases per-call buffers and resources."""
        self.buffer.reset()
        self.decision_engine.reset()
