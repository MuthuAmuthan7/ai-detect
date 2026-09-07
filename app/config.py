import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: str = "models/detector/model.onnx"
    VAD_MODEL_PATH: str = "models/vad/silero_vad.onnx"
    HF_MODEL_NAME: str = "garystafford/wav2vec2-deepfake-voice-detector"

    # Audio Sampling Settings
    AUDIO_SAMPLE_RATE: int = 16000
    INPUT_SAMPLE_RATE: int = 8000
    WINDOW_SECONDS: float = 3.0
    STRIDE_SECONDS: float = 1.0

    # VAD Settings
    VAD_THRESHOLD: float = 0.5
    MIN_SPEECH_DURATION_MS: int = 300
    MIN_SILENCE_DURATION_MS: int = 300
    SHORT_SILENCE_MS: int = 300
    LONG_SILENCE_MS: int = 1500

    # Thresholds & Aggregation
    AI_THRESHOLD: float = 0.90
    HUMAN_THRESHOLD: float = 0.10
    EMA_ALPHA: float = 0.4
    AI_CONFIRMATIONS: int = 2
    HUMAN_CONFIRMATIONS: int = 3

    # ONNX Runtime Execution
    ONNX_PROVIDER: str = "cpu"  # "cpu" or "cuda"
    ORT_INTRA_OP_THREADS: int = 4
    ORT_INTER_OP_THREADS: int = 1

    # Server Concurrency
    MAX_ACTIVE_CALLS: int = 100
    MAX_AUDIO_QUEUE_SIZE: int = 1000
    MAX_INFERENCE_QUEUE_SIZE: int = 100

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def window_samples(self) -> int:
        return int(self.AUDIO_SAMPLE_RATE * self.WINDOW_SECONDS)

    @property
    def stride_samples(self) -> int:
        return int(self.AUDIO_SAMPLE_RATE * self.STRIDE_SECONDS)


settings = Settings()
