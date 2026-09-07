import numpy as np
from transformers import AutoFeatureExtractor
from app.config import settings


class AudioPreprocessor:
    """
    Handles Hugging Face Wav2Vec2 feature extraction and normalization semantics.
    Converts 16kHz mono float32 audio array into normalized input tensor (1, seq_len).
    """

    def __init__(self, model_name_or_path: str = settings.HF_MODEL_NAME):
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        except Exception:
            # Fallback if offline or Hugging Face unavailable
            self.feature_extractor = None

    def preprocess(self, audio_16khz: np.ndarray) -> np.ndarray:
        """
        Normalizes float32 audio waveform for Wav2Vec2 model input.
        Returns array of shape (1, num_samples), float32.
        """
        if len(audio_16khz) == 0:
            raise ValueError("Empty audio input supplied to preprocessor.")

        # Ensure float32 1D
        audio_flat = audio_16khz.astype(np.float32).flatten()

        if self.feature_extractor is not None:
            processed = self.feature_extractor(
                audio_flat,
                sampling_rate=settings.AUDIO_SAMPLE_RATE,
                return_tensors="np"
            )
            input_values = processed.input_values.astype(np.float32)
            return input_values

        # Standard Wav2Vec2 zero-mean, unit-variance normalization fallback
        mean = np.mean(audio_flat)
        std = np.std(audio_flat)
        normalized = (audio_flat - mean) / (std + 1e-7)
        return np.expand_dims(normalized, axis=0).astype(np.float32)
