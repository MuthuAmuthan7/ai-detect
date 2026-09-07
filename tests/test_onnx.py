import pytest
import os
import numpy as np
from app.inference.onnx_model import VoiceDetector
from app.config import settings


def test_onnx_voice_detector_dummy():
    detector = VoiceDetector(model_path="models/detector/model.onnx")
    
    if os.path.exists("models/detector/model.onnx"):
        loaded = detector.load_model()
        assert loaded
        assert detector.is_ready

        dummy_tensor = np.zeros((1, settings.window_samples), dtype=np.float32)
        ai_prob, latency = detector.predict(dummy_tensor)

        assert 0.0 <= ai_prob <= 1.0
        assert latency >= 0.0
