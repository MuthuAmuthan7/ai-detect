import pytest
import numpy as np
from app.audio.vad import VoiceActivityDetector


def test_vad_speech_vs_silence():
    vad = VoiceActivityDetector()

    # Silence frame (all zeros)
    silence = np.zeros(16000, dtype=np.float32)
    assert not vad.is_speech(silence)

    # Active audio signal frame
    speech_signal = np.sin(np.linspace(0, 100, 16000)).astype(np.float32) * 0.5
    assert vad.is_speech(speech_signal)
