import pytest
import numpy as np
from app.audio.buffer import AudioBuffer
from app.config import settings


def test_buffer_sliding_window_stride():
    buffer = AudioBuffer(sample_rate=16000, window_seconds=3.0, stride_seconds=1.0)
    
    # 1 second of speech (16000 samples)
    speech_1s = np.random.uniform(-0.2, 0.2, size=(16000,)).astype(np.float32)

    # Adding 1s & 2s speech -> should yield NO window (less than 3s)
    w1 = buffer.add_frame(speech_1s, is_speech=True)
    assert len(w1) == 0

    w2 = buffer.add_frame(speech_1s, is_speech=True)
    assert len(w2) == 0

    # Adding 3rd second of speech -> should yield 1 window of exactly 48000 samples (3.0s)
    w3 = buffer.add_frame(speech_1s, is_speech=True)
    assert len(w3) == 1
    assert len(w3[0]) == 48000

    # Adding 4th second of speech -> should yield 2nd sliding window (1.0s stride)
    w4 = buffer.add_frame(speech_1s, is_speech=True)
    assert len(w4) == 1
    assert len(w4[0]) == 48000


def test_buffer_long_silence_reset():
    buffer = AudioBuffer(sample_rate=16000, window_seconds=3.0, stride_seconds=1.0, long_silence_ms=1500)
    speech_2s = np.random.uniform(-0.2, 0.2, size=(32000,)).astype(np.float32)
    silence_2s = np.zeros((32000,), dtype=np.float32)

    buffer.add_frame(speech_2s, is_speech=True)
    assert len(buffer.buffer) == 32000

    # Long silence should trigger buffer reset
    buffer.add_frame(silence_2s, is_speech=False)
    assert len(buffer.buffer) == 0
