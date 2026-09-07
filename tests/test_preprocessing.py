import pytest
import numpy as np
from app.audio.resampler import AudioResampler
from app.audio.preprocessing import AudioPreprocessor


def test_resampler_8k_to_16k():
    resampler = AudioResampler(target_sample_rate=16000)

    # 1 second of 8000 Hz audio = 8000 samples
    pcm8k = np.int16(np.sin(np.linspace(0, 100, 8000)) * 10000).tobytes()

    audio16k = resampler.process_raw_pcm_chunk(pcm8k, source_sample_rate=8000)
    assert isinstance(audio16k, np.ndarray)
    assert audio16k.dtype == np.float32
    assert len(audio16k) == 16000


def test_preprocessor_output_shape():
    preprocessor = AudioPreprocessor()
    audio_3s = np.random.uniform(-0.5, 0.5, size=(48000,)).astype(np.float32)

    tensor = preprocessor.preprocess(audio_3s)
    assert isinstance(tensor, np.ndarray)
    assert tensor.ndim == 2
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 48000
