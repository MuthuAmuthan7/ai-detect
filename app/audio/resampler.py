import numpy as np
from scipy import signal


class AudioResampler:
    """
    In-memory resampler converting incoming PCM bytes or audio arrays
    from an arbitrary input sample rate (e.g., 8000 Hz) to target sample rate (16000 Hz float32).
    """

    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate

    def pcm16_bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        """
        Converts raw 16-bit PCM integer byte buffer to float32 normalized [-1.0, 1.0].
        """
        if not pcm_bytes:
            return np.array([], dtype=np.float32)
        int16_arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        return int16_arr.astype(np.float32) / 32768.0

    def resample(self, audio_array: np.ndarray, source_sample_rate: int) -> np.ndarray:
        """
        Resamples a float32 1D audio array from source_sample_rate to self.target_sample_rate.
        Uses scipy.signal.resample_poly for speed and accuracy.
        """
        if len(audio_array) == 0:
            return np.array([], dtype=np.float32)

        # Force float32 mono
        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        audio_array = audio_array.astype(np.float32)

        if source_sample_rate == self.target_sample_rate:
            return audio_array

        # Compute polyphase resampling factors
        gcd = np.gcd(source_sample_rate, self.target_sample_rate)
        up = self.target_sample_rate // gcd
        down = source_sample_rate // gcd

        resampled = signal.resample_poly(audio_array, up, down)
        return resampled.astype(np.float32)

    def process_raw_pcm_chunk(self, raw_bytes: bytes, source_sample_rate: int) -> np.ndarray:
        """
        Converts PCM16 bytes directly to 16kHz float32 audio numpy array.
        """
        float32_audio = self.pcm16_bytes_to_float32(raw_bytes)
        return self.resample(float32_audio, source_sample_rate)
