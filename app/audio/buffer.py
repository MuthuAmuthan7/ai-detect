import numpy as np
from typing import Optional, List
from app.config import settings


class AudioBuffer:
    """
    Streaming sliding audio ring buffer for 16kHz speech.
    Accumulates incoming speech frames, buffers across short pauses,
    and yields 3-second audio windows with 1-second strides.
    """

    def __init__(
        self,
        sample_rate: int = settings.AUDIO_SAMPLE_RATE,
        window_seconds: float = settings.WINDOW_SECONDS,
        stride_seconds: float = settings.STRIDE_SECONDS,
        long_silence_ms: int = settings.LONG_SILENCE_MS
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.stride_samples = int(sample_rate * stride_seconds)
        self.long_silence_samples = int(sample_rate * (long_silence_ms / 1000.0))

        self.buffer = np.array([], dtype=np.float32)
        self.accumulated_since_last_stride = 0
        self.consecutive_silence_samples = 0
        self.total_samples_processed = 0

    def add_frame(self, frame_16khz: np.ndarray, is_speech: bool) -> List[np.ndarray]:
        """
        Adds a 16kHz float32 audio frame to the buffer.
        Returns a list of 3.0-second audio windows ready for ONNX model inference.
        """
        if len(frame_16khz) == 0:
            return []

        frame_len = len(frame_16khz)
        self.total_samples_processed += frame_len

        if not is_speech:
            self.consecutive_silence_samples += frame_len
            # Reset buffer if long silence pause detected
            if self.consecutive_silence_samples >= self.long_silence_samples:
                self.reset()
            return []

        # Active speech detected
        self.consecutive_silence_samples = 0
        self.buffer = np.concatenate([self.buffer, frame_16khz.astype(np.float32)])
        self.accumulated_since_last_stride += frame_len

        ready_windows: List[np.ndarray] = []

        # Extract windows if we have at least 3 seconds of speech audio
        while len(self.buffer) >= self.window_samples:
            # Check if we reached a 1-second stride boundary (or initial 3s window)
            window = self.buffer[: self.window_samples].copy()
            ready_windows.append(window)

            # Slide buffer forward by stride_samples
            self.buffer = self.buffer[self.stride_samples:]
            self.accumulated_since_last_stride = max(0, self.accumulated_since_last_stride - self.stride_samples)

        return ready_windows

    def reset(self):
        """Resets the buffer and counters."""
        self.buffer = np.array([], dtype=np.float32)
        self.accumulated_since_last_stride = 0
        self.consecutive_silence_samples = 0

    @property
    def current_buffer_duration_seconds(self) -> float:
        return len(self.buffer) / float(self.sample_rate)
