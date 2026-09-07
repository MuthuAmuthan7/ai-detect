import os
import numpy as np
import onnxruntime as ort
from app.config import settings


class VoiceActivityDetector:
    """
    Voice Activity Detection (VAD) using Silero VAD (ONNX) with fallback to energy-based VAD.
    Determines whether audio frames contain human/synthetic speech or background silence.
    """

    def __init__(self, model_path: str = settings.VAD_MODEL_PATH, threshold: float = settings.VAD_THRESHOLD):
        self.model_path = model_path
        self.threshold = threshold
        self.session = None
        self._init_vad_session()

    def _init_vad_session(self):
        if os.path.exists(self.model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                self.session = ort.InferenceSession(self.model_path, opts, providers=["CPUExecutionProvider"])
            except Exception:
                self.session = None

    def is_speech(self, audio_16khz: np.ndarray) -> bool:
        """
        Evaluates whether a 16kHz float32 audio chunk contains active speech.
        """
        if len(audio_16khz) == 0:
            return False

        if self.session is not None:
            try:
                # Silero VAD ONNX expects input of shape (1, N) or chunked 512 samples
                audio_input = audio_16khz.astype(np.float32).reshape(1, -1)
                sr_input = np.array([settings.AUDIO_SAMPLE_RATE], dtype=np.int64)
                
                # Check Silero input names
                input_names = [i.name for i in self.session.get_inputs()]
                inputs = {input_names[0]: audio_input}
                if len(input_names) > 1 and "sr" in input_names[1].lower():
                    inputs[input_names[1]] = sr_input

                outputs = self.session.run(None, inputs)
                speech_prob = float(outputs[0][0][0])
                return speech_prob >= self.threshold
            except Exception:
                pass

        # Energy-based VAD fallback (Root Mean Square energy)
        rms = np.sqrt(np.mean(np.square(audio_16khz)))
        # RMS threshold ~ 0.015 for normalized float32 audio
        return bool(rms >= 0.012)

    def get_speech_probability(self, audio_16khz: np.ndarray) -> float:
        """
        Returns continuous speech probability [0.0, 1.0].
        """
        if len(audio_16khz) == 0:
            return 0.0

        if self.session is not None:
            try:
                audio_input = audio_16khz.astype(np.float32).reshape(1, -1)
                input_names = [i.name for i in self.session.get_inputs()]
                inputs = {input_names[0]: audio_input}
                if len(input_names) > 1:
                    inputs[input_names[1]] = np.array([settings.AUDIO_SAMPLE_RATE], dtype=np.int64)
                outputs = self.session.run(None, inputs)
                return float(outputs[0][0][0])
            except Exception:
                pass

        rms = float(np.sqrt(np.mean(np.square(audio_16khz))))
        return min(1.0, rms / 0.03)
