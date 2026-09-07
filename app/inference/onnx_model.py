import os
import time
import logging
import numpy as np
import onnxruntime as ort
from typing import Tuple, List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class VoiceDetector:
    """
    Singleton ONNX Runtime Inference engine for Wav2Vec2 Audio Classifier.
    Loaded once at server startup and shared across concurrent calls.
    """

    def __init__(self, model_path: str = settings.MODEL_PATH, provider: str = settings.ONNX_PROVIDER):
        self.model_path = model_path
        self.provider = provider
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.is_ready: bool = False

    def load_model(self) -> bool:
        """
        Loads the ONNX model and configures session options and providers.
        """
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX model file not found at {self.model_path}. Model loading deferred.")
            return False

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
            opts.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            available_providers = ort.get_available_providers()
            providers: List[str] = []

            if self.provider.lower() == "cuda" and "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            logger.info(f"Loading ONNX model from {self.model_path} with providers: {providers}")
            self.session = ort.InferenceSession(self.model_path, opts, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

            # Perform model warmup
            self.warmup()
            self.is_ready = True
            logger.info("ONNX VoiceDetector model loaded and warmed up successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}", exc_info=True)
            self.is_ready = False
            return False

    def warmup(self, num_runs: int = 3):
        """
        Runs dummy float32 audio inference requests to warm up execution engine.
        """
        if self.session is None:
            return

        dummy_audio = np.zeros((1, settings.window_samples), dtype=np.float32)
        for _ in range(num_runs):
            self.session.run([self.output_name], {self.input_name: dummy_audio})

    def predict(self, input_tensor: np.ndarray) -> Tuple[float, float]:
        """
        Performs model inference on a normalized 16kHz audio tensor (1, num_samples).
        Returns:
            Tuple[ai_probability, latency_ms]
        """
        if self.session is None:
            raise RuntimeError("ONNX session is not initialized. Model not loaded.")

        start_time = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        logits = outputs[0][0]  # shape (num_classes,) e.g. [human_logit, ai_logit] or [ai_logit, human_logit]
        
        # Apply Softmax to obtain probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Assumes index 1 is AI/synthetic class, index 0 is Human class
        # (or max prob if binary classifier format)
        if len(probs) >= 2:
            ai_probability = float(probs[1])
        else:
            # Sigmoid if single output logit
            ai_probability = float(1.0 / (1.0 + np.exp(-logits[0])))

        return ai_probability, latency_ms


# Global singleton instance
voice_detector = VoiceDetector()
