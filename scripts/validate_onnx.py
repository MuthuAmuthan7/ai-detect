import sys
import numpy as np
import torch
from pathlib import Path
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def validate_onnx_numerical_equivalence(tolerance: float = 1e-3):
    model_name = settings.HF_MODEL_NAME
    onnx_path = Path("models/detector/model.onnx")

    if not onnx_path.exists():
        print(f"ERROR: ONNX model not found at {onnx_path}. Run 'python scripts/export_onnx.py' first.")
        sys.exit(1)

    print(f"--> Loading PyTorch model {model_name}...")
    pytorch_model = AutoModelForAudioClassification.from_pretrained(model_name)
    pytorch_model.eval()

    print(f"--> Loading ONNX model from {onnx_path}...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # Generate synthetic 3-second 16kHz audio sample
    np.random.seed(42)
    sample_audio = np.random.uniform(-0.5, 0.5, size=(1, settings.window_samples)).astype(np.float32)

    # 1. Run PyTorch Inference
    with torch.no_grad():
        pt_input = torch.from_numpy(sample_audio)
        pt_logits = pytorch_model(pt_input).logits.numpy()
        pt_probs = torch.softmax(torch.from_numpy(pt_logits), dim=-1).numpy()[0]

    # 2. Run ONNX Runtime Inference
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: sample_audio})
    onnx_logits = onnx_output[0]
    
    exp_logits = np.exp(onnx_logits - np.max(onnx_logits))
    onnx_probs = (exp_logits / np.sum(exp_logits))[0]

    max_abs_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
    prob_diff = float(np.max(np.abs(pt_probs - onnx_probs)))

    print("\n--------------------------------------------------")
    print(f"PyTorch logits: {pt_logits[0].tolist()}")
    print(f"ONNX logits:    {onnx_logits[0].tolist()}")
    print(f"Max Absolute Logit Difference: {max_abs_diff:.6f}")
    print(f"PyTorch AI Probability: {pt_probs[1]:.4f}")
    print(f"ONNX AI Probability:    {onnx_probs[1]:.4f}")
    print("--------------------------------------------------")

    if max_abs_diff <= tolerance:
        print("Prediction Match: YES (Numerical equivalence validated within tolerance)")
    else:
        print(f"Prediction Match: FAIL (Difference {max_abs_diff:.6f} exceeds tolerance {tolerance})")
        sys.exit(1)


if __name__ == "__main__":
    validate_onnx_numerical_equivalence()
