import os
import sys
import torch
from pathlib import Path
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings


def export_model_to_onnx():
    model_name = settings.HF_MODEL_NAME
    output_dir = Path("models/detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"

    print(f"--> Loading Hugging Face model: {model_name}")
    processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()

    # Create 3-second 16kHz dummy input
    dummy_input = torch.randn(1, settings.window_samples, dtype=torch.float32)

    print(f"--> Exporting Wav2Vec2 model to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={
            "input_values": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"}
        }
    )

    processor.save_pretrained(str(output_dir))
    print(f"Successfully exported ONNX model to {onnx_path}")

    # Silero VAD download
    vad_dir = Path("models/vad")
    vad_dir.mkdir(parents=True, exist_ok=True)
    vad_path = vad_dir / "silero_vad.onnx"

    if not vad_path.exists() or vad_path.stat().st_size == 0:
        print(f"--> Downloading Silero VAD ONNX model to {vad_path}...")
        try:
            import urllib.request
            url = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
            urllib.request.urlretrieve(url, str(vad_path))
            print(f"Successfully downloaded Silero VAD model to {vad_path}")
        except Exception as e:
            print(f"Could not download Silero VAD model: {e}")


if __name__ == "__main__":
    export_model_to_onnx()
