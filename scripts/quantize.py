import sys
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def quantize_onnx_model():
    input_path = Path("models/detector/model.onnx")
    output_path = Path("models/detector/model_int8.onnx")

    if not input_path.exists():
        print(f"ERROR: Base ONNX model not found at {input_path}. Run 'python scripts/export_onnx.py' first.")
        sys.exit(1)

    print(f"--> Performing INT8 Dynamic Quantization on {input_path}...")
    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8
    )
    print(f"Successfully generated INT8 quantized model at {output_path}!")


if __name__ == "__main__":
    quantize_onnx_model()
