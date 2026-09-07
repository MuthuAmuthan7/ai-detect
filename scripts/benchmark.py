import sys
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoModelForAudioClassification
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings
from app.audio.vad import VoiceActivityDetector
from app.audio.preprocessing import AudioPreprocessor


def measure_latencies(runner_fn, sample_data, iterations: int = 50, warmup: int = 5) -> List[float]:
    latencies = []
    # Warmup
    for _ in range(warmup):
        runner_fn(sample_data)

    # Benchmark runs
    for _ in range(iterations):
        t0 = time.perf_counter()
        runner_fn(sample_data)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    return latencies


def run_benchmark(iterations: int = 50):
    print("==================================================")
    print(" REAL-TIME AI VOICE DETECTOR LATENCY BENCHMARK")
    print("==================================================")

    sample_rate = settings.AUDIO_SAMPLE_RATE
    audio_3s = np.random.uniform(-0.3, 0.3, size=(settings.window_samples,)).astype(np.float32)

    # 1. Pipeline Component Benchmarks
    vad = VoiceActivityDetector()
    preprocessor = AudioPreprocessor()

    vad_lats = measure_latencies(lambda a: vad.is_speech(a), audio_3s, iterations=iterations)
    prep_lats = measure_latencies(lambda a: preprocessor.preprocess(a), audio_3s, iterations=iterations)

    avg_vad = np.mean(vad_lats)
    avg_prep = np.mean(prep_lats)

    print(f"\n--- Component Latencies (3.0s Audio Input) ---")
    print(f"VAD Latency:            {avg_vad:.2f} ms")
    print(f"Preprocessing Latency:  {avg_prep:.2f} ms")

    # 2. Model Benchmarks
    onnx_fp32_path = Path("models/detector/model.onnx")
    onnx_int8_path = Path("models/detector/model_int8.onnx")

    results: Dict[str, List[float]] = {}

    # Benchmark ONNX FP32
    if onnx_fp32_path.exists():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
        opts.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
        session_fp32 = ort.InferenceSession(str(onnx_fp32_path), opts, providers=["CPUExecutionProvider"])
        inp_name = session_fp32.get_inputs()[0].name
        tensor_in = preprocessor.preprocess(audio_3s)

        lats = measure_latencies(lambda t: session_fp32.run(None, {inp_name: t}), tensor_in, iterations=iterations)
        results["ONNX FP32 (CPU)"] = lats

    # Benchmark ONNX INT8
    if onnx_int8_path.exists():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = settings.ORT_INTRA_OP_THREADS
        opts.inter_op_num_threads = settings.ORT_INTER_OP_THREADS
        session_int8 = ort.InferenceSession(str(onnx_int8_path), opts, providers=["CPUExecutionProvider"])
        inp_name = session_int8.get_inputs()[0].name
        tensor_in = preprocessor.preprocess(audio_3s)

        lats = measure_latencies(lambda t: session_int8.run(None, {inp_name: t}), tensor_in, iterations=iterations)
        results["ONNX INT8 (CPU)"] = lats

    # Print Detailed Benchmark Table
    print("\n----------------------------------------------------------------------")
    print(f"{'Engine / Precision':<22} | {'Mean (ms)':<10} | {'P50 (ms)':<10} | {'P95 (ms)':<10} | {'P99 (ms)':<10}")
    print("----------------------------------------------------------------------")

    for name, lats in results.items():
        mean_lat = np.mean(lats)
        p50 = np.percentile(lats, 50)
        p95 = np.percentile(lats, 95)
        p99 = np.percentile(lats, 99)
        print(f"{name:<22} | {mean_lat:<10.2f} | {p50:<10.2f} | {p95:<10.2f} | {p99:<10.2f}")

    print("----------------------------------------------------------------------\n")


if __name__ == "__main__":
    run_benchmark()
