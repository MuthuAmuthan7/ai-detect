import sys
import numpy as np
import soundfile as sf
from pathlib import Path

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.audio.resampler import AudioResampler
from app.audio.preprocessing import AudioPreprocessor
from app.inference.onnx_model import VoiceDetector
from app.inference.decision import DecisionEngine


def infer_audio_file(audio_file_path: str):
    """
    Direct model inference on an audio file (WAV, MP3, FLAC, OGG).
    """
    print(f"--> Loading audio file: {audio_file_path}")
    audio_data, sample_rate = sf.read(audio_file_path, dtype="float32")

    # 1. Resample audio to 16kHz mono float32
    resampler = AudioResampler(target_sample_rate=16000)
    audio_16khz = resampler.resample(audio_data, source_sample_rate=sample_rate)

    print(f"--> Processed Audio: Duration = {len(audio_16khz) / 16000:.2f} seconds")

    # 2. Load ONNX Model Singleton
    detector = VoiceDetector(model_path="models/detector/model.onnx")
    if not detector.load_model():
        print("Error: Could not load ONNX model.")
        return

    preprocessor = AudioPreprocessor()
    decision_engine = DecisionEngine()

    # 3. Process audio in 3.0 second windows with 1.0 second stride
    window_samples = 48000  # 3s at 16kHz
    stride_samples = 16000  # 1s at 16kHz

    start_idx = 0
    print("\n------------------------------------------------------------")
    print(f"{'Time Window':<15} | {'Raw AI Score':<15} | {'EMA Score':<12} | {'Status':<10}")
    print("------------------------------------------------------------")

    while start_idx + window_samples <= len(audio_16khz):
        chunk_3s = audio_16khz[start_idx : start_idx + window_samples]
        
        # Preprocess waveform tensor (1, 48000)
        input_tensor = preprocessor.preprocess(chunk_3s)

        # Run ONNX model inference
        raw_ai_score, latency_ms = detector.predict(input_tensor)

        # Update EMA rolling score & status decision
        status, ema_score = decision_engine.update(raw_ai_score)

        window_str = f"{start_idx/16000:.1f}s - {(start_idx+window_samples)/16000:.1f}s"
        print(f"{window_str:<15} | {raw_ai_score:<15.4f} | {ema_score:<12.4f} | {status.upper():<10}")

        start_idx += stride_samples

    print("------------------------------------------------------------\n")
    print(f"Final Decision for Call: {decision_engine.current_status.upper()}")
    print(f"Final Smoothed Confidence: {decision_engine.ema_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        infer_audio_file(sys.argv[1])
    else:
        # Create dummy 5-second sample audio file for demo
        sample_path = "sample_test.wav"
        print(f"No audio file provided. Creating dummy 5-second sample audio file '{sample_path}'...")
        t = np.linspace(0, 5, 5 * 16000)
        dummy_voice = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
        sf.write(sample_path, dummy_voice, 16000)
        
        infer_audio_file(sample_path)
