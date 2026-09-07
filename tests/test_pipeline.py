import pytest
import numpy as np
from app.pipeline.call_detector import CallDetectorPipeline


def test_call_detector_pipeline_execution():
    pipeline = CallDetectorPipeline(call_id="test-call-1", source_sample_rate=8000)

    # Generate 4 seconds of 8kHz PCM16 speech bytes
    samples_8k = np.int16(np.sin(np.linspace(0, 500, 32000)) * 15000)
    pcm_bytes = samples_8k.tobytes()

    # Process PCM bytes
    events = pipeline.process_pcm_chunk(pcm_bytes)

    # Should have processed audio and produced detection events
    assert pipeline.total_audio_duration_ms == 4000
    assert len(events) >= 1
    assert events[0].call_id == "test-call-1"
    assert events[0].status in ["ai", "human", "unknown"]
    assert 0.0 <= events[0].confidence <= 1.0
