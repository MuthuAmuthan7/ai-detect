import asyncio
import json
import numpy as np
import websockets


async def simulate_call_stream(call_id: str = "test-call-101", server_url: str = "ws://127.0.0.1:8000"):
    uri = f"{server_url}/ws/call/{call_id}"
    print(f"Connecting to WebSocket: {uri}")

    async with websockets.connect(uri) as websocket:
        print("--> WebSocket connected! Simulating 8kHz PCM audio stream...")

        # Generate 10 seconds of synthetic 8kHz PCM16 audio (100ms chunks = 800 samples = 1600 bytes)
        sample_rate = 8000
        chunk_duration_sec = 0.1
        samples_per_chunk = int(sample_rate * chunk_duration_sec)

        for second in range(1, 10):
            print(f"\n[Streaming Second {second}/10] Sending audio chunks...")
            for _ in range(10):  # 10 x 100ms = 1 second
                # Generate audio signal (simulating voice waveform)
                time_samples = np.linspace(0, chunk_duration_sec, samples_per_chunk, endpoint=False)
                audio_signal = (np.sin(2 * np.pi * 440 * time_samples) * 15000).astype(np.int16)
                pcm_bytes = audio_signal.tobytes()

                # Send raw binary PCM audio bytes over WebSocket
                await websocket.send(pcm_bytes)
                await asyncio.sleep(0.05)  # Real-time streaming pace

                # Check if server emitted a detection event
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    event = json.loads(message)
                    print("  >>> DETECTION EVENT RECEIVED FROM SERVER:")
                    print(f"      Status:            {event.get('status').upper()}")
                    print(f"      Confidence:        {event.get('confidence')}")
                    print(f"      Raw AI Score:      {event.get('raw_ai_score')}")
                    print(f"      Timestamp:         {event.get('timestamp_ms')} ms")
                    print(f"      Inference Latency: {event.get('inference_latency_ms')} ms")
                except asyncio.TimeoutError:
                    pass

        print("\n--> Call streaming finished. Closing connection.")


if __name__ == "__main__":
    asyncio.run(simulate_call_stream())
