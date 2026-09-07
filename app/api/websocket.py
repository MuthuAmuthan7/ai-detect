import asyncio
import logging
import json
from typing import Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.pipeline.call_detector import CallDetectorPipeline
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Global tracking of active call pipelines
active_calls: Dict[str, CallDetectorPipeline] = {}
metrics_counters = {
    "total_calls": 0,
    "total_inferences": 0,
    "total_latency_ms": 0.0,
    "ai_predictions": 0,
    "human_predictions": 0,
    "unknown_predictions": 0,
}


@router.websocket("/ws/call/{call_id}")
async def call_websocket_endpoint(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time telephone audio streaming.
    Receives raw PCM binary chunks, processes audio in background queue,
    and returns JSON detection events back over WebSocket.
    """
    if len(active_calls) >= settings.MAX_ACTIVE_CALLS:
        await websocket.close(code=1013, reason="Server overloaded: Maximum active calls reached.")
        return

    await websocket.accept()
    logger.info(f"WebSocket connection established for call_id={call_id}")

    pipeline = CallDetectorPipeline(call_id=call_id, source_sample_rate=settings.INPUT_SAMPLE_RATE)
    active_calls[call_id] = pipeline
    metrics_counters["total_calls"] += 1

    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=settings.MAX_AUDIO_QUEUE_SIZE)
    is_active = True

    async def audio_processor_worker():
        """Decoupled background worker processing audio queue and pushing detection events."""
        nonlocal is_active
        while is_active:
            try:
                raw_chunk = await audio_queue.get()
                if raw_chunk is None:
                    break

                # Process chunk using thread pool if needed or direct call
                events = pipeline.process_pcm_chunk(raw_chunk)
                for event in events:
                    # Update global metrics
                    metrics_counters["total_inferences"] += 1
                    metrics_counters["total_latency_ms"] += event.inference_latency_ms
                    if event.status == "ai":
                        metrics_counters["ai_predictions"] += 1
                    elif event.status == "human":
                        metrics_counters["human_predictions"] += 1
                    else:
                        metrics_counters["unknown_predictions"] += 1

                    # Send detection response to client
                    await websocket.send_json(event.model_dump())

                audio_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing audio in call {call_id}: {e}", exc_info=True)

    worker_task = asyncio.create_task(audio_processor_worker())

    try:
        while True:
            # Receive binary raw PCM audio bytes or JSON configuration/audio packet
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if not audio_queue.full():
                    await audio_queue.put(raw_bytes)
                else:
                    logger.warning(f"Audio queue full for call {call_id}, dropping stale frame.")
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                except Exception:
                    pass
    except WebSocketDisconnect:
        logger.info(f"Client disconnected call_id={call_id}")
    except Exception as e:
        logger.error(f"WebSocket error for call_id={call_id}: {e}")
    finally:
        is_active = False
        await audio_queue.put(None)  # Signal worker to stop
        worker_task.cancel()
        pipeline.cleanup()
        active_calls.pop(call_id, None)
        logger.info(f"Cleaned up call_id={call_id}")
