"""FastAPI server for SignBridge AI — real-time video stream, WebSocket telemetry, and voice/ML APIs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .hand_tracker import VisionPipeline
from .ml_model import SignLanguageAIModel
from .signs import SIGNS
from .voice_engine import VoiceEngine

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

ai_model: Optional[SignLanguageAIModel] = None
voice_engine: Optional[VoiceEngine] = None
vision_pipeline: Optional[VisionPipeline] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ai_model, voice_engine, vision_pipeline
    ai_model = SignLanguageAIModel()
    voice_engine = VoiceEngine()
    vision_pipeline = VisionPipeline(ai_model=ai_model)
    yield
    if vision_pipeline is not None:
        vision_pipeline.shutdown()
    if voice_engine is not None:
        voice_engine.shutdown()


app = FastAPI(title="SignBridge AI", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class SpeakRequest(BaseModel):
    text: str
    rate: float = 1.0
    force: bool = False


class SettingsRequest(BaseModel):
    show_skeleton: Optional[bool] = None
    mirror: Optional[bool] = None
    paused: Optional[bool] = None


class CaptureRequest(BaseModel):
    label: str


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/signs")
async def get_signs():
    return {"signs": SIGNS}


@app.get("/api/status")
async def get_status():
    assert ai_model is not None and voice_engine is not None and vision_pipeline is not None
    return {
        "ok": True,
        "model_accuracy": round(ai_model.model_accuracy * 100, 1),
        "custom_counts": ai_model.custom_counts,
        "voice_backend": voice_engine.backend,
        "detector_mode": vision_pipeline.detector.mode,
    }


@app.post("/api/speak")
async def speak(req: SpeakRequest):
    assert voice_engine is not None
    queued = voice_engine.speak(req.text, rate=req.rate, force=req.force)
    return {"ok": queued, "backend": voice_engine.backend}


@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    assert vision_pipeline is not None
    if req.show_skeleton is not None:
        vision_pipeline.show_skeleton = req.show_skeleton
    if req.mirror is not None:
        vision_pipeline.mirror = req.mirror
    if req.paused is not None:
        vision_pipeline.paused = req.paused
    return {
        "ok": True,
        "show_skeleton": vision_pipeline.show_skeleton,
        "mirror": vision_pipeline.mirror,
        "paused": vision_pipeline.paused,
    }


@app.post("/api/camera/reconnect")
async def reconnect_camera():
    assert vision_pipeline is not None
    vision_pipeline.request_reconnect()
    return {"ok": True}


@app.post("/api/train/capture")
async def capture_training_sample(req: CaptureRequest):
    assert vision_pipeline is not None and ai_model is not None
    lm = vision_pipeline.get_primary_landmarks()
    if lm is None:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": "No hand detected in camera frame. Hold your hand clearly in view."},
        )
    result = ai_model.add_custom_sample(req.label, lm)
    return {"ok": True, **result}


@app.post("/api/train/retrain")
async def retrain_model():
    assert ai_model is not None
    metrics = await asyncio.to_thread(ai_model.train_and_save)
    return {"ok": True, **metrics}


@app.post("/api/train/reset")
async def reset_custom_samples():
    assert ai_model is not None
    await asyncio.to_thread(ai_model.clear_custom_samples)
    return {"ok": True}


@app.get("/video_feed")
async def video_feed():
    assert vision_pipeline is not None

    async def frame_generator():
        while True:
            jpeg = vision_pipeline.get_latest_jpeg()
            if jpeg:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.033)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def telemetry_ws(websocket: WebSocket):
    assert vision_pipeline is not None
    await websocket.accept()
    try:
        while True:
            telemetry = vision_pipeline.get_latest_telemetry()
            await websocket.send_json(telemetry)
            await asyncio.sleep(0.08)
    except WebSocketDisconnect:
        return
    except Exception:
        return
