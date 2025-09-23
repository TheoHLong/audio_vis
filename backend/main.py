from __future__ import annotations
import json
import logging
from typing import Any, Dict

import numpy as np
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import PipelineConfig
from .pipeline import CometPipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_app() -> FastAPI:
    config = PipelineConfig()
    pipeline = CometPipeline(config=config)

    app = FastAPI(title="Speech-to-Comet Visualization", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse("frontend/index.html")

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {"status": "ok", "mode": pipeline.mode}

    @app.post("/mode")
    async def set_mode(payload: Dict[str, str] = Body(...)) -> Dict[str, str]:
        mode = payload.get("mode")
        try:
            pipeline.set_mode(mode)
        except ValueError as exc:  # pragma: no cover - simple validation
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"mode": pipeline.mode}

    @app.websocket("/ws/audio")
    async def websocket_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {
                "type": "hello",
                "sampleRate": config.sample_rate,
                "frameMs": config.frame_ms,
                "hopMs": config.hop_ms,
                "palette": pipeline.palette,
                "defaultColor": pipeline.default_color,
            }
        )
        try:
            while True:
                message = await websocket.receive()
                if "text" in message and message["text"] is not None:
                    await _handle_control_message(pipeline, websocket, message["text"])
                    continue
                if "bytes" in message and message["bytes"] is not None:
                    chunk = np.frombuffer(message["bytes"], dtype=np.float32)
                else:
                    continue
                payload = pipeline.process_samples(chunk)
                if payload:
                    await websocket.send_text(pipeline.to_json(payload))
        except WebSocketDisconnect:
            logger.info("Client disconnected")
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.exception("Websocket error: %s", exc)
            await websocket.close(code=1011)

    return app


async def _handle_control_message(pipeline: CometPipeline, websocket: WebSocket, message: str) -> None:
    try:
        data = json.loads(message)
    except json.JSONDecodeError:
        return
    msg_type = data.get("type")
    if msg_type == "mode":
        mode = data.get("mode")
        try:
            pipeline.set_mode(mode)
        except ValueError:
            return
        await websocket.send_json({"type": "mode", "mode": pipeline.mode})
    elif msg_type == "reset":
        pipeline.reset()
        await websocket.send_json({"type": "reset"})


app = create_app()
