"""VERS v5.0 Production FastAPI Backend.

Replaces the legacy Flask mock server with a fully production-ready API
providing REST endpoints for alert ingestion, health monitoring, and
system statistics.

Launch:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import cv2
import base64
from src.vers_dashboard import get_runtime

from src.api.schemas import (
    AlertPayload,
    AlertReceiptResponse,
    HealthResponse,
    RecentAlertsResponse,
    StatsResponse,
)
from src.infrastructure.logging_config import setup_logging
from src.infrastructure.metrics import get_metrics
from src.infrastructure.security import APIKeyMiddleware, get_or_create_api_key

# ---------------------------------------------------------------------------
# Initialise logging
# ---------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger("vers.api")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Lifespan (modern replacement for deprecated on_event)
# ---------------------------------------------------------------------------
runtime = get_runtime()

async def broadcast_camera_feed():
    """Polls DashboardRuntime and broadcasts over WebSocket."""
    runtime.start(conf_threshold=0.6, distress_threshold=0.6)
    try:
        while True:
            await asyncio.sleep(0.05)  # ~20 FPS
            if not manager.active_connections:
                continue

            snapshot = runtime.snapshot()
            if snapshot["frame"] is not None:
                bgr_frame = cv2.cvtColor(snapshot["frame"], cv2.COLOR_RGB2BGR)
                ret, buffer = cv2.imencode('.jpg', bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    b64_img = base64.b64encode(buffer).decode('utf-8')

                    payload = {
                        "telemetry": {
                            "fps": snapshot.get("fps", 0),
                            "gesture": snapshot.get("gesture", "NONE"),
                            "confidence": snapshot.get("confidence", 0),
                            "distress_score": snapshot.get("distress_score", 0),
                            "dominant_emotion": snapshot.get("dominant_emotion", "neutral"),
                            "threat_level": snapshot.get("threat_level", "NONE"),
                            "sign_word": snapshot.get("sign_word", ""),
                            "sign_buffer_words": snapshot.get("sign_buffer_words", []),
                        },
                        "alerts": snapshot.get("alerts", []),
                        "image": b64_img
                    }
                    await manager.broadcast_json(payload)
    except Exception as e:
        logger.error(f"Camera broadcast task failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifecycle manager — replaces deprecated on_event hooks."""
    api_key = get_or_create_api_key()
    logger.info("VERS API v5.0 started. Docs at http://localhost:8000/docs")
    logger.info("API Key: %s", api_key)
    asyncio.create_task(broadcast_camera_feed())
    yield
    runtime.stop()


app = FastAPI(
    title="VERS Emergency Response API",
    description=(
        "Production REST API for the Vision-Based Emergency Response System. "
        "Receives multimodal alert payloads, tracks system health, and exposes "
        "real-time operational statistics."
    ),
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- CORS (allow React dashboard to call the API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Key security middleware ---
app.add_middleware(APIKeyMiddleware)

# ---------------------------------------------------------------------------
# In-memory alert store (production would use Redis / PostgreSQL)
# ---------------------------------------------------------------------------
_alert_store: deque[dict[str, Any]] = deque(maxlen=200)
_start_time = time.time()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast_json(self, message: dict):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Public health check — exempt from API key requirement."""
    metrics = get_metrics()
    return HealthResponse(
        status="healthy",
        version="VERS-5.0-Production",
        uptime_seconds=round(time.time() - _start_time, 1),
        camera_available=True,
        models_loaded=True,
        tts_active=True,
    )


@app.post("/api/v1/alerts", response_model=AlertReceiptResponse, tags=["Alerts"])
async def receive_alert(payload: AlertPayload) -> AlertReceiptResponse:
    """Receive and store an alert payload from the vision pipeline."""
    alert_dict = payload.model_dump()
    _alert_store.appendleft(alert_dict)

    # Record in metrics
    metrics = get_metrics()
    metrics.record_alert(payload.ThreatLevel.value)

    logger.info(
        "Alert received: %s [%s] severity=%.3f",
        payload.MainGesture,
        payload.ThreatLevel.value,
        payload.SeverityScore,
    )

    return AlertReceiptResponse(
        status="received",
        alert_id=alert_dict.get("Timestamp", "unknown"),
        message=f"Alert for {payload.MainGesture} processed.",
    )


# Backward compatibility with legacy Flask endpoint
@app.post("/alert", tags=["Legacy"])
async def legacy_alert(payload: dict[str, Any]) -> dict[str, str]:
    """Legacy endpoint for backward compatibility with existing pipeline."""
    _alert_store.appendleft(payload)
    return {"status": "received", "message": "Alert processed by VERS API v3.0"}


@app.get("/api/v1/alerts/recent", response_model=RecentAlertsResponse, tags=["Alerts"])
async def recent_alerts(limit: int = 20) -> RecentAlertsResponse:
    """Retrieve the most recent alerts."""
    alerts = list(_alert_store)[:limit]
    return RecentAlertsResponse(count=len(alerts), alerts=alerts)


@app.get("/api/v1/stats", response_model=StatsResponse, tags=["System"])
async def system_stats() -> StatsResponse:
    """Real-time system statistics and operational metrics."""
    metrics = get_metrics()
    snapshot = metrics.snapshot()
    return StatsResponse(**snapshot)

@app.post("/api/v1/trigger", tags=["Alerts"])
async def manual_trigger(gesture: str = "HELP", threat_level: str = "Critical") -> dict[str, Any]:
    """Manually dispatch a test alert for demonstration."""
    from src.services.alert_dispatcher import dispatch as dispatch_alert
    payload = dispatch_alert(
        gesture_label=gesture,
        gesture_confidence=0.95,
        dominant_emotion="fear",
        emotion_distress=0.85,
        severity_score=0.92,
        threat_level=threat_level,
        distress_flag=True,
        enable_tts=True,
        force=True,
    )
    if payload:
        runtime.add_alert(payload)
        _alert_store.appendleft(payload)
        metrics = get_metrics()
        metrics.record_alert(threat_level)
    return {"status": "dispatched", "payload": payload}


@app.websocket("/api/v1/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for live telemetry and camera frames."""
    import json
    import numpy as np

    await manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            if text:
                try:
                    data = json.loads(text)
                    if "image" in data and data["image"]:
                        # Direct Browser Camera frame processing
                        raw_b64 = data["image"]
                        if "," in raw_b64:
                            raw_b64 = raw_b64.split(",", 1)[1]
                        img_bytes = base64.b64decode(raw_b64)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            sign_mode = bool(data.get("sign_mode", False))
                            overlay_rgb, snapshot = runtime.process_frame(frame, use_sign_mode=sign_mode, is_flipped=True)
                            bgr_overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                            ret, buffer = cv2.imencode('.jpg', bgr_overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                b64_img = base64.b64encode(buffer).decode('utf-8')
                                payload = {
                                    "telemetry": {
                                        "fps": snapshot.get("fps", 0),
                                        "gesture": snapshot.get("gesture", "NONE"),
                                        "confidence": snapshot.get("confidence", 0),
                                        "distress_score": snapshot.get("distress_score", 0),
                                        "dominant_emotion": snapshot.get("dominant_emotion", "neutral"),
                                        "threat_level": snapshot.get("threat_level", "NONE"),
                                        "sign_word": snapshot.get("sign_word", ""),
                                        "sign_buffer_words": snapshot.get("sign_buffer_words", []),
                                    },
                                    "alerts": snapshot.get("alerts", []),
                                    "image": b64_img,
                                }
                                await websocket.send_json(payload)
                except Exception as exc:
                    logger.debug("Error processing websocket payload: %s", exc)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

