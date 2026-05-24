"""
WebSocket routes and dashboard endpoint for RoboFleet Nexus.

Drop these lines into your existing src/robofleet_nexus/api/main.py:

    from robofleet_nexus.api.ws_routes import ws_router, lifespan
    # Replace your existing app = FastAPI(...) with:
    app = FastAPI(title="RoboFleet Nexus", lifespan=lifespan)
    app.include_router(ws_router)

Then the dashboard is live at:  http://localhost:8000/dashboard
WebSocket endpoint:              ws://localhost:8000/ws
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from robofleet_nexus.api.ws_manager import manager
from robofleet_nexus.core.gpu_monitor import gpu_poll_loop

logger = logging.getLogger(__name__)

ws_router = APIRouter(tags=["dashboard"])

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan — start background tasks when the server starts
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app) -> AsyncGenerator[None, None]:  # type: ignore[type-arg]
    """
    FastAPI lifespan context manager.
    Starts GPU polling and any other background loops.
    """
    logger.info("RoboFleet Nexus starting up — launching background tasks")
    tasks = [
        asyncio.create_task(
            gpu_poll_loop(manager.emit_gpu_snapshot, interval_seconds=3.0),
            name="gpu-poll",
        ),
    ]
    yield
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("RoboFleet Nexus shut down cleanly")


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Main dashboard WebSocket.  Clients connect here to receive all events.
    The server never expects messages from the client (read-only dashboard),
    but keeps the connection alive with ping/pong.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep-alive: wait for a message (client sends nothing,
            # so this just blocks until the connection closes)
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
        await manager.disconnect(websocket)


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard HTML endpoint
# ──────────────────────────────────────────────────────────────────────────────

@ws_router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    """Serve the live fleet dashboard."""
    html_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    # Fallback: inline minimal dashboard if file is missing
    return HTMLResponse(
        "<h1>RoboFleet Nexus</h1>"
        "<p>Dashboard HTML not found. Place it at "
        "<code>src/robofleet_nexus/dashboard/index.html</code></p>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience hook — call from telemetry ingest handler
# ──────────────────────────────────────────────────────────────────────────────

async def on_telemetry_ingested(robot_id: str, payload: dict) -> None:
    """
    Call this from your existing telemetry POST handler to forward
    events to all connected dashboard clients in real time.

    Example in your existing ingest route:
        from robofleet_nexus.api.ws_routes import on_telemetry_ingested
        ...
        await on_telemetry_ingested(telemetry.robot_id, telemetry.dict())
    """
    await manager.emit_telemetry(robot_id, payload)


async def on_diagnostic_findings(robot_id: str, findings: list[dict]) -> None:
    await manager.emit_diagnostic(robot_id, findings)


async def on_audit_event(entry: dict) -> None:
    await manager.emit_audit(entry)


async def on_sim_job_update(job: dict) -> None:
    await manager.emit_sim_job(job)
