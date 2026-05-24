"""
WebSocket connection manager for RoboFleet Nexus dashboard.

Manages all connected browser clients and broadcasts real-time
telemetry, GPU metrics, diagnostic findings, and audit events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

# Replay buffer — new clients receive recent history immediately
_REPLAY_BUFFER_SIZE = 50


class ConnectionManager:
    """
    Manages all active WebSocket connections and the in-memory event buffer.

    Thread-safe for use with asyncio. Call broadcast() from any coroutine;
    it fans out to all connected clients and silently drops stale connections.
    """

    def __init__(self) -> None:
        self._active: list[WebSocket] = []
        self._replay_buffer: Deque[dict[str, Any]] = deque(maxlen=_REPLAY_BUFFER_SIZE)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._active.append(websocket)

        # Replay recent events so the dashboard is instantly populated
        for event in list(self._replay_buffer):
            try:
                await websocket.send_text(json.dumps(event))
            except Exception:
                break  # client disconnected during replay

        logger.info(
            "Dashboard client connected. Total clients: %d", len(self._active)
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            try:
                self._active.remove(websocket)
            except ValueError:
                pass
        logger.info(
            "Dashboard client disconnected. Total clients: %d", len(self._active)
        )

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    async def broadcast(self, event: dict[str, Any]) -> None:
        """
        Broadcast a typed event dict to all connected clients.
        Stale/disconnected clients are pruned automatically.
        """
        # Stamp and buffer
        event.setdefault("ts", datetime.now(timezone.utc).isoformat())
        self._replay_buffer.append(event)

        message = json.dumps(event)
        dead: list[WebSocket] = []

        async with self._lock:
            clients = list(self._active)

        for ws in clients:
            try:
                await ws.send_text(message)
            except (WebSocketDisconnect, RuntimeError):
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    try:
                        self._active.remove(ws)
                    except ValueError:
                        pass

    # ------------------------------------------------------------------
    # Typed event helpers
    # ------------------------------------------------------------------

    async def emit_telemetry(self, robot_id: str, payload: dict[str, Any]) -> None:
        await self.broadcast(
            {"type": "telemetry", "robot_id": robot_id, "data": payload}
        )

    async def emit_gpu_snapshot(self, gpus: list[dict[str, Any]]) -> None:
        await self.broadcast({"type": "gpu_snapshot", "gpus": gpus})

    async def emit_diagnostic(
        self, robot_id: str, findings: list[dict[str, Any]]
    ) -> None:
        await self.broadcast(
            {"type": "diagnostic", "robot_id": robot_id, "findings": findings}
        )

    async def emit_sim_job(self, job: dict[str, Any]) -> None:
        await self.broadcast({"type": "sim_job", "job": job})

    async def emit_audit(self, entry: dict[str, Any]) -> None:
        await self.broadcast({"type": "audit", "entry": entry})

    async def emit_ros2_event(
        self, robot_id: str, topic: str, payload: dict[str, Any]
    ) -> None:
        await self.broadcast(
            {
                "type": "ros2_event",
                "robot_id": robot_id,
                "topic": topic,
                "data": payload,
            }
        )

    @property
    def client_count(self) -> int:
        return len(self._active)


# Module-level singleton — import this everywhere
manager = ConnectionManager()
