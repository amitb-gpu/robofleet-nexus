"""
Dataset API routes for RoboFleet Nexus LeRobot v3 exporter.

Add to main.py:
    from robofleet_nexus.lerobot.routes import dataset_router
    app.include_router(dataset_router)

Endpoints:
    GET  /dataset/buffer          — buffer stats per robot
    POST /dataset/export          — export completed episodes to disk
    GET  /dataset/exports         — list previous exports
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from robofleet_nexus.lerobot.episode_buffer import episode_buffer
from robofleet_nexus.lerobot.exporter import export_dataset, export_summary

logger = logging.getLogger("robofleet.lerobot.routes")

dataset_router = APIRouter(prefix="/dataset", tags=["dataset"])

# Track all export output dirs this session
_export_history: list[dict] = []

DEFAULT_EXPORT_ROOT = Path.home() / "robofleet_datasets"


class ExportRequest(BaseModel):
    robot_id: str
    task: str = "unspecified"
    fps: float = 10.0
    output_dir: str | None = None
    include_open_episode: bool = False


@dataset_router.get("/buffer")
def buffer_stats() -> dict:
    """Return episode buffer statistics for all robots."""
    robot_ids = episode_buffer.all_robot_ids()
    if not robot_ids:
        return {"robots": [], "message": "No telemetry received yet."}
    return {
        "robots": [episode_buffer.stats(rid) for rid in robot_ids]
    }


@dataset_router.post("/export")
def export_episodes(req: ExportRequest) -> dict:
    """
    Flush completed episodes for a robot and write a LeRobot v3 dataset.

    The export is written to:
      ~/robofleet_datasets/<robot_id>/<episode_count>eps/
    or to req.output_dir if provided.
    """
    episodes = episode_buffer.flush(
        robot_id=req.robot_id,
        task=req.task,
        include_open=req.include_open_episode,
    )

    if not episodes:
        stats = episode_buffer.stats(req.robot_id)
        raise HTTPException(
            status_code=404,
            detail=(
                f"No completed episodes found for robot '{req.robot_id}'. "
                f"Buffer has {stats['joint_state_events']} joint_state events. "
                f"Need at least {5} frames and a {30}s gap to close an episode, "
                f"or set include_open_episode=true."
            ),
        )

    out_dir = Path(req.output_dir) if req.output_dir else (
        DEFAULT_EXPORT_ROOT / req.robot_id /
        f"{len(episodes)}eps_{episodes[0].started_at.strftime('%Y%m%d_%H%M%S')}"
    )

    export_dataset(
        episodes=episodes,
        output_dir=out_dir,
        robot_id=req.robot_id,
        task=req.task,
        fps=req.fps,
    )

    summary = export_summary(out_dir)
    _export_history.append(summary)

    logger.info(
        "Exported %d episodes for %s → %s",
        len(episodes), req.robot_id, out_dir,
    )
    return summary


@dataset_router.get("/exports")
def list_exports() -> dict:
    """Return all dataset exports created this session."""
    return {"exports": _export_history, "count": len(_export_history)}
