"""
LeRobot v3 dataset schema and frame assembly for RoboFleet Nexus.

Specification reference:
  https://github.com/huggingface/lerobot/blob/main/lerobot/common/datasets/README.md

A LeRobot v3 dataset is organised as:
  <root>/
    meta/
      info.json        — dataset-level metadata and feature schema
      episodes.jsonl   — one JSON object per episode
      tasks.jsonl      — one JSON object per task label
    data/
      chunk-000/
        episode_000000.parquet
        episode_000001.parquet
        ...

Each parquet row = one frame (one timestep). Mandatory columns:
  timestamp           float32   seconds from episode start
  frame_index         int64     0-based index within episode
  episode_index       int64     global episode index
  task_index          int64     index into tasks.jsonl
  next.done           bool      True on the last frame of an episode
  action              list[f32] joint position targets (what the robot did)
  observation.state   list[f32] observed joint positions

Optional columns added by this exporter:
  observation.velocity  list[f32]  joint velocities
  observation.effort    list[f32]  joint efforts (torques)
  observation.pose      list[f32]  [x, y, z, vx] from odometry
  observation.battery   list[f32]  [battery_pct]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ── Frame ─────────────────────────────────────────────────────────────────────

@dataclass
class LeRobotFrame:
    """One timestep in a LeRobot v3 episode."""
    timestamp: float              # seconds from episode start
    frame_index: int
    episode_index: int
    task_index: int
    action: list[float]           # joint position targets
    observation_state: list[float]  # observed joint positions
    observation_velocity: list[float]
    observation_effort: list[float]
    observation_pose: list[float]   # [x, y, z, vx]
    observation_battery: list[float]  # [battery_pct]
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "episode_index": self.episode_index,
            "task_index": self.task_index,
            "next.done": self.done,
            "action": self.action,
            "observation.state": self.observation_state,
            "observation.velocity": self.observation_velocity,
            "observation.effort": self.observation_effort,
            "observation.pose": self.observation_pose,
            "observation.battery": self.observation_battery,
        }


# ── Episode ───────────────────────────────────────────────────────────────────

@dataclass
class LeRobotEpisode:
    """A complete episode ready for export."""
    episode_index: int
    robot_id: str
    task: str
    started_at: datetime
    ended_at: datetime
    frames: list[LeRobotFrame] = field(default_factory=list)

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return (self.ended_at - self.started_at).total_seconds()

    def to_meta(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "task": self.task,
            "robot_id": self.robot_id,
            "length": self.num_frames,
            "duration_seconds": round(self.duration_seconds, 3),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
        }


# ── Dataset info ──────────────────────────────────────────────────────────────

def build_info_json(
    robot_id: str,
    joint_names: list[str],
    episodes: list[LeRobotEpisode],
    fps: float = 10.0,
    robot_type: str = "nexus_robot",
) -> dict[str, Any]:
    """Build the meta/info.json dict for a LeRobot v3 dataset."""
    n_joints = len(joint_names)
    total_frames = sum(e.num_frames for e in episodes)
    n_episodes = len(episodes)
    n_chunks = max(1, (n_episodes + 999) // 1000)

    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "robot_id": robot_id,
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_chunks": n_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "action": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": joint_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": joint_names,
            },
            "observation.velocity": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": [f"{j}.vel" for j in joint_names],
            },
            "observation.effort": {
                "dtype": "float32",
                "shape": [n_joints],
                "names": [f"{j}.eff" for j in joint_names],
            },
            "observation.pose": {
                "dtype": "float32",
                "shape": [4],
                "names": ["x", "y", "z", "vx"],
            },
            "observation.battery": {
                "dtype": "float32",
                "shape": [1],
                "names": ["battery_pct"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
