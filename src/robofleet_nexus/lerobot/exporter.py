"""
LeRobot v3 Parquet exporter for RoboFleet Nexus.

Writes a fully spec-compliant LeRobot v3 dataset from a list of
LeRobotEpisode objects.

Output structure:
    <output_dir>/
        meta/
            info.json
            episodes.jsonl
            tasks.jsonl
        data/
            chunk-000/
                episode_000000.parquet
                episode_000001.parquet
                ...

The dataset can be loaded directly with the HuggingFace LeRobot library:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset.from_preloaded(path)

Or used standalone with pandas:
    import pandas as pd
    df = pd.read_parquet("data/chunk-000/episode_000000.parquet")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from robofleet_nexus.lerobot.schema import (
    LeRobotEpisode,
    build_info_json,
)

logger = logging.getLogger("robofleet.lerobot.exporter")

CHUNKS_SIZE = 1000  # episodes per chunk directory


def export_dataset(
    episodes: list[LeRobotEpisode],
    output_dir: Path | str,
    robot_id: str,
    task: str = "unspecified",
    fps: float = 10.0,
    robot_type: str = "nexus_robot",
) -> Path:
    """
    Export a list of LeRobotEpisode objects to a LeRobot v3 dataset on disk.

    Args:
        episodes:    Episodes to export (from EpisodeBuffer.flush()).
        output_dir:  Root directory for the dataset. Created if absent.
        robot_id:    Robot identifier embedded in info.json.
        task:        Task label for all episodes.
        fps:         Nominal recording frame rate.
        robot_type:  Robot type string for info.json.

    Returns:
        Path to the output directory.

    Raises:
        ValueError: If episodes list is empty.
    """
    if not episodes:
        raise ValueError("No episodes to export.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_dir = output_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    # ── Infer joint names from first episode ──────────────────────────────────
    first_frame = episodes[0].frames[0]
    n_joints = len(first_frame.action)
    joint_names = [f"joint_{i}" for i in range(n_joints)]

    # ── Write per-episode Parquet files ───────────────────────────────────────
    for episode in episodes:
        chunk_idx = episode.episode_index // CHUNKS_SIZE
        chunk_dir = output_dir / "data" / f"chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = chunk_dir / f"episode_{episode.episode_index:06d}.parquet"
        _write_episode_parquet(episode, parquet_path)
        logger.info(
            "Wrote %s (%d frames)",
            parquet_path.relative_to(output_dir),
            episode.num_frames,
        )

    # ── meta/info.json ────────────────────────────────────────────────────────
    info = build_info_json(
        robot_id=robot_id,
        joint_names=joint_names,
        episodes=episodes,
        fps=fps,
        robot_type=robot_type,
    )
    (meta_dir / "info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )

    # ── meta/episodes.jsonl ───────────────────────────────────────────────────
    with open(meta_dir / "episodes.jsonl", "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_meta()) + "\n")

    # ── meta/tasks.jsonl ──────────────────────────────────────────────────────
    (meta_dir / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Dataset exported to %s — %d episodes, %d total frames",
        output_dir,
        len(episodes),
        sum(e.num_frames for e in episodes),
    )
    return output_dir


def _write_episode_parquet(episode: LeRobotEpisode, path: Path) -> None:
    """Serialise a single episode to a Parquet file."""
    rows = [f.to_dict() for f in episode.frames]
    df = pd.DataFrame(rows)

    # Cast to correct dtypes for LeRobot compatibility
    df["timestamp"]     = df["timestamp"].astype("float32")
    df["frame_index"]   = df["frame_index"].astype("int64")
    df["episode_index"] = df["episode_index"].astype("int64")
    df["task_index"]    = df["task_index"].astype("int64")
    df["next.done"]     = df["next.done"].astype("bool")

    for col in ("action", "observation.state",
                "observation.velocity", "observation.effort",
                "observation.pose", "observation.battery"):
        # Store list columns as fixed-size list arrays in Arrow
        if col in df.columns:
            arr = pa.array(df[col].tolist(), type=pa.list_(pa.float32()))
            df = df.drop(columns=[col])
            table = pa.Table.from_pandas(df, preserve_index=False)
            table = table.append_column(col, arr)
            pq.write_table(table, path, compression="snappy")
            return

    # Fallback if no list columns (shouldn't happen)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="snappy")


def export_summary(output_dir: Path | str) -> dict:
    """
    Read back an exported dataset and return a summary dict.
    Useful for verifying the export and building the API response.
    """
    output_dir = Path(output_dir)
    info_path = output_dir / "meta" / "info.json"
    episodes_path = output_dir / "meta" / "episodes.jsonl"

    if not info_path.exists():
        return {"error": f"No dataset found at {output_dir}"}

    info = json.loads(info_path.read_text())
    episodes = []
    if episodes_path.exists():
        for line in episodes_path.read_text().splitlines():
            if line.strip():
                episodes.append(json.loads(line))

    return {
        "output_dir": str(output_dir),
        "robot_type": info.get("robot_type"),
        "robot_id": info.get("robot_id"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "episodes": episodes,
    }
