"""
Observation baseline for out-of-distribution (OOD) detection.

A learned policy is only trustworthy on observations that look like its training
data. The baseline records per-dimension statistics of the observations in a
policy's demonstration dataset; at runtime the guard scores each observation
against it and halts when the robot has drifted somewhere the policy never saw
(a slipped encoder, an unexpected load, a scene the policy wasn't trained on).

Baselines are fitted from LeRobot v3 datasets, including the ones RoboFleet Nexus
exports from fleet telemetry.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy.typing import NDArray
from pydantic import BaseModel, Field

STD_FLOOR = 1e-3


class ObservationBaseline(BaseModel):
    """Per-dimension statistics of a policy's training observations."""

    names: list[str]
    mean: list[float]
    std: list[float]
    minimum: list[float]
    maximum: list[float]
    max_step: list[float] = Field(
        description="Largest frame-to-frame change seen in the data, per dimension."
    )
    num_frames: int
    source: str | None = None

    @property
    def dim(self) -> int:
        return len(self.names)

    # ── Fitting ───────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        episodes: Sequence[NDArray[np.float64] | Sequence[Sequence[float]]],
        names: list[str] | None = None,
        source: str | None = None,
    ) -> ObservationBaseline:
        """Fit from a list of episodes, each an array of shape (frames, dim)."""
        arrays = [np.asarray(ep, dtype=np.float64) for ep in episodes if len(ep) > 0]
        if not arrays:
            raise ValueError("cannot fit a baseline from zero frames")
        dims = {a.shape[1] for a in arrays}
        if len(dims) != 1:
            raise ValueError(f"episodes have inconsistent dimensions: {sorted(dims)}")
        (dim,) = dims

        stacked = np.vstack(arrays)
        steps = [np.abs(np.diff(a, axis=0)) for a in arrays if a.shape[0] > 1]
        max_step = np.max(np.vstack(steps), axis=0) if steps else np.zeros(dim)

        return cls(
            names=names or [f"joint_{i}" for i in range(dim)],
            mean=stacked.mean(axis=0).tolist(),
            std=stacked.std(axis=0).tolist(),
            minimum=stacked.min(axis=0).tolist(),
            maximum=stacked.max(axis=0).tolist(),
            max_step=max_step.tolist(),
            num_frames=int(stacked.shape[0]),
            source=source,
        )

    @classmethod
    def from_lerobot_dataset(
        cls, root: Path | str, column: str = "observation.state"
    ) -> ObservationBaseline:
        """Fit from a LeRobot v3 dataset directory (meta/ + data/chunk-*/*.parquet)."""
        root = Path(root)
        files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
        if not files:
            raise FileNotFoundError(f"no LeRobot episode parquet files under {root / 'data'}")

        episodes: list[NDArray[np.float64]] = []
        for f in files:
            table = pq.read_table(f, columns=[column])
            rows = table.column(column).to_pylist()
            if rows:
                episodes.append(np.asarray(rows, dtype=np.float64))

        names = _names_from_info(root, column)
        return cls.fit(episodes, names=names, source=str(root))

    # ── Scoring ───────────────────────────────────────────────────────────

    def zscores(self, observation: Sequence[float]) -> NDArray[np.float64]:
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (self.dim,):
            raise ValueError(f"observation has {obs.size} values, baseline expects {self.dim}")
        std = np.maximum(np.asarray(self.std), STD_FLOOR)
        result: NDArray[np.float64] = np.abs(obs - np.asarray(self.mean)) / std
        return result

    def score(self, observation: Sequence[float]) -> tuple[float, str]:
        """Return the worst per-dimension z-score and the dimension it came from."""
        z = self.zscores(observation)
        worst = int(np.argmax(z))
        return float(z[worst]), self.names[worst]

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: Path | str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path | str) -> ObservationBaseline:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _names_from_info(root: Path, column: str) -> list[str] | None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        return None
    info = json.loads(info_path.read_text(encoding="utf-8"))
    names = info.get("features", {}).get(column, {}).get("names")
    if isinstance(names, list) and all(isinstance(n, str) for n in names):
        return names
    return None
