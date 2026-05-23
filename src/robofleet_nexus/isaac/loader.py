from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

from robofleet_nexus.isaac.job_spec import IsaacSimulationJob


class SimulationJobLoadError(ValueError):
    """Raised when a simulation job file cannot be loaded or validated."""


def load_simulation_job(path: Path) -> IsaacSimulationJob:
    """
    Load an Isaac simulation job from YAML or JSON.

    The loader is intentionally small and strict. It validates the loaded
    document against the IsaacSimulationJob Pydantic model before returning it.
    """
    if not path.exists():
        raise SimulationJobLoadError(f"Simulation job file does not exist: {path}")

    if not path.is_file():
        raise SimulationJobLoadError(f"Simulation job path is not a file: {path}")

    suffix = path.suffix.lower()

    try:
        raw_text = path.read_text()
    except OSError as exc:
        raise SimulationJobLoadError(f"Could not read simulation job file: {exc}") from exc

    try:
        if suffix in {".yaml", ".yml"}:
            raw_data = yaml.safe_load(raw_text)
        elif suffix == ".json":
            raw_data = json.loads(raw_text)
        else:
            raise SimulationJobLoadError(
                f"Unsupported simulation job file extension: {suffix}. "
                "Expected .yaml, .yml, or .json."
            )
    except yaml.YAMLError as exc:
        raise SimulationJobLoadError(f"Invalid YAML simulation job file: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SimulationJobLoadError(f"Invalid JSON simulation job file: {exc}") from exc

    data = cast(Any, raw_data)

    if not isinstance(data, dict):
        raise SimulationJobLoadError("Simulation job file must contain an object at the top level.")

    try:
        return IsaacSimulationJob.model_validate(data)
    except Exception as exc:
        raise SimulationJobLoadError(f"Simulation job failed schema validation: {exc}") from exc
