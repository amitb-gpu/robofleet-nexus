"""
Shared 6-joint arm trajectory used by the in-process demo and the ROS2 fake arm.

Kept free of pandas/pyarrow so it imports under the system Python that runs ROS2.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
HZ = 30.0
AMPLITUDE = np.array([0.6, 0.4, 0.5, 0.3, 0.3, 0.2])
OFFSET = np.array([0.0, -0.8, 1.2, -0.4, 0.0, 0.0])
OMEGA = np.array([0.5, 0.7, 0.6, 0.9, 1.1, 0.8])


def nominal(t: float, phase: NDArray[np.float64] | None = None) -> NDArray[np.float64]:
    """Joint positions of the nominal pick-and-place motion at time t (seconds)."""
    p = phase if phase is not None else np.zeros(len(JOINTS))
    result: NDArray[np.float64] = OFFSET + AMPLITUDE * np.sin(OMEGA * t + p)
    return result
