"""
Episode buffer for RoboFleet Nexus LeRobot v3 exporter.

Accumulates RobotEvent objects from the telemetry stream and groups them
into discrete episodes based on time-gap detection. An episode boundary
is declared when consecutive joint_state events for a robot are separated
by more than EPISODE_GAP_SECONDS.

Frame assembly strategy:
  - Primary timeline: joint_state events (one frame per joint_state)
  - For each joint_state frame, the nearest odometry event within ±2s
    is merged in to provide pose data
  - For each joint_state frame, the nearest battery_state event within ±10s
    is merged in for battery level
  - action = joint positions at this frame (what the robot executed)
  - observation.state = same joint positions (current observed state)

This matches the LeRobot convention where for teleoperation data the
action is the joint target the operator commanded, which equals the
observed position at that timestep (no lag in the dataset recording).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from robofleet_nexus.lerobot.schema import LeRobotEpisode, LeRobotFrame
from robofleet_nexus.telemetry.schemas import RobotEvent

logger = logging.getLogger("robofleet.lerobot")

# ── Configuration ──────────────────────────────────────────────────────────────
EPISODE_GAP_SECONDS = 30.0   # gap between joint_state events → new episode
MIN_EPISODE_FRAMES  = 5      # discard episodes shorter than this
DEFAULT_JOINT_NAMES = [f"joint_{i}" for i in range(6)]
DEFAULT_TASK        = "unspecified"


# ── EpisodeBuffer ──────────────────────────────────────────────────────────────

class EpisodeBuffer:
    """
    Maintains a per-robot sliding window of RobotEvents and extracts
    complete LeRobotEpisodes on demand.

    Usage:
        buf = EpisodeBuffer()
        buf.ingest(event)          # called from telemetry ingest handler
        episodes = buf.flush("bot_001")   # extract completed episodes
    """

    def __init__(self) -> None:
        # raw events per robot, oldest first
        self._events: dict[str, list[RobotEvent]] = defaultdict(list)
        # episode counter per robot
        self._episode_counters: dict[str, int] = defaultdict(int)

    def ingest(self, event: RobotEvent) -> None:
        """Append a telemetry event to the buffer."""
        if event.event_type in ("joint_state", "odometry", "battery_state"):
            self._events[event.robot_id].append(event)

    def flush(
        self,
        robot_id: str,
        task: str = DEFAULT_TASK,
        include_open: bool = False,
    ) -> list[LeRobotEpisode]:
        """
        Extract completed episodes for a robot from the buffer.

        A completed episode is a contiguous run of joint_state events
        where each consecutive pair is ≤ EPISODE_GAP_SECONDS apart,
        terminated by a gap > EPISODE_GAP_SECONDS or end of buffer.

        Args:
            robot_id:     Robot to extract episodes for.
            task:         Task label to embed in the episode metadata.
            include_open: If True, also extract the currently-open (last)
                          episode even if no trailing gap has been detected.
                          Useful for manual export during a live session.

        Returns:
            List of LeRobotEpisode objects ready for export.
        """
        all_events = self._events.get(robot_id, [])
        if not all_events:
            return []

        joint_events = [e for e in all_events if e.event_type == "joint_state"]
        odom_events  = [e for e in all_events if e.event_type == "odometry"]
        batt_events  = [e for e in all_events if e.event_type == "battery_state"]

        if not joint_events:
            return []

        # ── Split joint events into runs separated by EPISODE_GAP_SECONDS ──
        runs: list[list[RobotEvent]] = []
        current_run: list[RobotEvent] = [joint_events[0]]

        for prev, curr in zip(joint_events, joint_events[1:]):
            gap = (curr.timestamp - prev.timestamp).total_seconds()
            if gap > EPISODE_GAP_SECONDS:
                runs.append(current_run)
                current_run = [curr]
            else:
                current_run.append(curr)

        # The last run is open (no trailing gap confirmed)
        if include_open:
            runs.append(current_run)
        else:
            # Only closed runs (all but the last)
            if len(runs) == 0:
                return []  # only one run and it's still open

        episodes: list[LeRobotEpisode] = []
        consumed_events: list[RobotEvent] = []

        for run in runs:
            if len(run) < MIN_EPISODE_FRAMES:
                consumed_events.extend(run)
                continue

            ep_idx = self._episode_counters[robot_id]
            self._episode_counters[robot_id] += 1

            t0 = run[0].timestamp
            frames: list[LeRobotFrame] = []

            for i, je in enumerate(run):
                t_sec = (je.timestamp - t0).total_seconds()
                joint_names = _extract_joint_names(je)
                positions   = _extract_joint_metric(je, joint_names, ".pos")
                velocities  = _extract_joint_metric(je, joint_names, ".vel")
                efforts     = _extract_joint_metric(je, joint_names, ".eff")

                # nearest odometry snapshot
                nearest_odom = _nearest_event(odom_events, je.timestamp, window=2.0)
                pose = _extract_pose(nearest_odom)

                # nearest battery snapshot
                nearest_batt = _nearest_event(batt_events, je.timestamp, window=10.0)
                battery = _extract_battery(nearest_batt)

                frames.append(LeRobotFrame(
                    timestamp=round(t_sec, 4),
                    frame_index=i,
                    episode_index=ep_idx,
                    task_index=0,
                    action=positions,          # joint positions as action targets
                    observation_state=positions,
                    observation_velocity=velocities,
                    observation_effort=efforts,
                    observation_pose=pose,
                    observation_battery=battery,
                    done=(i == len(run) - 1),
                ))

            if frames:
                frames[-1].done = True
                episode = LeRobotEpisode(
                    episode_index=ep_idx,
                    robot_id=robot_id,
                    task=task,
                    started_at=run[0].timestamp,
                    ended_at=run[-1].timestamp,
                    frames=frames,
                )
                episodes.append(episode)
                logger.info(
                    "Episode %d assembled: robot=%s frames=%d duration=%.1fs",
                    ep_idx, robot_id, len(frames),
                    episode.duration_seconds,
                )

            consumed_events.extend(run)

        # Remove consumed events from the buffer
        remaining = [e for e in all_events if e not in consumed_events]
        self._events[robot_id] = remaining

        return episodes

    def stats(self, robot_id: str) -> dict[str, Any]:
        """Return buffer statistics for a robot."""
        events = self._events.get(robot_id, [])
        joint_count = sum(1 for e in events if e.event_type == "joint_state")
        return {
            "robot_id": robot_id,
            "buffered_events": len(events),
            "joint_state_events": joint_count,
            "completed_episodes": self._episode_counters[robot_id],
            "estimated_frames_in_buffer": joint_count,
        }

    def all_robot_ids(self) -> list[str]:
        return list(self._events.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_joint_names(event: RobotEvent) -> list[str]:
    """Infer joint names from metrics keys (e.g. joint_0.pos → joint_0)."""
    names = sorted({
        k.rsplit(".", 1)[0]
        for k in event.metrics
        if k.endswith(".pos")
    })
    return names if names else DEFAULT_JOINT_NAMES


def _extract_joint_metric(
    event: RobotEvent,
    joint_names: list[str],
    suffix: str,
) -> list[float]:
    return [float(event.metrics.get(f"{j}{suffix}", 0.0)) for j in joint_names]


def _nearest_event(
    events: list[RobotEvent],
    target: datetime,
    window: float,
) -> RobotEvent | None:
    """Find the event closest in time to target within ±window seconds."""
    best: RobotEvent | None = None
    best_dt = float("inf")
    for e in events:
        dt = abs((e.timestamp - target).total_seconds())
        if dt < best_dt and dt <= window:
            best_dt = dt
            best = e
    return best


def _extract_pose(event: RobotEvent | None) -> list[float]:
    if event is None:
        return [0.0, 0.0, 0.0, 0.0]
    m = event.metrics
    return [
        float(m.get("x", 0.0)),
        float(m.get("y", 0.0)),
        float(m.get("z", 0.0)),
        float(m.get("vx", 0.0)),
    ]


def _extract_battery(event: RobotEvent | None) -> list[float]:
    if event is None:
        return [100.0]
    return [float(event.metrics.get("battery_pct", 100.0))]


# ── Module-level singleton ─────────────────────────────────────────────────────
episode_buffer = EpisodeBuffer()
