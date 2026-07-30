# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Go2 ROS2 Proxy — controls Go2 via ROS2 topics instead of direct MuJoCo.

Used when the MuJoCo simulation is managed by an external process
(e.g., launch_explore.sh) and we need to send commands via ROS2.
"""
from __future__ import annotations

import functools
import inspect
import json
import math
import os
import threading
import time
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from vector_os_nano.navigation.frames import (
    body_to_sensor_position,
    sensor_to_body_position,
)
from vector_os_nano.navigation.runtime_files import (
    nav_active_file,
    nav_stalled_file,
)

logger = logging.getLogger(__name__)

# Goal-scoped navigation telemetry.  These are deliberately separate from the
# CMU navigation topics: PointStamped has no request identifier and changing its
# frame_id would break the planner's "map" frame contract.
NAV_GOAL_CONTROL_TOPIC: str = "/vector_os/nav_goal_control"
NAV_GOAL_TELEMETRY_TOPIC: str = "/vector_os/nav_goal_telemetry"
NAV_GOAL_TELEMETRY_VERSION: int = 1
NAV_SEGMENT_CONTROL_TOPIC: str = "/vector_os/nav_segment_control"
NAV_SEGMENT_ACK_TOPIC: str = "/vector_os/nav_segment_ack"
NAV_SEGMENT_CONTROL_VERSION: int = 1
NAV_SEGMENT_ACK_TIMEOUT_S: float = 0.60
NAV_SEGMENT_ACK_RETRY_S: float = 0.15
FAR_GOAL_MATCH_TOLERANCE_MAX_M: float = 1.0
FAR_GLOBAL_PATH_TOPIC: str = "/far/global_path"
FAR_ROUTE_MARKER_TOPIC: str = "/viz_path_topic"
FAR_VGRAPH_MARKER_TOPIC: str = "/viz_graph_topic"
FAR_VGRAPH_GLOBAL_VERTEX_NS: str = "global_vertex"
LOCAL_PLANNER_PATH_TOPIC: str = "/local_planner/path"
EXECUTED_PATH_TOPIC: str = "/nav/executed_path"
DOOR_PATH_TOPIC: str = "/scene_graph/door_path"
CURRENT_GOAL_TOPIC: str = "/nav/current_goal"
# Keep the result-envelope ``actor_caused`` alias aligned with the native actor
# grader: a confirmed command alone is not enough; planar displacement must also
# exceed ordinary gait/odometry jitter.
NAV_GOAL_DISPLACEMENT_EPS_M: float = 0.02


@dataclass
class _NavigationProgressWatchdog:
    """Detect a genuine stall without rejecting valid local-avoidance motion.

    The former loop required a 0.1 m improvement on every 0.5 s sample. Slow,
    steady motion therefore accumulated "stall" time even while odometry moved.
    A pure distance-only replacement still rejected large-angle turns and can
    reject a local-planner detour that initially moves away from the final goal.
    Count actual body translation, actual body rotation, or a reduction in goal
    heading error as progress too.  The segment deadline remains the hard upper
    bound if the robot moves but never completes the route.
    """

    progress_threshold_m: float
    timeout_s: float
    heading_progress_threshold_rad: float = math.radians(3.0)
    # GUI rendering can starve the Python/ROS executor for several wall-clock
    # seconds between pose observations.  Count at most this much no-progress
    # time per observation so a sparse-but-live control loop is not mistaken
    # for a stationary robot.  ``None`` preserves wall-clock behaviour for
    # callers that sample at a known, stable rate.
    max_observation_gap_s: float | None = None
    anchor_distance_m: float | None = None
    anchor_time_s: float | None = None
    anchor_heading_error_rad: float | None = None
    anchor_position_xy: tuple[float, float] | None = None
    anchor_heading_rad: float | None = None
    last_observation_time_s: float | None = None
    no_progress_observed_s: float = 0.0

    def _record_progress(self, now_s: float) -> None:
        self.anchor_time_s = now_s
        self.no_progress_observed_s = 0.0

    def stalled(
        self,
        distance_m: float,
        now_s: float,
        *,
        heading_error_rad: float | None = None,
        position_xy: Any = None,
        heading_rad: float | None = None,
    ) -> bool:
        distance = float(distance_m)
        now = float(now_s)
        position = _xy(position_xy)
        heading = (
            float(heading_rad)
            if heading_rad is not None and math.isfinite(float(heading_rad))
            else None
        )
        heading_error = (
            abs(float(heading_error_rad))
            if heading_error_rad is not None
            and math.isfinite(float(heading_error_rad))
            else None
        )
        if self.anchor_distance_m is None or self.anchor_time_s is None:
            self.anchor_distance_m = distance
            self.anchor_time_s = now
            self.last_observation_time_s = now
            self.no_progress_observed_s = 0.0
            self.anchor_heading_error_rad = heading_error
            self.anchor_position_xy = position
            self.anchor_heading_rad = heading
            return False
        previous_observation = self.last_observation_time_s
        observation_delta = (
            max(0.0, now - previous_observation)
            if previous_observation is not None
            else 0.0
        )
        self.last_observation_time_s = now
        if self.max_observation_gap_s is not None:
            observation_delta = min(
                observation_delta,
                max(0.0, float(self.max_observation_gap_s)),
            )
        self.no_progress_observed_s += observation_delta
        if self.anchor_distance_m - distance >= self.progress_threshold_m:
            self.anchor_distance_m = distance
            self._record_progress(now)
            self.anchor_heading_error_rad = heading_error
            self.anchor_position_xy = position
            self.anchor_heading_rad = heading
            return False
        if position is not None:
            if self.anchor_position_xy is None:
                self.anchor_position_xy = position
            elif (
                math.hypot(
                    position[0] - self.anchor_position_xy[0],
                    position[1] - self.anchor_position_xy[1],
                )
                >= self.progress_threshold_m
            ):
                self.anchor_position_xy = position
                self._record_progress(now)
                return False
        if heading is not None:
            if self.anchor_heading_rad is None:
                self.anchor_heading_rad = heading
            elif (
                abs(
                    math.atan2(
                        math.sin(heading - self.anchor_heading_rad),
                        math.cos(heading - self.anchor_heading_rad),
                    )
                )
                >= self.heading_progress_threshold_rad
            ):
                self.anchor_heading_rad = heading
                self._record_progress(now)
                return False
        if heading_error is not None:
            if self.anchor_heading_error_rad is None:
                self.anchor_heading_error_rad = heading_error
            elif (
                self.anchor_heading_error_rad - heading_error
                >= self.heading_progress_threshold_rad
            ):
                # Do not move the linear anchor: once translation begins, all
                # distance accumulated since the last real linear milestone
                # still counts.  Only the no-progress clock and angular anchor
                # need to advance during alignment.
                self.anchor_heading_error_rad = heading_error
                self._record_progress(now)
                return False
        return self.no_progress_observed_s >= self.timeout_s


def _normalise_goal_id(goal_id: Any) -> str:
    """Return a bounded, non-empty navigation goal identifier."""
    if goal_id is None:
        raise ValueError("goal_id is required")
    value = str(goal_id).strip()
    if not value or len(value) > 128:
        raise ValueError("goal_id must contain 1..128 characters")
    return value


def _normalise_segment_id(segment_id: Any) -> str:
    """Return a bounded, non-empty segment generation identifier."""

    if segment_id is None:
        raise ValueError("segment_id is required")
    value = str(segment_id).strip()
    if not value or len(value) > 128:
        raise ValueError("segment_id must contain 1..128 characters")
    return value


def _xy(position: Any) -> tuple[float, float] | None:
    """Best-effort finite planar position coercion used by telemetry only."""
    if position is None:
        return None
    try:
        x, y = float(position[0]), float(position[1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return (x, y)


@dataclass(frozen=True)
class NavigationSegmentConstraints:
    """Goal-scoped motion limits for one structured navigation waypoint.

    This value object is ROS-free on purpose: the proxy serialises it onto the
    control topic and the bridge applies the exact same validation and velocity
    limiting at its final motor boundary.
    """

    goal_id: str | None = None
    segment_id: str | None = None
    kind: str = "room_goal"
    speed_limit_mps: float | None = None
    allow_reverse: bool = True
    tolerance: float | None = None

    _VALID_KINDS = frozenset(
        {"door_pre", "door_center", "door_post", "room_goal"}
    )

    def __post_init__(self) -> None:
        if self.goal_id is not None:
            object.__setattr__(self, "goal_id", _normalise_goal_id(self.goal_id))
        if self.segment_id is not None:
            object.__setattr__(
                self,
                "segment_id",
                _normalise_segment_id(self.segment_id),
            )
        kind = str(self.kind).strip()
        if kind not in self._VALID_KINDS:
            raise ValueError(f"unsupported navigation waypoint kind: {kind!r}")
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.allow_reverse, bool):
            raise ValueError("allow_reverse must be a bool")
        if self.speed_limit_mps is not None:
            speed = float(self.speed_limit_mps)
            if not math.isfinite(speed) or speed < 0.0:
                raise ValueError("speed_limit_mps must be finite and non-negative")
            object.__setattr__(self, "speed_limit_mps", speed)
        if self.tolerance is not None:
            tolerance = float(self.tolerance)
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError("tolerance must be finite and positive")
            object.__setattr__(self, "tolerance", tolerance)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NavigationSegmentConstraints":
        """Parse a validated ``set`` control payload."""
        if int(payload.get("version", -1)) != NAV_SEGMENT_CONTROL_VERSION:
            raise ValueError("unsupported navigation segment control version")
        if str(payload.get("event", "")) != "set":
            raise ValueError("navigation segment payload is not a set event")
        if payload.get("goal_id") is None:
            raise ValueError("navigation segment payload requires goal_id")
        if payload.get("segment_id") is None:
            raise ValueError("navigation segment payload requires segment_id")
        return cls(
            goal_id=payload.get("goal_id"),
            segment_id=payload.get("segment_id"),
            kind=payload.get("kind", "room_goal"),
            speed_limit_mps=payload.get("speed_limit_mps"),
            allow_reverse=payload.get("allow_reverse", True),
            tolerance=payload.get("tolerance"),
        )

    def to_payload(self, *, event: str = "set") -> dict[str, Any]:
        """Return the canonical JSON-compatible topic payload."""
        if event not in {"set", "clear"}:
            raise ValueError(f"unsupported segment control event: {event!r}")
        payload: dict[str, Any] = {
            "version": NAV_SEGMENT_CONTROL_VERSION,
            "event": event,
            "goal_id": self.goal_id,
            "segment_id": self.segment_id,
        }
        if event == "set":
            payload.update(
                {
                    "kind": self.kind,
                    "speed_limit_mps": self.speed_limit_mps,
                    "allow_reverse": self.allow_reverse,
                    "tolerance": self.tolerance,
                }
            )
        return payload

    def constrain_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
    ) -> tuple[float, float, float]:
        """Clamp planar velocity while leaving the yaw controller intact."""
        bounded_vx = float(vx)
        bounded_vy = float(vy)
        bounded_vyaw = float(vyaw)
        if not self.allow_reverse and bounded_vx < 0.0:
            bounded_vx = 0.0
        limit = self.speed_limit_mps
        planar_speed = math.hypot(bounded_vx, bounded_vy)
        if limit is not None and planar_speed > limit:
            if limit == 0.0:
                bounded_vx = 0.0
                bounded_vy = 0.0
            else:
                scale = limit / planar_speed
                bounded_vx *= scale
                bounded_vy *= scale
        return (bounded_vx, bounded_vy, bounded_vyaw)


@dataclass
class _ActiveGoalMotion:
    """Mutable bridge-side accumulator for one accepted goal."""

    goal_id: str
    started_at: float
    target_xy: tuple[float, float] | None
    seq: int = 0
    nonzero_cmd_count: int = 0
    cmd_motion_count: float = 0.0
    nonzero_cmd_duration_s: float = 0.0
    moved_distance_m: float = 0.0
    last_sample_at: float | None = None
    last_position: tuple[float, float] | None = None
    previous_command_nonzero: bool = False


class GoalMotionTracker:
    """Pure-Python bridge accumulator for goal-scoped motor evidence.

    The tracker is intentionally ROS-free so the causation contract can be
    tested without a ROS installation.  A command is counted only when the
    bridge supplies a positive ``executed_motion`` value measured from the
    simulator's own cumulative command counter before/after ``set_velocity``.

    Duration and travelled distance are charged to the interval *following* a
    confirmed non-zero command.  Consequently idle drift before the first
    command, or all drift while no goal is active, cannot become actor evidence.
    """

    _MOTION_EPS: float = 1e-9
    _MAX_SAMPLE_GAP_S: float = 0.5

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._active: _ActiveGoalMotion | None = None

    @property
    def active_goal_id(self) -> str | None:
        return self._active.goal_id if self._active is not None else None

    def begin(
        self,
        goal_id: str,
        *,
        target_xy: Any = None,
        position: Any = None,
    ) -> dict[str, Any] | None:
        """Accept a new goal; repeat announcements for the same goal are no-ops."""
        goal_id = _normalise_goal_id(goal_id)
        if self._active is not None and self._active.goal_id == goal_id:
            return None
        if self._active is not None:
            raise RuntimeError(
                f"goal {self._active.goal_id!r} must be finalized before {goal_id!r}"
            )
        now = float(self._clock())
        self._active = _ActiveGoalMotion(
            goal_id=goal_id,
            started_at=now,
            target_xy=_xy(target_xy),
            last_sample_at=now,
            last_position=_xy(position),
        )
        return self._snapshot("accepted", "active", now=now)

    def record_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        *,
        executed_motion: float,
        position: Any = None,
        source: str = "bridge",
    ) -> dict[str, Any] | None:
        """Record one actual bridge ``set_velocity`` result.

        ``executed_motion`` must be the positive delta of the plant-side
        cumulative command-magnitude counter.  A requested non-zero velocity
        whose delta is zero was gated/rejected and is therefore not evidence.
        Zero commands still close the previous non-zero time/distance interval.
        """
        state = self._active
        if state is None:
            return None

        now = float(self._clock())
        position_xy = _xy(position)
        self._close_previous_interval(state, now, position_xy)

        try:
            requested_motion = abs(float(vx)) + abs(float(vy)) + abs(float(vyaw))
            actual_motion = float(executed_motion)
        except (TypeError, ValueError):
            requested_motion = 0.0
            actual_motion = 0.0
        actual_nonzero = (
            math.isfinite(requested_motion)
            and math.isfinite(actual_motion)
            and requested_motion > self._MOTION_EPS
            and actual_motion > self._MOTION_EPS
        )

        state.previous_command_nonzero = actual_nonzero
        state.last_sample_at = now
        if position_xy is not None:
            state.last_position = position_xy

        if not actual_nonzero:
            return None

        state.nonzero_cmd_count += 1
        state.cmd_motion_count += actual_motion
        event = self._snapshot("velocity", "active", now=now)
        event.update(
            {
                "source": str(source),
                "velocity": [float(vx), float(vy), float(vyaw)],
            }
        )
        return event

    def finalize(
        self,
        goal_id: str,
        *,
        status: str,
        position: Any = None,
    ) -> dict[str, Any] | None:
        """Close *goal_id* and return its terminal snapshot.

        A stale finalizer for G123 cannot close a newer active G124.
        """
        state = self._active
        if state is None or state.goal_id != str(goal_id):
            return None
        now = float(self._clock())
        self._close_previous_interval(state, now, _xy(position))
        state.previous_command_nonzero = False
        state.last_sample_at = now
        position_xy = _xy(position)
        if position_xy is not None:
            state.last_position = position_xy
        event = self._snapshot("finalized", str(status), now=now)
        self._active = None
        return event

    def _close_previous_interval(
        self,
        state: _ActiveGoalMotion,
        now: float,
        position_xy: tuple[float, float] | None,
    ) -> None:
        """Accumulate only motion following a confirmed non-zero command."""
        if not state.previous_command_nonzero:
            return
        if state.last_sample_at is not None:
            dt = max(0.0, now - state.last_sample_at)
            state.nonzero_cmd_duration_s += min(dt, self._MAX_SAMPLE_GAP_S)
        if position_xy is not None and state.last_position is not None:
            state.moved_distance_m += math.hypot(
                position_xy[0] - state.last_position[0],
                position_xy[1] - state.last_position[1],
            )

    def _snapshot(self, event: str, status: str, *, now: float) -> dict[str, Any]:
        state = self._active
        if state is None:  # pragma: no cover - internal invariant
            raise RuntimeError("no active goal")
        state.seq += 1
        actual = (
            state.nonzero_cmd_count > 0
            and state.cmd_motion_count > self._MOTION_EPS
        )
        actor_caused = (
            actual and state.moved_distance_m > NAV_GOAL_DISPLACEMENT_EPS_M
        )
        return {
            "version": NAV_GOAL_TELEMETRY_VERSION,
            "event": event,
            "goal_id": state.goal_id,
            "seq": state.seq,
            "status": status,
            "target_xy": list(state.target_xy) if state.target_xy is not None else None,
            "nonzero_cmd_count": state.nonzero_cmd_count,
            "cmd_motion_count": state.cmd_motion_count,
            "nonzero_cmd_duration_s": state.nonzero_cmd_duration_s,
            "moved_distance_m": state.moved_distance_m,
            "actual_velocity_observed": actual,
            # Checklist-compatible aliases.
            "cmd_vel_count": state.nonzero_cmd_count,
            "executed_command_count": state.nonzero_cmd_count,
            "distance_travelled_m": state.moved_distance_m,
            "actor_caused": actor_caused,
            "elapsed_s": max(0.0, now - state.started_at),
        }


def _goal_scoped_navigation(method: Callable[..., bool]) -> Callable[..., bool]:
    """Wrap a blocking proxy navigation method in a goal lifecycle."""

    default_timeout = float(inspect.signature(method).parameters["timeout"].default)

    @functools.wraps(method)
    def wrapped(
        self: "Go2ROS2Proxy",
        x: float,
        y: float,
        timeout: float | None = None,
        on_progress: Callable[[float, float], None] | None = None,
        goal_id: str | None = None,
        timeout_s: float | None = None,
        **navigation_options: Any,
    ) -> bool:
        if timeout is not None and timeout_s is not None:
            if float(timeout) != float(timeout_s):
                raise ValueError("timeout and timeout_s disagree")
        selected_timeout = timeout_s if timeout_s is not None else timeout
        effective_timeout = (
            default_timeout if selected_timeout is None else float(selected_timeout)
        )
        with self._navigation_state_lock:
            navigation_generation = self._navigation_cancel_generation
        with self._navigation_call_lock:
            active_goal, owns_goal = self._acquire_navigation_goal(
                goal_id, target_xy=(x, y)
            )
            atomic_segment = any(
                navigation_options.get(name) is not None
                for name in (
                    "waypoint_kind",
                    "speed_limit_mps",
                    "allow_reverse",
                    "arrival_tolerance",
                )
            )
            try:
                segment_policy_ready = True
                if atomic_segment:
                    requested_kind = (
                        navigation_options.get("waypoint_kind") or "room_goal"
                    )
                    requested_reverse = (
                        True
                        if navigation_options.get("allow_reverse") is None
                        else navigation_options["allow_reverse"]
                    )
                    active_policy = self._active_segment_constraints
                    policy_matches = (
                        active_policy is not None
                        and active_policy.goal_id == active_goal
                        and active_policy.kind == requested_kind
                        and active_policy.speed_limit_mps
                        == navigation_options.get("speed_limit_mps")
                        and active_policy.allow_reverse == requested_reverse
                        and active_policy.tolerance
                        == navigation_options.get("arrival_tolerance")
                    )
                    if not policy_matches:
                        segment_policy_ready = (
                            self.set_navigation_segment_constraints(
                                kind=requested_kind,
                                speed_limit_mps=navigation_options.get(
                                    "speed_limit_mps"
                                ),
                                allow_reverse=requested_reverse,
                                tolerance=navigation_options.get(
                                    "arrival_tolerance"
                                ),
                                goal_id=active_goal,
                            )
                        )
                if atomic_segment and not segment_policy_ready:
                    # A connected proxy without a working bridge policy channel
                    # must not start a door segment: doing so would bypass the
                    # speed/no-reverse guarantee at the final motor boundary.
                    self._fail_closed_navigation_segment(
                        goal_id=active_goal,
                        target_xy=(x, y),
                        reason="segment_constraint_unavailable",
                        segment_kind=navigation_options.get("waypoint_kind"),
                        state="ERROR",
                    )
                    arrived = False
                else:
                    with self._navigation_state_lock:
                        still_active = (
                            navigation_generation
                            == self._navigation_cancel_generation
                            and self._active_navigation_goal_id == active_goal
                        )
                    if not still_active:
                        self.set_velocity(0.0, 0.0, 0.0)
                        arrived = False
                    else:
                        arrived = bool(
                            method(
                                self,
                                x,
                                y,
                                timeout=effective_timeout,
                                on_progress=on_progress,
                                goal_id=active_goal,
                                _navigation_generation=navigation_generation,
                                **navigation_options,
                            )
                        )
            except BaseException:
                if atomic_segment:
                    self.clear_navigation_segment_constraints(goal_id=active_goal)
                if owns_goal:
                    self.finalize_navigation_goal(active_goal, status="failed")
                raise
            if atomic_segment:
                self.clear_navigation_segment_constraints(goal_id=active_goal)
            if owns_goal:
                self.finalize_navigation_goal(
                    active_goal, status="succeeded" if arrived else "failed"
                )
            return arrived

    return wrapped

# ---------------------------------------------------------------------------
# Nav config loader (lazy, module-level cache)
# ---------------------------------------------------------------------------

_NAV_CFG: dict | None = None


def _load_nav_config() -> dict:
    """Load nav.yaml with defaults. Searches relative paths then falls back."""
    global _NAV_CFG
    if _NAV_CFG is not None:
        return _NAV_CFG

    import yaml

    _search = [
        "config/nav.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config", "nav.yaml"),
    ]
    for path in _search:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                _NAV_CFG = data
                return _NAV_CFG
            except Exception as exc:
                logger.warning("nav.yaml load failed (%s), using defaults", exc)
    _NAV_CFG = {}
    return _NAV_CFG


def _nav(key: str, default: float) -> float:
    """Look up a navigation parameter by key, return default if absent."""
    cfg = _load_nav_config()
    nav_section = cfg.get("navigation", {})
    return float(nav_section.get(key, default))


class Go2ROS2Proxy:
    """Proxy that implements the same interface as MuJoCoGo2 but via ROS2 topics.

    Publishes: /cmd_vel_nav (Twist) for velocity commands
    Subscribes: /state_estimation (Odometry) for position/heading
                /camera/image (Image) for VLM perception
    """

    _NODE_NAME: str = "go2_agent_proxy"

    def __init__(self) -> None:
        self._node: Any = None
        self._cmd_pub: Any = None
        self._position: tuple[float, float, float] = (0.0, 0.0, 0.28)
        self._sensor_position: tuple[float, float, float] | None = None
        self._state_estimation_child_frame: str = ""
        self._heading: float = 0.0
        self._connected: bool = False
        self._last_odom: Any = None
        self._last_camera_frame: Any = None  # numpy (H, W, 3) uint8
        self._last_depth_frame: Any = None   # numpy (H, W) float32 metres
        self._last_camera_ts: float = 0.0    # monotonic time of last /camera/image
        self._last_depth_ts: float = 0.0     # monotonic time of last /camera/depth
        # Real d435_rgb world pose published by the bridge over /camera/pose
        # (D36). When present, get_camera_pose() returns this EXACT pose so the
        # depth back-projection matches the camera the image was rendered from.
        # None until the first message → falls back to the mount approximation.
        self._last_cam_xpos: Any = None      # numpy (3,) world position
        self._last_cam_xmat: Any = None      # numpy (9,) row-major rotation
        self._last_cam_pose_ts: float = 0.0  # monotonic time of last /camera/pose
        # Legacy localPlanner-path diagnostic.  It is deliberately not used to
        # prove a FAR response because localPlanner can publish without a valid
        # global route.
        self._last_path_time: float = 0.0
        self._shared_runtime_used: bool = False
        # FAR's PointStamped waypoint has no request identifier.  A waypoint is
        # therefore accepted as a response only after the source-labelled FAR
        # global path proves that its terminal endpoint matches the currently
        # published planner goal.
        self._far_response_lock = threading.RLock()
        self._far_probe_generation: int = 0
        self._active_far_probe_generation: int | None = None
        self._active_far_goal_xy: tuple[float, float] | None = None
        self._active_far_goal_tolerance_m: float = 0.0
        self._active_far_goal_first_publish_time: float = 0.0
        self._active_far_probe_start_route_sequence: int = 0
        self._far_path_match_generation: int | None = None
        self._far_path_match_time: float = 0.0
        self._far_path_mismatch_count: int = 0
        self._far_empty_route_count: int = 0
        self._far_fresh_waypoint_count: int = 0
        self._associated_waypoint_time: float = 0.0
        self._associated_waypoint_pos: tuple[float, float] | None = None
        self._last_waypoint_time: float = 0.0
        self._last_waypoint_pos: tuple[float, float] | None = None
        self._last_far_route_marker_time: float = 0.0
        self._last_far_route_endpoint: tuple[float, float] | None = None
        self._last_far_route_goal_error_m: float | None = None
        self._far_route_marker_sequence: int = 0
        # FAR ignores /goal_point until its global V-Graph is non-empty.  Topic
        # discovery alone therefore is not a readiness signal: retain the
        # complete /viz_graph_topic snapshot and expose it to startup gating.
        self._far_vgraph_lock = threading.RLock()
        self._far_vgraph_subscription: Any = None
        self._far_vgraph_node_count: int = 0
        self._far_vgraph_global_vertex_seen: bool = False
        self._far_vgraph_message_count: int = 0
        self._last_far_vgraph_marker_time: float = 0.0

        # P1-05 goal-scoped navigation evidence.  The bridge owns the actual
        # motor controller; this proxy only advances actor-causation counters
        # from bridge-confirmed telemetry for goals it issued itself.
        self._goal_control_pub: Any = None
        self._segment_control_pub: Any = None
        self._segment_ack_subscription: Any = None
        self._current_goal_pub: Any = None
        self._door_path_pub: Any = None
        self._active_segment_constraints: NavigationSegmentConstraints | None = None
        self._pending_segment_constraints: NavigationSegmentConstraints | None = None
        self._current_goal_state: dict[str, Any] = {
            "goal_id": None,
            "state": "IDLE",
            "reason": "",
        }
        self._last_navigation_plan: dict[str, Any] | None = None
        self._navigation_call_lock = threading.RLock()
        self._navigation_state_lock = threading.RLock()
        self._navigation_cancel_generation: int = 0
        self._segment_ack_lock = threading.RLock()
        self._segment_ack_changed = threading.Condition(self._segment_ack_lock)
        self._segment_set_lock = threading.Lock()
        self._segment_acknowledgements: dict[str, dict[str, Any]] = {}
        self._goal_telemetry_lock = threading.RLock()
        self._goal_telemetry_changed = threading.Condition(self._goal_telemetry_lock)
        self._active_navigation_goal_id: str | None = None
        self._active_navigation_goal_owner_tid: int | None = None
        self._active_navigation_goal_external: bool = False
        self._active_navigation_target: tuple[float, float] | None = None
        self._issued_navigation_goal_ids: set[str] = set()
        self._goal_finalization_pending: set[str] = set()
        self._navigation_telemetry: dict[str, dict[str, Any]] = {}
        self._last_navigation_goal_id: str | None = None
        self._cmd_motion_total: float = 0.0
        self._nonzero_cmd_count_total: int = 0
        self._nonzero_cmd_duration_total: float = 0.0
        self._moved_distance_total: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Initialise rclpy node, publisher, and odometry subscriber."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from nav_msgs.msg import Odometry, Path
            import threading

            use_shared_runtime = (
                os.environ.get("VECTOR_SHARED_EXECUTOR", "1") == "1"
            )
            shared_runtime: Any = None
            if use_shared_runtime:
                from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

                shared_runtime = get_ros2_runtime()
                # The runtime owns the rclpy context in shared mode.  In
                # particular, it must select/initialise the requested ROS
                # domain before Node() captures that context.
                ensure_initialized = getattr(
                    shared_runtime,
                    "ensure_initialized",
                    None,
                )
                if callable(ensure_initialized):
                    ensure_initialized()
                elif not rclpy.ok():
                    # Compatibility with older external runtime adapters.
                    rclpy.init()
            elif not rclpy.ok():
                rclpy.init()

            self._node = Node(self._NODE_NAME)

            reliable_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                depth=5,
            )
            path_state_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            far_vgraph_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )

            from geometry_msgs.msg import Twist, PointStamped  # noqa: F401
            from sensor_msgs.msg import Image
            from visualization_msgs.msg import Marker, MarkerArray

            self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel_nav", 10)
            # /goal_point → FAR planner (global route planning)
            self._goal_pub = self._node.create_publisher(
                PointStamped, "/goal_point", 10
            )
            # /way_point → localPlanner (direct goal, overrides TARE at 2Hz)
            self._waypoint_pub = self._node.create_publisher(
                PointStamped, "/way_point", 10
            )
            self._node.create_subscription(
                Odometry, "/state_estimation", self._odom_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/image", self._camera_cb, reliable_qos
            )
            self._node.create_subscription(
                Image, "/camera/depth", self._depth_cb, reliable_qos
            )
            # Real d435_rgb world pose from the bridge (D36) — the single source
            # of truth for back-projecting /camera/depth in get_camera_pose().
            from std_msgs.msg import Float64MultiArray, String
            self._node.create_subscription(
                Float64MultiArray, "/camera/pose", self._cam_pose_cb, reliable_qos
            )
            # Goal identity and plant-confirmed motor evidence travel on their
            # own topics.  Do not overload PointStamped.frame_id="map".
            self._goal_control_pub = self._node.create_publisher(
                String, NAV_GOAL_CONTROL_TOPIC, reliable_qos
            )
            self._node.create_subscription(
                String,
                NAV_GOAL_TELEMETRY_TOPIC,
                self._goal_telemetry_cb,
                reliable_qos,
            )
            self._segment_control_pub = self._node.create_publisher(
                String, NAV_SEGMENT_CONTROL_TOPIC, reliable_qos
            )
            self._segment_ack_subscription = self._node.create_subscription(
                String,
                NAV_SEGMENT_ACK_TOPIC,
                self._segment_ack_cb,
                reliable_qos,
            )
            self._current_goal_pub = self._node.create_publisher(
                String, CURRENT_GOAL_TOPIC, reliable_qos
            )
            self._door_path_pub = self._node.create_publisher(
                Path, DOOR_PATH_TOPIC, path_state_qos
            )
            self._publish_empty_door_path()
            # A /way_point alone cannot identify the goal that produced it.
            # Pair it directly with FAR's native LINE_STRIP route marker: only a
            # marker whose endpoint matches the current /goal_point can arm
            # Phase 2.  FAR publishes this Marker reliable/volatile with depth 5.
            self._node.create_subscription(
                PointStamped, "/way_point", self._waypoint_cb, 10
            )
            self._node.create_subscription(
                Marker,
                FAR_ROUTE_MARKER_TOPIC,
                self._far_route_marker_cb,
                reliable_qos,
            )
            self._far_vgraph_subscription = self._node.create_subscription(
                MarkerArray,
                FAR_VGRAPH_MARKER_TOPIC,
                self._far_vgraph_marker_cb,
                far_vgraph_qos,
            )

            # Scene graph marker publisher (agent sets self._scene_graph)
            self._scene_graph = None
            self._nav_goal: tuple[float, float] | None = None
            self._trajectory: list[tuple[float, float]] = []
            self._last_marker_hash: int | None = None
            self._last_marker_publish_time: float = 0.0
            try:
                self._marker_pub = self._node.create_publisher(
                    MarkerArray, "/scene_graph_markers", 5
                )
                self._node.create_timer(3.0, self._publish_markers)
            except ImportError:
                self._marker_pub = None

            # Route to shared executor or legacy per-proxy spin.
            if use_shared_runtime:
                shared_runtime.add_node(self._node)
                self._shared_runtime_used = True
            else:
                # Legacy per-proxy spin (rollback: VECTOR_SHARED_EXECUTOR=0)
                self._spin_thread = threading.Thread(
                    target=lambda: rclpy.spin(self._node), daemon=True
                )
                self._spin_thread.start()
                self._shared_runtime_used = False
            self._connected = True

            # Wait up to 5 s for the first odometry message.
            for _ in range(50):
                if self._position != (0.0, 0.0, 0.28):
                    break
                time.sleep(0.1)

            logger.info("Go2ROS2Proxy connected")
        except ImportError as exc:
            # ROS2/rclpy not installed (expected on a macOS/Windows sim host):
            # the bridge is simply unavailable, NOT an error — so it must not
            # bleed an ERROR line into the REPL panels. Visible under --verbose.
            logger.debug("Go2ROS2Proxy: ROS2 unavailable, running without bridge: %s", exc)
            self._connected = False
        except Exception as exc:
            logger.error("Failed to connect Go2ROS2Proxy: %s", exc)
            self._connected = False

    def disconnect(self) -> None:
        """Destroy the rclpy node and mark proxy as disconnected."""
        active_goal = self._active_navigation_goal_id
        if active_goal is not None:
            self.finalize_navigation_goal(active_goal, status="cancelled")
        self._publish_empty_door_path()
        if self._shared_runtime_used and self._node is not None:
            try:
                from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime
                runtime = get_ros2_runtime()
                runtime.remove_node(self._node)
                # If this was the final shared node, drain callback futures
                # before destroying their publishers/subscriptions.
                runtime.shutdown_if_idle()
            except Exception:
                pass  # best effort — don't block teardown
        self._shared_runtime_used = False
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass
            self._node = None
        self._goal_control_pub = None
        self._segment_control_pub = None
        self._segment_ack_subscription = None
        self._current_goal_pub = None
        self._door_path_pub = None
        self._far_vgraph_subscription = None
        with self._far_vgraph_lock:
            self._far_vgraph_node_count = 0
            self._far_vgraph_global_vertex_seen = False
            self._far_vgraph_message_count = 0
            self._last_far_vgraph_marker_time = 0.0
        self._active_segment_constraints = None
        self._pending_segment_constraints = None
        with self._segment_ack_changed:
            self._segment_acknowledgements.clear()
            self._segment_ack_changed.notify_all()
        self._connected = False

    # ------------------------------------------------------------------
    # Internal callback
    # ------------------------------------------------------------------

    def _odom_cb(self, msg: Any) -> None:
        """Update the body pose from the navigation stack's odometry message.

        The bundled CMU stack uses ``child_frame_id=sensor`` and publishes the
        front lidar pose.  ``BaseProtocol.get_position()`` is explicitly the
        robot body centre, so sensor-frame messages are converted exactly once.
        Other simulators publishing ``base_link`` (or no child frame) retain
        their existing body-pose behaviour.
        """
        self._last_odom = msg
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._heading = math.atan2(siny_cosp, cosy_cosp)
        raw_position = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
            float(msg.pose.pose.position.z),
        )
        child = getattr(msg, "child_frame_id", "")
        child_frame = (
            child.strip().lstrip("/")
            if isinstance(child, str)
            else ""
        )
        self._state_estimation_child_frame = child_frame
        if child_frame == "sensor":
            self._sensor_position = raw_position
            self._position = sensor_to_body_position(
                raw_position,
                self._heading,
            )
        else:
            self._sensor_position = None
            self._position = raw_position

    def _camera_cb(self, msg: Any) -> None:
        """Cache latest camera frame from /camera/image (RGB8 240x320)."""
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.uint8)
            frame = frame.reshape((msg.height, msg.width, 3))
            self._last_camera_frame = frame
            self._last_camera_ts = time.monotonic()
        except Exception:
            pass

    def _depth_cb(self, msg: Any) -> None:
        """Cache latest depth frame from /camera/depth (32FC1 240x320).

        Bridge publishes depth as 32FC1 (float32, single channel) in metres.
        """
        try:
            import numpy as np
            frame = np.frombuffer(msg.data, dtype=np.float32)
            frame = frame.reshape((msg.height, msg.width))
            self._last_depth_frame = frame
            self._last_depth_ts = time.monotonic()
        except Exception:
            pass

    def _cam_pose_cb(self, msg: Any) -> None:
        """Cache the real d435_rgb world pose from /camera/pose (D36).

        Flat layout [xpos(3), xmat(9)] published by the bridge from the live
        MjData. Cached as numpy arrays so get_camera_pose() can return the EXACT
        pose the depth was rendered from instead of a mount approximation.
        """
        try:
            import numpy as np
            data = list(msg.data)
            if len(data) != 12:
                return
            self._last_cam_xpos = np.array(data[:3], dtype=float)
            self._last_cam_xmat = np.array(data[3:], dtype=float)
            self._last_cam_pose_ts = time.monotonic()
        except Exception:
            pass

    def _waypoint_cb(self, msg: Any) -> None:
        """Record a waypoint, associating it only with a proven current path.

        FAR waypoints are legitimate intermediate route points and need not be
        close to the final target.  We therefore never compare the waypoint
        coordinate itself with the goal.  Instead, ``_far_route_marker_cb`` must
        first observe FAR's LINE_STRIP marker ending at the current planner
        goal; only a waypoint received after that proof is associated.
        """

        position = _xy(
            (
                getattr(getattr(msg, "point", None), "x", None),
                getattr(getattr(msg, "point", None), "y", None),
            )
        )
        if position is None:
            return
        received_at = time.time()
        with self._far_response_lock:
            self._last_waypoint_time = received_at
            self._last_waypoint_pos = position
            generation = self._active_far_probe_generation
            first_publish = self._active_far_goal_first_publish_time
            if (
                generation is None
                or first_publish <= 0.0
                or received_at < first_publish
            ):
                return
            self._far_fresh_waypoint_count += 1
            if (
                self._far_path_match_generation == generation
                and received_at >= self._far_path_match_time
            ):
                self._associated_waypoint_time = received_at
                self._associated_waypoint_pos = position

    def _far_route_marker_cb(self, msg: Any) -> None:
        """Associate FAR's native LINE_STRIP marker with the current goal."""

        points = list(getattr(msg, "points", ()) or ())
        received_at = time.time()
        with self._far_response_lock:
            self._far_route_marker_sequence += 1
            route_sequence = self._far_route_marker_sequence
            self._last_far_route_marker_time = received_at
            if not points:
                # FAR publishes an empty marker to mean that no route is
                # currently available.  It advances the observed generation and
                # clears prior proof, but can never associate a later waypoint.
                self._last_far_route_endpoint = None
                self._last_far_route_goal_error_m = None
                self._far_empty_route_count += 1
                self._far_path_match_generation = None
                self._far_path_match_time = 0.0
                self._associated_waypoint_time = 0.0
                self._associated_waypoint_pos = None
                return

            endpoint = _xy(
                (
                    getattr(points[-1], "x", None),
                    getattr(points[-1], "y", None),
                )
            )
            if endpoint is None:
                return
            self._last_far_route_endpoint = endpoint
            generation = self._active_far_probe_generation
            expected = self._active_far_goal_xy
            first_publish = self._active_far_goal_first_publish_time
            if (
                generation is None
                or expected is None
                or first_publish <= 0.0
                or received_at < first_publish
                or route_sequence <= self._active_far_probe_start_route_sequence
            ):
                self._last_far_route_goal_error_m = None
                return
            error = math.hypot(endpoint[0] - expected[0], endpoint[1] - expected[1])
            self._last_far_route_goal_error_m = error
            if error <= self._active_far_goal_tolerance_m:
                # Keep the first matching timestamp.  The bridge can republish
                # the same path more frequently than FAR publishes waypoints;
                # resetting this timestamp on every path would starve the
                # required "waypoint after matching path" evidence.
                if self._far_path_match_generation != generation:
                    self._far_path_match_generation = generation
                    self._far_path_match_time = received_at
                    self._associated_waypoint_time = 0.0
                    self._associated_waypoint_pos = None
                return
            # A newer non-empty route supersedes the earlier proof.  Keeping a
            # previous match here would let the next stale waypoint inherit it.
            self._far_path_mismatch_count += 1
            self._far_path_match_generation = None
            self._far_path_match_time = 0.0
            self._associated_waypoint_time = 0.0
            self._associated_waypoint_pos = None

    def _far_vgraph_marker_cb(self, msg: Any) -> None:
        """Record whether FAR's latest global V-Graph is actually non-empty.

        ``/viz_graph_topic`` is a complete MarkerArray snapshot.  FAR's
        ``DPVisualizer::VizGraph`` identifies graph vertices with the
        ``global_vertex`` namespace and emits one point per node.  Treat an
        absent or empty marker as an explicit not-ready update so a reset graph
        cannot inherit readiness from an older message.
        """

        node_count = 0
        global_vertex_seen = False
        for marker in list(getattr(msg, "markers", ()) or ()):
            if getattr(marker, "ns", "") != FAR_VGRAPH_GLOBAL_VERTEX_NS:
                continue
            global_vertex_seen = True
            node_count += len(list(getattr(marker, "points", ()) or ()))

        with self._far_vgraph_lock:
            self._far_vgraph_node_count = node_count
            self._far_vgraph_global_vertex_seen = global_vertex_seen
            self._far_vgraph_message_count += 1
            self._last_far_vgraph_marker_time = time.time()

    def far_vgraph_ready(self) -> bool:
        """Return True only after FAR reports at least one global graph node."""

        with self._far_vgraph_lock:
            return self._far_vgraph_node_count > 0

    def far_vgraph_diagnostics(self) -> dict[str, Any]:
        """Return a thread-safe snapshot of FAR graph readiness evidence."""

        with self._far_vgraph_lock:
            node_count = self._far_vgraph_node_count
            marker_time = self._last_far_vgraph_marker_time
            message_count = self._far_vgraph_message_count
            global_vertex_seen = self._far_vgraph_global_vertex_seen
        if node_count > 0:
            status = "ready"
        elif message_count > 0:
            status = "empty_graph"
        else:
            status = "waiting_for_marker"
        return {
            "topic": FAR_VGRAPH_MARKER_TOPIC,
            "status": status,
            "ready": node_count > 0,
            "node_count": node_count,
            "global_vertex_marker_seen": global_vertex_seen,
            "message_count": message_count,
            "last_marker_time": marker_time,
            "marker_age_s": (
                max(0.0, time.time() - marker_time)
                if marker_time > 0.0
                else None
            ),
        }

    def _begin_far_goal_probe(
        self,
        expected_goal_xy: tuple[float, float],
        *,
        match_tolerance_m: float,
    ) -> int:
        """Start an isolated response generation for one planner goal."""

        expected = _xy(expected_goal_xy)
        tolerance = float(match_tolerance_m)
        if expected is None or not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("FAR probe requires a finite goal and positive tolerance")
        # A generous metre covers planner endpoint quantisation while remaining
        # far below the separation between distinct room/door goals.  Never let
        # configuration widen association enough to bind another room's route.
        tolerance = min(tolerance, FAR_GOAL_MATCH_TOLERANCE_MAX_M)
        with self._far_response_lock:
            self._far_probe_generation += 1
            generation = self._far_probe_generation
            self._active_far_probe_generation = generation
            self._active_far_goal_xy = expected
            self._active_far_goal_tolerance_m = tolerance
            self._active_far_goal_first_publish_time = 0.0
            self._active_far_probe_start_route_sequence = (
                self._far_route_marker_sequence
            )
            self._far_path_match_generation = None
            self._far_path_match_time = 0.0
            self._far_path_mismatch_count = 0
            self._far_empty_route_count = 0
            self._far_fresh_waypoint_count = 0
            self._associated_waypoint_time = 0.0
            self._associated_waypoint_pos = None
            # Retain the historical fields for diagnostics and compatibility,
            # but clear them so a transient-local old path/waypoint delivered
            # before the first current publish cannot satisfy this generation.
            self._last_waypoint_time = 0.0
            self._last_waypoint_pos = None
            self._last_far_route_marker_time = 0.0
            self._last_far_route_endpoint = None
            self._last_far_route_goal_error_m = None
            return generation

    def _mark_far_goal_published(self, generation: int) -> None:
        """Open the response window after the first publish of this generation."""

        published_at = time.time()
        with self._far_response_lock:
            if (
                self._active_far_probe_generation == generation
                and self._active_far_goal_first_publish_time <= 0.0
            ):
                self._active_far_goal_first_publish_time = published_at

    def _far_probe_snapshot(self, generation: int) -> dict[str, Any]:
        """Return immutable association evidence for the probe loop/telemetry."""

        with self._far_response_lock:
            active = self._active_far_probe_generation == generation
            matched = active and self._far_path_match_generation == generation
            associated = matched and self._associated_waypoint_time > 0.0
            expected = self._active_far_goal_xy if active else None
            endpoint = self._last_far_route_endpoint if active else None
            waypoint = self._last_waypoint_pos if active else None
            return {
                "far_response_generation": generation,
                "far_response_associated": bool(associated),
                "planner_goal_xy": list(expected) if expected is not None else None,
                "observed_waypoint_xy": (
                    list(waypoint) if waypoint is not None else None
                ),
                "far_path_endpoint_xy": (
                    list(endpoint) if endpoint is not None else None
                ),
                "far_path_goal_error_m": self._last_far_route_goal_error_m,
                "far_path_match_tolerance_m": self._active_far_goal_tolerance_m,
                "far_path_mismatch_count": self._far_path_mismatch_count,
                "far_empty_route_count": self._far_empty_route_count,
                "far_path_sequence": self._far_route_marker_sequence,
                "far_probe_start_path_sequence": (
                    self._active_far_probe_start_route_sequence
                ),
                "fresh_waypoint_count": self._far_fresh_waypoint_count,
            }

    def _goal_telemetry_cb(self, msg: Any) -> None:
        """Ingest one JSON telemetry snapshot published by the bridge."""
        self._ingest_navigation_telemetry(getattr(msg, "data", ""))

    def _segment_ack_cb(self, msg: Any) -> None:
        """Record a bridge-confirmed segment generation for the waiting caller."""

        try:
            payload = json.loads(getattr(msg, "data", ""))
            if int(payload.get("version", -1)) != NAV_SEGMENT_CONTROL_VERSION:
                return
            event = str(payload.get("event", "")).upper()
            if event not in {"APPLIED", "REJECTED"}:
                return
            goal_id = _normalise_goal_id(payload.get("goal_id"))
            segment_id = _normalise_segment_id(payload.get("segment_id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        with self._segment_ack_changed:
            pending = self._pending_segment_constraints
            if (
                pending is None
                or pending.goal_id != goal_id
                or pending.segment_id != segment_id
            ):
                return
            self._segment_acknowledgements[segment_id] = {
                **payload,
                "event": event,
                "goal_id": goal_id,
                "segment_id": segment_id,
            }
            self._segment_ack_changed.notify_all()

    def _ingest_navigation_telemetry(self, payload: str | dict[str, Any]) -> bool:
        """Validate and merge one monotonic bridge telemetry snapshot.

        Returns True only when a newer snapshot for a goal issued by this proxy
        was accepted.  Stray goals, duplicates, regressions, and malformed
        numeric values fail closed and never advance actor-causation counters.
        """
        try:
            data = json.loads(payload) if isinstance(payload, str) else dict(payload)
            if int(data.get("version", -1)) != NAV_GOAL_TELEMETRY_VERSION:
                return False
            goal_id = _normalise_goal_id(data.get("goal_id", ""))
            seq = int(data.get("seq", -1))
            nonzero_count = int(data.get("nonzero_cmd_count", 0))
            cmd_motion = float(data.get("cmd_motion_count", 0.0))
            duration = float(data.get("nonzero_cmd_duration_s", 0.0))
            moved = float(data.get("moved_distance_m", 0.0))
            elapsed = float(data.get("elapsed_s", 0.0))
            incoming_target = _xy(data.get("target_xy"))
        except (OverflowError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if (
            seq < 1
            or nonzero_count < 0
            or cmd_motion < 0.0
            or duration < 0.0
            or moved < 0.0
            or elapsed < 0.0
            or not all(
                math.isfinite(v) for v in (cmd_motion, duration, moved, elapsed)
            )
        ):
            return False
        has_nonzero_count = nonzero_count > 0
        has_motion = cmd_motion > GoalMotionTracker._MOTION_EPS
        if has_nonzero_count != has_motion:
            # Cumulative command count and cumulative plant-confirmed magnitude
            # describe the same events. Accepting a one-sided sample would let a
            # malformed/stale publisher advance actor counters without evidence.
            return False

        with self._goal_telemetry_changed:
            if goal_id not in self._issued_navigation_goal_ids:
                return False
            if (
                goal_id != self._active_navigation_goal_id
                and goal_id not in self._goal_finalization_pending
            ):
                # Goal-scoped isolation: a delayed G123 sample must never move
                # the cumulative counter inside a later G124 actor window.
                return False
            previous = self._navigation_telemetry.get(
                goal_id, self._empty_navigation_telemetry(goal_id)
            )
            if seq <= int(previous.get("seq", 0)):
                return False
            previous_count = int(previous.get("nonzero_cmd_count", 0))
            previous_motion = float(previous.get("cmd_motion_count", 0.0))
            previous_duration = float(previous.get("nonzero_cmd_duration_s", 0.0))
            previous_moved = float(previous.get("moved_distance_m", 0.0))
            # A cumulative metric may never go backwards within one goal.
            if (
                nonzero_count < previous_count
                or cmd_motion < previous_motion
                or duration < previous_duration
                or moved < previous_moved
            ):
                return False

            event = str(data.get("event", "velocity"))
            if event not in {"accepted", "velocity", "finalized"}:
                return False
            actual = nonzero_count > 0 and cmd_motion > GoalMotionTracker._MOTION_EPS
            actor_caused = actual and moved > NAV_GOAL_DISPLACEMENT_EPS_M
            incoming_status = str(data.get("status", "active"))
            previous_status = str(previous.get("status", "active"))
            # A delayed non-terminal sample can contribute metrics, but cannot
            # overwrite a locally requested failed/cancelled terminal state.
            terminal = previous_status not in {"issued", "active"}
            status = previous_status if terminal and event != "finalized" else incoming_status
            merged = {
                **previous,
                "version": NAV_GOAL_TELEMETRY_VERSION,
                "event": event,
                "goal_id": goal_id,
                "seq": seq,
                "status": status,
                "target_xy": (
                    list(incoming_target)
                    if incoming_target is not None
                    else previous.get("target_xy")
                ),
                "nonzero_cmd_count": nonzero_count,
                "cmd_motion_count": cmd_motion,
                "nonzero_cmd_duration_s": duration,
                "moved_distance_m": moved,
                "actual_velocity_observed": actual,
                "cmd_vel_count": nonzero_count,
                "executed_command_count": nonzero_count,
                "distance_travelled_m": moved,
                "actor_caused": actor_caused,
                "elapsed_s": elapsed,
            }
            self._navigation_telemetry[goal_id] = merged
            if event == "finalized":
                self._goal_finalization_pending.discard(goal_id)
                if self._active_navigation_goal_id == goal_id:
                    self._active_navigation_goal_id = None
                    self._active_navigation_goal_owner_tid = None
                    self._active_navigation_goal_external = False
                    self._active_navigation_target = None
            self._cmd_motion_total += cmd_motion - previous_motion
            self._nonzero_cmd_count_total += nonzero_count - previous_count
            self._nonzero_cmd_duration_total += duration - previous_duration
            self._moved_distance_total += moved - previous_moved
            self._goal_telemetry_changed.notify_all()
            return True

    @staticmethod
    def _empty_navigation_telemetry(
        goal_id: str, target_xy: Any = None
    ) -> dict[str, Any]:
        target = _xy(target_xy)
        return {
            "version": NAV_GOAL_TELEMETRY_VERSION,
            "event": "issued",
            "goal_id": goal_id,
            "seq": 0,
            "status": "issued",
            "target_xy": list(target) if target is not None else None,
            "nonzero_cmd_count": 0,
            "cmd_motion_count": 0.0,
            "nonzero_cmd_duration_s": 0.0,
            "moved_distance_m": 0.0,
            "actual_velocity_observed": False,
            "cmd_vel_count": 0,
            "executed_command_count": 0,
            "distance_travelled_m": 0.0,
            "actor_caused": False,
            "elapsed_s": 0.0,
        }

    def _publish_goal_control(
        self,
        event: str,
        goal_id: str,
        *,
        status: str = "active",
        target_xy: Any = None,
    ) -> None:
        """Publish one idempotent goal-control announcement."""
        if self._node is None or self._goal_control_pub is None:
            return
        try:
            from std_msgs.msg import String

            target = _xy(target_xy)
            payload = {
                "version": NAV_GOAL_TELEMETRY_VERSION,
                "event": str(event),
                "goal_id": _normalise_goal_id(goal_id),
                "status": str(status),
                "target_xy": list(target) if target is not None else None,
                "published_at": time.time(),
            }
            msg = String()
            msg.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            self._goal_control_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish goal control: %s", exc)

    def _publish_current_goal_state(
        self,
        state: str,
        *,
        goal_id: str | None = None,
        target_xy: Any = None,
        reason: str = "",
        **details: Any,
    ) -> None:
        """Publish an explicit JSON state without conflicting ROS topic types."""
        selected_goal = goal_id or self._active_navigation_goal_id
        target = _xy(target_xy)
        payload: dict[str, Any] = {
            "version": 1,
            "goal_id": selected_goal,
            "state": str(state),
            "target_xy": (
                list(target)
                if target is not None
                else (
                    list(self._active_navigation_target)
                    if self._active_navigation_target is not None
                    else None
                )
            ),
            "reason": str(reason),
            "published_at": time.time(),
        }
        payload.update({key: value for key, value in details.items() if value is not None})
        self._current_goal_state = payload
        if self._node is None or self._current_goal_pub is None:
            return
        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = json.dumps(payload, separators=(",", ":"), sort_keys=True)
            self._current_goal_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish current goal state: %s", exc)

    def set_navigation_segment_constraints(
        self,
        *,
        kind: str = "room_goal",
        speed_limit_mps: float | None = None,
        allow_reverse: bool = True,
        tolerance: float | None = None,
        goal_id: str | None = None,
    ) -> bool:
        """Serialise policy handshakes so a pending ACK cannot be bypassed."""

        with self._segment_set_lock:
            return self._set_navigation_segment_constraints(
                kind=kind,
                speed_limit_mps=speed_limit_mps,
                allow_reverse=allow_reverse,
                tolerance=tolerance,
                goal_id=goal_id,
            )

    def _set_navigation_segment_constraints(
        self,
        *,
        kind: str = "room_goal",
        speed_limit_mps: float | None = None,
        allow_reverse: bool = True,
        tolerance: float | None = None,
        goal_id: str | None = None,
    ) -> bool:
        """Install and await bridge ACK for one goal-scoped segment policy."""
        selected_goal = goal_id or self._active_navigation_goal_id
        if selected_goal is None:
            raise ValueError("segment constraints require an active goal_id")
        selected_goal = _normalise_goal_id(selected_goal)
        if self._active_navigation_goal_id != selected_goal:
            logger.warning(
                "[NAV] Refusing segment policy for inactive goal %s",
                selected_goal,
            )
            return False
        constraints = NavigationSegmentConstraints(
            goal_id=selected_goal,
            segment_id=f"S{uuid.uuid4().hex[:12]}",
            kind=kind,
            speed_limit_mps=speed_limit_mps,
            allow_reverse=allow_reverse,
            tolerance=tolerance,
        )
        with self._segment_ack_changed:
            if (
                self._active_segment_constraints is not None
                or self._pending_segment_constraints is not None
            ):
                logger.warning(
                    "[NAV] Refusing overlapping segment policy handshake"
                )
                return False
            self._pending_segment_constraints = constraints
            self._segment_acknowledgements.pop(
                str(constraints.segment_id), None
            )
        self._publish_current_goal_state(
            self._current_goal_state.get("state", "GOAL_ACCEPTED"),
            goal_id=constraints.goal_id,
            segment_kind=constraints.kind,
            speed_limit_mps=constraints.speed_limit_mps,
            allow_reverse=constraints.allow_reverse,
            arrival_tolerance=constraints.tolerance,
            segment_id=constraints.segment_id,
        )
        if (
            self._node is None
            or self._segment_control_pub is None
            or self._segment_ack_subscription is None
        ):
            with self._segment_ack_changed:
                if self._pending_segment_constraints == constraints:
                    self._pending_segment_constraints = None
                    self._segment_ack_changed.notify_all()
            return False
        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = json.dumps(
                constraints.to_payload(), separators=(",", ":"), sort_keys=True
            )
            segment_id = str(constraints.segment_id)
            deadline = time.monotonic() + NAV_SEGMENT_ACK_TIMEOUT_S

            while time.monotonic() < deadline:
                with self._segment_ack_changed:
                    if self._pending_segment_constraints != constraints:
                        break
                # Repeating begin before each set repairs cross-topic reordering:
                # a bridge that still owns the previous goal rejects this
                # generation, then accepts the retry after begin is processed.
                self._announce_active_navigation_goal()
                self._segment_control_pub.publish(msg)
                retry_deadline = min(
                    deadline,
                    time.monotonic() + NAV_SEGMENT_ACK_RETRY_S,
                )
                with self._segment_ack_changed:
                    while (
                        segment_id not in self._segment_acknowledgements
                        and self._pending_segment_constraints == constraints
                        and time.monotonic() < retry_deadline
                    ):
                        self._segment_ack_changed.wait(
                            max(0.0, retry_deadline - time.monotonic())
                        )
                    ack = self._segment_acknowledgements.pop(segment_id, None)
                    if (
                        ack is not None
                        and ack.get("goal_id") == selected_goal
                        and ack.get("event") == "APPLIED"
                        and self._active_navigation_goal_id == selected_goal
                        and self._pending_segment_constraints == constraints
                    ):
                        self._pending_segment_constraints = None
                        self._active_segment_constraints = constraints
                        return True

            logger.warning(
                "[NAV] Bridge did not apply segment policy %s for goal %s",
                constraints.segment_id,
                selected_goal,
            )
            # Same-publisher ordering ensures this clear follows every retry;
            # a late APPLIED callback cannot leave an orphan policy installed.
            clear_msg = String()
            clear_msg.data = json.dumps(
                constraints.to_payload(event="clear"),
                separators=(",", ":"),
                sort_keys=True,
            )
            self._segment_control_pub.publish(clear_msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish segment constraints: %s", exc)
        with self._segment_ack_changed:
            if self._pending_segment_constraints == constraints:
                self._pending_segment_constraints = None
            self._segment_acknowledgements.pop(
                str(constraints.segment_id), None
            )
            self._segment_ack_changed.notify_all()
        return False

    def clear_navigation_segment_constraints(
        self,
        *,
        goal_id: str | None = None,
    ) -> bool:
        """Clear a matching segment policy; stale clears cannot affect a new goal."""
        selected_goal = goal_id or self._active_navigation_goal_id
        with self._segment_ack_changed:
            policy = (
                self._active_segment_constraints
                or self._pending_segment_constraints
            )
            if policy is None:
                return False
            if selected_goal is None:
                selected_goal = policy.goal_id
            if selected_goal is None:
                return False
            selected_goal = _normalise_goal_id(selected_goal)
            if policy.goal_id != selected_goal:
                return False
            if self._active_segment_constraints == policy:
                self._active_segment_constraints = None
            if self._pending_segment_constraints == policy:
                self._pending_segment_constraints = None
            self._segment_acknowledgements.pop(
                str(policy.segment_id), None
            )
            self._segment_ack_changed.notify_all()
        if self._node is None or self._segment_control_pub is None:
            return False
        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = json.dumps(
                policy.to_payload(event="clear"),
                separators=(",", ":"),
                sort_keys=True,
            )
            self._segment_control_pub.publish(msg)
            return True
        except Exception as exc:
            logger.warning("[NAV] Failed to clear segment constraints: %s", exc)
            return False

    def _fail_closed_navigation_segment(
        self,
        *,
        goal_id: str | None,
        target_xy: tuple[float, float],
        reason: str,
        segment_kind: str | None,
        state: str = "NO_PATH",
        **details: Any,
    ) -> bool:
        """Disarm autonomous following and hold zero after a segment failure."""

        self._invalidate_navigation_gate()
        self._nav_goal = None
        self.set_velocity(0.0, 0.0, 0.0)
        self._publish_empty_door_path()
        self._publish_current_goal_state(
            state,
            goal_id=goal_id,
            target_xy=target_xy,
            reason=reason,
            segment_kind=segment_kind,
            **details,
        )
        return False

    def _invalidate_navigation_gate(self) -> None:
        """Atomically invalidate in-flight starts and remove the bridge gate."""

        with self._navigation_state_lock:
            self._navigation_cancel_generation += 1
            try:
                os.remove(nav_active_file())
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("[NAV] Could not remove nav gate: %s", exc)

    def _navigation_generation_is_active(
        self,
        generation: int | None,
        goal_id: str | None,
    ) -> bool:
        with self._navigation_state_lock:
            return (
                generation is not None
                and generation == self._navigation_cancel_generation
                and goal_id is not None
                and self._active_navigation_goal_id == goal_id
            )

    @staticmethod
    def _route_waypoint_xy(waypoint: Any) -> tuple[float, float] | None:
        raw_xy = (
            waypoint.get("xy")
            if isinstance(waypoint, dict)
            else getattr(waypoint, "xy", None)
        )
        return _xy(raw_xy)

    def publish_navigation_plan(
        self,
        route: Any,
        *,
        goal_id: str | None = None,
    ) -> bool:
        """Publish the high-level door chain as ``/scene_graph/door_path``."""
        if hasattr(route, "to_dict"):
            route_data = route.to_dict()
        elif isinstance(route, dict):
            route_data = dict(route)
        else:
            route_data = {"waypoints": list(route)}
        waypoints = list(route_data.get("waypoints") or [])
        points = [
            point
            for waypoint in waypoints
            if (point := self._route_waypoint_xy(waypoint)) is not None
        ]
        self._last_navigation_plan = route_data
        selected_goal = goal_id or self._active_navigation_goal_id
        self._publish_current_goal_state(
            "GOAL_ACCEPTED",
            goal_id=selected_goal,
            door_path_points=len(points),
            route_rooms=route_data.get("room_path"),
        )
        if self._node is None or self._door_path_pub is None:
            return False
        try:
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Path

            stamp = self._node.get_clock().now().to_msg()
            msg = Path()
            msg.header.stamp = stamp
            msg.header.frame_id = "map"
            for x, y in points:
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = "map"
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.08
                pose.pose.orientation.w = 1.0
                msg.poses.append(pose)
            self._door_path_pub.publish(msg)
            return True
        except Exception as exc:
            logger.warning("[NAV] Failed to publish door path: %s", exc)
            return False

    def _publish_empty_door_path(self) -> bool:
        """Clear the proxy-owned topology path at a goal lifecycle boundary."""

        self._last_navigation_plan = None
        if self._node is None or self._door_path_pub is None:
            return False
        try:
            from nav_msgs.msg import Path

            msg = Path()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            self._door_path_pub.publish(msg)
            return True
        except Exception as exc:
            logger.warning("[NAV] Failed to clear door path: %s", exc)
            return False

    def _begin_navigation_goal(
        self,
        goal_id: str | None,
        *,
        target_xy: Any,
        externally_managed: bool,
    ) -> str:
        """Create the local half of a navigation goal lifecycle."""
        resolved_id = (
            _normalise_goal_id(goal_id)
            if goal_id is not None
            else f"G{uuid.uuid4().hex[:12]}"
        )
        current = self._active_navigation_goal_id
        if current is not None and current != resolved_id:
            self.finalize_navigation_goal(current, status="superseded")
        # A new goal must never inherit the old route if planning fails before
        # publish_navigation_plan() can produce a replacement.
        self._publish_empty_door_path()

        target = _xy(target_xy)
        with self._goal_telemetry_changed:
            self._issued_navigation_goal_ids.add(resolved_id)
            self._navigation_telemetry.setdefault(
                resolved_id,
                self._empty_navigation_telemetry(resolved_id, target),
            )
            self._active_navigation_goal_id = resolved_id
            self._active_navigation_goal_owner_tid = threading.get_ident()
            self._active_navigation_goal_external = bool(externally_managed)
            self._active_navigation_target = target
            self._last_navigation_goal_id = resolved_id
        self._publish_goal_control(
            "begin", resolved_id, status="active", target_xy=target
        )
        self._publish_current_goal_state(
            "GOAL_ACCEPTED", goal_id=resolved_id, target_xy=target
        )
        return resolved_id

    def begin_navigation_goal(
        self, goal_id: str | None = None, target_xy: Any = None
    ) -> str:
        """Begin an externally managed goal spanning one or more nav calls.

        Native adapters that delegate through multiple skills may call this
        before the first operation, then call ``finalize_navigation_goal`` once
        the whole operation succeeds, fails, or is cancelled.
        """
        with self._navigation_call_lock:
            return self._begin_navigation_goal(
                goal_id, target_xy=target_xy, externally_managed=True
            )

    def _acquire_navigation_goal(
        self, goal_id: str | None, *, target_xy: Any
    ) -> tuple[str, bool]:
        """Join a matching active goal or start a method-owned one."""
        requested = _normalise_goal_id(goal_id) if goal_id is not None else None
        current = self._active_navigation_goal_id
        same_thread = self._active_navigation_goal_owner_tid == threading.get_ident()
        if current is not None and (
            requested == current or (requested is None and same_thread)
        ):
            return current, False
        new_id = self._begin_navigation_goal(
            requested, target_xy=target_xy, externally_managed=False
        )
        return new_id, True

    def _announce_active_navigation_goal(self) -> None:
        """Repeat the active begin message; the bridge treats it idempotently."""
        goal_id = self._active_navigation_goal_id
        if goal_id is not None:
            self._publish_goal_control(
                "begin",
                goal_id,
                status="active",
                target_xy=self._active_navigation_target,
            )

    def finalize_navigation_goal(self, goal_id: str, status: str) -> dict[str, Any]:
        """Finalize a goal locally and ask the bridge for its terminal snapshot."""
        goal_id = _normalise_goal_id(goal_id)
        status = str(status).strip().lower() or "failed"
        terminal_target = self._active_navigation_target
        previous_goal_state = (
            str(self._current_goal_state.get("state", ""))
            if self._current_goal_state.get("goal_id") == goal_id
            else ""
        )
        with self._goal_telemetry_changed:
            stats = self._navigation_telemetry.get(goal_id)
            if stats is None:
                stats = self._empty_navigation_telemetry(goal_id)
                self._navigation_telemetry[goal_id] = stats
                self._issued_navigation_goal_ids.add(goal_id)
            already_terminal = (
                str(stats.get("status", "issued")) not in {"issued", "active"}
                and self._active_navigation_goal_id != goal_id
            )
            if already_terminal:
                self._publish_empty_door_path()
                return dict(stats)
            if self._active_navigation_goal_id == goal_id:
                self._active_navigation_goal_id = None
                self._active_navigation_goal_owner_tid = None
                self._active_navigation_goal_external = False
                self._active_navigation_target = None
            self._goal_finalization_pending.add(goal_id)
            bridge_seen = int(stats.get("seq", 0)) > 0

        self._publish_goal_control("finalize", goal_id, status=status)
        self.clear_navigation_segment_constraints(goal_id=goal_id)
        self._publish_empty_door_path()
        terminal_state = {
            "succeeded": "ARRIVED",
            "cancelled": "CANCELLED",
            "superseded": "CANCELLED",
        }.get(status, "ERROR")
        preserve_terminal_details = (
            status == "failed" and previous_goal_state in {"NO_PATH", "ERROR"}
        )
        if preserve_terminal_details:
            terminal_state = previous_goal_state
        preserved_details = (
            {
                key: value
                for key, value in self._current_goal_state.items()
                if key
                not in {
                    "version",
                    "goal_id",
                    "state",
                    "target_xy",
                    "reason",
                    "published_at",
                }
            }
            if preserve_terminal_details
            else {}
        )
        self._publish_current_goal_state(
            terminal_state,
            goal_id=goal_id,
            target_xy=terminal_target,
            reason=(
                str(self._current_goal_state.get("reason", ""))
                if preserve_terminal_details
                else ("" if terminal_state == "ARRIVED" else status)
            ),
            **preserved_details,
        )

        # If the bridge has acknowledged this goal, give its reliable terminal
        # snapshot a short opportunity to arrive before the blocking nav method
        # returns and actor_causation captures the post-state.
        if bridge_seen:
            with self._goal_telemetry_changed:
                self._goal_telemetry_changed.wait_for(
                    lambda: self._navigation_telemetry[goal_id].get("event")
                    == "finalized",
                    timeout=0.5,
                )

        with self._goal_telemetry_changed:
            current = self._navigation_telemetry[goal_id]
            if current.get("event") != "finalized":
                current = {
                    **current,
                    "event": "finalized_local",
                    "status": status,
                }
                self._navigation_telemetry[goal_id] = current
            self._goal_finalization_pending.discard(goal_id)
            self._goal_telemetry_changed.notify_all()
            return dict(current)

    def get_navigation_telemetry(self, goal_id: str | None = None) -> dict[str, Any]:
        """Return a frozen telemetry snapshot for *goal_id* (or the last goal)."""
        selected = goal_id or self._last_navigation_goal_id
        if selected is None:
            return self._empty_navigation_telemetry("")
        selected = str(selected)
        with self._goal_telemetry_lock:
            return dict(
                self._navigation_telemetry.get(
                    selected, self._empty_navigation_telemetry(selected)
                )
            )

    def get_navigation_goal_state(
        self, goal_id: str | None = None,
    ) -> dict[str, Any]:
        """Return the latest structured planner/executor state.

        This is intentionally separate from actor-causation telemetry: callers
        need the transport reason (NO_PATH, segment timeout, or stall) instead
        of collapsing every false return from ``navigate_to`` into "no path".
        """

        with self._navigation_state_lock:
            state = dict(self._current_goal_state)
        if goal_id is not None and state.get("goal_id") != str(goal_id):
            return {}
        return state

    def cmd_motion(self) -> float:
        """Cumulative plant-confirmed goal velocity magnitude for R2b capture."""
        with self._goal_telemetry_lock:
            return float(self._cmd_motion_total)

    def nonzero_cmd_count(self) -> int:
        with self._goal_telemetry_lock:
            return int(self._nonzero_cmd_count_total)

    def nonzero_cmd_duration_s(self) -> float:
        with self._goal_telemetry_lock:
            return float(self._nonzero_cmd_duration_total)

    def moved_distance_m(self) -> float:
        with self._goal_telemetry_lock:
            return float(self._moved_distance_total)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_position(self) -> tuple[float, float, float]:
        """Return the last known robot body-centre position in metres."""
        return self._position

    def get_sensor_position(self) -> tuple[float, float, float] | None:
        """Return raw sensor-frame odometry when the transport provides it."""

        return self._sensor_position

    def get_heading(self) -> float:
        """Return last known heading in radians (yaw from odometry)."""
        return self._heading

    def get_camera_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest RGB camera frame as (H, W, 3) uint8 numpy array.

        Received from /camera/image topic published by Go2VNavBridge.
        Returns a black frame if no image has been received yet.
        Logs a warning if the cached frame is more than 1 second old.
        """
        import numpy as np
        if self._last_camera_ts > 0:
            age = time.monotonic() - self._last_camera_ts
            if age > 1.0:
                logger.warning("[GO2] Camera frame is %.1fs old", age)
        if self._last_camera_frame is not None:
            return self._last_camera_frame.copy()
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_depth_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return latest depth frame as (H, W) float32 array in metres.

        Received from /camera/depth topic (32FC1) published by Go2VNavBridge.
        Returns a zero frame if no depth has been received yet.
        Logs a warning if the cached frame is more than 1 second old.
        """
        import numpy as np
        if self._last_depth_ts > 0:
            age = time.monotonic() - self._last_depth_ts
            if age > 1.0:
                logger.warning("[GO2] Depth frame is %.1fs old", age)
        if self._last_depth_frame is not None:
            return self._last_depth_frame.copy()
        return np.zeros((height, width), dtype=np.float32)

    def get_rgbd_frame(self, width: int = 320, height: int = 240) -> Any:
        """Return aligned (rgb, depth) tuple.

        Sim-to-real compatible: same interface as MuJoCoGo2.get_rgbd_frame().
        """
        return self.get_camera_frame(width, height), self.get_depth_frame(width, height)

    def get_camera_pose(self) -> tuple:
        """Return the D435 camera world pose (cam_xpos, cam_xmat).

        Preferred path (D36): the real d435_rgb pose published by the bridge on
        /camera/pose, read straight from the live MjData. This is the EXACT pose
        the depth pixels were rendered from, so back-projecting /camera/depth
        lands on the object. Returned as numpy (3,) + (9,) matching
        MuJoCoGo2.get_camera_pose() semantics.

        Fallback (no /camera/pose seen yet): compute the pose from robot
        odometry + the MJCF mount approximation below, so manipulation still
        degrades gracefully if the topic is absent. NOTE this approximation does
        NOT match the real d435_rgb camera and is back-project-inaccurate — it
        exists only so nothing breaks before the first /camera/pose arrives.

        Mount geometry sourced from MJCF go2_piper.xml::d435_camera body:
            pos="0.25 0 0.1" quat="0.999054 0 0.0434863 0"
        which is 0.25 m forward, 0.1 m up above base_link, pitched -5° down.

        Frame convention (REP-103):
            World X = forward, Y = left, Z = up.
            For a robot heading 0 (facing +X), body-right is world -Y
            so ``right = (sin(h), -cos(h), 0)``.
        xmat columns are [camera_right, camera_up, -camera_forward].
        ``up = cross(right, forward)`` keeps the basis right-handed and
        gives world +Z for a level, +X-facing camera. v2.4 G3 fix.
        """
        import numpy as np

        # Preferred: real pose from the bridge (single source of truth).
        if self._last_cam_xpos is not None and self._last_cam_xmat is not None:
            return (self._last_cam_xpos.copy(), self._last_cam_xmat.copy())

        pos = self._position
        heading = self._heading

        # MJCF-grounded mount offsets
        mount_fwd, mount_up = 0.25, 0.1
        pitch = math.radians(-5.0)

        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        # Camera world position
        cam_x = pos[0] + cos_h * mount_fwd
        cam_y = pos[1] + sin_h * mount_fwd
        cam_z = pos[2] + mount_up
        cam_xpos = np.array([cam_x, cam_y, cam_z])

        # Body frame: forward = (cos_h·cos_p, sin_h·cos_p, sin_p),
        #             right   = (sin_h, -cos_h, 0)   ← REP-103 right
        # up = cross(right, forward) → world +Z for a level, +X-facing camera.
        fwd = np.array([cos_h * cos_p, sin_h * cos_p, sin_p])
        right = np.array([sin_h, -cos_h, 0.0])
        up = np.cross(right, fwd)

        # MuJoCo xmat: columns = [right, up, -forward]
        cam_xmat = np.column_stack([right, up, -fwd]).flatten()

        return (cam_xpos, cam_xmat)

    def get_odometry(self) -> Any:
        """Return latest Odometry data as a types.Odometry dataclass."""
        from vector_os_nano.core.types import Odometry
        pos = self._position
        return Odometry(
            timestamp=time.time(),
            x=pos[0], y=pos[1], z=pos[2],
            qx=0.0, qy=0.0, qz=0.0, qw=1.0,
            vx=0.0, vy=0.0, vz=0.0, vyaw=0.0,
        )

    @property
    def name(self) -> str:
        return "go2_ros2_proxy"

    @property
    def supports_lidar(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Motion interface (matches MuJoCoGo2 public API)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, vyaw: float) -> None:
        """Publish a Twist command on /cmd_vel_nav (non-blocking)."""
        if self._node is None:
            return
        from geometry_msgs.msg import Twist

        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(vyaw)
        self._cmd_pub.publish(msg)

    def walk(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vyaw: float = 0.0,
        duration: float = 1.0,
    ) -> bool:
        """Walk at the given velocity for *duration* seconds, then stop.

        Publishes velocity at 4Hz to keep overriding the bridge path follower
        (which reclaims control 0.5s after last /cmd_vel_nav message).
        """
        deadline = time.time() + duration
        while time.time() < deadline:
            self.set_velocity(vx, vy, vyaw)
            time.sleep(0.25)  # 4Hz — keeps _teleop_until fresh
        self.set_velocity(0.0, 0.0, 0.0)
        return True

    def stop(self) -> None:
        """Emergency stop — immediately halt all motion."""
        self.set_velocity(0.0, 0.0, 0.0)

    def stand(self, duration: float = 1.0) -> bool:
        """Stop motion and hold position for *duration* seconds."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)
        return True

    def sit(self, duration: float = 1.0) -> bool:
        """Best-effort sit: stop motion (cannot command sit via ROS2 velocity)."""
        self.set_velocity(0.0, 0.0, 0.0)
        time.sleep(duration)
        return True

    # ------------------------------------------------------------------
    # Navigation via FAR planner / nav stack
    # ------------------------------------------------------------------

    def _planner_goal_for_body_target(
        self,
        x: float,
        y: float,
    ) -> tuple[float, float]:
        """Return the FAR sensor goal that corresponds to a body-centre goal.

        FAR tests convergence using sensor-frame odometry.  Without this small
        forward projection it can declare convergence while the robot body is
        still one mounting offset short of a doorway waypoint.  The proxy still
        decides arrival exclusively from the nominal body target.
        """

        target_x, target_y = float(x), float(y)
        if self._state_estimation_child_frame != "sensor":
            return target_x, target_y
        body_x, body_y, _body_z = self.get_position()
        dx, dy = target_x - body_x, target_y - body_y
        approach_heading = (
            math.atan2(dy, dx)
            if math.hypot(dx, dy) > 1e-9
            else self._heading
        )
        sensor_x, sensor_y, _sensor_z = body_to_sensor_position(
            (target_x, target_y, 0.0),
            approach_heading,
        )
        return sensor_x, sensor_y

    @_goal_scoped_navigation
    def navigate_to(
        self,
        x: float,
        y: float,
        timeout: float = 60.0,
        on_progress: Callable[[float, float], None] | None = None,
        goal_id: str | None = None,
        allow_door_fallback: bool = True,
        waypoint_kind: str | None = None,
        speed_limit_mps: float | None = None,
        allow_reverse: bool | None = None,
        arrival_tolerance: float | None = None,
        timeout_s: float | None = None,
        _navigation_generation: int | None = None,
    ) -> bool:
        """Navigate to (x, y) via FAR planner global route planning.

        Two-phase strategy:

        Phase 1 — FAR probe (3 s, /goal_point only):
          Publish /goal_point and require two pieces of current-generation
          evidence: FAR's native LINE_STRIP route marker whose terminal point
          matches this goal, followed by a FAR /way_point.  A fresh waypoint
          alone may be stale/TARE replay and cannot arm navigation.  If no
          associated response arrives within 3 s we perform an early_fallback:
          return False immediately so NavigateSkill can use its door-aware
          fallback when that policy is enabled.

        Phase 2 — Full navigation (FAR responded):
          Publish /goal_point only at 2 Hz. FAR routes through V-Graph and
          publishes intermediate /way_point to localPlanner. We do NOT publish
          /way_point directly — that would override FAR's door routing.

        FAR's 5 Hz /way_point naturally overrides TARE's 1 Hz exploration
        waypoints during navigation.

        Returns True when within the arrival tolerance, False on timeout or
        when FAR has no route.  Structured room navigation passes
        ``allow_door_fallback=False`` so a failed door segment stops instead of
        silently jumping into the proxy's legacy door-chain fallback.

        The waypoint safety arguments are installed atomically by the lifecycle
        wrapper and cleared after this call.  ``goal_id`` is consumed by that
        wrapper and retained here for native/tool compatibility.
        """
        if not isinstance(allow_door_fallback, bool):
            raise ValueError("allow_door_fallback must be a bool")
        if self._node is None:
            logger.warning("[NAV] navigate_to called but node not connected")
            self._publish_current_goal_state(
                "ERROR",
                goal_id=goal_id,
                target_xy=(x, y),
                reason="proxy_not_connected",
            )
            return False

        # Arm the bridge only if no concurrent stop/cancel invalidated this
        # blocking call while it waited for the segment-policy ACK.
        with self._navigation_state_lock:
            if (
                _navigation_generation != self._navigation_cancel_generation
                or self._active_navigation_goal_id != goal_id
            ):
                self.set_velocity(0.0, 0.0, 0.0)
                return False
            try:
                with open(nav_active_file(), "w") as fh:
                    fh.write("1")
            except OSError as exc:
                logger.warning("[NAV] Could not create nav flag: %s", exc)
                self.set_velocity(0.0, 0.0, 0.0)
                return False

        self._nav_goal = (float(x), float(y))
        logger.info("[NAV] navigate_to(%.2f, %.2f) timeout=%.0fs", x, y, timeout)
        self._publish_current_goal_state(
            "BUILDING_GRAPH",
            goal_id=goal_id,
            target_xy=(x, y),
            segment_kind=waypoint_kind,
            allow_door_fallback=allow_door_fallback,
        )

        start_time = time.time()
        _ARRIVAL_DIST: float = (
            float(arrival_tolerance)
            if arrival_tolerance is not None
            else _nav("arrival_radius", 0.8)
        )
        initial_pos = self.get_position()
        initial_dist = math.hypot(initial_pos[0] - x, initial_pos[1] - y)
        if initial_dist <= _ARRIVAL_DIST:
            logger.info(
                "[NAV] Already at (%.2f, %.2f) — distance=%.2fm",
                x,
                y,
                initial_dist,
            )
            self._nav_goal = None
            self.set_velocity(0.0, 0.0, 0.0)
            self._publish_current_goal_state(
                "ARRIVED",
                goal_id=goal_id,
                target_xy=(x, y),
                segment_kind=waypoint_kind,
            )
            return True
        _FAR_PROBE_S: float = _nav("far_probe_timeout", 3.0)  # shorter probe — door-chain is reliable backup
        _MIN_PROBE_S: float = 1.5    # minimum wait
        planner_x, planner_y = self._planner_goal_for_body_target(x, y)
        if (planner_x, planner_y) != (float(x), float(y)):
            sensor = self.get_sensor_position()
            logger.info(
                "[NAV] body_goal=(%.2f, %.2f) planner_sensor_goal=(%.2f, %.2f) "
                "body=(%.2f, %.2f) sensor=%s",
                x,
                y,
                planner_x,
                planner_y,
                self._position[0],
                self._position[1],
                (
                    f"({sensor[0]:.2f}, {sensor[1]:.2f})"
                    if sensor is not None
                    else "unavailable"
                ),
            )

        # PointStamped has no goal ID, so temporal freshness alone is not
        # sufficient.  Isolate this probe generation and require FAR's native
        # /viz_path_topic marker to end at the current planner goal before
        # accepting a later waypoint.
        far_probe_generation = self._begin_far_goal_probe(
            (planner_x, planner_y),
            match_tolerance_m=_nav("far_goal_match_tolerance", 1.0),
        )
        # Kept explicit for compatibility diagnostics: _begin_far_goal_probe()
        # reset self._last_waypoint_time = 0 for this generation.

        # Phase 1: probe FAR — send /goal_point, wait for goal-associated route.
        probe_deadline = start_time + _FAR_PROBE_S
        while time.time() < probe_deadline:
            if (
                not self._navigation_generation_is_active(
                    _navigation_generation, goal_id,
                )
                or not os.path.exists(nav_active_file())
            ):
                logger.info("[NAV] Cancelled by stop command")
                self._nav_goal = None
                self.set_velocity(0.0, 0.0, 0.0)
                self._publish_current_goal_state(
                    "CANCELLED",
                    goal_id=goal_id,
                    target_xy=(x, y),
                    reason="nav_gate_disabled",
                )
                return False
            self._mark_far_goal_published(far_probe_generation)
            self._publish_goal_point(planner_x, planner_y)
            time.sleep(0.5)
            elapsed = time.time() - start_time
            far_probe = self._far_probe_snapshot(far_probe_generation)
            if far_probe["far_response_associated"] and elapsed >= _MIN_PROBE_S:
                logger.info(
                    "[NAV] FAR responded with current-goal path + /way_point "
                    "after %.1fs",
                    elapsed,
                )
                break

        far_probe = self._far_probe_snapshot(far_probe_generation)
        far_available = bool(far_probe["far_response_associated"])
        if not far_available:
            stale_planner_response = int(far_probe["fresh_waypoint_count"]) > 0
            if not allow_door_fallback:
                # P2 fail-closed segment semantics: NO_PATH is terminal for this
                # topological waypoint.  Never skip it and head for a later door
                # or the final room centre.
                reason = (
                    "stale_planner_response"
                    if stale_planner_response
                    else "far_probe_timeout"
                )
                if stale_planner_response:
                    logger.warning(
                        "[NAV] Rejected unassociated FAR waypoint after %.0fs: "
                        "expected=%s observed=%s path_end=%s",
                        _FAR_PROBE_S,
                        far_probe["planner_goal_xy"],
                        far_probe["observed_waypoint_xy"],
                        far_probe["far_path_endpoint_xy"],
                    )
                else:
                    logger.warning(
                        "[NAV] FAR returned NO_PATH for required segment after %.0fs",
                        _FAR_PROBE_S,
                    )
                return self._fail_closed_navigation_segment(
                    goal_id=goal_id,
                    target_xy=(x, y),
                    reason=reason,
                    segment_kind=waypoint_kind,
                    **far_probe,
                )
            # FAR has no V-Graph — use door-chain fallback via localPlanner.
            # Publish door waypoints to /way_point one by one. localPlanner
            # handles obstacle avoidance for each segment.
            logger.warning(
                "[NAV] No FAR response after %.0fs — using door-chain fallback",
                _FAR_PROBE_S,
            )
            remaining = timeout - (time.time() - start_time)
            return self._navigate_via_doors(
                x,
                y,
                remaining,
                arrival_tolerance=_ARRIVAL_DIST,
            )

        self._publish_current_goal_state(
            "VALID_PATH",
            goal_id=goal_id,
            target_xy=(x, y),
            segment_kind=waypoint_kind,
            **far_probe,
        )

        # Phase 2: full navigation loop (FAR is routing)
        # ONLY publish /goal_point — let FAR handle /way_point routing.
        # FAR routes through doors via V-Graph and publishes intermediate
        # /way_point at 5Hz.
        # Phase 2: full navigation loop (FAR is routing)
        deadline = start_time + timeout
        _last_diag = 0.0
        progress_watchdog = _NavigationProgressWatchdog(
            progress_threshold_m=_nav("stall_threshold", 0.3),
            timeout_s=_nav("stall_timeout", 30.0),
            heading_progress_threshold_rad=math.radians(
                _nav("stall_heading_progress_deg", 3.0)
            ),
            max_observation_gap_s=_nav("stall_max_observation_gap", 1.0),
        )
        _last_progress = time.time()
        while time.time() < deadline:
            # --- Cancel check: stop command removes nav flag ---
            if (
                not self._navigation_generation_is_active(
                    _navigation_generation, goal_id,
                )
                or not os.path.exists(nav_active_file())
            ):
                logger.info("[NAV] Cancelled by stop command")
                self._nav_goal = None
                self.set_velocity(0.0, 0.0, 0.0)
                self._publish_current_goal_state(
                    "CANCELLED",
                    goal_id=goal_id,
                    target_xy=(x, y),
                    reason="nav_gate_disabled",
                )
                return False

            # --- Abort check (cognitive layer) ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    logger.info("[NAV] Abort requested — cancelling navigate_to")
                    if not allow_door_fallback:
                        return self._fail_closed_navigation_segment(
                            goal_id=goal_id,
                            target_xy=(x, y),
                            reason="abort_requested",
                            segment_kind=waypoint_kind,
                            state="CANCELLED",
                        )
                    self._nav_goal = None
                    self.set_velocity(0.0, 0.0, 0.0)
                    self._publish_current_goal_state(
                        "CANCELLED",
                        goal_id=goal_id,
                        target_xy=(x, y),
                        reason="abort_requested",
                    )
                    return False
            except ImportError:
                pass

            # --- Stall flag written by bridge _stuck_detector ---
            stalled_file = nav_stalled_file()
            if os.path.exists(stalled_file):
                os.remove(stalled_file)
                logger.warning("[NAV] Stall flag detected — aborting navigate_to")
                if not allow_door_fallback:
                    return self._fail_closed_navigation_segment(
                        goal_id=goal_id,
                        target_xy=(x, y),
                        reason="bridge_stall",
                        segment_kind=waypoint_kind,
                        state="ERROR",
                    )
                self._nav_goal = None
                self.set_velocity(0.0, 0.0, 0.0)
                self._publish_current_goal_state(
                    "NO_PATH",
                    goal_id=goal_id,
                    target_xy=(x, y),
                    reason="bridge_stall",
                    segment_kind=waypoint_kind,
                )
                return False

            self._publish_goal_point(planner_x, planner_y)
            time.sleep(0.5)

            pos = self.get_position()
            dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            target_heading = math.atan2(y - pos[1], x - pos[0])
            heading = self.get_heading()
            heading_error = abs(
                math.atan2(
                    math.sin(target_heading - heading),
                    math.cos(target_heading - heading),
                )
            )

            # --- Progress callback every 2s ---
            now = time.time()
            if on_progress is not None and now - _last_progress >= 2.0:
                _last_progress = now
                on_progress(dist, now - start_time)

            if dist <= _ARRIVAL_DIST:
                logger.info(
                    "[NAV] Arrived at (%.2f, %.2f) — distance=%.2fm", x, y, dist
                )
                self._nav_goal = None
                # Clear the bridge's old local path and hold still before the
                # decorator releases this segment's speed/reverse policy.
                self.set_velocity(0.0, 0.0, 0.0)
                self._publish_current_goal_state(
                    "ARRIVED",
                    goal_id=goal_id,
                    target_xy=(x, y),
                    segment_kind=waypoint_kind,
                )
                return True

            # --- Stall detection: linear or angular progress over a real window ---
            if progress_watchdog.stalled(
                dist,
                now,
                heading_error_rad=heading_error,
                position_xy=pos,
                heading_rad=heading,
            ):
                if not allow_door_fallback:
                    logger.warning(
                        "[NAV] FAR segment stalled %.0fs at dist=%.1fm — NO_PATH",
                        progress_watchdog.timeout_s,
                        dist,
                    )
                    return self._fail_closed_navigation_segment(
                        goal_id=goal_id,
                        target_xy=(x, y),
                        reason="segment_stalled",
                        segment_kind=waypoint_kind,
                        state="ERROR",
                    )
                logger.warning(
                    "[NAV] Stalled %.0fs (dist=%.1fm not decreasing) — switching to door-chain",
                    progress_watchdog.timeout_s, dist,
                )
                remaining = timeout - (time.time() - start_time)
                return self._navigate_via_doors(
                    x,
                    y,
                    remaining,
                    arrival_tolerance=_ARRIVAL_DIST,
                )

            elapsed = time.time() - start_time
            if elapsed - _last_diag >= 5.0:
                _last_diag = elapsed
                wp_age = time.time() - self._last_waypoint_time if self._last_waypoint_time > 0 else -1
                wp_pos = getattr(self, "_last_waypoint_pos", None)
                wp_str = f"wp=({wp_pos[0]:.1f},{wp_pos[1]:.1f})" if wp_pos else "wp=?"
                logger.info(
                    "[NAV] t=%.0fs pos=(%.1f,%.1f) goal=(%.1f,%.1f) "
                    "dist=%.1fm %s age=%.1fs",
                    elapsed, pos[0], pos[1], x, y, dist, wp_str, wp_age,
                )

        logger.warning(
            "[NAV] navigate_to(%.2f, %.2f) far_timeout after %.0fs", x, y, timeout
        )
        if not allow_door_fallback:
            return self._fail_closed_navigation_segment(
                goal_id=goal_id,
                target_xy=(x, y),
                reason="segment_timeout",
                segment_kind=waypoint_kind,
                state="ERROR",
            )
        self._nav_goal = None
        self.set_velocity(0.0, 0.0, 0.0)
        self._publish_current_goal_state(
            "ERROR",
            goal_id=goal_id,
            target_xy=(x, y),
            reason="far_timeout",
            segment_kind=waypoint_kind,
        )
        return False

    def navigate_far_segment(
        self,
        x: float,
        y: float,
        *,
        timeout_s: float = 60.0,
        on_progress: Callable[[float, float], None] | None = None,
        goal_id: str | None = None,
        waypoint_kind: str = "room_goal",
        speed_limit_mps: float | None = None,
        allow_reverse: bool = True,
        arrival_tolerance: float | None = None,
    ) -> bool:
        """Explicit fail-closed FAR entry point for one required door waypoint."""
        return self.navigate_to(
            x,
            y,
            timeout_s=timeout_s,
            on_progress=on_progress,
            goal_id=goal_id,
            allow_door_fallback=False,
            waypoint_kind=waypoint_kind,
            speed_limit_mps=speed_limit_mps,
            allow_reverse=allow_reverse,
            arrival_tolerance=arrival_tolerance,
        )

    def _publish_waypoint(self, x: float, y: float) -> None:
        """Publish PointStamped to /way_point (localPlanner goal topic)."""
        if self._node is None:
            return
        self._announce_active_navigation_goal()
        try:
            from geometry_msgs.msg import PointStamped

            msg = PointStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = 0.0
            self._waypoint_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish waypoint: %s", exc)

    @_goal_scoped_navigation
    def go_to_waypoint(
        self,
        x: float,
        y: float,
        timeout: float = 30.0,
        on_progress: Callable[[float, float], None] | None = None,
        goal_id: str | None = None,
        arrival_tolerance: float | None = None,
        _navigation_generation: int | None = None,
    ) -> bool:
        """Navigate to (x, y) by publishing /way_point directly to localPlanner.

        Unlike navigate_to(), this does NOT probe FAR and does NOT fall back
        to door-chain.  It simply publishes /way_point at 2 Hz and waits for
        localPlanner + path follower to reach the goal.

        Used by dead_reckoning (NavigateSkill) to avoid recursive cascades
        where navigate_to → door-chain → navigate_to → door-chain → ...

        Returns True when within arrival radius, False on timeout or stall.
        """
        if self._node is None:
            return False
        if not self._navigation_generation_is_active(
            _navigation_generation, goal_id
        ):
            self.set_velocity(0.0, 0.0, 0.0)
            return False

        _ARRIVAL = (
            float(arrival_tolerance)
            if arrival_tolerance is not None
            else _nav("arrival_radius", 0.8)
        )
        progress_watchdog = _NavigationProgressWatchdog(
            progress_threshold_m=_nav("stall_threshold", 0.3),
            timeout_s=_nav("stall_timeout", 30.0),
            heading_progress_threshold_rad=math.radians(
                _nav("stall_heading_progress_deg", 3.0)
            ),
            max_observation_gap_s=_nav("stall_max_observation_gap", 1.0),
        )

        start = time.time()
        deadline = start + timeout
        last_progress_time = start
        initial_pos = self.get_position()
        if math.hypot(initial_pos[0] - x, initial_pos[1] - y) <= _ARRIVAL:
            logger.info("[NAV] go_to_waypoint already within arrival tolerance")
            return True

        logger.info("[NAV] go_to_waypoint(%.1f, %.1f) timeout=%.0fs", x, y, timeout)

        while time.time() < deadline:
            # Cancel check
            if (
                not self._navigation_generation_is_active(
                    _navigation_generation, goal_id
                )
                or not os.path.exists(nav_active_file())
            ):
                self.set_velocity(0.0, 0.0, 0.0)
                return False

            # Abort check
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    return False
            except ImportError:
                pass

            self._publish_waypoint(x, y)
            time.sleep(0.5)

            pos = self.get_position()
            dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            target_heading = math.atan2(y - pos[1], x - pos[0])
            heading = self.get_heading()
            heading_error = abs(
                math.atan2(
                    math.sin(target_heading - heading),
                    math.cos(target_heading - heading),
                )
            )

            # Progress callback
            now = time.time()
            if on_progress is not None and now - last_progress_time >= 2.0:
                last_progress_time = now
                on_progress(dist, now - start)

            # Arrival check
            if dist <= _ARRIVAL:
                logger.info("[NAV] go_to_waypoint arrived (dist=%.1fm)", dist)
                return True

            # Stall detection — no recursive fallback, just return False.
            if progress_watchdog.stalled(
                dist,
                now,
                heading_error_rad=heading_error,
                position_xy=pos,
                heading_rad=heading,
            ):
                logger.warning(
                    "[NAV] go_to_waypoint stalled %.0fs at dist=%.1fm",
                    progress_watchdog.timeout_s, dist,
                )
                return False

        logger.warning("[NAV] go_to_waypoint timeout after %.0fs", timeout)
        return False

    def _navigate_via_doors(
        self,
        x: float,
        y: float,
        timeout: float,
        *,
        arrival_tolerance: float | None = None,
    ) -> bool:
        """Navigate to (x,y) by publishing door waypoints to /way_point.

        Uses SceneGraph door chain instead of hardcoded room map.
        localPlanner provides obstacle avoidance for each segment.
        """
        sg = self._scene_graph
        if sg is None or not hasattr(sg, "get_door_chain"):
            logger.warning("[NAV] No SceneGraph available for door-chain navigation")
            return False

        pos = self.get_position()
        src_room = sg.nearest_room(float(pos[0]), float(pos[1]))
        dst_room = sg.nearest_room(float(x), float(y))

        if src_room is None or dst_room is None:
            logger.warning(
                "[NAV] Cannot determine rooms for door-chain: src=%s dst=%s",
                src_room, dst_room,
            )
            return False

        # Get waypoint chain from SceneGraph BFS
        chain = sg.get_door_chain(src_room, dst_room)
        if not chain:
            logger.warning("[NAV] No door chain found: %s -> %s", src_room, dst_room)
            return False

        # Replace final waypoint with exact goal coordinates
        # (door chain ends at dst room center, but actual goal may differ)
        chain[-1] = (x, y, chain[-1][2])

        logger.info(
            "[NAV] Door-chain: %s -> %s (%d waypoints)",
            src_room, dst_room, len(chain),
        )

        start = time.time()
        deadline = start + timeout

        for i, (wx, wy, label) in enumerate(chain):
            remaining = deadline - time.time()
            if remaining <= 0:
                logger.warning("[NAV] Door-chain global timeout")
                return False
            n_remaining = len(chain) - i
            per_wp = max(remaining / n_remaining, 5.0)

            logger.info("[NAV] Door-chain -> %s (%.1f, %.1f) budget=%.0fs",
                        label, wx, wy, per_wp)

            final_waypoint = i == len(chain) - 1
            ok = self.go_to_waypoint(
                wx,
                wy,
                timeout=per_wp,
                arrival_tolerance=(
                    arrival_tolerance if final_waypoint else None
                ),
            )
            if not ok:
                logger.warning("[NAV] Door-chain: failed to reach %s", label)
                return False

        # Final arrival check
        pos = self.get_position()
        final_dist = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
        final_tolerance = (
            float(arrival_tolerance)
            if arrival_tolerance is not None
            else _nav("arrival_radius", 0.8)
        )
        arrived = final_dist <= final_tolerance
        logger.info(
            "[NAV] Door-chain %s -- final dist=%.1fm",
            "arrived" if arrived else "failed", final_dist,
        )
        return arrived

    def _publish_goal_point(self, x: float, y: float) -> None:
        """Publish PointStamped to /goal_point (FAR planner input for routing)."""
        if self._node is None:
            return
        self._announce_active_navigation_goal()
        try:
            from geometry_msgs.msg import PointStamped

            msg = PointStamped()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            msg.point.x = float(x)
            msg.point.y = float(y)
            msg.point.z = 0.0
            self._goal_pub.publish(msg)
        except Exception as exc:
            logger.warning("[NAV] Failed to publish goal point: %s", exc)

    def cancel_navigation(self) -> None:
        """Cancel active navigation and disarm every in-flight nav start."""
        active_goal = self._active_navigation_goal_id
        self._invalidate_navigation_gate()
        self.set_velocity(0.0, 0.0, 0.0)
        self._nav_goal = None
        self._publish_empty_door_path()
        if active_goal is not None:
            self.finalize_navigation_goal(active_goal, status="cancelled")
        logger.info("[NAV] Navigation cancelled and nav gate removed")

    def stop_navigation(self) -> None:
        """Fully stop navigation: remove nav flag, zero velocity, clear goal.

        Use this when navigation is complete AND no further nav-stack
        motion is expected (e.g., exploration resumes via its own flag
        logic).
        """
        active_goal = self._active_navigation_goal_id
        self._invalidate_navigation_gate()
        self.set_velocity(0.0, 0.0, 0.0)
        self._nav_goal = None
        self._publish_empty_door_path()
        if active_goal is not None:
            self.finalize_navigation_goal(active_goal, status="cancelled")
        logger.info("[NAV] Navigation stopped, nav flag removed")

    def _scene_graph_hash(self) -> int:
        """Compute a lightweight hash of the current scene graph state.

        Combines rooms count, viewpoints count, objects count, and robot
        position rounded to 0.5 m grid so minor drift does not trigger
        a re-publish.  Returns 0 when no scene graph is available.
        """
        sg = self._scene_graph
        rooms_count = 0
        vp_count = 0
        obj_count = 0
        if sg is not None:
            try:
                rooms = sg.get_all_rooms()
                rooms_count = len(list(rooms))
                for room in sg.get_all_rooms():
                    vp_count += len(sg.get_viewpoints_in_room(room.room_id))
                    obj_count += len(sg.find_objects_in_room(room.room_id))
            except Exception:
                pass
        pos = self._position
        rx = round(pos[0] / 0.5)
        ry = round(pos[1] / 0.5)
        nav_goal = getattr(self, "_nav_goal", None)
        goal_key = (
            (round(nav_goal[0], 3), round(nav_goal[1], 3))
            if nav_goal is not None
            else None
        )
        return hash((rooms_count, vp_count, obj_count, rx, ry, goal_key))

    def _publish_markers(self) -> None:
        """Publish scene graph visualization as MarkerArray at 3 Hz.

        Records current position into trajectory history on every call.
        Caps trajectory at 200 entries to avoid unbounded memory growth.

        Only rebuilds and publishes the MarkerArray when the scene graph
        state hash changes, or every 10 seconds as a keep-alive fallback.
        This prevents unnecessary RViz re-renders that cause flickering.
        """
        if self._marker_pub is None:
            return
        try:
            from vector_os_nano.ros2.nodes.scene_graph_viz import (
                build_scene_graph_markers,
                _TRAJECTORY_MAX_POINTS,
            )
            pos = self._position
            # Always record trajectory (position history should be continuous)
            self._trajectory.append((pos[0], pos[1]))
            if len(self._trajectory) > _TRAJECTORY_MAX_POINTS:
                del self._trajectory[: len(self._trajectory) - _TRAJECTORY_MAX_POINTS]

            # Decide whether to publish: state changed OR 10 s fallback
            now = time.time()
            current_hash = self._scene_graph_hash()
            elapsed = now - self._last_marker_publish_time
            state_changed = current_hash != self._last_marker_hash
            fallback_due = elapsed >= 10.0

            if not (state_changed or fallback_due):
                return

            ma = build_scene_graph_markers(
                scene_graph=self._scene_graph,
                robot_x=pos[0],
                robot_y=pos[1],
                robot_heading=self._heading,
                nav_goal=self._nav_goal,
                trajectory=self._trajectory,
            )
            if ma is not None:
                self._marker_pub.publish(ma)
                self._last_marker_hash = current_hash
                self._last_marker_publish_time = now
        except Exception:
            pass
