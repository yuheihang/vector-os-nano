# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""NavigateSkill -- hardware-agnostic room-to-room navigation.

Supports two navigation modes:
1. NavStackClient (real navigation): when context.services.get("nav") is available
   and nav.is_available is True, publishes a waypoint goal and waits for
   goal_reached feedback from the navigation stack.
2. Dead-reckoning fallback: when no nav stack is present, uses SceneGraph door
   chain to navigate between named rooms via turn+walk sequences.

Room positions and door coordinates come entirely from the SceneGraph
(populated during exploration).  No hardcoded coordinates are used.

This module has no ROS2 imports at the top level.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from collections.abc import Mapping
from typing import Any

from vector_os_nano.core.skill import SkillContext, skill
from vector_os_nano.core.types import SkillResult
from vector_os_nano.navigation.room_resolver import (
    ROOM_ALIASES,
    RoomPositionUnknown,
    RoomResolver,
    UnknownRoom,
    normalize_room_query,
)
from vector_os_nano.navigation.runtime_files import nav_active_file
from vector_os_nano.navigation.world_mode import WorldMode
from vector_os_nano.vcli.worlds.go2_sim_oracle import AT_POSITION_TOL_M

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nav config loader (lazy, module-level cache)
# ---------------------------------------------------------------------------

_NAV_CFG: dict | None = None


def _load_nav_config() -> dict:
    """Load nav.yaml with defaults. Searches relative paths then falls back."""
    import os
    import yaml

    global _NAV_CFG
    if _NAV_CFG is not None:
        return _NAV_CFG

    _search = [
        "config/nav.yaml",
        os.path.join(os.path.dirname(__file__), "..", "..", "config", "nav.yaml"),
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


# ---------------------------------------------------------------------------
# Aliases -> canonical room name (Chinese + English + shortcuts)
# ---------------------------------------------------------------------------

# Backward-compatible export.  The authoritative language map is shared with
# native navigation and verify predicates; it intentionally contains no
# coordinates.
_ROOM_ALIASES: dict[str, str] = ROOM_ALIASES

_WALK_SPEED: float = 0.6     # m/s
_TURN_SPEED: float = 0.8     # rad/s
_ARRIVAL_RADIUS: float = 0.5  # meters -- close enough to target (dead-reckoning helper)
_DOORCHAIN_ARRIVAL_RADIUS: float = 0.8  # meters -- arrival threshold for nav stack door-chain
# Loaded from config/nav.yaml at first use; fallback keeps original behaviour
_DOORCHAIN_WAYPOINT_TIMEOUT: float = _nav("waypoint_timeout", 30.0)

_MIN_VISIT_COUNT: int = 1  # trust SceneGraph position after first visit

# Coordinate-goal navigation (R38). Slow local-planner motion in the textured
# house can take several minutes across rooms. One bounded transport call is
# preferable to many model-level retries. Its arrival radius is deliberately
# single-sourced from the deterministic verifier.
_COORD_NAV_TIMEOUT_S: float = _nav("coordinate_timeout", 240.0)

# Executable layout-v2 routes are deliberately segmented.  A single room-centre
# goal can cut through a wall, while an unbounded per-segment timeout can leave a
# failed doorway blocking the whole native turn indefinitely.
_SAFE_ROUTE_TOTAL_TIMEOUT_S: float = _nav("room_route_timeout", 360.0)
_SAFE_ROUTE_MIN_SEGMENT_TIMEOUT_S: float = _nav(
    "room_route_min_segment_timeout", 35.0,
)
_SAFE_ROUTE_TIMEOUT_PER_DOOR_S: float = max(
    0.0, _nav("room_route_timeout_per_door", 300.0),
)
_SAFE_ROUTE_TIMEOUT_PER_METER_S: float = max(
    0.0, _nav("room_route_timeout_per_meter", 55.0),
)
_SAFE_ROUTE_MAX_TIMEOUT_S: float = max(
    _SAFE_ROUTE_TOTAL_TIMEOUT_S,
    _nav("room_route_max_timeout", 1200.0),
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_room(name: str, sg: Any = None) -> str | None:
    """Backward-compatible thin wrapper over :class:`RoomResolver`.

    With a SceneGraph, resolution is constrained to the live room set.  Without
    one, this compatibility helper resolves language aliases only; navigation
    itself always supplies a SceneGraph and never gets coordinates here.
    """

    if not name:
        return None
    if sg is not None:
        try:
            return RoomResolver(sg).canonicalize(name)
        except (UnknownRoom, ValueError):
            return None
    key = normalize_room_query(name)
    alias_result = _ROOM_ALIASES.get(key)
    canonical = alias_result or key.replace(" ", "_")
    if canonical in _ROOM_ALIASES.values():
        return canonical
    return _fuzzy_room_match(key, sorted(set(_ROOM_ALIASES.values())))


def _fuzzy_room_match(query: str, room_ids: list[str]) -> str | None:
    """Return a unique whole-token match; ambiguous/substring guesses fail."""

    key = normalize_room_query(query)
    stop_words = {"room", "the", "a", "to", "go", "去", "到"}
    query_words = set(key.split()) - stop_words
    if not query_words:
        return None
    matches: set[str] = set()
    for rid in room_ids:
        rid_words = set(str(rid).replace("_", " ").split()) - stop_words
        if query_words <= rid_words:
            matches.add(str(rid))
            continue
        for alias, target in _ROOM_ALIASES.items():
            if target == rid and query_words <= (set(alias.split()) - stop_words):
                matches.add(str(rid))
                break
    return next(iter(matches)) if len(matches) == 1 else None


def _angle_between(x1: float, y1: float, x2: float, y2: float) -> float:
    """Bearing angle from (x1,y1) to (x2,y2) in radians."""
    return math.atan2(y2 - y1, x2 - x1)


def _distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _bounded_safe_route_budget(
    *,
    base_timeout_s: float,
    door_count: int,
    total_polyline_length_m: float,
) -> dict[str, Any]:
    """Return the bounded wall-clock budget for one executable room route.

    A fixed 360 s deadline is ample for a one-door route, but it starves a
    three-door/ten-segment route: the 35 s reservation for each future segment
    consumes 350 s before distance weighting can help the current leg.  Scale
    the deadline by whichever route-complexity estimate is larger (door
    transitions or polyline length), while retaining both the configured
    baseline and one absolute upper bound.
    """

    base = float(base_timeout_s)
    if not math.isfinite(base) or base <= 0.0:
        base = _SAFE_ROUTE_TOTAL_TIMEOUT_S
    doors = max(0, int(door_count))
    length = float(total_polyline_length_m)
    if not math.isfinite(length) or length < 0.0:
        length = 0.0

    door_budget = doors * _SAFE_ROUTE_TIMEOUT_PER_DOOR_S
    distance_budget = length * _SAFE_ROUTE_TIMEOUT_PER_METER_S
    effective = min(
        _SAFE_ROUTE_MAX_TIMEOUT_S,
        max(base, door_budget, distance_budget),
    )
    return {
        "policy": "bounded_route_complexity_v1",
        "base_timeout_s": round(base, 3),
        "effective_timeout_s": round(effective, 3),
        "max_timeout_s": round(_SAFE_ROUTE_MAX_TIMEOUT_S, 3),
        "door_count": doors,
        "timeout_per_door_s": round(_SAFE_ROUTE_TIMEOUT_PER_DOOR_S, 3),
        "door_budget_s": round(door_budget, 3),
        "total_polyline_length_m": round(length, 3),
        "timeout_per_meter_s": round(_SAFE_ROUTE_TIMEOUT_PER_METER_S, 3),
        "distance_budget_s": round(distance_budget, 3),
    }


def _new_goal_id(params: dict[str, Any]) -> str:
    """Return a caller-supplied internal goal ID or create a collision-safe one."""

    supplied = params.get("_goal_id") or params.get("goal_id")
    if supplied is not None and str(supplied).strip():
        return str(supplied).strip()
    return f"nav-{uuid.uuid4().hex}"


def _safe_position(base: Any) -> tuple[float, float] | None:
    getter = getattr(base, "get_position", None)
    if not callable(getter):
        return None
    try:
        pos = getter()
        x, y = float(pos[0]), float(pos[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _segment_failure(
    base: Any,
    goal_id: str,
    *,
    index: int,
    label: str,
) -> tuple[str, str]:
    """Translate the proxy's structured terminal reason without guessing."""

    state: Mapping[str, Any] = {}
    getter = getattr(base, "get_navigation_goal_state", None)
    if callable(getter):
        try:
            snapshot = getter(goal_id)
            if isinstance(snapshot, Mapping):
                state = snapshot
        except TypeError:
            try:
                snapshot = getter()
                if isinstance(snapshot, Mapping):
                    state = snapshot
            except Exception:  # noqa: BLE001
                pass
        except Exception:  # noqa: BLE001
            pass
    reason = str(state.get("reason", "")).strip()
    if reason == "segment_timeout":
        return (
            "segment_timeout",
            f"FAR path execution timed out for segment {index} ({label}).",
        )
    if reason in {"segment_stalled", "bridge_stall"}:
        return (
            "segment_stalled",
            f"Navigation made no progress on segment {index} ({label}).",
        )
    if reason == "stale_planner_response":
        expected = state.get("planner_goal_xy")
        observed = state.get("observed_waypoint_xy")
        endpoint = state.get("far_path_endpoint_xy")
        return (
            "stale_planner_response",
            (
                f"Rejected a stale FAR response for segment {index} ({label}); "
                f"expected goal {expected}, observed waypoint {observed}, "
                f"path endpoint {endpoint}."
            ),
        )
    if reason in {"abort_requested", "nav_gate_disabled"}:
        return (
            "aborted",
            f"Navigation was cancelled on segment {index} ({label}).",
        )
    # A missing structured state means a legacy/fake base; retain the
    # fail-closed historical interpretation for compatibility.
    return (
        "segment_no_path",
        f"No FAR path for segment {index} ({label}); later segments were not issued.",
    )


def _begin_navigation_goal(
    base: Any, goal_id: str, target: tuple[float, float],
) -> None:
    """Open a bridge-visible goal scope when the base supports P1 telemetry."""

    begin = getattr(base, "begin_navigation_goal", None)
    if not callable(begin):
        return
    try:
        begin(goal_id, target_xy=target)
    except TypeError:
        try:
            begin(goal_id, target)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] begin goal telemetry failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[NAV] begin goal telemetry failed: %s", exc)


def _finalize_navigation_goal(base: Any, goal_id: str, status: str) -> None:
    finalize = getattr(base, "finalize_navigation_goal", None)
    if not callable(finalize):
        return
    try:
        finalize(goal_id, status=status)
    except TypeError:
        try:
            finalize(goal_id, status)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] finalize goal telemetry failed: %s", exc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("[NAV] finalize goal telemetry failed: %s", exc)


def _navigation_stats(
    base: Any,
    goal_id: str,
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> dict[str, Any]:
    """Read goal-scoped bridge evidence, with a conservative local fallback."""

    for method_name in ("get_navigation_telemetry", "get_navigation_goal_stats"):
        getter = getattr(base, method_name, None)
        if not callable(getter):
            continue
        try:
            raw = getter(goal_id)
        except TypeError:
            try:
                raw = getter()
            except Exception:  # noqa: BLE001
                continue
        except Exception:  # noqa: BLE001
            continue
        if isinstance(raw, dict):
            stats = dict(raw)
            stats.setdefault("goal_id", goal_id)
            stats.setdefault("nonzero_cmd_count", 0)
            stats.setdefault("cmd_motion_count", stats["nonzero_cmd_count"])
            stats.setdefault(
                "actual_velocity_observed",
                bool(stats.get("nonzero_cmd_count") or stats.get("cmd_motion_count")),
            )
            if "moved_distance_m" not in stats and start is not None and end is not None:
                stats["moved_distance_m"] = round(_distance(*start, *end), 3)
            return stats

    moved = _distance(*start, *end) if start is not None and end is not None else 0.0
    # Position displacement is useful context, but is deliberately not promoted to
    # command evidence.  P1 actor causation trusts actual bridge velocity counters.
    return {
        "goal_id": goal_id,
        "nonzero_cmd_count": 0,
        "cmd_motion_count": 0,
        "moved_distance_m": round(moved, 3),
        "actual_velocity_observed": False,
    }


def _stop_navigation(base: Any) -> None:
    """Disarm path following and command a stationary hold, best-effort."""

    try:
        import os

        active_file = nav_active_file()
        if os.path.exists(active_file):
            os.remove(active_file)
    except Exception:
        pass

    stop_navigation = getattr(base, "stop_navigation", None)
    if callable(stop_navigation):
        try:
            stop_navigation()
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] stop_navigation failed: %s", exc)

    stop = getattr(base, "stop", None)
    if callable(stop):
        try:
            stop()
            return
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] stop failed: %s", exc)

    set_velocity = getattr(base, "set_velocity", None)
    if callable(set_velocity):
        try:
            set_velocity(0.0, 0.0, 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[NAV] zero velocity failed: %s", exc)


def _with_navigation_contract(
    result: SkillResult,
    *,
    goal_id: str,
    goal_type: str,
    target: tuple[float, float] | None,
    base: Any,
    start: tuple[float, float] | None,
    requested_room: str | None = None,
    canonical_room: str | None = None,
    source: str | None = None,
    verification_mode: str = "at_position",
) -> SkillResult:
    """Add the stable P1 result envelope while retaining compatibility fields."""

    end = _safe_position(base)
    data = dict(result.result_data or {})
    mode = str(data.get("mode", ""))
    planner = {
        "proxy_nav_stack": "far",
        "proxy_coord": "far",
        "nav_stack": "nav_stack",
        "dead_reckoning": "door_chain",
        "safe_door_route": "far_segmented",
    }.get(mode, mode or "unavailable")
    data.update(
        {
            "goal_id": goal_id,
            "goal_type": goal_type,
            "target_xy": list(target) if target is not None else None,
            "planner": planner,
            "arrived": bool(result.success),
            "verification_mode": verification_mode,
            "navigation_stats": _navigation_stats(base, goal_id, start, end),
        }
    )
    if end is not None:
        data["position_xy"] = [round(end[0], 3), round(end[1], 3)]
    if requested_room is not None:
        data["requested_room"] = requested_room
    if canonical_room is not None:
        data["canonical_room"] = canonical_room
        data.setdefault("room", canonical_room)
    if source is not None:
        data["source"] = source
    return SkillResult(
        success=result.success,
        result_data=data,
        error_message=result.error_message,
        diagnosis_code=result.diagnosis_code or "",
    )


def _normalize_angle(a: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _detect_current_room(x: float, y: float, sg: Any = None) -> str:
    """Guess which room the robot is in based on SceneGraph nearest_room.

    If sg is provided and has rooms, delegates to sg.nearest_room(x, y).
    Returns "unknown" if SceneGraph is absent or empty.
    """
    if sg is not None and hasattr(sg, "nearest_room"):
        room = sg.nearest_room(x, y)
        if room is not None:
            return room
    return "unknown"


def _get_room_center_from_memory(
    memory: Any, room_key: str,
) -> tuple[float, float] | None:
    """Look up explored room center from spatial memory (SceneGraph).

    Only trusts positions that have visit_count >= _MIN_VISIT_COUNT
    (not just a doorway drive-by).

    Uses get_room() if available (SceneGraph API), otherwise falls back
    to the older get_location() API (legacy SpatialMemory).

    Returns None if room not in memory or position not trustworthy.
    """
    # SceneGraph direct API — preferred, enforces visit_count threshold
    if hasattr(memory, "get_room"):
        room_node = memory.get_room(room_key)
        if room_node is not None:
            if (room_node.center_x != 0.0 or room_node.center_y != 0.0) and room_node.visit_count >= _MIN_VISIT_COUNT:
                return (room_node.center_x, room_node.center_y)
        # get_room is present but room not found or insufficient visits — do not
        # fall through to get_location(), which would bypass the visit threshold.
        return None

    # Backward-compatible get_location() API (legacy SpatialMemory only)
    if hasattr(memory, "get_location"):
        loc = memory.get_location(room_key)
        if loc is not None:
            x, y = getattr(loc, "x", 0.0), getattr(loc, "y", 0.0)
            if x != 0.0 or y != 0.0:
                return (x, y)

    return None


def _navigate_to_waypoint(
    base: Any,
    target_x: float,
    target_y: float,
    label: str,
) -> bool:
    """Turn toward waypoint and walk to it via dead-reckoning.

    Returns True if arrived upright, False if robot fell.
    """
    pos = base.get_position()
    cx, cy = pos[0], pos[1]
    heading = base.get_heading()

    dist = _distance(cx, cy, target_x, target_y)
    if dist < _ARRIVAL_RADIUS:
        logger.info("[NAV] Already at %s (%.1fm away)", label, dist)
        return True

    # Calculate required heading change
    target_angle = _angle_between(cx, cy, target_x, target_y)
    turn_needed = _normalize_angle(target_angle - heading)

    # Turn in place if heading delta > ~5.7 degrees
    if abs(turn_needed) > 0.1:
        vyaw = _TURN_SPEED if turn_needed > 0 else -_TURN_SPEED
        turn_dur = abs(turn_needed) / _TURN_SPEED
        logger.info("[NAV] Turn %.0f deg toward %s", math.degrees(turn_needed), label)
        base.walk(0.0, 0.0, vyaw, turn_dur)

    # Walk forward to waypoint
    walk_dur = dist / _WALK_SPEED
    logger.info("[NAV] Walk %.1fm to %s", dist, label)
    base.walk(_WALK_SPEED, 0.0, 0.0, walk_dur)

    # Upright check (z < 0.12 means robot has fallen)
    pos = base.get_position()
    if pos[2] < 0.12:
        logger.error("[NAV] Robot fell during navigation to %s", label)
        return False
    return True


# ---------------------------------------------------------------------------
# NavigateSkill
# ---------------------------------------------------------------------------

@skill(
    aliases=[
        "navigate", "go to", "goto",
        "去", "到", "走到", "去到", "导航",
    ],
    direct=True,
)
class NavigateSkill:
    """Navigate the robot to a room discovered during exploration.

    Mode selection (in priority order):
    1. NavStackClient (context.services["nav"]) -- full navigation stack,
       publishes waypoint goal and waits for goal_reached confirmation.
    2. Dead-reckoning -- turns toward waypoints and walks using SceneGraph
       door chain data.

    Room coordinates come exclusively from the SceneGraph populated
    during explore.  No hardcoded room positions are used.

    Works with ANY BaseProtocol implementation (not Go2-specific).
    """

    name: str = "navigate"
    description: str = (
        "Navigate the robot to a named room. "
        "Use this when the user says 'go to X' or '去X'. "
        "Returns an error if the room has not been discovered yet."
    )
    parameters: dict = {
        "room": {
            "type": "string",
            "required": True,
            "description": (
                "Target room name. Examples: kitchen, bedroom, study, "
                "bathroom, living_room, dining_room, guest_bedroom, hallway. "
                "Chinese: 厨房, 卧室, 书房, 卫生间, 客厅, 餐厅, 客房, 走廊."
            ),
        },
    }
    preconditions: list[str] = []
    postconditions: list[str] = []
    effects: dict = {"position": "changed"}
    failure_modes: list[str] = [
        "no_base",
        "unknown_room",
        "navigation_failed",
        "source_room_unknown",
        "layout_not_executable",
        "no_route",
        "invalid_topology",
        "door_too_narrow",
        "segment_no_path",
        "segment_timeout",
        "segment_stalled",
        "stale_planner_response",
        "segment_out_of_tolerance",
        "aborted",
        "ambiguous_goal",
    ]

    def execute(self, params: dict, context: SkillContext) -> SkillResult:
        goal_id = _new_goal_id(params)
        if context.base is None:
            return SkillResult(
                success=False,
                result_data={
                    "goal_id": goal_id,
                    "goal_type": "room",
                    "arrived": False,
                },
                error_message="No base connected",
                diagnosis_code="no_base",
            )

        coordinate_fields = (
            "x", "y", "target", "goal", "coordinate", "coord",
        )
        room_input = str(params.get("room", ""))
        if room_input.strip() and any(key in params for key in coordinate_fields):
            # A named-room request is a topology contract.  Never let callers
            # smuggle a room centre (or any other coordinate) through the
            # coordinate branch and bypass its mandatory door chain.
            _stop_navigation(context.base)
            result = SkillResult(
                success=False,
                error_message=(
                    "Navigation goal is ambiguous: provide either 'room' or "
                    "coordinates, never both."
                ),
                diagnosis_code="ambiguous_goal",
            )
            return _with_navigation_contract(
                result,
                goal_id=goal_id,
                goal_type="room",
                target=None,
                base=context.base,
                start=_safe_position(context.base),
                requested_room=room_input,
                verification_mode="unavailable",
            )

        # --- Coordinate goal (R38): a navigate step may target a world COORDINATE
        # rather than a named SceneGraph room (e.g. "去桌子那里" -> the table
        # standoff, which has no room entry). When x/y (or target=[x, y]) are
        # present, drive the FAR planner directly via base.navigate_to(x, y),
        # bypassing room resolution. World-agnostic and additive — the named-room
        # path below is unchanged when no coordinates are given. The step's
        # at_position verify (graded RAN per D14) is the source of truth for
        # arrival; we surface the actual position regardless of FAR's verdict.
        coord = self._parse_coordinate(params)
        if coord is not None:
            return self._navigate_to_coordinate(coord, context, goal_id=goal_id)
        if any(key in params for key in coordinate_fields):
            _stop_navigation(context.base)
            result = SkillResult(
                success=False,
                error_message="Coordinate goal requires two finite numeric values.",
                diagnosis_code="invalid_coordinate",
            )
            return _with_navigation_contract(
                result,
                goal_id=goal_id,
                goal_type="xy",
                target=None,
                base=context.base,
                start=_safe_position(context.base),
                verification_mode="unavailable",
            )

        sg = context.services.get("spatial_memory")
        world_mode = None
        config = getattr(context, "config", None)
        if isinstance(config, Mapping):
            world_mode = config.get("world_mode")
        resolver = RoomResolver(sg, world_mode=world_mode)
        try:
            resolved = resolver.resolve(room_input)
        except UnknownRoom as exc:
            # Fail closed: an unresolved name never reaches any coordinate API.
            _stop_navigation(context.base)
            available = list(exc.available_rooms)
            if available:
                message = (
                    f"Unknown room: {room_input!r}. "
                    f"Available rooms: {', '.join(available)}"
                )
            else:
                message = "No rooms learned. Run explore first."
            result = SkillResult(
                success=False,
                result_data={"available_rooms": available},
                error_message=message,
                diagnosis_code="unknown_room",
            )
            return _with_navigation_contract(
                result,
                goal_id=goal_id,
                goal_type="room",
                target=None,
                base=context.base,
                start=_safe_position(context.base),
                requested_room=room_input,
                verification_mode="unavailable",
            )
        except RoomPositionUnknown as exc:
            _stop_navigation(context.base)
            result = SkillResult(
                success=False,
                error_message=f"Room '{exc.canonical}' position unknown. Explore more.",
                diagnosis_code="room_not_explored",
            )
            return _with_navigation_contract(
                result,
                goal_id=goal_id,
                goal_type="room",
                target=None,
                base=context.base,
                start=_safe_position(context.base),
                requested_room=room_input,
                canonical_room=exc.canonical,
                source="scene_graph",
                verification_mode="unavailable",
            )

        room_key = resolved.canonical
        target = resolved.navigation_target

        if target != resolved.center:
            logger.info(
                "[NAV] Using collision-free navigation goal for %s: "
                "(%.1f, %.1f), semantic center=(%.1f, %.1f)",
                room_key,
                target[0],
                target[1],
                resolved.center[0],
                resolved.center[1],
            )
        else:
            logger.info(
                "[NAV] Using learned position for %s: (%.1f, %.1f)",
                room_key,
                target[0],
                target[1],
            )

        # Cancel background exploration if running (navigate takes priority)
        try:
            from vector_os_nano.skills.go2.explore import cancel_exploration, is_exploring
            if is_exploring():
                cancel_exploration()
                logger.info("[NAV] Cancelled background exploration for navigation")
        except Exception:
            pass

        # Ensure nav flag exists so bridge path follower is armed
        try:
            import os
            active_file = nav_active_file()
            if not os.path.exists(active_file):
                with open(active_file, "w") as fh:
                    fh.write("1")
        except Exception:
            pass

        start = _safe_position(context.base)
        _begin_navigation_goal(context.base, goal_id, target)

        try:
            # An executable layout-v2 prior is a topology contract, not merely a
            # room-centre lookup.  In known-layout mode it must win over every
            # direct-centre and open-loop fallback below.
            layout_schema_version = int(
                getattr(sg, "layout_schema_version", 0) or 0
            )
            if (
                resolver.world_mode is WorldMode.KNOWN_LAYOUT
                and layout_schema_version > 0
            ):
                if bool(getattr(sg, "has_executable_layout", False)):
                    result = self._navigate_known_layout_route(
                        resolver,
                        room_key,
                        target,
                        context,
                        goal_id=goal_id,
                        start=start,
                    )
                else:
                    _stop_navigation(context.base)
                    result = SkillResult(
                        success=False,
                        error_message=(
                            f"Room layout schema v{layout_schema_version} is "
                            "not executable; safe doorway geometry is required."
                        ),
                        diagnosis_code="layout_not_executable",
                        result_data={
                            "room": room_key,
                            "mode": "safe_door_route",
                            "layout_schema_version": layout_schema_version,
                            "completed_segments": 0,
                        },
                    )
            # --- Legacy Mode 0: Direct nav stack via proxy ---
            elif hasattr(context.base, "navigate_to"):
                result = self._navigate_with_proxy(
                    room_key, target, context, goal_id=goal_id,
                )
            else:
                # --- Mode 1: NavStackClient ---
                nav = context.services.get("nav")
                if nav is not None and nav.is_available:
                    result = self._navigate_with_nav_stack(nav, room_key, target, context)
                else:
                    # --- Mode 2: Dead-reckoning fallback ---
                    result = self._dead_reckoning(room_key, context)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[NAV] Navigation execution failed for %s", room_key)
            result = SkillResult(
                success=False,
                error_message=f"Navigation to {room_key} failed: {exc}",
                diagnosis_code="navigation_failed",
            )

        _finalize_navigation_goal(
            context.base, goal_id, "arrived" if result.success else "failed",
        )
        _stop_navigation(context.base)
        end = _safe_position(context.base)
        location = (
            resolver.locate(end[0], end[1])
            if end is not None
            else None
        )
        verification_mode = (
            location.verification_mode if location is not None else "unavailable"
        )
        return _with_navigation_contract(
            result,
            goal_id=goal_id,
            goal_type="room",
            target=target,
            base=context.base,
            start=start,
            requested_room=resolved.requested,
            canonical_room=room_key,
            source=resolved.source,
            verification_mode=verification_mode,
        )

    # ------------------------------------------------------------------
    # Navigation modes (private)
    # ------------------------------------------------------------------

    def _navigate_known_layout_route(
        self,
        resolver: RoomResolver,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
        *,
        goal_id: str,
        start: tuple[float, float] | None,
        total_timeout: float = _SAFE_ROUTE_TOTAL_TIMEOUT_S,
    ) -> SkillResult:
        """Execute one validated layout-v2 route, segment by segment through FAR.

        This path intentionally has no direct-room-centre, compact door-chain, or
        turn-and-walk fallback.  If source-room membership, topology, one FAR
        segment, or its arrival tolerance cannot be established, the robot holds
        position and no later waypoint is issued.
        """

        base = context.base
        sg = context.services.get("spatial_memory")
        plan = getattr(sg, "plan_door_route", None)
        navigate_to = getattr(base, "navigate_to", None)

        if start is None:
            _stop_navigation(base)
            return SkillResult(
                success=False,
                error_message=(
                    "Cannot safely plan a known-layout route: current position "
                    "is unavailable."
                ),
                diagnosis_code="source_room_unknown",
                result_data={
                    "room": room_key,
                    "mode": "safe_door_route",
                    "completed_segments": 0,
                },
            )

        location = resolver.locate(start[0], start[1])
        src_room = location.canonical
        if not src_room:
            _stop_navigation(base)
            return SkillResult(
                success=False,
                error_message=(
                    "Cannot safely plan a known-layout route: the current "
                    "position is outside every known room."
                ),
                diagnosis_code="source_room_unknown",
                result_data={
                    "room": room_key,
                    "mode": "safe_door_route",
                    "source_xy": [start[0], start[1]],
                    "source_verification_mode": location.verification_mode,
                    "completed_segments": 0,
                },
            )

        if not callable(plan):
            _stop_navigation(base)
            return SkillResult(
                success=False,
                error_message="Executable room layout has no topology planner.",
                diagnosis_code="invalid_topology",
                result_data={
                    "room": room_key,
                    "from_room": src_room,
                    "mode": "safe_door_route",
                    "completed_segments": 0,
                },
            )

        route = plan(src_room, room_key, goal_xy=target)
        route_payload = (
            route.to_dict()
            if callable(getattr(route, "to_dict", None))
            else {}
        )
        if not bool(getattr(route, "success", False)):
            _stop_navigation(base)
            diagnosis = str(
                getattr(route, "diagnosis_code", "") or "invalid_topology"
            )
            message = str(
                getattr(route, "message", "")
                or f"No safe door route from {src_room} to {room_key}."
            )
            return SkillResult(
                success=False,
                error_message=message,
                diagnosis_code=diagnosis,
                result_data={
                    "room": room_key,
                    "from_room": src_room,
                    "mode": "safe_door_route",
                    "route": route_payload,
                    "completed_segments": 0,
                },
            )

        waypoints = tuple(getattr(route, "waypoints", ()) or ())
        if not waypoints:
            _stop_navigation(base)
            return SkillResult(
                success=False,
                error_message="Safe topology planner returned no route segments.",
                diagnosis_code="invalid_topology",
                result_data={
                    "room": room_key,
                    "from_room": src_room,
                    "mode": "safe_door_route",
                    "route": route_payload,
                    "completed_segments": 0,
                },
            )
        if not callable(navigate_to):
            _stop_navigation(base)
            return SkillResult(
                success=False,
                error_message=(
                    "Safe known-layout navigation requires segmented FAR "
                    "(base.navigate_to is unavailable)."
                ),
                diagnosis_code="navigation_failed",
                result_data={
                    "room": room_key,
                    "from_room": src_room,
                    "mode": "safe_door_route",
                    "route": route_payload,
                    "completed_segments": 0,
                },
            )

        initial_leg_lengths = self._safe_route_leg_lengths(start, waypoints)
        total_polyline_length = sum(initial_leg_lengths)
        route_door_ids = tuple(getattr(route, "door_ids", ()) or ())
        door_count = len(route_door_ids)
        if door_count == 0:
            # Compatibility for route-like planner fakes that predate door_ids.
            door_count = sum(
                self._safe_waypoint_payload(waypoint)["kind"] == "door_center"
                for waypoint in waypoints
            )
        route_budget = _bounded_safe_route_budget(
            base_timeout_s=total_timeout,
            door_count=door_count,
            total_polyline_length_m=total_polyline_length,
        )
        effective_total_timeout = float(route_budget["effective_timeout_s"])

        publish_plan = getattr(base, "publish_navigation_plan", None)
        if callable(publish_plan):
            try:
                publish_plan(route_payload, goal_id=goal_id)
            except TypeError:
                try:
                    publish_plan(route_payload)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("[NAV] route visualization publish failed: %s", exc)
            except Exception as exc:  # noqa: BLE001
                logger.debug("[NAV] route visualization publish failed: %s", exc)

        logger.info(
            "[NAV] Safe layout route %s -> %s: %d FAR segments, %d doors, "
            "%.1fm polyline, budget=%.0fs (base=%.0fs door=%.0fs "
            "distance=%.0fs cap=%.0fs)",
            src_room,
            room_key,
            len(waypoints),
            door_count,
            total_polyline_length,
            effective_total_timeout,
            route_budget["base_timeout_s"],
            route_budget["door_budget_s"],
            route_budget["distance_budget_s"],
            route_budget["max_timeout_s"],
        )
        route_started = time.monotonic()
        completed: list[dict[str, Any]] = []
        segment_budgets: list[dict[str, Any]] = []

        for index, waypoint in enumerate(waypoints):
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested

                if is_abort_requested():
                    _stop_navigation(base)
                    return self._safe_route_failure(
                        room_key=room_key,
                        src_room=src_room,
                        route_payload=route_payload,
                        completed=completed,
                        waypoint=waypoint,
                        index=index,
                        route_budget=route_budget,
                        segment_budgets=segment_budgets,
                        diagnosis_code="aborted",
                        message="Navigation aborted",
                    )
            except ImportError:
                pass

            elapsed = time.monotonic() - route_started
            remaining = effective_total_timeout - elapsed
            if remaining <= 0:
                _stop_navigation(base)
                return self._safe_route_failure(
                    room_key=room_key,
                    src_room=src_room,
                    route_payload=route_payload,
                    completed=completed,
                    waypoint=waypoint,
                    index=index,
                    route_budget=route_budget,
                    segment_budgets=segment_budgets,
                    diagnosis_code="navigation_failed",
                    message="Safe room route timed out before the next segment.",
                )

            remaining_segments = len(waypoints) - index
            # A room approach can be several metres while door-centre/post
            # segments are only ~0.6 m.  Equal division starved the long first
            # leg and wasted the same budget on tiny
            # threshold crossings.  Allocate the remaining wall-time by the
            # actual remaining polyline length while reserving a bounded
            # minimum for every later fail-closed segment.
            budget_position = _safe_position(base) or start
            remaining_lengths = self._safe_route_leg_lengths(
                budget_position,
                waypoints[index:],
            )
            total_remaining_length = sum(remaining_lengths)
            distance_share = (
                remaining_lengths[0] / total_remaining_length
                if total_remaining_length > 1e-6
                else 1.0 / remaining_segments
            )
            # Give *every* remaining segment a fixed planning/turning allowance,
            # then distribute only the surplus by polyline length.  Weighting
            # the whole budget by distance underfunds short approaches in
            # confined rooms: master_bedroom -> dining_room, for example, spent
            # most of its time turning around the bed and reached 0.36 m from a
            # 0.30 m pre-door tolerance just as its old 63 s budget expired.
            # A genuinely stationary segment is still bounded by the 30 s
            # progress watchdog, so this allowance does not weaken fail-closed
            # behaviour.
            per_segment_floor = min(
                _SAFE_ROUTE_MIN_SEGMENT_TIMEOUT_S,
                remaining / remaining_segments,
            )
            distributable = max(
                0.0,
                remaining - per_segment_floor * remaining_segments,
            )
            segment_timeout = min(
                remaining,
                per_segment_floor + distributable * distance_share,
            )
            waypoint_payload = self._safe_waypoint_payload(waypoint)
            wx, wy = waypoint_payload["xy"]
            segment_budget = {
                "index": index,
                "kind": waypoint_payload["kind"],
                "label": waypoint_payload["label"],
                "leg_distance_m": round(remaining_lengths[0], 3),
                "remaining_polyline_length_m": round(
                    total_remaining_length, 3,
                ),
                "route_elapsed_s": round(elapsed, 3),
                "remaining_route_timeout_s": round(remaining, 3),
                "allocated_timeout_s": round(segment_timeout, 3),
            }
            segment_budgets.append(segment_budget)
            logger.info(
                "[NAV] Segment %d/%d %s: leg=%.2fm remaining_path=%.2fm "
                "allocated=%.1fs route_remaining=%.1fs",
                index + 1,
                len(waypoints),
                waypoint_payload["label"],
                remaining_lengths[0],
                total_remaining_length,
                segment_timeout,
                remaining,
            )

            def _progress(dist: float, elapsed_s: float) -> None:
                logger.info(
                    "[NAV] %s distance=%.1fm elapsed=%ds",
                    waypoint_payload["kind"],
                    dist,
                    int(elapsed_s),
                )

            set_constraints = getattr(
                base, "set_navigation_segment_constraints", None,
            )
            try:
                if callable(set_constraints):
                    policy_ready = set_constraints(
                        goal_id=goal_id,
                        kind=waypoint_payload["kind"],
                        speed_limit_mps=waypoint_payload["speed_limit_mps"],
                        allow_reverse=waypoint_payload["allow_reverse"],
                        tolerance=waypoint_payload["tolerance"],
                    )
                    if policy_ready is False:
                        _stop_navigation(base)
                        return self._safe_route_failure(
                            room_key=room_key,
                            src_room=src_room,
                            route_payload=route_payload,
                            completed=completed,
                            waypoint=waypoint,
                            index=index,
                            route_budget=route_budget,
                            segment_budgets=segment_budgets,
                            diagnosis_code="navigation_failed",
                            message=(
                                "Bridge did not acknowledge the segment safety "
                                f"policy for {waypoint_payload['label']}."
                            ),
                        )

                # The explicit false is safety-critical: a failed segment may not
                # recurse into the proxy's legacy compact-door or open-loop path.
                nav_result = bool(
                    navigate_to(
                        wx,
                        wy,
                        timeout=segment_timeout,
                        on_progress=_progress,
                        goal_id=goal_id,
                        allow_door_fallback=False,
                        waypoint_kind=waypoint_payload["kind"],
                        speed_limit_mps=waypoint_payload["speed_limit_mps"],
                        allow_reverse=waypoint_payload["allow_reverse"],
                        arrival_tolerance=waypoint_payload["tolerance"],
                    )
                )
            except TypeError as exc:
                _stop_navigation(base)
                return self._safe_route_failure(
                    room_key=room_key,
                    src_room=src_room,
                    route_payload=route_payload,
                    completed=completed,
                    waypoint=waypoint,
                    index=index,
                    route_budget=route_budget,
                    segment_budgets=segment_budgets,
                    diagnosis_code="navigation_failed",
                    message=(
                        "Base does not support fail-closed segmented FAR "
                        f"navigation: {exc}"
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                _stop_navigation(base)
                return self._safe_route_failure(
                    room_key=room_key,
                    src_room=src_room,
                    route_payload=route_payload,
                    completed=completed,
                    waypoint=waypoint,
                    index=index,
                    route_budget=route_budget,
                    segment_budgets=segment_budgets,
                    diagnosis_code="navigation_failed",
                    message=(
                        f"FAR segment {index} ({waypoint_payload['label']}) "
                        f"failed: {exc}"
                    ),
                )
            finally:
                clear_constraints = getattr(
                    base, "clear_navigation_segment_constraints", None,
                )
                if callable(clear_constraints):
                    try:
                        clear_constraints(goal_id=goal_id)
                    except TypeError:
                        try:
                            clear_constraints()
                        except Exception:  # noqa: BLE001
                            pass
                    except Exception:  # noqa: BLE001
                        pass

            landed = _safe_position(base)
            landing_distance = (
                _distance(landed[0], landed[1], wx, wy)
                if landed is not None
                else None
            )
            segment_budget["transport_succeeded"] = nav_result
            segment_budget["end_xy"] = (
                [round(landed[0], 3), round(landed[1], 3)]
                if landed is not None
                else None
            )
            segment_budget["distance_to_waypoint_m"] = (
                round(landing_distance, 3)
                if landing_distance is not None
                else None
            )

            if not nav_result:
                diagnosis_code, message = _segment_failure(
                    base,
                    goal_id,
                    index=index,
                    label=waypoint_payload["label"],
                )
                _stop_navigation(base)
                return self._safe_route_failure(
                    room_key=room_key,
                    src_room=src_room,
                    route_payload=route_payload,
                    completed=completed,
                    waypoint=waypoint,
                    index=index,
                    route_budget=route_budget,
                    segment_budgets=segment_budgets,
                    diagnosis_code=diagnosis_code,
                    message=message,
                )

            tolerance = float(waypoint_payload["tolerance"])
            if landing_distance is None or landing_distance > tolerance:
                segment_budget["arrived_within_tolerance"] = False
                _stop_navigation(base)
                return self._safe_route_failure(
                    room_key=room_key,
                    src_room=src_room,
                    route_payload=route_payload,
                    completed=completed,
                    waypoint=waypoint,
                    index=index,
                    route_budget=route_budget,
                    segment_budgets=segment_budgets,
                    diagnosis_code="segment_out_of_tolerance",
                    message=(
                        f"FAR segment {index} did not finish within "
                        f"{tolerance:.2f} m"
                        + (
                            f" (distance {landing_distance:.2f} m)."
                            if landing_distance is not None
                            else "."
                        )
                    ),
                )

            segment_budget["arrived_within_tolerance"] = True
            completed.append(waypoint_payload)

        final_position = _safe_position(base)
        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "from_room": src_room,
                "position": (
                    [round(final_position[0], 1), round(final_position[1], 1)]
                    if final_position is not None
                    else None
                ),
                "mode": "safe_door_route",
                "route": route_payload,
                "route_budget": route_budget,
                "segment_budgets": segment_budgets,
                "completed_segments": len(completed),
                "segment_count": len(waypoints),
            },
        )

    @staticmethod
    def _safe_route_leg_lengths(
        start: tuple[float, float],
        waypoints: tuple[Any, ...],
    ) -> list[float]:
        """Return consecutive route leg lengths from the current body pose."""

        cursor = (float(start[0]), float(start[1]))
        lengths: list[float] = []
        for waypoint in waypoints:
            waypoint_xy = NavigateSkill._safe_waypoint_payload(waypoint)["xy"]
            target = (float(waypoint_xy[0]), float(waypoint_xy[1]))
            lengths.append(_distance(cursor[0], cursor[1], target[0], target[1]))
            cursor = target
        return lengths

    @staticmethod
    def _safe_waypoint_payload(waypoint: Any) -> dict[str, Any]:
        """Return the stable structured waypoint shape used at the motor seam."""

        to_dict = getattr(waypoint, "to_dict", None)
        if callable(to_dict):
            payload = dict(to_dict())
        else:
            xy = getattr(waypoint, "xy")
            speed_limit = getattr(waypoint, "speed_limit_mps")
            payload = {
                "kind": str(getattr(waypoint, "kind")),
                "room_from": str(getattr(waypoint, "room_from")),
                "room_to": str(getattr(waypoint, "room_to")),
                "xy": [float(xy[0]), float(xy[1])],
                "tolerance": float(getattr(waypoint, "tolerance")),
                "speed_limit_mps": (
                    None if speed_limit is None else float(speed_limit)
                ),
                "allow_reverse": bool(getattr(waypoint, "allow_reverse")),
                "door_id": getattr(waypoint, "door_id", None),
                "label": str(getattr(waypoint, "label", "")),
            }
        payload["xy"] = [
            float(payload["xy"][0]),
            float(payload["xy"][1]),
        ]
        payload["tolerance"] = float(payload["tolerance"])
        speed_limit = payload.get("speed_limit_mps")
        payload["speed_limit_mps"] = (
            None if speed_limit is None else float(speed_limit)
        )
        payload["allow_reverse"] = bool(payload["allow_reverse"])
        return payload

    @staticmethod
    def _safe_route_failure(
        *,
        room_key: str,
        src_room: str,
        route_payload: dict[str, Any],
        completed: list[dict[str, Any]],
        waypoint: Any,
        index: int,
        route_budget: dict[str, Any],
        segment_budgets: list[dict[str, Any]],
        diagnosis_code: str,
        message: str,
    ) -> SkillResult:
        return SkillResult(
            success=False,
            error_message=message,
            diagnosis_code=diagnosis_code,
            result_data={
                "room": room_key,
                "from_room": src_room,
                "mode": "safe_door_route",
                "route": route_payload,
                "route_budget": route_budget,
                "segment_budgets": segment_budgets,
                "completed_segments": len(completed),
                "failed_segment_index": index,
                "failed_waypoint": NavigateSkill._safe_waypoint_payload(waypoint),
            },
        )

    @staticmethod
    def _parse_coordinate(params: dict) -> tuple[float, float] | None:
        """Extract a world (x, y) coordinate goal from params, or None.

        Accepts ``{"x": .., "y": ..}`` or ``{"target": [x, y]}`` /
        ``{"goal": [x, y]}`` (a 2-list/tuple). Returns None when no coordinate is
        present (the named-room path then runs) — back-compatible.
        """
        def _finite(value: Any) -> float | None:
            if isinstance(value, bool):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) else None

        x = params.get("x")
        y = params.get("y")
        if x is not None and y is not None:
            fx, fy = _finite(x), _finite(y)
            return (fx, fy) if fx is not None and fy is not None else None
        for key in ("target", "goal", "coordinate", "coord"):
            val = params.get(key)
            if isinstance(val, (list, tuple)) and len(val) == 2:
                fx, fy = _finite(val[0]), _finite(val[1])
                return (fx, fy) if fx is not None and fy is not None else None
        return None

    def _navigate_to_coordinate(
        self,
        coord: tuple[float, float],
        context: SkillContext,
        *,
        goal_id: str | None = None,
    ) -> SkillResult:
        """Drive the base to a world coordinate via the FAR planner (navigate_to).

        Used for a coordinate navigate goal (no SceneGraph room). Honest: the
        step's success is the planner's own arrival verdict and the actual landing
        position is reported; the at_position verify oracle (RAN) is what grades the
        step. If the base lacks navigate_to (no nav stack) we fail loud.
        """
        base = context.base
        tx, ty = coord
        goal_id = goal_id or f"nav-{uuid.uuid4().hex}"
        start = _safe_position(base)
        navigate_to = getattr(base, "navigate_to", None)
        if not callable(navigate_to):
            result = SkillResult(
                success=False,
                error_message="Coordinate navigation needs a nav stack (base.navigate_to absent).",
                diagnosis_code="navigation_failed",
            )
            return _with_navigation_contract(
                result,
                goal_id=goal_id,
                goal_type="xy",
                target=coord,
                base=base,
                start=start,
                verification_mode="at_position",
            )

        # Ensure the bridge path follower is armed (same as the room path).
        try:
            import os
            active_file = nav_active_file()
            if not os.path.exists(active_file):
                with open(active_file, "w") as fh:
                    fh.write("1")
        except Exception:
            pass

        logger.info("[NAV] Coordinate goal -> navigate_to(%.2f, %.2f) via FAR", tx, ty)

        def _progress(dist: float, elapsed: float) -> None:
            logger.info(
                "[NAV] Coordinate progress distance=%.1fm elapsed=%ds",
                dist,
                int(elapsed),
            )

        _begin_navigation_goal(base, goal_id, coord)
        nav_exception = ""
        try:
            ok = bool(
                navigate_to(
                    tx,
                    ty,
                    timeout=_COORD_NAV_TIMEOUT_S,
                    on_progress=_progress,
                    goal_id=goal_id,
                    arrival_tolerance=AT_POSITION_TOL_M,
                )
            )
        except TypeError:
            # A base whose navigate_to does not accept the extra kwargs.
            try:
                ok = bool(
                    navigate_to(
                        tx, ty, timeout=_COORD_NAV_TIMEOUT_S, on_progress=_progress,
                    )
                )
            except TypeError:
                try:
                    ok = bool(navigate_to(tx, ty))
                except Exception as exc:  # noqa: BLE001
                    nav_exception = str(exc)
                    ok = False
            except Exception as exc:  # noqa: BLE001
                nav_exception = str(exc)
                ok = False
        except Exception as exc:  # noqa: BLE001
            nav_exception = str(exc)
            ok = False

        # HOLD at the goal. navigate_to leaves the dog under the planner/TARE,
        # which keeps publishing /way_point and DRIFTS the dog away from the goal
        # (observed: dog wandered to (7.7, 4.5) before the next skill perceived).
        # A downstream manipulation step needs the dog STATIONARY at the table, so
        # disarm the nav flag and stop the base now. (Symmetric with how a stop
        # command clears the flag; the proxy's loops treat the missing flag as
        # "no active nav".)
        pos = base.get_position()
        dist = _distance(pos[0], pos[1], tx, ty)
        # The action and verifier share one geometric contract. A transport-level
        # ``ok`` outside this radius is not arrival; conversely, a timeout that
        # physically landed inside it is accepted and then independently verified.
        arrived = dist <= AT_POSITION_TOL_M
        _finalize_navigation_goal(
            base, goal_id, "arrived" if arrived else "failed",
        )
        _stop_navigation(base)
        result = SkillResult(
            success=arrived,
            error_message="" if arrived else (
                (
                    f"navigate_to({tx}, {ty}) failed: {nav_exception}"
                    if nav_exception
                    else (
                        f"navigate_to({tx}, {ty}) did not reach the "
                        f"{AT_POSITION_TOL_M:.1f} m acceptance radius "
                        f"(ended {dist:.1f} m away)"
                    )
                )
            ),
            diagnosis_code=None if arrived else "navigation_failed",
            result_data={
                "target": [round(tx, 1), round(ty, 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "far_confirmed": ok,
                "mode": "proxy_coord",
            },
        )
        return _with_navigation_contract(
            result,
            goal_id=goal_id,
            goal_type="xy",
            target=coord,
            base=base,
            start=start,
            verification_mode="at_position",
        )

    def _navigate_with_proxy(
        self,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
        *,
        goal_id: str | None = None,
    ) -> SkillResult:
        """Mode 0: Navigate via Go2ROS2Proxy.navigate_to() — FAR planner path.

        Called when context.base exposes navigate_to() (i.e. the proxy is
        connected to the live nav stack).  Falls back to dead-reckoning if
        the proxy call returns False.
        """
        logger.info(
            "[NAV] Proxy mode -> room=%s target=(%.1f, %.1f)",
            room_key, target[0], target[1],
        )

        def _progress(dist: float, elapsed: float) -> None:
            logger.info(
                "[NAV] Room progress distance=%.1fm elapsed=%ds",
                dist,
                int(elapsed),
            )

        try:
            nav_result = context.base.navigate_to(
                target[0],
                target[1],
                timeout=45.0,
                on_progress=_progress,
                goal_id=goal_id,
            )
        except TypeError:
            nav_result = context.base.navigate_to(
                target[0], target[1], timeout=45.0, on_progress=_progress
            )

        pos = context.base.get_position()
        dist = _distance(pos[0], pos[1], target[0], target[1])

        if not nav_result:
            logger.warning(
                "[NAV] Proxy navigate_to timed out; falling back to dead-reckoning"
            )
            return self._dead_reckoning(room_key, context)

        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "target": [round(target[0], 1), round(target[1], 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "mode": "proxy_nav_stack",
            },
        )

    def _navigate_with_nav_stack(
        self,
        nav: Any,
        room_key: str,
        target: tuple[float, float],
        context: SkillContext,
    ) -> SkillResult:
        """Delegate navigation to NavStackClient.

        Sends /way_point and monitors position. Does not rely on /goal_reached
        since the nav stack doesn't always publish it reliably.
        """
        logger.info("[NAV] Using nav stack -> room=%s target=(%.1f, %.1f)",
                    room_key, target[0], target[1])

        # Send the waypoint and wait for result
        nav_result = nav.navigate_to(target[0], target[1], timeout=30.0)

        # Check actual position after navigation
        pos = context.base.get_position() if context.base else None
        if pos is None:
            odom = nav.get_state_estimation()
            pos = [odom.x, odom.y, odom.z] if odom else [0, 0, 0]

        dist = _distance(pos[0], pos[1], target[0], target[1])

        if not nav_result:
            return SkillResult(
                success=False,
                error_message=f"Navigation to {room_key} failed (timeout or rejected)",
                diagnosis_code="navigation_failed",
                result_data={
                    "room": room_key,
                    "target": [round(target[0], 1), round(target[1], 1)],
                    "position": [round(pos[0], 1), round(pos[1], 1)],
                    "distance_to_target": round(dist, 1),
                    "mode": "nav_stack",
                },
            )

        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "target": [round(target[0], 1), round(target[1], 1)],
                "position": [round(pos[0], 1), round(pos[1], 1)],
                "distance_to_target": round(dist, 1),
                "mode": "nav_stack",
            },
        )

    def _dead_reckoning(
        self,
        room_key: str,
        context: SkillContext,
        total_timeout: float = 45.0,
    ) -> SkillResult:
        """Navigate via nav stack door chain using SceneGraph waypoints.

        Publishes each waypoint to /way_point via base.navigate_to() so the
        localPlanner handles obstacle avoidance.  The total_timeout budget is
        divided dynamically across remaining waypoints (min 5s each); arrival
        is confirmed when within _DOORCHAIN_ARRIVAL_RADIUS meters of target.
        """
        base = context.base
        sg = context.services.get("spatial_memory")

        pos = base.get_position()
        cx, cy = pos[0], pos[1]
        src_room = _detect_current_room(cx, cy, sg=sg)

        # Get door chain from SceneGraph
        if sg is None or not hasattr(sg, "get_door_chain"):
            return SkillResult(
                success=False,
                error_message="No door data. Explore first.",
                diagnosis_code="room_not_explored",
            )

        # Check if already at destination
        target_room_node = sg.get_room(room_key) if hasattr(sg, "get_room") else None
        if target_room_node is not None:
            target_cx = target_room_node.center_x
            target_cy = target_room_node.center_y
            if _distance(cx, cy, target_cx, target_cy) < _DOORCHAIN_ARRIVAL_RADIUS:
                return SkillResult(
                    success=True,
                    result_data={
                        "room": room_key,
                        "position": [round(cx, 1), round(cy, 1)],
                        "note": "already here",
                    },
                )

        logger.info("[NAV] Door-chain (nav stack): %s -> %s", src_room, room_key)

        # Get waypoint sequence from SceneGraph door chain
        waypoints = sg.get_door_chain(src_room, room_key)

        if not waypoints:
            return SkillResult(
                success=False,
                error_message=(
                    f"No door data between '{src_room}' and '{room_key}'. "
                    "Explore first."
                ),
                diagnosis_code="room_not_explored",
            )

        # Execute each waypoint via nav stack (obstacle avoidance)
        # Dynamic per-waypoint timeout: divide remaining budget evenly across
        # remaining waypoints, but never less than 5s per waypoint.
        start_time = time.monotonic()

        for i, (wx, wy, label) in enumerate(waypoints):
            # --- Abort check between waypoints ---
            try:
                from vector_os_nano.vcli.cognitive.abort import is_abort_requested
                if is_abort_requested():
                    _stop_navigation(base)
                    return SkillResult(
                        success=False,
                        error_message="Navigation aborted",
                        diagnosis_code="aborted",
                    )
            except ImportError:
                pass

            # Compute remaining budget for this waypoint
            elapsed = time.monotonic() - start_time
            remaining = total_timeout - elapsed
            if remaining <= 0:
                _stop_navigation(base)
                return SkillResult(
                    success=False,
                    error_message="Navigation timeout",
                    diagnosis_code="navigation_failed",
                )
            n_remaining = len(waypoints) - i
            per_wp = max(remaining / n_remaining, 5.0)

            # Check arrival before sending — skip waypoint if already close enough
            cur_pos = base.get_position()
            if _distance(cur_pos[0], cur_pos[1], wx, wy) < _DOORCHAIN_ARRIVAL_RADIUS:
                logger.info("[NAV] Already within %.1fm of %s — skipping", _DOORCHAIN_ARRIVAL_RADIUS, label)
                continue

            logger.info(
                "[NAV] Navigate to waypoint %s (%.1f, %.1f) timeout=%.0fs",
                label, wx, wy, per_wp,
            )
            cur_pos2 = base.get_position()
            seg_dist = _distance(cur_pos2[0], cur_pos2[1], wx, wy)
            logger.info(
                "[NAV] Door-chain heading to %s distance=%.1fm",
                label,
                seg_dist,
            )

            def _progress(dist: float, elapsed_s: float) -> None:
                logger.info(
                    "[NAV] Door-chain progress distance=%.1fm elapsed=%ds",
                    dist,
                    int(elapsed_s),
                )

            # Use go_to_waypoint (simple /way_point) to avoid recursive
            # navigate_to → FAR probe → door-chain → navigate_to cascade.
            _go_fn = (
                getattr(base, "go_to_waypoint", None)
                or getattr(base, "navigate_to", None)
            )
            if not callable(_go_fn):
                _stop_navigation(base)
                return SkillResult(
                    success=False,
                    error_message=(
                        f"Navigation planner unavailable near {label}; "
                        "refusing open-loop motion"
                    ),
                    diagnosis_code="navigation_failed",
                )

            ok = _go_fn(
                float(wx), float(wy),
                timeout=per_wp,
                on_progress=_progress,
            )
            if not ok:
                # navigate_to returned False — timed out or rejected by nav stack
                _stop_navigation(base)
                return SkillResult(
                    success=False,
                    error_message=f"Navigation timed out near {label}",
                    diagnosis_code="navigation_failed",
                )

        final_pos = base.get_position()
        return SkillResult(
            success=True,
            result_data={
                "room": room_key,
                "from_room": src_room,
                "position": [round(final_pos[0], 1), round(final_pos[1], 1)],
                "mode": "dead_reckoning",
            },
        )
