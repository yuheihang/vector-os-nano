# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Navigation primitives — wrap SceneGraph + NavStackClient for CaP-X generated code.

All functions are module-level and read from the module-global _ctx.
Requires init_primitives() to be called before use.
"""
from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

from vector_os_nano.navigation.room_resolver import (
    RoomPositionUnknown,
    RoomResolver,
    UnknownRoom,
)
from vector_os_nano.vcli.primitives import PrimitiveContext

if TYPE_CHECKING:
    pass

_ctx: PrimitiveContext | None = None


def _require_scene_graph() -> object:
    """Return _ctx.scene_graph or raise RuntimeError if unavailable."""
    if _ctx is None or _ctx.scene_graph is None:
        raise RuntimeError(
            "No SceneGraph connected. Call init_primitives() with a valid scene_graph."
        )
    return _ctx.scene_graph


# ---------------------------------------------------------------------------
# Room queries
# ---------------------------------------------------------------------------


def nearest_room() -> str | None:
    """Current room name based on robot position, or None if unknown.

    Returns:
        Room name string or None if no rooms are known.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    from vector_os_nano.vcli.primitives import locomotion
    sg = _require_scene_graph()
    pos = locomotion.get_position()
    return sg.nearest_room(pos[0], pos[1])


# ---------------------------------------------------------------------------
# Goal sending
# ---------------------------------------------------------------------------


def publish_goal(x: float, y: float) -> None:
    """Send a navigation goal to the planner.

    Tries nav_client first; falls back to base.navigate_to if available.

    Args:
        x: Target x coordinate in world frame (meters).
        y: Target y coordinate in world frame (meters).

    Raises:
        RuntimeError: If neither nav_client nor base is available.
    """
    if _ctx is None:
        raise RuntimeError(
            "Primitives not initialized. Call init_primitives() first."
        )
    if _ctx.nav_client is not None:
        _ctx.nav_client.navigate_to(x, y)
    elif _ctx.base is not None and hasattr(_ctx.base, "navigate_to"):
        _ctx.base.navigate_to(x, y)
    else:
        raise RuntimeError(
            "No navigation interface available. Provide nav_client or a base with navigate_to."
        )


# ---------------------------------------------------------------------------
# Blocking wait
# ---------------------------------------------------------------------------


def wait_until_near(
    x: float,
    y: float,
    tolerance: float = 0.8,
    timeout: float = 60.0,
) -> bool:
    """Block until the robot is within tolerance of the target position.

    Polls at 2 Hz. Returns immediately if already within tolerance.

    Args:
        x: Target x coordinate in world frame (meters).
        y: Target y coordinate in world frame (meters).
        tolerance: Acceptable radius in meters. Default 0.8 m.
        timeout: Maximum wait time in seconds. Default 60 s.

    Returns:
        True if position reached within timeout, False on timeout.
    """
    from vector_os_nano.vcli.primitives import locomotion
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pos = locomotion.get_position()
        dx = pos[0] - x
        dy = pos[1] - y
        if math.sqrt(dx * dx + dy * dy) <= tolerance:
            return True
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------


def get_door_chain(from_room: str, to_room: str) -> list[tuple[float, float, str]]:
    """Get waypoints between rooms via BFS on the SceneGraph.

    Args:
        from_room: Source room name.
        to_room: Destination room name.

    Returns:
        List of (x, y, label) tuples. Empty list if no path found.

    Raises:
        RuntimeError: If no SceneGraph is connected.
    """
    sg = _require_scene_graph()
    return list(sg.get_door_chain(from_room, to_room))


def _formal_navigate_skill() -> Any | None:
    if _ctx is None or _ctx.skill_registry is None:
        return None
    getter = getattr(_ctx.skill_registry, "get", None)
    return getter("navigate") if callable(getter) else None


def _skill_context() -> Any:
    from vector_os_nano.core.skill import SkillContext

    services = {"spatial_memory": _ctx.scene_graph}
    if _ctx.nav_client is not None:
        services["nav"] = _ctx.nav_client
    return SkillContext(
        base=_ctx.base,
        services=services,
    )


def _hold_position() -> None:
    """Best-effort fail-closed stop for a rejected navigation request."""

    if _ctx is None or _ctx.base is None:
        return
    stop_navigation = getattr(_ctx.base, "stop_navigation", None)
    if callable(stop_navigation):
        stop_navigation()
        return
    set_velocity = getattr(_ctx.base, "set_velocity", None)
    if callable(set_velocity):
        set_velocity(0.0, 0.0, 0.0)


def navigate_room(room: str) -> bool:
    """Navigate to a named room through the shared resolver/formal skill.

    Unknown names fail closed before any coordinate is published.

    Args:
        room: Target room name.

    Returns:
        True if the robot arrived, False on failure or timeout.

    Raises:
        RuntimeError: If neither skill_registry nor SceneGraph is available.
    """
    if _ctx is None:
        raise RuntimeError(
            "Primitives not initialized. Call init_primitives() first."
        )

    sg = _require_scene_graph()
    try:
        resolved = RoomResolver(sg).resolve(room)
    except (UnknownRoom, RoomPositionUnknown):
        _hold_position()
        return False

    skill = _formal_navigate_skill()
    if skill is not None:
        result = skill.execute({"room": resolved.canonical}, _skill_context())
        return bool(getattr(result, "success", False))

    # A named room is not a coordinate shortcut.  Without the formal skill
    # there is no source-room membership check or executable door topology, so
    # publishing the room centre could send the robot through a wall.
    _hold_position()
    return False


def navigate_xy(x: float, y: float) -> bool:
    """Navigate to an explicit finite world coordinate."""

    if _ctx is None:
        raise RuntimeError(
            "Primitives not initialized. Call init_primitives() first."
        )
    if isinstance(x, bool) or isinstance(y, bool):
        return False
    try:
        tx, ty = float(x), float(y)
    except (TypeError, ValueError):
        return False
    if not (math.isfinite(tx) and math.isfinite(ty)):
        return False
    skill = _formal_navigate_skill()
    if skill is not None:
        result = skill.execute({"x": tx, "y": ty}, _skill_context())
        return bool(getattr(result, "success", False))
    publish_goal(tx, ty)
    return wait_until_near(tx, ty)


def navigate_to_room(room: str) -> bool:
    """Backward-compatible alias for :func:`navigate_room`."""

    return navigate_room(room)
