# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deterministic sim-oracle verify predicates over a connected mobile base.

The go2 counterpart of ``arm_sim_oracle``: these are the callables a base
sub-goal's ``verify`` expression evaluates against. They read the sim's
DETERMINISTIC ground truth off the connected base — ``get_position`` /
``get_heading`` — never the VLM (ADR-008: generator and verifier independent).

Single-sourced HERE in the kernel so BOTH the playground go2 world AND the plain
``RobotWorld`` can consume them without the kernel importing the playground
(ADR-008 / kernel rule 2: the dependency edge is one-way, playground -> kernel).
``playground/verify/base_predicates.py`` is a thin re-export shim.

Grounding contract:
- The base is reached from the agent via ``getattr(agent, "_base", None)`` (the
  same accessor the engine's SkillContext builder and robot_context use). When
  the base is absent or not connected, every predicate FAILS SAFE (returns
  ``False``) — it must NEVER raise into the GoalVerifier sandbox.
- Each predicate is a thin factory bound to the connected ``agent`` (and, for
  ``visited`` / ``in_room``, the scenario or SceneGraph room source), so a world
  can drop them straight into the verify namespace.

The predicates are side-effect-free: position / heading reads do not advance the
sim. Legacy ``visited`` checks an axis-aligned scenario box. The primary
``in_room`` predicate delegates aliases and membership to the shared
``RoomResolver``: ``room_at`` / polygon / bounds geometry is preferred, with an
explicitly recorded ``nearest_center`` fallback only when geometry is unavailable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable

from vector_os_nano.navigation.room_resolver import RoomResolver, UnknownRoom

logger = logging.getLogger(__name__)

# Tolerances (metres / radians). ``AT_POSITION_TOL_M`` is public because motion
# transports must not declare arrival outside the verifier's acceptance radius.
# The private alias remains for compatibility with older imports.
AT_POSITION_TOL_M: float = 0.5
_AT_POSITION_TOL_M: float = AT_POSITION_TOL_M
_FACING_TOL_RAD: float = math.radians(20.0)


def _get_base(agent: Any) -> Any | None:
    """Return the connected base reachable from *agent*, or None (fail-safe).

    Mirrors the kernel accessor ``getattr(agent, "_base", None)``. Returns None
    when no agent, no base, or the base reports itself disconnected — so callers
    can fail safe without raising.
    """
    if agent is None:
        return None
    base = getattr(agent, "_base", None)
    if base is None:
        return None
    # Respect an explicit connected flag when present (MuJoCoGo2 exposes
    # ``_connected``); absence of the attr means the base has no such notion, so
    # treat it as usable.
    if getattr(base, "_connected", True) is False:
        return None
    return base


def _base_position(base: Any) -> list[float] | None:
    """Return the base's current xyz, or None (fail-safe)."""
    try:
        pos = base.get_position()
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle base.get_position failed: %s", exc)
        return None


def _base_heading(base: Any) -> float | None:
    """Return the base's current yaw (radians), or None (fail-safe)."""
    try:
        return float(base.get_heading())
    except Exception as exc:  # noqa: BLE001
        logger.debug("sim-oracle base.get_heading failed: %s", exc)
        return None


def _angle_delta(a: float, b: float) -> float:
    """Smallest absolute difference between two angles (radians), in [0, pi]."""
    return abs(math.atan2(math.sin(a - b), math.cos(a - b)))


def make_at_position(agent: Any) -> Callable[..., bool]:
    """Build ``at_position(x, y, tol=...)`` bound to *agent*.

    True when the base's planar (xy) position is within ``tol`` metres of the
    target ``(x, y)``. ``tol`` defaults to ``_AT_POSITION_TOL_M``. Reads
    deterministic ground truth; fails safe to ``False`` (bad args or no base).
    """

    def at_position(x: Any, y: Any, tol: Any = _AT_POSITION_TOL_M) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        try:
            tx, ty, t = float(x), float(y), float(tol)
        except (TypeError, ValueError):
            return False
        pos = _base_position(base)
        if pos is None:
            return False
        return math.dist((pos[0], pos[1]), (tx, ty)) <= t

    return at_position


def make_facing(agent: Any) -> Callable[..., bool]:
    """Build ``facing(heading, tol=...)`` bound to *agent*.

    True when the base's yaw is within ``tol`` radians of the target ``heading``
    (radians), wrapping correctly across the +/-pi seam. ``tol`` defaults to
    ``_FACING_TOL_RAD``. Reads deterministic ground truth; fails safe to
    ``False`` (bad args or no base).
    """

    def facing(heading: Any, tol: Any = _FACING_TOL_RAD) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        try:
            target, t = float(heading), float(tol)
        except (TypeError, ValueError):
            return False
        yaw = _base_heading(base)
        if yaw is None:
            return False
        return _angle_delta(yaw, target) <= t

    return facing


@dataclass(frozen=True)
class _BoundsRoom:
    """Minimal immutable room node used for scenario-owned room bounds."""

    room_id: str
    center_x: float
    center_y: float
    bounds: tuple[float, float, float, float]
    # Scenario rooms are explicitly supplied by the active world rather than
    # leaked layout priors, so they remain visible in either world mode.
    visit_count: int = 1


class _BoundsSceneGraph:
    """Read-only SceneGraph adapter over a scenario room-box mapping.

    ``RoomResolver`` takes one SceneGraph-like source for both name availability
    and membership. Playground scenarios already own exact room bounds but do not
    need persistent spatial memory, so this adapter lets them use the same
    resolver instead of duplicating alias or point-in-room logic.
    """

    def __init__(self, rooms: dict[str, tuple[float, float, float, float]]) -> None:
        self._rooms: dict[str, _BoundsRoom] = {}
        for name, box in rooms.items():
            x_min, y_min, x_max, y_max = box
            room_id = str(name)
            self._rooms[room_id] = _BoundsRoom(
                room_id=room_id,
                center_x=(x_min + x_max) / 2.0,
                center_y=(y_min + y_max) / 2.0,
                bounds=box,
            )

    def get_all_rooms(self) -> list[_BoundsRoom]:
        return list(self._rooms.values())

    def get_room(self, room_id: str) -> _BoundsRoom | None:
        return self._rooms.get(str(room_id))

    def nearest_room(self, x: float, y: float) -> str | None:
        if not self._rooms:
            return None
        return min(
            self._rooms.values(),
            key=lambda room: (
                math.dist((float(x), float(y)), (room.center_x, room.center_y)),
                room.room_id,
            ),
        ).room_id


def _valid_room_boxes(
    rooms: dict[str, tuple[float, float, float, float]] | None,
) -> dict[str, tuple[float, float, float, float]]:
    """Return a finite, normalized copy of a room-box mapping."""

    normalized: dict[str, tuple[float, float, float, float]] = {}
    for name, raw_box in (rooms or {}).items():
        if not _is_box(raw_box):
            continue
        x_min, y_min, x_max, y_max = (float(v) for v in raw_box)
        box = (x_min, y_min, x_max, y_max)
        if (
            all(math.isfinite(v) for v in box)
            and x_min <= x_max
            and y_min <= y_max
        ):
            normalized[str(name)] = box
    return normalized


class _InRoomPredicate:
    """Callable room predicate with inspectable verification details.

    ``GoalVerifier`` needs a boolean callable, while P1 also requires any
    geometry downgrade to be observable. The latest call therefore records
    ``verification_mode`` (``room_at`` / ``polygon`` / ``bounds`` /
    ``nearest_center`` / ``unavailable``), its canonical target, and the current
    room as public attributes without changing the boolean return contract.
    """

    def __init__(self, agent: Any, resolver: RoomResolver) -> None:
        self._agent = agent
        self._resolver = resolver
        self.verification_mode: str = "unavailable"
        self.canonical_room: str | None = None
        self.current_room: str | None = None

    def __call__(self, room_id: Any) -> bool:
        self.verification_mode = "unavailable"
        self.canonical_room = None
        self.current_room = None

        try:
            canonical = self._resolver.canonicalize(room_id)
        except (UnknownRoom, TypeError, ValueError):
            # Unknown aliases fail closed before touching the robot.
            return False
        self.canonical_room = canonical

        base = _get_base(self._agent)
        if base is None:
            return False
        pos = _base_position(base)
        if pos is None:
            return False

        try:
            location = self._resolver.locate(pos[0], pos[1])
        except Exception as exc:  # noqa: BLE001
            logger.debug("sim-oracle room resolution failed: %s", exc)
            return False
        self.verification_mode = location.verification_mode
        self.current_room = location.canonical
        return location.canonical == canonical


def make_in_room(
    agent: Any,
    *,
    scene_graph: Any = None,
    rooms: dict[str, tuple[float, float, float, float]] | None = None,
    resolver: RoomResolver | None = None,
) -> Callable[..., bool]:
    """Build ``in_room(room_id)`` over deterministic base + room ground truth.

    Name canonicalization and point membership are single-sourced through the
    shared :class:`RoomResolver`. A live SceneGraph may expose ``room_at``,
    polygon/bounds data, or only room centres; the resolver uses them in that
    order and records ``nearest_center`` when it must degrade. A playground can
    instead supply scenario-owned ``rooms`` bounds, adapted to the same contract.

    Unknown room names, unavailable bases, malformed geometry, and oracle read
    errors all fail safe to ``False`` and never escape into ``GoalVerifier``.
    """

    if resolver is None:
        from vector_os_nano.navigation.world_mode import world_mode_for_agent

        room_boxes = _valid_room_boxes(rooms)
        source = scene_graph
        if source is None and room_boxes:
            source = _BoundsSceneGraph(room_boxes)
        resolver = RoomResolver(
            source,
            world_mode=world_mode_for_agent(agent),
            room_bounds=room_boxes,
        )
    return _InRoomPredicate(agent, resolver)


def make_visited(agent: Any, rooms: dict[str, tuple[float, float, float, float]]) -> Callable[..., bool]:
    """Build ``visited(room)`` bound to *agent* + the scenario's named rooms.

    True when the base's current planar position lies inside the named room's
    axis-aligned bounding box ``(x_min, y_min, x_max, y_max)``. An unknown room
    name fails safe to ``False`` (it is not silently treated as "anywhere").
    Reads deterministic ground truth; fails safe to ``False`` when the base is
    unavailable. ``rooms`` is the scenario-owned source of truth for box names.
    """

    room_boxes = {
        str(name): tuple(float(v) for v in box)
        for name, box in (rooms or {}).items()
        if _is_box(box)
    }

    def visited(room: Any) -> bool:
        base = _get_base(agent)
        if base is None:
            return False
        box = room_boxes.get(str(room))
        if box is None:
            return False
        pos = _base_position(base)
        if pos is None:
            return False
        x_min, y_min, x_max, y_max = box
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max

    return visited


def _is_box(box: Any) -> bool:
    """True if *box* coerces to a 4-tuple of floats ``(x_min, y_min, x_max, y_max)``."""
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    return True


def make_rooms_producer(
    rooms: dict[str, tuple[float, float, float, float]],
) -> Callable[..., dict[str, Any]]:
    """Build a rooms PRODUCING-STEP callable over the scenario's named rooms.

    The go2 counterpart of ``arm_sim_oracle.make_detect_producer``: an EXECUTOR
    primitive (NOT a verify predicate) that "locates" the rooms a navigation chain
    should visit and wraps them as a producing step's structured output —
    ``{"rooms": [{"name", "x", "y"}, ...], "count": N}``. The executor captures
    that dict to the run Blackboard under the step name, so a downstream
    ``foreach`` whose ``source_step`` points at this step resolves
    ``source_step.rooms`` to the REAL room list (pure path traversal, never eval).

    Each emitted room carries its NAME (the contract the ``visited(room)`` verify
    predicate reads) plus the centre ``(x, y)`` of its axis-aligned box, so a body
    template can navigate by name (``visited('${room.name}')``) or by coordinate
    (``at_position(${room.x}, ${room.y})``). The room set is the SAME
    scenario-owned source of truth ``visited`` reads, so producer and verifier
    never diverge.

    Deterministic and fail-safe: an empty / malformed room map yields
    ``{"rooms": [], "count": 0}`` — never raises into the executor. Rooms are
    emitted in sorted name order so the produced list is stable across runs.
    """

    room_boxes = {
        str(name): tuple(float(v) for v in box)
        for name, box in (rooms or {}).items()
        if _is_box(box)
    }

    def rooms_producer(**_: Any) -> dict[str, Any]:
        out: list[dict[str, Any]] = []
        for name in sorted(room_boxes):
            x_min, y_min, x_max, y_max = room_boxes[name]
            out.append(
                {
                    "name": name,
                    "x": (x_min + x_max) / 2.0,
                    "y": (y_min + y_max) / 2.0,
                }
            )
        return {"rooms": out, "count": len(out)}

    return rooms_producer
