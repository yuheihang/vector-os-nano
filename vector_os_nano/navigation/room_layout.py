# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Validated room-layout configuration for deterministic indoor navigation.

Schema v1 (legacy) stores only room and door centres.  It remains readable for
old persisted fixtures, but it does not claim to be an executable safety
topology.  Schema v2 adds room polygons and enough door geometry to construct a
pre/centre/post crossing without asking an LLM or planner to infer wall gaps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


DEFAULT_FOOTPRINT_WIDTH_M = 0.55
DEFAULT_DOOR_CLEARANCE_M = 0.15
DEFAULT_STANDOFF_DISTANCE_M = 0.60
_EPS = 1e-6


class RoomLayoutError(ValueError):
    """Raised when a layout cannot be used without unsafe inference."""


@dataclass(frozen=True)
class RoomLayoutRoom:
    """One room prior parsed from ``room_layout.yaml``."""

    room_id: str
    center: tuple[float, float]
    polygon: tuple[tuple[float, float], ...] = ()
    aliases: tuple[str, ...] = ()
    source: str = "layout_prior"
    # A semantic room centre is often occupied by representative furniture
    # (for example, the dining table).  Keep that centre for labels and room
    # membership, while giving navigation an explicitly authored free-space
    # destination when one is available.
    navigation_goal: tuple[float, float] | None = None


@dataclass(frozen=True)
class RoomLayoutDoor:
    """One oriented doorway prior.

    ``normal`` points from ``room_a`` to ``room_b``.  The two standoffs must
    therefore have negative and positive signed distances from the centre,
    respectively.
    """

    door_id: str
    room_a: str
    room_b: str
    center: tuple[float, float]
    width: float | None = None
    normal: tuple[float, float] | None = None
    room_a_standoff: tuple[float, float] | None = None
    room_b_standoff: tuple[float, float] | None = None
    source: str = "layout_prior"
    confidence: float = 1.0

    @property
    def executable(self) -> bool:
        return (
            self.width is not None
            and self.normal is not None
            and self.room_a_standoff is not None
            and self.room_b_standoff is not None
        )


@dataclass(frozen=True)
class RoomLayout:
    """A fully parsed layout, committed atomically by ``SceneGraph``."""

    schema_version: int
    rooms: tuple[RoomLayoutRoom, ...]
    doors: tuple[RoomLayoutDoor, ...]
    footprint_width_m: float = DEFAULT_FOOTPRINT_WIDTH_M
    door_clearance_m: float = DEFAULT_DOOR_CLEARANCE_M

    @property
    def executable(self) -> bool:
        return (
            self.schema_version >= 2
            and all(room.polygon for room in self.rooms)
            and all(door.executable for door in self.doors)
        )


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise RoomLayoutError(f"{field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RoomLayoutError(f"{field} must be a finite number") from exc
    if not math.isfinite(number):
        raise RoomLayoutError(f"{field} must be a finite number")
    return number


def _point(value: Any, *, field: str) -> tuple[float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 2
    ):
        raise RoomLayoutError(f"{field} must be [x, y]")
    return (
        _finite_number(value[0], field=f"{field}[0]"),
        _finite_number(value[1], field=f"{field}[1]"),
    )


def _polygon(value: Any, *, field: str) -> tuple[tuple[float, float], ...]:
    if value in (None, ()):
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RoomLayoutError(f"{field} must be a list of at least three [x, y] points")
    points = tuple(_point(item, field=f"{field}[{idx}]") for idx, item in enumerate(value))
    if len(points) < 3:
        raise RoomLayoutError(f"{field} must contain at least three points")
    area2 = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area2 += x1 * y2 - x2 * y1
    if abs(area2) <= _EPS:
        raise RoomLayoutError(f"{field} has zero area")
    return points


def _aliases(value: Any, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise RoomLayoutError(f"{field} must be a list of strings")
    result: list[str] = []
    for idx, alias in enumerate(value):
        if not isinstance(alias, str) or not alias.strip():
            raise RoomLayoutError(f"{field}[{idx}] must be a non-empty string")
        normalized = alias.strip()
        if normalized not in result:
            result.append(normalized)
    return tuple(result)


def _door_rooms(door_id: str, raw: Any) -> tuple[str, str]:
    explicit = raw.get("rooms") if isinstance(raw, Mapping) else None
    if explicit is not None:
        if (
            isinstance(explicit, (str, bytes))
            or not isinstance(explicit, Sequence)
            or len(explicit) != 2
        ):
            raise RoomLayoutError(f"doors.{door_id}.rooms must contain two room IDs")
        room_a, room_b = str(explicit[0]).strip(), str(explicit[1]).strip()
    else:
        parts = door_id.split("-", 1)
        if len(parts) != 2:
            raise RoomLayoutError(
                f"door key {door_id!r} must be 'room_a-room_b' or define rooms"
            )
        room_a, room_b = parts[0].strip(), parts[1].strip()
    if not room_a or not room_b or room_a == room_b:
        raise RoomLayoutError(f"door {door_id!r} must connect two distinct rooms")
    return room_a, room_b


def _unit_normal(value: Any, *, field: str) -> tuple[float, float]:
    nx, ny = _point(value, field=field)
    length = math.hypot(nx, ny)
    if length <= _EPS:
        raise RoomLayoutError(f"{field} must be non-zero")
    return (nx / length, ny / length)


def _signed_side(
    point: tuple[float, float],
    center: tuple[float, float],
    normal: tuple[float, float],
) -> float:
    return (
        (point[0] - center[0]) * normal[0]
        + (point[1] - center[1]) * normal[1]
    )


def point_in_polygon(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
) -> bool:
    """Return whether *point* is inside or on the boundary of *polygon*."""

    px, py = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)
        if (
            abs(cross) <= _EPS
            and min(x1, x2) - _EPS <= px <= max(x1, x2) + _EPS
            and min(y1, y2) - _EPS <= py <= max(y1, y2) + _EPS
        ):
            return True
        if (y1 > py) != (y2 > py):
            crossing_x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < crossing_x:
                inside = not inside
    return inside


def point_on_polygon_boundary(
    point: tuple[float, float],
    polygon: Sequence[tuple[float, float]],
    *,
    tolerance: float = _EPS,
) -> bool:
    """Return whether *point* lies on a polygon edge within *tolerance*."""

    px, py = point
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        dx, dy = x2 - x1, y2 - y1
        length_sq = dx * dx + dy * dy
        if length_sq <= _EPS * _EPS:
            continue
        projection = ((px - x1) * dx + (py - y1) * dy) / length_sq
        projection = min(1.0, max(0.0, projection))
        closest_x = x1 + projection * dx
        closest_y = y1 + projection * dy
        if math.hypot(px - closest_x, py - closest_y) <= tolerance:
            return True
    return False


def _parse_room(room_id: str, raw: Any, *, schema_version: int) -> RoomLayoutRoom:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
        return RoomLayoutRoom(room_id=room_id, center=_point(raw, field=f"rooms.{room_id}"))
    if not isinstance(raw, Mapping):
        raise RoomLayoutError(
            f"rooms.{room_id} must be [x, y] or a schema-v2 mapping"
        )
    if "center" not in raw:
        raise RoomLayoutError(f"rooms.{room_id}.center is required")
    polygon = _polygon(raw.get("polygon"), field=f"rooms.{room_id}.polygon")
    if schema_version >= 2 and not polygon:
        raise RoomLayoutError(f"rooms.{room_id}.polygon is required in schema v2")
    center = _point(raw["center"], field=f"rooms.{room_id}.center")
    if polygon and not point_in_polygon(center, polygon):
        raise RoomLayoutError(
            f"rooms.{room_id}.center must lie inside its polygon"
        )
    navigation_goal = (
        _point(
            raw["navigation_goal"],
            field=f"rooms.{room_id}.navigation_goal",
        )
        if raw.get("navigation_goal") is not None
        else None
    )
    if (
        navigation_goal is not None
        and polygon
        and (
            not point_in_polygon(navigation_goal, polygon)
            or point_on_polygon_boundary(navigation_goal, polygon)
        )
    ):
        raise RoomLayoutError(
            f"rooms.{room_id}.navigation_goal must lie strictly inside its polygon"
        )
    return RoomLayoutRoom(
        room_id=room_id,
        center=center,
        polygon=polygon,
        aliases=_aliases(raw.get("aliases"), field=f"rooms.{room_id}.aliases"),
        source=str(raw.get("source", "layout_prior") or "layout_prior"),
        navigation_goal=navigation_goal,
    )


def _parse_door(
    door_id: str,
    raw: Any,
    *,
    schema_version: int,
    room_ids: set[str],
    room_polygons: Mapping[str, Sequence[tuple[float, float]]],
) -> RoomLayoutDoor:
    room_a, room_b = _door_rooms(door_id, raw)
    missing = [room for room in (room_a, room_b) if room not in room_ids]
    if missing:
        raise RoomLayoutError(
            f"door {door_id!r} references unknown room(s): {', '.join(missing)}"
        )

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
        return RoomLayoutDoor(
            door_id=door_id,
            room_a=room_a,
            room_b=room_b,
            center=_point(raw, field=f"doors.{door_id}"),
        )
    if not isinstance(raw, Mapping):
        raise RoomLayoutError(
            f"doors.{door_id} must be [x, y] or a schema-v2 mapping"
        )
    if "center" not in raw:
        raise RoomLayoutError(f"doors.{door_id}.center is required")

    center = _point(raw["center"], field=f"doors.{door_id}.center")
    if schema_version < 2:
        return RoomLayoutDoor(
            door_id=door_id,
            room_a=room_a,
            room_b=room_b,
            center=center,
        )

    width = _finite_number(raw.get("width"), field=f"doors.{door_id}.width")
    if width <= 0:
        raise RoomLayoutError(f"doors.{door_id}.width must be positive")
    normal = _unit_normal(raw.get("normal"), field=f"doors.{door_id}.normal")
    room_a_standoff = _point(
        raw.get("room_a_standoff"),
        field=f"doors.{door_id}.room_a_standoff",
    )
    room_b_standoff = _point(
        raw.get("room_b_standoff"),
        field=f"doors.{door_id}.room_b_standoff",
    )
    side_a = _signed_side(room_a_standoff, center, normal)
    side_b = _signed_side(room_b_standoff, center, normal)
    if not (side_a < -_EPS and side_b > _EPS):
        raise RoomLayoutError(
            f"door {door_id!r} standoffs must lie on opposite normal sides "
            f"(room_a negative, room_b positive)"
        )
    polygon_a = room_polygons.get(room_a, ())
    polygon_b = room_polygons.get(room_b, ())
    if polygon_a and not point_in_polygon(room_a_standoff, polygon_a):
        raise RoomLayoutError(
            f"doors.{door_id}.room_a_standoff must lie inside room {room_a!r}"
        )
    if polygon_b and not point_in_polygon(room_b_standoff, polygon_b):
        raise RoomLayoutError(
            f"doors.{door_id}.room_b_standoff must lie inside room {room_b!r}"
        )
    if (
        polygon_a
        and polygon_b
        and (
            not point_on_polygon_boundary(center, polygon_a)
            or not point_on_polygon_boundary(center, polygon_b)
        )
    ):
        raise RoomLayoutError(
            f"doors.{door_id}.center must lie on the shared room boundary"
        )
    tangent = (-normal[1], normal[0])
    lateral_a = abs(
        (room_a_standoff[0] - center[0]) * tangent[0]
        + (room_a_standoff[1] - center[1]) * tangent[1]
    )
    lateral_b = abs(
        (room_b_standoff[0] - center[0]) * tangent[0]
        + (room_b_standoff[1] - center[1]) * tangent[1]
    )
    if max(lateral_a, lateral_b) > width * 0.5 + _EPS:
        raise RoomLayoutError(
            f"door {door_id!r} standoffs must stay within the doorway corridor"
        )
    confidence = _finite_number(
        raw.get("confidence", 1.0),
        field=f"doors.{door_id}.confidence",
    )
    if not 0.0 <= confidence <= 1.0:
        raise RoomLayoutError(f"doors.{door_id}.confidence must be in [0, 1]")
    return RoomLayoutDoor(
        door_id=door_id,
        room_a=room_a,
        room_b=room_b,
        center=center,
        width=width,
        normal=normal,
        room_a_standoff=room_a_standoff,
        room_b_standoff=room_b_standoff,
        source=str(raw.get("source", "layout_prior") or "layout_prior"),
        confidence=confidence,
    )


def load_room_layout(path: str | Path) -> RoomLayout:
    """Parse and validate a v1/v2 layout without mutating a SceneGraph."""

    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, Mapping):
        raise RoomLayoutError("layout root must be a mapping")

    try:
        schema_version = int(raw.get("schema_version", 1))
    except (TypeError, ValueError) as exc:
        raise RoomLayoutError("schema_version must be an integer") from exc
    if schema_version not in (1, 2):
        raise RoomLayoutError(f"unsupported room layout schema_version={schema_version}")

    rooms_raw = raw.get("rooms")
    doors_raw = raw.get("doors", {})
    if not isinstance(rooms_raw, Mapping) or not rooms_raw:
        raise RoomLayoutError("rooms must be a non-empty mapping")
    if not isinstance(doors_raw, Mapping):
        raise RoomLayoutError("doors must be a mapping")

    rooms = tuple(
        _parse_room(str(room_id), room_raw, schema_version=schema_version)
        for room_id, room_raw in rooms_raw.items()
    )
    room_ids = {room.room_id for room in rooms}
    room_polygons = {room.room_id: room.polygon for room in rooms}
    if len(room_ids) != len(rooms):
        raise RoomLayoutError("room IDs must be unique")
    doors = tuple(
        _parse_door(
            str(door_id),
            door_raw,
            schema_version=schema_version,
            room_ids=room_ids,
            room_polygons=room_polygons,
        )
        for door_id, door_raw in doors_raw.items()
    )

    robot = raw.get("robot", {})
    if robot is None:
        robot = {}
    if not isinstance(robot, Mapping):
        raise RoomLayoutError("robot must be a mapping")
    footprint = _finite_number(
        robot.get("footprint_width_m", DEFAULT_FOOTPRINT_WIDTH_M),
        field="robot.footprint_width_m",
    )
    clearance = _finite_number(
        robot.get("door_clearance_m", DEFAULT_DOOR_CLEARANCE_M),
        field="robot.door_clearance_m",
    )
    if footprint <= 0 or clearance < 0:
        raise RoomLayoutError(
            "robot footprint_width_m must be positive and door_clearance_m non-negative"
        )

    if schema_version >= 2:
        duplicate_pairs: set[tuple[str, str]] = set()
        for door in doors:
            key = tuple(sorted((door.room_a, door.room_b)))
            if key in duplicate_pairs:
                raise RoomLayoutError(
                    f"duplicate door connection: {door.room_a}-{door.room_b}"
                )
            duplicate_pairs.add(key)

    return RoomLayout(
        schema_version=schema_version,
        rooms=rooms,
        doors=doors,
        footprint_width_m=footprint,
        door_clearance_m=clearance,
    )
