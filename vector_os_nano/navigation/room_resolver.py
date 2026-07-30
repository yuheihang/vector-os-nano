# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deterministic named-room resolution over the live SceneGraph.

Language aliases live here, but room coordinates never do.  A room is
navigable only when it exists in the supplied SceneGraph and exposes a finite
centre.  Unknown names fail closed with the live set of available room IDs.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from vector_os_nano.navigation.world_mode import WorldMode, get_world_mode


ROOM_ALIASES: dict[str, str] = {
    # English
    "living room": "living_room",
    "living": "living_room",
    "lounge": "living_room",
    "dining room": "dining_room",
    "dining": "dining_room",
    "kitchen": "kitchen",
    "study": "study",
    "office": "study",
    "master bedroom": "master_bedroom",
    "master": "master_bedroom",
    "bedroom": "master_bedroom",
    "guest bedroom": "guest_bedroom",
    "guest room": "guest_bedroom",
    "guest": "guest_bedroom",
    "bathroom": "bathroom",
    "bath": "bathroom",
    "restroom": "bathroom",
    "toilet": "bathroom",
    "hallway": "hallway",
    "hall": "hallway",
    "corridor": "hallway",
    "laundry": "hallway",
    # Chinese
    "客厅": "living_room",
    "大厅": "living_room",
    "餐厅": "dining_room",
    "饭厅": "dining_room",
    "厨房": "kitchen",
    "书房": "study",
    "办公室": "study",
    "工作室": "study",
    "主卧": "master_bedroom",
    "卧室": "master_bedroom",
    "主卧室": "master_bedroom",
    "客卧": "guest_bedroom",
    "客房": "guest_bedroom",
    "次卧": "guest_bedroom",
    "卫生间": "bathroom",
    "浴室": "bathroom",
    "洗手间": "bathroom",
    "厕所": "bathroom",
    "走廊": "hallway",
    "过道": "hallway",
    "大厅走廊": "hallway",
    "洗衣房": "hallway",
}

_EN_PREFIXES: tuple[str, ...] = (
    "please navigate to ",
    "please go to ",
    "navigate to ",
    "take me to ",
    "go to ",
    "the ",
)
_ZH_PREFIXES: tuple[str, ...] = ("请导航到", "请带我去", "导航到", "走到", "去到", "请去", "到", "去")
_STOP_WORDS: frozenset[str] = frozenset({"room", "the", "a", "to", "go"})


@dataclass(frozen=True)
class ResolvedRoom:
    """A room query grounded in the live SceneGraph."""

    requested: str
    canonical: str
    center: tuple[float, float]
    source: str = "scene_graph"
    navigation_goal: tuple[float, float] | None = None

    @property
    def navigation_target(self) -> tuple[float, float]:
        """Executable destination without changing the semantic room centre."""

        return self.navigation_goal or self.center


@dataclass(frozen=True)
class RoomLocation:
    """Room membership at a world coordinate and how it was verified."""

    canonical: str | None
    verification_mode: str


class UnknownRoom(ValueError):
    """Raised when a query cannot be mapped to an available SceneGraph room."""

    def __init__(self, requested: str, available_rooms: Iterable[str] = ()) -> None:
        self.requested = requested
        self.available_rooms = tuple(sorted({str(r) for r in available_rooms}))
        available = ", ".join(self.available_rooms) if self.available_rooms else "none"
        super().__init__(
            f"unknown_room: {requested!r}. Available rooms are [{available}]"
        )


class RoomPositionUnknown(ValueError):
    """Raised when a known room has no usable SceneGraph centre."""

    def __init__(self, canonical: str) -> None:
        self.canonical = canonical
        super().__init__(
            f"room_position_unknown: {canonical!r} has no finite SceneGraph centre"
        )


def normalize_room_query(value: Any) -> str:
    """Normalize Unicode, case, separators, whitespace and common route prefixes."""

    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    for prefix in _EN_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for prefix in _ZH_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


class RoomResolver:
    """Resolve names and membership against one live SceneGraph."""

    def __init__(
        self,
        scene_graph: Any,
        *,
        world_mode: str | WorldMode | None = None,
        room_bounds: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        self._scene_graph = scene_graph
        self._world_mode = get_world_mode(world_mode)
        self._room_bounds = {
            str(name): tuple(values)
            for name, values in (room_bounds or {}).items()
        }

    @property
    def world_mode(self) -> WorldMode:
        return self._world_mode

    def _room_nodes(self) -> tuple[Any, ...]:
        getter = getattr(self._scene_graph, "get_all_rooms", None)
        if not callable(getter):
            return ()
        try:
            rooms = tuple(getter() or ())
        except Exception:
            return ()
        if self._world_mode is WorldMode.KNOWN_LAYOUT:
            return rooms
        # Unknown exploration may only reveal rooms actually discovered online.
        discovered: list[Any] = []
        for room in rooms:
            explicit = getattr(room, "discovered", None)
            if explicit is True or (
                explicit is None and int(getattr(room, "visit_count", 0) or 0) > 0
            ):
                discovered.append(room)
        return tuple(discovered)

    def available_room_ids(self) -> tuple[str, ...]:
        """Sorted canonical room IDs visible in the current world mode."""

        return tuple(
            sorted(
                {
                    str(getattr(room, "room_id"))
                    for room in self._room_nodes()
                    if getattr(room, "room_id", None)
                }
            )
        )

    def canonicalize(self, query: Any) -> str:
        """Map an alias to one available canonical room ID, or raise UnknownRoom."""

        requested = str(query or "")
        key = normalize_room_query(requested)
        available = self.available_room_ids()
        if not key:
            raise UnknownRoom(requested, available)

        canonical_form = key.replace(" ", "_")
        alias_target = ROOM_ALIASES.get(key)
        dynamic_alias_targets: set[str] = set()
        for room in self._room_nodes():
            room_id = str(getattr(room, "room_id", "") or "")
            for alias in getattr(room, "aliases", ()) or ():
                if normalize_room_query(alias) == key and room_id in available:
                    dynamic_alias_targets.add(room_id)
        if len(dynamic_alias_targets) == 1:
            return next(iter(dynamic_alias_targets))
        if len(dynamic_alias_targets) > 1:
            raise UnknownRoom(requested, available)

        for candidate in (alias_target, canonical_form):
            if candidate and candidate in available:
                return candidate

        # Preserve the historical useful matches ("master room", "guest") while
        # rejecting permissive substring guesses such as "bed".  Only complete
        # normalized tokens count, and a match must be unique.
        query_tokens = set(key.split()) - _STOP_WORDS
        matches: list[str] = []
        if query_tokens:
            for room_id in available:
                room_tokens = set(room_id.replace("_", " ").split()) - _STOP_WORDS
                if query_tokens <= room_tokens:
                    matches.append(room_id)
                    continue
                aliases = {
                    alias
                    for alias, target in ROOM_ALIASES.items()
                    if target == room_id
                }
                room = next(
                    (
                        node
                        for node in self._room_nodes()
                        if str(getattr(node, "room_id", "")) == room_id
                    ),
                    None,
                )
                if room is not None:
                    aliases.update(
                        normalize_room_query(alias)
                        for alias in (getattr(room, "aliases", ()) or ())
                    )
                if any(query_tokens <= (set(alias.split()) - _STOP_WORDS) for alias in aliases):
                    matches.append(room_id)
        unique = sorted(set(matches))
        if len(unique) == 1:
            return unique[0]
        raise UnknownRoom(requested, available)

    def resolve(self, query: Any) -> ResolvedRoom:
        """Resolve *query* and read its finite centre from the live SceneGraph."""

        requested = str(query or "")
        canonical = self.canonicalize(requested)
        getter = getattr(self._scene_graph, "get_room", None)
        room = getter(canonical) if callable(getter) else None
        if room is None:
            raise UnknownRoom(requested, self.available_room_ids())
        try:
            x = float(getattr(room, "center_x"))
            y = float(getattr(room, "center_y"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise RoomPositionUnknown(canonical) from exc
        if not (math.isfinite(x) and math.isfinite(y)):
            raise RoomPositionUnknown(canonical)
        navigation_goal = None
        # Authored collision-free goals belong to the known-layout contract.
        # Unknown-world discovery must navigate to what it actually observed,
        # never to a prior accidentally left on a reused test/graph object.
        raw_goal = (
            getattr(room, "navigation_goal", None)
            if self._world_mode is WorldMode.KNOWN_LAYOUT
            else None
        )
        if raw_goal is not None:
            try:
                gx, gy = float(raw_goal[0]), float(raw_goal[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise RoomPositionUnknown(canonical) from exc
            if not (math.isfinite(gx) and math.isfinite(gy)):
                raise RoomPositionUnknown(canonical)
            navigation_goal = (gx, gy)
        source = str(getattr(room, "source", "") or "scene_graph")
        return ResolvedRoom(
            requested=requested,
            canonical=canonical,
            center=(x, y),
            source=source,
            navigation_goal=navigation_goal,
        )

    def locate(self, x: Any, y: Any) -> RoomLocation:
        """Locate a coordinate by room API, polygon/bounds, then nearest centre."""

        try:
            px, py = float(x), float(y)
        except (TypeError, ValueError):
            return RoomLocation(None, "unavailable")
        if not (math.isfinite(px) and math.isfinite(py)):
            return RoomLocation(None, "unavailable")

        # Use one live-room snapshot for the whole lookup so a concurrent
        # SceneGraph update cannot make room_at, geometry, and nearest-centre
        # disagree about which IDs are currently available.
        room_nodes = self._room_nodes()
        available_ids = frozenset(
            str(getattr(room, "room_id"))
            for room in room_nodes
            if getattr(room, "room_id", None)
        )

        room_at = getattr(self._scene_graph, "room_at", None)
        if callable(room_at):
            try:
                hit = room_at(px, py)
            except Exception:
                hit = None
            room_id = _room_id(hit)
            if room_id in available_ids:
                return RoomLocation(room_id, "room_at")

        polygon_hits: list[str] = []
        bounds_hits: list[str] = []
        rooms_with_polygon: set[str] = set()
        rooms_with_geometry: set[str] = set()
        for room in room_nodes:
            room_id = str(getattr(room, "room_id", ""))
            polygon = _room_polygon(room)
            if polygon:
                rooms_with_geometry.add(room_id)
                rooms_with_polygon.add(room_id)
                if _point_in_polygon(px, py, polygon):
                    polygon_hits.append(room_id)
                # A polygon is more precise than its axis-aligned envelope.  Do
                # not let a room's optional bounds turn a miss in a concave
                # polygon into a false hit.
                continue
            bounds = _room_bounds(room)
            if bounds is not None:
                rooms_with_geometry.add(room_id)
                if _point_in_bounds(px, py, bounds):
                    bounds_hits.append(room_id)

        for room_id, bounds in self._room_bounds.items():
            if room_id not in available_ids or room_id in rooms_with_polygon:
                continue
            parsed = _parse_bounds(bounds)
            if parsed is None:
                continue
            rooms_with_geometry.add(room_id)
            if _point_in_bounds(px, py, parsed):
                bounds_hits.append(room_id)

        if polygon_hits:
            return RoomLocation(sorted(set(polygon_hits))[0], "polygon")
        if bounds_hits:
            # Shared boundaries have deterministic ownership.
            return RoomLocation(sorted(set(bounds_hits))[0], "bounds")
        # Geometry is authoritative for every room that owns it.  Rooms whose
        # geometry has not been learned yet may still use the documented
        # nearest-centre downgrade; restrict that fallback to those rooms so an
        # exterior point can never be assigned back through a polygon/bounds wall.
        fallback_centres: list[tuple[float, str]] = []
        for room in room_nodes:
            room_id = str(getattr(room, "room_id", ""))
            if not room_id or room_id in rooms_with_geometry:
                continue
            try:
                cx = float(getattr(room, "center_x"))
                cy = float(getattr(room, "center_y"))
            except (AttributeError, TypeError, ValueError):
                continue
            if math.isfinite(cx) and math.isfinite(cy):
                fallback_centres.append((math.hypot(cx - px, cy - py), room_id))
        if fallback_centres:
            _, room_id = min(fallback_centres, key=lambda item: (item[0], item[1]))
            return RoomLocation(room_id, "nearest_center")
        if rooms_with_geometry:
            return RoomLocation(None, "geometry")
        return RoomLocation(None, "unavailable")


def _room_id(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        raw = value.get("room_id") or value.get("id")
        return str(raw) if raw is not None else None
    raw = getattr(value, "room_id", None)
    return str(raw) if raw is not None else None


def _room_polygon(room: Any) -> tuple[tuple[float, float], ...]:
    for attr in ("polygon", "boundary", "vertices"):
        raw = getattr(room, attr, None)
        if not raw:
            continue
        try:
            points = tuple((float(p[0]), float(p[1])) for p in raw)
        except (TypeError, ValueError, IndexError):
            continue
        if len(points) >= 3:
            return points
    return ()


def _room_bounds(room: Any) -> tuple[float, float, float, float] | None:
    for attr in ("bounds", "bbox", "aabb"):
        raw = getattr(room, attr, None)
        if raw is None:
            continue
        parsed = _parse_bounds(raw)
        if parsed is not None:
            return parsed
    return None


def _parse_bounds(values: Sequence[float]) -> tuple[float, float, float, float] | None:
    try:
        x_min, y_min, x_max, y_max = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x_min, y_min, x_max, y_max)):
        return None
    if x_min > x_max or y_min > y_max:
        return None
    return x_min, y_min, x_max, y_max


def _point_in_bounds(
    x: float, y: float, values: Sequence[float]
) -> bool:
    bounds = _parse_bounds(values)
    if bounds is None:
        return False
    x_min, y_min, x_max, y_max = bounds
    return x_min <= x <= x_max and y_min <= y <= y_max


def _point_in_polygon(
    x: float, y: float, polygon: Sequence[tuple[float, float]]
) -> bool:
    # Boundary counts as inside.  The later sorted tie-break provides
    # deterministic ownership when two room polygons share an edge.
    for idx, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9:
            if min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9:
                return True

    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x <= x_cross:
                inside = not inside
        j = i
    return inside
