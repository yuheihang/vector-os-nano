# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Three-layer scene graph for spatial memory (SysNav-inspired).

Replaces flat SpatialMemory with a hierarchical representation:

    RoomNode  →  ViewpointNode  →  ObjectNode

Room nodes represent semantically meaningful spaces (kitchen, bedroom, etc.).
Viewpoint nodes are discrete camera positions within a room, each with a VLM
scene description and coverage area.  Object nodes are individual items
detected by the VLM, with category, position, and on-demand attributes.

Backward-compatible: implements the same public API as SpatialMemory
(visit, observe, get_visited_rooms, get_room_summary, etc.) so existing
skills work unchanged.

Reference: SysNav (arxiv 2603.06914v1) — three-layer scene representation
with VLM-guided room-level reasoning.

No ROS2 dependency.  Thread-safe via threading.Lock.
"""
from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import yaml

from vector_os_nano.navigation.room_layout import (
    DEFAULT_DOOR_CLEARANCE_M,
    DEFAULT_FOOTPRINT_WIDTH_M,
    RoomLayoutError,
    load_room_layout,
    point_in_polygon,
    point_on_polygon_boundary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TextLLM Protocol — narrow injectable adapter for text-only LLM calls.
# core/ depends ONLY on this Protocol; no vcli import needed.
# ---------------------------------------------------------------------------


class TextLLM(Protocol):
    """Minimal protocol for a text-generation adapter.

    Implementations live outside core/ (e.g. vcli/backends/text_llm_adapter.py)
    so that core/ never imports vcli.  Stubs can be used in tests.
    """

    def complete_text(self, prompt: str) -> str:
        """Send *prompt* to the language model and return the text response.

        Args:
            prompt: The full user-side prompt string.

        Returns:
            The model's text response as a string.
        """
        ...


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VIEWPOINT_MIN_DISTANCE: float = 1.5   # meters — minimum distance between viewpoints
_VIEWPOINT_FOV_DEG: float = 60.0       # camera horizontal FOV (degrees)
_VIEWPOINT_RANGE: float = 3.0          # meters — max observation range
_ROOM_AREA_DEFAULT: float = 15.0       # m² — fallback room area for coverage calc
_MAX_EVENTS: int = 200


# ---------------------------------------------------------------------------
# Node dataclasses (frozen — immutable after construction)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectNode:
    """An object detected by VLM in the scene."""

    object_id: str
    category: str                       # "chair", "sofa", "fridge"
    description: str = ""               # VLM description
    confidence: float = 0.8
    room_id: str = ""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    attributes: dict = field(default_factory=dict)  # {"color": "red"} — queried on demand
    viewpoint_ids: tuple[str, ...] = ()
    first_seen: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "room_id": self.room_id,
            "x": self.x, "y": self.y, "z": self.z,
            "attributes": dict(self.attributes),
            "viewpoint_ids": list(self.viewpoint_ids),
            "first_seen": self.first_seen,
        }


@dataclass(frozen=True)
class ViewpointNode:
    """A camera viewpoint within a room."""

    viewpoint_id: str
    room_id: str
    x: float
    y: float
    heading: float = 0.0                # radians
    timestamp: float = field(default_factory=time.time)
    scene_summary: str = ""             # VLM scene description
    object_ids: tuple[str, ...] = ()    # objects visible from here
    frame_b64: str = ""                 # optional cached frame (base64 jpeg)

    @property
    def coverage_area(self) -> float:
        """Estimated coverage area in m² (FOV cone approximation)."""
        half_angle = math.radians(_VIEWPOINT_FOV_DEG / 2)
        return 0.5 * _VIEWPOINT_RANGE**2 * math.sin(2 * half_angle)

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewpoint_id": self.viewpoint_id,
            "room_id": self.room_id,
            "x": self.x, "y": self.y,
            "heading": self.heading,
            "timestamp": self.timestamp,
            "scene_summary": self.scene_summary,
            "object_ids": list(self.object_ids),
        }


@dataclass(frozen=True)
class RoomNode:
    """A room in the house."""

    room_id: str
    center_x: float = 0.0
    center_y: float = 0.0
    area: float = _ROOM_AREA_DEFAULT
    visit_count: int = 0
    last_visited: float = 0.0
    representative_description: str = ""
    connected_rooms: tuple[str, ...] = ()
    polygon: tuple[tuple[float, float], ...] = ()
    aliases: tuple[str, ...] = ()
    source: str = "observed"
    confidence: float = 0.8
    # Optional collision-free destination authored independently of the
    # semantic centre.  Appended to preserve RoomNode's positional API.
    navigation_goal: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_id": self.room_id,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
            "visit_count": self.visit_count,
            "last_visited": self.last_visited,
            "representative_description": self.representative_description,
            "connected_rooms": list(self.connected_rooms),
            "polygon": [list(point) for point in self.polygon],
            "aliases": list(self.aliases),
            "source": self.source,
            "confidence": self.confidence,
            "navigation_goal": (
                list(self.navigation_goal)
                if self.navigation_goal is not None
                else None
            ),
        }


@dataclass(frozen=True)
class DoorEdge:
    """A bidirectional doorway with oriented safety geometry."""

    door_id: str
    room_a: str
    room_b: str
    center_x: float
    center_y: float
    width: float | None = None
    normal_x: float | None = None
    normal_y: float | None = None
    room_a_standoff: tuple[float, float] | None = None
    room_b_standoff: tuple[float, float] | None = None
    source: str = "observed"
    confidence: float = 0.6
    observation_count: int = 1
    last_observed_center: tuple[float, float] | None = None
    last_observed_confidence: float | None = None

    @property
    def executable(self) -> bool:
        """Whether this edge contains self-consistent crossing geometry.

        Layout parsing already rejects malformed door geometry, but rich door
        records can also arrive through persistence or online observation.
        Rechecking here ensures a corrupt/stale record can never become a
        planner edge merely because all optional fields happen to be present.
        """

        if (
            self.width is None
            or self.normal_x is None
            or self.normal_y is None
            or self.room_a_standoff is None
            or self.room_b_standoff is None
        ):
            return False
        try:
            (
                center_x,
                center_y,
                width,
                normal_x,
                normal_y,
                standoff_ax,
                standoff_ay,
                standoff_bx,
                standoff_by,
            ) = tuple(
                float(value)
                for value in (
                    self.center_x,
                    self.center_y,
                    self.width,
                    self.normal_x,
                    self.normal_y,
                    *self.room_a_standoff,
                    *self.room_b_standoff,
                )
            )
        except (TypeError, ValueError, OverflowError):
            return False
        values = (
            center_x,
            center_y,
            width,
            normal_x,
            normal_y,
            standoff_ax,
            standoff_ay,
            standoff_bx,
            standoff_by,
        )
        if not all(math.isfinite(value) for value in values) or width <= 0:
            return False
        normal_length = math.hypot(normal_x, normal_y)
        if normal_length <= 1e-9:
            return False
        normal_x /= normal_length
        normal_y /= normal_length
        side_a = (
            (standoff_ax - center_x) * normal_x
            + (standoff_ay - center_y) * normal_y
        )
        side_b = (
            (standoff_bx - center_x) * normal_x
            + (standoff_by - center_y) * normal_y
        )
        tangent_x, tangent_y = -normal_y, normal_x
        lateral_a = abs(
            (standoff_ax - center_x) * tangent_x
            + (standoff_ay - center_y) * tangent_y
        )
        lateral_b = abs(
            (standoff_bx - center_x) * tangent_x
            + (standoff_by - center_y) * tangent_y
        )
        return (
            side_a < -1e-9
            and side_b > 1e-9
            and max(lateral_a, lateral_b) <= width * 0.5 + 1e-9
        )

    def standoff_for(self, room_id: str) -> tuple[float, float] | None:
        if room_id == self.room_a:
            return self.room_a_standoff
        if room_id == self.room_b:
            return self.room_b_standoff
        return None

    def other_room(self, room_id: str) -> str | None:
        if room_id == self.room_a:
            return self.room_b
        if room_id == self.room_b:
            return self.room_a
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "door_id": self.door_id,
            "room_a": self.room_a,
            "room_b": self.room_b,
            "center": [self.center_x, self.center_y],
            "width": self.width,
            "normal": (
                [self.normal_x, self.normal_y]
                if self.normal_x is not None and self.normal_y is not None
                else None
            ),
            "room_a_standoff": (
                list(self.room_a_standoff)
                if self.room_a_standoff is not None
                else None
            ),
            "room_b_standoff": (
                list(self.room_b_standoff)
                if self.room_b_standoff is not None
                else None
            ),
            "source": self.source,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "last_observed_center": (
                list(self.last_observed_center)
                if self.last_observed_center is not None
                else None
            ),
            "last_observed_confidence": self.last_observed_confidence,
        }


@dataclass(frozen=True)
class RouteWaypoint:
    """One planner segment in a safe room-to-room route.

    ``speed_limit_mps=None`` deliberately delegates speed selection to the
    normal adaptive path follower; the remaining segment policy still carries
    reverse permission and the required arrival tolerance.
    """

    kind: str
    room_from: str
    room_to: str
    xy: tuple[float, float]
    tolerance: float
    speed_limit_mps: float | None
    allow_reverse: bool
    door_id: str | None = None

    @property
    def label(self) -> str:
        if self.door_id:
            return f"{self.door_id}:{self.kind}"
        return self.room_to

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "room_from": self.room_from,
            "room_to": self.room_to,
            "xy": [self.xy[0], self.xy[1]],
            "tolerance": self.tolerance,
            "speed_limit_mps": self.speed_limit_mps,
            "allow_reverse": self.allow_reverse,
            "door_id": self.door_id,
            "label": self.label,
        }


@dataclass(frozen=True)
class DoorRoute:
    """Structured result of topology planning."""

    src_room: str
    dst_room: str
    room_path: tuple[str, ...] = ()
    door_ids: tuple[str, ...] = ()
    waypoints: tuple[RouteWaypoint, ...] = ()
    diagnosis_code: str = ""
    message: str = ""
    required_width_m: float = 0.0

    @property
    def success(self) -> bool:
        return not self.diagnosis_code and bool(self.waypoints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "src_room": self.src_room,
            "dst_room": self.dst_room,
            "room_path": list(self.room_path),
            "door_ids": list(self.door_ids),
            "waypoints": [waypoint.to_dict() for waypoint in self.waypoints],
            "diagnosis_code": self.diagnosis_code,
            "message": self.message,
            "required_width_m": self.required_width_m,
        }


# ---------------------------------------------------------------------------
# SceneGraph
# ---------------------------------------------------------------------------


class SceneGraph:
    """Three-layer hierarchical scene graph.

    Layers:
        rooms      — dict[room_id, RoomNode]
        viewpoints — dict[viewpoint_id, ViewpointNode]
        objects    — dict[object_id, ObjectNode]

    Backward-compatible with SpatialMemory: visit(), observe(),
    get_visited_rooms(), get_room_summary(), etc.

    Thread-safe: all mutations guarded by a single Lock.
    """

    def __init__(self, persist_path: str | None = None) -> None:
        self._rooms: dict[str, RoomNode] = {}
        self._viewpoints: dict[str, ViewpointNode] = {}
        self._objects: dict[str, ObjectNode] = {}
        # key = tuple(sorted([room_a, room_b])) — bidirectional lookup.
        # DoorEdge retains its declared orientation so normal/standoff semantics
        # are reversible without losing which point belongs to which room.
        self._doors: dict[tuple[str, str], DoorEdge] = {}
        self._layout_schema_version: int = 0
        self._layout_executable: bool = False
        self._robot_footprint_width_m: float = DEFAULT_FOOTPRINT_WIDTH_M
        self._door_clearance_m: float = DEFAULT_DOOR_CLEARANCE_M
        self._last_layout_error: str = ""
        self._events: list[dict] = []
        self._lock = threading.RLock()
        self._persist_path = persist_path

    # ------------------------------------------------------------------
    # Room operations
    # ------------------------------------------------------------------

    def add_room(self, room: RoomNode) -> None:
        with self._lock:
            self._rooms[room.room_id] = room

    def get_room(self, room_id: str) -> RoomNode | None:
        with self._lock:
            return self._rooms.get(room_id)

    def get_all_rooms(self) -> list[RoomNode]:
        with self._lock:
            return list(self._rooms.values())

    # ------------------------------------------------------------------
    # Door operations
    # ------------------------------------------------------------------

    @staticmethod
    def _door_key(room_a: str, room_b: str) -> tuple[str, str]:
        return tuple(sorted((str(room_a), str(room_b))))  # type: ignore[return-value]

    @property
    def layout_schema_version(self) -> int:
        return self._layout_schema_version

    @property
    def has_executable_layout(self) -> bool:
        return self._layout_executable

    @property
    def last_layout_error(self) -> str:
        return self._last_layout_error

    def navigation_profile(self) -> dict[str, float | int | bool]:
        return {
            "schema_version": self._layout_schema_version,
            "executable": self._layout_executable,
            "footprint_width_m": self._robot_footprint_width_m,
            "door_clearance_m": self._door_clearance_m,
            "required_door_width_m": (
                self._robot_footprint_width_m + 2.0 * self._door_clearance_m
            ),
        }

    def _connect_rooms(self, room_a: str, room_b: str) -> None:
        """Update bidirectional room adjacency. Caller holds ``_lock``."""

        for src, dst in ((room_a, room_b), (room_b, room_a)):
            room = self._rooms.get(src)
            if room is None:
                self._rooms[src] = RoomNode(
                    room_id=src,
                    connected_rooms=(dst,),
                )
            elif dst not in room.connected_rooms:
                self._rooms[src] = replace(
                    room,
                    connected_rooms=room.connected_rooms + (dst,),
                )

    def add_door(
        self,
        room_a: str,
        room_b: str,
        x: float,
        y: float,
        *,
        door_id: str | None = None,
        width: float | None = None,
        normal: tuple[float, float] | None = None,
        room_a_standoff: tuple[float, float] | None = None,
        room_b_standoff: tuple[float, float] | None = None,
        source: str = "observed",
        confidence: float = 0.6,
        authoritative: bool = False,
    ) -> None:
        """Record a doorway without silently erasing a stronger layout prior.

        Legacy callers may still provide only ``(room_a, room_b, x, y)``. Such
        an observation remains useful for the compatibility door-chain API but
        is deliberately not executable by the strict schema-v2 route planner.
        """

        room_a, room_b = str(room_a), str(room_b)
        if not room_a or not room_b or room_a == room_b:
            raise ValueError("a door must connect two distinct room IDs")
        x, y, confidence = float(x), float(y), float(confidence)
        if not all(math.isfinite(value) for value in (x, y, confidence)):
            raise ValueError("door coordinates/confidence must be finite")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("door confidence must be in [0, 1]")
        if width is not None:
            width = float(width)
            if not math.isfinite(width) or width <= 0:
                raise ValueError("door width must be finite and positive")

        key = self._door_key(room_a, room_b)
        with self._lock:
            existing = self._doors.get(key)
            observation_count = (
                existing.observation_count + 1
                if existing is not None and source == "observed"
                else (existing.observation_count if existing is not None else 1)
            )

            if (
                existing is not None
                and existing.source == "layout_prior"
                and source == "observed"
                and not authoritative
            ):
                # Retain the deterministic prior, but record that an online
                # observation occurred and where it landed.
                self._doors[key] = replace(
                    existing,
                    observation_count=observation_count,
                    last_observed_center=(x, y),
                    last_observed_confidence=confidence,
                )
            elif (
                existing is not None
                and source == "observed"
                and existing.source == "observed"
                and not authoritative
            ):
                n = max(1, existing.observation_count)
                self._doors[key] = replace(
                    existing,
                    center_x=(existing.center_x * n + x) / (n + 1),
                    center_y=(existing.center_y * n + y) / (n + 1),
                    confidence=max(existing.confidence, confidence),
                    observation_count=n + 1,
                    last_observed_center=(x, y),
                    last_observed_confidence=confidence,
                )
            else:
                self._doors[key] = DoorEdge(
                    door_id=door_id or f"{room_a}-{room_b}",
                    room_a=room_a,
                    room_b=room_b,
                    center_x=x,
                    center_y=y,
                    width=width,
                    normal_x=(float(normal[0]) if normal is not None else None),
                    normal_y=(float(normal[1]) if normal is not None else None),
                    room_a_standoff=room_a_standoff,
                    room_b_standoff=room_b_standoff,
                    source=str(source or "observed"),
                    confidence=confidence,
                    observation_count=observation_count,
                    last_observed_center=(
                        existing.last_observed_center if existing is not None else None
                    ),
                    last_observed_confidence=(
                        existing.last_observed_confidence
                        if existing is not None
                        else None
                    ),
                )
            self._connect_rooms(room_a, room_b)

    def get_door(self, room_a: str, room_b: str) -> tuple[float, float] | None:
        """Return the compatibility ``(x, y)`` centre for one door."""

        with self._lock:
            edge = self._doors.get(self._door_key(room_a, room_b))
            return (
                (edge.center_x, edge.center_y)
                if edge is not None
                else None
            )

    def get_door_edge(self, room_a: str, room_b: str) -> DoorEdge | None:
        with self._lock:
            return self._doors.get(self._door_key(room_a, room_b))

    def get_all_doors(self) -> dict[tuple[str, str], tuple[float, float]]:
        """Compatibility view of all door centres."""

        with self._lock:
            return {
                key: (edge.center_x, edge.center_y)
                for key, edge in self._doors.items()
            }

    def get_all_door_edges(self) -> dict[tuple[str, str], DoorEdge]:
        with self._lock:
            return dict(self._doors)

    def _find_room_path(
        self,
        src_room: str,
        dst_room: str,
        *,
        edge_allowed: Any,
    ) -> tuple[str, ...]:
        """Deterministic BFS over actual DoorEdges. Caller holds ``_lock``."""

        if src_room not in self._rooms or dst_room not in self._rooms:
            return ()
        if src_room == dst_room:
            return (src_room,)
        adjacency: dict[str, list[str]] = {}
        for edge in self._doors.values():
            if not edge_allowed(edge):
                continue
            adjacency.setdefault(edge.room_a, []).append(edge.room_b)
            adjacency.setdefault(edge.room_b, []).append(edge.room_a)
        queue: deque[str] = deque([src_room])
        parent: dict[str, str | None] = {src_room: None}
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in parent:
                    continue
                parent[neighbor] = current
                if neighbor == dst_room:
                    path: list[str] = [dst_room]
                    node = current
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    return tuple(reversed(path))
                queue.append(neighbor)
        return ()

    def _edge_is_trusted_for_route(self, edge: DoorEdge) -> bool:
        """Reject online shortcuts while an executable layout prior is active."""

        return (
            not self._layout_executable
            or edge.source == "layout_prior"
        )

    def _edge_geometry_matches_rooms(self, edge: DoorEdge) -> bool:
        """Revalidate a rich/persisted edge against its current room polygons."""

        if not edge.executable:
            return False
        room_a = self._rooms.get(edge.room_a)
        room_b = self._rooms.get(edge.room_b)
        if room_a is None or room_b is None:
            return False
        try:
            center = (float(edge.center_x), float(edge.center_y))
            standoff_a = (
                float(edge.room_a_standoff[0]),  # type: ignore[index]
                float(edge.room_a_standoff[1]),  # type: ignore[index]
            )
            standoff_b = (
                float(edge.room_b_standoff[0]),  # type: ignore[index]
                float(edge.room_b_standoff[1]),  # type: ignore[index]
            )
        except (IndexError, TypeError, ValueError, OverflowError):
            return False
        if room_a.polygon:
            if (
                not point_in_polygon(standoff_a, room_a.polygon)
                or not point_on_polygon_boundary(
                    center, room_a.polygon,
                )
            ):
                return False
        if room_b.polygon:
            if (
                not point_in_polygon(standoff_b, room_b.polygon)
                or not point_on_polygon_boundary(
                    center, room_b.polygon,
                )
            ):
                return False
        return True

    def plan_door_route(
        self,
        src_room: str,
        dst_room: str,
        *,
        goal_xy: tuple[float, float] | None = None,
        footprint_width_m: float | None = None,
        clearance_m: float | None = None,
    ) -> DoorRoute:
        """Build a safe pre/centre/post route or return a structured failure."""

        src_room, dst_room = str(src_room), str(dst_room)
        footprint = (
            self._robot_footprint_width_m
            if footprint_width_m is None
            else float(footprint_width_m)
        )
        clearance = (
            self._door_clearance_m
            if clearance_m is None
            else float(clearance_m)
        )
        if (
            not math.isfinite(footprint)
            or not math.isfinite(clearance)
            or footprint <= 0
            or clearance < 0
        ):
            return DoorRoute(
                src_room,
                dst_room,
                diagnosis_code="invalid_topology",
                message="invalid robot footprint or door clearance",
            )
        required_width = footprint + 2.0 * clearance

        with self._lock:
            if src_room not in self._rooms or dst_room not in self._rooms:
                return DoorRoute(
                    src_room,
                    dst_room,
                    diagnosis_code="no_route",
                    message=f"unknown route endpoint: {src_room} -> {dst_room}",
                    required_width_m=required_width,
                )
            dst_node = self._rooms[dst_room]
            target = goal_xy or dst_node.navigation_goal or (
                dst_node.center_x,
                dst_node.center_y,
            )
            try:
                target = (float(target[0]), float(target[1]))
            except (TypeError, ValueError, IndexError):
                return DoorRoute(
                    src_room,
                    dst_room,
                    diagnosis_code="invalid_topology",
                    message="room goal is not a finite [x, y] point",
                    required_width_m=required_width,
                )
            if not all(math.isfinite(value) for value in target):
                return DoorRoute(
                    src_room,
                    dst_room,
                    diagnosis_code="invalid_topology",
                    message="room goal is not a finite [x, y] point",
                    required_width_m=required_width,
                )
            if dst_node.polygon and not point_in_polygon(target, dst_node.polygon):
                return DoorRoute(
                    src_room,
                    dst_room,
                    diagnosis_code="invalid_topology",
                    message=(
                        f"room goal {target!r} lies outside destination "
                        f"polygon {dst_room!r}"
                    ),
                    required_width_m=required_width,
                )

            if src_room == dst_room:
                waypoint = RouteWaypoint(
                    kind="room_goal",
                    room_from=src_room,
                    room_to=dst_room,
                    xy=target,
                    tolerance=0.50,
                    speed_limit_mps=0.60,
                    allow_reverse=True,
                )
                return DoorRoute(
                    src_room,
                    dst_room,
                    room_path=(src_room,),
                    waypoints=(waypoint,),
                    required_width_m=required_width,
                )

            topological = self._find_room_path(
                src_room,
                dst_room,
                edge_allowed=self._edge_is_trusted_for_route,
            )
            if not topological:
                return DoorRoute(
                    src_room,
                    dst_room,
                    diagnosis_code="no_route",
                    message=f"no door topology connects {src_room} to {dst_room}",
                    required_width_m=required_width,
                )
            executable = self._find_room_path(
                src_room,
                dst_room,
                edge_allowed=lambda edge: (
                    self._edge_is_trusted_for_route(edge)
                    and self._edge_geometry_matches_rooms(edge)
                ),
            )
            if not executable:
                return DoorRoute(
                    src_room,
                    dst_room,
                    room_path=topological,
                    diagnosis_code="invalid_topology",
                    message="route contains a legacy door without width/standoff geometry",
                    required_width_m=required_width,
                )
            passable = self._find_room_path(
                src_room,
                dst_room,
                edge_allowed=lambda edge: (
                    self._edge_is_trusted_for_route(edge)
                    and self._edge_geometry_matches_rooms(edge)
                    and edge.width is not None
                    and edge.width + 1e-9 >= required_width
                ),
            )
            if not passable:
                return DoorRoute(
                    src_room,
                    dst_room,
                    room_path=executable,
                    diagnosis_code="door_too_narrow",
                    message=(
                        f"all routes contain a door narrower than "
                        f"{required_width:.2f} m"
                    ),
                    required_width_m=required_width,
                )

            waypoints: list[RouteWaypoint] = []
            door_ids: list[str] = []
            for room_from, room_to in zip(passable, passable[1:]):
                edge = self._doors.get(self._door_key(room_from, room_to))
                if (
                    edge is None
                    or not self._edge_is_trusted_for_route(edge)
                    or not self._edge_geometry_matches_rooms(edge)
                ):
                    return DoorRoute(
                        src_room,
                        dst_room,
                        room_path=passable,
                        diagnosis_code="invalid_topology",
                        message=f"missing executable door: {room_from}-{room_to}",
                        required_width_m=required_width,
                    )
                pre = edge.standoff_for(room_from)
                post = edge.standoff_for(room_to)
                if pre is None or post is None:
                    return DoorRoute(
                        src_room,
                        dst_room,
                        room_path=passable,
                        diagnosis_code="invalid_topology",
                        message=f"missing oriented standoff: {edge.door_id}",
                        required_width_m=required_width,
                    )
                door_ids.append(edge.door_id)
                waypoints.extend(
                    (
                        RouteWaypoint(
                            kind="door_pre",
                            room_from=room_from,
                            room_to=room_to,
                            xy=pre,
                            tolerance=0.30,
                            speed_limit_mps=None,
                            # This segment approaches the doorway from an
                            # arbitrary point in the current room; it is not
                            # yet crossing the threshold. Normal controller
                            # manoeuvring (including a short reverse while
                            # aligning) is therefore safe and avoids trapping
                            # a robot whose initial heading faces away.
                            allow_reverse=True,
                            door_id=edge.door_id,
                        ),
                        RouteWaypoint(
                            kind="door_center",
                            room_from=room_from,
                            room_to=room_to,
                            xy=(edge.center_x, edge.center_y),
                            tolerance=0.22,
                            speed_limit_mps=None,
                            allow_reverse=False,
                            door_id=edge.door_id,
                        ),
                        RouteWaypoint(
                            kind="door_post",
                            room_from=room_from,
                            room_to=room_to,
                            xy=post,
                            tolerance=0.30,
                            speed_limit_mps=None,
                            allow_reverse=False,
                            door_id=edge.door_id,
                        ),
                    )
                )
            waypoints.append(
                RouteWaypoint(
                    kind="room_goal",
                    room_from=passable[-2],
                    room_to=dst_room,
                    xy=target,
                    tolerance=0.50,
                    speed_limit_mps=0.60,
                    allow_reverse=True,
                )
            )
            return DoorRoute(
                src_room,
                dst_room,
                room_path=passable,
                door_ids=tuple(door_ids),
                waypoints=tuple(waypoints),
                required_width_m=required_width,
            )

    def get_door_chain(
        self,
        src_room: str,
        dst_room: str,
    ) -> list[tuple[float, float, str]]:
        """Backward-compatible compact door-centre chain.

        This adapter never infers a missing edge from ``connected_rooms``: the
        BFS walks actual DoorEdges, eliminating the historical "skip a missing
        door and append the destination centre" wall-crossing failure.
        """

        with self._lock:
            path = self._find_room_path(
                str(src_room), str(dst_room), edge_allowed=lambda edge: True,
            )
            if not path:
                return []
            if len(path) == 1:
                room = self._rooms.get(str(dst_room))
                return (
                    [
                        (
                            *(room.navigation_goal or (room.center_x, room.center_y)),
                            str(dst_room),
                        )
                    ]
                    if room is not None
                    else []
                )
            waypoints: list[tuple[float, float, str]] = []
            for room_from, room_to in zip(path, path[1:]):
                edge = self._doors.get(self._door_key(room_from, room_to))
                if edge is None:
                    return []
                waypoints.append(
                    (
                        edge.center_x,
                        edge.center_y,
                        f"{room_from}_{room_to}_door",
                    )
                )
            dst_node = self._rooms.get(str(dst_room))
            if dst_node is None:
                return []
            dst_goal = dst_node.navigation_goal or (
                dst_node.center_x,
                dst_node.center_y,
            )
            waypoints.append((*dst_goal, str(dst_room)))
            return waypoints

    # ------------------------------------------------------------------
    # Viewpoint operations
    # ------------------------------------------------------------------

    def add_viewpoint(self, vp: ViewpointNode) -> None:
        with self._lock:
            self._viewpoints[vp.viewpoint_id] = vp
            # Update room's visit info
            room = self._rooms.get(vp.room_id)
            if room is not None:
                self._rooms[vp.room_id] = replace(
                    room,
                    representative_description=(
                        vp.scene_summary or room.representative_description
                    ),
                )

    def get_viewpoints_in_room(self, room_id: str) -> list[ViewpointNode]:
        with self._lock:
            return [
                vp for vp in self._viewpoints.values()
                if vp.room_id == room_id
            ]

    def should_add_viewpoint(
        self, room_id: str, x: float, y: float,
    ) -> bool:
        """Check if a new viewpoint should be added at (x, y).

        Returns True if no existing viewpoint in this room is within
        _VIEWPOINT_MIN_DISTANCE meters of the proposed position.
        """
        with self._lock:
            for vp in self._viewpoints.values():
                if vp.room_id != room_id:
                    continue
                dist = math.sqrt((vp.x - x)**2 + (vp.y - y)**2)
                if dist < _VIEWPOINT_MIN_DISTANCE:
                    return False
            return True

    # ------------------------------------------------------------------
    # Object operations
    # ------------------------------------------------------------------

    def add_object(self, obj: ObjectNode) -> None:
        with self._lock:
            self._objects[obj.object_id] = obj

    def find_objects_by_category(self, category: str) -> list[ObjectNode]:
        cat = category.lower().strip()
        with self._lock:
            return [
                o for o in self._objects.values()
                if cat in o.category.lower()
            ]

    def find_objects_in_room(self, room_id: str) -> list[ObjectNode]:
        with self._lock:
            return [
                o for o in self._objects.values()
                if o.room_id == room_id
            ]

    def merge_object(
        self,
        category: str,
        room_id: str,
        viewpoint_id: str,
        description: str = "",
        confidence: float = 0.8,
        x: float = 0.0,
        y: float = 0.0,
    ) -> ObjectNode:
        """Add or merge an object in a room.

        If an object with the same category already exists in this room,
        update it (add viewpoint, update description if confidence is higher).
        Otherwise create a new ObjectNode.

        Returns the resulting ObjectNode.
        """
        with self._lock:
            # Check for existing object with same category in same room
            for oid, existing in self._objects.items():
                if (existing.category.lower() == category.lower()
                        and existing.room_id == room_id):
                    # Merge: add viewpoint, keep higher confidence
                    vp_ids = set(existing.viewpoint_ids)
                    vp_ids.add(viewpoint_id)
                    merged = ObjectNode(
                        object_id=existing.object_id,
                        category=existing.category,
                        description=(
                            description if confidence > existing.confidence
                            else existing.description
                        ),
                        confidence=max(confidence, existing.confidence),
                        room_id=room_id,
                        x=x if x != 0.0 else existing.x,
                        y=y if y != 0.0 else existing.y,
                        z=existing.z,
                        attributes=existing.attributes,
                        viewpoint_ids=tuple(sorted(vp_ids)),
                        first_seen=existing.first_seen,
                    )
                    self._objects[oid] = merged
                    return merged

            # New object
            obj = ObjectNode(
                object_id=f"obj_{uuid.uuid4().hex[:8]}",
                category=category,
                description=description,
                confidence=confidence,
                room_id=room_id,
                x=x, y=y,
                viewpoint_ids=(viewpoint_id,),
            )
            self._objects[obj.object_id] = obj
            return obj

    # ------------------------------------------------------------------
    # Coverage tracking
    # ------------------------------------------------------------------

    def get_room_coverage(self, room_id: str) -> float:
        """Estimate what fraction of a room has been observed.

        Uses a simple model: each viewpoint covers coverage_area m².
        Total coverage is capped at room area. Returns 0.0-1.0.
        """
        vps = self.get_viewpoints_in_room(room_id)
        if not vps:
            return 0.0
        room = self.get_room(room_id)
        room_area = room.area if room else _ROOM_AREA_DEFAULT
        total_coverage = sum(vp.coverage_area for vp in vps)
        return min(1.0, total_coverage / room_area)

    # ------------------------------------------------------------------
    # VLM-guided room selection
    # ------------------------------------------------------------------

    def rank_rooms_for_goal(
        self, goal: str, text_llm: TextLLM,
    ) -> list[tuple[str, str]]:
        """Ask a TextLLM which room most likely contains the goal target.

        Args:
            goal: Natural language goal (e.g. "find the red chair").
            text_llm: A TextLLM adapter whose complete_text() will be called
                with a text-only prompt.  No provider, key, or model is
                hardcoded here — those details belong in the adapter.

        Returns:
            List of (room_id, reasoning) sorted by relevance.
        """
        rooms_info = []
        with self._lock:
            for rid, room in self._rooms.items():
                objs = [
                    o.category for o in self._objects.values()
                    if o.room_id == rid
                ]
                desc = room.representative_description or "not yet explored"
                rooms_info.append(
                    f"- {rid}: {desc}. Objects: {', '.join(objs) if objs else 'none seen'}"
                )

        if not rooms_info:
            return []

        prompt = (
            f"Goal: {goal}\n\n"
            f"Known rooms and their contents:\n"
            + "\n".join(rooms_info) + "\n\n"
            "Which room most likely contains what the goal describes? "
            "Rank rooms from most to least likely. "
            'Respond in JSON: [{"room": "...", "reasoning": "..."}]'
        )

        try:
            import json
            import re

            text = text_llm.complete_text(prompt)

            # Parse JSON — try direct parse first, then regex fallback
            clean = text.strip()
            data = None
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", clean, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))

            if isinstance(data, list):
                return [
                    (item.get("room", ""), item.get("reasoning", ""))
                    for item in data
                    if isinstance(item, dict)
                ]
        except Exception as exc:
            logger.warning("[SceneGraph] rank_rooms_for_goal failed: %s", exc)

        return []

    # ------------------------------------------------------------------
    # Backward-compatible SpatialMemory API
    # ------------------------------------------------------------------

    def visit(self, room: str, x: float, y: float) -> None:
        """Record visiting a room. Creates RoomNode if needed.

        Room center is computed as running average of all visit positions.
        This gives a more accurate center than a single detection point
        (which is often at the doorway).
        """
        with self._lock:
            existing = self._rooms.get(room)
            if existing:
                n = existing.visit_count
                # Online observations refine discovered rooms, but a deterministic
                # layout prior keeps its authored geometry within the same run.
                if existing.source == "layout_prior":
                    new_cx, new_cy = existing.center_x, existing.center_y
                else:
                    new_cx = (existing.center_x * n + x) / (n + 1)
                    new_cy = (existing.center_y * n + y) / (n + 1)
                self._rooms[room] = replace(
                    existing,
                    center_x=new_cx,
                    center_y=new_cy,
                    visit_count=n + 1,
                    last_visited=time.time(),
                )
            else:
                self._rooms[room] = RoomNode(
                    room_id=room,
                    center_x=x,
                    center_y=y,
                    visit_count=1,
                    last_visited=time.time(),
                )
            self._append_event({
                "type": "visit", "room": room, "x": x, "y": y,
                "timestamp": time.time(),
            })

    def observe(
        self,
        room: str,
        objects: list[str],
        description: str = "",
    ) -> None:
        """Record VLM observation. Creates viewpoint + objects.

        Compatible with SpatialMemory.observe() signature.
        """
        with self._lock:
            # Ensure room exists
            if room not in self._rooms:
                self._rooms[room] = RoomNode(
                    room_id=room, visit_count=0, last_visited=time.time(),
                )

            # Create viewpoint
            room_node = self._rooms[room]
            vp_id = f"vp_{uuid.uuid4().hex[:8]}"
            vp = ViewpointNode(
                viewpoint_id=vp_id,
                room_id=room,
                x=room_node.center_x,
                y=room_node.center_y,
                scene_summary=description,
            )
            self._viewpoints[vp_id] = vp

            # Update room description
            if description:
                self._rooms[room] = replace(
                    room_node,
                    representative_description=description,
                )

            # Create/merge objects
            for obj_name in objects:
                self.merge_object(
                    category=obj_name,
                    room_id=room,
                    viewpoint_id=vp_id,
                )

            self._append_event({
                "type": "observe", "room": room,
                "objects": objects, "timestamp": time.time(),
            })

    def observe_with_viewpoint(
        self,
        room: str,
        x: float,
        y: float,
        heading: float,
        objects: list[str],
        description: str = "",
        detected_objects: list[tuple[str, float, float]] | None = None,
    ) -> ViewpointNode | None:
        """Full viewpoint-aware observation (new API).

        Only adds a viewpoint if position is far enough from existing ones.
        Returns the ViewpointNode if created, None if skipped.

        Args:
            room: Room identifier.
            x: Robot x position (world frame).
            y: Robot y position (world frame).
            heading: Robot heading in radians.
            objects: Plain list of object name strings (used when
                detected_objects is None or empty).
            description: VLM scene description.
            detected_objects: Optional list of (category, obj_x, obj_y)
                tuples carrying per-object world coordinates.
                When non-empty this overrides the plain ``objects`` list.
        """
        # Determine which object source to use.
        use_detected = bool(detected_objects)

        if not self.should_add_viewpoint(room, x, y):
            # Still record objects even if viewpoint not added.
            with self._lock:
                nearest_vp = ""
                for vp in self._viewpoints.values():
                    if vp.room_id == room:
                        nearest_vp = vp.viewpoint_id
                        break
                if nearest_vp:
                    if use_detected:
                        for category, obj_x, obj_y in detected_objects:  # type: ignore[union-attr]
                            self.merge_object(
                                category=category, room_id=room,
                                viewpoint_id=nearest_vp,
                                x=obj_x, y=obj_y,
                            )
                    else:
                        for obj_name in objects:
                            self.merge_object(
                                category=obj_name, room_id=room,
                                viewpoint_id=nearest_vp,
                            )
            return None

        # Build object_ids tuple for the ViewpointNode record.
        if use_detected:
            vp_object_ids = tuple(cat for cat, _, _ in detected_objects)  # type: ignore[union-attr]
        else:
            vp_object_ids = tuple(objects)

        vp_id = f"vp_{uuid.uuid4().hex[:8]}"
        vp = ViewpointNode(
            viewpoint_id=vp_id,
            room_id=room,
            x=x, y=y,
            heading=heading,
            scene_summary=description,
            object_ids=vp_object_ids,
        )
        self.add_viewpoint(vp)

        # Visit room if not yet visited.
        self.visit(room, x, y)

        # Merge objects with or without per-object world coordinates.
        with self._lock:
            if use_detected:
                for category, obj_x, obj_y in detected_objects:  # type: ignore[union-attr]
                    self.merge_object(
                        category=category, room_id=room,
                        viewpoint_id=vp_id, x=obj_x, y=obj_y,
                    )
            else:
                for obj_name in objects:
                    self.merge_object(
                        category=obj_name, room_id=room,
                        viewpoint_id=vp_id,
                    )

        return vp

    def get_location(self, name: str) -> Any:
        """Backward compat: return a LocationRecord-like object."""
        room = self.get_room(name)
        if room is None:
            return None
        # Return a duck-typed object with .name, .x, .y
        from vector_os_nano.core.spatial_memory import LocationRecord
        objs = self.find_objects_in_room(name)
        return LocationRecord(
            name=room.room_id,
            x=room.center_x,
            y=room.center_y,
            visit_count=room.visit_count,
            last_visited=room.last_visited,
            objects_seen=tuple(o.category for o in objs),
            description=room.representative_description,
        )

    def get_all_locations(self) -> list:
        """Backward compat: return all rooms as LocationRecord-like objects."""
        return [self.get_location(r.room_id) for r in self.get_all_rooms()]

    def remember_location(self, name: str, x: float, y: float) -> None:
        """Save a custom named location."""
        self.visit(name, x, y)

    def get_visited_rooms(self) -> list[str]:
        with self._lock:
            return [
                r.room_id for r in self._rooms.values()
                if r.visit_count > 0
            ]

    def nearest_room(self, x: float, y: float) -> str | None:
        """Return room_id of the nearest room center, or None if no rooms exist.

        Only considers rooms with visit_count > 0 (actually explored).
        Used to replace the hardcoded _detect_current_room() throughout the codebase.
        """
        with self._lock:
            best_id: str | None = None
            best_dist = float("inf")
            for room in self._rooms.values():
                if room.visit_count <= 0:
                    continue
                dist = math.sqrt(
                    (room.center_x - x) ** 2 + (room.center_y - y) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = room.room_id
            return best_id

    def get_unvisited_rooms(self, all_rooms: list[str]) -> list[str]:
        visited = set(self.get_visited_rooms())
        return [r for r in all_rooms if r not in visited]

    def get_room_summary(self) -> str:
        """Human-readable summary for LLM system prompt.

        Includes room descriptions, objects, and coverage.
        """
        with self._lock:
            if not self._rooms:
                return "No rooms explored yet."

            visited = [r for r in self._rooms.values() if r.visit_count > 0]
            unvisited = [r for r in self._rooms.values() if r.visit_count == 0]

            parts = []
            for r in visited:
                objs = [
                    o.category for o in self._objects.values()
                    if o.room_id == r.room_id
                ]
                vp_count = sum(
                    1 for vp in self._viewpoints.values()
                    if vp.room_id == r.room_id
                )
                coverage = self.get_room_coverage(r.room_id)
                room_str = f"{r.room_id} ({r.visit_count} visits"
                if vp_count > 0:
                    room_str += f", {vp_count} viewpoints, {coverage:.0%} coverage"
                if objs:
                    room_str += f", saw: {', '.join(objs[:8])}"
                if r.representative_description:
                    room_str += f" — {r.representative_description[:80]}"
                room_str += ")"
                parts.append(room_str)

            total = len(self._rooms)
            summary = f"Rooms explored ({len(visited)}/{total}): " + ", ".join(parts)

            if unvisited:
                summary += f"\nUnexplored: {', '.join(r.room_id for r in unvisited)}"

            return summary

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return counts of rooms, viewpoints, and objects."""
        with self._lock:
            return {
                "rooms": len(self._rooms),
                "viewpoints": len(self._viewpoints),
                "objects": len(self._objects),
                "visited_rooms": sum(
                    1 for r in self._rooms.values() if r.visit_count > 0
                ),
            }

    # ------------------------------------------------------------------
    # Layout seeding (simulation)
    # ------------------------------------------------------------------

    def load_layout(self, layout_path: str, *, overwrite: bool = True) -> int:
        """Parse, validate, then atomically seed a v1/v2 layout prior.

        A failed parse never leaves a half-loaded graph. Schema v1 stays
        readable for compatibility, while only a complete schema-v2 layout is
        marked executable for strict known-layout navigation.
        """
        try:
            layout = load_room_layout(layout_path)
        except FileNotFoundError:
            self._last_layout_error = f"layout file not found: {layout_path}"
            return 0
        except (RoomLayoutError, OSError, yaml.YAMLError) as exc:
            self._last_layout_error = str(exc)
            logger.warning("[SceneGraph] load_layout failed: %s", exc)
            return 0

        def _polygon_area(points: tuple[tuple[float, float], ...]) -> float:
            if not points:
                return _ROOM_AREA_DEFAULT
            area2 = sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
            )
            return abs(area2) * 0.5

        with self._lock:
            now = time.time()
            for room_spec in layout.rooms:
                existing = self._rooms.get(room_spec.room_id)
                if existing is not None and not overwrite:
                    continue
                if existing is None:
                    self._rooms[room_spec.room_id] = RoomNode(
                        room_id=room_spec.room_id,
                        center_x=room_spec.center[0],
                        center_y=room_spec.center[1],
                        area=_polygon_area(room_spec.polygon),
                        visit_count=10,
                        last_visited=now,
                        polygon=room_spec.polygon,
                        aliases=room_spec.aliases,
                        source=room_spec.source,
                        confidence=1.0,
                        navigation_goal=room_spec.navigation_goal,
                    )
                else:
                    self._rooms[room_spec.room_id] = replace(
                        existing,
                        center_x=room_spec.center[0],
                        center_y=room_spec.center[1],
                        # ``area`` is learned/semantic SceneGraph memory and
                        # may carry a measured value.  The layout polygon is
                        # authoritative navigation geometry, but attaching a
                        # prior must not erase that accumulated memory.
                        area=existing.area,
                        polygon=room_spec.polygon or existing.polygon,
                        aliases=room_spec.aliases or existing.aliases,
                        source=room_spec.source,
                        confidence=1.0,
                        navigation_goal=room_spec.navigation_goal,
                    )

            layout_door_keys = {
                self._door_key(door.room_a, door.room_b)
                for door in layout.doors
            }
            if overwrite and layout.schema_version >= 2:
                # A schema-v2 document is the current authoritative prior.
                # Drop superseded prior edges so an old persisted topology
                # cannot survive a layout correction and form a shortcut.
                stale_prior_keys = [
                    key
                    for key, edge in self._doors.items()
                    if edge.source == "layout_prior"
                    and key not in layout_door_keys
                ]
                for key in stale_prior_keys:
                    del self._doors[key]

            for door_spec in layout.doors:
                key = self._door_key(door_spec.room_a, door_spec.room_b)
                if key in self._doors and not overwrite:
                    continue
                self.add_door(
                    door_spec.room_a,
                    door_spec.room_b,
                    door_spec.center[0],
                    door_spec.center[1],
                    door_id=door_spec.door_id,
                    width=door_spec.width,
                    normal=door_spec.normal,
                    room_a_standoff=door_spec.room_a_standoff,
                    room_b_standoff=door_spec.room_b_standoff,
                    source=door_spec.source,
                    confidence=door_spec.confidence,
                    authoritative=True,
                )

            # DoorEdge is the single source of truth for topology.  Rebuild
            # compatibility adjacency after removing/replacing prior edges so
            # stale ``connected_rooms`` values cannot advertise impossible
            # links to tools or downstream predictors.
            neighbors: dict[str, set[str]] = {
                room_id: set() for room_id in self._rooms
            }
            for edge in self._doors.values():
                if edge.room_a in neighbors and edge.room_b in neighbors:
                    neighbors[edge.room_a].add(edge.room_b)
                    neighbors[edge.room_b].add(edge.room_a)
            for room_id, room in tuple(self._rooms.items()):
                self._rooms[room_id] = replace(
                    room,
                    connected_rooms=tuple(sorted(neighbors[room_id])),
                )

            self._layout_schema_version = layout.schema_version
            self._layout_executable = layout.executable
            self._robot_footprint_width_m = layout.footprint_width_m
            self._door_clearance_m = layout.door_clearance_m
            self._last_layout_error = ""

        logger.info(
            "[SceneGraph] Loaded layout v%d: %d rooms, %d doors from %s",
            layout.schema_version,
            len(layout.rooms),
            len(layout.doors),
            layout_path,
        )
        return len(layout.rooms)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self) -> None:
        if not self._persist_path:
            return
        with self._lock:
            doors_serialized = {
                f"{key[0]}|{key[1]}": edge.to_dict()
                for key, edge in self._doors.items()
            }
            data = {
                "rooms": {k: v.to_dict() for k, v in self._rooms.items()},
                "viewpoints": {k: v.to_dict() for k, v in self._viewpoints.items()},
                "objects": {k: v.to_dict() for k, v in self._objects.items()},
                "doors": doors_serialized,
                "layout_profile": {
                    "schema_version": self._layout_schema_version,
                    "executable": self._layout_executable,
                    "footprint_width_m": self._robot_footprint_width_m,
                    "door_clearance_m": self._door_clearance_m,
                },
            }
        with open(self._persist_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    def load(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                return
            # Rooms
            for rid, rd in data.get("rooms", {}).items():
                self._rooms[rid] = RoomNode(
                    room_id=rd["room_id"],
                    center_x=float(rd.get("center_x", 0)),
                    center_y=float(rd.get("center_y", 0)),
                    area=float(rd.get("area", _ROOM_AREA_DEFAULT)),
                    visit_count=int(rd.get("visit_count", 0)),
                    last_visited=float(rd.get("last_visited", 0)),
                    representative_description=rd.get("representative_description", ""),
                    connected_rooms=tuple(rd.get("connected_rooms", ())),
                    polygon=tuple(
                        (float(point[0]), float(point[1]))
                        for point in rd.get("polygon", ())
                    ),
                    aliases=tuple(str(alias) for alias in rd.get("aliases", ())),
                    source=str(rd.get("source", "observed")),
                    confidence=float(rd.get("confidence", 0.8)),
                    navigation_goal=(
                        (
                            float(rd["navigation_goal"][0]),
                            float(rd["navigation_goal"][1]),
                        )
                        if rd.get("navigation_goal") is not None
                        else None
                    ),
                )
            # Viewpoints
            for vid, vd in data.get("viewpoints", {}).items():
                self._viewpoints[vid] = ViewpointNode(
                    viewpoint_id=vd["viewpoint_id"],
                    room_id=vd["room_id"],
                    x=float(vd.get("x", 0)),
                    y=float(vd.get("y", 0)),
                    heading=float(vd.get("heading", 0)),
                    timestamp=float(vd.get("timestamp", 0)),
                    scene_summary=vd.get("scene_summary", ""),
                    object_ids=tuple(vd.get("object_ids", ())),
                )
            # Objects
            for oid, od in data.get("objects", {}).items():
                self._objects[oid] = ObjectNode(
                    object_id=od["object_id"],
                    category=od.get("category", ""),
                    description=od.get("description", ""),
                    confidence=float(od.get("confidence", 0.8)),
                    room_id=od.get("room_id", ""),
                    x=float(od.get("x", 0)),
                    y=float(od.get("y", 0)),
                    z=float(od.get("z", 0)),
                    attributes=od.get("attributes", {}),
                    viewpoint_ids=tuple(od.get("viewpoint_ids", ())),
                    first_seen=float(od.get("first_seen", 0)),
                )
            # Doors — accept both the old {x,y,count} and rich DoorEdge form.
            for door_key, dv in data.get("doors", {}).items():
                parts = door_key.split("|", 1)
                if len(parts) != 2 or not isinstance(dv, dict):
                    continue
                center_raw = dv.get("center")
                if center_raw is None:
                    center = (float(dv.get("x", 0)), float(dv.get("y", 0)))
                else:
                    center = (float(center_raw[0]), float(center_raw[1]))
                normal_raw = dv.get("normal")
                a_standoff_raw = dv.get("room_a_standoff")
                b_standoff_raw = dv.get("room_b_standoff")
                room_a = str(dv.get("room_a", parts[0]))
                room_b = str(dv.get("room_b", parts[1]))
                key = self._door_key(room_a, room_b)
                self._doors[key] = DoorEdge(
                    door_id=str(dv.get("door_id", f"{room_a}-{room_b}")),
                    room_a=room_a,
                    room_b=room_b,
                    center_x=center[0],
                    center_y=center[1],
                    width=(
                        float(dv["width"])
                        if dv.get("width") is not None
                        else None
                    ),
                    normal_x=(
                        float(normal_raw[0]) if normal_raw is not None else None
                    ),
                    normal_y=(
                        float(normal_raw[1]) if normal_raw is not None else None
                    ),
                    room_a_standoff=(
                        (float(a_standoff_raw[0]), float(a_standoff_raw[1]))
                        if a_standoff_raw is not None
                        else None
                    ),
                    room_b_standoff=(
                        (float(b_standoff_raw[0]), float(b_standoff_raw[1]))
                        if b_standoff_raw is not None
                        else None
                    ),
                    source=str(dv.get("source", "observed")),
                    confidence=float(dv.get("confidence", 0.6)),
                    observation_count=int(
                        dv.get("observation_count", dv.get("count", 1))
                    ),
                    last_observed_center=(
                        (
                            float(dv["last_observed_center"][0]),
                            float(dv["last_observed_center"][1]),
                        )
                        if dv.get("last_observed_center") is not None
                        else None
                    ),
                    last_observed_confidence=(
                        float(dv["last_observed_confidence"])
                        if dv.get("last_observed_confidence") is not None
                        else None
                    ),
                )
                self._connect_rooms(room_a, room_b)
            profile = data.get("layout_profile", {})
            if isinstance(profile, dict):
                self._layout_schema_version = int(profile.get("schema_version", 0))
                self._layout_executable = bool(profile.get("executable", False))
                self._robot_footprint_width_m = float(
                    profile.get("footprint_width_m", DEFAULT_FOOTPRINT_WIDTH_M)
                )
                self._door_clearance_m = float(
                    profile.get("door_clearance_m", DEFAULT_DOOR_CLEARANCE_M)
                )
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("[SceneGraph] load failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append_event(self, event: dict) -> None:
        """Append event to log, trimming if over limit. Caller holds lock."""
        self._events.append(event)
        if len(self._events) > _MAX_EVENTS:
            del self._events[:len(self._events) - _MAX_EVENTS]
