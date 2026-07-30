# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from vector_os_nano.core.scene_graph import RoomNode, SceneGraph
from vector_os_nano.navigation.room_layout import RoomLayoutError, load_room_layout
from vector_os_nano.navigation.room_resolver import RoomResolver


REPO_LAYOUT = Path(__file__).resolve().parents[2] / "config" / "room_layout.yaml"
ROOM_XML = (
    Path(__file__).resolve().parents[2]
    / "vector_os_nano"
    / "hardware"
    / "sim"
    / "go2_room.xml"
)


def test_repository_layout_is_valid_executable_v2() -> None:
    layout = load_room_layout(REPO_LAYOUT)
    assert layout.schema_version == 2
    assert layout.executable is True
    assert len(layout.rooms) == 8
    assert len(layout.doors) == 9
    assert all(door.width == pytest.approx(1.2) for door in layout.doors)
    rooms = {room.room_id: room for room in layout.rooms}
    assert rooms["dining_room"].center == pytest.approx((3.0, 7.5))
    assert rooms["dining_room"].navigation_goal == pytest.approx((4.8, 6.0))
    assert rooms["kitchen"].navigation_goal == pytest.approx((15.0, 3.5))
    assert rooms["guest_bedroom"].navigation_goal == pytest.approx((15.0, 12.0))


def test_kitchen_study_door_swept_corridor_is_not_blocked_by_bookshelf() -> None:
    """The authored door edge must also be passable in the MuJoCo world."""

    layout = load_room_layout(REPO_LAYOUT)
    door = next(item for item in layout.doors if item.door_id == "kitchen-study")
    assert door.center is not None
    assert door.room_b_standoff is not None

    root = ET.parse(ROOM_XML).getroot()
    bookshelf = root.find(".//body[@name='bookshelf']")
    assert bookshelf is not None
    collision = bookshelf.find("./geom[@group='3']")
    assert collision is not None

    body_xy = tuple(float(value) for value in bookshelf.attrib["pos"].split()[:2])
    geom_xy = tuple(
        float(value) for value in collision.attrib.get("pos", "0 0 0").split()[:2]
    )
    half_xy = tuple(float(value) for value in collision.attrib["size"].split()[:2])
    obstacle = (
        body_xy[0] + geom_xy[0] - half_xy[0],
        body_xy[0] + geom_xy[0] + half_xy[0],
        body_xy[1] + geom_xy[1] - half_xy[1],
        body_xy[1] + geom_xy[1] + half_xy[1],
    )

    required_width = layout.footprint_width_m + 2.0 * layout.door_clearance_m
    corridor = (
        door.center[0] - required_width / 2.0,
        door.center[0] + required_width / 2.0,
        min(door.center[1], door.room_b_standoff[1]),
        max(door.center[1], door.room_b_standoff[1]),
    )
    overlaps = not (
        obstacle[1] <= corridor[0]
        or obstacle[0] >= corridor[1]
        or obstacle[3] <= corridor[2]
        or obstacle[2] >= corridor[3]
    )
    assert not overlaps, (
        f"bookshelf collision AABB {obstacle} blocks kitchen-study "
        f"swept corridor {corridor}"
    )


def test_v2_geometry_matches_real_wall_planes_and_topology() -> None:
    layout = load_room_layout(REPO_LAYOUT)
    by_id = {door.door_id: door for door in layout.doors}
    assert by_id["living_room-dining_room"].center == pytest.approx((3.0, 5.0))
    assert by_id["living_room-hallway"].center == pytest.approx((6.0, 3.0))
    assert by_id["kitchen-hallway"].center == pytest.approx((14.0, 3.0))
    assert by_id["kitchen-study"].center == pytest.approx((17.0, 5.0))
    assert by_id["master_bedroom-dining_room"].center == pytest.approx((3.0, 10.0))
    assert by_id["guest_bedroom-study"].center == pytest.approx((16.0, 10.0))


def test_v1_layout_still_loads_but_is_not_executable(tmp_path: Path) -> None:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
rooms:
  a: [0.0, 0.0]
  b: [2.0, 0.0]
doors:
  a-b: [1.0, 0.0]
""",
        encoding="utf-8",
    )
    graph = SceneGraph()
    assert graph.load_layout(str(path)) == 2
    assert graph.layout_schema_version == 1
    assert graph.has_executable_layout is False
    assert graph.get_door("a", "b") == pytest.approx((1.0, 0.0))
    route = graph.plan_door_route("a", "b")
    assert route.success is False
    assert route.diagnosis_code == "invalid_topology"


def test_v2_cannot_claim_executable_without_room_membership_polygons(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mixed.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a: [0.0, 0.0]
  b: [2.0, 0.0]
doors:
  a-b:
    center: [1.0, 0.0]
    width: 1.2
    normal: [1.0, 0.0]
    room_a_standoff: [0.4, 0.0]
    room_b_standoff: [1.6, 0.0]
""",
        encoding="utf-8",
    )

    graph = SceneGraph()
    assert graph.load_layout(str(path)) == 2
    assert graph.layout_schema_version == 2
    assert graph.has_executable_layout is False


def test_layout_aliases_and_polygons_reach_live_room_resolver() -> None:
    graph = SceneGraph()
    assert graph.load_layout(str(REPO_LAYOUT)) == 8
    assert RoomResolver(graph).resolve("饭厅").canonical == "dining_room"
    room = graph.get_room("dining_room")
    assert room is not None
    assert len(room.polygon) == 4
    assert room.source == "layout_prior"


def test_every_repository_door_standoff_belongs_to_its_declared_room() -> None:
    graph = SceneGraph()
    graph.load_layout(str(REPO_LAYOUT))
    resolver = RoomResolver(graph, world_mode="known_layout")

    for edge in graph.get_all_door_edges().values():
        assert edge.room_a_standoff is not None
        assert edge.room_b_standoff is not None
        assert resolver.locate(*edge.room_a_standoff).canonical == edge.room_a
        assert resolver.locate(*edge.room_b_standoff).canonical == edge.room_b


@pytest.mark.parametrize(
    ("door_body", "message"),
    [
        (
            """
    center: [1.0, 0.0]
    width: 1.2
    normal: [1.0, 0.0]
    room_a_standoff: [1.4, 0.0]
    room_b_standoff: [0.6, 0.0]
""",
            "opposite normal sides",
        ),
        (
            """
    center: [.nan, 0.0]
    width: 1.2
    normal: [1.0, 0.0]
    room_a_standoff: [0.4, 0.0]
    room_b_standoff: [1.6, 0.0]
""",
            "finite",
        ),
        (
            """
    center: [1.0, 0.0]
    width: 0.0
    normal: [1.0, 0.0]
    room_a_standoff: [0.4, 0.0]
    room_b_standoff: [1.6, 0.0]
""",
            "positive",
        ),
    ],
)
def test_v2_rejects_invalid_door_geometry(
    tmp_path: Path,
    door_body: str,
    message: str,
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        f"""
schema_version: 2
rooms:
  a:
    center: [0.0, 0.0]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
  b:
    center: [2.0, 0.0]
    polygon: [[1,-1], [3,-1], [3,1], [1,1]]
doors:
  a-b:
{door_body}
""",
        encoding="utf-8",
    )
    with pytest.raises(RoomLayoutError, match=message):
        load_room_layout(path)


def test_v2_rejects_standoff_outside_its_declared_room(tmp_path: Path) -> None:
    path = tmp_path / "outside.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a:
    center: [0, 0]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
  b:
    center: [2, 0]
    polygon: [[1,-1], [3,-1], [3,1], [1,1]]
doors:
  a-b:
    center: [1, 0]
    width: 1.2
    normal: [1, 0]
    room_a_standoff: [-2, 0]
    room_b_standoff: [1.6, 0]
""",
        encoding="utf-8",
    )

    with pytest.raises(RoomLayoutError, match="inside room 'a'"):
        load_room_layout(path)


def test_v2_rejects_room_center_outside_membership_polygon(
    tmp_path: Path,
) -> None:
    path = tmp_path / "outside_center.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a:
    center: [99, 99]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
doors: {}
""",
        encoding="utf-8",
    )

    with pytest.raises(RoomLayoutError, match="center must lie inside"):
        load_room_layout(path)


@pytest.mark.parametrize(
    "navigation_goal",
    ("[.nan, 0]", "[2, 0]", "[1, 0, 2]"),
)
def test_v2_rejects_invalid_navigation_goal(
    tmp_path: Path,
    navigation_goal: str,
) -> None:
    path = tmp_path / "bad_navigation_goal.yaml"
    path.write_text(
        f"""
schema_version: 2
rooms:
  a:
    center: [0, 0]
    navigation_goal: {navigation_goal}
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
doors: {{}}
""",
        encoding="utf-8",
    )

    with pytest.raises(RoomLayoutError, match="navigation_goal"):
        load_room_layout(path)


def test_v2_rejects_door_center_away_from_shared_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_door_center.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a:
    center: [0, 0]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
  b:
    center: [2, 0]
    polygon: [[1,-1], [3,-1], [3,1], [1,1]]
doors:
  a-b:
    center: [1, 100]
    width: 1.2
    normal: [1, 0]
    room_a_standoff: [0.4, 0]
    room_b_standoff: [1.6, 0]
""",
        encoding="utf-8",
    )

    with pytest.raises(RoomLayoutError, match="shared room boundary"):
        load_room_layout(path)


def test_v2_rejects_standoffs_outside_doorway_corridor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bad_door_corridor.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a:
    center: [0, 0]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
  b:
    center: [2, 0]
    polygon: [[1,-1], [3,-1], [3,1], [1,1]]
doors:
  a-b:
    center: [1, 0]
    width: 1.2
    normal: [1, 0]
    room_a_standoff: [0.4, 0.8]
    room_b_standoff: [1.6, 0.8]
""",
        encoding="utf-8",
    )

    with pytest.raises(RoomLayoutError, match="doorway corridor"):
        load_room_layout(path)


def test_unknown_door_endpoint_fails_before_scene_graph_mutation(tmp_path: Path) -> None:
    path = tmp_path / "bad_ref.yaml"
    path.write_text(
        """
schema_version: 2
rooms:
  a:
    center: [0, 0]
    polygon: [[-1,-1], [1,-1], [1,1], [-1,1]]
doors:
  a-missing:
    center: [1, 0]
    width: 1.2
    normal: [1, 0]
    room_a_standoff: [0.4, 0]
    room_b_standoff: [1.6, 0]
""",
        encoding="utf-8",
    )
    graph = SceneGraph()
    graph.add_room(RoomNode("sentinel", center_x=9.0, center_y=9.0))
    assert graph.load_layout(str(path)) == 0
    assert "unknown room" in graph.last_layout_error
    assert [room.room_id for room in graph.get_all_rooms()] == ["sentinel"]
    assert graph.get_all_doors() == {}


def test_observation_is_recorded_without_overwriting_layout_prior() -> None:
    graph = SceneGraph()
    graph.load_layout(str(REPO_LAYOUT))
    original = graph.get_door_edge("living_room", "hallway")
    assert original is not None
    graph.add_door(
        "living_room",
        "hallway",
        6.2,
        3.1,
        source="observed",
        confidence=0.75,
    )
    updated = graph.get_door_edge("living_room", "hallway")
    assert updated is not None
    assert (updated.center_x, updated.center_y) == pytest.approx((6.0, 3.0))
    assert updated.source == "layout_prior"
    assert updated.last_observed_center == pytest.approx((6.2, 3.1))
    assert updated.last_observed_confidence == pytest.approx(0.75)
    assert updated.observation_count == original.observation_count + 1


def test_rich_layout_metadata_round_trips_persistence(tmp_path: Path) -> None:
    persist = tmp_path / "scene_graph.yaml"
    graph = SceneGraph(persist_path=str(persist))
    graph.load_layout(str(REPO_LAYOUT))
    graph.add_door("living_room", "hallway", 6.1, 3.0, confidence=0.7)
    graph.save()

    restored = SceneGraph(persist_path=str(persist))
    restored.load()
    edge = restored.get_door_edge("living_room", "hallway")
    room = restored.get_room("living_room")
    assert edge is not None and edge.executable
    assert edge.source == "layout_prior"
    assert edge.last_observed_center == pytest.approx((6.1, 3.0))
    assert room is not None and len(room.polygon) == 4
    assert room.navigation_goal is None
    dining = restored.get_room("dining_room")
    assert dining is not None
    assert dining.navigation_goal == pytest.approx((4.8, 6.0))
    assert restored.layout_schema_version == 2
    assert restored.has_executable_layout is True
