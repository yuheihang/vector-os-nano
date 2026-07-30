# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_p2_room_navigation import _ALL_ROOMS_TARGETS
from vector_os_nano.core.scene_graph import RoomNode, SceneGraph


REPO_LAYOUT = Path(__file__).resolve().parents[2] / "config" / "room_layout.yaml"


def _layout_graph() -> SceneGraph:
    graph = SceneGraph()
    assert graph.load_layout(str(REPO_LAYOUT)) == 8
    return graph


def test_dining_to_kitchen_route_is_expanded_through_hallway() -> None:
    route = _layout_graph().plan_door_route("dining_room", "kitchen")
    assert route.success
    assert route.room_path == ("dining_room", "hallway", "kitchen")
    assert route.door_ids == (
        "dining_room-hallway",
        "kitchen-hallway",
    )
    assert [waypoint.kind for waypoint in route.waypoints] == [
        "door_pre",
        "door_center",
        "door_post",
        "door_pre",
        "door_center",
        "door_post",
        "room_goal",
    ]


def test_every_directed_repository_room_pair_has_an_executable_route() -> None:
    graph = _layout_graph()
    rooms = sorted(room.room_id for room in graph.get_all_rooms())
    expected_doors = {
        edge.door_id for edge in graph.get_all_door_edges().values()
    }
    traversed_doors: set[str] = set()
    checked = 0

    for source in rooms:
        for target in rooms:
            if source == target:
                continue
            route = graph.plan_door_route(source, target)
            assert route.success, (
                f"{source}->{target}: {route.diagnosis_code}: {route.message}"
            )
            assert route.room_path[0] == source
            assert route.room_path[-1] == target
            assert len(route.door_ids) == len(route.room_path) - 1
            traversed_doors.update(route.door_ids)
            checked += 1

    assert checked == 56
    assert traversed_doors == expected_doors


def test_all_rooms_acceptance_preset_covers_every_room_and_physical_door() -> None:
    graph = _layout_graph()
    current = "hallway"
    rooms = {current}
    doors: set[str] = set()

    for target in _ALL_ROOMS_TARGETS:
        route = graph.plan_door_route(current, target)
        assert route.success
        assert len(route.door_ids) == 1, (
            f"acceptance leg {current}->{target} is not adjacent: "
            f"{route.room_path}"
        )
        rooms.add(target)
        doors.update(route.door_ids)
        current = target

    assert rooms == {room.room_id for room in graph.get_all_rooms()}
    assert doors == {
        edge.door_id for edge in graph.get_all_door_edges().values()
    }
    assert len(_ALL_ROOMS_TARGETS) == 11


def test_living_to_dining_policy_uses_physical_direct_door() -> None:
    route = _layout_graph().plan_door_route("living_room", "dining_room")
    assert route.room_path == ("living_room", "dining_room")
    assert route.door_ids == ("living_room-dining_room",)
    assert [waypoint.xy for waypoint in route.waypoints] == pytest.approx(
        [(3.0, 4.4), (3.0, 5.0), (3.0, 5.6), (4.8, 6.0)]
    )


def test_reverse_route_swaps_pre_and_post_standoffs() -> None:
    graph = _layout_graph()
    forward = graph.plan_door_route("living_room", "hallway")
    reverse = graph.plan_door_route("hallway", "living_room")
    assert forward.waypoints[0].xy == pytest.approx((5.4, 3.0))
    assert forward.waypoints[2].xy == pytest.approx((6.6, 3.0))
    assert reverse.waypoints[0].xy == pytest.approx((6.6, 3.0))
    assert reverse.waypoints[2].xy == pytest.approx((5.4, 3.0))


def test_door_crossing_uses_normal_speed_and_disallows_reverse() -> None:
    route = _layout_graph().plan_door_route("living_room", "kitchen")
    door_waypoints = [waypoint for waypoint in route.waypoints if waypoint.door_id]
    assert door_waypoints
    assert all(
        waypoint.allow_reverse
        for waypoint in door_waypoints
        if waypoint.kind == "door_pre"
    )
    assert all(
        not waypoint.allow_reverse
        for waypoint in door_waypoints
        if waypoint.kind in {"door_center", "door_post"}
    )
    assert all(waypoint.speed_limit_mps is None for waypoint in door_waypoints)
    assert all(waypoint.tolerance < 0.5 for waypoint in door_waypoints)
    assert route.waypoints[-1].kind == "room_goal"
    assert route.waypoints[-1].allow_reverse is True


def test_same_room_has_only_room_goal() -> None:
    route = _layout_graph().plan_door_route("kitchen", "kitchen")
    assert route.success
    assert route.room_path == ("kitchen",)
    assert len(route.waypoints) == 1
    assert route.waypoints[0].kind == "room_goal"


def test_disconnected_rooms_return_no_route() -> None:
    graph = SceneGraph()
    graph.add_room(RoomNode("a", center_x=0.0, center_y=0.0))
    graph.add_room(RoomNode("b", center_x=2.0, center_y=0.0))
    route = graph.plan_door_route("a", "b")
    assert route.success is False
    assert route.diagnosis_code == "no_route"


def test_connected_rooms_tuple_without_real_edge_cannot_skip_to_goal() -> None:
    graph = SceneGraph()
    graph.add_room(
        RoomNode("a", center_x=0.0, center_y=0.0, connected_rooms=("b",))
    )
    graph.add_room(
        RoomNode("b", center_x=2.0, center_y=0.0, connected_rooms=("a",))
    )
    assert graph.get_door_chain("a", "b") == []
    assert graph.plan_door_route("a", "b").diagnosis_code == "no_route"


def test_all_topological_routes_too_narrow_are_distinguished() -> None:
    graph = _layout_graph()
    route = graph.plan_door_route(
        "living_room",
        "kitchen",
        footprint_width_m=1.1,
        clearance_m=0.1,
    )
    assert route.success is False
    assert route.diagnosis_code == "door_too_narrow"
    assert route.required_width_m == pytest.approx(1.3)


def test_legacy_door_is_invalid_topology_not_a_safe_route() -> None:
    graph = SceneGraph()
    graph.add_room(RoomNode("a", center_x=0.0, center_y=0.0))
    graph.add_room(RoomNode("b", center_x=2.0, center_y=0.0))
    graph.add_door("a", "b", 1.0, 0.0)
    assert graph.get_door_chain("a", "b")
    route = graph.plan_door_route("a", "b")
    assert route.success is False
    assert route.diagnosis_code == "invalid_topology"


def test_rich_edge_with_inconsistent_standoff_is_never_executable() -> None:
    graph = SceneGraph()
    graph.add_room(RoomNode("a", center_x=0.0, center_y=0.0))
    graph.add_room(RoomNode("b", center_x=2.0, center_y=0.0))
    graph.add_door(
        "a",
        "b",
        1.0,
        0.0,
        width=1.2,
        normal=(1.0, 0.0),
        # Both points are on room_b's side of the door.
        room_a_standoff=(1.4, 0.0),
        room_b_standoff=(1.6, 0.0),
    )

    edge = graph.get_door_edge("a", "b")
    assert edge is not None and edge.executable is False
    assert graph.plan_door_route("a", "b").diagnosis_code == "invalid_topology"


def test_loading_v2_removes_superseded_layout_prior_shortcuts() -> None:
    graph = SceneGraph()
    for room_id, center in (
        ("living_room", (3.0, 2.5)),
        ("study", (17.0, 7.5)),
    ):
        graph.add_room(RoomNode(room_id, *center))
    graph.add_door(
        "living_room",
        "study",
        10.0,
        5.0,
        door_id="obsolete-impossible-shortcut",
        width=1.2,
        normal=(1.0, 0.0),
        room_a_standoff=(9.4, 5.0),
        room_b_standoff=(10.6, 5.0),
        source="layout_prior",
        confidence=1.0,
        authoritative=True,
    )

    assert graph.load_layout(str(REPO_LAYOUT)) == 8
    assert graph.get_door_edge("living_room", "study") is None
    route = graph.plan_door_route("living_room", "study")
    assert "obsolete-impossible-shortcut" not in route.door_ids


def test_observed_door_cannot_create_known_layout_shortcut() -> None:
    graph = _layout_graph()
    graph.add_door(
        "living_room",
        "study",
        10.0,
        5.0,
        door_id="untrusted-observed-shortcut",
        width=1.2,
        normal=(1.0, 0.0),
        room_a_standoff=(9.4, 5.0),
        room_b_standoff=(10.6, 5.0),
        source="observed",
        confidence=0.01,
        authoritative=True,
    )

    route = graph.plan_door_route("living_room", "study")
    assert route.success
    assert route.room_path == ("living_room", "hallway", "study")
    assert route.door_ids == (
        "living_room-hallway",
        "study-hallway",
    )


def test_room_goal_outside_destination_polygon_fails_closed() -> None:
    route = _layout_graph().plan_door_route(
        "living_room",
        "dining_room",
        goal_xy=(17.0, 2.5),
    )

    assert route.success is False
    assert route.diagnosis_code == "invalid_topology"
    assert "outside destination polygon" in route.message
