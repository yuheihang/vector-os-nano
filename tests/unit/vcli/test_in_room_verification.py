# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P1-04 room-level verification, without MuJoCo or ROS2.

The room predicate must use the same RoomResolver as named navigation, prefer
real geometry, record an honest nearest-centre downgrade, fail closed for
unknown rooms, and participate in both evidence and actor-causation grading.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from vector_os_nano.playground.catalog import GO2_ROOM
from vector_os_nano.playground.world import PlaygroundWorld
from vector_os_nano.vcli.cognitive.actor_causation import (
    ActorBaseline,
    ActorCaused,
    grade,
    is_robot_predicate,
)
from vector_os_nano.vcli.cognitive.evidence_classifier import classify_verify_expr
from vector_os_nano.vcli.cognitive.goal_verifier import GoalVerifier
from vector_os_nano.vcli.worlds.go2_sim_oracle import make_in_room
from vector_os_nano.vcli.worlds.robot import RobotWorld


class _Base:
    _connected = True

    def __init__(self, x: float, y: float) -> None:
        self.position = [float(x), float(y), 0.3]
        self.position_reads = 0

    def get_position(self) -> list[float]:
        self.position_reads += 1
        return list(self.position)

    def get_heading(self) -> float:
        return 0.0


class _SceneGraph:
    def __init__(
        self,
        rooms: list[Any],
        *,
        room_at_result: Any = None,
    ) -> None:
        self._rooms = {str(room.room_id): room for room in rooms}
        self._room_at_result = room_at_result

    def get_all_rooms(self) -> list[Any]:
        return list(self._rooms.values())

    def get_room(self, room_id: str) -> Any:
        return self._rooms.get(room_id)

    def room_at(self, _x: float, _y: float) -> Any:
        return self._room_at_result

    def nearest_room(self, x: float, y: float) -> str | None:
        if not self._rooms:
            return None
        return min(
            self._rooms.values(),
            key=lambda room: (
                (float(room.center_x) - x) ** 2
                + (float(room.center_y) - y) ** 2,
                room.room_id,
            ),
        ).room_id


def _room(room_id: str, x: float, y: float, **geometry: Any) -> SimpleNamespace:
    return SimpleNamespace(
        room_id=room_id,
        center_x=float(x),
        center_y=float(y),
        visit_count=1,
        **geometry,
    )


def _agent(base: Any, scene_graph: Any = None) -> SimpleNamespace:
    return SimpleNamespace(_base=base, _spatial_memory=scene_graph)


def test_in_room_prefers_scene_graph_room_at_and_resolves_alias() -> None:
    dining = _room("dining_room", 3.0, 7.5)
    kitchen = _room("kitchen", 17.0, 2.5)
    graph = _SceneGraph([dining, kitchen], room_at_result=dining)
    predicate = make_in_room(
        _agent(_Base(3.0, 7.5), graph),
        scene_graph=graph,
    )

    assert predicate("dining room") is True
    assert predicate.verification_mode == "room_at"
    assert predicate.canonical_room == "dining_room"
    assert predicate.current_room == "dining_room"


def test_in_room_prefers_polygon_over_bounds_and_nearest_center() -> None:
    dining = _room(
        "dining_room",
        3.0,
        7.5,
        polygon=((0.0, 5.0), (6.0, 5.0), (6.0, 10.0), (0.0, 10.0)),
        bounds=(-100.0, -100.0, 100.0, 100.0),
    )
    kitchen = _room("kitchen", 17.0, 2.5)
    graph = _SceneGraph([dining, kitchen])
    predicate = make_in_room(
        _agent(_Base(2.0, 8.0), graph),
        scene_graph=graph,
    )

    assert predicate("餐厅") is True
    assert predicate.verification_mode == "polygon"


def test_in_room_uses_scene_graph_bounds_when_available() -> None:
    kitchen = _room("kitchen", 17.0, 2.5, bounds=(14.0, 0.0, 20.0, 5.0))
    graph = _SceneGraph([kitchen])
    predicate = make_in_room(
        _agent(_Base(18.0, 4.0), graph),
        scene_graph=graph,
    )

    assert predicate("厨房") is True
    assert predicate.verification_mode == "bounds"


def test_in_room_does_not_use_nearest_center_outside_known_geometry() -> None:
    dining = _room(
        "dining_room",
        3.0,
        7.5,
        polygon=((0.0, 5.0), (6.0, 5.0), (6.0, 10.0), (0.0, 10.0)),
    )
    graph = _SceneGraph([dining])
    predicate = make_in_room(
        _agent(_Base(50.0, 50.0), graph),
        scene_graph=graph,
    )

    # nearest_room() would return dining_room because it is the sole room, but
    # exact geometry says the robot is outside the known room set.
    assert predicate("dining_room") is False
    assert predicate.verification_mode == "geometry"
    assert predicate.current_room is None


def test_in_room_mixed_geometry_falls_back_only_for_rooms_without_geometry() -> None:
    dining = _room(
        "dining_room",
        3.0,
        7.5,
        polygon=((0.0, 5.0), (6.0, 5.0), (6.0, 10.0), (0.0, 10.0)),
    )
    kitchen = _room("kitchen", 17.0, 2.5)
    graph = _SceneGraph([dining, kitchen])
    predicate = make_in_room(
        _agent(_Base(17.0, 2.5), graph),
        scene_graph=graph,
    )

    assert predicate("kitchen") is True
    assert predicate.verification_mode == "nearest_center"
    assert predicate.current_room == "kitchen"


def test_in_room_unknown_mode_hides_undiscovered_room() -> None:
    dining = _room("dining_room", 3.0, 7.5)
    kitchen = _room("kitchen", 17.0, 2.5)
    kitchen.visit_count = 0
    agent = _agent(_Base(17.0, 2.5), _SceneGraph([dining, kitchen]))
    agent._world_mode = "unknown_exploration"
    predicate = make_in_room(agent, scene_graph=agent._spatial_memory)

    assert predicate("kitchen") is False
    assert predicate.verification_mode == "unavailable"
    assert predicate.canonical_room is None


def test_in_room_records_nearest_center_fallback_without_geometry() -> None:
    dining = _room("dining_room", 3.0, 7.5)
    kitchen = _room("kitchen", 17.0, 2.5)
    graph = _SceneGraph([dining, kitchen])
    predicate = make_in_room(
        _agent(_Base(2.5, 7.0), graph),
        scene_graph=graph,
    )

    assert predicate("dining_room") is True
    assert predicate.verification_mode == "nearest_center"
    assert predicate.current_room == "dining_room"


def test_unknown_room_fails_closed_before_reading_robot_position() -> None:
    base = _Base(3.0, 7.5)
    graph = _SceneGraph([_room("dining_room", 3.0, 7.5)])
    predicate = make_in_room(_agent(base, graph), scene_graph=graph)

    assert predicate("atlantis") is False
    assert predicate.verification_mode == "unavailable"
    assert predicate.canonical_room is None
    assert predicate.current_room is None
    assert base.position_reads == 0


def test_unavailable_scene_graph_and_base_fail_safe() -> None:
    predicate = make_in_room(_agent(None), scene_graph=None)
    assert predicate("kitchen") is False
    assert predicate.verification_mode == "unavailable"


def test_playground_in_room_uses_scenario_bounds_and_aliases() -> None:
    base = _Base(17.0, 2.5)
    namespace = PlaygroundWorld(GO2_ROOM).build_verify_namespace(_agent(base))
    predicate = namespace["in_room"]

    assert predicate("厨房") is True
    assert predicate.verification_mode == "bounds"
    assert predicate("guest bedroom") is False
    assert predicate.verification_mode == "bounds"


def test_robot_world_in_room_uses_live_spatial_memory() -> None:
    dining = _room("dining_room", 3.0, 7.5)
    graph = _SceneGraph([dining], room_at_result="dining_room")
    namespace = RobotWorld().build_verify_namespace(_agent(_Base(3.0, 7.5), graph))

    assert namespace["in_room"]("餐厅") is True
    assert namespace["in_room"].verification_mode == "room_at"


def test_robot_world_accepts_base_scene_graph_compatibility_binding() -> None:
    graph = _SceneGraph(
        [_room("kitchen", 17.0, 2.5, bounds=(14.0, 0.0, 20.0, 5.0))]
    )
    base = _Base(17.0, 2.5)
    base._scene_graph = graph
    agent = SimpleNamespace(_base=base)
    namespace = RobotWorld().build_verify_namespace(agent)

    assert namespace["in_room"]("kitchen") is True
    assert namespace["in_room"].verification_mode == "bounds"


def test_goal_verifier_evaluates_in_room_predicate() -> None:
    graph = _SceneGraph(
        [_room("kitchen", 17.0, 2.5, bounds=(14.0, 0.0, 20.0, 5.0))]
    )
    predicate = make_in_room(
        _agent(_Base(17.0, 2.5), graph),
        scene_graph=graph,
    )
    verifier = GoalVerifier({"in_room": predicate})

    assert verifier.verify("in_room('厨房')") is True
    assert verifier.verify("in_room('atlantis')") is False


def test_in_room_is_a_grounded_predicate_oracle() -> None:
    oracles = frozenset({"in_room"})
    assert classify_verify_expr("in_room('dining_room')", oracles) == "GROUNDED"
    assert classify_verify_expr(
        "in_room('dining_room') or True", oracles
    ) == "RAN"


def test_in_room_is_actor_graded_on_planar_base_channel() -> None:
    oracles = frozenset({"in_room"})
    before = ActorBaseline(
        base_cmd_motion=0.0,
        base_pos=(2.0, 7.5, 0.3),
        base_heading=0.0,
    )
    after = ActorBaseline(
        base_cmd_motion=0.4,
        base_pos=(3.0, 7.5, 0.3),
        base_heading=0.0,
    )

    assert is_robot_predicate("in_room('dining_room')", oracles) is True
    assert (
        grade(before, after, "in_room('dining_room')", oracles)
        is ActorCaused.CAUSED
    )


def test_in_room_drift_without_command_is_uncaused() -> None:
    oracles = frozenset({"in_room"})
    before = ActorBaseline(
        base_cmd_motion=0.0,
        base_pos=(2.0, 7.5, 0.3),
        base_heading=0.0,
    )
    drifted = ActorBaseline(
        base_cmd_motion=0.0,
        base_pos=(3.0, 7.5, 0.3),
        base_heading=0.0,
    )

    assert (
        grade(before, drifted, "in_room('dining_room')", oracles)
        is ActorCaused.UNCAUSED
    )
