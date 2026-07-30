# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P1 native named-room navigation contracts (pure fakes, no simulator/ROS2)."""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from vector_os_nano.core.skill import SkillContext, SkillRegistry
from vector_os_nano.skills.navigate import NavigateSkill
from vector_os_nano.vcli.native_loop import (
    NativeStepRunner,
    _build_motor_tools,
    _build_tool_context,
    _native_system_prompt,
    _scene_room_names,
)


class _Room:
    def __init__(
        self,
        room_id: str,
        center_x: float,
        center_y: float,
        *,
        visit_count: int = 10,
    ) -> None:
        self.room_id = room_id
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.visit_count = int(visit_count)


class _SceneGraph:
    """Small live SceneGraph surface consumed by RoomResolver/NavigateSkill."""

    def __init__(self, rooms: list[_Room]) -> None:
        # Preserve deliberately non-sorted insertion order; public results must
        # still sort canonical IDs.
        self._rooms = {room.room_id: room for room in rooms}

    def get_all_rooms(self) -> list[_Room]:
        return list(self._rooms.values())

    def get_room(self, room_id: str) -> _Room | None:
        return self._rooms.get(room_id)

    def nearest_room(self, x: float, y: float) -> str | None:
        if not self._rooms:
            return None
        return min(
            self._rooms.values(),
            key=lambda room: (
                math.dist(
                    (float(x), float(y)),
                    (room.center_x, room.center_y),
                ),
                room.room_id,
            ),
        ).room_id


class _Base:
    """Planner-capable fake base that records every requested world target."""

    __module__ = "vector_os_nano.hardware.sim.fake"
    _connected = True

    def __init__(self) -> None:
        self._position = [0.0, 0.0, 0.3]
        self._cmd_motion = 0.0
        self.navigate_calls: list[tuple[float, float, dict[str, Any]]] = []
        self.stop_calls = 0
        self._goal_stats: dict[str, dict[str, Any]] = {}

    def get_position(self) -> list[float]:
        return list(self._position)

    def get_heading(self) -> float:
        return 0.0

    def cmd_motion(self) -> float:
        return self._cmd_motion

    def navigate_to(self, x: float, y: float, **kwargs: Any) -> bool:
        tx, ty = float(x), float(y)
        self.navigate_calls.append((tx, ty, dict(kwargs)))
        self._cmd_motion += 1.0
        self._position[:2] = [tx, ty]
        goal_id = str(kwargs.get("goal_id") or "")
        if goal_id:
            self._goal_stats[goal_id] = {
                "goal_id": goal_id,
                "nonzero_cmd_count": 1,
                "cmd_motion_count": 1,
                "actual_velocity_observed": True,
                "moved_distance_m": math.dist((0.0, 0.0), (tx, ty)),
            }
        return True

    def begin_navigation_goal(
        self, goal_id: str, target_xy: tuple[float, float]
    ) -> None:
        self._goal_stats[str(goal_id)] = {
            "goal_id": str(goal_id),
            "target_xy": list(target_xy),
            "nonzero_cmd_count": 0,
        }

    def finalize_navigation_goal(self, goal_id: str, status: str) -> None:
        self._goal_stats.setdefault(str(goal_id), {})["status"] = status

    def get_navigation_telemetry(self, goal_id: str) -> dict[str, Any]:
        return dict(self._goal_stats.get(str(goal_id), {}))

    def stop_navigation(self) -> None:
        self.stop_calls += 1


class _PartialNavigationBase(_Base):
    """Issues real command evidence and moves, but does not reach the goal."""

    def navigate_to(self, x: float, y: float, **kwargs: Any) -> bool:
        tx, ty = float(x), float(y)
        self.navigate_calls.append((tx, ty, dict(kwargs)))
        self._cmd_motion += 0.5
        self._position[:2] = [0.5, 0.0]
        goal_id = str(kwargs.get("goal_id") or "")
        if goal_id:
            self._goal_stats[goal_id] = {
                "goal_id": goal_id,
                "nonzero_cmd_count": 1,
                "cmd_motion_count": 0.5,
                "actual_velocity_observed": True,
                "moved_distance_m": 0.5,
                "actor_caused": True,
            }
        return False


def _make_agent(
    rooms: list[_Room] | None = None,
    *,
    base: _Base | None = None,
) -> tuple[SimpleNamespace, _Base, _SceneGraph]:
    graph = _SceneGraph(
        rooms
        if rooms is not None
        else [
            _Room("kitchen", 17.0, 2.5),
            _Room("dining_room", 3.0, 7.5),
        ]
    )
    base = base or _Base()
    registry = SkillRegistry()
    registry.register(NavigateSkill())
    agent = SimpleNamespace(
        _base=base,
        _arm=None,
        _gripper=None,
        _spatial_memory=graph,
        _skill_registry=registry,
        _world_mode="known_layout",
    )

    def _build_context() -> SkillContext:
        return SkillContext(
            base=base,
            services={
                "spatial_memory": graph,
                "skill_registry": registry,
            },
            config={"world_mode": agent._world_mode},
        )

    agent._build_context = _build_context
    agent._sync_robot_state = lambda: None
    return agent, base, graph


def _engine() -> SimpleNamespace:
    return SimpleNamespace(_registry=None, _permissions=None)


def _tools_and_context(
    agent: SimpleNamespace,
) -> tuple[dict[str, Any], Any]:
    engine = _engine()
    tools = _build_motor_tools(agent, engine)
    return tools, _build_tool_context(agent, None, None, engine)


def _payload(result: Any) -> dict[str, Any]:
    payload = json.loads(result.content)
    assert isinstance(payload, dict)
    return payload


def test_native_motor_surface_splits_formal_navigate_without_old_name() -> None:
    agent, _, _ = _make_agent()
    tools, _ = _tools_and_context(agent)

    navigation_names = {name for name in tools if name.startswith("navigate")}
    assert navigation_names == {"navigate_room", "navigate_xy"}
    assert "navigate" not in tools

    # Both native names are adapters over the one official NavigateSkill, not
    # independent implementations or a second room database.
    assert isinstance(tools["navigate_room"]._formal_tool._skill, NavigateSkill)
    assert (
        tools["navigate_room"]._formal_tool
        is tools["navigate_xy"]._formal_tool
    )


@pytest.mark.parametrize(
    ("requested", "canonical"),
    [
        ("dining room", "dining_room"),
        ("餐厅", "dining_room"),
    ],
)
def test_navigate_room_aliases_delegate_to_formal_skill_and_return_json(
    requested: str,
    canonical: str,
) -> None:
    agent, base, _ = _make_agent()
    tools, context = _tools_and_context(agent)

    result = tools["navigate_room"].execute({"room": requested}, context)
    data = _payload(result)

    assert result.is_error is False
    assert {
        "goal_id",
        "goal_type",
        "requested_room",
        "canonical_room",
        "target_xy",
        "source",
        "planner",
        "arrived",
    } <= data.keys()
    assert data["goal_id"].startswith("nav-")
    assert data["goal_type"] == "room"
    assert data["requested_room"] == requested
    assert data["canonical_room"] == canonical
    assert data["target_xy"] == [3.0, 7.5]
    assert data["source"] == "scene_graph"
    assert data["planner"] == "far"
    assert data["arrived"] is True
    assert base.navigate_calls[-1][:2] == (3.0, 7.5)


def test_dining_room_target_comes_from_live_scene_graph_not_native_loop() -> None:
    agent, base, graph = _make_agent()
    tools, context = _tools_and_context(agent)

    first = _payload(
        tools["navigate_room"].execute({"room": "dining_room"}, context)
    )
    assert first["target_xy"] == [3.0, 7.5]
    assert base.navigate_calls[-1][:2] == (3.0, 7.5)

    # Mutating the fake LIVE SceneGraph changes the target on the next call. A
    # native-loop room table would keep returning the old coordinates.
    dining = graph.get_room("dining_room")
    assert dining is not None
    dining.center_x, dining.center_y = 8.25, 9.5
    second = _payload(
        tools["navigate_room"].execute({"room": "dining_room"}, context)
    )
    assert second["target_xy"] == [8.25, 9.5]
    assert base.navigate_calls[-1][:2] == (8.25, 9.5)

    import vector_os_nano.vcli.native_loop as native_loop

    native_source = Path(inspect.getsourcefile(native_loop) or "").read_text()
    assert "(3.0, 7.5)" not in native_source
    assert '"dining_room": (' not in native_source


def test_unknown_room_is_owned_failure_and_blocks_xy_for_same_turn() -> None:
    # Reverse lexical order in the live graph to prove the error payload sorts it.
    agent, base, _ = _make_agent(
        [
            _Room("kitchen", 17.0, 2.5),
            _Room("dining_room", 3.0, 7.5),
        ]
    )
    tools, context = _tools_and_context(agent)

    class _UnexpectedAction:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, _params: dict[str, Any], _context: Any) -> Any:
            self.calls += 1
            raise AssertionError("action must be frozen after unknown_room")

    unexpected_walk = _UnexpectedAction()
    tools["walk"] = unexpected_walk

    class _Verifier:
        def verify(self, _expr: str) -> bool:
            return False

    runner = NativeStepRunner(
        agent,
        _Verifier(),
        frozenset({"in_room", "at_position"}),
        tools,
        context,
    )
    room_result = runner.dispatch_skill(
        "navigate_room", {"room": "observatory"}
    )
    room_data = _payload(room_result)

    assert room_result.is_error is True
    assert room_data["diagnosis_code"] == "unknown_room"
    assert room_data["available_rooms"] == ["dining_room", "kitchen"]
    assert room_data["arrived"] is False
    assert base.navigate_calls == []

    xy_result = runner.dispatch_skill("navigate_xy", {"x": 3.0, "y": 7.5})
    xy_data = _payload(xy_result)
    assert xy_result.is_error is True
    assert xy_data["diagnosis_code"] == "room_resolution_failed"
    assert base.navigate_calls == []

    walk_result = runner.dispatch_skill(
        "walk", {"direction": "forward", "distance": 1.0}
    )
    walk_data = _payload(walk_result)
    assert walk_result.is_error is True
    assert walk_data["diagnosis_code"] == "room_resolution_failed"
    assert unexpected_walk.calls == 0
    assert base.navigate_calls == []

    trace = runner.build_trace("go to the observatory")
    assert trace.success is False
    assert len(trace.steps) == 1
    assert len(trace.goal_tree.sub_goals) == 1
    assert trace.steps[0].success is False
    assert trace.steps[0].strategy == "navigate_room"
    assert trace.steps[0].result_data["diagnosis_code"] == "unknown_room"
    assert trace.goal_tree.sub_goals[0].description.startswith("native:")


def test_navigate_xy_schema_is_strict_and_requires_finite_numbers() -> None:
    agent, base, _ = _make_agent()
    tools, context = _tools_and_context(agent)
    tool = tools["navigate_xy"]

    assert tool.input_schema == {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "Target x in world-frame metres.",
            },
            "y": {
                "type": "number",
                "description": "Target y in world-frame metres.",
            },
        },
        "required": ["x", "y"],
        "additionalProperties": False,
    }

    invalid_inputs = [
        {},
        {"x": 1.0},
        {"x": 1.0, "y": 2.0, "room": "kitchen"},
        {"x": True, "y": 2.0},
        {"x": "1.0", "y": 2.0},
        {"x": 10**400, "y": 2.0},
        {"x": float("nan"), "y": 2.0},
        {"x": float("inf"), "y": 2.0},
        {"x": 1.0, "y": float("-inf")},
    ]
    for params in invalid_inputs:
        result = tool.execute(params, context)
        assert result.is_error is True, params
        assert _payload(result)["diagnosis_code"] == "invalid_coordinate"
    assert base.navigate_calls == []

    valid = tool.execute({"x": 2.5, "y": 1.2}, context)
    assert valid.is_error is False
    assert _payload(valid)["target_xy"] == [2.5, 1.2]
    assert base.navigate_calls[-1][:2] == (2.5, 1.2)


def test_partial_navigation_with_real_motion_is_ran_not_failed() -> None:
    from vector_os_nano.vcli.cognitive.actor_causation import ActorCaused
    from vector_os_nano.vcli.verdict import VerdictReport

    partial_base = _PartialNavigationBase()
    agent, _, _ = _make_agent(base=partial_base)
    tools, context = _tools_and_context(agent)

    class _Verifier:
        def verify(self, _expr: str) -> bool:
            return False

    runner = NativeStepRunner(
        agent,
        _Verifier(),
        frozenset({"at_position", "in_room"}),
        tools,
        context,
    )
    result = runner.dispatch_skill("navigate_xy", {"x": 2.5, "y": 1.2})
    data = _payload(result)
    assert result.is_error is True
    assert data["diagnosis_code"] == "navigation_failed"
    assert data["navigation_stats"]["actual_velocity_observed"] is True
    assert data["navigation_stats"]["moved_distance_m"] == pytest.approx(0.5)

    runner.handle_verify("at_position(2.5, 1.2)")
    trace = runner.build_trace("go to (2.5, 1.2)")
    assert trace.success is True
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.success is True
    assert step.verify_result is False
    assert step.actor_caused is ActorCaused.CAUSED
    assert "navigation_failed" in step.error

    report = VerdictReport.from_trace(trace, frozenset({"at_position"}))
    assert report.verified is False
    assert report.evidence == "RAN"
    assert report.per_step[0].evidence == "RAN"


def test_prompt_lists_only_live_canonical_rooms_without_layout_coordinates() -> None:
    agent, _, _ = _make_agent(
        [
            _Room("kitchen", 17.0, 2.5),
            _Room("dining_room", 3.0, 7.5),
        ]
    )
    names = _scene_room_names(agent)
    assert names == ("dining_room", "kitchen")

    blocks = _native_system_prompt(
        None,
        frozenset({"at_position", "in_room"}),
        room_names=names,
        world_mode="known_layout",
        has_navigate_room=True,
        has_navigate_xy=True,
    )
    text = "\n".join(str(block["text"]) for block in blocks)

    assert "Named-room vocabulary" in text
    assert "dining_room, kitchen" in text
    for absent_room in (
        "living_room",
        "study",
        "master_bedroom",
        "guest_bedroom",
        "bathroom",
        "hallway",
    ):
        assert absent_room not in text
    assert "3.0, 7.5" not in text
    assert "17.0, 2.5" not in text
    assert "room_layout.yaml" not in text
    assert "navigate_room(room='<name-or-alias>')" in text
    assert "in_room('<canonical_room>')" in text
    assert "NEVER invent, expose, or infer room coordinates" in text
    assert "prior trusted perception/planning tool result" in text


def test_unknown_exploration_vocab_hides_undiscovered_rooms() -> None:
    agent, _, _ = _make_agent(
        [
            _Room("kitchen", 17.0, 2.5, visit_count=0),
            _Room("dining_room", 3.0, 7.5, visit_count=1),
        ]
    )
    agent._world_mode = "unknown_exploration"

    names = _scene_room_names(agent)
    assert names == ("dining_room",)

    blocks = _native_system_prompt(
        None,
        frozenset({"at_position", "in_room"}),
        room_names=names,
        world_mode="unknown_exploration",
        has_navigate_room=True,
        has_navigate_xy=True,
    )
    text = "\n".join(str(block["text"]) for block in blocks)

    assert "rooms discovered so far in this unknown world" in text
    assert "dining_room" in text
    assert "kitchen" not in text
    assert "3.0" not in text
    assert "7.5" not in text

    tools, context = _tools_and_context(agent)
    hidden = tools["navigate_room"].execute({"room": "kitchen"}, context)
    hidden_data = _payload(hidden)
    assert hidden.is_error is True
    assert hidden_data["diagnosis_code"] == "unknown_room"
    assert hidden_data["available_rooms"] == ["dining_room"]
    assert agent._base.navigate_calls == []


def test_known_layout_attach_corrects_persisted_geometry_and_keeps_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vector_os_nano.core.scene_graph import ObjectNode, RoomNode, SceneGraph
    from vector_os_nano.vcli.tools.sim_tool import _attach_sim_scene_graph

    home = tmp_path / "home"
    persist = home / ".vector_os_nano" / "scene_graph.yaml"
    persist.parent.mkdir(parents=True)
    stale = SceneGraph(persist_path=str(persist))
    stale.add_room(
        RoomNode(
            room_id="dining_room",
            center_x=99.0,
            center_y=98.0,
            area=42.0,
            visit_count=3,
            last_visited=123.0,
            representative_description="sunny dining area",
            connected_rooms=("hallway",),
        )
    )
    stale.add_room(RoomNode("custom_lab", 1.0, 1.0, visit_count=2))
    stale.add_door("dining_room", "hallway", 90.0, 90.0)
    stale.add_object(
        ObjectNode(
            object_id="obj-1", category="chair", room_id="dining_room"
        )
    )
    stale.save()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VECTOR_WORLD_MODE", "known_layout")
    agent = SimpleNamespace(_config={})
    base = SimpleNamespace()
    repo = str(Path(__file__).resolve().parents[3])
    attached = _attach_sim_scene_graph(agent, base, repo)

    dining = attached.get_room("dining_room")
    assert dining is not None
    assert (dining.center_x, dining.center_y) == (3.0, 7.5)
    assert dining.area == 42.0
    assert dining.visit_count == 3
    assert dining.last_visited == 123.0
    assert dining.representative_description == "sunny dining area"
    assert attached.get_room("custom_lab") is not None
    # P2 schema-v2 geometry uses the physical wall plane (x=6.0); the
    # historical x=6.5 value was a one-sided approach point, not a door centre.
    assert attached.get_door("dining_room", "hallway") == (6.0, 8.0)
    edge = attached.get_door_edge("dining_room", "hallway")
    assert edge is not None and edge.executable
    assert edge.room_a_standoff == (5.4, 8.0)
    assert edge.room_b_standoff == (6.6, 8.0)
    assert [obj.object_id for obj in attached.find_objects_in_room("dining_room")] == [
        "obj-1"
    ]
    assert base._scene_graph is attached
    assert agent._world_mode == "known_layout"

    reloaded = SceneGraph(persist_path=str(persist))
    reloaded.load()
    assert reloaded.get_room("dining_room") is not None
    assert (
        reloaded.get_room("dining_room").center_x,
        reloaded.get_room("dining_room").center_y,
    ) == (3.0, 7.5)


def test_sim_scene_graph_keeps_unknown_exploration_persistence_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vector_os_nano.core.scene_graph import RoomNode, SceneGraph
    from vector_os_nano.vcli.tools.sim_tool import _attach_sim_scene_graph

    home = tmp_path / "home"
    known_path = home / ".vector_os_nano" / "scene_graph.yaml"
    unknown_path = (
        home / ".vector_os_nano" / "scene_graph_unknown_exploration.yaml"
    )
    unknown_path.parent.mkdir(parents=True)
    known = SceneGraph(persist_path=str(known_path))
    known.add_room(RoomNode("kitchen", 17.0, 2.5, visit_count=10))
    known.save()
    unknown = SceneGraph(persist_path=str(unknown_path))
    unknown.add_room(RoomNode("discovered_den", 4.0, 4.0, visit_count=1))
    unknown.save()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("VECTOR_WORLD_MODE", "unknown_exploration")
    agent = SimpleNamespace(_config={})
    base = SimpleNamespace()
    repo = str(Path(__file__).resolve().parents[3])
    attached = _attach_sim_scene_graph(agent, base, repo)

    assert attached.get_room("discovered_den") is not None
    assert attached.get_room("kitchen") is None
    assert attached.get_room("dining_room") is None
    assert attached._persist_path == str(unknown_path)
