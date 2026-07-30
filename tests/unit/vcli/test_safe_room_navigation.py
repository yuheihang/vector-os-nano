# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.unit.test_navigate_known_layout_route import _SegmentedFarBase
from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext, SkillRegistry
from vector_os_nano.skills.navigate import NavigateSkill
from vector_os_nano.vcli.primitives import PrimitiveContext
from vector_os_nano.vcli.native_loop import (
    _build_motor_tools,
    _build_tool_context,
)


REPO_LAYOUT = Path(__file__).resolve().parents[3] / "config" / "room_layout.yaml"


def _native_tool(
    base: _SegmentedFarBase,
) -> tuple[Any, Any]:
    graph = SceneGraph()
    assert graph.load_layout(str(REPO_LAYOUT)) == 8
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
            config={"world_mode": "known_layout"},
        )

    agent._build_context = _build_context
    agent._sync_robot_state = lambda: None
    engine = SimpleNamespace(_registry=None, _permissions=None)
    tools = _build_motor_tools(agent, engine)
    return tools["navigate_room"], _build_tool_context(
        agent, None, None, engine,
    )


def test_native_navigate_room_surfaces_structured_safe_route() -> None:
    base = _SegmentedFarBase()
    tool, context = _native_tool(base)

    result = tool.execute({"room": "餐厅"}, context)
    payload = json.loads(result.content)

    assert result.is_error is False
    assert payload["canonical_room"] == "dining_room"
    assert payload["planner"] == "far_segmented"
    assert payload["completed_segments"] == 4
    assert payload["route"]["room_path"] == [
        "living_room",
        "dining_room",
    ]
    assert [call[2]["waypoint_kind"] for call in base.navigate_calls] == [
        "door_pre",
        "door_center",
        "door_post",
        "room_goal",
    ]


def test_native_navigate_room_reports_failed_segment_without_fallback() -> None:
    base = _SegmentedFarBase(fail_index=1)
    tool, context = _native_tool(base)

    result = tool.execute({"room": "dining room"}, context)
    payload = json.loads(result.content)

    assert result.is_error is True
    assert payload["diagnosis_code"] == "segment_no_path"
    assert payload["failed_segment_index"] == 1
    assert payload["failed_waypoint"]["kind"] == "door_center"
    assert payload["completed_segments"] == 1
    assert len(base.navigate_calls) == 2
    assert (3.0, 7.5) not in [call[:2] for call in base.navigate_calls]
    assert base.stop_calls >= 1


def test_primitive_named_room_without_formal_skill_fails_closed(
    monkeypatch: Any,
) -> None:
    from vector_os_nano.vcli.primitives import navigation

    graph = SceneGraph()
    assert graph.load_layout(str(REPO_LAYOUT)) == 8
    base = _SegmentedFarBase()
    monkeypatch.setattr(
        navigation,
        "_ctx",
        PrimitiveContext(
            base=base,
            scene_graph=graph,
            skill_registry=None,
        ),
    )

    assert navigation.navigate_room("dining room") is False
    assert base.navigate_calls == []
    assert base.stop_calls >= 1
