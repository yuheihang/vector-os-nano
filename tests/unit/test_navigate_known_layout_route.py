# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from vector_os_nano.core.scene_graph import SceneGraph
from vector_os_nano.core.skill import SkillContext
from vector_os_nano.skills.navigate import (
    NavigateSkill,
    _bounded_safe_route_budget,
)


REPO_LAYOUT = Path(__file__).resolve().parents[2] / "config" / "room_layout.yaml"


class _SegmentedFarBase:
    """Planner fake that records the complete structured motor seam."""

    def __init__(
        self,
        *,
        position: tuple[float, float] = (3.0, 2.5),
        fail_index: int | None = None,
        miss_index: int | None = None,
        constraint_ready: bool | None = None,
    ) -> None:
        self.position = [position[0], position[1], 0.3]
        self.fail_index = fail_index
        self.miss_index = miss_index
        self.constraint_ready = constraint_ready
        self.navigate_calls: list[tuple[float, float, dict[str, Any]]] = []
        self.constraint_calls: list[dict[str, Any]] = []
        self.clear_calls = 0
        self.stop_calls = 0
        self.published_plans: list[tuple[dict[str, Any], str | None]] = []

    def get_position(self) -> list[float]:
        return list(self.position)

    def navigate_to(self, x: float, y: float, **kwargs: Any) -> bool:
        index = len(self.navigate_calls)
        self.navigate_calls.append((float(x), float(y), dict(kwargs)))
        if index == self.fail_index:
            return False
        if index == self.miss_index:
            self.position[:2] = [float(x) + 0.5, float(y)]
        else:
            self.position[:2] = [float(x), float(y)]
        return True

    def set_navigation_segment_constraints(self, **kwargs: Any) -> bool | None:
        self.constraint_calls.append(dict(kwargs))
        return self.constraint_ready

    def clear_navigation_segment_constraints(self, **_kwargs: Any) -> None:
        self.clear_calls += 1

    def publish_navigation_plan(
        self, route: dict[str, Any], *, goal_id: str | None = None,
    ) -> None:
        self.published_plans.append((dict(route), goal_id))

    def stop_navigation(self) -> None:
        self.stop_calls += 1


def _layout_graph() -> SceneGraph:
    graph = SceneGraph()
    assert graph.load_layout(str(REPO_LAYOUT)) == 8
    assert graph.has_executable_layout
    return graph


def _context(
    base: _SegmentedFarBase,
    graph: SceneGraph,
    *,
    world_mode: str = "known_layout",
) -> SkillContext:
    return SkillContext(
        base=base,
        services={"spatial_memory": graph},
        config={"world_mode": world_mode},
    )


def test_known_layout_executes_every_structured_waypoint_through_far() -> None:
    base = _SegmentedFarBase()
    result = NavigateSkill().execute(
        {"room": "dining_room", "_goal_id": "p2-route"},
        _context(base, _layout_graph()),
    )

    assert result.success
    assert result.result_data["planner"] == "far_segmented"
    assert result.result_data["completed_segments"] == 4
    assert [call[:2] for call in base.navigate_calls] == [
        (3.0, 4.4),
        (3.0, 5.0),
        (3.0, 5.6),
        (4.8, 6.0),
    ]
    kinds = [call[2]["waypoint_kind"] for call in base.navigate_calls]
    assert kinds == [
        "door_pre",
        "door_center",
        "door_post",
        "room_goal",
    ]
    assert all(
        call[2]["allow_door_fallback"] is False
        for call in base.navigate_calls
    )
    assert [call[2]["allow_reverse"] for call in base.navigate_calls] == [
        True,
        False,
        False,
        True,
    ]
    assert all(
        call[2]["speed_limit_mps"] is None
        for call in base.navigate_calls[:-1]
    )
    assert len(base.constraint_calls) == 4
    assert base.clear_calls == 4
    assert base.published_plans[0][0]["room_path"] == [
        "living_room",
        "dining_room",
    ]
    assert base.published_plans[0][1] == "p2-route"


def test_failed_far_segment_stops_and_never_issues_later_waypoints() -> None:
    base = _SegmentedFarBase(fail_index=2)
    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "segment_no_path"
    assert len(base.navigate_calls) == 3
    assert [call[2]["waypoint_kind"] for call in base.navigate_calls] == [
        "door_pre",
        "door_center",
        "door_post",
    ]
    assert result.result_data["completed_segments"] == 2
    assert result.result_data["failed_segment_index"] == 2
    assert result.result_data["failed_waypoint"]["kind"] == "door_post"
    assert base.stop_calls >= 1
    assert (3.0, 7.5) not in [call[:2] for call in base.navigate_calls]


def test_segment_timeout_is_not_misreported_as_no_path() -> None:
    base = _SegmentedFarBase(fail_index=0)
    base.get_navigation_goal_state = lambda _goal_id=None: {
        "state": "ERROR",
        "reason": "segment_timeout",
    }
    result = NavigateSkill().execute(
        {"room": "dining_room", "_goal_id": "p2-timeout"},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "segment_timeout"
    assert "timed out" in result.error_message
    assert len(base.navigate_calls) == 1
    assert result.result_data["route_budget"]["effective_timeout_s"] >= 360.0
    failed_budget = result.result_data["segment_budgets"][0]
    assert failed_budget["index"] == 0
    assert failed_budget["allocated_timeout_s"] == pytest.approx(
        base.navigate_calls[0][2]["timeout"], abs=0.001,
    )
    assert failed_budget["transport_succeeded"] is False


def test_long_room_approach_receives_distance_weighted_budget() -> None:
    base = _SegmentedFarBase(position=(10.0, 3.0))
    result = NavigateSkill().execute(
        {"room": "dining_room", "_goal_id": "p2-budget"},
        _context(base, _layout_graph()),
    )

    assert result.success
    timeouts = [float(call[2]["timeout"]) for call in base.navigate_calls]
    assert timeouts[0] > 100.0
    assert all(timeout >= 35.0 for timeout in timeouts)


def test_short_confined_room_approach_gets_floor_plus_distance_budget() -> None:
    """Turning near furniture must not consume a distance-only 63 s budget."""

    base = _SegmentedFarBase(position=(3.53, 11.54))
    result = NavigateSkill().execute(
        {"room": "dining_room", "_goal_id": "p2-master-reverse-budget"},
        _context(base, _layout_graph()),
    )

    assert result.success
    first_timeout = float(base.navigate_calls[0][2]["timeout"])
    assert first_timeout > 70.0
    assert first_timeout < 90.0


def test_three_door_route_scales_budget_without_starving_first_leg() -> None:
    """Ten segments must not consume the 360 s baseline as 10 x 35 s floors."""

    # Exact body pose recorded at the start of the reported failing segment.
    base = _SegmentedFarBase(position=(4.36, 6.16))
    result = NavigateSkill().execute(
        {"room": "guest_bedroom", "_goal_id": "p2-three-door-budget"},
        _context(base, _layout_graph()),
    )

    assert result.success
    assert result.result_data["segment_count"] == 10
    assert result.result_data["route"]["room_path"] == [
        "dining_room",
        "hallway",
        "study",
        "guest_bedroom",
    ]

    route_budget = result.result_data["route_budget"]
    assert route_budget["base_timeout_s"] == pytest.approx(360.0)
    assert route_budget["door_count"] == 3
    assert route_budget["door_budget_s"] == pytest.approx(900.0)
    assert route_budget["effective_timeout_s"] == pytest.approx(900.0)
    assert route_budget["max_timeout_s"] == pytest.approx(1200.0)

    timeouts = [float(call[2]["timeout"]) for call in base.navigate_calls]
    # Regression: the old fixed deadline allocated only ~36 s here and stopped
    # around (5.029, 7.347), still 0.75 m from the first pre-door point.
    assert timeouts[0] == pytest.approx(106.7, abs=0.2)
    # The 6.8 m hallway traversal is segment 3 and must retain a useful
    # distance-weighted share rather than another near-floor timeout.
    assert timeouts[3] >= 300.0
    assert timeouts[3] > timeouts[0]

    # Budget scaling must not weaken the physical waypoint contract.
    assert [call[2]["arrival_tolerance"] for call in base.navigate_calls] == [
        0.30,
        0.22,
        0.30,
        0.30,
        0.22,
        0.30,
        0.30,
        0.22,
        0.30,
        0.50,
    ]
    segment_budgets = result.result_data["segment_budgets"]
    assert len(segment_budgets) == 10
    assert segment_budgets[0]["allocated_timeout_s"] == pytest.approx(
        timeouts[0], abs=0.001,
    )
    assert segment_budgets[3]["leg_distance_m"] == pytest.approx(6.8)
    assert all(item["arrived_within_tolerance"] for item in segment_budgets)


def test_safe_route_complexity_budget_has_an_absolute_cap() -> None:
    budget = _bounded_safe_route_budget(
        base_timeout_s=360.0,
        door_count=99,
        total_polyline_length_m=999.0,
    )

    assert budget["effective_timeout_s"] == pytest.approx(1200.0)
    assert budget["door_budget_s"] > budget["max_timeout_s"]
    assert budget["distance_budget_s"] > budget["max_timeout_s"]


def test_far_true_outside_waypoint_tolerance_fails_closed() -> None:
    base = _SegmentedFarBase(miss_index=1)
    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "segment_out_of_tolerance"
    assert len(base.navigate_calls) == 2
    assert result.result_data["completed_segments"] == 1
    assert result.result_data["failed_waypoint"]["kind"] == "door_center"
    assert base.stop_calls >= 1


def test_unknown_source_room_never_sends_target_or_fallback_motion() -> None:
    base = _SegmentedFarBase(position=(-2.0, -2.0))
    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "source_room_unknown"
    assert base.navigate_calls == []
    assert base.stop_calls >= 1


def test_topology_failure_never_degrades_to_direct_room_center() -> None:
    graph = _layout_graph()
    # Make the validated layout impassable for this robot envelope.
    graph._robot_footprint_width_m = 1.5
    base = _SegmentedFarBase()

    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, graph),
    )

    assert not result.success
    assert result.diagnosis_code == "door_too_narrow"
    assert base.navigate_calls == []
    assert base.stop_calls >= 1


def test_named_room_cannot_be_smuggled_through_coordinate_branch() -> None:
    base = _SegmentedFarBase()
    result = NavigateSkill().execute(
        {"room": "kitchen", "x": 17.0, "y": 2.5},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "ambiguous_goal"
    assert base.navigate_calls == []
    assert base.stop_calls >= 1


def test_unacknowledged_segment_policy_stops_before_far_submission() -> None:
    base = _SegmentedFarBase(constraint_ready=False)
    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, _layout_graph()),
    )

    assert not result.success
    assert result.diagnosis_code == "navigation_failed"
    assert len(base.constraint_calls) == 1
    assert base.navigate_calls == []
    assert base.stop_calls >= 1


def test_v2_does_not_retry_with_an_unsafe_legacy_navigate_signature() -> None:
    class _LegacyFarBase:
        def __init__(self) -> None:
            self.position = [3.0, 2.5, 0.3]
            self.calls = 0
            self.stop_calls = 0

        def get_position(self) -> list[float]:
            return list(self.position)

        def navigate_to(self, _x: float, _y: float) -> bool:
            self.calls += 1
            return True

        def stop_navigation(self) -> None:
            self.stop_calls += 1

    base = _LegacyFarBase()
    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, _layout_graph()),  # type: ignore[arg-type]
    )

    assert not result.success
    assert result.diagnosis_code == "navigation_failed"
    # Python rejects the safety kwargs before entering the legacy method.  The
    # skill must not retry it with a weaker signature.
    assert base.calls == 0
    assert base.stop_calls >= 1
    assert "fail-closed segmented FAR" in result.error_message


def test_known_layout_v1_fails_closed_without_direct_navigation(
    tmp_path: Path,
) -> None:
    layout = tmp_path / "legacy.yaml"
    layout.write_text(
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
    assert graph.load_layout(str(layout)) == 2
    assert not graph.has_executable_layout
    base = _SegmentedFarBase(position=(0.0, 0.0))

    result = NavigateSkill().execute(
        {"room": "b"},
        _context(base, graph),
    )

    assert not result.success
    assert result.diagnosis_code == "layout_not_executable"
    assert base.navigate_calls == []
    assert base.stop_calls >= 1


def test_schema_zero_manual_graph_keeps_legacy_direct_compatibility() -> None:
    from vector_os_nano.core.scene_graph import RoomNode

    graph = SceneGraph()
    graph.add_room(RoomNode("a", center_x=0.0, center_y=0.0, visit_count=1))
    graph.add_room(RoomNode("b", center_x=2.0, center_y=0.0, visit_count=1))
    assert graph.layout_schema_version == 0
    base = _SegmentedFarBase(position=(0.0, 0.0))

    result = NavigateSkill().execute(
        {"room": "b"},
        _context(base, graph),
    )

    assert result.success
    assert len(base.navigate_calls) == 1
    assert base.navigate_calls[0][:2] == pytest.approx((2.0, 0.0))
    assert base.navigate_calls[0][2].get("waypoint_kind") is None
    assert result.result_data["planner"] == "far"


def test_unknown_exploration_mode_preserves_legacy_direct_behavior() -> None:
    graph = _layout_graph()
    graph.visit("living_room", 3.0, 2.5)
    graph.visit("dining_room", 3.0, 7.5)
    base = _SegmentedFarBase()

    result = NavigateSkill().execute(
        {"room": "dining_room"},
        _context(base, graph, world_mode="unknown_exploration"),
    )

    assert result.success
    assert len(base.navigate_calls) == 1
    assert base.navigate_calls[0][:2] == pytest.approx((3.0, 7.5))
    assert result.result_data["planner"] == "far"
    assert math.dist(base.position[:2], (3.0, 7.5)) == pytest.approx(0.0)
