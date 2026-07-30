# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P1-05 goal-scoped navigation telemetry tests.

These tests exercise the protocol accumulator and proxy ingestion directly.
They intentionally require neither rclpy nor any ROS message package.
"""
from __future__ import annotations

import symtable
from pathlib import Path

import pytest

from vector_os_nano.hardware.sim.go2_ros2_proxy import (
    GoalMotionTracker,
    Go2ROS2Proxy,
    NAV_GOAL_DISPLACEMENT_EPS_M,
    NAV_GOAL_TELEMETRY_VERSION,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _event(
    goal_id: str,
    seq: int,
    *,
    event: str,
    status: str,
    count: int,
    motion: float,
    duration: float,
    distance: float,
) -> dict:
    return {
        "version": NAV_GOAL_TELEMETRY_VERSION,
        "event": event,
        "goal_id": goal_id,
        "seq": seq,
        "status": status,
        "target_xy": [3.0, 7.5],
        "nonzero_cmd_count": count,
        "cmd_motion_count": motion,
        "nonzero_cmd_duration_s": duration,
        "moved_distance_m": distance,
        "actual_velocity_observed": count > 0 and motion > 0.0,
        "elapsed_s": duration,
    }


def test_tracker_counts_only_plant_confirmed_nonzero_velocity() -> None:
    clock = _Clock()
    tracker = GoalMotionTracker(clock)

    accepted = tracker.begin("G123", target_xy=(3.0, 7.5), position=(0.0, 0.0))
    assert accepted is not None
    assert accepted["goal_id"] == "G123"

    clock.now = 0.1
    first = tracker.record_velocity(
        0.4, 0.0, 0.1, executed_motion=0.5, position=(0.0, 0.0)
    )
    clock.now = 0.2
    second = tracker.record_velocity(
        0.4, 0.0, 0.1, executed_motion=0.5, position=(0.1, 0.0)
    )
    clock.now = 0.3
    stopped = tracker.record_velocity(
        0.0, 0.0, 0.0, executed_motion=0.0, position=(0.2, 0.0)
    )
    clock.now = 0.4
    final = tracker.finalize("G123", status="succeeded", position=(0.2, 0.0))

    assert first is not None and second is not None
    assert stopped is None
    assert final is not None
    assert final["nonzero_cmd_count"] == 2
    assert final["cmd_motion_count"] == pytest.approx(1.0)
    assert final["nonzero_cmd_duration_s"] == pytest.approx(0.2)
    assert final["moved_distance_m"] == pytest.approx(0.2)
    assert final["actual_velocity_observed"] is True
    assert final["actor_caused"] is True


def test_no_goal_drift_and_gated_commands_are_not_actor_evidence() -> None:
    clock = _Clock()
    tracker = GoalMotionTracker(clock)

    # Background movement with no native navigation goal has nowhere to attach.
    assert tracker.record_velocity(
        0.5, 0.0, 0.0, executed_motion=0.5, position=(5.0, 0.0)
    ) is None

    tracker.begin("G123", position=(5.0, 0.0))
    clock.now = 0.1
    # Requested non-zero, but plant cmd_motion did not advance: the control gate
    # rejected it. Position jitter/drift must not launder it into causation.
    assert tracker.record_velocity(
        0.5, 0.0, 0.0, executed_motion=0.0, position=(5.4, 0.0)
    ) is None
    clock.now = 0.2
    final = tracker.finalize("G123", status="failed", position=(5.8, 0.0))

    assert final is not None
    assert final["nonzero_cmd_count"] == 0
    assert final["cmd_motion_count"] == 0.0
    assert final["moved_distance_m"] == 0.0
    assert final["actual_velocity_observed"] is False


def test_confirmed_command_without_displacement_is_not_actor_caused() -> None:
    clock = _Clock()
    tracker = GoalMotionTracker(clock)
    tracker.begin("G123", position=(0.0, 0.0))
    tracker.record_velocity(
        0.3, 0.0, 0.0, executed_motion=0.3, position=(0.0, 0.0)
    )
    clock.now = 0.1
    final = tracker.finalize("G123", status="failed", position=(0.0, 0.0))

    assert final is not None
    assert final["actual_velocity_observed"] is True
    assert final["moved_distance_m"] <= NAV_GOAL_DISPLACEMENT_EPS_M
    assert final["actor_caused"] is False


def test_tracker_keeps_g123_and_g124_lifecycles_isolated() -> None:
    clock = _Clock()
    tracker = GoalMotionTracker(clock)

    tracker.begin("G123", position=(0.0, 0.0))
    tracker.record_velocity(
        0.2, 0.0, 0.0, executed_motion=0.2, position=(0.0, 0.0)
    )
    g123 = tracker.finalize("G123", status="cancelled", position=(0.0, 0.0))

    tracker.begin("G124", position=(10.0, 0.0))
    # A stale G123 finalizer cannot close or mutate active G124.
    assert tracker.finalize("G123", status="failed", position=(10.0, 0.0)) is None
    tracker.record_velocity(
        0.3, 0.0, 0.0, executed_motion=0.3, position=(10.0, 0.0)
    )
    g124 = tracker.finalize("G124", status="failed", position=(10.0, 0.0))

    assert g123 is not None and g124 is not None
    assert g123["goal_id"] == "G123"
    assert g123["cmd_motion_count"] == pytest.approx(0.2)
    assert g124["goal_id"] == "G124"
    assert g124["cmd_motion_count"] == pytest.approx(0.3)


def test_proxy_exposes_required_fields_and_rejects_late_cross_goal_sample() -> None:
    proxy = Go2ROS2Proxy()

    proxy.begin_navigation_goal("G123", target_xy=(3.0, 7.5))
    assert proxy._ingest_navigation_telemetry(
        _event(
            "G123", 1, event="accepted", status="active",
            count=0, motion=0.0, duration=0.0, distance=0.0,
        )
    )
    assert proxy._ingest_navigation_telemetry(
        _event(
            "G123", 2, event="velocity", status="active",
            count=2, motion=0.8, duration=0.1, distance=0.05,
        )
    )
    assert proxy._ingest_navigation_telemetry(
        _event(
            "G123", 3, event="finalized", status="succeeded",
            count=2, motion=0.8, duration=0.2, distance=0.1,
        )
    )

    proxy.begin_navigation_goal("G124", target_xy=(1.0, 2.0))
    # G123 is terminal and G124 is active. A delayed old sample is rejected
    # instead of advancing the cumulative actor counter in G124's window.
    assert not proxy._ingest_navigation_telemetry(
        _event(
            "G123", 4, event="velocity", status="active",
            count=99, motion=99.0, duration=99.0, distance=99.0,
        )
    )
    assert proxy._ingest_navigation_telemetry(
        _event(
            "G124", 1, event="accepted", status="active",
            count=0, motion=0.0, duration=0.0, distance=0.0,
        )
    )
    assert proxy._ingest_navigation_telemetry(
        _event(
            "G124", 2, event="velocity", status="active",
            count=1, motion=0.3, duration=0.0, distance=0.0,
        )
    )

    g123 = proxy.get_navigation_telemetry("G123")
    g124 = proxy.get_navigation_telemetry("G124")
    required = {
        "goal_id",
        "nonzero_cmd_count",
        "cmd_motion_count",
        "moved_distance_m",
        "actual_velocity_observed",
    }
    assert required <= g124.keys()
    assert g123["cmd_motion_count"] == pytest.approx(0.8)
    assert g124["cmd_motion_count"] == pytest.approx(0.3)
    assert g124["nonzero_cmd_count"] == 1
    assert g124["actual_velocity_observed"] is True
    assert proxy.cmd_motion() == pytest.approx(1.1)


@pytest.mark.parametrize(
    ("count", "motion"),
    [(0, 0.5), (1, 0.0)],
)
def test_proxy_rejects_inconsistent_command_count_and_motion(
    count: int, motion: float
) -> None:
    proxy = Go2ROS2Proxy()
    proxy.begin_navigation_goal("G123", target_xy=(3.0, 7.5))

    assert not proxy._ingest_navigation_telemetry(
        _event(
            "G123", 1, event="velocity", status="active",
            count=count, motion=motion, duration=0.1, distance=0.1,
        )
    )
    assert proxy.cmd_motion() == 0.0
    stats = proxy.get_navigation_telemetry("G123")
    assert stats["nonzero_cmd_count"] == 0
    assert stats["cmd_motion_count"] == 0.0


def test_explicit_failed_navigation_goal_is_finalized_without_ros() -> None:
    proxy = Go2ROS2Proxy()
    assert proxy.navigate_to(1.0, 2.0, timeout=0.01, goal_id="GFAIL") is False

    stats = proxy.get_navigation_telemetry("GFAIL")
    assert stats["goal_id"] == "GFAIL"
    assert stats["status"] == "failed"
    assert stats["event"] == "finalized_local"
    assert stats["actual_velocity_observed"] is False


def test_bridge_routes_all_motor_writes_through_telemetry_boundary() -> None:
    """Static wiring check; importing the bridge would require ROS2."""
    bridge = (
        Path(__file__).resolve().parents[2] / "scripts" / "go2_vnav_bridge.py"
    ).read_text()
    assert NAV_GOAL_TELEMETRY_VERSION == 1
    assert "def _goal_control_cb" in bridge
    assert "def _apply_velocity" in bridge
    # Exactly one actual plant boundary remains, inside _apply_velocity.
    assert bridge.count("self._go2.set_velocity(") == 2  # call + explanatory doc literal
    assert 'source="path_follower"' in bridge


def test_bridge_initializer_does_not_shadow_ros_string_message() -> None:
    """A local String import makes every earlier String use unbound at runtime."""
    bridge_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "go2_vnav_bridge.py"
    )
    table = symtable.symtable(
        bridge_path.read_text(encoding="utf-8"),
        str(bridge_path),
        "exec",
    )
    bridge_class = next(
        child for child in table.get_children()
        if child.get_name() == "Go2VNavBridge"
    )
    initializer = next(
        child for child in bridge_class.get_children()
        if child.get_name() == "__init__"
    )
    string_symbol = initializer.lookup("String")

    assert string_symbol.is_global()
    assert not string_symbol.is_local()
