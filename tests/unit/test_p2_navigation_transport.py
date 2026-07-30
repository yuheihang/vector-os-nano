# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P2 fail-closed FAR segment and bridge motion-boundary contracts."""
from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from vector_os_nano.hardware.sim import go2_ros2_proxy as proxy_module
from vector_os_nano.hardware.sim.go2_ros2_proxy import (
    CURRENT_GOAL_TOPIC,
    DOOR_PATH_TOPIC,
    EXECUTED_PATH_TOPIC,
    FAR_GLOBAL_PATH_TOPIC,
    FAR_ROUTE_MARKER_TOPIC,
    FAR_VGRAPH_MARKER_TOPIC,
    LOCAL_PLANNER_PATH_TOPIC,
    NAV_SEGMENT_ACK_TOPIC,
    NAV_SEGMENT_CONTROL_TOPIC,
    NAV_SEGMENT_CONTROL_VERSION,
    Go2ROS2Proxy,
    NavigationSegmentConstraints,
    _NavigationProgressWatchdog,
)
from vector_os_nano.skills.navigate import _segment_failure


class _Publisher:
    def __init__(self) -> None:
        self.messages: list[object] = []

    def publish(self, msg: object) -> None:
        self.messages.append(msg)


def test_progress_watchdog_accepts_slow_cumulative_motion() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
    )
    distance = 5.0
    now = 0.0
    assert watchdog.stalled(distance, now) is False
    for _ in range(60):
        # 0.04 m/sample is below the old 0.1 m requirement but is steady,
        # meaningful progress over the configured window.
        distance -= 0.04
        now += 0.5
        assert watchdog.stalled(distance, now) is False


def test_progress_watchdog_stalls_only_after_stationary_timeout() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
    )
    assert watchdog.stalled(5.0, 0.0) is False
    assert watchdog.stalled(5.0, 9.9) is False
    assert watchdog.stalled(5.0, 10.0) is True


def test_progress_watchdog_accepts_turn_in_place_before_translation() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
    )
    distance = 5.0
    heading_error = math.radians(170.0)
    assert watchdog.stalled(
        distance,
        0.0,
        heading_error_rad=heading_error,
    ) is False

    # A loaded quadruped can take longer than one linear-progress window to
    # reverse its heading.  Cumulative heading-error reduction is real progress
    # even though the goal distance has not changed yet.
    for second in range(1, 31):
        heading_error -= math.radians(2.0)
        assert watchdog.stalled(
            distance,
            float(second),
            heading_error_rad=heading_error,
        ) is False


def test_progress_watchdog_does_not_accept_static_heading() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
    )
    heading_error = math.radians(170.0)
    assert watchdog.stalled(
        5.0,
        0.0,
        heading_error_rad=heading_error,
    ) is False
    assert watchdog.stalled(
        5.0,
        10.0,
        heading_error_rad=heading_error,
    ) is True


def test_progress_watchdog_accepts_local_detour_away_from_goal() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
    )
    assert watchdog.stalled(
        5.0,
        0.0,
        heading_error_rad=0.0,
        position_xy=(0.0, 0.0),
        heading_rad=0.0,
    ) is False

    # The local path first moves laterally around an obstacle, so radial
    # distance and direct goal-heading error both get worse.  Real body motion
    # must keep the stationary watchdog from rejecting the valid detour.
    for second in range(1, 31):
        lateral_y = 0.05 * second
        assert watchdog.stalled(
            5.0 + 0.01 * second,
            float(second),
            heading_error_rad=math.radians(0.5 * second),
            position_xy=(0.0, lateral_y),
            heading_rad=math.pi / 2.0,
        ) is False


def test_progress_watchdog_stalls_when_pose_and_goal_error_are_static() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
    )
    kwargs = {
        "heading_error_rad": math.radians(20.0),
        "position_xy": (1.0, 2.0),
        "heading_rad": math.radians(10.0),
    }
    assert watchdog.stalled(5.0, 0.0, **kwargs) is False
    assert watchdog.stalled(5.0, 10.0, **kwargs) is True


def test_progress_watchdog_does_not_charge_unobserved_gui_time() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
        max_observation_gap_s=1.0,
    )
    kwargs = {
        "heading_error_rad": math.radians(170.0),
        "position_xy": (1.0, 2.0),
        "heading_rad": 0.0,
    }
    assert watchdog.stalled(5.0, 0.0, **kwargs) is False

    # A GUI frame can block the executor for five wall-clock seconds.  Six
    # sparse observations represent only six seconds of inspected stationary
    # state, so a 30-second wall interval must not trigger a 10-second stall.
    for now in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
        assert watchdog.stalled(5.0, now, **kwargs) is False

    for now in (35.0, 40.0, 45.0):
        assert watchdog.stalled(5.0, now, **kwargs) is False
    assert watchdog.stalled(5.0, 50.0, **kwargs) is True


def test_progress_watchdog_accepts_sparse_but_valid_turning() -> None:
    watchdog = _NavigationProgressWatchdog(
        progress_threshold_m=0.3,
        timeout_s=10.0,
        heading_progress_threshold_rad=math.radians(3.0),
        max_observation_gap_s=1.0,
    )
    assert watchdog.stalled(
        5.0,
        0.0,
        heading_error_rad=math.radians(170.0),
        position_xy=(1.0, 2.0),
        heading_rad=0.0,
    ) is False
    for sample in range(1, 13):
        heading = math.radians(0.6 * sample)
        assert watchdog.stalled(
            5.0,
            float(sample * 5),
            heading_error_rad=math.radians(170.0) - heading,
            position_xy=(1.0, 2.0),
            heading_rad=heading,
        ) is False


class _LoopbackSegmentPublisher(_Publisher):
    def __init__(self, proxy: Go2ROS2Proxy) -> None:
        super().__init__()
        self.proxy = proxy

    def publish(self, msg: object) -> None:
        super().publish(msg)
        payload = json.loads(getattr(msg, "data"))
        ack = _String()
        ack.data = json.dumps(
            {
                "version": NAV_SEGMENT_CONTROL_VERSION,
                "event": "APPLIED",
                "goal_id": payload["goal_id"],
                "segment_id": payload["segment_id"],
                "reason": "",
            }
        )
        self.proxy._segment_ack_cb(ack)


class _RetryingAckPublisher(_Publisher):
    """Inject a stale/rejected first ACK, then apply the exact retry."""

    def __init__(
        self,
        proxy: Go2ROS2Proxy,
        *,
        first_event: str,
        wrong_segment: bool = False,
    ) -> None:
        super().__init__()
        self.proxy = proxy
        self.first_event = first_event
        self.wrong_segment = wrong_segment
        self.set_count = 0

    def publish(self, msg: object) -> None:
        super().publish(msg)
        payload = json.loads(getattr(msg, "data"))
        if payload["event"] != "set":
            return
        self.set_count += 1
        ack = _String()
        ack.data = json.dumps(
            {
                "version": NAV_SEGMENT_CONTROL_VERSION,
                "event": (
                    self.first_event if self.set_count == 1 else "APPLIED"
                ),
                "goal_id": payload["goal_id"],
                "segment_id": (
                    "S-stale-generation"
                    if self.set_count == 1 and self.wrong_segment
                    else payload["segment_id"]
                ),
                "reason": "injected_test_ack",
            }
        )
        self.proxy._segment_ack_cb(ack)


class _CancelDuringAckPublisher(_Publisher):
    """Cancel synchronously before a matching APPLIED ACK can be consumed."""

    def __init__(self, proxy: Go2ROS2Proxy) -> None:
        super().__init__()
        self.proxy = proxy
        self.cancelled = False

    def publish(self, msg: object) -> None:
        super().publish(msg)
        payload = json.loads(getattr(msg, "data"))
        if payload["event"] != "set" or self.cancelled:
            return
        self.cancelled = True
        self.proxy.stop_navigation()
        ack = _String()
        ack.data = json.dumps(
            {
                "version": NAV_SEGMENT_CONTROL_VERSION,
                "event": "APPLIED",
                "goal_id": payload["goal_id"],
                "segment_id": payload["segment_id"],
                "reason": "",
            }
        )
        self.proxy._segment_ack_cb(ack)


class _StampClock:
    class _Now:
        @staticmethod
        def to_msg() -> object:
            return object()

    @staticmethod
    def now() -> "_StampClock._Now":
        return _StampClock._Now()


class _Node:
    @staticmethod
    def get_clock() -> _StampClock:
        return _StampClock()


class _Header:
    def __init__(self) -> None:
        self.stamp = None
        self.frame_id = ""


class _Point:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class _Orientation:
    def __init__(self) -> None:
        self.w = 0.0


class _Pose:
    def __init__(self) -> None:
        self.position = _Point()
        self.orientation = _Orientation()


class _PoseStamped:
    def __init__(self) -> None:
        self.header = _Header()
        self.pose = _Pose()


class _Path:
    def __init__(self) -> None:
        self.header = _Header()
        self.poses: list[_PoseStamped] = []


class _Marker:
    def __init__(self) -> None:
        self.ns = ""
        self.points: list[_Point] = []


class _MarkerArray:
    def __init__(self) -> None:
        self.markers: list[_Marker] = []


class _String:
    def __init__(self) -> None:
        self.data = ""


@pytest.fixture
def fake_ros_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    std_msg = types.ModuleType("std_msgs.msg")
    std_msg.String = _String
    std = types.ModuleType("std_msgs")
    std.msg = std_msg
    geometry_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msg.PoseStamped = _PoseStamped
    geometry = types.ModuleType("geometry_msgs")
    geometry.msg = geometry_msg
    nav_msg = types.ModuleType("nav_msgs.msg")
    nav_msg.Path = _Path
    nav = types.ModuleType("nav_msgs")
    nav.msg = nav_msg
    visualization_msg = types.ModuleType("visualization_msgs.msg")
    visualization_msg.Marker = _Marker
    visualization_msg.MarkerArray = _MarkerArray
    visualization = types.ModuleType("visualization_msgs")
    visualization.msg = visualization_msg
    monkeypatch.setitem(sys.modules, "std_msgs", std)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msg)
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msg)
    monkeypatch.setitem(sys.modules, "nav_msgs", nav)
    monkeypatch.setitem(sys.modules, "nav_msgs.msg", nav_msg)
    monkeypatch.setitem(sys.modules, "visualization_msgs", visualization)
    monkeypatch.setitem(sys.modules, "visualization_msgs.msg", visualization_msg)


def _point_stamped(x: float, y: float) -> object:
    return types.SimpleNamespace(point=types.SimpleNamespace(x=x, y=y))


def _route_marker_ending_at(x: float, y: float) -> _Marker:
    marker = _Marker()
    point = _Point()
    point.x = x
    point.y = y
    marker.points.append(point)
    return marker


def _vgraph_marker_array(node_count: int, *, namespace: str = "global_vertex") -> _MarkerArray:
    marker_array = _MarkerArray()
    marker = _Marker()
    marker.ns = namespace
    marker.points = [_Point() for _ in range(node_count)]
    marker_array.markers.append(marker)
    return marker_array


def test_far_vgraph_readiness_requires_nonempty_global_vertex_marker() -> None:
    proxy = Go2ROS2Proxy()

    assert proxy.far_vgraph_ready() is False
    assert proxy.far_vgraph_diagnostics()["status"] == "waiting_for_marker"

    # Other visualization namespaces cannot prove that FAR accepts goals.
    proxy._far_vgraph_marker_cb(
        _vgraph_marker_array(4, namespace="updating_vertex")
    )
    diagnostics = proxy.far_vgraph_diagnostics()
    assert diagnostics["ready"] is False
    assert diagnostics["node_count"] == 0
    assert diagnostics["global_vertex_marker_seen"] is False
    assert diagnostics["status"] == "empty_graph"

    proxy._far_vgraph_marker_cb(_vgraph_marker_array(3))
    diagnostics = proxy.far_vgraph_diagnostics()
    assert proxy.far_vgraph_ready() is True
    assert diagnostics["ready"] is True
    assert diagnostics["node_count"] == 3
    assert diagnostics["global_vertex_marker_seen"] is True
    assert diagnostics["message_count"] == 2
    assert diagnostics["last_marker_time"] > 0.0
    assert diagnostics["marker_age_s"] is not None
    assert diagnostics["status"] == "ready"


def test_empty_far_vgraph_update_clears_prior_readiness() -> None:
    proxy = Go2ROS2Proxy()
    proxy._far_vgraph_marker_cb(_vgraph_marker_array(2))
    assert proxy.far_vgraph_ready() is True

    proxy._far_vgraph_marker_cb(_vgraph_marker_array(0))

    diagnostics = proxy.far_vgraph_diagnostics()
    assert proxy.far_vgraph_ready() is False
    assert diagnostics["node_count"] == 0
    assert diagnostics["global_vertex_marker_seen"] is True
    assert diagnostics["status"] == "empty_graph"


def test_disconnect_clears_far_vgraph_readiness_state() -> None:
    proxy = Go2ROS2Proxy()
    proxy._far_vgraph_subscription = object()
    proxy._far_vgraph_marker_cb(_vgraph_marker_array(2))

    proxy.disconnect()

    assert proxy._far_vgraph_subscription is None
    assert proxy.far_vgraph_ready() is False
    assert proxy.far_vgraph_diagnostics() == {
        "topic": FAR_VGRAPH_MARKER_TOPIC,
        "status": "waiting_for_marker",
        "ready": False,
        "node_count": 0,
        "global_vertex_marker_seen": False,
        "message_count": 0,
        "last_marker_time": 0.0,
        "marker_age_s": None,
    }


def test_far_probe_requires_matching_global_path_before_waypoint() -> None:
    proxy = Go2ROS2Proxy()
    generation = proxy._begin_far_goal_probe(
        (6.3, 3.0),
        match_tolerance_m=99.0,
    )
    proxy._mark_far_goal_published(generation)

    # A freshly replayed dining-room response is still stale for this living
    # goal.  Freshness must not substitute for current-goal association.
    proxy._far_route_marker_cb(_route_marker_ending_at(6.6, 8.0))
    for _ in range(3):
        proxy._waypoint_cb(_point_stamped(6.6, 8.0))

    snapshot = proxy._far_probe_snapshot(generation)
    assert snapshot["far_response_associated"] is False
    assert snapshot["fresh_waypoint_count"] == 3
    assert snapshot["far_path_mismatch_count"] == 1
    assert snapshot["far_path_match_tolerance_m"] == 1.0
    assert snapshot["planner_goal_xy"] == [6.3, 3.0]
    assert snapshot["observed_waypoint_xy"] == [6.6, 8.0]
    assert snapshot["far_path_endpoint_xy"] == [6.6, 8.0]


def test_far_probe_preserves_legitimate_intermediate_waypoint() -> None:
    proxy = Go2ROS2Proxy()
    generation = proxy._begin_far_goal_probe(
        (3.2, 2.5),
        match_tolerance_m=0.75,
    )
    proxy._mark_far_goal_published(generation)
    proxy._far_route_marker_cb(_route_marker_ending_at(3.2, 2.5))

    # Intermediate waypoints can be far from the final target; the matching
    # global-path endpoint, not waypoint proximity, proves their generation.
    proxy._waypoint_cb(_point_stamped(9.3, 3.3))

    snapshot = proxy._far_probe_snapshot(generation)
    assert snapshot["far_response_associated"] is True
    assert snapshot["observed_waypoint_xy"] == [9.3, 3.3]
    assert snapshot["far_path_endpoint_xy"] == [3.2, 2.5]


def test_far_probe_does_not_retroactively_bind_pre_path_waypoint() -> None:
    proxy = Go2ROS2Proxy()
    generation = proxy._begin_far_goal_probe(
        (3.2, 2.5),
        match_tolerance_m=1.0,
    )
    proxy._mark_far_goal_published(generation)
    proxy._waypoint_cb(_point_stamped(6.6, 8.0))
    proxy._far_route_marker_cb(_route_marker_ending_at(3.2, 2.5))

    assert proxy._far_probe_snapshot(generation)["far_response_associated"] is False

    proxy._waypoint_cb(_point_stamped(9.3, 3.3))
    assert proxy._far_probe_snapshot(generation)["far_response_associated"] is True


def test_empty_far_route_marker_clears_association_proof() -> None:
    proxy = Go2ROS2Proxy()
    generation = proxy._begin_far_goal_probe(
        (3.2, 2.5),
        match_tolerance_m=1.0,
    )
    proxy._mark_far_goal_published(generation)
    proxy._far_route_marker_cb(_route_marker_ending_at(3.2, 2.5))
    proxy._far_route_marker_cb(_Marker())
    proxy._waypoint_cb(_point_stamped(9.3, 3.3))

    snapshot = proxy._far_probe_snapshot(generation)
    assert snapshot["far_response_associated"] is False
    assert snapshot["far_empty_route_count"] == 1
    assert snapshot["far_path_endpoint_xy"] is None


def test_mismatching_far_route_marker_supersedes_prior_match() -> None:
    proxy = Go2ROS2Proxy()
    generation = proxy._begin_far_goal_probe(
        (3.2, 2.5),
        match_tolerance_m=1.0,
    )
    proxy._mark_far_goal_published(generation)
    proxy._far_route_marker_cb(_route_marker_ending_at(3.2, 2.5))
    proxy._far_route_marker_cb(_route_marker_ending_at(6.6, 8.0))
    proxy._waypoint_cb(_point_stamped(6.6, 8.0))

    snapshot = proxy._far_probe_snapshot(generation)
    assert snapshot["far_response_associated"] is False
    assert snapshot["far_path_mismatch_count"] == 1
    assert snapshot["far_path_endpoint_xy"] == [6.6, 8.0]


def test_segment_failure_keeps_stale_planner_diagnosis() -> None:
    class _Base:
        @staticmethod
        def get_navigation_goal_state(_goal_id: str) -> dict[str, object]:
            return {
                "reason": "stale_planner_response",
                "planner_goal_xy": [6.3, 3.0],
                "observed_waypoint_xy": [6.6, 8.0],
                "far_path_endpoint_xy": [6.6, 8.0],
            }

    code, message = _segment_failure(
        _Base(),
        "G-STALE",
        index=0,
        label="living_room door_pre",
    )
    assert code == "stale_planner_response"
    assert "expected goal [6.3, 3.0]" in message
    assert "path endpoint [6.6, 8.0]" in message


def test_constraints_disallow_reverse_and_cap_planar_speed() -> None:
    constraints = NavigationSegmentConstraints(
        goal_id="G-P2",
        segment_id="S-P2",
        kind="door_center",
        speed_limit_mps=0.2,
        allow_reverse=False,
        tolerance=0.22,
    )

    vx, vy, vyaw = constraints.constrain_velocity(-0.4, 0.3, 0.7)

    assert vx == 0.0
    assert vy == pytest.approx(0.2)
    assert vyaw == pytest.approx(0.7)
    assert constraints.to_payload() == {
        "version": NAV_SEGMENT_CONTROL_VERSION,
        "event": "set",
        "goal_id": "G-P2",
        "segment_id": "S-P2",
        "kind": "door_center",
        "speed_limit_mps": 0.2,
        "allow_reverse": False,
        "tolerance": 0.22,
    }


def test_door_policy_without_cap_keeps_normal_adaptive_forward_speed() -> None:
    constraints = NavigationSegmentConstraints(
        goal_id="G-P2-NORMAL",
        segment_id="S-P2-NORMAL",
        kind="door_center",
        speed_limit_mps=None,
        allow_reverse=False,
        tolerance=0.22,
    )

    forward = constraints.constrain_velocity(0.55, 0.12, 0.4)
    reverse = constraints.constrain_velocity(-0.55, 0.12, 0.4)

    assert forward == pytest.approx((0.55, 0.12, 0.4))
    assert reverse == pytest.approx((0.0, 0.12, 0.4))
    assert constraints.to_payload()["speed_limit_mps"] is None


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 99, "event": "set", "kind": "door_center"},
        {
            "version": NAV_SEGMENT_CONTROL_VERSION,
            "event": "set",
            "kind": "door_center",
        },
        {
            "version": NAV_SEGMENT_CONTROL_VERSION,
            "event": "set",
            "kind": "unknown",
        },
        {
            "version": NAV_SEGMENT_CONTROL_VERSION,
            "event": "set",
            "kind": "door_center",
            "speed_limit_mps": float("nan"),
        },
        {
            "version": NAV_SEGMENT_CONTROL_VERSION,
            "event": "set",
            "kind": "door_center",
            "allow_reverse": "false",
        },
    ],
)
def test_constraints_reject_malformed_control_payload(payload: dict) -> None:
    with pytest.raises(ValueError):
        NavigationSegmentConstraints.from_payload(payload)


def test_proxy_setter_publishes_goal_scoped_set_and_clear(
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = _Node()
    proxy._active_navigation_goal_id = "G-DOOR"
    proxy._segment_ack_subscription = object()
    proxy._segment_control_pub = _LoopbackSegmentPublisher(proxy)
    proxy._current_goal_pub = _Publisher()

    assert proxy.set_navigation_segment_constraints(
        kind="door_center",
        speed_limit_mps=0.16,
        allow_reverse=False,
        tolerance=0.2,
        goal_id="G-DOOR",
    )
    assert proxy.clear_navigation_segment_constraints(goal_id="G-DOOR")

    payloads = [
        json.loads(message.data)
        for message in proxy._segment_control_pub.messages
    ]
    assert payloads[0]["event"] == "set"
    assert payloads[0]["goal_id"] == "G-DOOR"
    assert payloads[0]["segment_id"].startswith("S")
    assert payloads[0]["allow_reverse"] is False
    assert payloads[0]["speed_limit_mps"] == pytest.approx(0.16)
    assert payloads[1] == {
        "version": NAV_SEGMENT_CONTROL_VERSION,
        "event": "clear",
        "goal_id": "G-DOOR",
        "segment_id": payloads[0]["segment_id"],
    }


@pytest.mark.parametrize(
    ("first_event", "wrong_segment"),
    [
        ("REJECTED", False),
        ("APPLIED", True),
    ],
)
def test_proxy_retries_until_exact_segment_generation_is_applied(
    monkeypatch: pytest.MonkeyPatch,
    fake_ros_messages: None,
    first_event: str,
    wrong_segment: bool,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = _Node()
    proxy._active_navigation_goal_id = "G-RETRY"
    proxy._segment_ack_subscription = object()
    publisher = _RetryingAckPublisher(
        proxy,
        first_event=first_event,
        wrong_segment=wrong_segment,
    )
    proxy._segment_control_pub = publisher
    monkeypatch.setattr(proxy_module, "NAV_SEGMENT_ACK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(proxy_module, "NAV_SEGMENT_ACK_RETRY_S", 0.001)

    assert proxy.set_navigation_segment_constraints(
        kind="door_center",
        speed_limit_mps=0.16,
        allow_reverse=False,
        tolerance=0.2,
        goal_id="G-RETRY",
    )
    assert publisher.set_count >= 2
    assert proxy._pending_segment_constraints is None
    assert proxy._active_segment_constraints is not None


def test_cancel_during_segment_ack_cannot_submit_far_goal(
    monkeypatch: pytest.MonkeyPatch,
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._segment_ack_subscription = object()
    publisher = _CancelDuringAckPublisher(proxy)
    proxy._segment_control_pub = publisher
    goal_publish_attempts: list[tuple[float, float]] = []
    velocity: list[tuple[float, float, float]] = []
    monkeypatch.setattr(proxy_module.os, "remove", lambda _path: None)
    monkeypatch.setattr(
        proxy,
        "_publish_goal_point",
        lambda x, y: goal_publish_attempts.append((x, y)),
    )
    monkeypatch.setattr(
        proxy,
        "set_velocity",
        lambda x, y, z: velocity.append((x, y, z)),
    )

    arrived = proxy.navigate_to(
        6.0,
        3.0,
        timeout_s=10.0,
        goal_id="G-CANCEL-DURING-ACK",
        allow_door_fallback=False,
        waypoint_kind="door_center",
        speed_limit_mps=0.16,
        allow_reverse=False,
        arrival_tolerance=0.22,
    )

    assert arrived is False
    assert publisher.cancelled
    assert goal_publish_attempts == []
    assert velocity
    assert velocity[-1] == (0.0, 0.0, 0.0)
    assert proxy._pending_segment_constraints is None
    assert proxy._active_segment_constraints is None
    assert proxy._active_navigation_goal_id is None


class _AdvancingClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 0.55
        return self.value


def test_navigate_to_already_arrived_does_not_publish_far_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    goal_publish_attempts: list[tuple[float, float]] = []
    velocity: list[tuple[float, float, float]] = []
    monkeypatch.setattr(proxy, "get_position", lambda: (3.05, 4.02, 0.28))
    monkeypatch.setattr(
        proxy,
        "_publish_goal_point",
        lambda x, y: goal_publish_attempts.append((x, y)),
    )
    monkeypatch.setattr(
        proxy,
        "set_velocity",
        lambda x, y, z: velocity.append((x, y, z)),
    )
    monkeypatch.setattr(proxy_module.os, "remove", lambda _path: None)

    with patch("builtins.open", mock_open()):
        arrived = proxy.navigate_to(
            3.0,
            4.0,
            timeout_s=10.0,
            goal_id="G-ALREADY-ARRIVED",
        )

    assert arrived is True
    assert goal_publish_attempts == []
    assert velocity == [(0.0, 0.0, 0.0)]
    assert proxy._current_goal_state["state"] == "ARRIVED"
    assert proxy.get_navigation_telemetry("G-ALREADY-ARRIVED")["status"] == "succeeded"


def test_required_far_segment_no_path_stops_without_door_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fake_ros_messages: None,
    tmp_path: Path,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._segment_ack_subscription = object()
    proxy._segment_control_pub = _LoopbackSegmentPublisher(proxy)
    proxy._last_waypoint_time = 0.0
    velocity: list[tuple[float, float, float]] = []
    fallback_calls: list[tuple[float, float, float]] = []
    removed_paths: list[str] = []
    clock = _AdvancingClock()
    active_file = tmp_path / "nav_active"

    monkeypatch.setenv("VECTOR_NAV_ACTIVE_FILE", str(active_file))
    monkeypatch.setattr(proxy_module.time, "time", clock)
    monkeypatch.setattr(proxy_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(proxy_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(
        proxy_module.os,
        "remove",
        lambda path: removed_paths.append(str(path)),
    )
    monkeypatch.setattr(proxy, "_publish_goal_point", lambda _x, _y: None)
    monkeypatch.setattr(proxy, "set_velocity", lambda x, y, z: velocity.append((x, y, z)))
    monkeypatch.setattr(
        proxy,
        "_navigate_via_doors",
        lambda x, y, timeout, **_kwargs: (
            fallback_calls.append((x, y, timeout)) or True
        ),
    )

    with patch("builtins.open", mock_open()):
        arrived = proxy.navigate_to(
            6.0,
            3.0,
            timeout_s=10.0,
            goal_id="G-REQUIRED",
            allow_door_fallback=False,
            waypoint_kind="door_center",
            speed_limit_mps=0.16,
            allow_reverse=False,
            arrival_tolerance=0.22,
        )

    assert arrived is False
    assert fallback_calls == []
    assert velocity[-1] == (0.0, 0.0, 0.0)
    assert str(active_file) in removed_paths
    assert proxy._current_goal_state["state"] == "NO_PATH"
    assert proxy._active_segment_constraints is None
    assert proxy.get_navigation_telemetry("G-REQUIRED")["status"] == "failed"


def test_required_far_segment_rejects_replayed_waypoint_as_stale(
    monkeypatch: pytest.MonkeyPatch,
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._segment_ack_subscription = object()
    proxy._segment_control_pub = _LoopbackSegmentPublisher(proxy)
    velocity: list[tuple[float, float, float]] = []
    fallback_calls: list[tuple[float, float, float]] = []
    clock = _AdvancingClock()
    stale_route_marker = _route_marker_ending_at(6.6, 8.0)
    stale_waypoint = _point_stamped(6.6, 8.0)

    monkeypatch.setattr(proxy_module.time, "time", clock)
    monkeypatch.setattr(proxy_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(proxy_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(proxy_module.os, "remove", lambda _path: None)
    monkeypatch.setattr(
        proxy,
        "_publish_goal_point",
        lambda _x, _y: (
            proxy._far_route_marker_cb(stale_route_marker),
            proxy._waypoint_cb(stale_waypoint),
        ),
    )
    monkeypatch.setattr(
        proxy,
        "set_velocity",
        lambda x, y, z: velocity.append((x, y, z)),
    )
    monkeypatch.setattr(
        proxy,
        "_navigate_via_doors",
        lambda x, y, timeout, **_kwargs: (
            fallback_calls.append((x, y, timeout)) or True
        ),
    )

    with patch("builtins.open", mock_open()):
        arrived = proxy.navigate_to(
            6.3,
            3.0,
            timeout_s=30.0,
            goal_id="G-STALE-REPLAY",
            allow_door_fallback=False,
            waypoint_kind="door_pre",
            speed_limit_mps=0.35,
            allow_reverse=True,
            arrival_tolerance=0.3,
        )

    assert arrived is False
    assert fallback_calls == []
    assert velocity[-1] == (0.0, 0.0, 0.0)
    state = proxy.get_navigation_goal_state("G-STALE-REPLAY")
    assert state["state"] == "NO_PATH"
    assert state["reason"] == "stale_planner_response"
    assert state["planner_goal_xy"] == [6.3, 3.0]
    assert state["observed_waypoint_xy"] == [6.6, 8.0]
    assert state["far_path_endpoint_xy"] == [6.6, 8.0]
    # The probe terminates in a few simulated seconds; it never enters the
    # former 30-second Phase-2 stall window.
    assert clock.value < 10.0


def test_connected_proxy_refuses_segment_when_policy_channel_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._segment_control_pub = None
    velocity: list[tuple[float, float, float]] = []
    goal_publish_attempts: list[tuple[float, float]] = []
    monkeypatch.setattr(proxy, "set_velocity", lambda x, y, z: velocity.append((x, y, z)))
    monkeypatch.setattr(proxy_module.os, "remove", lambda _path: None)
    monkeypatch.setattr(
        proxy,
        "_publish_goal_point",
        lambda x, y: goal_publish_attempts.append((x, y)),
    )

    arrived = proxy.navigate_to(
        6.0,
        3.0,
        timeout_s=10.0,
        goal_id="G-NO-POLICY-CHANNEL",
        allow_door_fallback=False,
        waypoint_kind="door_center",
        speed_limit_mps=0.16,
        allow_reverse=False,
        arrival_tolerance=0.22,
    )

    assert arrived is False
    assert goal_publish_attempts == []
    assert velocity == [(0.0, 0.0, 0.0)]
    assert proxy._current_goal_state["state"] == "ERROR"
    assert (
        proxy._current_goal_state["reason"]
        == "segment_constraint_unavailable"
    )
    assert proxy._active_navigation_goal_id is None
    assert proxy.get_navigation_telemetry("G-NO-POLICY-CHANNEL")["status"] == "failed"


def test_publish_without_matching_bridge_ack_never_submits_far_goal(
    monkeypatch: pytest.MonkeyPatch,
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._segment_control_pub = _Publisher()
    proxy._segment_ack_subscription = object()
    goal_publish_attempts: list[tuple[float, float]] = []
    velocity: list[tuple[float, float, float]] = []
    monkeypatch.setattr(proxy_module, "NAV_SEGMENT_ACK_TIMEOUT_S", 0.01)
    monkeypatch.setattr(proxy_module, "NAV_SEGMENT_ACK_RETRY_S", 0.002)
    monkeypatch.setattr(proxy_module.os, "remove", lambda _path: None)
    monkeypatch.setattr(
        proxy,
        "_publish_goal_point",
        lambda x, y: goal_publish_attempts.append((x, y)),
    )
    monkeypatch.setattr(
        proxy,
        "set_velocity",
        lambda x, y, z: velocity.append((x, y, z)),
    )

    arrived = proxy.navigate_to(
        6.0,
        3.0,
        timeout_s=10.0,
        goal_id="G-NO-ACK",
        allow_door_fallback=False,
        waypoint_kind="door_center",
        speed_limit_mps=0.16,
        allow_reverse=False,
        arrival_tolerance=0.22,
    )

    assert arrived is False
    assert goal_publish_attempts == []
    assert velocity == [(0.0, 0.0, 0.0)]
    assert proxy._current_goal_state["reason"] == "segment_constraint_unavailable"


def test_invalid_atomic_segment_policy_finalizes_owned_goal() -> None:
    proxy = Go2ROS2Proxy()

    with pytest.raises(ValueError, match="unsupported navigation waypoint kind"):
        proxy.navigate_to(
            1.0,
            2.0,
            goal_id="G-INVALID",
            waypoint_kind="wall_shortcut",
        )

    assert proxy._active_navigation_goal_id is None
    assert proxy.get_navigation_telemetry("G-INVALID")["status"] == "failed"


def test_legacy_navigation_can_still_opt_into_door_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    proxy._last_waypoint_time = 0.0
    fallback_calls: list[tuple[float, float, float]] = []
    clock = _AdvancingClock()

    monkeypatch.setattr(proxy_module.time, "time", clock)
    monkeypatch.setattr(proxy_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(proxy_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(proxy, "_publish_goal_point", lambda _x, _y: None)
    monkeypatch.setattr(
        proxy,
        "_navigate_via_doors",
        lambda x, y, timeout, **_kwargs: (
            fallback_calls.append((x, y, timeout)) or True
        ),
    )

    with patch("builtins.open", mock_open()):
        arrived = proxy.navigate_to(
            6.0,
            3.0,
            timeout_s=10.0,
            goal_id="G-LEGACY",
            allow_door_fallback=True,
        )

    assert arrived is True
    assert len(fallback_calls) == 1


def test_legacy_waypoint_path_receives_navigation_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = object()
    monkeypatch.setattr(proxy_module.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(proxy_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(proxy, "_publish_waypoint", lambda _x, _y: None)
    monkeypatch.setattr(proxy, "get_position", lambda: (1.0, 2.0, 0.28))

    assert proxy.go_to_waypoint(
        1.0,
        2.0,
        timeout_s=1.0,
        goal_id="G-WAYPOINT-GENERATION",
    )


def test_proxy_publishes_door_path_and_current_goal_as_distinct_types(
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = _Node()
    proxy._door_path_pub = _Publisher()
    proxy._current_goal_pub = _Publisher()

    assert proxy.publish_navigation_plan(
        {
            "room_path": ["living_room", "hallway"],
            "waypoints": [
                {"kind": "door_pre", "xy": [5.4, 3.0]},
                {"kind": "door_center", "xy": [6.0, 3.0]},
                {"kind": "door_post", "xy": [6.6, 3.0]},
            ],
        },
        goal_id="G-VIZ",
    )

    door_path = proxy._door_path_pub.messages[-1]
    current_goal = json.loads(proxy._current_goal_pub.messages[-1].data)
    assert isinstance(door_path, _Path)
    assert door_path.header.frame_id == "map"
    assert [(pose.pose.position.x, pose.pose.position.y) for pose in door_path.poses] == [
        (5.4, 3.0),
        (6.0, 3.0),
        (6.6, 3.0),
    ]
    assert current_goal["goal_id"] == "G-VIZ"
    assert current_goal["state"] == "GOAL_ACCEPTED"


def test_proxy_can_clear_the_published_door_path(
    fake_ros_messages: None,
) -> None:
    proxy = Go2ROS2Proxy()
    proxy._node = _Node()
    proxy._door_path_pub = _Publisher()
    proxy._current_goal_pub = _Publisher()

    assert proxy.publish_navigation_plan(
        {
            "room_path": ["living_room", "dining_room"],
            "waypoints": [
                {"kind": "door_center", "xy": [3.0, 5.0]},
                {"kind": "room_goal", "xy": [4.8, 6.0]},
            ],
        },
        goal_id="G-CLEAR",
    )
    assert proxy._door_path_pub.messages[-1].poses

    assert proxy._publish_empty_door_path()
    assert proxy._door_path_pub.messages[-1].poses == []
    assert proxy._last_navigation_plan is None


def test_far_route_association_matches_native_planner_marker_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root.parent
        / "third_party"
        / "far_planner"
        / "src"
        / "far_planner"
        / "src"
        / "planner_visualizer.cpp",
        root.parent
        / "vector_navigation_stack"
        / "src"
        / "far_planner"
        / "src"
        / "planner_visualizer.cpp",
    )
    planner_source_path = next((path for path in candidates if path.is_file()), None)
    if planner_source_path is None:
        pytest.skip("sibling FAR planner source is unavailable in this checkout")
    planner = planner_source_path.read_text(encoding="utf-8")
    compact_planner = " ".join(planner.split())
    assert FAR_ROUTE_MARKER_TOPIC == "/viz_path_topic"
    assert (
        'create_publisher<visualization_msgs::msg::Marker>("/viz_path_topic", 5)'
        in compact_planner
    )
    assert "path_marker.points.push_back" in planner

    proxy = (
        root / "vector_os_nano" / "hardware" / "sim" / "go2_ros2_proxy.py"
    ).read_text(encoding="utf-8")
    assert 'FAR_ROUTE_MARKER_TOPIC: str = "/viz_path_topic"' in proxy
    subscription = proxy[
        proxy.index("                Marker,\n                FAR_ROUTE_MARKER_TOPIC")
        : proxy.index(
            "            )",
            proxy.index("                Marker,\n                FAR_ROUTE_MARKER_TOPIC"),
        )
    ]
    assert "self._far_route_marker_cb" in subscription
    assert "reliable_qos" in subscription
    assert "path_state_qos" not in subscription
    assert "getattr(msg, \"points\"" in proxy


def test_far_vgraph_readiness_matches_native_planner_marker_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root.parent
        / "third_party"
        / "far_planner"
        / "src"
        / "far_planner"
        / "src"
        / "planner_visualizer.cpp",
        root.parent
        / "vector_navigation_stack"
        / "src"
        / "far_planner"
        / "src"
        / "planner_visualizer.cpp",
    )
    planner_source_path = next((path for path in candidates if path.is_file()), None)
    if planner_source_path is None:
        pytest.skip("sibling FAR planner source is unavailable in this checkout")
    planner = planner_source_path.read_text(encoding="utf-8")
    compact_planner = " ".join(planner.split())
    assert FAR_VGRAPH_MARKER_TOPIC == "/viz_graph_topic"
    assert (
        "create_publisher<visualization_msgs::msg::MarkerArray>"
        '("/viz_graph_topic", 5)'
        in compact_planner
    )
    assert '"global_vertex"' in planner
    assert "nav_node_marker.points.resize(graph_size)" in planner
    far_master_source_path = planner_source_path.with_name("far_planner.cpp")
    assert far_master_source_path.is_file()
    far_master = " ".join(
        far_master_source_path.read_text(encoding="utf-8").split()
    )
    assert "if (!is_graph_init_ && !nav_graph_.empty())" in far_master
    assert (
        "void FARMaster::WaypointCallBack("
        "const geometry_msgs::msg::PointStamped& route_goal) "
        "{ if (!is_graph_init_)"
        in far_master
    )

    proxy = (
        root / "vector_os_nano" / "hardware" / "sim" / "go2_ros2_proxy.py"
    ).read_text(encoding="utf-8")
    assert 'FAR_VGRAPH_MARKER_TOPIC: str = "/viz_graph_topic"' in proxy
    qos = proxy[
        proxy.index("            far_vgraph_qos = QoSProfile(")
        : proxy.index(
            "\n\n            from geometry_msgs.msg",
            proxy.index("            far_vgraph_qos = QoSProfile("),
        )
    ]
    assert "ReliabilityPolicy.RELIABLE" in qos
    assert "DurabilityPolicy.VOLATILE" in qos
    assert "HistoryPolicy.KEEP_LAST" in qos
    assert "depth=5" in qos
    subscription = proxy[
        proxy.index("            self._far_vgraph_subscription =")
        : proxy.index(
            "\n            )",
            proxy.index("            self._far_vgraph_subscription ="),
        )
    ]
    assert "MarkerArray" in subscription
    assert "FAR_VGRAPH_MARKER_TOPIC" in subscription
    assert "self._far_vgraph_marker_cb" in subscription
    assert "far_vgraph_qos" in subscription


def test_bridge_wires_topics_and_enforces_policy_at_final_motor_boundary() -> None:
    bridge = (
        Path(__file__).resolve().parents[2] / "scripts" / "go2_vnav_bridge.py"
    ).read_text(encoding="utf-8")

    for topic_symbol in (
        "FAR_GLOBAL_PATH_TOPIC",
        "LOCAL_PLANNER_PATH_TOPIC",
        "EXECUTED_PATH_TOPIC",
        "NAV_SEGMENT_CONTROL_TOPIC",
        "NAV_SEGMENT_ACK_TOPIC",
    ):
        assert topic_symbol in bridge
    assert "def _segment_control_cb" in bridge
    assert "def _publish_segment_ack" in bridge
    assert 'Marker, "/viz_path_topic", self._far_path_marker_cb' in bridge
    assert "def _publish_empty_navigation_paths" in bridge
    assert "def _cancel_far_goal_at_current_position" in bridge
    assert "DurabilityPolicy.TRANSIENT_LOCAL" in bridge
    cancel_far = bridge[
        bridge.index("    def _cancel_far_goal_at_current_position(") :
        bridge.index(
            "\n    def _path_visualization_active",
            bridge.index("    def _cancel_far_goal_at_current_position("),
        )
    ]
    assert "body_to_sensor_position(" in cancel_far
    assert "segment_id" in bridge
    segment_control = bridge[
        bridge.index("    def _segment_control_cb(") : bridge.index(
            "\n    def _goal_control_cb",
            bridge.index("    def _segment_control_cb("),
        )
    ]
    assert "current.segment_id != segment_id" in segment_control
    clear_start = segment_control.index('if event == "clear":')
    set_start = segment_control.index('if event != "set":')
    clear_branch = segment_control[clear_start:set_start]
    assert clear_branch.index("self._apply_velocity(") < clear_branch.index(
        "self._segment_constraints = None"
    )
    follow_path = bridge[
        bridge.index("    def _follow_path(") : bridge.index(
            "\n    def _safety_check", bridge.index("    def _follow_path(")
        )
    ]
    assert "if not os.path.exists(nav_active_file()):" in follow_path
    assert "nav_fail_closed_hold" in follow_path
    assert follow_path.index("nav_fail_closed_hold") < follow_path.index(
        "Wall escape mode"
    )
    path_cb = bridge[
        bridge.index("    def _path_cb(") : bridge.index(
            "\n    def _check_front_obstacle", bridge.index("    def _path_cb(")
        )
    ]
    assert path_cb.index("if not self._path_visualization_active():") < path_cb.index(
        "self._local_path_pub.publish(map_path)"
    )
    assert path_cb.index("if not self._path_visualization_active():") < path_cb.index(
        "self._current_path = new_path"
    )
    assert '{"map", "vehicle", "base_link", "sensor"}' in path_cb
    assert "Rejected /path with unsupported frame" in path_cb
    assert 'frame == "sensor"' in path_cb
    goal_control = bridge[
        bridge.index("    def _goal_control_cb(") : bridge.index(
            "\n    def _check_reset_flag", bridge.index("    def _goal_control_cb(")
        )
    ]
    assert "self._exploration_finished = False" in goal_control
    turn_branch = bridge[
        bridge.index("            if abs_err > 2.1:") : bridge.index(
            "\n            elif _in_narrow:",
            bridge.index("            if abs_err > 2.1:"),
        )
    ]
    assert "vx = 0.0" in turn_branch
    assert "vy = 0.0" in turn_branch
    assert "vx = -0.15" not in turn_branch
    stuck_detector = bridge[
        bridge.index("    def _stuck_detector(") : bridge.index(
            "\n    def save_terrain", bridge.index("    def _stuck_detector(")
        )
    ]
    assert 'stall_heading_progress_deg", 3.0' in stuck_detector
    assert "self._stuck_count == 2" in stuck_detector
    assert "self._stuck_count == 4" in stuck_detector
    assert "self._active_segment_constraints() is not None" in stuck_detector
    assert "self._stuck_heading = heading" in stuck_detector
    apply_velocity = bridge[
        bridge.index("    def _apply_velocity(") : bridge.index(
            "\n    def _finalize_active_goal", bridge.index("    def _apply_velocity(")
        )
    ]
    assert "_SEGMENT_CONSTRAINED_SOURCES" in apply_velocity
    assert "constraints.constrain_velocity" in apply_velocity
    assert bridge.count("self._go2.set_velocity(") == 2
    assert DOOR_PATH_TOPIC == "/scene_graph/door_path"
    assert CURRENT_GOAL_TOPIC == "/nav/current_goal"
    assert NAV_SEGMENT_ACK_TOPIC == "/vector_os/nav_segment_ack"


def test_far_convergence_is_tighter_than_every_door_waypoint() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    far = yaml.safe_load(
        (root / "config" / "far_go2_indoor.yaml").read_text(encoding="utf-8")
    )
    convergence = float(
        far["far_planner"]["ros__parameters"]["g_planner/converge_distance"]
    )

    # P2's strictest waypoint is door_center (0.22 m).  FAR must continue
    # driving until the application-level segment can honestly pass its check.
    assert convergence < 0.22


def test_navigation_launcher_rejects_a_second_live_stack() -> None:
    launcher = (
        Path(__file__).resolve().parents[2] / "scripts" / "launch_explore.sh"
    ).read_text(encoding="utf-8")

    assert "flock -n 9" in launcher
    assert "another Vector navigation stack is still running" in launcher
    assert "exit 73" in launcher
    assert '--params-file "$LOCAL_PLANNER_CONFIG"' in launcher
    assert "ros2 run local_planner localPlanner" in launcher
    assert "ros2 launch local_planner local_planner.launch" not in launcher
