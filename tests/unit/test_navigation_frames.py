# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Body/sensor pose contract for the CMU navigation transport."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
from vector_os_nano.hardware.sim.piper_ros2_proxy import PiperROS2Proxy
from vector_os_nano.navigation.frames import (
    body_to_sensor_position,
    sensor_to_body_position,
)


def _odom(
    x: float,
    y: float,
    z: float,
    yaw: float,
    *,
    child_frame_id: str,
) -> SimpleNamespace:
    q = SimpleNamespace(
        x=0.0,
        y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0),
    )
    pose = SimpleNamespace(
        position=SimpleNamespace(x=x, y=y, z=z),
        orientation=q,
    )
    return SimpleNamespace(
        child_frame_id=child_frame_id,
        pose=SimpleNamespace(pose=pose),
    )


@pytest.mark.parametrize("heading", [0.0, math.pi / 2.0, math.pi, -math.pi / 2.0])
def test_body_sensor_conversion_round_trip(heading: float) -> None:
    body = (3.2, -1.4, 0.28)
    sensor = body_to_sensor_position(body, heading)
    assert sensor_to_body_position(sensor, heading) == pytest.approx(body)


def test_proxy_converts_sensor_odometry_to_body_pose() -> None:
    proxy = Go2ROS2Proxy()
    sensor = body_to_sensor_position((10.0, 3.0, 0.28), math.pi / 2.0)

    proxy._odom_cb(_odom(*sensor, math.pi / 2.0, child_frame_id="sensor"))

    assert proxy.get_sensor_position() == pytest.approx(sensor)
    assert proxy.get_position() == pytest.approx((10.0, 3.0, 0.28))
    assert proxy.get_heading() == pytest.approx(math.pi / 2.0)


def test_proxy_does_not_reapply_offset_to_base_link_odometry() -> None:
    proxy = Go2ROS2Proxy()

    proxy._odom_cb(_odom(3.0, 4.0, 0.28, 0.0, child_frame_id="base_link"))

    assert proxy.get_position() == pytest.approx((3.0, 4.0, 0.28))
    assert proxy.get_sensor_position() is None


def test_sensor_inside_door_tolerance_does_not_mean_body_arrived() -> None:
    proxy = Go2ROS2Proxy()
    # Facing west: the front sensor is at the door centre while the body is
    # still 0.30 m on the hallway side.
    proxy._odom_cb(_odom(6.0, 8.0, 0.48, math.pi, child_frame_id="sensor"))

    body = proxy.get_position()
    assert math.hypot(body[0] - 6.0, body[1] - 8.0) == pytest.approx(0.3)
    assert math.hypot(body[0] - 6.0, body[1] - 8.0) > 0.22
    # FAR receives a sensor goal 0.3 m beyond the body waypoint, while the
    # proxy continues to grade arrival at the nominal body goal.
    assert proxy._planner_goal_for_body_target(6.0, 8.0) == pytest.approx(
        (5.7, 8.0)
    )


def test_piper_ik_uses_canonical_body_pose_without_second_offset() -> None:
    piper = PiperROS2Proxy.__new__(PiperROS2Proxy)
    piper._base = SimpleNamespace(
        get_position=lambda: (10.0, 3.0, 0.28),
        get_heading=lambda: math.pi / 2.0,
    )
    piper._ik_data = SimpleNamespace(
        qpos=np.zeros(20),
        qvel=np.ones(20),
        qacc=np.ones(20),
    )
    piper._ik_arm_qpos_adr = [7, 8, 9, 10, 11, 12]

    piper._sync_ik_base([0.1] * 6)

    assert piper._ik_data.qpos[:3] == pytest.approx((10.0, 3.0, 0.28))
    assert piper._ik_data.qpos[3:7] == pytest.approx(
        (
            math.cos(math.pi / 4.0),
            0.0,
            0.0,
            math.sin(math.pi / 4.0),
        )
    )
    assert piper._ik_data.qpos[7:13] == pytest.approx([0.1] * 6)
