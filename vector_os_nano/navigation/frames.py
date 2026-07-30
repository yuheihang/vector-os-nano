# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Navigation pose-frame contract shared by the bridge and ROS proxy.

The CMU navigation stack publishes ``/state_estimation`` as the lidar sensor
pose, while application-level goals and :class:`BaseProtocol` positions refer
to the robot body centre.  Keeping the conversion here prevents either side
from silently applying a different mounting offset.
"""
from __future__ import annotations

import math


NAV_SENSOR_OFFSET_X_M: float = 0.3
NAV_SENSOR_OFFSET_Y_M: float = 0.0
NAV_SENSOR_OFFSET_Z_M: float = 0.2


def body_to_sensor_position(
    position: tuple[float, float, float],
    heading: float,
) -> tuple[float, float, float]:
    """Convert a body-centre map pose to the mounted sensor map position."""

    x, y, z = (float(value) for value in position)
    cos_h = math.cos(float(heading))
    sin_h = math.sin(float(heading))
    return (
        x + cos_h * NAV_SENSOR_OFFSET_X_M - sin_h * NAV_SENSOR_OFFSET_Y_M,
        y + sin_h * NAV_SENSOR_OFFSET_X_M + cos_h * NAV_SENSOR_OFFSET_Y_M,
        z + NAV_SENSOR_OFFSET_Z_M,
    )


def sensor_to_body_position(
    position: tuple[float, float, float],
    heading: float,
) -> tuple[float, float, float]:
    """Convert a mounted sensor map pose to the robot body-centre map pose."""

    x, y, z = (float(value) for value in position)
    cos_h = math.cos(float(heading))
    sin_h = math.sin(float(heading))
    return (
        x - cos_h * NAV_SENSOR_OFFSET_X_M + sin_h * NAV_SENSOR_OFFSET_Y_M,
        y - sin_h * NAV_SENSOR_OFFSET_X_M - cos_h * NAV_SENSOR_OFFSET_Y_M,
        z - NAV_SENSOR_OFFSET_Z_M,
    )


__all__ = [
    "NAV_SENSOR_OFFSET_X_M",
    "NAV_SENSOR_OFFSET_Y_M",
    "NAV_SENSOR_OFFSET_Z_M",
    "body_to_sensor_position",
    "sensor_to_body_position",
]
