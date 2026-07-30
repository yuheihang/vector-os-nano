# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""ROS2 integration layer for Vector OS Nano. Optional.

Usage:
    from vector_os_nano.ros2 import ROS2_AVAILABLE

    if ROS2_AVAILABLE:
        from vector_os_nano.ros2 import HardwareBridgeNode, PerceptionBridgeNode

This module intentionally does NOT import any ROS2 modules at the top level.
All ROS2 imports are guarded so the SDK works without ROS2 installed.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

ROS2_AVAILABLE: bool = False
try:
    import rclpy  # noqa: F401
    ROS2_AVAILABLE = True
except ImportError:
    pass

_NODE_EXPORTS: dict[str, tuple[str, str]] = {
    "HardwareBridgeNode": (
        "vector_os_nano.ros2.nodes.hardware_bridge",
        "HardwareBridgeNode",
    ),
    "PerceptionBridgeNode": (
        "vector_os_nano.ros2.nodes.perception_node",
        "PerceptionBridgeNode",
    ),
    "SkillServerNode": (
        "vector_os_nano.ros2.nodes.skill_server",
        "SkillServerNode",
    ),
    "WorldModelServiceNode": (
        "vector_os_nano.ros2.nodes.world_model_node",
        "WorldModelServiceNode",
    ),
    "AgentNode": (
        "vector_os_nano.ros2.nodes.agent_node",
        "AgentNode",
    ),
}

__all__ = (
    ["ROS2_AVAILABLE", *_NODE_EXPORTS]
    if ROS2_AVAILABLE
    else ["ROS2_AVAILABLE"]
)


def __getattr__(name: str) -> Any:
    """Load a requested ROS node without importing unrelated message stacks."""

    if not ROS2_AVAILABLE or name not in _NODE_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _NODE_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
