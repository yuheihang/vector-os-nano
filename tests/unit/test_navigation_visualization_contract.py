# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P2-04 contract tests for source-labelled navigation visualisation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RVIZ_CONFIG = _REPO_ROOT / "config" / "vnav.rviz"

_PATH_CONTRACT = {
    "/scene_graph/door_path": ("DoorTopologyPath", "170; 85; 255"),
    "/far/global_path": ("FARGlobalPath", "0; 85; 255"),
    "/local_planner/path": ("LocalPlannerPath", "25; 255; 0"),
    "/nav/executed_path": ("ExecutedPath", "255; 221; 0"),
}


def _displays() -> list[dict[str, Any]]:
    config = _config()
    displays = config["Visualization Manager"]["Displays"]
    assert isinstance(displays, list)
    return displays


def _config() -> dict[str, Any]:
    config = yaml.safe_load(_RVIZ_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(config, dict)
    return config


def _display_by_topic() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for display in _displays():
        topic = display.get("Topic")
        if isinstance(topic, dict) and isinstance(topic.get("Value"), str):
            result[topic["Value"]] = display
    return result


def test_all_p2_navigation_path_sources_have_enabled_path_displays() -> None:
    displays = _display_by_topic()

    for topic, (name, _color) in _PATH_CONTRACT.items():
        display = displays[topic]
        assert display["Class"] == "rviz_default_plugins/Path"
        assert display["Name"] == name
        assert display["Enabled"] is True
        assert display["Value"] is True
        assert display["Topic"]["Durability Policy"] == "Transient Local"


def test_p2_navigation_path_colors_are_unambiguous() -> None:
    displays = _display_by_topic()

    colors = []
    for topic, (_name, color) in _PATH_CONTRACT.items():
        assert displays[topic]["Color"] == color
        colors.append(color)
    assert len(set(colors)) == len(colors)


def test_legacy_paths_are_disabled_in_favour_of_lifecycle_owned_displays() -> None:
    """Raw planner topics bypass goal cleanup and must stay hidden by default."""
    displays = _display_by_topic()

    assert displays["/path"]["Name"] == "Path"
    assert displays["/path"]["Enabled"] is False
    assert displays["/path"]["Value"] is False
    assert displays["/viz_path_topic"]["Name"] == "FARPath"
    assert displays["/viz_path_topic"]["Enabled"] is False
    assert displays["/viz_path_topic"]["Value"] is False
    assert displays["/far/global_path"]["Name"] == "FARGlobalPath"
    assert displays["/local_planner/path"]["Name"] == "LocalPlannerPath"
    for topic in ("/exploration_path", "/global_path_full"):
        assert displays[topic]["Enabled"] is False
        assert displays[topic]["Value"] is False


def test_historical_trajectory_is_disabled_during_navigation_acceptance() -> None:
    """Historical tracks must not look like a newly planned navigation path."""
    displays = _display_by_topic()

    assert displays["/trajectory"]["Name"] == "Trajectory"
    assert displays["/trajectory"]["Enabled"] is False
    scene_graph = displays["/scene_graph_markers"]
    assert scene_graph["Namespaces"]["trajectory"] is False


def test_current_goal_red_marker_namespace_is_enabled() -> None:
    """The JSON goal state is paired with the existing red nav-goal marker."""
    display = _display_by_topic()["/scene_graph_markers"]

    assert display["Class"] == "rviz_default_plugins/MarkerArray"
    assert display["Enabled"] is True
    assert display["Namespaces"]["nav_goal"] is True
    assert display["Namespaces"]["nav_goal_label"] is True


def test_scene_graph_marker_refresh_key_tracks_goal_changes() -> None:
    from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

    proxy = Go2ROS2Proxy()
    proxy._scene_graph = None
    proxy._position = (0.0, 0.0, 0.3)
    proxy._nav_goal = None
    idle_hash = proxy._scene_graph_hash()

    proxy._nav_goal = (6.0, 3.0)
    assert proxy._scene_graph_hash() != idle_hash

    proxy._nav_goal = None
    assert proxy._scene_graph_hash() == idle_hash


def test_rviz_acceptance_config_uses_only_portable_core_plugins() -> None:
    """Native CLI acceptance must not depend on unbuilt FAR demo plugins."""
    config = _config()
    manager = config["Visualization Manager"]
    plugin_classes = [
        panel["Class"] for panel in config["Panels"]
    ] + [
        tool["Class"] for tool in manager["Tools"]
    ]

    assert all(
        plugin_class.startswith(("rviz_common/", "rviz_default_plugins/"))
        for plugin_class in plugin_classes
    )
    source = _RVIZ_CONFIG.read_text(encoding="utf-8")
    assert "goalpoint_rviz_plugin" not in source
    assert "teleop_rviz_plugin" not in source
    assert manager["Global Options"]["Fixed Frame"] == "map"


def test_vehicle_axes_uses_a_frame_published_by_the_nav_stack() -> None:
    vehicle_axes = next(
        display for display in _displays()
        if display.get("Name") == "Vehicle"
    )
    assert vehicle_axes["Reference Frame"] == "vehicle"
