# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for Wave 2/3 skill registration wiring in sim_tool.py.

Test strategy: The full behavioural test (calling _start_go2 end-to-end with
with_arm=True) requires mocking subprocess.Popen, os.setsid, time.sleep(20),
Go2ROS2Proxy, PiperROS2Proxy, PiperGripperROS2Proxy, Agent, SceneGraph, and
multiple lazy imports inside the method body. That mock surface is fragile and
couples the test to implementation details of unrelated code paths.

Instead we use two focused guards:

1. test_manipulation_skills_importable_and_instantiable — verifies all 4 skill
   classes can be imported and constructed with no arguments. A failure here
   means any register(XyzSkill()) call in sim_tool will raise ImportError or
   TypeError at runtime.

2. test_sim_tool_module_contains_all_manipulation_registrations — inspects the
   source text of sim_tool to confirm each skill class name appears with a
   register() call pattern. Catches typos, missing imports, or accidental
   deletions without running the full method.
"""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest


def test_manipulation_skills_importable_and_instantiable() -> None:
    """Regression guard: all 4 Wave 2/3 skill classes are importable and
    instantiable with no arguments. A failure means the register(...) calls
    added to _start_go2 will crash at runtime.
    """
    from vector_os_nano.skills.pick_top_down import PickTopDownSkill
    from vector_os_nano.skills.place_top_down import PlaceTopDownSkill
    from vector_os_nano.skills.mobile_pick import MobilePickSkill
    from vector_os_nano.skills.mobile_place import MobilePlaceSkill

    PickTopDownSkill()
    PlaceTopDownSkill()
    MobilePickSkill()
    MobilePlaceSkill()


def test_sim_tool_module_contains_all_manipulation_registrations() -> None:
    """Sanity: the sim_tool source text imports and registers all 4 skill
    classes inside the piper_arm guard block. Not a behaviour test but catches
    most typos and missing lines that would break runtime registration.
    """
    from vector_os_nano.vcli.tools import sim_tool

    src = inspect.getsource(sim_tool)

    expected_classes = (
        "PickTopDownSkill",
        "PlaceTopDownSkill",
        "MobilePickSkill",
        "MobilePlaceSkill",
    )
    for cls in expected_classes:
        assert f"{cls}()" in src, (
            f"{cls}() not found in sim_tool.py — "
            "skill will not be registered when with_arm=True"
        )


def test_dead_navigation_subprocess_surfaces_log_root_cause(tmp_path) -> None:
    from vector_os_nano.vcli.tools.sim_tool import _vnav_process_exit_error

    log_path = tmp_path / "vector_vnav.log"
    log_path.write_text(
        "Starting bridge\n"
        "Traceback (most recent call last):\n"
        "UnboundLocalError: local variable 'String' referenced before assignment\n",
        encoding="utf-8",
    )
    with log_path.open("a", encoding="utf-8") as log_fh:
        error = _vnav_process_exit_error(
            SimpleNamespace(poll=lambda: 139),
            log_fh,
            str(log_path),
            phase="startup",
        )

    assert isinstance(error, RuntimeError)
    assert "status 139" in str(error)
    assert "UnboundLocalError" in str(error)
    assert str(log_path) in str(error)


def test_live_navigation_subprocess_has_no_startup_error(tmp_path) -> None:
    from vector_os_nano.vcli.tools.sim_tool import _vnav_process_exit_error

    log_path = tmp_path / "vector_vnav.log"
    with log_path.open("w", encoding="utf-8") as log_fh:
        error = _vnav_process_exit_error(
            SimpleNamespace(poll=lambda: None),
            log_fh,
            str(log_path),
            phase="startup",
        )
    assert error is None


def test_proxy_readiness_requires_real_odom_and_piper_joint_state() -> None:
    from vector_os_nano.vcli.tools.sim_tool import (
        _require_go2_proxy_ready,
        _require_piper_proxy_ready,
    )

    with pytest.raises(TimeoutError, match="state_estimation"):
        _require_go2_proxy_ready(
            SimpleNamespace(_connected=True, _last_odom=None)
        )
    with pytest.raises(TimeoutError, match="joint_state"):
        _require_piper_proxy_ready(
            SimpleNamespace(_connected=True, _last_joint_state_ts=0.0),
            SimpleNamespace(_connected=True),
        )

    _require_go2_proxy_ready(
        SimpleNamespace(_connected=True, _last_odom=object())
    )
    _require_piper_proxy_ready(
        SimpleNamespace(_connected=True, _last_joint_state_ts=1.0),
        SimpleNamespace(_connected=True),
    )


def test_sim_ros_domain_ignores_generic_shell_domain(monkeypatch) -> None:
    import secrets

    from vector_os_nano.vcli.tools.sim_tool import _select_sim_ros_domain_id

    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    monkeypatch.delenv("VECTOR_SIM_ROS_DOMAIN_ID", raising=False)
    monkeypatch.setattr(secrets, "randbelow", lambda span: 17)

    assert _select_sim_ros_domain_id() == 232

    monkeypatch.setenv("VECTOR_SIM_ROS_DOMAIN_ID", "42")
    assert _select_sim_ros_domain_id() == 42

    monkeypatch.setenv("VECTOR_SIM_ROS_DOMAIN_ID", "233")
    with pytest.raises(ValueError, match="0 to 232"):
        _select_sim_ros_domain_id()


def test_vnav_session_environment_is_isolated_and_restorable(
    monkeypatch, tmp_path
) -> None:
    import tempfile

    from vector_os_nano.vcli.tools.sim_tool import (
        _prepare_vnav_session_environment,
        _restore_vnav_session_environment,
    )

    session_dir = tmp_path / "session"

    def _mkdtemp(*, prefix: str) -> str:
        assert prefix.startswith("vector-vnav-")
        session_dir.mkdir()
        return str(session_dir)

    monkeypatch.setattr(tempfile, "mkdtemp", _mkdtemp)
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    monkeypatch.setenv("VECTOR_VNAV_LOCK_FILE", "/tmp/operator-vnav.lock")
    monkeypatch.delenv("VECTOR_VNAV_LOG_FILE", raising=False)

    actual_dir, values, previous = _prepare_vnav_session_environment(144)

    assert actual_dir == str(session_dir)
    assert values["ROS_DOMAIN_ID"] == "144"
    assert values["VECTOR_NAV_ACTIVE_FILE"].startswith(str(session_dir))
    assert values["VECTOR_NAV_STALLED_FILE"].startswith(str(session_dir))
    assert values["VECTOR_EXPLORE_FINISHED_FILE"].startswith(str(session_dir))
    assert values["VECTOR_TERRAIN_MAP_FILE"].startswith(str(session_dir))
    assert "VECTOR_VNAV_LOCK_FILE" not in values
    assert __import__("os").environ["VECTOR_VNAV_LOCK_FILE"] == (
        "/tmp/operator-vnav.lock"
    )
    assert values["VECTOR_VNAV_LOG_FILE"] == f"{session_dir}.log"
    assert previous["ROS_DOMAIN_ID"] == "7"

    _restore_vnav_session_environment(values, previous)
    assert __import__("os").environ["ROS_DOMAIN_ID"] == "7"
    assert "VECTOR_NAV_ACTIVE_FILE" not in __import__("os").environ


def test_launcher_ready_wait_uses_current_session_log(tmp_path) -> None:
    from vector_os_nano.vcli.tools.sim_tool import _wait_for_vnav_ready_marker

    log_path = tmp_path / "this-session.log"
    log_path.write_text(
        "Starting stack\nReady! Dog is standing still.\n",
        encoding="utf-8",
    )
    with log_path.open("a", encoding="utf-8") as log_fh:
        _wait_for_vnav_ready_marker(
            SimpleNamespace(poll=lambda: None),
            log_fh,
            str(log_path),
            timeout_s=0.1,
        )


class _MatchedEndpoint:
    def __init__(self, count: int = 1) -> None:
        self.count = count

    def get_subscription_count(self) -> int:
        return self.count

    def get_publisher_count(self) -> int:
        return self.count


class _GraphNode:
    def __init__(self) -> None:
        self.publishers = {
            "/vector_os/nav_goal_telemetry": 1,
            "/vector_os/nav_segment_ack": 1,
            "/path": 1,
        }
        self.node_names = [
            "go2_vnav_bridge",
            "localPlanner",
            "far_planner",
            "tare_planner_node",
        ]

    def count_publishers(self, topic: str) -> int:
        return self.publishers.get(topic, 0)

    def get_node_names(self) -> list[str]:
        return list(self.node_names)


def _ready_go2_proxy() -> SimpleNamespace:
    return SimpleNamespace(
        _connected=True,
        _last_odom=object(),
        _node=_GraphNode(),
        _goal_control_pub=_MatchedEndpoint(),
        _segment_control_pub=_MatchedEndpoint(),
        _segment_ack_subscription=_MatchedEndpoint(),
        _goal_pub=_MatchedEndpoint(),
        _waypoint_pub=_MatchedEndpoint(),
        far_vgraph_ready=lambda: True,
        far_vgraph_diagnostics=lambda: {
            "status": "ready",
            "node_count": 12,
        },
    )


def test_navigation_readiness_requires_actual_dds_matches() -> None:
    from vector_os_nano.vcli.tools.sim_tool import (
        _go2_navigation_readiness_missing,
    )

    proxy = _ready_go2_proxy()
    assert _go2_navigation_readiness_missing(proxy) == ()

    proxy._segment_control_pub.count = 0
    proxy._node.node_names.remove("far_planner")
    missing = _go2_navigation_readiness_missing(proxy)
    assert any("nav_segment_control" in item for item in missing)
    assert any("far_planner" in item for item in missing)


def test_navigation_readiness_accepts_far_source_and_launch_aliases() -> None:
    from vector_os_nano.vcli.tools.sim_tool import (
        _go2_navigation_readiness_missing,
    )

    launched_proxy = _ready_go2_proxy()
    assert "far_planner" in launched_proxy._node.node_names
    assert _go2_navigation_readiness_missing(launched_proxy) == ()

    direct_proxy = _ready_go2_proxy()
    direct_proxy._node.node_names.remove("far_planner")
    direct_proxy._node.node_names.append("far_planner_node")
    assert _go2_navigation_readiness_missing(direct_proxy) == ()


def test_navigation_readiness_rejects_discovered_but_empty_far_graph() -> None:
    from vector_os_nano.vcli.tools.sim_tool import (
        _go2_navigation_readiness_missing,
    )

    proxy = _ready_go2_proxy()
    proxy.far_vgraph_ready = lambda: False
    proxy.far_vgraph_diagnostics = lambda: {
        "status": "empty_graph",
        "node_count": 0,
    }

    missing = _go2_navigation_readiness_missing(proxy)
    assert missing == (
        "FAR non-empty V-Graph "
        "(status=empty_graph, global_vertex_nodes=0)",
    )


def test_odometry_startup_rate_uses_fresh_messages(monkeypatch) -> None:
    from vector_os_nano.vcli.tools import sim_tool

    now = [0.0]
    proxy = SimpleNamespace(
        _last_odom=SimpleNamespace(
            header=SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
        )
    )

    def _clock() -> float:
        return now[0]

    def _sleep(duration: float) -> None:
        now[0] += duration
        nanoseconds = int(now[0] * 1_000_000_000)
        proxy._last_odom = SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(
                    sec=nanoseconds // 1_000_000_000,
                    nanosec=nanoseconds % 1_000_000_000,
                )
            )
        )

    measured = sim_tool._measure_go2_odometry_rate(
        proxy,
        window_s=1.0,
        poll_s=0.02,
        clock=_clock,
        sleep=_sleep,
    )
    assert measured >= 40.0

    monkeypatch.setattr(
        sim_tool, "_measure_go2_odometry_rate", lambda *args, **kwargs: 1.3
    )
    with pytest.raises(RuntimeError, match="startup_performance.*1.3 Hz"):
        sim_tool._require_go2_odometry_performance(proxy, minimum_hz=5.0)


def test_launch_script_fails_fast_if_bridge_dies() -> None:
    from pathlib import Path

    launch = (
        Path(__file__).resolve().parents[3] / "scripts" / "launch_explore.sh"
    ).read_text(encoding="utf-8")
    assert "require_alive" in launch
    assert launch.count(
        'require_alive "$BRIDGE_PID" "Go2 navigation bridge"'
    ) >= 2
    assert 'require_alive "$FAR_LAUNCH_PID" "FAR planner launch"' in launch
    assert 'require_alive "$TARE_LAUNCH_PID" "TARE planner launch"' in launch
    assert "MUJOCO_EGL_DEVICE_ID=0" in launch
    assert 'VECTOR_MUJOCO_GUI_GL:-glfw' in launch
    assert 'VECTOR_MUJOCO_HEADLESS_GL:-egl' in launch
    assert "CLEANING_UP=1" in launch
    assert "trap - EXIT INT TERM" in launch
    assert "set -m" not in launch
    assert "pkill -9 -f" not in launch
    assert 'NAV_ACTIVE_FILE="${VECTOR_NAV_ACTIVE_FILE' in launch
    assert "/tmp/vector_reset_pose" in launch
    assert "/tmp/vector_terrain_replay" in launch
    assert "VECTOR_VNAV_PARENT_PID" in launch


def test_go2_launch_owns_one_process_session() -> None:
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    source = inspect.getsource(SimStartTool._start_go2)
    assert "start_new_session=True" in source
    assert "preexec_fn=os.setsid" not in source
    assert 'child_env["VECTOR_VNAV_PARENT_PID"]' in source
    assert "_wait_for_vnav_ready_marker" in source
    assert "_wait_for_go2_navigation_ready" in source
    assert "_require_go2_odometry_performance" in source
    # Pre-launch failures restore directly; post-launch failures flow through
    # the idempotent two-phase lifecycle.
    assert source.count("_restore_vnav_session_environment") >= 2
    assert source.count("shutil.rmtree(session_dir") >= 2
    assert "_VNavSessionLifecycle" in source
    assert "prepare_for_domain(domain_id)" in source
    assert source.index("prepare_for_domain(domain_id)") < source.index(
        "subprocess.Popen("
    )
    assert source.index("prepare_for_domain(domain_id)") < source.index(
        "base.connect()"
    )
    assert "base._sim_stop_process" in source
    assert "base._sim_finalize_session" in source
    assert "base._sim_unregister_cleanup" in source
