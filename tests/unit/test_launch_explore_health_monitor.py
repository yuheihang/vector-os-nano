# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Static contracts for the managed navigation launcher's health boundary."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


_LAUNCHER = (
    Path(__file__).resolve().parents[2] / "scripts" / "launch_explore.sh"
)


def test_ready_session_monitors_every_navigation_critical_process() -> None:
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert "monitor_critical_processes()" in source
    assert source.rstrip().endswith("monitor_critical_processes")
    for variable in (
        "BRIDGE_PID",
        "LOCAL_PLANNER_PID",
        "VEHICLE_TF_PID",
        "CAMERA_TF_PID",
        "SENSOR_SCAN_PID",
        "TERRAIN_PID",
        "TERRAIN_EXT_PID",
        "TARE_LAUNCH_PID",
    ):
        assert f'require_alive "${variable}"' in source

    # ros2 launch can stay alive through graph_decoder after far_planner dies,
    # so the child planner identity must be monitored separately.
    assert 'find_descendant_process "$launch_pid" "$expected_comm"' in source
    assert '"$FAR_PLANNER_PID" "$FAR_PLANNER_START_TICKS" "far_planner"' in source
    assert source.count("require_process_identity_alive") >= 3
    assert 'require_alive "$FAR_LAUNCH_PID"' in source


def test_launcher_does_not_hide_planner_death_behind_bridge_wait() -> None:
    source = _LAUNCHER.read_text(encoding="utf-8")

    assert 'wait "$BRIDGE_PID"' not in source
    assert "exited during the navigation session" in source


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_far_planner_child_death_stops_launcher_while_ros2_launch_lives(
    tmp_path: Path,
) -> None:
    """Exercise the real launcher against cheap fake ROS processes.

    The fake FAR launch deliberately keeps graph_decoder alive after its
    far_planner child is killed.  Monitoring only FAR_LAUNCH_PID would hang.
    """

    if not Path("/opt/ros/humble/setup.bash").is_file():
        pytest.skip("launcher integration contract requires ROS 2 Humble setup")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_nav = tmp_path / "nav"
    (fake_nav / "install").mkdir(parents=True)
    (fake_nav / "install/far_planner/share/far_planner/config").mkdir(
        parents=True
    )
    (fake_nav / "install/tare_planner/share/tare_planner").mkdir(parents=True)
    local_prefix = tmp_path / "local_planner"
    (local_prefix / "share/local_planner/paths").mkdir(parents=True)

    (fake_nav / "install/setup.bash").write_text(
        'export PATH="${FAKE_ROS_BIN}:$PATH"\n',
        encoding="utf-8",
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/bin/bash
if [ "${1:-}" = "60" ]; then
    exec /bin/sleep "$@"
fi
exec /bin/sleep 0.01
""",
    )
    _write_executable(
        fake_bin / "python3",
        """#!/bin/bash
exec /bin/sleep 60
""",
    )
    _write_executable(
        fake_bin / "ros2",
        """#!/bin/bash
if [ "${1:-}" = "pkg" ] && [ "${2:-}" = "prefix" ]; then
    printf '%s\n' "$FAKE_LOCAL_PREFIX"
    exit 0
fi

if [ "${1:-}" = "launch" ] && [ "${2:-}" = "far_planner" ]; then
    printf '%s\n' "$$" > "$FAKE_FAR_LAUNCH_PID_FILE"
    "$FAKE_HOST_PYTHON" -c \
        'import ctypes, time; ctypes.CDLL(None).prctl(15, b"far_planner", 0, 0, 0); time.sleep(60)' &
    far_pid=$!
    printf '%s\n' "$far_pid" > "$FAKE_FAR_PID_FILE"

    /bin/sleep 60 &
    decoder_pid=$!
    printf '%s\n' "$decoder_pid" > "$FAKE_DECODER_PID_FILE"

    cleanup_fake_launch() {
        trap - EXIT INT TERM
        kill "$far_pid" "$decoder_pid" 2>/dev/null || true
        wait "$far_pid" "$decoder_pid" 2>/dev/null || true
        exit 0
    }
    trap cleanup_fake_launch EXIT INT TERM
    wait
    exit 0
fi

exec /bin/sleep 60
""",
    )

    far_pid_file = tmp_path / "far.pid"
    far_launch_pid_file = tmp_path / "far_launch.pid"
    decoder_pid_file = tmp_path / "decoder.pid"
    log_path = tmp_path / "launcher.log"
    env = os.environ.copy()
    env.update(
        {
            "FAKE_ROS_BIN": str(fake_bin),
            "FAKE_HOST_PYTHON": sys.executable,
            "FAKE_LOCAL_PREFIX": str(local_prefix),
            "FAKE_FAR_PID_FILE": str(far_pid_file),
            "FAKE_FAR_LAUNCH_PID_FILE": str(far_launch_pid_file),
            "FAKE_DECODER_PID_FILE": str(decoder_pid_file),
            "GO2ARM_ROOT": str(tmp_path),
            "NAV_STACK": str(fake_nav),
            "TARE_ROOT": str(tmp_path / "tare"),
            "VECTOR_VNAV_LOCK_FILE": str(tmp_path / "vnav.lock"),
            "VECTOR_NAV_ACTIVE_FILE": str(tmp_path / "nav_active"),
            "VECTOR_NAV_STALLED_FILE": str(tmp_path / "nav_stalled"),
            "VECTOR_NAV_RESET_FILE": str(tmp_path / "reset_pose"),
            "VECTOR_NAV_REPLAY_FILE": str(tmp_path / "terrain_replay"),
        }
    )
    env.pop("VECTOR_VNAV_PARENT_PID", None)
    env.pop("VECTOR_VNAV_MANAGED_SESSION", None)

    log_handle = log_path.open("w", encoding="utf-8")
    launcher = subprocess.Popen(
        ["bash", str(_LAUNCHER), "--no-gui"],
        cwd=_LAUNCHER.parents[1],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    try:
        deadline = time.monotonic() + 8.0
        ready = False
        while time.monotonic() < deadline:
            if (
                far_pid_file.is_file()
                and far_launch_pid_file.is_file()
                and "Ready! Dog is standing still." in log_path.read_text(
                    encoding="utf-8"
                )
            ):
                ready = True
                break
            if launcher.poll() is not None:
                break
            time.sleep(0.02)

        assert launcher.poll() is None, log_path.read_text(encoding="utf-8")
        assert ready, log_path.read_text(encoding="utf-8")
        assert far_pid_file.is_file()
        assert far_launch_pid_file.is_file()
        far_pid = int(far_pid_file.read_text(encoding="utf-8"))
        far_launch_pid = int(far_launch_pid_file.read_text(encoding="utf-8"))
        assert _pid_exists(far_pid)
        assert _pid_exists(far_launch_pid)

        os.kill(far_pid, signal.SIGKILL)
        return_code = launcher.wait(timeout=5.0)
        log_handle.close()
        output = log_path.read_text(encoding="utf-8")

        assert return_code != 0, output
        assert "ERROR: far_planner exited during the navigation session." in output
    finally:
        if launcher.poll() is None:
            os.killpg(launcher.pid, signal.SIGTERM)
            try:
                launcher.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                os.killpg(launcher.pid, signal.SIGKILL)
                launcher.wait(timeout=3.0)
        if not log_handle.closed:
            log_handle.close()
