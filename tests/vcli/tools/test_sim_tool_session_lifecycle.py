# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Focused regression tests for managed SimStart session teardown."""
from __future__ import annotations

import inspect
import os
import signal
from types import SimpleNamespace

import numpy as np
import pytest

from vector_os_nano.vcli.tools import sim_tool


def _write_terrain(
    path,
    *,
    ix=(0,),
    iy=(0,),
    z=(0.0,),
    voxel_size=0.1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ix=np.asarray(ix),
        iy=np.asarray(iy),
        z=np.asarray(z),
        voxel_size=np.asarray(voxel_size),
    )


def _read_grid(path) -> tuple[dict[tuple[int, int], float], float]:
    loaded = sim_tool._load_valid_terrain_map(str(path))
    assert loaded is not None
    return loaded


def test_load_valid_terrain_coalesces_duplicate_voxels_with_max_z(
    tmp_path,
) -> None:
    terrain = tmp_path / "terrain.npz"
    _write_terrain(
        terrain,
        ix=(1, 1, -2),
        iy=(3, 3, 4),
        z=(0.4, 0.9, -0.1),
    )

    grid, voxel_size = _read_grid(terrain)

    assert grid == {(1, 3): pytest.approx(0.9), (-2, 4): pytest.approx(-0.1)}
    assert voxel_size == pytest.approx(0.1)


@pytest.mark.parametrize(
    "payload",
    [
        {"ix": (0,), "iy": (0,), "z": (0.0,)},  # missing voxel_size
        {"ix": (0, 1), "iy": (0,), "z": (0.0,), "voxel_size": 0.1},
        {"ix": (), "iy": (), "z": (), "voxel_size": 0.1},
        {"ix": (0.5,), "iy": (0,), "z": (0.0,), "voxel_size": 0.1},
        {"ix": (np.nan,), "iy": (0,), "z": (0.0,), "voxel_size": 0.1},
        {"ix": (0,), "iy": (0,), "z": (np.inf,), "voxel_size": 0.1},
        {"ix": (0,), "iy": (0,), "z": (0.0,), "voxel_size": 0.0},
        {"ix": (0,), "iy": (0,), "z": (0.0,), "voxel_size": np.nan},
        {"ix": (0,), "iy": (0,), "z": (0.0,), "voxel_size": (0.1, 0.2)},
    ],
)
def test_load_valid_terrain_rejects_malformed_npz(tmp_path, payload) -> None:
    terrain = tmp_path / "invalid.npz"
    np.savez_compressed(terrain, **payload)

    assert sim_tool._load_valid_terrain_map(str(terrain)) is None


def test_load_valid_terrain_rejects_non_npz(tmp_path) -> None:
    terrain = tmp_path / "invalid.npz"
    terrain.write_bytes(b"not an npz")

    assert sim_tool._load_valid_terrain_map(str(terrain)) is None


def test_merge_is_union_max_z_and_atomic(monkeypatch, tmp_path) -> None:
    canonical = tmp_path / "canonical" / "terrain.npz"
    session = tmp_path / "session" / "terrain.npz"
    _write_terrain(
        canonical,
        ix=(0, 1),
        iy=(0, 1),
        z=(0.2, 0.8),
    )
    _write_terrain(
        session,
        ix=(0, 1, 2),
        iy=(0, 1, 2),
        z=(0.7, 0.3, 1.1),
    )

    real_replace = os.replace
    replacements: list[tuple[str, str]] = []

    def _replace(source: str, target: str) -> None:
        replacements.append((source, target))
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", _replace)

    assert sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )

    grid, voxel_size = _read_grid(canonical)
    assert grid == {
        (0, 0): pytest.approx(0.7),
        (1, 1): pytest.approx(0.8),
        (2, 2): pytest.approx(1.1),
    }
    assert voxel_size == pytest.approx(0.1)
    assert len(replacements) == 1
    temporary, target = replacements[0]
    assert os.path.dirname(temporary) == str(canonical.parent)
    assert target == str(canonical)
    assert temporary != target


def test_smaller_session_cannot_shrink_canonical(tmp_path) -> None:
    canonical = tmp_path / "canonical.npz"
    session = tmp_path / "session.npz"
    _write_terrain(
        canonical,
        ix=(1, 2, 3),
        iy=(1, 2, 3),
        z=(0.1, 0.2, 0.3),
    )
    _write_terrain(session, ix=(2,), iy=(2,), z=(0.15,))

    assert sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )
    assert _read_grid(canonical)[0] == {
        (1, 1): pytest.approx(0.1),
        (2, 2): pytest.approx(0.2),
        (3, 3): pytest.approx(0.3),
    }


@pytest.mark.parametrize("failure", ["corrupt", "voxel_mismatch"])
def test_invalid_or_incompatible_session_never_changes_canonical(
    tmp_path, failure
) -> None:
    canonical = tmp_path / "canonical.npz"
    session = tmp_path / "session.npz"
    _write_terrain(canonical, ix=(1, 2), iy=(3, 4), z=(0.5, 0.6))
    original = canonical.read_bytes()

    if failure == "corrupt":
        session.write_bytes(b"broken")
    else:
        _write_terrain(
            session,
            ix=(8,),
            iy=(9,),
            z=(1.0,),
            voxel_size=0.2,
        )

    assert not sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )
    assert canonical.read_bytes() == original


def test_invalid_canonical_is_not_blindly_overwritten(tmp_path) -> None:
    canonical = tmp_path / "canonical.npz"
    session = tmp_path / "session.npz"
    canonical.write_bytes(b"unknown legacy or damaged payload")
    original = canonical.read_bytes()
    _write_terrain(session, ix=(1,), iy=(2,), z=(0.5,))

    assert not sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )
    assert canonical.read_bytes() == original


def test_atomic_replace_failure_preserves_canonical(monkeypatch, tmp_path) -> None:
    canonical = tmp_path / "canonical.npz"
    session = tmp_path / "session.npz"
    _write_terrain(canonical, ix=(1,), iy=(1,), z=(0.5,))
    _write_terrain(session, ix=(2,), iy=(2,), z=(0.8,))
    original = canonical.read_bytes()

    def _fail_replace(source: str, target: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _fail_replace)

    assert not sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )
    assert canonical.read_bytes() == original
    assert list(tmp_path.glob(".terrain-map-*.npz")) == []


def test_clear_memory_deletion_is_not_resurrected(tmp_path) -> None:
    canonical = tmp_path / "canonical.npz"
    session = tmp_path / "session.npz"
    _write_terrain(canonical, ix=(1,), iy=(1,), z=(0.5,))
    _write_terrain(session, ix=(1, 2), iy=(1, 2), z=(0.7, 0.8))
    canonical.unlink()  # /clear_memory explicitly removed the durable seed

    assert not sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=True,
    )
    assert not canonical.exists()


def test_first_session_can_create_missing_canonical_seed(tmp_path) -> None:
    canonical = tmp_path / "canonical" / "terrain.npz"
    session = tmp_path / "session.npz"
    _write_terrain(session, ix=(2,), iy=(3,), z=(0.8,))

    assert sim_tool._merge_session_terrain_map(
        str(session),
        str(canonical),
        canonical_existed_at_start=False,
    )
    assert _read_grid(canonical)[0] == {(2, 3): pytest.approx(0.8)}


class _FakeProcess:
    pid = 424242

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.wait_calls = 0

    def wait(self, *, timeout: float) -> int:
        self.events.append("wait")
        self.wait_calls += 1
        return 0


class _FakeLog:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.closed = False

    def close(self) -> None:
        self.events.append("log_close")
        self.closed = True


def _make_lifecycle(monkeypatch, tmp_path):
    events: list[str] = []
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session_terrain = session_dir / "terrain.npz"
    canonical = tmp_path / "canonical.npz"
    _write_terrain(canonical, ix=(1,), iy=(1,), z=(0.2,))
    _write_terrain(session_terrain, ix=(2,), iy=(2,), z=(0.6,))

    session_values = {
        "ROS_DOMAIN_ID": "221",
        "VECTOR_NAV_ACTIVE_FILE": str(session_dir / "nav_active"),
        "VECTOR_NAV_STALLED_FILE": str(session_dir / "nav_stalled"),
        "VECTOR_NAV_RESET_FILE": str(session_dir / "nav_reset"),
        "VECTOR_NAV_REPLAY_FILE": str(session_dir / "nav_replay"),
        "VECTOR_EXPLORE_FINISHED_FILE": str(session_dir / "explore_finished"),
        "VECTOR_TERRAIN_MAP_FILE": str(session_terrain),
        "VECTOR_VNAV_LOG_FILE": str(tmp_path / "session.log"),
    }
    for key, value in session_values.items():
        monkeypatch.setenv(key, value)
    previous = {key: None for key in session_values}
    previous["ROS_DOMAIN_ID"] = "9"

    def _killpg(group_id: int, sig: int) -> None:
        assert group_id == _FakeProcess.pid
        if sig == 0:
            raise ProcessLookupError
        events.append(f"signal:{sig}")

    monkeypatch.setattr(os, "killpg", _killpg)
    process = _FakeProcess(events)
    log = _FakeLog(events)
    lifecycle = sim_tool._VNavSessionLifecycle(
        process=process,
        log_fh=log,
        session_dir=str(session_dir),
        session_values=session_values,
        previous_environment=previous,
        canonical_terrain_path=str(canonical),
        canonical_terrain_existed_at_start=True,
    )
    return lifecycle, process, log, events, session_dir, canonical


def test_lifecycle_stop_and_finalize_are_split_and_idempotent(
    monkeypatch, tmp_path
) -> None:
    (
        lifecycle,
        process,
        log,
        events,
        session_dir,
        canonical,
    ) = _make_lifecycle(monkeypatch, tmp_path)

    lifecycle.stop_process()
    lifecycle.stop_process()

    assert events == [f"signal:{signal.SIGTERM}", "wait", "log_close"]
    assert process.wait_calls == 1
    assert log.closed
    assert session_dir.exists()
    assert os.environ["ROS_DOMAIN_ID"] == "221"
    assert _read_grid(canonical)[0] == {(1, 1): pytest.approx(0.2)}

    lifecycle.finalize_session()
    lifecycle.finalize_session()

    assert os.environ["ROS_DOMAIN_ID"] == "9"
    assert "VECTOR_TERRAIN_MAP_FILE" not in os.environ
    assert not session_dir.exists()
    assert _read_grid(canonical)[0] == {
        (1, 1): pytest.approx(0.2),
        (2, 2): pytest.approx(0.6),
    }
    assert process.wait_calls == 1


def test_one_step_cleanup_remains_compatible(monkeypatch, tmp_path) -> None:
    lifecycle, process, _log, _events, session_dir, canonical = (
        _make_lifecycle(monkeypatch, tmp_path)
    )

    lifecycle.cleanup()
    lifecycle.cleanup()

    assert process.wait_calls == 1
    assert not session_dir.exists()
    assert len(_read_grid(canonical)[0]) == 2


def test_finalize_restores_environment_even_if_persistence_raises(
    monkeypatch, tmp_path
) -> None:
    lifecycle, _process, _log, _events, session_dir, _canonical = (
        _make_lifecycle(monkeypatch, tmp_path)
    )

    def _raise_persistence_error(*_args, **_kwargs) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        sim_tool,
        "_merge_session_terrain_map",
        _raise_persistence_error,
    )

    with pytest.raises(RuntimeError, match="boom"):
        lifecycle.finalize_session()

    assert os.environ["ROS_DOMAIN_ID"] == "9"
    assert "VECTOR_TERRAIN_MAP_FILE" not in os.environ
    assert not session_dir.exists()
    # The finalizer is marked complete, so retries cannot repeat side effects.
    lifecycle.finalize_session()


def test_late_go2_agent_assembly_exception_runs_managed_rollback(
    monkeypatch,
    tmp_path,
) -> None:
    """Errors after ROS readiness must not wait for process-level atexit."""
    from vector_os_nano.core import agent as agent_module
    from vector_os_nano.skills import go2 as go2_skills

    events: list[str] = []
    fake_agent = SimpleNamespace(
        _skill_registry=SimpleNamespace(register=lambda _skill: None),
        _world_model=SimpleNamespace(),
    )
    monkeypatch.setattr(
        agent_module,
        "Agent",
        lambda **_kwargs: fake_agent,
    )
    monkeypatch.setattr(go2_skills, "get_go2_skills", lambda: [])

    def _fail_scene_graph(*_args, **_kwargs):
        events.append("late_setup")
        raise RuntimeError("late scene-graph failure")

    monkeypatch.setattr(sim_tool, "_attach_sim_scene_graph", _fail_scene_graph)

    with pytest.raises(RuntimeError, match="late scene-graph failure"):
        sim_tool._run_vnav_startup_stage(
            lambda: sim_tool._finish_go2_startup(
                repo=str(tmp_path),
                base=SimpleNamespace(),
                with_arm=False,
                startup_proxies=[],
                abort_startup=lambda *_args, **_kwargs: None,
                assert_stack_running=lambda _phase: events.append(
                    "final_liveness"
                ),
            ),
            lambda: events.append("rollback"),
        )

    assert events == ["late_setup", "rollback"]


def test_keyboard_interrupt_during_managed_startup_runs_rollback() -> None:
    events: list[str] = []
    interrupt = KeyboardInterrupt()

    def _interrupt_startup() -> None:
        events.append("startup")
        raise interrupt

    with pytest.raises(KeyboardInterrupt) as caught:
        sim_tool._run_vnav_startup_stage(
            _interrupt_startup,
            lambda: events.append("rollback"),
        )

    assert caught.value is interrupt
    assert events == ["startup", "rollback"]


def test_start_go2_routes_late_setup_through_managed_stage() -> None:
    source = inspect.getsource(sim_tool.SimStartTool._start_go2)

    assert "return _run_vnav_startup_stage(" in source
    assert "_finish_go2_startup(" in source
    assert source.index("_finish_go2_startup(") < source.rindex(
        "return _run_vnav_startup_stage("
    )


def test_sigkill_path_waits_again_to_reap_launcher(
    monkeypatch,
    tmp_path,
) -> None:
    (
        lifecycle,
        _old_process,
        log,
        _old_events,
        _session_dir,
        _canonical,
    ) = _make_lifecycle(monkeypatch, tmp_path)
    events: list[str] = []
    group_alive = True

    class _TimeoutThenExitProcess:
        pid = _FakeProcess.pid

        def __init__(self) -> None:
            self.wait_calls: list[float] = []

        def wait(self, *, timeout: float) -> int:
            self.wait_calls.append(timeout)
            events.append(f"wait:{timeout}")
            if len(self.wait_calls) == 1:
                raise TimeoutError("TERM timeout")
            return 0

    process = _TimeoutThenExitProcess()
    lifecycle._process = process
    log.events = events

    def _killpg(group_id: int, sig: int) -> None:
        nonlocal group_alive
        assert group_id == process.pid
        if sig == 0:
            if group_alive:
                return
            raise ProcessLookupError
        events.append(f"signal:{sig}")
        if sig == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(os, "killpg", _killpg)

    lifecycle.stop_process()
    lifecycle.stop_process()

    assert process.wait_calls == [5, 2]
    assert events == [
        f"signal:{signal.SIGTERM}",
        "wait:5",
        f"signal:{signal.SIGKILL}",
        "wait:2",
        "log_close",
    ]
    assert lifecycle._process_stopped is True


def test_shutdown_disconnects_all_proxies_before_session_finalize() -> None:
    events: list[str] = []

    class _Proxy:
        _shared_runtime_used = False

        def __init__(self, name: str) -> None:
            self.name = name

        def disconnect(self) -> None:
            events.append(f"disconnect:{self.name}")

    base = _Proxy("base")
    base._sim_stop_process = lambda: events.append("stop")
    base._sim_finalize_session = lambda: events.append("finalize")
    base._sim_unregister_cleanup = lambda: events.append("unregister")
    base._sim_cleanup = lambda: events.append("legacy_cleanup")
    base._sim_subprocess = None
    base._sim_log_fh = None
    arm = _Proxy("arm")
    gripper = _Proxy("gripper")
    agent = SimpleNamespace(_base=base, _arm=arm, _gripper=gripper)

    sim_tool.SimStartTool._shutdown_agent(agent)

    assert events == [
        "stop",
        "disconnect:base",
        "disconnect:gripper",
        "disconnect:arm",
        "finalize",
        "unregister",
    ]


def test_default_domain_range_and_explicit_override(monkeypatch) -> None:
    import secrets

    monkeypatch.delenv("VECTOR_SIM_ROS_DOMAIN_ID", raising=False)
    monkeypatch.setenv("ROS_DOMAIN_ID", "7")
    monkeypatch.setattr(secrets, "randbelow", lambda span: 0)
    assert sim_tool._select_sim_ros_domain_id() == 215

    monkeypatch.setattr(secrets, "randbelow", lambda span: span - 1)
    assert sim_tool._select_sim_ros_domain_id() == 232

    for explicit in ("0", "232"):
        monkeypatch.setenv("VECTOR_SIM_ROS_DOMAIN_ID", explicit)
        assert sim_tool._select_sim_ros_domain_id() == int(explicit)


def test_prepare_skips_invalid_canonical_seed(monkeypatch, tmp_path) -> None:
    import tempfile

    canonical = tmp_path / "canonical.npz"
    canonical.write_bytes(b"broken")
    session_dir = tmp_path / "session"

    monkeypatch.setattr(
        sim_tool, "_canonical_terrain_map_path", lambda: str(canonical)
    )
    monkeypatch.setattr(
        tempfile,
        "mkdtemp",
        lambda *, prefix: (session_dir.mkdir() or str(session_dir)),
    )
    monkeypatch.delenv("VECTOR_VNAV_LOG_FILE", raising=False)

    _actual, values, previous = sim_tool._prepare_vnav_session_environment(221)
    try:
        assert not os.path.exists(values["VECTOR_TERRAIN_MAP_FILE"])
    finally:
        sim_tool._restore_vnav_session_environment(values, previous)
