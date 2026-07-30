# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""SimStartTool — start/stop robot simulations at runtime.

Allows V to spin up simulations mid-conversation without requiring
--sim or --sim-go2 flags at startup.

Supported simulations:
  arm   — SO-101 6-DOF arm (MuJoCoArm)
  go2   — Unitree Go2 quadruped (MuJoCoGo2 / IsaacSimProxy / GazeboGo2Proxy)

Supported backends:
  mujoco — MuJoCo (default, physics + textured rendering)
  mujoco — MuJoCo 3.x (lightweight fallback)
  gazebo — Gz Sim Harmonic (ROS2-native, open-source)
"""
from __future__ import annotations

import logging
from typing import Any

from vector_os_nano.vcli.tools.base import (
    PermissionResult,
    ToolContext,
    ToolResult,
    tool,
)

logger = logging.getLogger(__name__)

_VNAV_READY_MARKER = "Ready! Dog is standing still."
# Linux DDS participants map ROS domains onto UDP ports.  Domains 215..232 put
# the base DDS ports above the default ephemeral-port range while still keeping
# CLI-owned sessions away from the conventional domain 0 graph.
_SIM_ROS_DOMAIN_MIN = 215
_SIM_ROS_DOMAIN_MAX = 232
_ROS_DOMAIN_MAX = 232
_GO2_REQUIRED_NAV_NODES = frozenset(
    {
        "go2_vnav_bridge",
        "localPlanner",
        "tare_planner_node",
    }
)
_GO2_REQUIRED_NAV_NODE_ALIASES = (
    # The C++ executable defaults to far_planner_node, while the installed
    # launch file deliberately remaps it to far_planner.
    frozenset({"far_planner", "far_planner_node"}),
)
_VNAV_SESSION_PATH_KEYS = (
    "VECTOR_NAV_ACTIVE_FILE",
    "VECTOR_NAV_STALLED_FILE",
    "VECTOR_NAV_RESET_FILE",
    "VECTOR_NAV_REPLAY_FILE",
    "VECTOR_EXPLORE_FINISHED_FILE",
    "VECTOR_TERRAIN_MAP_FILE",
)


def _select_sim_ros_domain_id() -> int:
    """Choose a simulation-only ROS domain, ignoring generic shell pollution.

    ``ROS_DOMAIN_ID`` may have been exported by an unrelated ROS workspace.  A
    fresh Go2 run must not silently join that graph.  Only the deliberately
    named ``VECTOR_SIM_ROS_DOMAIN_ID`` override is honoured; otherwise a high,
    random domain is selected for this CLI-owned simulation session.
    """

    import os
    import secrets

    override = os.environ.get("VECTOR_SIM_ROS_DOMAIN_ID")
    if override is not None:
        try:
            domain_id = int(override.strip())
        except (AttributeError, ValueError) as exc:
            raise ValueError(
                "VECTOR_SIM_ROS_DOMAIN_ID must be an integer from 0 to 232"
            ) from exc
        if not 0 <= domain_id <= _ROS_DOMAIN_MAX:
            raise ValueError(
                "VECTOR_SIM_ROS_DOMAIN_ID must be an integer from 0 to 232"
            )
        return domain_id

    span = _SIM_ROS_DOMAIN_MAX - _SIM_ROS_DOMAIN_MIN + 1
    return _SIM_ROS_DOMAIN_MIN + secrets.randbelow(span)


def _prepare_vnav_session_environment(
    domain_id: int,
) -> tuple[str, dict[str, str], dict[str, str | None]]:
    """Create and install process-local paths for one navigation session.

    The parent CLI and launch child intentionally receive the same values.
    Keeping flags, logs, and terrain persistence in a unique directory prevents
    a crashed process from contaminating a later run.  A pre-existing terrain
    map is copied once as a read/write seed rather than shared live.
    """

    import os
    import shutil
    import tempfile

    session_dir = tempfile.mkdtemp(prefix=f"vector-vnav-{os.getpid()}-")
    explicit_log = os.environ.get("VECTOR_VNAV_LOG_FILE")
    values = {
        "ROS_DOMAIN_ID": str(domain_id),
        "VECTOR_NAV_ACTIVE_FILE": os.path.join(session_dir, "nav_active"),
        "VECTOR_NAV_STALLED_FILE": os.path.join(session_dir, "nav_stalled"),
        "VECTOR_NAV_RESET_FILE": os.path.join(session_dir, "nav_reset"),
        "VECTOR_NAV_REPLAY_FILE": os.path.join(session_dir, "nav_replay"),
        "VECTOR_EXPLORE_FINISHED_FILE": os.path.join(
            session_dir, "explore_finished"
        ),
        "VECTOR_TERRAIN_MAP_FILE": os.path.join(session_dir, "terrain_map.npz"),
        "VECTOR_VNAV_LOG_FILE": (
            os.path.abspath(os.path.expanduser(explicit_log))
            if explicit_log
            # Keep the retained diagnostic beside (not inside) the disposable
            # control directory, so SimStop can remove session state without
            # deleting the log path it reports to the user.
            else f"{session_dir}.log"
        ),
    }
    previous = {key: os.environ.get(key) for key in values}

    default_terrain = _canonical_terrain_map_path()
    if os.path.isfile(default_terrain):
        try:
            shutil.copy2(default_terrain, values["VECTOR_TERRAIN_MAP_FILE"])
            if _load_valid_terrain_map(values["VECTOR_TERRAIN_MAP_FILE"]) is None:
                os.remove(values["VECTOR_TERRAIN_MAP_FILE"])
                logger.warning(
                    "[sim_tool] canonical terrain map is invalid; starting "
                    "without terrain seed: %s",
                    default_terrain,
                )
        except OSError as exc:
            logger.warning(
                "[sim_tool] could not seed session terrain map from %s: %s",
                default_terrain,
                exc,
            )

    os.environ.update(values)
    return session_dir, values, previous


def _restore_vnav_session_environment(
    values: dict[str, str],
    previous: dict[str, str | None],
) -> None:
    """Restore env keys only when they still belong to this simulation."""

    import os

    for key, session_value in values.items():
        if os.environ.get(key) != session_value:
            continue
        old_value = previous.get(key)
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def _canonical_terrain_map_path() -> str:
    """Return the canonical cross-session terrain seed path."""

    import os

    from vector_os_nano.navigation.runtime_files import (
        DEFAULT_TERRAIN_MAP_FILE,
    )

    return os.path.abspath(os.path.expanduser(DEFAULT_TERRAIN_MAP_FILE))


def _load_valid_terrain_map(
    path: str,
) -> tuple[dict[tuple[int, int], float], float] | None:
    """Load one terrain NPZ after strict, non-pickle validation.

    The bridge treats ``ix``/``iy`` as integer voxel indices and ``z`` as the
    maximum observed height.  Refuse malformed, non-finite, non-integral, empty,
    or out-of-range data instead of allowing a damaged session file to replace
    the durable seed.
    """

    import math

    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            if not {"ix", "iy", "z", "voxel_size"}.issubset(data.files):
                return None
            ix = np.asarray(data["ix"])
            iy = np.asarray(data["iy"])
            z = np.asarray(data["z"])
            voxel_raw = np.asarray(data["voxel_size"])
    except (OSError, ValueError, TypeError, KeyError):
        return None

    if ix.ndim != 1 or iy.ndim != 1 or z.ndim != 1:
        return None
    if not (len(ix) == len(iy) == len(z)) or len(ix) == 0:
        return None
    if voxel_raw.size != 1:
        return None
    if not all(
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.complexfloating)
        for array in (ix, iy, z, voxel_raw)
    ):
        return None

    try:
        ix_float = ix.astype(np.float64, copy=False)
        iy_float = iy.astype(np.float64, copy=False)
        z_float = z.astype(np.float64, copy=False)
        voxel_size = float(voxel_raw.reshape(-1)[0])
    except (TypeError, ValueError, OverflowError):
        return None
    if (
        not np.isfinite(ix_float).all()
        or not np.isfinite(iy_float).all()
        or not np.isfinite(z_float).all()
        or not math.isfinite(voxel_size)
        or voxel_size <= 0.0
    ):
        return None
    if not np.equal(ix_float, np.trunc(ix_float)).all():
        return None
    if not np.equal(iy_float, np.trunc(iy_float)).all():
        return None

    index_limit = float(np.iinfo(np.int32).max)
    index_minimum = float(np.iinfo(np.int32).min)
    if (
        (ix_float < index_minimum).any()
        or (ix_float > index_limit).any()
        or (iy_float < index_minimum).any()
        or (iy_float > index_limit).any()
    ):
        return None

    grid: dict[tuple[int, int], float] = {}
    for raw_ix, raw_iy, raw_z in zip(ix_float, iy_float, z_float):
        key = (int(raw_ix), int(raw_iy))
        height = float(raw_z)
        previous_height = grid.get(key)
        if previous_height is None or height > previous_height:
            grid[key] = height
    return grid, voxel_size


def _atomic_save_terrain_map(
    path: str,
    grid: dict[tuple[int, int], float],
    voxel_size: float,
) -> None:
    """Atomically save a validated non-empty terrain grid."""

    import os
    import tempfile

    import numpy as np

    if not grid:
        raise ValueError("refusing to save an empty terrain map")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    ordered = sorted(grid)
    ix = np.asarray([key[0] for key in ordered], dtype=np.int32)
    iy = np.asarray([key[1] for key in ordered], dtype=np.int32)
    z = np.asarray([grid[key] for key in ordered], dtype=np.float32)

    temporary_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".terrain-map-",
            suffix=".npz",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
        np.savez_compressed(
            temporary_path,
            ix=ix,
            iy=iy,
            z=z,
            voxel_size=np.float32(voxel_size),
        )
        if _load_valid_terrain_map(temporary_path) is None:
            raise ValueError("generated terrain map failed validation")
        with open(temporary_path, "r+b") as temporary:
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        temporary_path = ""
    finally:
        if temporary_path:
            try:
                os.remove(temporary_path)
            except FileNotFoundError:
                pass


def _merge_session_terrain_map(
    session_path: str,
    canonical_path: str,
    *,
    canonical_existed_at_start: bool,
) -> bool:
    """Merge a stopped session's terrain into the canonical durable seed.

    Existing canonical voxels are always retained, and duplicate keys keep the
    maximum ``z``.  If ``/clear_memory`` removed a seed that existed when this
    session started, the missing file is an explicit forget signal and must not
    be resurrected from the bridge's later autosave.
    """

    import math
    import os

    if canonical_existed_at_start and not os.path.isfile(canonical_path):
        logger.info(
            "[sim_tool] canonical terrain seed was removed during the "
            "session; skipping terrain persistence"
        )
        return False

    session_map = _load_valid_terrain_map(session_path)
    if session_map is None:
        if os.path.exists(session_path):
            logger.warning(
                "[sim_tool] refusing to persist invalid session terrain map: %s",
                session_path,
            )
        return False
    session_grid, session_voxel_size = session_map

    canonical_grid: dict[tuple[int, int], float] = {}
    voxel_size = session_voxel_size
    if os.path.isfile(canonical_path):
        canonical_map = _load_valid_terrain_map(canonical_path)
        if canonical_map is None:
            logger.warning(
                "[sim_tool] refusing to overwrite invalid canonical terrain "
                "map: %s",
                canonical_path,
            )
            return False
        canonical_grid, canonical_voxel_size = canonical_map
        if not math.isclose(
            canonical_voxel_size,
            session_voxel_size,
            rel_tol=1e-7,
            abs_tol=1e-9,
        ):
            logger.warning(
                "[sim_tool] terrain voxel-size mismatch (session %.9g, "
                "canonical %.9g); keeping canonical map unchanged",
                session_voxel_size,
                canonical_voxel_size,
            )
            return False
        voxel_size = canonical_voxel_size

    merged = dict(canonical_grid)
    for key, height in session_grid.items():
        previous_height = merged.get(key)
        if previous_height is None or height > previous_height:
            merged[key] = height
    if len(merged) < len(canonical_grid):
        raise AssertionError("terrain union must never shrink canonical data")

    try:
        _atomic_save_terrain_map(canonical_path, merged, voxel_size)
    except (OSError, ValueError) as exc:
        logger.warning(
            "[sim_tool] could not atomically persist terrain map to %s: %s",
            canonical_path,
            exc,
        )
        return False
    return True


class _VNavSessionLifecycle:
    """Idempotent two-phase teardown for a managed navigation session."""

    def __init__(
        self,
        *,
        process: Any,
        log_fh: Any,
        session_dir: str,
        session_values: dict[str, str],
        previous_environment: dict[str, str | None],
        canonical_terrain_path: str,
        canonical_terrain_existed_at_start: bool,
    ) -> None:
        self._process = process
        self._log_fh = log_fh
        self._session_dir = session_dir
        self._session_values = session_values
        self._previous_environment = previous_environment
        self._canonical_terrain_path = canonical_terrain_path
        self._canonical_terrain_existed_at_start = (
            canonical_terrain_existed_at_start
        )
        self._process_stopped = False
        self._session_finalized = False

    def stop_process(self) -> None:
        """Stop the child process group and close its log, exactly once."""

        import os
        import signal
        import time

        if self._process_stopped:
            return

        group_id = self._process.pid
        try:
            os.killpg(group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            logger.debug("[sim_tool] failed to TERM vnav group", exc_info=True)

        leader_reaped = False
        try:
            self._process.wait(timeout=5)
            leader_reaped = True
        except Exception:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except Exception:
                logger.debug(
                    "[sim_tool] failed to KILL vnav group after TERM timeout",
                    exc_info=True,
                )
            # SIGKILL does not reap the Popen child.  Waiting again is required
            # or the long-lived CLI retains a zombie until garbage collection.
            try:
                self._process.wait(timeout=2)
                leader_reaped = True
            except Exception:
                logger.warning(
                    "[sim_tool] vnav launcher could not be reaped after SIGKILL",
                    exc_info=True,
                )

        def _group_is_alive() -> bool:
            try:
                os.killpg(group_id, 0)
            except ProcessLookupError:
                return False
            except Exception:
                # Permission/OS errors mean disappearance cannot be proved.
                logger.debug(
                    "[sim_tool] could not inspect vnav process group",
                    exc_info=True,
                )
                return True
            return True

        group_alive = _group_is_alive()
        if group_alive:
            # The launcher may exit before one of its descendants.  Kill the
            # otherwise leaderless process group and wait briefly until the
            # kernel confirms that no member remains.
            try:
                os.killpg(group_id, signal.SIGKILL)
            except Exception:
                logger.debug(
                    "[sim_tool] failed to KILL surviving vnav descendants",
                    exc_info=True,
                )
            for _ in range(20):
                if not _group_is_alive():
                    group_alive = False
                    break
                time.sleep(0.05)
            else:
                group_alive = True

        try:
            self._log_fh.close()
        except Exception:
            pass
        # Successful calls are strictly idempotent.  If either the direct child
        # was not reaped or its process group still exists, leave the flag clear
        # so a subsequent cleanup/finalizer call gets one bounded retry.
        self._process_stopped = leader_reaped and not group_alive
        if not self._process_stopped:
            logger.warning(
                "[sim_tool] vnav process teardown incomplete "
                "(leader_reaped=%s, group_alive=%s)",
                leader_reaped,
                group_alive,
            )

    def finalize_session(self) -> None:
        """Persist terrain, restore the parent env, and delete session state."""

        import os
        import shutil

        if self._session_finalized:
            return
        # A direct finalizer call remains safe; normal shutdown deliberately
        # invokes stop_process(), disconnects proxies with the session env still
        # installed, then invokes this method.
        self.stop_process()
        try:
            _merge_session_terrain_map(
                self._session_values["VECTOR_TERRAIN_MAP_FILE"],
                self._canonical_terrain_path,
                canonical_existed_at_start=(
                    self._canonical_terrain_existed_at_start
                )
            )
        finally:
            # A persistence error must not strand a stale ROS domain or session
            # flag in the long-lived CLI process.
            _restore_vnav_session_environment(
                self._session_values,
                self._previous_environment,
            )
            for key in _VNAV_SESSION_PATH_KEYS:
                try:
                    os.remove(self._session_values[key])
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.debug(
                        "[sim_tool] failed to remove session file %s",
                        self._session_values[key],
                        exc_info=True,
                    )
            shutil.rmtree(self._session_dir, ignore_errors=True)
            self._session_finalized = True

    def cleanup(self) -> None:
        """Compatibility wrapper for callers that need one-step teardown."""

        self.stop_process()
        self.finalize_session()


def _wait_for_vnav_ready_marker(
    process: Any,
    log_fh: Any,
    log_path: str,
    *,
    timeout_s: float = 90.0,
    poll_s: float = 0.2,
) -> None:
    """Wait for the launcher's truthful Ready marker or fail with its log."""

    import time

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    offset = 0
    carry = ""
    while True:
        error = _vnav_process_exit_error(
            process, log_fh, log_path, phase="launcher readiness"
        )
        if error is not None:
            raise error
        try:
            log_fh.flush()
        except Exception:
            pass
        try:
            with open(log_path, encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
                candidate = carry + chunk
                if _VNAV_READY_MARKER in candidate:
                    return
                carry = candidate[-len(_VNAV_READY_MARKER) :]
        except OSError:
            pass
        if time.monotonic() >= deadline:
            excerpt = _startup_log_excerpt(log_path)
            detail = f" Diagnostic: {excerpt}" if excerpt else ""
            raise TimeoutError(
                "Go2 navigation launcher did not report Ready before "
                f"{timeout_s:.1f}s.{detail} Full log: {log_path}"
            )
        time.sleep(max(0.01, float(poll_s)))


def _endpoint_match_count(endpoint: Any, method_name: str) -> int:
    if endpoint is None:
        return 0
    method = getattr(endpoint, method_name, None)
    if not callable(method):
        return 0
    try:
        return int(method())
    except Exception:
        return 0


def _node_publisher_count(node: Any, topic: str) -> int:
    method = getattr(node, "count_publishers", None)
    if not callable(method):
        return 0
    try:
        return int(method(topic))
    except Exception:
        return 0


def _go2_navigation_readiness_missing(base: Any) -> tuple[str, ...]:
    """Return missing endpoints and planner-data readiness for a Go2 proxy."""

    missing: list[str] = []
    if not bool(getattr(base, "_connected", False)):
        missing.append("ROS proxy connection")
    if getattr(base, "_last_odom", None) is None:
        missing.append("/state_estimation odometry")

    node = getattr(base, "_node", None)
    if node is None:
        missing.append("go2_agent_proxy node")
        return tuple(missing)

    publisher_requirements = (
        ("_goal_control_pub", "/vector_os/nav_goal_control bridge subscriber"),
        (
            "_segment_control_pub",
            "/vector_os/nav_segment_control bridge subscriber",
        ),
        ("_goal_pub", "/goal_point FAR subscriber"),
        ("_waypoint_pub", "/way_point localPlanner subscriber"),
    )
    for attribute, label in publisher_requirements:
        if (
            _endpoint_match_count(
                getattr(base, attribute, None), "get_subscription_count"
            )
            < 1
        ):
            missing.append(label)

    ack_subscription = getattr(base, "_segment_ack_subscription", None)
    ack_publishers = _endpoint_match_count(
        ack_subscription, "get_publisher_count"
    )
    if ack_publishers < 1:
        ack_publishers = _node_publisher_count(
            node, "/vector_os/nav_segment_ack"
        )
    if ack_publishers < 1:
        missing.append("/vector_os/nav_segment_ack bridge publisher")

    topic_publishers = (
        (
            "/vector_os/nav_goal_telemetry",
            "/vector_os/nav_goal_telemetry bridge publisher",
        ),
        ("/path", "/path localPlanner publisher"),
    )
    for topic, label in topic_publishers:
        if _node_publisher_count(node, topic) < 1:
            missing.append(label)

    get_node_names = getattr(node, "get_node_names", None)
    if callable(get_node_names):
        try:
            node_names = {
                str(name).strip().lstrip("/") for name in get_node_names()
            }
        except Exception:
            node_names = set()
    else:
        node_names = set()
    for required_name in sorted(_GO2_REQUIRED_NAV_NODES - node_names):
        missing.append(f"ROS node {required_name}")
    for aliases in _GO2_REQUIRED_NAV_NODE_ALIASES:
        if node_names.isdisjoint(aliases):
            missing.append(f"ROS node {'/'.join(sorted(aliases))}")

    # FAR advertises its topics before it can accept a goal.  Its callback
    # explicitly drops /goal_point while is_graph_init_ is false, so endpoint
    # discovery alone is not a product readiness contract.
    vgraph_ready = getattr(base, "far_vgraph_ready", None)
    if not callable(vgraph_ready):
        missing.append("FAR non-empty V-Graph readiness signal")
    else:
        try:
            ready = bool(vgraph_ready())
        except Exception:
            ready = False
        if not ready:
            diagnostics_reader = getattr(base, "far_vgraph_diagnostics", None)
            diagnostics: dict[str, Any] = {}
            if callable(diagnostics_reader):
                try:
                    candidate = diagnostics_reader()
                    if isinstance(candidate, dict):
                        diagnostics = candidate
                except Exception:
                    pass
            status = str(diagnostics.get("status") or "not_ready")
            node_count = int(diagnostics.get("node_count") or 0)
            missing.append(
                "FAR non-empty V-Graph "
                f"(status={status}, global_vertex_nodes={node_count})"
            )

    return tuple(missing)


def _wait_for_go2_navigation_ready(
    base: Any,
    *,
    timeout_s: float = 35.0,
    stable_s: float = 0.75,
    poll_s: float = 0.1,
    liveness_check: Any = None,
) -> None:
    """Require endpoints and a non-empty FAR graph to remain stably ready."""

    import time

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    stable_since: float | None = None
    last_missing: tuple[str, ...] = ()
    while True:
        if callable(liveness_check):
            liveness_check()
        now = time.monotonic()
        last_missing = _go2_navigation_readiness_missing(base)
        if not last_missing:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= max(0.0, float(stable_s)):
                return
        else:
            stable_since = None
        if now >= deadline:
            detail = ", ".join(last_missing) or "DDS graph never stabilized"
            raise TimeoutError(
                "Go2 navigation control plane was not ready before "
                f"{timeout_s:.1f}s; missing: {detail}"
            )
        time.sleep(max(0.01, float(poll_s)))


def _odometry_sample_token(base: Any) -> tuple[Any, ...] | None:
    """Extract a changing identity from the latest odometry message."""

    message = getattr(base, "_last_odom", None)
    if message is None:
        return None
    stamp = getattr(getattr(message, "header", None), "stamp", None)
    return (
        getattr(stamp, "sec", None),
        getattr(stamp, "nanosec", None),
        id(message),
    )


def _measure_go2_odometry_rate(
    base: Any,
    *,
    window_s: float = 1.2,
    poll_s: float = 0.02,
    clock: Any = None,
    sleep: Any = None,
    liveness_check: Any = None,
) -> float:
    """Measure fresh odometry transitions over a bounded startup window."""

    import time

    now_fn = clock or time.monotonic
    sleep_fn = sleep or time.sleep
    window = max(0.1, float(window_s))
    start = now_fn()
    deadline = start + window
    previous_token = _odometry_sample_token(base)
    transitions = 0
    while now_fn() < deadline:
        if callable(liveness_check):
            liveness_check()
        remaining = deadline - now_fn()
        sleep_fn(min(max(0.005, float(poll_s)), max(0.0, remaining)))
        token = _odometry_sample_token(base)
        if token is not None and token != previous_token:
            transitions += 1
        previous_token = token
    elapsed = max(1e-6, now_fn() - start)
    return transitions / elapsed


def _require_go2_odometry_performance(
    base: Any,
    *,
    minimum_hz: float = 5.0,
    window_s: float = 1.2,
    liveness_check: Any = None,
) -> float:
    """Reject a discovered but overloaded simulation before its first goal."""

    measured_hz = _measure_go2_odometry_rate(
        base,
        window_s=window_s,
        liveness_check=liveness_check,
    )
    if measured_hz < float(minimum_hz):
        raise RuntimeError(
            "Go2 startup_performance failed: /state_estimation updated at "
            f"{measured_hz:.1f} Hz over {window_s:.1f}s; required at least "
            f"{minimum_hz:.1f} Hz before navigation"
        )
    return measured_hz


def _startup_log_excerpt(log_path: str, *, max_lines: int = 8) -> str:
    """Return a compact startup diagnostic suitable for the native CLI."""

    diagnostic_tokens = (
        "traceback",
        "error",
        "exception",
        "failed",
        "unboundlocalerror",
        "segmentation",
        "core dumped",
        "aborted",
    )
    diagnostics: list[str] = []
    fallback: list[str] = []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as stream:
            for raw_line in stream:
                line = raw_line.strip()
                if not line:
                    continue
                line = line[-300:]
                fallback.append(line)
                if len(fallback) > max_lines:
                    fallback.pop(0)
                lowered = line.lower()
                if any(token in lowered for token in diagnostic_tokens):
                    diagnostics.append(line)
                    if len(diagnostics) > max_lines:
                        diagnostics.pop(0)
    except OSError:
        return ""
    return " | ".join(diagnostics or fallback)


def _vnav_process_exit_error(
    process: Any,
    log_fh: Any,
    log_path: str,
    *,
    phase: str,
) -> RuntimeError | None:
    """Build a fail-loud error when the navigation subprocess has died."""

    return_code = process.poll()
    if return_code is None:
        return None
    try:
        log_fh.flush()
    except Exception:
        pass
    excerpt = _startup_log_excerpt(log_path)
    detail = f" Diagnostic: {excerpt}" if excerpt else ""
    return RuntimeError(
        f"Go2 navigation stack exited during {phase} "
        f"(status {return_code}).{detail} Full log: {log_path}"
    )


def _require_go2_proxy_ready(base: Any) -> None:
    """Require a live ROS connection and real bridge odometry."""

    if not bool(getattr(base, "_connected", False)):
        raise ConnectionError("Go2 ROS2 proxy did not connect")
    if getattr(base, "_last_odom", None) is None:
        raise TimeoutError(
            "Go2 bridge published no /state_estimation odometry during startup"
        )


def _require_piper_proxy_ready(arm: Any, gripper: Any) -> None:
    """Require the requested Piper embodiment to have real bridge state."""

    if not bool(getattr(arm, "_connected", False)):
        raise ConnectionError("Piper arm ROS2 proxy did not connect")
    if float(getattr(arm, "_last_joint_state_ts", 0.0)) <= 0.0:
        raise TimeoutError(
            "Piper bridge published no /piper/joint_state during startup"
        )
    if not bool(getattr(gripper, "_connected", False)):
        raise ConnectionError("Piper gripper ROS2 proxy did not connect")


def _attach_sim_scene_graph(agent: Any, base: Any, repo: str) -> Any:
    """Attach the mode-correct SceneGraph to a simulated mobile agent.

    Known-layout simulation gets room-layout priors immediately, so a fresh bare
    CLI can navigate by room name without first running ``explore``. Unknown-world
    exploration uses a separate persistence file and never imports those priors.
    """

    import os

    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.navigation.world_mode import WorldMode, world_mode_for_agent

    mode = world_mode_for_agent(agent)
    filename = (
        "scene_graph.yaml"
        if mode is WorldMode.KNOWN_LAYOUT
        else "scene_graph_unknown_exploration.yaml"
    )
    persist_path = os.path.expanduser(f"~/.vector_os_nano/{filename}")
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    scene_graph = SceneGraph(persist_path=persist_path)
    scene_graph.load()

    if mode is WorldMode.KNOWN_LAYOUT:
        layout_path = os.path.join(repo, "config", "room_layout.yaml")
        # The simulated house geometry is deterministic. Correct stale persisted
        # centres/doors on every startup while retaining learned objects, room
        # descriptions, connections, and visit history.
        scene_graph.load_layout(layout_path, overwrite=True)
        scene_graph.save()

    agent._spatial_memory = scene_graph
    agent._world_mode = mode.value
    config = getattr(agent, "_config", None)
    if isinstance(config, dict):
        config["world_mode"] = mode.value
    base._scene_graph = scene_graph

    stats = scene_graph.stats()
    logger.info(
        "[SceneGraph] mode=%s rooms=%d objects=%d persistence=%s",
        mode.value,
        stats["rooms"],
        stats["objects"],
        persist_path,
    )
    return scene_graph


def _run_vnav_startup_stage(operation: Any, teardown: Any) -> Any:
    """Run one managed-startup stage and roll back on every throwable exit.

    ``SimStartTool.execute`` converts ordinary startup exceptions into a
    ``ToolResult`` and keeps the CLI process alive, so relying on atexit here
    would strand the ROS context and child stack.  KeyboardInterrupt and
    SystemExit need the same rollback before they propagate.
    """

    try:
        return operation()
    except BaseException:
        try:
            teardown()
        except BaseException:
            logger.warning(
                "[sim_tool] managed startup rollback failed",
                exc_info=True,
            )
        raise


def _finish_go2_startup(
    *,
    repo: str,
    base: Any,
    with_arm: bool,
    startup_proxies: list[Any],
    abort_startup: Any,
    assert_stack_running: Any,
) -> Any:
    """Build the Go2 agent after transport readiness has been established."""

    import os

    # Load config for API key
    from vector_os_nano.core.config import load_config

    cfg_path = os.path.join(repo, "config", "user.yaml")
    cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
    api_key = cfg.get("llm", {}).get("api_key") or os.environ.get(
        "OPENROUTER_API_KEY", ""
    )

    # Piper arm + gripper proxies — bridge advertises /piper/* topics
    # when VECTOR_SIM_WITH_ARM=1 was set in child_env above.
    piper_arm = None
    piper_gripper = None
    piper_setup_error: Exception | None = None
    if with_arm:
        try:
            from vector_os_nano.hardware.sim.mujoco_go2 import (
                _build_room_scene_xml,
            )

            scene_xml = str(_build_room_scene_xml(with_arm=True))

            from vector_os_nano.hardware.sim.piper_ros2_proxy import (
                PiperGripperROS2Proxy,
                PiperROS2Proxy,
            )

            piper_arm = PiperROS2Proxy(
                base_proxy=base,
                scene_xml_path=scene_xml,
            )
            startup_proxies.append(piper_arm)
            piper_arm.connect()
            piper_gripper = PiperGripperROS2Proxy(scene_xml_path=scene_xml)
            startup_proxies.append(piper_gripper)
            piper_gripper.connect()
            _require_piper_proxy_ready(piper_arm, piper_gripper)
            logger.info("[sim_tool] Piper proxies connected (arm + gripper)")
        except ImportError as exc:
            piper_setup_error = exc
            logger.debug("[sim_tool] Piper proxy unavailable (no ROS2): %s", exc)
            piper_arm = None
            piper_gripper = None
        except Exception as exc:
            piper_setup_error = exc
            logger.error("[sim_tool] Piper proxy setup failed: %s", exc)
            for proxy in (piper_gripper, piper_arm):
                if proxy is not None:
                    try:
                        proxy.disconnect()
                    except Exception:
                        pass
            piper_arm = None
            piper_gripper = None
        if piper_arm is None or piper_gripper is None:
            reason = (
                f": {piper_setup_error}" if piper_setup_error is not None else ""
            )
            abort_startup(
                f"Go2 + Piper was requested, but Piper did not become ready{reason}",
                base_proxy=base,
            )

    from vector_os_nano.core.agent import Agent  # type: ignore[import]

    agent = Agent(
        base=base,
        arm=piper_arm,
        gripper=piper_gripper,
        llm_api_key=api_key,
        config=cfg,
    )

    # World model starts empty by design — objects are populated by the
    # perception pipeline at runtime (DetectSkill / LookSkill), NOT by
    # reading ground truth from the MJCF. This matches the SO-101 pattern:
    # camera -> VLM/tracker -> 3D pose -> world_model.
    #
    # Escape hatch for offline demos only: set VECTOR_SIM_DEMO_GROUND_TRUTH=1
    # to pre-populate from MJCF body names (treats sim as cheat knowledge).
    if with_arm and os.environ.get("VECTOR_SIM_DEMO_GROUND_TRUTH") == "1":
        try:
            from vector_os_nano.hardware.sim.mujoco_go2 import (
                _build_room_scene_xml,
            )

            scene_xml = str(_build_room_scene_xml(with_arm=True))
            n = SimStartTool._populate_pickables_from_mjcf(
                agent._world_model,
                scene_xml,
            )
            logger.warning(
                "[sim_tool] DEMO ground-truth populate: %d pickable objects "
                "registered from MJCF (VECTOR_SIM_DEMO_GROUND_TRUTH=1). "
                "This bypasses perception — use only for no-perception demos.",
                n,
            )
        except Exception as exc:
            logger.warning("[sim_tool] demo-populate failed: %s", exc)

    # Go2 perception is sourced from the SysNav sibling workspace via the
    # sysnav_bridge adapter (vector_os_nano/integrations/sysnav_bridge/).
    # We do NOT instantiate an in-process VLM detector here; the bridge
    # populates world_model when SysNav publishes /object_nodes_list.
    # Until the bridge is wired (v2.4), agent._perception stays None and
    # MobilePick returns object_not_found against an empty world_model.
    agent._perception = None
    agent._calibration = None

    # Go2 skills
    from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]

    for skill in get_go2_skills():
        agent._skill_registry.register(skill)
    # Local manipulation (Piper pick/place) is wired whenever the user
    # launched go2 WITH the arm — per the North Star, a capability behind a
    # flag is NOT done, and `with_arm=True` is the user explicitly asking for
    # the arm. Escape hatch to disable for a bare-mobility demo:
    # VECTOR_ENABLE_MANIPULATION=0 (default ON for with_arm).
    manipulation_on = os.environ.get("VECTOR_ENABLE_MANIPULATION", "1") != "0"
    if piper_arm is not None and manipulation_on:
        from vector_os_nano.skills.mobile_pick import MobilePickSkill
        from vector_os_nano.skills.mobile_place import MobilePlaceSkill
        from vector_os_nano.skills.pick_top_down import PickTopDownSkill
        from vector_os_nano.skills.place_top_down import PlaceTopDownSkill

        agent._skill_registry.register(PickTopDownSkill())
        agent._skill_registry.register(PlaceTopDownSkill())
        agent._skill_registry.register(MobilePickSkill())
        agent._skill_registry.register(MobilePlaceSkill())
        # Perception-driven grasp (the honest North-Star path): real RGB-D
        # from the go2 d435 (bridge -> /camera/image + /camera/depth -> proxy)
        # + Moondream VLM + EdgeTAM -> 3D grasp point (NOT ground truth).
        # Registered LAST so it wins the shared 抓/grab aliases on the empty-
        # world-model path (it needs no pre-populated world model; PickTopDown
        # does). holding_object grades GROUNDED on this bare-cli path: the
        # bridge welds the object on gripper-close and publishes per-body world
        # xpos + per-weld active over /piper/object_state; the Piper proxies
        # expose get_object_positions() + weld_is_active() + weld-backed
        # is_holding() the verify oracle + actor_causation read (D36).
        from vector_os_nano.perception.go2_grasp_perception import (
            Go2GraspPerception,
        )
        from vector_os_nano.skills.perception_grasp import PerceptionGraspSkill

        # Bridge publishes 320×240; intrinsics must match the actual frame size.
        agent._perception = Go2GraspPerception(base, width=320, height=240)
        agent._skill_registry.register(PerceptionGraspSkill())
        logger.info(
            "[sim_tool] perception-grasp wired: "
            "Go2GraspPerception + PerceptionGraspSkill"
        )

    # VLM perception (GPT-4o via OpenRouter)
    if api_key:
        try:
            from vector_os_nano.perception.vlm_go2 import Go2VLMPerception

            agent._vlm = Go2VLMPerception(config={"api_key": api_key})
        except Exception:
            agent._vlm = None

    # Mode-aware SceneGraph: known layout gets priors at startup; unknown
    # exploration stays isolated and discovery-only.
    _attach_sim_scene_graph(agent, base, repo)
    assert_stack_running("final readiness")

    return agent


def locate_mjpython(executable: str | None = None) -> str | None:
    """Locate the ``mjpython`` launcher for the running environment.

    mjpython lives next to the running interpreter (the venv's ``bin/``), so it is
    derived from ``sys.executable`` — deliberately NOT ``resolve()``-d: resolving
    follows the venv's ``python`` symlink to the base interpreter's ``bin``, where
    mjpython is absent. Falls back to ``shutil.which``. Returns an absolute path
    string, or ``None`` when mjpython is not installed.

    (Regression guard: a prior off-by-one ``parents[N]``-from-``__file__`` path
    resolved to ``$HOME``, so mjpython was never found and the viewer silently fell
    back to headless. Deriving from ``sys.executable`` is depth-independent.)
    """
    import os
    import shutil
    import sys
    from pathlib import Path

    exe = executable or sys.executable
    cand = Path(exe).parent / "mjpython"
    if cand.is_file() and os.access(str(cand), os.X_OK):
        return str(cand)
    return shutil.which("mjpython")


@tool(
    name="start_simulation",
    description="Start a robot simulation (arm or go2 quadruped) with isaac, mujoco, or gazebo backend. No restart needed.",
    read_only=False,
    permission="ask",
)
class SimStartTool:
    """Start a simulation and register its skills into the tool registry.

    Backends: mujoco (default), gazebo (Gz Sim Harmonic), isaac (Docker, archived).
    """

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "sim_type": {
                "type": "string",
                "enum": ["arm", "go2"],
                "description": "Which simulation to start: 'arm' (SO-101) or 'go2' (Unitree Go2)",
            },
            "gui": {
                "type": "boolean",
                "description": (
                    "Open the viewer window (default: true). When the user says "
                    "'headless' / '无窗口' / 'no window', pass gui=false to run "
                    "without a display. A window is the default; gui=false suppresses it."
                ),
                "default": True,
            },
            "backend": {
                "type": "string",
                "enum": ["isaac", "mujoco", "gazebo"],
                "default": "mujoco",
                "description": (
                    "Simulation backend: 'mujoco' (default, physics + textured rendering), "
                    "'gazebo' (Gz Sim Harmonic), or 'isaac' (Docker, archived)"
                ),
            },
            "with_arm": {
                "type": "boolean",
                "description": (
                    "ONLY for sim_type='go2'. True = mount Piper 6-DoF arm on "
                    "Go2's back (enables pick/place; forces sinusoidal gait "
                    "because convex_mpc is 12-DoF-only). False = pure Go2 "
                    "(smoother MPC gait, no manipulation). BEFORE calling this "
                    "tool, ASK the user which mode they want — both have real "
                    "tradeoffs. If the user gives an ambiguous command like "
                    "'go2sim' or '启动仿真', ask before calling."
                ),
            },
        },
        "required": ["sim_type"],
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        sim_type: str = params["sim_type"]
        gui: bool = params.get("gui", True)
        backend: str = params.get("backend", "mujoco")
        with_arm: bool = bool(params.get("with_arm", False))
        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)

        # Check if already running
        current_agent = app.get("agent")
        if current_agent is not None:
            current_arm = getattr(current_agent, "_arm", None)
            current_base = getattr(current_agent, "_base", None)
            if sim_type == "arm" and current_arm is not None:
                return ToolResult(content=f"Arm sim already running: {type(current_arm).__name__}")
            if sim_type == "go2" and current_base is not None:
                return ToolResult(content=f"Go2 sim already running: {type(current_base).__name__}")

        try:
            if backend == "isaac":
                if sim_type == "go2":
                    agent = self._start_isaac_go2()
                elif sim_type == "arm":
                    agent = self._start_isaac_arm()
                else:
                    return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
            elif backend == "gazebo":
                if sim_type == "go2":
                    agent = self._start_gazebo_go2()
                else:
                    return ToolResult(
                        content="Gazebo backend only supports go2",
                        is_error=True,
                    )
            else:
                # Default: mujoco backend (existing paths unchanged)
                if sim_type == "arm":
                    # Pure-NL case: user launched plain vector-cli (no --sim flag),
                    # then asked to start the arm sim. The startup re-exec guard
                    # did not fire. If a window is wanted but the viewer cannot open
                    # in this interpreter, re-exec the whole CLI under mjpython --sim.
                    if gui and not self._viewer_available():
                        import os as _os
                        import sys as _sys
                        _in_pytest = "pytest" in _sys.modules or bool(_os.environ.get("PYTEST_CURRENT_TEST"))
                        if not _in_pytest:
                            self._reexec_under_mjpython_with_sim()
                            # _reexec_under_mjpython_with_sim() returned → mjpython missing;
                            # fall through to headless launch.
                            gui = False
                    agent = self._start_arm(gui=gui)
                elif sim_type == "go2":
                    agent = self._start_go2(gui=gui, with_arm=with_arm)
                else:
                    return ToolResult(content=f"Unknown sim type: {sim_type}", is_error=True)
        except Exception as exc:
            return ToolResult(content=f"Failed to start {sim_type} sim: {exc}", is_error=True)

        # Update app state
        app["agent"] = agent
        app["scene_graph"] = getattr(agent, "_spatial_memory", None)
        app["skill_registry"] = getattr(agent, "_skill_registry", None)

        # Register skill tools under the 'robot' category (matches the --sim path)
        registry = app.get("registry")
        if registry is not None:
            from vector_os_nano.vcli.tools.skill_wrapper import wrap_skills
            for skill_tool in wrap_skills(agent):
                registry.register(skill_tool, category="robot")
            # The bare dev-world startup disabled robot/diag; re-enable now that a
            # robot is connected so skill + diag/status tools become visible.
            if hasattr(registry, "enable_category"):
                registry.enable_category("robot")
                registry.enable_category("diag")

        # Rebuild the system prompt as a LIVE DynamicSystemPrompt with an arm-aware
        # robot context, so subsequent turns correctly see the connected hardware
        # (a bare list would freeze state and drop the [Robot State] block).
        engine = app.get("engine")
        if engine is not None:
            from vector_os_nano.vcli.prompt import build_system_prompt
            from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
            from vector_os_nano.vcli.robot_context import RobotContextProvider
            from vector_os_nano.vcli.worlds import resolve_world
            provider = RobotContextProvider(
                base=getattr(agent, "_base", None),
                scene_graph=getattr(agent, "_spatial_memory", None),
                arm=getattr(agent, "_arm", None),
            )
            app["robot_ctx_provider"] = provider
            static_blocks = build_system_prompt(
                agent=agent, cwd=context.cwd, robot_context=provider,
                world=resolve_world(agent),
            )
            engine._system_prompt = DynamicSystemPrompt(static_blocks, provider)
            # Reinit VGG with new agent so verifier has live robot state
            try:
                engine.init_vgg(
                    agent=agent,
                    skill_registry=getattr(agent, "_skill_registry", None),
                    on_vgg_step=app.get("vgg_step_callback"),
                    world=resolve_world(agent),
                )
            except Exception as _exc:
                logger.warning("init_vgg after sim start failed: %s", _exc)

        # Report SceneGraph status
        sg = getattr(agent, "_spatial_memory", None)
        sg_stats = sg.stats() if sg else {}
        sg_info = ""
        if sg_stats.get("rooms", 0) > 0:
            sg_info = f" SceneGraph restored: {sg_stats['rooms']} rooms."

        base = getattr(agent, "_base", None)
        runtime_info = ""
        domain_id = getattr(base, "_sim_ros_domain_id", None)
        log_path = getattr(base, "_sim_log_path", None)
        startup_odom_hz = getattr(base, "_sim_startup_odom_hz", None)
        startup_vgraph_nodes = getattr(
            base, "_sim_startup_vgraph_nodes", None
        )
        if domain_id is not None and log_path:
            rate_info = (
                f", odom {float(startup_odom_hz):.1f} Hz"
                if startup_odom_hz is not None
                else ""
            )
            graph_info = (
                f", FAR graph {int(startup_vgraph_nodes)} nodes"
                if startup_vgraph_nodes is not None
                else ""
            )
            runtime_info = (
                f" ROS domain {domain_id}{rate_info}{graph_info}; "
                f"log: {log_path}."
            )

        hw_name = type(getattr(agent, "_arm", None) or base).__name__
        skill_count = len(agent._skill_registry.list_skills()) if hasattr(agent, "_skill_registry") else 0
        return ToolResult(
            content=(
                f"Started {sim_type} simulation: {hw_name}, {skill_count} "
                f"skills registered.{sg_info}{runtime_info}"
            )
        )

    @staticmethod
    def _shutdown_agent(agent: Any) -> str:
        """Tear down a running sim agent: kill subprocesses, disconnect hardware.

        Returns a short human-readable summary of what was stopped.
        """
        import os
        import signal
        parts: list[str] = []
        base = getattr(agent, "_base", None)
        arm = getattr(agent, "_arm", None)
        gripper = getattr(agent, "_gripper", None)
        used_shared_runtime = any(
            bool(getattr(proxy, "_shared_runtime_used", False))
            for proxy in (base, arm, gripper)
            if proxy is not None
        )
        sim_finalize_session = None
        sim_unregister_cleanup = None

        # Go2 phase 1: stop the child stack first, but deliberately leave the
        # session environment installed until every ROS proxy has disconnected.
        if base is not None:
            proc = getattr(base, "_sim_subprocess", None)
            sim_stop_process = getattr(base, "_sim_stop_process", None)
            sim_finalize_session = getattr(
                base, "_sim_finalize_session", None
            )
            sim_unregister_cleanup = getattr(
                base, "_sim_unregister_cleanup", None
            )
            sim_cleanup = getattr(base, "_sim_cleanup", None)
            if callable(sim_stop_process):
                try:
                    sim_stop_process()
                    parts.append("sim process tree stopped")
                except Exception as exc:
                    parts.append(f"sim process tree stop failed: {exc}")
            elif callable(sim_cleanup):
                # Compatibility with simulations created before the split
                # lifecycle API.  Such cleanup may also restore its env.
                try:
                    sim_cleanup()
                    parts.append("sim process tree stopped")
                except Exception as exc:
                    parts.append(f"sim process tree cleanup failed: {exc}")
            elif proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                    parts.append("sim subprocess stopped")
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        parts.append("sim subprocess force-killed")
                    except Exception as exc:
                        parts.append(f"subprocess kill failed: {exc}")
            log_fh = getattr(base, "_sim_log_fh", None)
            if log_fh is not None:
                try:
                    log_fh.close()
                except Exception:
                    pass
            try:
                base.disconnect()
                parts.append(f"{type(base).__name__} disconnected")
            except Exception:
                pass

        # Arm + gripper (SO-101 arm-only sim, OR PiperROS2Proxy in go2-with-arm)
        if gripper is not None:
            try:
                gripper.disconnect()
                parts.append(f"{type(gripper).__name__} disconnected")
            except Exception:
                pass
        if arm is not None:
            try:
                arm.disconnect()
                parts.append(f"{type(arm).__name__} disconnected")
            except Exception:
                pass

        if used_shared_runtime:
            try:
                from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

                get_ros2_runtime().shutdown_if_idle()
            except Exception:
                pass

        # Go2 phase 2: the proxy disconnect paths have finished resolving all
        # session-scoped flags.  It is now safe to persist terrain, restore the
        # parent environment, and delete the disposable session directory.
        if callable(sim_finalize_session):
            try:
                sim_finalize_session()
                parts.append("sim session finalized")
            except Exception as exc:
                parts.append(f"sim session finalization failed: {exc}")
            finally:
                if callable(sim_unregister_cleanup):
                    try:
                        sim_unregister_cleanup()
                    except Exception:
                        pass

        return "; ".join(parts) or "nothing to stop"

    @staticmethod
    def _viewer_available() -> bool:
        """Return True if the MuJoCo passive viewer can open a window.

        On macOS the viewer requires running under mjpython (which sets
        mujoco.viewer._MJPYTHON). Returns True on non-macOS or when already
        under mjpython; False otherwise.
        """
        import sys as _sys
        if _sys.platform != "darwin":
            return True
        try:
            import mujoco.viewer as _mjv  # type: ignore[import]
            return bool(getattr(_mjv, "_MJPYTHON", None))
        except Exception:
            return False

    @staticmethod
    def _reexec_under_mjpython_with_sim() -> None:
        """Re-exec the whole CLI under mjpython --sim for the pure-NL case.

        Called when the user asks for an arm sim from a plain `vector-cli`
        session (no --sim flag, so the startup re-exec guard did not fire)
        and a window was requested but is not available.

        Sets VECTOR_REEXEC=1 to prevent loops. If mjpython is missing, prints
        a warning and returns (falls through to headless launch).
        """
        import os as _os
        import sys as _sys

        if _os.environ.get("VECTOR_REEXEC") == "1":
            return  # already re-exec'd; do not loop

        mjpython: str | None = locate_mjpython()
        if not mjpython:
            print(
                "Warning: mjpython not found — arm sim will run headless "
                "(install mujoco into .venv-nano for a viewer window).",
                file=_sys.stderr,
            )
            return

        new_env = _os.environ.copy()
        new_env["VECTOR_REEXEC"] = "1"
        _os.execve(mjpython, [mjpython, "-m", "vector_os_nano.vcli.cli", "--sim"] + _sys.argv[1:], new_env)

    @staticmethod
    def _start_arm(gui: bool = True) -> Any:
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_arm import MuJoCoArm  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_gripper import MuJoCoGripper  # type: ignore[import]
        from vector_os_nano.hardware.sim.mujoco_perception import MuJoCoPerception  # type: ignore[import]
        from vector_os_nano.skills.pick import SIM_PICK_CONFIG
        arm = MuJoCoArm(gui=gui)
        arm.connect()
        gripper = MuJoCoGripper(arm)
        perception = MuJoCoPerception(arm)
        # Match the --sim flag path: full arm+gripper+perception, sim grasp offsets off.
        return Agent(
            arm=arm,
            gripper=gripper,
            perception=perception,
            config={"skills": {"pick": dict(SIM_PICK_CONFIG)}},
        )

    # ------------------------------------------------------------------
    # Pickable-object discovery: populate world_model from MJCF
    # ------------------------------------------------------------------

    @staticmethod
    def _populate_pickables_from_mjcf(world_model: Any, scene_xml_path: str) -> int:
        """Register every body whose name starts with 'pickable_' as an ObjectState.

        Loads the MJCF locally in the main process (independent of the
        bridge subprocess's MuJoCo instance). Uses the MJCF's DEFAULT
        body positions — i.e. what's written in the XML, not the post-
        physics-settled state. Sim-to-sim the drift is <1 cm, fine for
        grasp targeting since the skill has its own grasp-z offset.
        """
        import mujoco  # local import to avoid hard dep at module load
        from vector_os_nano.core.world_model import ObjectState

        model = mujoco.MjModel.from_xml_path(scene_xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        def _label(body_name: str) -> str:
            stem = body_name[len("pickable_"):]
            parts = stem.split("_")
            if len(parts) == 2:
                return f"{parts[1]} {parts[0]}"
            return stem.replace("_", " ")

        count = 0
        for bid in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
            if not name or not name.startswith("pickable_"):
                continue
            pos = data.body(bid).xpos
            world_model.add_object(ObjectState(
                object_id=name,
                label=_label(name),
                x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                confidence=1.0,
                state="on_table",
                properties={"source": "mjcf_scan"},
            ))
            count += 1
        return count

    @staticmethod
    def _start_go2(gui: bool = True, with_arm: bool = False) -> Any:
        import os
        import shutil
        import subprocess
        import atexit

        # Launch full stack as SEPARATE PROCESS (stable gait — no GIL contention)
        # This is the same architecture as run.py --sim-go2 --explore
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        # Use launch_explore.sh — all nodes (bridge + nav stack + TARE) must be
        # in ONE process group for reliable DDS communication. The nav flag
        # is NOT created here — dog stays still.
        # explore.py creates the flag to start movement.
        vnav_script = os.path.join(repo, "scripts", "launch_explore.sh")
        gui_flag = [] if gui else ["--no-gui"]

        # A generic ROS_DOMAIN_ID often leaks in from another sourced workspace.
        # Deliberately ignore it and install a simulation-scoped domain before
        # either the child stack or this process initialises rclpy.
        domain_id = _select_sim_ros_domain_id()
        canonical_terrain_path = _canonical_terrain_map_path()
        canonical_terrain_existed_at_start = os.path.isfile(
            canonical_terrain_path
        )
        session_dir, session_values, previous_environment = (
            _prepare_vnav_session_environment(domain_id)
        )

        # In shared-executor mode the runtime, not an individual proxy, owns the
        # default rclpy context.  Bind it to this session's domain before any
        # Node is constructed.  This is what makes SimStop -> SimStart on a new
        # random domain safe inside one long-lived vector-cli process.
        shared_runtime: Any = None
        if os.environ.get("VECTOR_SHARED_EXECUTOR", "1") == "1":
            try:
                from vector_os_nano.hardware.ros2.runtime import (
                    get_ros2_runtime,
                )

                shared_runtime = get_ros2_runtime()
                shared_runtime.prepare_for_domain(domain_id)
            except Exception:
                _restore_vnav_session_environment(
                    session_values, previous_environment
                )
                shutil.rmtree(session_dir, ignore_errors=True)
                raise

        def _release_prepared_runtime_if_idle() -> None:
            if shared_runtime is None:
                return
            try:
                shared_runtime.shutdown_if_idle()
            except Exception:
                logger.debug(
                    "[sim_tool] failed to release idle ROS2 runtime",
                    exc_info=True,
                )

        # Propagate mode and ownership to the sim subprocess via environment.
        # MuJoCoGo2._build_room_scene_xml reads VECTOR_SIM_WITH_ARM to pick
        # the scene (go2_piper vs bare go2).  The parent watcher makes abnormal
        # CLI exit terminate the complete launch session instead of orphaning it.
        child_env = os.environ.copy()
        child_env["VECTOR_SIM_WITH_ARM"] = "1" if with_arm else "0"
        child_env["VECTOR_VNAV_PARENT_PID"] = str(os.getpid())
        child_env["VECTOR_VNAV_MANAGED_SESSION"] = "1"

        log_path = session_values["VECTOR_VNAV_LOG_FILE"]
        try:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            log_fh = open(log_path, "w", buffering=1)
        except Exception:
            _release_prepared_runtime_if_idle()
            _restore_vnav_session_environment(
                session_values, previous_environment
            )
            shutil.rmtree(session_dir, ignore_errors=True)
            raise
        try:
            vnav_proc = subprocess.Popen(
                ["bash", vnav_script] + gui_flag,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=child_env,
            )
        except Exception:
            log_fh.close()
            _release_prepared_runtime_if_idle()
            _restore_vnav_session_environment(
                session_values, previous_environment
            )
            shutil.rmtree(session_dir, ignore_errors=True)
            raise

        lifecycle = _VNavSessionLifecycle(
            process=vnav_proc,
            log_fh=log_fh,
            session_dir=session_dir,
            session_values=session_values,
            previous_environment=previous_environment,
            canonical_terrain_path=canonical_terrain_path,
            canonical_terrain_existed_at_start=(
                canonical_terrain_existed_at_start
            ),
        )
        _stop_process = lifecycle.stop_process
        _finalize_session = lifecycle.finalize_session
        _cleanup = lifecycle.cleanup
        startup_proxies: list[Any] = []

        try:
            atexit.register(_cleanup)
        except BaseException:
            # The child already exists at this point, so even interpreter-level
            # registration failures must not leave its process group running.
            lifecycle.cleanup()
            raise

        def _forget_cleanup() -> None:
            try:
                atexit.unregister(_cleanup)
            except Exception:
                pass

        def _teardown_failed_start() -> None:
            """Tear down in dependency order while session paths remain active."""

            _forget_cleanup()
            try:
                _stop_process()
            except Exception:
                logger.debug(
                    "[sim_tool] failed to stop a rejected startup",
                    exc_info=True,
                )
            for proxy in reversed(startup_proxies):
                try:
                    proxy.disconnect()
                except Exception:
                    pass
            startup_proxies.clear()
            _release_prepared_runtime_if_idle()
            try:
                _finalize_session()
            except Exception:
                logger.warning(
                    "[sim_tool] failed to finalize rejected startup",
                    exc_info=True,
                )

        def _assert_stack_running(phase: str) -> None:
            error = _vnav_process_exit_error(
                vnav_proc, log_fh, log_path, phase=phase
            )
            if error is None:
                return
            raise error

        def _abort_startup(message: str, *, base_proxy: Any = None) -> None:
            try:
                log_fh.flush()
            except Exception:
                pass
            excerpt = _startup_log_excerpt(log_path)
            if base_proxy is not None and all(
                proxy is not base_proxy for proxy in startup_proxies
            ):
                startup_proxies.append(base_proxy)
            detail = f" Diagnostic: {excerpt}" if excerpt else ""
            raise RuntimeError(f"{message}.{detail} Full log: {log_path}")

        def _complete_managed_startup() -> Any:
            """Complete startup under one uninterrupted rollback boundary."""

            # The launch marker is emitted only after every critical child
            # remains alive.  It replaces the former fixed sleep, which could
            # return too early on a loaded GUI host and waste time on a fast
            # headless host.
            _wait_for_vnav_ready_marker(
                vnav_proc,
                log_fh,
                log_path,
                timeout_s=90.0,
            )
            _assert_stack_running("launcher readiness")

            # Connect via ROS2 proxy (same as run.py --explore).
            from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy

            base = Go2ROS2Proxy()
            startup_proxies.append(base)
            base.connect()
            _assert_stack_running("ROS2 proxy connection")

            # Stash subprocess handles on the base so SimStopTool can clean up
            # mid-session without waiting for atexit.
            base._sim_subprocess = vnav_proc  # type: ignore[attr-defined]
            base._sim_log_fh = log_fh  # type: ignore[attr-defined]
            base._sim_stop_process = _stop_process  # type: ignore[attr-defined]
            base._sim_finalize_session = (  # type: ignore[attr-defined]
                _finalize_session
            )
            base._sim_cleanup = _cleanup  # type: ignore[attr-defined]
            base._sim_unregister_cleanup = (  # type: ignore[attr-defined]
                _forget_cleanup
            )
            base._sim_log_path = log_path  # type: ignore[attr-defined]
            base._sim_session_dir = session_dir  # type: ignore[attr-defined]
            base._sim_ros_domain_id = domain_id  # type: ignore[attr-defined]
            base._sim_session_environment = (  # type: ignore[attr-defined]
                dict(session_values)
            )
            base._sim_environment_previous = (  # type: ignore[attr-defined]
                dict(previous_environment)
            )

            try:
                _require_go2_proxy_ready(base)
                _wait_for_go2_navigation_ready(
                    base,
                    timeout_s=35.0,
                    stable_s=0.75,
                    liveness_check=lambda: _assert_stack_running(
                        "navigation/FAR graph readiness"
                    ),
                )
                vgraph_diagnostics = base.far_vgraph_diagnostics()
                base._sim_startup_vgraph_nodes = int(  # type: ignore[attr-defined]
                    vgraph_diagnostics.get("node_count") or 0
                )
                odom_hz = _require_go2_odometry_performance(
                    base,
                    minimum_hz=5.0,
                    window_s=1.2,
                    liveness_check=lambda: _assert_stack_running(
                        "odometry performance readiness"
                    ),
                )
                base._sim_startup_odom_hz = odom_hz  # type: ignore[attr-defined]
            except (ConnectionError, TimeoutError, RuntimeError) as exc:
                _abort_startup(str(exc), base_proxy=base)

            # Everything after transport readiness remains inside this startup
            # transaction. A bad user config, Agent/skill import failure, scene
            # graph error, or Ctrl+C must release the child stack, ROS
            # nodes/context, and session environment immediately.
            return _finish_go2_startup(
                repo=repo,
                base=base,
                with_arm=with_arm,
                startup_proxies=startup_proxies,
                abort_startup=_abort_startup,
                assert_stack_running=_assert_stack_running,
            )

        # SimStartTool.execute converts ordinary exceptions into a ToolResult
        # while the CLI remains alive, so atexit alone is not a cleanup boundary.
        # This one guard covers every operation from launcher readiness through
        # the final Agent return, including KeyboardInterrupt/SystemExit.
        return _run_vnav_startup_stage(
            _complete_managed_startup,
            _teardown_failed_start,
        )

    @staticmethod
    def _start_isaac_go2() -> Any:
        """Connect to Go2 in Isaac Sim Docker (must already be running).

        Uses IsaacSimProxy over ROS2 topics — identical interface to
        Go2ROS2Proxy but assumes Isaac Sim is already live. No subprocess
        launch or sleep(20) needed.
        """
        import os
        from vector_os_nano.hardware.sim.isaac_sim_proxy import IsaacSimProxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.core.config import load_config

        proxy = IsaacSimProxy()
        proxy.connect()

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))
        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        agent = Agent(base=proxy, llm_api_key=api_key, config=cfg)

        # Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        # VLM perception (GPT-4o via OpenRouter)
        if api_key:
            try:
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
                agent._vlm = Go2VLMPerception(config={"api_key": api_key})
            except Exception:
                agent._vlm = None

        _attach_sim_scene_graph(agent, proxy, repo)

        return agent

    @staticmethod
    def _start_gazebo_go2() -> Any:
        """Start Go2 in Gazebo Harmonic via launch script + connect proxy.

        1. Launches Gazebo via scripts/launch_gazebo.sh (subprocess)
        2. Waits for /state_estimation topic (up to 60s)
        3. Connects GazeboGo2Proxy
        4. Builds Agent with skills + VLM + SceneGraph
        """
        import os
        import signal
        import subprocess
        import atexit
        import time as _time
        from vector_os_nano.hardware.sim.gazebo_go2_proxy import GazeboGo2Proxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]
        from vector_os_nano.core.config import load_config

        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))))

        # Launch Gazebo via script (handles sourcing quadruped_ros2_control)
        launch_script = os.path.join(repo, "scripts", "launch_gazebo.sh")
        log_fh = open("/tmp/vector_gazebo.log", "w")
        gz_proc = subprocess.Popen(
            ["bash", launch_script, "--world", "apartment"],
            stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        def _cleanup():
            try:
                os.killpg(os.getpgid(gz_proc.pid), signal.SIGTERM)
                gz_proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(gz_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            log_fh.close()

        atexit.register(_cleanup)

        # Wait for Gazebo to start (poll for /clock topic)
        import logging as _logging
        logger = _logging.getLogger(__name__)
        logger.info("[Gazebo] Waiting for Gazebo to start...")
        for i in range(60):
            if GazeboGo2Proxy.is_gazebo_running():
                logger.info("[Gazebo] Gazebo ready after %ds", i)
                break
            _time.sleep(1)
        else:
            raise ConnectionError(
                "Gazebo did not start within 60s. Check /tmp/vector_gazebo.log"
            )

        # Extra wait for controllers to activate
        _time.sleep(5)

        # Connect proxy
        proxy = GazeboGo2Proxy()
        proxy.connect()

        cfg_path = os.path.join(repo, "config", "user.yaml")
        cfg = load_config(cfg_path) if os.path.exists(cfg_path) else {}
        api_key = cfg.get("llm", {}).get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")

        agent = Agent(base=proxy, llm_api_key=api_key, config=cfg)

        # Go2 skills
        from vector_os_nano.skills.go2 import get_go2_skills  # type: ignore[import]
        for skill in get_go2_skills():
            agent._skill_registry.register(skill)

        # VLM perception
        if api_key:
            try:
                from vector_os_nano.perception.vlm_go2 import Go2VLMPerception
                agent._vlm = Go2VLMPerception(config={"api_key": api_key})
            except Exception:
                agent._vlm = None

        _attach_sim_scene_graph(agent, proxy, repo)

        return agent

    @staticmethod
    def _start_isaac_arm() -> Any:
        """Connect to a 6-DOF arm in Isaac Sim Docker (must already be running).

        Uses IsaacSimArmProxy over ROS2 topics. Isaac Sim must be running
        before calling this method.
        """
        from vector_os_nano.hardware.sim.isaac_sim_arm_proxy import IsaacSimArmProxy  # type: ignore[import]
        from vector_os_nano.core.agent import Agent  # type: ignore[import]

        arm = IsaacSimArmProxy()
        arm.connect()
        return Agent(arm=arm)

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Simulation startup")


@tool(
    name="stop_simulation",
    description=(
        "Stop a currently running robot simulation. Kills the MuJoCo + ROS2 "
        "bridge + nav stack subprocess, disconnects hardware, unregisters Go2 "
        "skills from the tool set. Call this when the user says '关闭仿真' / "
        "'stop sim' / 'shutdown simulation' etc."
    ),
    read_only=False,
    permission="ask",
)
class SimStopTool:
    """Stop the running simulation and clear the agent's hardware."""

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
    }

    def execute(self, params: dict[str, Any], context: ToolContext) -> ToolResult:
        app = context.app_state
        if app is None:
            return ToolResult(content="No app state available", is_error=True)

        agent = app.get("agent")
        if agent is None:
            return ToolResult(content="No simulation is running.")

        # Tear down hardware and subprocesses
        summary = SimStartTool._shutdown_agent(agent)

        # Unregister all go2/arm skill tools so the LLM stops offering them
        registry = app.get("registry")
        skills_dropped = 0
        if registry is not None and hasattr(registry, "list_tools"):
            for tool_name in list(registry.list_tools()):
                t = registry.get(tool_name)
                if t is not None and getattr(t, "_is_skill_wrapper", False):
                    try:
                        registry.unregister(tool_name)
                        skills_dropped += 1
                    except Exception:
                        pass

        # Clear app state references
        app["agent"] = None
        app["scene_graph"] = None
        app["skill_registry"] = None

        # Revert dev-world tool visibility (robot/diag hidden again)
        if registry is not None and hasattr(registry, "disable_category"):
            registry.disable_category("robot")
            registry.disable_category("diag")

        # Rebuild a live (empty) DynamicSystemPrompt so state cleanly reverts to dev
        engine = app.get("engine")
        if engine is not None:
            try:
                from vector_os_nano.vcli.prompt import build_system_prompt
                from vector_os_nano.vcli.dynamic_prompt import DynamicSystemPrompt
                from vector_os_nano.vcli.robot_context import RobotContextProvider
                provider = RobotContextProvider()
                app["robot_ctx_provider"] = provider
                static_blocks = build_system_prompt(
                    agent=None, cwd=context.cwd, robot_context=provider
                )
                engine._system_prompt = DynamicSystemPrompt(static_blocks, provider)
            except Exception as _exc:
                logger.warning("system-prompt rebuild after sim stop failed: %s", _exc)

        return ToolResult(
            content=f"Simulation stopped. {summary}. Dropped {skills_dropped} skill tools."
        )

    def check_permissions(
        self, params: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        return PermissionResult(behavior="allow", reason="Simulation shutdown")
