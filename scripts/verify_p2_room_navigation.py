#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Product acceptance for known-layout room navigation.

The default dynamic scenario reproduces both reported failures without an LLM:

1. fresh hallway -> dining_room;
2. dining_room -> living_room through their direct physical door;
3. living_room -> dining_room through the same direct door.

It also verifies idle/late-join RViz clearing, subscribes to every
source-labelled path before navigation starts, and records non-empty
``/registered_scan`` input.  The report therefore distinguishes topology,
FAR, local-planner participation, execution and terminal lifecycle evidence.
Use ``--targets`` for an arbitrary room sequence, ``--all-rooms`` for the
minimum adjacent-leg sequence that covers every room and physical door, or
``--plan-only`` for a ROS-free all-pairs topology check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parent.parent
_LAYOUT = _REPO / "config" / "room_layout.yaml"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_PATH_TOPICS = (
    "/scene_graph/door_path",
    "/far/global_path",
    "/local_planner/path",
    "/nav/executed_path",
)
_REQUIRED_STACK_TOPICS = {
    "/state_estimation",
    "/way_point",
    "/path",
    "/viz_path_topic",
}
_REQUIRED_STACK_NODES = {
    "go2_vnav_bridge",
    "localPlanner",
    "far_planner",
    "tare_planner_node",
}
_ALL_ROOMS_TARGETS = (
    "dining_room",
    "master_bedroom",
    "dining_room",
    "living_room",
    "hallway",
    "kitchen",
    "study",
    "guest_bedroom",
    "study",
    "hallway",
    "bathroom",
)


def _static_contract() -> dict[str, Any]:
    from vector_os_nano.core.scene_graph import SceneGraph

    graph = SceneGraph()
    loaded = graph.load_layout(str(_LAYOUT))
    if loaded != 8 or not graph.has_executable_layout:
        raise RuntimeError("room_layout.yaml is not an executable 8-room layout")

    room_nodes = sorted(graph.get_all_rooms(), key=lambda room: room.room_id)
    room_ids = [room.room_id for room in room_nodes]
    door_ids = sorted(
        {edge.door_id for edge in graph.get_all_door_edges().values()}
    )
    failures: list[dict[str, Any]] = []
    covered_doors: set[str] = set()
    hop_counts: dict[int, int] = {}
    pair_count = 0
    for source in room_ids:
        for target in room_ids:
            if source == target:
                continue
            route = graph.plan_door_route(source, target)
            pair_count += 1
            if not route.success:
                failures.append(route.to_dict())
                continue
            covered_doors.update(route.door_ids)
            hops = len(route.door_ids)
            hop_counts[hops] = hop_counts.get(hops, 0) + 1
    expected_pairs = len(room_ids) * (len(room_ids) - 1)
    if pair_count != expected_pairs or failures:
        raise RuntimeError(
            f"not every directed room pair is routable: "
            f"{pair_count - len(failures)}/{expected_pairs}; failures={failures}"
        )
    missing_doors = sorted(set(door_ids) - covered_doors)
    if missing_doors:
        raise RuntimeError(
            f"all-pairs routes do not exercise physical doors {missing_doors}"
        )

    direct = graph.plan_door_route("living_room", "dining_room")
    hallway = graph.plan_door_route("hallway", "dining_room")
    if direct.room_path != ("living_room", "dining_room"):
        raise RuntimeError(f"living->dining unexpectedly routes as {direct.room_path}")
    if direct.door_ids != ("living_room-dining_room",):
        raise RuntimeError(f"wrong living->dining door chain: {direct.door_ids}")
    if hallway.room_path != ("hallway", "dining_room"):
        raise RuntimeError(f"hallway->dining unexpectedly routes as {hallway.room_path}")

    dining = graph.get_room("dining_room")
    if dining is None or dining.navigation_goal != (4.8, 6.0):
        raise RuntimeError("dining_room collision-free navigation_goal is missing")
    guest = graph.get_room("guest_bedroom")
    if guest is None or guest.navigation_goal != (15.0, 12.0):
        raise RuntimeError("guest_bedroom collision-free navigation_goal is missing")

    return {
        "rooms": len(room_ids),
        "room_ids": room_ids,
        "doors": len(door_ids),
        "door_ids": door_ids,
        "all_directed_pairs": {
            "expected": expected_pairs,
            "routable": pair_count - len(failures),
            "failures": failures,
            "hop_counts": {
                str(hops): count for hops, count in sorted(hop_counts.items())
            },
            "covered_doors": sorted(covered_doors),
            "missing_doors": missing_doors,
        },
        "living_to_dining": direct.to_dict(),
        "hallway_to_dining": hallway.to_dict(),
        "navigation_targets": {
            room.room_id: list(
                room.navigation_goal or (room.center_x, room.center_y)
            )
            for room in room_nodes
        },
    }


def _target_sequence(args: argparse.Namespace) -> list[str]:
    if args.targets:
        return list(args.targets)
    if args.all_rooms:
        return list(_ALL_ROOMS_TARGETS)
    targets = ["dining_room"]
    if args.roundtrip:
        targets.extend(("living_room", "dining_room"))
    return targets


def _coverage(
    initial_room: str,
    cases: list[dict[str, Any]],
    *,
    all_rooms: set[str],
    all_doors: set[str],
) -> dict[str, Any]:
    covered_rooms = {initial_room}
    covered_doors: set[str] = set()
    for case in cases:
        result = case.get("result", {})
        if isinstance(result, dict) and result.get("success"):
            target = case.get("target_room")
            if isinstance(target, str):
                covered_rooms.add(target)
            covered_doors.update(
                door_id
                for door_id in case.get("door_ids", [])
                if isinstance(door_id, str)
            )
    return {
        "rooms": {
            "covered": sorted(covered_rooms),
            "missing": sorted(all_rooms - covered_rooms),
            "count": len(covered_rooms),
            "total": len(all_rooms),
        },
        "doors": {
            "covered": sorted(covered_doors),
            "missing": sorted(all_doors - covered_doors),
            "count": len(covered_doors),
            "total": len(all_doors),
        },
    }


def _write_progress(artifacts: Path, payload: dict[str, Any]) -> None:
    """Persist completed cases atomically so an interrupted run stays useful."""

    destination = artifacts / "progress.json"
    temporary = artifacts / "progress.json.tmp"
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(destination)


class _PathProbe:
    """ROS node wrapper recording non-empty and terminal path states."""

    def __init__(
        self,
        node_name: str = "vector_p2_room_nav_acceptance",
    ) -> None:
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from nav_msgs.msg import Path as RosPath
        from sensor_msgs.msg import PointCloud2

        class _Node(Node):
            pass

        self.node = _Node(node_name)
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        scan_qos = QoSProfile(depth=5)
        scan_qos.reliability = ReliabilityPolicy.RELIABLE
        scan_qos.durability = DurabilityPolicy.VOLATILE
        self._lock = threading.Lock()
        self._generation = 0
        self._states: dict[str, dict[str, Any]] = {}
        self._scan_state: dict[str, Any] = {}
        self._subscriptions = []
        for topic in _PATH_TOPICS:
            self._subscriptions.append(
                self.node.create_subscription(
                    RosPath,
                    topic,
                    lambda msg, topic=topic: self._path_cb(topic, msg),
                    qos,
                )
            )
        self._subscriptions.append(
            self.node.create_subscription(
                PointCloud2,
                "/registered_scan",
                self._scan_cb,
                scan_qos,
            )
        )
        self.begin_case()

    def _path_cb(self, topic: str, msg: Any) -> None:
        count = len(msg.poses)
        frame_id = str(getattr(msg.header, "frame_id", ""))
        stamp = getattr(msg.header, "stamp", None)
        stamp_ns = (
            int(getattr(stamp, "sec", 0)) * 1_000_000_000
            + int(getattr(stamp, "nanosec", 0))
        )
        with self._lock:
            state = self._states[topic]
            state["message_count"] += 1
            state["last_pose_count"] = count
            state["max_pose_count"] = max(state["max_pose_count"], count)
            state["saw_nonempty"] = state["saw_nonempty"] or count > 0
            state["saw_empty"] = state["saw_empty"] or count == 0
            if count > 0:
                state["nonempty_message_count"] += 1
                state["nonempty_frames"].add(frame_id)
                state["saw_nonzero_stamp"] = (
                    state["saw_nonzero_stamp"] or stamp_ns > 0
                )

    def _scan_cb(self, msg: Any) -> None:
        point_count = int(msg.width) * int(msg.height)
        with self._lock:
            self._scan_state["message_count"] += 1
            self._scan_state["max_point_count"] = max(
                self._scan_state["max_point_count"],
                point_count,
            )
            self._scan_state["saw_nonempty"] = (
                self._scan_state["saw_nonempty"] or point_count > 0
            )

    def begin_case(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self._generation += 1
            self._states = {
                topic: {
                    "message_count": 0,
                    "last_pose_count": None,
                    "max_pose_count": 0,
                    "saw_nonempty": False,
                    "saw_empty": False,
                    "nonempty_message_count": 0,
                    "nonempty_frames": set(),
                    "saw_nonzero_stamp": False,
                }
                for topic in _PATH_TOPICS
            }
            self._scan_state = {
                "message_count": 0,
                "max_point_count": 0,
                "saw_nonempty": False,
            }

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                topic: {
                    **state,
                    "nonempty_frames": sorted(state["nonempty_frames"]),
                }
                for topic, state in self._states.items()
            }

    def scan_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._scan_state)

    def topic_names(self) -> set[str]:
        return {
            name
            for name, _types in self.node.get_topic_names_and_types()
        }

    def node_names(self) -> set[str]:
        return set(self.node.get_node_names())


def _wait_for_empty_paths(
    probe: _PathProbe,
    *,
    timeout_s: float,
    reject_prior_nonempty: bool = False,
    stable_s: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """Wait until every lifecycle-owned path remains terminal-empty."""

    deadline = time.monotonic() + timeout_s
    empty_since: float | None = None
    snapshot = probe.snapshot()
    while time.monotonic() < deadline:
        now = time.monotonic()
        snapshot = probe.snapshot()
        received_empty = all(
            state["message_count"] > 0 and state["last_pose_count"] == 0
            for state in snapshot.values()
        )
        prior_is_clean = (
            not reject_prior_nonempty
            or all(not state["saw_nonempty"] for state in snapshot.values())
        )
        if received_empty and prior_is_clean:
            if empty_since is None:
                empty_since = now
            if now - empty_since >= max(0.0, float(stable_s)):
                return snapshot
        else:
            empty_since = None
        time.sleep(0.1)
    raise TimeoutError(f"lifecycle-owned paths did not become empty: {snapshot}")


def _spawn_stack(log_path: Path, *, with_arm: bool) -> subprocess.Popen:
    env = os.environ.copy()
    env["VECTOR_SIM_WITH_ARM"] = "1" if with_arm else "0"
    # Acceptance artifacts must remain self-contained even when the activation
    # script exported a shared ROS_LOG_DIR for interactive development.
    env["ROS_LOG_DIR"] = str(log_path.parent / "ros")
    Path(env["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["bash", str(_REPO / "scripts" / "launch_explore.sh"), "--no-gui"],
        cwd=str(_REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._acceptance_log_fh = log_fh  # type: ignore[attr-defined]
    return proc


def _stop_stack(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=3.0)
        except ProcessLookupError:
            pass
    log_fh = getattr(proc, "_acceptance_log_fh", None)
    if log_fh is not None:
        log_fh.close()


def _wait_for_stack(
    probe: _PathProbe,
    proc: subprocess.Popen,
    log_path: Path,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    missing_topics = set(_REQUIRED_STACK_TOPICS)
    missing_nodes = set(_REQUIRED_STACK_NODES)
    launch_ready = False
    while time.monotonic() < deadline:
        status = proc.poll()
        if status is not None:
            raise RuntimeError(f"navigation stack exited during startup (status {status})")
        missing_topics = _REQUIRED_STACK_TOPICS - probe.topic_names()
        missing_nodes = _REQUIRED_STACK_NODES - probe.node_names()
        try:
            launch_ready = "Ready! Dog is standing still." in log_path.read_text(
                encoding="utf-8", errors="replace",
            )
        except OSError:
            launch_ready = False
        if launch_ready and not missing_topics and not missing_nodes:
            return
        time.sleep(0.5)
    raise TimeoutError(
        "navigation stack not ready: "
        f"launch_ready={launch_ready}, "
        f"missing_topics={sorted(missing_topics)}, "
        f"missing_nodes={sorted(missing_nodes)}"
    )


def _skill_result_dict(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "diagnosis_code": result.diagnosis_code,
        "error_message": result.error_message,
        "result_data": result.result_data,
    }


def _run_dynamic(args: argparse.Namespace, artifacts: Path) -> dict[str, Any]:
    try:
        import rclpy
    except ImportError as exc:
        raise RuntimeError(
            "rclpy is unavailable; source ROS 2 and the navigation workspace first"
        ) from exc

    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime
    from vector_os_nano.hardware.sim.go2_ros2_proxy import Go2ROS2Proxy
    from vector_os_nano.navigation.room_resolver import RoomResolver
    from vector_os_nano.skills.navigate import NavigateSkill

    stack: subprocess.Popen | None = None
    base: Go2ROS2Proxy | None = None
    probe: _PathProbe | None = None
    late_probe: _PathProbe | None = None
    case_late_probes: list[_PathProbe] = []
    runtime = get_ros2_runtime()
    manually_initialized_rclpy = False
    cases: list[dict[str, Any]] = []
    requested_targets = _target_sequence(args)
    canonical_targets: list[tuple[str, str]] = []
    initial_room = "unknown"
    all_room_ids: set[str] = set()
    all_door_ids: set[str] = set()
    try:
        domain_id = (
            args.ros_domain_id
            if args.ros_domain_id is not None
            else 100 + (os.getpid() % 100)
        )
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        print(f"[accept] starting headless stack (with_arm={args.with_arm})", flush=True)
        stack_log = artifacts / "sim.log"
        stack = _spawn_stack(stack_log, with_arm=args.with_arm)
        if not rclpy.ok():
            rclpy.init()
            manually_initialized_rclpy = True
        probe = _PathProbe()
        runtime.add_node(probe.node)
        _wait_for_stack(probe, stack, stack_log, args.startup_timeout)
        print("[accept] navigation topics ready; waiting for map settling", flush=True)
        time.sleep(args.settle_seconds)

        base = Go2ROS2Proxy()
        base.connect()
        graph = SceneGraph()
        graph.load_layout(str(_LAYOUT))
        resolver = RoomResolver(graph, world_mode="known_layout")
        all_room_ids = {
            room.room_id for room in graph.get_all_rooms()
        }
        all_door_ids = {
            edge.door_id for edge in graph.get_all_door_edges().values()
        }
        canonical_targets = [
            (requested, resolver.canonicalize(requested))
            for requested in requested_targets
        ]
        context = SkillContext(
            base=base,
            services={"spatial_memory": graph},
            config={"world_mode": "known_layout"},
        )

        idle_before = tuple(float(v) for v in base.get_position()[:2])
        startup_paths = _wait_for_empty_paths(
            probe,
            timeout_s=5.0,
            reject_prior_nonempty=True,
            stable_s=1.0,
        )
        startup_paths = probe.snapshot()
        if any(state["saw_nonempty"] for state in startup_paths.values()):
            raise RuntimeError(
                f"idle startup published a stale non-empty path: {startup_paths}"
            )
        if os.path.exists("/tmp/vector_nav_active"):
            raise RuntimeError("navigation gate is active before any goal was issued")
        idle_after = tuple(float(v) for v in base.get_position()[:2])
        idle_displacement = math.hypot(
            idle_after[0] - idle_before[0],
            idle_after[1] - idle_before[1],
        )
        if idle_displacement > 0.05:
            raise RuntimeError(
                "robot moved during startup idle window: "
                f"{idle_displacement:.3f}m"
            )

        # A second subscriber joins after every publisher has started.  With
        # TRANSIENT_LOCAL it must still receive the retained empty state.
        late_probe = _PathProbe("vector_p2_room_nav_late_join")
        runtime.add_node(late_probe.node)
        late_join_paths = _wait_for_empty_paths(
            late_probe,
            timeout_s=3.0,
            reject_prior_nonempty=True,
            stable_s=1.0,
        )
        runtime.remove_node(late_probe.node)

        start = tuple(float(v) for v in base.get_position()[:2])
        if not all(math.isfinite(v) for v in start):
            raise RuntimeError(f"invalid initial odometry: {start}")
        initial_room = resolver.locate(*start).canonical
        if initial_room != "hallway":
            raise RuntimeError(
                f"expected fresh simulation in hallway, got {initial_room!r} at {start}"
            )
        print(
            f"[accept] initial room={initial_room}, xy=({start[0]:.2f}, {start[1]:.2f})",
            flush=True,
        )

        for index, (requested_target, target) in enumerate(
            canonical_targets,
            start=1,
        ):
            before = tuple(float(v) for v in base.get_position()[:2])
            source_room = resolver.locate(*before).canonical
            if source_room is None:
                raise RuntimeError(
                    f"cannot locate source room at ({before[0]:.2f}, "
                    f"{before[1]:.2f})"
                )
            if source_room == target:
                raise RuntimeError(
                    f"target {requested_target!r} resolves to the current room "
                    f"{source_room!r}; cross-room acceptance requires a new room"
                )
            expected_route = graph.plan_door_route(source_room, target)
            if not expected_route.success:
                raise RuntimeError(
                    f"no expected topology route for {source_room}->{target}: "
                    f"{expected_route.diagnosis_code}: {expected_route.message}"
                )
            expected_path = list(expected_route.room_path)
            expected_doors = list(expected_route.door_ids)
            resolved_target = resolver.resolve(target)

            probe.begin_case()
            goal_id = f"P2-DYNAMIC-{index}"
            print(
                f"[accept] {goal_id}: {source_room} -> {target} "
                f"(expected doors={expected_doors})",
                flush=True,
            )
            started = time.monotonic()
            result = NavigateSkill().execute(
                {"room": target, "_goal_id": goal_id},
                context,
            )
            duration = time.monotonic() - started
            path_states = _wait_for_empty_paths(
                probe,
                timeout_s=7.0,
                stable_s=1.0,
            )
            # Join after a completed non-empty route.  Every publisher must
            # retain its terminal empty sample, never the previous blue path.
            case_late_probe = _PathProbe(
                f"vector_p2_room_nav_terminal_late_{index}"
            )
            case_late_probes.append(case_late_probe)
            runtime.add_node(case_late_probe.node)
            terminal_late_join_paths = _wait_for_empty_paths(
                case_late_probe,
                timeout_s=3.0,
                reject_prior_nonempty=True,
                stable_s=1.0,
            )
            runtime.remove_node(case_late_probe.node)
            after = tuple(float(v) for v in base.get_position()[:2])
            actual_room = resolver.locate(*after).canonical
            scan_state = probe.scan_snapshot()
            result_data = (
                result.result_data
                if isinstance(result.result_data, dict)
                else {}
            )
            route = result_data.get("route", {})
            if not isinstance(route, dict):
                route = {}
            room_path = route.get("room_path", [])
            door_ids = route.get("door_ids", [])
            route_ok = room_path == expected_path
            door_chain_ok = door_ids == expected_doors
            arrived_room_ok = actual_room == target
            all_path_layers_seen = all(
                state["saw_nonempty"] for state in path_states.values()
            )
            all_path_frames_valid = all(
                state["nonempty_frames"] == ["map"]
                and state["saw_nonzero_stamp"]
                for state in path_states.values()
            )
            local_planner_participated = (
                scan_state["saw_nonempty"]
                and path_states["/local_planner/path"]["saw_nonempty"]
            )
            terminal_clear = all(
                state["last_pose_count"] == 0
                for state in path_states.values()
            )
            case = {
                "goal_id": goal_id,
                "source_room": source_room,
                "requested_target": requested_target,
                "target_room": target,
                "target_navigation_xy": list(
                    resolved_target.navigation_target
                ),
                "before_xy": list(before),
                "after_xy": list(after),
                "actual_room": actual_room,
                "duration_s": round(duration, 3),
                "expected_room_path": expected_path,
                "expected_door_ids": expected_doors,
                "room_path": room_path,
                "door_ids": door_ids,
                "paths": path_states,
                "terminal_late_join_paths": terminal_late_join_paths,
                "registered_scan": scan_state,
                "result": _skill_result_dict(result),
                "checks": {
                    "route_exact": route_ok,
                    "door_chain_exact": door_chain_ok,
                    "arrived_in_target_room": arrived_room_ok,
                    "all_path_layers_seen": all_path_layers_seen,
                    "path_frames_and_stamps_valid": all_path_frames_valid,
                    "registered_scan_and_local_planner_seen": (
                        local_planner_participated
                    ),
                    "terminal_paths_cleared": terminal_clear,
                },
            }
            cases.append(case)
            coverage = _coverage(
                initial_room,
                cases,
                all_rooms=all_room_ids,
                all_doors=all_door_ids,
            )
            _write_progress(
                artifacts,
                {
                    "complete": False,
                    "requested_targets": requested_targets,
                    "canonical_targets": [
                        canonical for _requested, canonical in canonical_targets
                    ],
                    "coverage": coverage,
                    "cases": cases,
                },
            )
            print(
                f"[accept] {goal_id}: success={result.success}, "
                f"duration={duration:.1f}s, after=({after[0]:.2f}, "
                f"{after[1]:.2f}), room={actual_room}",
                flush=True,
            )

            if not result.success:
                raise RuntimeError(
                    f"{source_room}->{target} failed: "
                    f"{result.diagnosis_code}: {result.error_message}"
                )
            if not route_ok:
                raise RuntimeError(
                    f"{source_room}->{target} used unexpected route {room_path}; "
                    f"expected {expected_path}"
                )
            if not door_chain_ok:
                raise RuntimeError(
                    f"{source_room}->{target} used unexpected doors {door_ids}; "
                    f"expected {expected_doors}"
                )
            if not arrived_room_ok:
                raise RuntimeError(
                    f"{source_room}->{target} reported success but ended in "
                    f"{actual_room!r} at ({after[0]:.2f}, {after[1]:.2f})"
                )
            if not all_path_layers_seen:
                raise RuntimeError(
                    f"{source_room}->{target} did not publish all four path "
                    f"layers: {path_states}"
                )
            if not all_path_frames_valid:
                raise RuntimeError(
                    f"{source_room}->{target} published invalid path frames or "
                    f"timestamps: {path_states}"
                )
            if not local_planner_participated:
                raise RuntimeError(
                    f"{source_room}->{target} lacks registered-scan/localPlanner "
                    f"participation evidence: scan={scan_state}"
                )
            if not terminal_clear:
                raise RuntimeError(
                    f"{source_room}->{target} did not clear all RViz paths: "
                    f"{path_states}"
                )

        coverage = _coverage(
            initial_room,
            cases,
            all_rooms=all_room_ids,
            all_doors=all_door_ids,
        )
        dynamic_report = {
            "success": True,
            "with_arm": args.with_arm,
            "ros_domain_id": domain_id,
            "requested_targets": requested_targets,
            "canonical_targets": [
                canonical for _requested, canonical in canonical_targets
            ],
            "initial_xy": list(start),
            "initial_room": initial_room,
            "startup_idle": {
                "paths": startup_paths,
                "late_join_paths": late_join_paths,
                "displacement_m": round(idle_displacement, 4),
                "nav_gate_active": False,
            },
            "coverage": coverage,
            "cases": cases,
        }
        _write_progress(
            artifacts,
            {
                "complete": True,
                **dynamic_report,
            },
        )
        return dynamic_report
    except Exception as exc:
        failure_report = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "with_arm": args.with_arm,
            "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
            "requested_targets": requested_targets,
            "canonical_targets": [
                canonical for _requested, canonical in canonical_targets
            ],
            "coverage": _coverage(
                initial_room,
                cases,
                all_rooms=all_room_ids,
                all_doors=all_door_ids,
            ),
            "cases": cases,
        }
        _write_progress(
            artifacts,
            {
                "complete": True,
                **failure_report,
            },
        )
        return failure_report
    finally:
        # Unregister passive probes first.  The proxy is then the last shared
        # node and its disconnect drains the executor before any node handles
        # are destroyed, avoiding rclpy "Destroyable" teardown warnings.
        if probe is not None:
            runtime.remove_node(probe.node)
        if late_probe is not None:
            runtime.remove_node(late_probe.node)
        for case_late_probe in case_late_probes:
            runtime.remove_node(case_late_probe.node)
        if base is not None:
            try:
                base.disconnect()
            except Exception:
                pass
        if probe is not None:
            probe.node.destroy_node()
        if late_probe is not None:
            late_probe.node.destroy_node()
        for case_late_probe in case_late_probes:
            case_late_probe.node.destroy_node()
        runtime.shutdown_if_idle()
        if manually_initialized_rclpy and rclpy.ok():
            rclpy.shutdown()
        _stop_stack(stack)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="validate topology and navigation goals without ROS or MuJoCo",
    )
    scenario = parser.add_mutually_exclusive_group()
    scenario.add_argument(
        "--single",
        dest="roundtrip",
        action="store_false",
        help="only run fresh hallway -> dining_room",
    )
    scenario.add_argument(
        "--targets",
        nargs="+",
        metavar="ROOM",
        help=(
            "navigate through this ordered room/alias sequence; every leg is "
            "checked against the exact SceneGraph route"
        ),
    )
    scenario.add_argument(
        "--all-rooms",
        action="store_true",
        help=(
            "run the minimum adjacent-leg sequence covering all 8 rooms and "
            "all 9 physical doors"
        ),
    )
    parser.add_argument(
        "--without-arm",
        dest="with_arm",
        action="store_false",
        help="run the lighter bare-Go2 model (default uses the dog-arm model)",
    )
    parser.add_argument("--startup-timeout", type=float, default=60.0)
    parser.add_argument("--settle-seconds", type=float, default=8.0)
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=None,
        help="isolated ROS domain (default: a process-derived value from 100 to 199)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="output directory (default: timestamped artifacts/benchmarks path)",
    )
    parser.set_defaults(
        roundtrip=True,
        targets=None,
        all_rooms=False,
        with_arm=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    static = _static_contract()
    if args.plan_only:
        print(json.dumps({"success": True, "topology": static}, indent=2))
        return 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    artifacts = (
        args.artifacts_dir
        if args.artifacts_dir is not None
        else _REPO / "artifacts" / "benchmarks" / "afterp2" / "nav" / f"dynamic_{stamp}"
    )
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {
        "topology": static,
        "dynamic": _run_dynamic(args, artifacts),
        "artifacts_dir": str(artifacts),
    }
    (artifacts / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["dynamic"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
