#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Deterministic product smoke for bare-CLI managed SimStart navigation.

This intentionally calls the same managed Go2 startup/teardown path used by
``start_simulation`` but bypasses the network LLM.  Each cycle starts on a
different ROS domain and issues the room command immediately after startup
returns, so it guards both first-goal readiness and stop -> start isolation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _result_dict(result: Any) -> dict[str, Any]:
    return {
        "success": bool(result.success),
        "diagnosis_code": str(result.diagnosis_code or ""),
        "error_message": str(result.error_message or ""),
        "result_data": result.result_data,
    }


def _run_cycle(
    *,
    cycle: int,
    target: str,
    with_arm: bool,
    domain_id: int,
    artifacts: Path,
) -> dict[str, Any]:
    from vector_os_nano.core.skill import SkillContext
    from vector_os_nano.navigation.room_resolver import RoomResolver
    from vector_os_nano.skills.navigate import NavigateSkill
    from vector_os_nano.vcli.tools.sim_tool import SimStartTool

    cycle_dir = artifacts / f"cycle_{cycle:02d}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VECTOR_SIM_ROS_DOMAIN_ID"] = str(domain_id)
    os.environ["VECTOR_VNAV_LOG_FILE"] = str(cycle_dir / "sim.log")
    os.environ["ROS_LOG_DIR"] = str(cycle_dir / "ros")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    agent: Any = None
    started = time.monotonic()
    try:
        agent = SimStartTool._start_go2(gui=False, with_arm=with_arm)
        startup_duration = time.monotonic() - started
        base = agent._base
        graph = agent._spatial_memory
        resolver = RoomResolver(graph, world_mode="known_layout")
        before = tuple(float(value) for value in base.get_position()[:2])
        source_room = resolver.locate(*before).canonical
        startup_graph = base.far_vgraph_diagnostics()

        context = SkillContext(
            base=base,
            services={"spatial_memory": graph},
            config={"world_mode": "known_layout"},
        )
        navigation_started = time.monotonic()
        result = NavigateSkill().execute(
            {
                "room": target,
                "_goal_id": f"MANAGED-FRESH-{cycle}",
            },
            context,
        )
        navigation_duration = time.monotonic() - navigation_started
        after = tuple(float(value) for value in base.get_position()[:2])
        actual_room = resolver.locate(*after).canonical
        payload = {
            "cycle": cycle,
            "domain_id": domain_id,
            "with_arm": with_arm,
            "startup_duration_s": round(startup_duration, 3),
            "startup_odom_hz": float(base._sim_startup_odom_hz),
            "startup_far_vgraph": startup_graph,
            "source_room": source_room,
            "target_room": target,
            "before_xy": list(before),
            "after_xy": list(after),
            "actual_room": actual_room,
            "navigation_duration_s": round(navigation_duration, 3),
            "result": _result_dict(result),
        }
        payload["success"] = bool(result.success) and actual_room == target
        return payload
    except Exception as exc:
        return {
            "cycle": cycle,
            "domain_id": domain_id,
            "with_arm": with_arm,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if agent is not None:
            shutdown = SimStartTool._shutdown_agent(agent)
            (cycle_dir / "shutdown.txt").write_text(
                shutdown + "\n",
                encoding="utf-8",
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="living_room")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--domain-start", type=int, default=221)
    parser.add_argument(
        "--without-arm",
        dest="with_arm",
        action="store_false",
        help="use bare Go2 instead of the default Go2 + Piper embodiment",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=_REPO / "artifacts" / "benchmarks" / "fresh_start_fix",
    )
    parser.set_defaults(with_arm=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cycles < 1:
        raise SystemExit("--cycles must be at least 1")
    if args.domain_start < 0 or args.domain_start + args.cycles - 1 > 232:
        raise SystemExit("requested ROS domains must stay within 0..232")

    artifacts = args.artifacts_dir.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    report = {
        "target": args.target,
        "cycles": [],
    }
    for index in range(args.cycles):
        domain_id = args.domain_start + index
        print(
            f"[managed-smoke] cycle={index + 1} domain={domain_id} "
            f"target={args.target}",
            flush=True,
        )
        cycle = _run_cycle(
            cycle=index + 1,
            target=args.target,
            with_arm=args.with_arm,
            domain_id=domain_id,
            artifacts=artifacts,
        )
        report["cycles"].append(cycle)
        print(
            f"[managed-smoke] cycle={index + 1} success={cycle['success']}",
            flush=True,
        )
        if not cycle["success"]:
            break

    report["success"] = (
        len(report["cycles"]) == args.cycles
        and all(cycle["success"] for cycle in report["cycles"])
    )
    report_path = artifacts / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[managed-smoke] report={report_path}", flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
