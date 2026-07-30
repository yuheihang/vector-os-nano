# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""P1/P2 product acceptance through bare ``cli.main`` + natural language.

The network model alone is scripted.  Tool registration, prompt construction,
formal NavigateSkill execution, live SceneGraph resolution, deterministic
``in_room`` verification, actor causation, and the REPL verdict are production
paths.
"""
from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console


class _Base:
    _connected = True
    name = "p1-fake-mobile-base"

    def __init__(self) -> None:
        self.position = [0.0, 0.0, 0.3]
        self.motion = 0.0
        self.navigate_calls: list[tuple[float, float, dict[str, Any]]] = []
        self.walk_calls = 0
        self.stop_calls = 0
        self.goal_stats: dict[str, dict[str, Any]] = {}

    def get_position(self) -> list[float]:
        return list(self.position)

    def get_heading(self) -> float:
        return 0.0

    def cmd_motion(self) -> float:
        return self.motion

    def begin_navigation_goal(
        self, goal_id: str, target_xy: tuple[float, float]
    ) -> None:
        self.goal_stats[str(goal_id)] = {
            "goal_id": str(goal_id),
            "target_xy": list(target_xy),
            "nonzero_cmd_count": 0,
            "cmd_motion_count": 0.0,
            "moved_distance_m": 0.0,
            "actual_velocity_observed": False,
        }

    def navigate_to(self, x: float, y: float, **kwargs: Any) -> bool:
        tx, ty = float(x), float(y)
        start = tuple(self.position[:2])
        self.navigate_calls.append((tx, ty, dict(kwargs)))
        self.motion += 1.0
        self.position[:2] = [tx, ty]
        goal_id = str(kwargs.get("goal_id") or "")
        if goal_id:
            self.goal_stats[goal_id] = {
                "goal_id": goal_id,
                "nonzero_cmd_count": 1,
                "cmd_motion_count": 1.0,
                "moved_distance_m": math.dist(start, (tx, ty)),
                "actual_velocity_observed": True,
                "actor_caused": True,
            }
        return True

    def finalize_navigation_goal(self, goal_id: str, status: str) -> None:
        self.goal_stats.setdefault(str(goal_id), {})["status"] = str(status)

    def get_navigation_telemetry(self, goal_id: str) -> dict[str, Any]:
        return dict(self.goal_stats.get(str(goal_id), {}))

    def stop_navigation(self) -> None:
        self.stop_calls += 1

    def walk(self, *_args: Any, **_kwargs: Any) -> bool:
        self.walk_calls += 1
        self.motion += 1.0
        return True


def _agent() -> tuple[Any, _Base]:
    from vector_os_nano.core.agent import Agent
    from vector_os_nano.core.scene_graph import SceneGraph
    from vector_os_nano.skills.navigate import NavigateSkill

    base = _Base()
    graph = SceneGraph()
    repo = Path(__file__).resolve().parents[2]
    assert graph.load_layout(str(repo / "config" / "room_layout.yaml")) == 8
    agent = Agent(base=base, config={"world_mode": "known_layout"})
    agent._world_mode = "known_layout"
    agent._spatial_memory = graph
    base._scene_graph = graph
    agent._skill_registry.register(NavigateSkill())
    return agent, base


class _RecordingBackend:
    def __init__(self, turns: list[Any]) -> None:
        from tests.harness.fake_backend import FakeToolScriptBackend

        self._inner = FakeToolScriptBackend.from_tool_script(turns)
        self.requests: list[dict[str, Any]] = []

    def call(self, **kwargs: Any) -> Any:
        self.requests.append(dict(kwargs))
        return self._inner.call(**kwargs)


def _run_bare_repl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    prompt: str,
    turns: list[Any],
) -> tuple[_Base, _RecordingBackend, str]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-used")
    monkeypatch.setenv("VECTOR_REPL_NATIVE", "1")

    from vector_os_nano.vcli import cli
    from vector_os_nano.vcli import config as config_module
    from vector_os_nano.vcli import session as session_module

    agent, base = _agent()
    backend = _RecordingBackend(turns)
    rendered = io.StringIO()

    vector_dir = tmp_path / ".vector"
    monkeypatch.setattr(
        session_module, "DEFAULT_SESSION_DIR", vector_dir / "sessions"
    )
    monkeypatch.setattr(config_module, "_CONFIG_DIR", vector_dir)
    monkeypatch.setattr(config_module, "_CONFIG_PATH", vector_dir / "config.yaml")
    monkeypatch.setattr(
        config_module,
        "_CLAUDE_CREDS_PATH",
        tmp_path / ".claude" / ".credentials.json",
    )
    monkeypatch.setattr(cli, "_init_agent", lambda _args: agent)
    monkeypatch.setattr(
        cli,
        "create_backend_with_fake_seam",
        lambda **_kwargs: backend,
    )
    monkeypatch.setattr(
        cli,
        "console",
        Console(file=rendered, force_terminal=False, color_system=None),
    )
    monkeypatch.setattr(cli.time, "sleep", lambda *_args, **_kwargs: None)

    inputs = iter([prompt])

    class _PromptSession:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def prompt(self, *_args: Any, **_kwargs: Any) -> str:
            try:
                return next(inputs)
            except StopIteration as exc:
                raise EOFError from exc

    monkeypatch.setattr(cli, "PromptSession", _PromptSession)
    cli.main(["--no-permission"])
    return base, backend, rendered.getvalue()


@pytest.mark.cli_main
@pytest.mark.capability
def test_bare_cli_named_room_navigation_is_grounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.fake_backend import tool_turn

    base, backend, rendered = _run_bare_repl(
        monkeypatch,
        tmp_path,
        prompt="导航到 dining room",
        turns=[
            tool_turn(("navigate_room", {"room": "dining room"})),
            tool_turn(("verify", {"expr": "in_room('dining_room')"})),
            tool_turn(("finish", {}), end=True),
        ],
    )

    assert [call[:2] for call in base.navigate_calls] == [
        (3.0, 4.4),
        (3.0, 5.0),
        (3.0, 5.6),
        (4.8, 6.0),
    ]
    assert [call[2]["waypoint_kind"] for call in base.navigate_calls] == [
        "door_pre",
        "door_center",
        "door_post",
        "room_goal",
    ]
    assert all(
        call[2]["allow_door_fallback"] is False
        for call in base.navigate_calls
    )
    assert [call[2]["allow_reverse"] for call in base.navigate_calls] == [
        True,
        False,
        False,
        True,
    ]
    assert all(
        call[2]["speed_limit_mps"] is None
        for call in base.navigate_calls[:-1]
    )
    assert base.position[:2] == [4.8, 6.0]
    assert "navigate_room" in rendered
    assert "GROUNDED" in rendered
    assert "verified=True" in rendered

    first_request = backend.requests[0]
    tool_names = {tool["name"] for tool in first_request["tools"]}
    assert {"navigate_room", "navigate_xy", "verify", "finish"} <= tool_names
    assert "navigate" not in tool_names
    system_text = "\n".join(
        str(block.get("text", "")) for block in first_request["system"]
    )
    assert "dining_room" in system_text
    assert "3.0, 7.5" not in system_text
    assert "room_layout.yaml" not in system_text


@pytest.mark.cli_main
@pytest.mark.capability
def test_bare_cli_unknown_room_owns_turn_and_never_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.harness.fake_backend import tool_turn

    base, _backend, rendered = _run_bare_repl(
        monkeypatch,
        tmp_path,
        prompt="导航到一个不存在的 observatory",
        turns=[
            tool_turn(("navigate_room", {"room": "observatory"})),
            tool_turn(("navigate_xy", {"x": 3.0, "y": 7.5})),
            tool_turn(("walk", {"direction": "forward", "distance": 1.0})),
            tool_turn(("finish", {}), end=True),
        ],
    )

    assert base.navigate_calls == []
    assert base.walk_calls == 0
    assert base.position[:2] == [0.0, 0.0]
    assert base.stop_calls >= 1
    assert "navigate_room" in rendered
    assert "FAILED" in rendered
    assert "verified=False" in rendered
