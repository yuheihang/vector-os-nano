# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Lazy context resources are capability-gated and attempted only once."""

from __future__ import annotations

from pathlib import Path

import pytest

from vector_os_nano.core.agent import Agent


class _PiperLikeArm:
    """Piper owns internal IK and deliberately has no set_ik_solver method."""

    def __init__(self) -> None:
        self.fk_calls = 0

    def get_joint_positions(self) -> list[float]:
        return [0.0] * 6

    def fk(
        self,
        joints: list[float],
    ) -> tuple[list[float], list[list[float]]]:
        assert len(joints) == 6
        self.fk_calls += 1
        return [0.41, -0.12, 0.73], [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]


class _ExternalIKArm(_PiperLikeArm):
    def set_ik_solver(self, _solver: object) -> None:
        pass


def test_piper_like_arm_never_constructs_so101_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vector_os_nano.hardware.so101.ik_solver as ik_module

    calls = 0

    def _unexpected_solver() -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("Piper must not construct the SO101 solver")

    monkeypatch.setattr(ik_module, "IKSolver", _unexpected_solver)
    agent = Agent(arm=_PiperLikeArm(), config={})
    agent._calibration_init_attempted = True

    agent._build_context()
    assert calls == 0


def test_robot_state_uses_piper_owned_fk() -> None:
    arm = _PiperLikeArm()

    agent = Agent(arm=arm, config={})

    assert arm.fk_calls == 1
    assert agent.world.get_robot().ee_position == pytest.approx(
        (0.41, -0.12, 0.73),
    )


def test_external_ik_construction_failure_is_not_retried_per_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vector_os_nano.hardware.so101.ik_solver as ik_module

    calls = 0

    def _failing_solver() -> object:
        nonlocal calls
        calls += 1
        raise RuntimeError("missing optional IK dependency")

    monkeypatch.setattr(ik_module, "IKSolver", _failing_solver)
    agent = Agent(arm=_ExternalIKArm(), config={})
    agent._calibration_init_attempted = True

    agent._build_context()
    agent._build_context()
    assert calls == 1


def test_calibration_load_failure_is_not_retried_per_action(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vector_os_nano.perception.calibration import Calibration

    calls = 0

    def _failing_load(_path: str) -> object:
        nonlocal calls
        calls += 1
        raise FileNotFoundError("missing calibration")

    monkeypatch.setattr(Calibration, "load", staticmethod(_failing_load))
    agent = Agent(
        config={"calibration": {"file": str(tmp_path / "missing.yaml")}},
    )

    agent._build_context()
    agent._build_context()
    assert calls == 1
