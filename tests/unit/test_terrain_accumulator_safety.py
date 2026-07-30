# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Regression tests for non-shrinking, validated terrain persistence."""

from pathlib import Path

import numpy as np


_BRIDGE = Path(__file__).resolve().parents[2] / "scripts" / "go2_vnav_bridge.py"


def _accumulator_type():
    source = _BRIDGE.read_text(encoding="utf-8")
    start = source.index("class TerrainAccumulator")
    end = source.index("\nclass Go2VNavBridge", start)
    namespace: dict[str, object] = {}
    exec(source[start:end], namespace)
    return namespace["TerrainAccumulator"]


def test_loaded_seed_remains_in_next_save(tmp_path) -> None:
    accumulator_type = _accumulator_type()
    terrain = tmp_path / "terrain.npz"
    np.savez_compressed(
        terrain,
        ix=np.array([1, 2], dtype=np.int32),
        iy=np.array([3, 4], dtype=np.int32),
        z=np.array([0.2, 0.4], dtype=np.float32),
        voxel_size=np.float32(0.1),
    )

    accumulator = accumulator_type()
    assert accumulator.load(str(terrain))
    accumulator.add([(0.5, 0.6, 0.7, 1.0)])
    assert accumulator.save(str(terrain))

    reloaded = accumulator_type()
    assert reloaded.load(str(terrain))
    assert reloaded.size == 3


def test_invalid_npz_is_rejected_without_destroying_live_grid(tmp_path) -> None:
    accumulator_type = _accumulator_type()
    accumulator = accumulator_type()
    accumulator.add([(1.0, 2.0, 0.4, 1.0)])

    malformed = tmp_path / "malformed.npz"
    np.savez_compressed(
        malformed,
        ix=np.array([1, 2]),
        iy=np.array([3]),
        z=np.array([0.1, 0.2]),
        voxel_size=np.float32(0.1),
    )

    assert not accumulator.load(str(malformed))
    assert accumulator.size == 1


def test_bridge_loads_seed_into_live_accumulator_not_replay_only() -> None:
    source = _BRIDGE.read_text(encoding="utf-8")

    assert "if self._terrain_acc.load(terrain_path):" in source
    assert "acc = TerrainAccumulator()" not in source
