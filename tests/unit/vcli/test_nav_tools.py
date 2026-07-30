# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for NavStateTool and TerrainStatusTool.

TDD RED phase — all tests must fail before implementation is written.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vector_os_nano.vcli.tools.base import ToolContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_context() -> MagicMock:
    ctx = MagicMock(spec=ToolContext)
    ctx.app_state = {}
    return ctx


@pytest.fixture()
def nav_tool():
    from vector_os_nano.vcli.tools.nav_tools import NavStateTool
    return NavStateTool()


@pytest.fixture()
def terrain_tool():
    from vector_os_nano.vcli.tools.nav_tools import TerrainStatusTool
    return TerrainStatusTool()


# ---------------------------------------------------------------------------
# NavStateTool tests
# ---------------------------------------------------------------------------


class TestNavStateTool:

    @patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=["kitchen", "hallway"])
    def test_nav_state_all_fields(self, mock_rooms, mock_nav, mock_explore, nav_tool):
        """Output must include all required diagnostic fields."""
        ctx = _make_context()
        result = nav_tool.execute({}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert "exploring" in data
        assert "nav_stack_running" in data
        assert "nav_flag_active" in data
        assert "explored_rooms" in data
        assert "tare_running" in data

    @patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=True)
    @patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=["kitchen", "hallway"])
    def test_nav_state_values(self, mock_rooms, mock_nav, mock_explore, nav_tool):
        """Mocked values must flow through to the output JSON."""
        ctx = _make_context()
        result = nav_tool.execute({}, ctx)
        data = json.loads(result.content)
        assert data["exploring"] is True
        assert data["nav_stack_running"] is True
        assert data["explored_rooms"] == ["kitchen", "hallway"]

    def test_nav_state_no_explore_module(self, nav_tool):
        """Tool must succeed gracefully even when explore module is unavailable.

        When _is_exploring / _is_nav_stack_running / _get_explored_rooms raise
        ImportError internally they return False / []. The tool itself must never
        raise and must return a valid JSON result.
        """
        ctx = _make_context()
        # Patch helpers to simulate ImportError behaviour (returns defaults)
        with patch("vector_os_nano.vcli.tools.nav_tools._is_exploring", return_value=False), \
             patch("vector_os_nano.vcli.tools.nav_tools._is_nav_stack_running", return_value=False), \
             patch("vector_os_nano.vcli.tools.nav_tools._get_explored_rooms", return_value=[]):
            result = nav_tool.execute({}, ctx)
        assert not result.is_error
        data = json.loads(result.content)
        assert data["exploring"] is False
        assert data["explored_rooms"] == []

    def test_nav_tool_is_read_only(self, nav_tool):
        assert nav_tool.is_read_only({}) is True

    def test_nav_tool_is_concurrency_safe(self, nav_tool):
        assert nav_tool.is_concurrency_safe({}) is True

    def test_nav_state_reads_active_file_override_at_execute_time(
        self, nav_tool, tmp_path, monkeypatch
    ):
        active_file = tmp_path / "session" / "nav_active"
        active_file.parent.mkdir()
        active_file.write_text("1", encoding="utf-8")
        monkeypatch.setenv("VECTOR_NAV_ACTIVE_FILE", str(active_file))

        result = nav_tool.execute({}, _make_context())

        assert json.loads(result.content)["nav_flag_active"] is True


# ---------------------------------------------------------------------------
# TerrainStatusTool tests
# ---------------------------------------------------------------------------


class TestTerrainStatusTool:

    def test_terrain_status_file_exists(self, terrain_tool, tmp_path, monkeypatch):
        """When terrain file exists, file_exists=True and size is reported."""
        npz_file = tmp_path / "terrain_map.npz"
        ix = np.array([1, 2, 3])
        np.savez(str(npz_file), ix=ix)

        ctx = _make_context()
        monkeypatch.setenv("VECTOR_TERRAIN_MAP_FILE", str(npz_file))
        result = terrain_tool.execute({}, ctx)

        assert not result.is_error
        data = json.loads(result.content)
        assert data["file_exists"] is True
        assert data["file_size_kb"] > 0
        assert data["voxel_count"] == 3

    def test_terrain_status_file_missing(self, terrain_tool, tmp_path, monkeypatch):
        """When terrain file is absent, file_exists=False with graceful output."""
        missing = tmp_path / "no_terrain_here.npz"
        ctx = _make_context()
        monkeypatch.setenv("VECTOR_TERRAIN_MAP_FILE", str(missing))
        result = terrain_tool.execute({}, ctx)

        assert not result.is_error
        data = json.loads(result.content)
        assert data["file_exists"] is False
        assert data["file_size_kb"] == 0
        assert data["voxel_count"] == 0

    def test_terrain_status_fields(self, terrain_tool, tmp_path, monkeypatch):
        """All expected fields must be present in the output."""
        npz_file = tmp_path / "terrain_map.npz"
        np.savez(str(npz_file), ix=np.array([10, 20]))

        ctx = _make_context()
        replay_file = tmp_path / "nav_replay"
        replay_file.write_text("1", encoding="utf-8")
        monkeypatch.setenv("VECTOR_TERRAIN_MAP_FILE", str(npz_file))
        monkeypatch.setenv("VECTOR_NAV_REPLAY_FILE", str(replay_file))
        result = terrain_tool.execute({}, ctx)

        data = json.loads(result.content)
        for key in ("file_exists", "file_path", "file_size_kb", "replay_triggered", "voxel_count"):
            assert key in data, f"Missing field: {key}"
        assert data["file_path"] == str(npz_file)
        assert data["replay_triggered"] is True

    def test_terrain_tool_is_read_only(self, terrain_tool):
        assert terrain_tool.is_read_only({}) is True

    def test_terrain_tool_is_concurrency_safe(self, terrain_tool):
        assert terrain_tool.is_concurrency_safe({}) is True
