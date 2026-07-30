# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Session-path contracts for navigation runtime files."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vector_os_nano.navigation.runtime_files import (
    DEFAULT_EXPLORE_FINISHED_FILE,
    DEFAULT_NAV_ACTIVE_FILE,
    DEFAULT_NAV_REPLAY_FILE,
    DEFAULT_NAV_RESET_FILE,
    DEFAULT_NAV_STALLED_FILE,
    DEFAULT_TERRAIN_MAP_FILE,
    explore_finished_file,
    nav_active_file,
    nav_replay_file,
    nav_reset_file,
    nav_stalled_file,
    terrain_map_file,
)


@pytest.mark.parametrize(
    ("environment_key", "resolver", "default"),
    [
        ("VECTOR_NAV_ACTIVE_FILE", nav_active_file, DEFAULT_NAV_ACTIVE_FILE),
        ("VECTOR_NAV_STALLED_FILE", nav_stalled_file, DEFAULT_NAV_STALLED_FILE),
        ("VECTOR_NAV_RESET_FILE", nav_reset_file, DEFAULT_NAV_RESET_FILE),
        ("VECTOR_NAV_REPLAY_FILE", nav_replay_file, DEFAULT_NAV_REPLAY_FILE),
        ("VECTOR_TERRAIN_MAP_FILE", terrain_map_file, DEFAULT_TERRAIN_MAP_FILE),
        (
            "VECTOR_EXPLORE_FINISHED_FILE",
            explore_finished_file,
            DEFAULT_EXPLORE_FINISHED_FILE,
        ),
    ],
)
def test_runtime_file_defaults_remain_legacy_compatible(
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str,
    resolver,
    default: str,
) -> None:
    monkeypatch.delenv(environment_key, raising=False)
    assert resolver() == os.path.abspath(os.path.expanduser(default))


@pytest.mark.parametrize(
    ("environment_key", "resolver"),
    [
        ("VECTOR_NAV_ACTIVE_FILE", nav_active_file),
        ("VECTOR_NAV_STALLED_FILE", nav_stalled_file),
        ("VECTOR_NAV_RESET_FILE", nav_reset_file),
        ("VECTOR_NAV_REPLAY_FILE", nav_replay_file),
        ("VECTOR_TERRAIN_MAP_FILE", terrain_map_file),
        ("VECTOR_EXPLORE_FINISHED_FILE", explore_finished_file),
    ],
)
def test_runtime_files_resolve_environment_after_module_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    environment_key: str,
    resolver,
) -> None:
    first = tmp_path / "session-a" / environment_key.lower()
    second = tmp_path / "session-b" / environment_key.lower()

    monkeypatch.setenv(environment_key, str(first))
    assert resolver() == str(first)
    monkeypatch.setenv(environment_key, str(second))
    assert resolver() == str(second)


def test_blank_runtime_file_override_uses_legacy_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VECTOR_NAV_ACTIVE_FILE", "   ")
    assert nav_active_file() == DEFAULT_NAV_ACTIVE_FILE


def test_navigation_production_code_uses_runtime_file_resolvers() -> None:
    root = Path(__file__).resolve().parents[2]
    sources = {
        "proxy": root
        / "vector_os_nano"
        / "hardware"
        / "sim"
        / "go2_ros2_proxy.py",
        "navigate": root / "vector_os_nano" / "skills" / "navigate.py",
        "explore": root / "vector_os_nano" / "skills" / "go2" / "explore.py",
        "bridge": root / "scripts" / "go2_vnav_bridge.py",
        "cli": root / "vector_os_nano" / "vcli" / "cli.py",
        "nav_tools": root / "vector_os_nano" / "vcli" / "tools" / "nav_tools.py",
    }
    text = {name: path.read_text(encoding="utf-8") for name, path in sources.items()}

    assert "nav_active_file" in text["proxy"]
    assert "nav_stalled_file" in text["proxy"]
    assert "nav_active_file" in text["navigate"]
    assert "nav_active_file" in text["explore"]
    assert "nav_replay_file" in text["explore"]
    assert "explore_finished_file" in text["explore"]
    for resolver in (
        "nav_active_file",
        "nav_stalled_file",
        "nav_reset_file",
        "nav_replay_file",
        "terrain_map_file",
        "explore_finished_file",
    ):
        assert resolver in text["bridge"]
    assert "terrain_map_file" in text["cli"]
    assert "nav_reset_file" in text["cli"]
    assert "nav_active_file" in text["nav_tools"]
    assert "nav_replay_file" in text["nav_tools"]
    assert "terrain_map_file" in text["nav_tools"]

    combined = "\n".join(text.values())
    for legacy_literal in (
        "/tmp/vector_nav_active",
        "/tmp/vector_nav_stalled",
        "/tmp/vector_reset_pose",
        "/tmp/vector_terrain_replay",
        "/tmp/vector_explore_finished",
        "~/.vector_os_nano/terrain_map.npz",
    ):
        assert legacy_literal not in combined


def test_cli_reset_writes_the_session_reset_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vector_os_nano.vcli import cli

    reset_file = tmp_path / "session" / "nav_reset"
    reset_file.parent.mkdir()
    monkeypatch.setenv("VECTOR_NAV_RESET_FILE", str(reset_file))

    assert cli._handle_slash_command("reset", [], MagicMock(), None, {}) is True
    assert reset_file.read_text(encoding="utf-8") == "1"


def test_cli_clear_memory_removes_session_terrain_and_canonical_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from vector_os_nano.vcli import cli

    home = tmp_path / "home"
    session_terrain = tmp_path / "session" / "terrain_map.npz"
    canonical_terrain = home / ".vector_os_nano" / "terrain_map.npz"
    session_terrain.parent.mkdir()
    canonical_terrain.parent.mkdir(parents=True)
    session_terrain.write_bytes(b"session")
    canonical_terrain.write_bytes(b"canonical")
    original_expanduser = os.path.expanduser
    monkeypatch.setattr(
        os.path,
        "expanduser",
        lambda value: (
            str(home / value[2:])
            if isinstance(value, str) and value.startswith("~/")
            else original_expanduser(value)
        ),
    )
    monkeypatch.setenv("VECTOR_TERRAIN_MAP_FILE", str(session_terrain))

    assert (
        cli._handle_slash_command("clear_memory", [], MagicMock(), None, {})
        is True
    )
    assert not session_terrain.exists()
    assert not canonical_terrain.exists()
