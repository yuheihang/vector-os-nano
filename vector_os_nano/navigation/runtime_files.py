# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Session-aware filesystem paths shared by the navigation runtime.

The CLI can start more than one isolated simulation over its lifetime and sets
these environment variables only when a session starts.  Resolve them on every
call rather than caching values at import time so modules imported before
``SimStartTool`` still consume the active session paths.
"""
from __future__ import annotations

import os

VECTOR_NAV_ACTIVE_FILE = "VECTOR_NAV_ACTIVE_FILE"
VECTOR_NAV_STALLED_FILE = "VECTOR_NAV_STALLED_FILE"
VECTOR_NAV_RESET_FILE = "VECTOR_NAV_RESET_FILE"
VECTOR_NAV_REPLAY_FILE = "VECTOR_NAV_REPLAY_FILE"
VECTOR_TERRAIN_MAP_FILE = "VECTOR_TERRAIN_MAP_FILE"
VECTOR_EXPLORE_FINISHED_FILE = "VECTOR_EXPLORE_FINISHED_FILE"

DEFAULT_NAV_ACTIVE_FILE = "/tmp/vector_nav_active"
DEFAULT_NAV_STALLED_FILE = "/tmp/vector_nav_stalled"
DEFAULT_NAV_RESET_FILE = "/tmp/vector_reset_pose"
DEFAULT_NAV_REPLAY_FILE = "/tmp/vector_terrain_replay"
DEFAULT_TERRAIN_MAP_FILE = "~/.vector_os_nano/terrain_map.npz"
DEFAULT_EXPLORE_FINISHED_FILE = "/tmp/vector_explore_finished"


def _runtime_file(environment_key: str, default: str) -> str:
    configured = os.environ.get(environment_key)
    selected = configured.strip() if configured and configured.strip() else default
    return os.path.abspath(os.path.expanduser(selected))


def nav_active_file() -> str:
    """Return the path that gates autonomous navigation."""

    return _runtime_file(VECTOR_NAV_ACTIVE_FILE, DEFAULT_NAV_ACTIVE_FILE)


def nav_stalled_file() -> str:
    """Return the bridge-to-proxy stall signal path."""

    return _runtime_file(VECTOR_NAV_STALLED_FILE, DEFAULT_NAV_STALLED_FILE)


def nav_reset_file() -> str:
    """Return the CLI-to-bridge reset request path."""

    return _runtime_file(VECTOR_NAV_RESET_FILE, DEFAULT_NAV_RESET_FILE)


def nav_replay_file() -> str:
    """Return the terrain replay request path."""

    return _runtime_file(VECTOR_NAV_REPLAY_FILE, DEFAULT_NAV_REPLAY_FILE)


def terrain_map_file() -> str:
    """Return the persistent/session terrain map path."""

    return _runtime_file(VECTOR_TERRAIN_MAP_FILE, DEFAULT_TERRAIN_MAP_FILE)


def explore_finished_file() -> str:
    """Return the bridge-to-skill exploration completion signal path."""

    return _runtime_file(
        VECTOR_EXPLORE_FINISHED_FILE,
        DEFAULT_EXPLORE_FINISHED_FILE,
    )


__all__ = [
    "DEFAULT_NAV_ACTIVE_FILE",
    "DEFAULT_EXPLORE_FINISHED_FILE",
    "DEFAULT_NAV_REPLAY_FILE",
    "DEFAULT_NAV_RESET_FILE",
    "DEFAULT_NAV_STALLED_FILE",
    "DEFAULT_TERRAIN_MAP_FILE",
    "VECTOR_NAV_ACTIVE_FILE",
    "VECTOR_EXPLORE_FINISHED_FILE",
    "VECTOR_NAV_REPLAY_FILE",
    "VECTOR_NAV_RESET_FILE",
    "VECTOR_NAV_STALLED_FILE",
    "VECTOR_TERRAIN_MAP_FILE",
    "nav_active_file",
    "explore_finished_file",
    "nav_replay_file",
    "nav_reset_file",
    "nav_stalled_file",
    "terrain_map_file",
]
