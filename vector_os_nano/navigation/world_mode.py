# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Explicit world-mode selection for semantic navigation."""

from __future__ import annotations

import enum
import os
from typing import Any, Mapping

WORLD_MODE_ENV = "VECTOR_WORLD_MODE"


class WorldMode(str, enum.Enum):
    """How room semantics become available to navigation."""

    KNOWN_LAYOUT = "known_layout"
    UNKNOWN_EXPLORATION = "unknown_exploration"


def get_world_mode(
    value: str | WorldMode | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorldMode:
    """Return the configured world mode.

    The historical Go2 room simulation is a known-layout simulation, so an
    *unset* mode preserves that behaviour.  An explicitly invalid value fails
    loud instead of silently loading layout priors into an exploration run.
    """

    if isinstance(value, WorldMode):
        return value
    if value is None:
        env = os.environ if environ is None else environ
        value = env.get(WORLD_MODE_ENV, WorldMode.KNOWN_LAYOUT.value)
    normalized = str(value).strip().lower().replace("-", "_")
    try:
        return WorldMode(normalized)
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in WorldMode)
        raise ValueError(
            f"Invalid {WORLD_MODE_ENV}={value!r}. Expected one of: {valid}"
        ) from exc


def world_mode_for_agent(
    agent: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> WorldMode:
    """Resolve world mode from one live agent, then fall back to the environment.

    Native prompting, named-room execution, and ``in_room`` verification must see
    the same room visibility rules.  Keeping the agent lookup here avoids three
    subtly different copies of that policy across those surfaces.
    """

    if agent is not None:
        explicit = getattr(agent, "_world_mode", None)
        if explicit is not None:
            return get_world_mode(explicit, environ=environ)
        config = getattr(agent, "_config", None)
        if isinstance(config, Mapping) and config.get("world_mode") is not None:
            return get_world_mode(config["world_mode"], environ=environ)
    return get_world_mode(environ=environ)
