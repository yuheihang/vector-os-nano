# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Product logging contract for the interactive Vector CLI.

Rich owns the terminal presentation. Python/SDK diagnostics go to a private
rotating file, so enabling ``--verbose`` must never dump HTTP state, prompts, or
project DEBUG records through a live spinner.
"""

from __future__ import annotations

import logging
import stat
from pathlib import Path

import pytest

from vector_os_nano.vcli import cli


_TOUCHED_LOGGERS = (
    *cli._QUIET_LOGGERS,
    *cli._TRANSPORT_LOGGERS,
    "vector_os_nano.vcli.engine",
)


@pytest.fixture(autouse=True)
def _restore_logging_state():
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_root_level = root.level
    saved_levels = {
        name: logging.getLogger(name).level for name in _TOUCHED_LOGGERS
    }
    saved_active_path = cli._ACTIVE_LOG_PATH
    try:
        yield
    finally:
        cli._remove_cli_handlers(root)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_root_level)
        for name, level in saved_levels.items():
            logging.getLogger(name).setLevel(level)
        cli._ACTIVE_LOG_PATH = saved_active_path


def _flush_cli_handlers() -> None:
    for handler in logging.getLogger().handlers:
        if getattr(handler, "_vector_cli_owned", False):
            handler.flush()


def _read(path: Path) -> str:
    _flush_cli_handlers()
    return path.read_text(encoding="utf-8")


def test_non_verbose_keeps_console_quiet_and_retains_info(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "vector-cli.log"
    assert cli._setup_logging(verbose=False, log_path=path) == path

    log = logging.getLogger("vector_os_nano.vcli.engine")
    log.debug("internal-debug")
    log.info("internal-info")
    log.warning("internal-warning")

    assert capsys.readouterr().err == ""
    text = _read(path)
    assert "internal-debug" not in text
    assert "internal-info" in text
    assert "internal-warning" in text


def test_verbose_writes_vector_debug_to_file_not_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "vector-cli.log"
    cli._setup_logging(verbose=True, log_path=path)

    logging.getLogger("vector_os_nano.vcli.engine").debug("vector-debug-detail")

    assert capsys.readouterr().err == ""
    assert "vector-debug-detail" in _read(path)


def test_transport_wire_debug_is_suppressed_even_when_verbose(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "vector-cli.log"
    cli._setup_logging(verbose=True, log_path=path)

    logging.getLogger("anthropic._base_client").debug(
        "Request options: secret prompt and tool schemas"
    )
    logging.getLogger("httpcore.http11").debug("raw response headers")
    logging.getLogger("OpenGL.platform").info("loaded optional accelerator")
    logging.getLogger("httpx").warning("request retry warning")

    assert capsys.readouterr().err == ""
    text = _read(path)
    assert "secret prompt" not in text
    assert "raw response headers" not in text
    assert "optional accelerator" not in text
    assert "request retry warning" in text


def test_only_errors_cross_into_interactive_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "vector-cli.log"
    cli._setup_logging(verbose=True, log_path=path)

    log = logging.getLogger("vector_os_nano.skills.navigate")
    log.warning("planner retry detail")
    log.error("navigation process unavailable")

    terminal = capsys.readouterr().err
    assert "planner retry detail" not in terminal
    assert "navigation process unavailable" in terminal
    text = _read(path)
    assert "planner retry detail" in text
    assert "navigation process unavailable" in text


def test_repeated_setup_is_idempotent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = tmp_path / "first.log"
    second = tmp_path / "second.log"
    cli._setup_logging(verbose=False, log_path=first)
    cli._setup_logging(verbose=True, log_path=second)

    owned = [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_vector_cli_owned", False)
    ]
    assert len(owned) == 2

    logging.getLogger("vector_os_nano.vcli.engine").info("one-record")
    assert capsys.readouterr().err == ""
    assert _read(second).count("one-record") == 1


def test_log_directory_and_file_are_private(tmp_path: Path) -> None:
    path = tmp_path / "private" / "vector-cli.log"
    cli._setup_logging(verbose=True, log_path=path)

    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700


def test_rollover_keeps_each_log_generation_private(tmp_path: Path) -> None:
    path = tmp_path / "private" / "vector-cli.log"
    cli._setup_logging(verbose=True, log_path=path)
    file_handler = next(
        handler
        for handler in logging.getLogger().handlers
        if isinstance(handler, cli._PrivateRotatingFileHandler)
    )

    logging.getLogger("vector_os_nano.vcli.engine").info("before-rollover")
    file_handler.flush()
    file_handler.doRollover()
    logging.getLogger("vector_os_nano.vcli.engine").info("after-rollover")
    file_handler.flush()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(f"{path}.1").stat().st_mode) == 0o600


def test_existing_custom_log_directory_mode_is_preserved(tmp_path: Path) -> None:
    shared = tmp_path / "existing"
    shared.mkdir(mode=0o750)
    shared.chmod(0o750)

    cli._setup_logging(verbose=True, log_path=shared / "vector-cli.log")

    assert stat.S_IMODE(shared.stat().st_mode) == 0o750


def test_unwritable_log_target_does_not_block_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A directory cannot be opened as a RotatingFileHandler target.
    assert cli._setup_logging(verbose=True, log_path=tmp_path) is None

    logging.getLogger("vector_os_nano.vcli.engine").error("still-running")
    assert "still-running" in capsys.readouterr().err


def test_verbose_help_describes_file_diagnostics() -> None:
    args = cli.parse_args(["--verbose"])
    assert args.verbose is True
