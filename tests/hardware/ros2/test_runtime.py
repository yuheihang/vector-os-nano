# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Tests for Ros2Runtime process-singleton.

Strategy: inject a fake rclpy into sys.modules so tests run without
a live ROS2 installation.  Each test gets a fresh singleton by resetting
the module-level _runtime variable in teardown.
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a minimal fake rclpy tree
# ---------------------------------------------------------------------------


def _make_fake_rclpy() -> tuple[MagicMock, MagicMock]:
    """Return (fake_rclpy, fake_executor_class).

    fake_rclpy.ok()         -> True
    fake_rclpy.init()       -> None
    fake_rclpy.shutdown()   -> None
    fake_rclpy.executors.MultiThreadedExecutor(num_threads=4) -> MagicMock executor
    """
    fake_rclpy = MagicMock(name="rclpy")
    fake_rclpy.ok.return_value = True  # rclpy already initialised — default path
    fake_context = MagicMock(name="rclpy_default_context")
    fake_context.get_domain_id.return_value = 0
    fake_rclpy.get_default_context.return_value = fake_context

    def _init(*, domain_id=None, **_kwargs):
        fake_rclpy.ok.return_value = True
        if domain_id is not None:
            fake_context.get_domain_id.return_value = int(domain_id)

    def _shutdown(**_kwargs):
        fake_rclpy.ok.return_value = False

    fake_rclpy.init.side_effect = _init
    fake_rclpy.shutdown.side_effect = _shutdown

    fake_executors_mod = MagicMock(name="rclpy.executors")

    # Each call to MultiThreadedExecutor() returns a NEW mock executor instance
    # (we verify that only ONE instance is created across multiple add_node calls)
    executor_instance = MagicMock(name="executor_instance")
    # spin is a blocking method; the real executor.spin() loops forever.
    # We replace it with a no-op so the daemon thread exits quickly.
    executor_instance.spin = MagicMock(return_value=None)

    fake_executors_mod.MultiThreadedExecutor.return_value = executor_instance
    fake_rclpy.executors = fake_executors_mod

    return fake_rclpy, fake_executors_mod.MultiThreadedExecutor, executor_instance


# ---------------------------------------------------------------------------
# Fixture — isolate singleton + sys.modules per test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton(monkeypatch):
    """Reset the module-level _runtime to None between tests and inject fake rclpy."""
    fake_rclpy, fake_executor_cls, executor_instance = _make_fake_rclpy()
    monkeypatch.delenv("ROS_DOMAIN_ID", raising=False)

    # Inject before importing the module under test
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.executors", fake_rclpy.executors)

    # Import (or reload) the runtime module so it picks up the mock
    import importlib

    import vector_os_nano.hardware.ros2.runtime as rt_mod

    importlib.reload(rt_mod)

    # Store references on the fixture so tests can access them
    yield {
        "rt_mod": rt_mod,
        "fake_rclpy": fake_rclpy,
        "fake_executor_cls": fake_executor_cls,
        "executor_instance": executor_instance,
    }

    # Teardown: explicitly shut down so atexit doesn't fire against a stale
    # runtime with real rclpy imported after monkeypatch has been reverted.
    if rt_mod._runtime is not None:
        try:
            rt_mod._runtime.shutdown()
        except Exception:  # noqa: BLE001
            pass
    rt_mod._runtime = None


# ---------------------------------------------------------------------------
# Test 1 — singleton identity
# ---------------------------------------------------------------------------


def test_ros2_runtime_singleton_returns_same_instance(reset_singleton):
    """Two get_ros2_runtime() calls must return the exact same object."""
    rt_mod = reset_singleton["rt_mod"]

    a = rt_mod.get_ros2_runtime()
    b = rt_mod.get_ros2_runtime()

    assert a is b, "get_ros2_runtime() must return the same object on every call"


# ---------------------------------------------------------------------------
# Test 2 — rclpy.init called once across multiple add_node calls
# ---------------------------------------------------------------------------


def test_ros2_runtime_add_node_initialises_rclpy_once(reset_singleton):
    """First add_node must call rclpy.init() exactly once (when not already ok).
    Second add_node reuses the same executor instance."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_executor_cls = reset_singleton["fake_executor_cls"]
    executor_instance = reset_singleton["executor_instance"]

    # Simulate rclpy NOT yet initialised so init() gets triggered
    fake_rclpy.ok.return_value = False

    runtime = rt_mod.get_ros2_runtime()
    node1 = MagicMock(name="node1")
    node2 = MagicMock(name="node2")

    runtime.add_node(node1)
    runtime.add_node(node2)

    # rclpy.init() must be called exactly once
    assert fake_rclpy.init.call_count == 1, (
        f"Expected rclpy.init() called once, got {fake_rclpy.init.call_count}"
    )
    # MultiThreadedExecutor constructed exactly once
    assert fake_executor_cls.call_count == 1, (
        f"Expected one MultiThreadedExecutor, got {fake_executor_cls.call_count}"
    )
    # Both add_node calls target the SAME executor
    assert executor_instance.add_node.call_count == 2


# ---------------------------------------------------------------------------
# Test 3 — add then remove node
# ---------------------------------------------------------------------------


def test_ros2_runtime_add_then_remove_node_ok(reset_singleton):
    """add_node registers the node; remove_node unregisters it. No exceptions."""
    rt_mod = reset_singleton["rt_mod"]
    executor_instance = reset_singleton["executor_instance"]

    runtime = rt_mod.get_ros2_runtime()
    node = MagicMock(name="node_a")

    runtime.add_node(node)
    assert node in runtime._nodes, "Node should be in _nodes after add_node"

    runtime.remove_node(node)
    assert node not in runtime._nodes, "Node should not be in _nodes after remove_node"

    # Executor should have received both calls
    executor_instance.add_node.assert_called_once_with(node)
    executor_instance.remove_node.assert_called_once_with(node)


# ---------------------------------------------------------------------------
# Test 4 — three concurrent nodes on the SAME executor (Bug 1 regression guard)
# ---------------------------------------------------------------------------


def test_ros2_runtime_supports_three_concurrent_nodes(reset_singleton):
    """Adding 3 nodes must all land on a single executor instance.

    This is the regression guard for the 'Executor is already spinning'
    crash that occurred when each ROS2 proxy created its own spin thread.
    """
    rt_mod = reset_singleton["rt_mod"]
    fake_executor_cls = reset_singleton["fake_executor_cls"]
    executor_instance = reset_singleton["executor_instance"]

    runtime = rt_mod.get_ros2_runtime()

    nodes = [MagicMock(name=f"node_{i}") for i in range(3)]
    for n in nodes:
        runtime.add_node(n)  # must NOT raise

    # Still only ONE executor created
    assert fake_executor_cls.call_count == 1, (
        "Multiple executors created — this would cause 'Executor is already spinning'"
    )
    # All three nodes registered on that single executor
    assert executor_instance.add_node.call_count == 3
    for n in nodes:
        executor_instance.add_node.assert_any_call(n)


# ---------------------------------------------------------------------------
# Test 5 — shutdown joins thread and calls rclpy.shutdown when we inited
# ---------------------------------------------------------------------------


def test_ros2_runtime_shutdown_joins_thread(reset_singleton):
    """shutdown() must stop executor, join the spin thread, and call rclpy.shutdown
    iff _we_inited_rclpy is True."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    executor_instance = reset_singleton["executor_instance"]

    # Simulate rclpy not yet initialised so we own the lifecycle
    fake_rclpy.ok.return_value = False

    runtime = rt_mod.get_ros2_runtime()
    node = MagicMock(name="node_shutdown_test")

    # Patch threading.Thread so we can assert join() was called without
    # actually starting a real thread.
    mock_thread = MagicMock(name="spin_thread")
    mock_thread.is_alive.return_value = False

    with patch("threading.Thread", return_value=mock_thread):
        runtime.add_node(node)
        runtime.shutdown()

    # Executor was stopped
    executor_instance.shutdown.assert_called_once()
    # Thread was joined
    mock_thread.join.assert_called_once()
    # rclpy.shutdown was called because we inited it
    assert fake_rclpy.shutdown.call_count == 1, (
        "rclpy.shutdown() must be called when _we_inited_rclpy is True"
    )
    # is_running must be False after shutdown
    assert runtime.is_running is False


def test_shutdown_if_idle_preserves_other_nodes_then_drains(reset_singleton):
    rt_mod = reset_singleton["rt_mod"]
    executor_instance = reset_singleton["executor_instance"]
    runtime = rt_mod.get_ros2_runtime()
    node = MagicMock(name="live_node")

    runtime.add_node(node)
    assert runtime.shutdown_if_idle() is False
    executor_instance.shutdown.assert_not_called()

    runtime.remove_node(node)
    assert runtime.shutdown_if_idle() is True
    executor_instance.shutdown.assert_called_once()
    assert runtime.is_running is False


def test_prepare_for_domain_restarts_idle_owned_context(reset_singleton, monkeypatch):
    """A new session may replace an idle runtime-owned context and DDS domain."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_rclpy.ok.return_value = False
    runtime = rt_mod.get_ros2_runtime()

    monkeypatch.setenv("ROS_DOMAIN_ID", "41")
    assert runtime.prepare_for_domain(41) == 41
    assert runtime.domain_id == 41

    monkeypatch.setenv("ROS_DOMAIN_ID", "42")
    assert runtime.prepare_for_domain(42) == 42
    assert runtime.domain_id == 42

    assert fake_rclpy.init.call_args_list == [
        call(domain_id=41),
        call(domain_id=42),
    ]
    fake_rclpy.shutdown.assert_called_once()


def test_shutdown_if_idle_allows_stop_then_start_on_new_domain(
    reset_singleton,
    monkeypatch,
):
    """Final-node teardown releases the context before the next session."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_rclpy.ok.return_value = False
    runtime = rt_mod.get_ros2_runtime()
    node = MagicMock(name="session_node")

    monkeypatch.setenv("ROS_DOMAIN_ID", "51")
    runtime.prepare_for_domain(51)
    runtime.add_node(node)
    runtime.remove_node(node)
    assert runtime.shutdown_if_idle() is True
    assert fake_rclpy.ok() is False
    assert runtime.domain_id is None

    monkeypatch.setenv("ROS_DOMAIN_ID", "52")
    runtime.prepare_for_domain(52)
    assert runtime.domain_id == 52
    assert fake_rclpy.init.call_args_list == [
        call(domain_id=51),
        call(domain_id=52),
    ]


def test_prepare_for_domain_borrows_matching_external_context(
    reset_singleton,
    monkeypatch,
):
    """A verified same-domain external context is borrowed, never shut down."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_rclpy.ok.return_value = True
    fake_rclpy.get_default_context().get_domain_id.return_value = 61
    monkeypatch.setenv("ROS_DOMAIN_ID", "61")
    runtime = rt_mod.get_ros2_runtime()

    assert runtime.prepare_for_domain(61) == 61
    assert runtime.domain_id == 61
    fake_rclpy.init.assert_not_called()

    assert runtime.shutdown_if_idle() is True
    fake_rclpy.shutdown.assert_not_called()
    assert fake_rclpy.ok() is True


def test_prepare_for_domain_rejects_mismatched_external_context(
    reset_singleton,
    monkeypatch,
):
    """An external context on an old domain must not be silently reused."""
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_rclpy.ok.return_value = True
    fake_rclpy.get_default_context().get_domain_id.return_value = 71
    monkeypatch.setenv("ROS_DOMAIN_ID", "72")
    runtime = rt_mod.get_ros2_runtime()

    with pytest.raises(RuntimeError, match=r"externally initialised.*71.*72"):
        runtime.prepare_for_domain(72)

    fake_rclpy.init.assert_not_called()
    fake_rclpy.shutdown.assert_not_called()


def test_prepare_for_domain_rejects_switch_while_nodes_active(
    reset_singleton,
    monkeypatch,
):
    rt_mod = reset_singleton["rt_mod"]
    fake_rclpy = reset_singleton["fake_rclpy"]
    fake_rclpy.ok.return_value = False
    runtime = rt_mod.get_ros2_runtime()

    monkeypatch.setenv("ROS_DOMAIN_ID", "81")
    runtime.prepare_for_domain(81)
    runtime.add_node(MagicMock(name="active_node"))

    monkeypatch.setenv("ROS_DOMAIN_ID", "82")
    with pytest.raises(RuntimeError, match=r"nodes are active.*81.*82"):
        runtime.prepare_for_domain(82)
