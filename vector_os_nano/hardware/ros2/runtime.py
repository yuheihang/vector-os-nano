# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""Process-singleton ROS2 runtime.

Owns ONE MultiThreadedExecutor and ONE spin thread for the entire
process lifetime.  Replaces the pattern where each ROS2 proxy calls
``rclpy.spin(node)`` in its own thread, which triggers the
``Executor is already spinning`` crash in rclpy's global default executor.

Usage::

    from vector_os_nano.hardware.ros2.runtime import get_ros2_runtime

    runtime = get_ros2_runtime()
    runtime.add_node(my_node)
    ...
    runtime.remove_node(my_node)
    # shutdown is registered with atexit automatically
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton state
# ---------------------------------------------------------------------------

_runtime: Ros2Runtime | None = None
_singleton_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Ros2Runtime
# ---------------------------------------------------------------------------


class Ros2Runtime:
    """Process-singleton holder for rclpy executor + nodes.

    Do not instantiate directly — use :func:`get_ros2_runtime`.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._executor: Any | None = None
        self._spin_thread: threading.Thread | None = None
        self._nodes: set[Any] = set()
        self._we_inited_rclpy: bool = False
        # Known for a context initialised by this runtime, or for a borrowed
        # Humble+ context whose public ``Context.get_domain_id()`` succeeds.
        # Ownership remains tracked separately by ``_we_inited_rclpy``.
        self._domain_id: int | None = None
        self._atexit_registered: bool = False
        self._is_running: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_initialized(self, expected_domain_id: int | None = None) -> int | None:
        """Ensure that a usable rclpy context exists before constructing Nodes.

        When this runtime owns an idle context and ``ROS_DOMAIN_ID`` changes,
        the old context is shut down and a new one is initialised on the
        configured domain.  A context created outside this runtime is borrowed
        only when its public ``Context.get_domain_id()`` agrees with the
        requested domain (or, for older rclpy compatibility, when no explicit
        domain was requested and the domain cannot be inspected).

        Args:
            expected_domain_id: Require this exact ROS domain.  This strict
                mode is intended for simulation-session startup.

        Returns:
            The known domain for an owned or inspectable borrowed context,
            otherwise ``None`` for a legacy external context whose domain
            cannot be inspected.
        """
        import rclpy  # noqa: PLC0415

        configured_domain = self._configured_domain_id()
        strict = expected_domain_id is not None
        requested_domain = (
            self._validate_domain_id(expected_domain_id)
            if strict
            else configured_domain
        )
        if strict and configured_domain != requested_domain:
            raise RuntimeError(
                "ROS_DOMAIN_ID does not match the requested ROS2 runtime "
                f"domain: environment={configured_domain}, "
                f"requested={requested_domain}"
            )

        with self._lock:
            self._ensure_initialized_locked(
                rclpy,
                requested_domain=requested_domain,
                strict=strict,
            )
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True
            return self._domain_id

    def prepare_for_domain(self, domain_id: int) -> int:
        """Prepare an idle, runtime-owned context for an exact ROS domain.

        A same-domain external context may be borrowed without taking shutdown
        ownership.  A mismatched or uninspectable external context is refused
        instead of silently binding Nodes to the wrong DDS domain.
        """
        requested = self._validate_domain_id(domain_id)
        actual = self.ensure_initialized(expected_domain_id=requested)
        if actual != requested:  # defensive; strict mode must know the domain
            raise RuntimeError(
                f"Unable to guarantee ROS2 domain {requested}; active domain is unknown"
            )
        return requested

    def add_node(self, node: Any) -> None:
        """Register *node* with the shared executor.

        Idempotent on the same node object.  Starts the singleton spin
        thread on the first call.  Thread-safe.
        """
        # Deferred import so ``import runtime`` works without rclpy installed.
        import rclpy  # noqa: PLC0415
        import rclpy.executors  # noqa: PLC0415

        with self._lock:
            # --- rclpy lifecycle ---
            self._ensure_initialized_locked(
                rclpy,
                requested_domain=self._configured_domain_id(),
                strict=False,
            )

            # --- executor ---
            if self._executor is None:
                self._executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
                logger.debug("MultiThreadedExecutor created (num_threads=4)")

            # --- spin thread ---
            if self._spin_thread is None:
                self._spin_thread = threading.Thread(
                    target=self._executor.spin,
                    daemon=True,
                    name="ros2-runtime-spin",
                )
                self._spin_thread.start()
                self._is_running = True
                logger.debug("Spin thread started")

            # --- atexit registration (once only) ---
            if not self._atexit_registered:
                atexit.register(self.shutdown)
                self._atexit_registered = True

            # --- register node ---
            if node not in self._nodes:
                self._executor.add_node(node)
                self._nodes.add(node)
                logger.debug("Node added: %s", node)

    @staticmethod
    def _validate_domain_id(value: int | None) -> int:
        """Return a valid DDS domain id or raise a concise configuration error."""
        if value is None or isinstance(value, bool):
            raise RuntimeError(f"Invalid ROS_DOMAIN_ID: {value!r}")
        try:
            domain_id = int(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid ROS_DOMAIN_ID: {value!r}") from exc
        if not 0 <= domain_id <= 232:
            raise RuntimeError(
                f"ROS_DOMAIN_ID must be in the inclusive range 0..232, got {domain_id}"
            )
        return domain_id

    @classmethod
    def _configured_domain_id(cls) -> int:
        raw = os.environ.get("ROS_DOMAIN_ID", "0").strip()
        try:
            return cls._validate_domain_id(int(raw))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid ROS_DOMAIN_ID: {raw!r}") from exc

    def _ensure_initialized_locked(
        self,
        rclpy: Any,
        *,
        requested_domain: int,
        strict: bool,
    ) -> None:
        """Context lifecycle implementation; caller must hold ``self._lock``."""
        context_ok = bool(rclpy.ok())

        if self._nodes:
            if not context_ok:
                raise RuntimeError(
                    "rclpy context stopped while ROS2 runtime nodes are still registered"
                )
            if self._domain_id is not None and self._domain_id != requested_domain:
                raise RuntimeError(
                    "Cannot change ROS_DOMAIN_ID while ROS2 runtime nodes are active: "
                    f"active={self._domain_id}, requested={requested_domain}"
                )
            if strict and self._domain_id is None:
                raise RuntimeError(
                    "Cannot guarantee the requested ROS_DOMAIN_ID: active rclpy "
                    "context domain cannot be inspected"
                )
            return

        if self._we_inited_rclpy:
            if context_ok and self._domain_id == requested_domain:
                return
            # No registered nodes remain, so it is safe to drain the executor
            # and replace a stopped or differently-configured owned context.
            self._shutdown_locked()
            context_ok = bool(rclpy.ok())

        if context_ok:
            external_domain = self._external_context_domain_id(rclpy)
            if external_domain is not None and external_domain != requested_domain:
                raise RuntimeError(
                    "Cannot change ROS_DOMAIN_ID of an externally initialised "
                    "rclpy context: "
                    f"active={external_domain}, requested={requested_domain}"
                )
            if strict and external_domain is None:
                raise RuntimeError(
                    "Cannot guarantee the requested ROS_DOMAIN_ID because the "
                    "active external rclpy context domain cannot be inspected"
                )
            # Compatibility for callers that intentionally own rclpy outside
            # this singleton (for example ROS integration tests).  Borrowing
            # never transfers shutdown ownership to this runtime.
            self._domain_id = external_domain
            return

        rclpy.init(domain_id=requested_domain)
        self._we_inited_rclpy = True
        self._domain_id = requested_domain
        logger.debug(
            "rclpy.init() called by Ros2Runtime (ROS_DOMAIN_ID=%d)",
            requested_domain,
        )

    @classmethod
    def _external_context_domain_id(cls, rclpy: Any) -> int | None:
        """Inspect an active default context without assuming ownership."""
        try:
            context = rclpy.get_default_context()
            raw_domain = context.get_domain_id()
        except (AttributeError, RuntimeError):
            return None
        # Avoid coercing MagicMock/arbitrary objects to misleading integers.
        if isinstance(raw_domain, bool) or not isinstance(raw_domain, int):
            return None
        try:
            return cls._validate_domain_id(raw_domain)
        except RuntimeError:
            return None

    def remove_node(self, node: Any) -> None:
        """Unregister *node* from the executor.

        Does NOT destroy the node — the caller owns it.  Thread-safe.
        """
        with self._lock:
            if self._executor is not None and node in self._nodes:
                self._executor.remove_node(node)
                self._nodes.discard(node)
                logger.debug("Node removed: %s", node)

    def shutdown(self) -> None:
        """Stop executor, join spin thread, call rclpy.shutdown if we initialised it.

        Called at process exit via atexit or by explicit teardown code.
        Thread-safe; safe to call multiple times (idempotent after first call).
        """
        with self._lock:
            self._shutdown_locked()

    def shutdown_if_idle(self) -> bool:
        """Drain the executor only when no registered nodes remain.

        Proxy disconnect removes its node before destroying it. Once the final
        proxy is gone, joining the spin thread here prevents an in-flight
        callback future from reporting ``Destroyable ... destruction was
        requested`` during native-CLI teardown. If another subsystem still
        owns a node, its executor is left untouched.
        """
        with self._lock:
            if self._nodes:
                return False
            self._shutdown_locked()
            return True

    def _shutdown_locked(self) -> None:
        """Shutdown implementation; caller must hold ``self._lock``."""
        executor = self._executor
        spin_thread = self._spin_thread

        if executor is not None:
            try:
                executor.shutdown()
                logger.debug("Executor shut down")
            except Exception:  # noqa: BLE001
                logger.warning("Exception during executor.shutdown()", exc_info=True)
            self._executor = None

        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
            self._spin_thread = None
            logger.debug("Spin thread joined")

        if self._we_inited_rclpy:
            try:
                import rclpy  # noqa: PLC0415

                rclpy.shutdown()
                logger.debug("rclpy.shutdown() called")
            except Exception:  # noqa: BLE001
                logger.warning("Exception during rclpy.shutdown()", exc_info=True)
            self._we_inited_rclpy = False
        self._domain_id = None

        self._is_running = False

    @property
    def is_running(self) -> bool:
        """True while the spin thread is active."""
        return self._is_running

    @property
    def domain_id(self) -> int | None:
        """Known domain of the owned or inspected borrowed context."""
        with self._lock:
            return self._domain_id


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_ros2_runtime() -> Ros2Runtime:
    """Return the process-singleton :class:`Ros2Runtime` (lazy, thread-safe)."""
    global _runtime  # noqa: PLW0603

    if _runtime is None:
        with _singleton_lock:
            if _runtime is None:
                _runtime = Ros2Runtime()
                logger.debug("Ros2Runtime singleton created")

    return _runtime
