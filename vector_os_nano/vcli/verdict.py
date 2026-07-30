# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics

"""verdict — the machine-checkable acceptance signal for a single cli.main turn.

This is the acceptance INSTRUMENT for the orchestration redesign. The honest
verdict the engine already computes for a VGG run
(``evidence_passed(trace, verify_oracle_names(agent, engine))``) previously NEVER
escaped cli.main as a machine signal — the REPL only ``console.print``ed Rich
prose. ``VerdictReport`` turns that exact same computation into a frozen,
JSON-serializable record so a non-interactive ``-p/--json`` turn can emit ONE
stdout line a harness asserts against.

Honesty by construction (CLAUDE.md rule 5 — verify is the moat):
``VerdictReport.from_trace`` re-uses the EXISTING ``classify_step_evidence`` /
``evidence_passed`` from ``trace_store`` for EVERY field — it NEVER re-derives a
verdict with its own logic. The contract test pins this:

    VerdictReport.from_trace(trace, oracle_names).verified
        == evidence_passed(trace, oracle_names)

so the machine signal can only ever AGREE with the gate the engine itself uses.
An empty oracle set, a sentinel ``""``/``"True"`` verify, an absent oracle, a
tautology, or a VLM visual override all classify RAN (not GROUNDED), so the
signal fails CLOSED — the moat only ever gets stricter.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from vector_os_nano.vcli.cognitive.trace_store import (
    classify_step_evidence,
    evidence_passed,
)

# Fixed stdout sentinel a harness scans for. NEVER changed lightly — it is the
# machine contract between cli.main and the PTY harness / CI gate.
VERDICT_SENTINEL = "VECTOR_VERDICT"

# The top-level evidence verdict for a whole turn.
#   GROUNDED — verified: success backed by deterministic, oracle-consuming evidence.
#   RAN      — the turn ran (some/all steps succeeded) but carries NO grounded evidence.
#   FAILED   — the trace did not succeed (>=1 step failed / aborted).
#   NO_TRACE — no VGG trace was produced (e.g. a chat-only / tool_use turn, or error).
EVIDENCE_GROUNDED = "GROUNDED"
EVIDENCE_RAN = "RAN"
EVIDENCE_FAILED = "FAILED"
EVIDENCE_NO_TRACE = "NO_TRACE"

# Exit codes the harness asserts against (verified == (exit == 0)).
EXIT_VERIFIED = 0
EXIT_ERROR = 1
EXIT_RAN_NOT_VERIFIED = 2

_DIAGNOSTIC_VALUE_MAX_CHARS = 96


def _short_scalar(value: Any) -> str:
    """Return one bounded, single-line scalar for a human-facing diagnostic."""

    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float):
            return f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value)
    text = " ".join(str(value).split())
    if len(text) > _DIAGNOSTIC_VALUE_MAX_CHARS:
        return text[: _DIAGNOSTIC_VALUE_MAX_CHARS - 1] + "…"
    return text


def _short_value(value: Any) -> str:
    """Format only the useful identity/position part of a structured value."""

    if value is None:
        return ""
    if isinstance(value, dict):
        label = _short_scalar(
            value.get("label")
            or value.get("name")
            or value.get("room")
            or ""
        )
        point = (
            value.get("xy")
            or value.get("position_xy")
            or value.get("target_xy")
            or value.get("position")
        )
        point_text = _short_value(point) if point is not None else ""
        if label and point_text:
            return f"{label}@{point_text}"
        return label or point_text
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        if (
            2 <= len(value) <= 3
            and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
        ):
            return f"({', '.join(_short_scalar(item) for item in value)})"
        items = ", ".join(_short_scalar(item) for item in value[:4])
        suffix = ", …" if len(value) > 4 else ""
        return f"[{items}{suffix}]"
    return _short_scalar(value)


def _error_payload(step: Any) -> dict[str, Any]:
    """Decode the native tool's JSON error without exposing its raw prose."""

    raw = str(getattr(step, "error", "") or "").strip()
    if not raw.startswith("{"):
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_value(
    payloads: tuple[dict[str, Any], ...],
    keys: tuple[str, ...],
) -> Any:
    for payload in payloads:
        for key in keys:
            value = payload.get(key)
            if value is not None and value != "":
                return value
    return None


def _step_failure_diagnostic(
    step: Any,
    *,
    verify: str,
    evidence: str,
) -> tuple[str, str, str]:
    """Project a full StepRecord failure into bounded code/expected/observed data.

    The raw error and result_data deliberately stay out of VerdictReport and the
    terminal.  Native navigation metadata is stable enough to retain the useful
    comparison: the failed/current goal on the expected side and the reported
    waypoint/robot position on the observed side.
    """

    if evidence == EVIDENCE_GROUNDED:
        return "", "", ""

    result_data = getattr(step, "result_data", {}) or {}
    structured = dict(result_data) if isinstance(result_data, dict) else {}
    error_data = _error_payload(step)
    payloads = (structured, error_data)

    diagnosis = _short_value(
        _first_value(
            payloads,
            ("diagnosis_code", "error_code", "failure_code", "diagnosis"),
        )
    )
    step_success = bool(getattr(step, "success", False))
    verify_result = bool(getattr(step, "verify_result", False))
    actor = getattr(getattr(step, "actor_caused", None), "value", "")
    if not diagnosis:
        if not step_success:
            diagnosis = "execution_failed"
        elif not verify_result:
            diagnosis = "verification_failed"
        elif actor == "UNCAUSED":
            diagnosis = "actor_not_caused"
        else:
            diagnosis = "evidence_not_grounded"

    expected_value = _first_value(
        payloads,
        (
            "expected_goal",
            "expected_waypoint",
            "expected_xy",
            "expected",
            "failed_waypoint",
            "target_waypoint",
            "target_xy",
            "goal_xy",
            "target",
            "requested_room",
            "canonical_room",
            "room",
        ),
    )
    observed_keys = (
        (
            "available_rooms",
            "observed_waypoint",
            "observed_goal",
            "observed_xy",
            "observed",
            "actual_waypoint",
            "actual_xy",
            "actual",
            "position_xy",
        )
        if diagnosis in {"unknown_room", "invalid_room"}
        else (
            "observed_waypoint",
            "observed_goal",
            "observed_xy",
            "observed",
            "actual_waypoint",
            "actual_xy",
            "actual",
            "position_xy",
            "available_rooms",
        )
    )
    observed_value = _first_value(payloads, observed_keys)
    if observed_value is None:
        robot_state = _first_value(payloads, ("robot_state_after",))
        if isinstance(robot_state, dict):
            observed_value = (
                robot_state.get("position_xy")
                or robot_state.get("position")
            )

    expected = _short_value(expected_value)
    observed = _short_value(observed_value)
    if not expected and verify:
        expected = _short_value(verify)
    if not observed and (not verify_result or evidence != EVIDENCE_GROUNDED):
        observed = _short_value(verify_result)
    return diagnosis, expected, observed


@dataclass(frozen=True)
class StepVerdict:
    """Per-step evidence row — a pure projection of one StepRecord+SubGoal."""

    name: str
    strategy: str
    success: bool
    verify: str
    verify_result: bool
    evidence: str  # GROUNDED | RAN | FAILED (from classify_step_evidence)
    # Bounded human/machine diagnostic projection. Full StepRecord error/result_data
    # remain in the trace/log and are intentionally not copied into the verdict.
    diagnosis_code: str = ""
    expected: str = ""
    observed: str = ""


@dataclass(frozen=True)
class VerdictReport:
    """A frozen, JSON-serializable verdict for ONE executed turn.

    Built ONLY from ``trace_store.evidence_passed`` /
    ``trace_store.classify_step_evidence`` — never re-derived (see module docstring
    + the contract test ``test_verdict_matches_evidence_passed``).
    """

    verified: bool
    success: bool
    evidence: str  # GROUNDED | RAN | FAILED | NO_TRACE
    goal: str
    n_steps: int
    n_grounded: int
    oracle_names: tuple[str, ...]
    per_step: tuple[StepVerdict, ...] = ()
    error: str = ""

    # ------------------------------------------------------------------
    # Construction — the ONLY supported way to build a verdict from a run.
    # ------------------------------------------------------------------

    @classmethod
    def from_trace(
        cls, trace: Any, oracle_names: frozenset[str]
    ) -> "VerdictReport":
        """Build a verdict from an ExecutionTrace using the EXISTING gate.

        ``verified`` is delegated VERBATIM to ``evidence_passed`` (no second
        opinion). Per-step ``evidence`` is delegated to ``classify_step_evidence``.
        ``n_grounded`` counts the steps the classifier calls GROUNDED. The
        top-level ``evidence`` summarizes: GROUNDED iff verified, else FAILED if
        the trace did not succeed, else RAN.
        """
        sg_by_name = {sg.name: sg for sg in trace.goal_tree.sub_goals}
        verified = bool(evidence_passed(trace, oracle_names))

        per_step: list[StepVerdict] = []
        n_grounded = 0
        for s in trace.steps:
            sg = sg_by_name.get(s.sub_goal_name)
            if sg is None:
                # A step with no matching sub-goal cannot be classified by the
                # gate — record it as RAN-shaped metadata only (it never counts
                # toward GROUNDED, mirroring evidence_passed which ignores it).
                ev = EVIDENCE_RAN if s.success else EVIDENCE_FAILED
                verify_str = ""
            else:
                ev = classify_step_evidence(s, sg, oracle_names, trace.goal_tree.goal)
                verify_str = sg.verify
            if ev == EVIDENCE_GROUNDED:
                n_grounded += 1
            diagnosis, expected, observed = _step_failure_diagnostic(
                s,
                verify=verify_str,
                evidence=ev,
            )
            per_step.append(
                StepVerdict(
                    name=s.sub_goal_name,
                    strategy=s.strategy,
                    success=bool(s.success),
                    verify=verify_str,
                    verify_result=bool(s.verify_result),
                    evidence=ev,
                    diagnosis_code=diagnosis,
                    expected=expected,
                    observed=observed,
                )
            )

        if verified:
            top_evidence = EVIDENCE_GROUNDED
        elif not trace.success:
            top_evidence = EVIDENCE_FAILED
        else:
            top_evidence = EVIDENCE_RAN

        return cls(
            verified=verified,
            success=bool(trace.success),
            evidence=top_evidence,
            goal=trace.goal_tree.goal,
            n_steps=len(trace.steps),
            n_grounded=n_grounded,
            oracle_names=tuple(sorted(oracle_names)),
            per_step=tuple(per_step),
        )

    @classmethod
    def no_trace(cls, goal: str = "", error: str = "") -> "VerdictReport":
        """A fail-closed verdict for a turn that produced NO VGG trace.

        A chat-only / tool_use turn (or an error before any trace) has no
        deterministic per-step evidence to grade, so it can NEVER be verified.
        """
        return cls(
            verified=False,
            success=False,
            evidence=EVIDENCE_NO_TRACE,
            goal=goal,
            n_steps=0,
            n_grounded=0,
            oracle_names=(),
            per_step=(),
            error=error,
        )

    def compact_failure_reason(self) -> str:
        """Return exactly one concise reason, preferring the final failed retry."""

        for step in reversed(self.per_step):
            if not step.diagnosis_code:
                continue
            parts = [step.diagnosis_code]
            if step.expected:
                parts.append(f"expected={step.expected}")
            if step.observed:
                parts.append(f"observed={step.observed}")
            return " · ".join(parts)
        return ""

    # ------------------------------------------------------------------
    # Serialization + exit-code contract.
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """A plain JSON-safe dict (frozen dataclasses -> dicts, tuples -> lists)."""
        d = asdict(self)
        d["oracle_names"] = list(self.oracle_names)
        d["per_step"] = [asdict(s) for s in self.per_step]
        return d

    def to_sentinel_line(self) -> str:
        """The single stdout line a harness scans for: ``VECTOR_VERDICT {<json>}``."""
        return f"{VERDICT_SENTINEL} {json.dumps(self.to_dict(), ensure_ascii=False)}"

    def exit_code(self) -> int:
        """0 = verified, 2 = ran-not-verified, 1 = error / no trace.

        ``verified == (exit_code() == 0)`` is the harness's invariant; this method
        is the single source of that mapping.
        """
        if self.verified:
            return EXIT_VERIFIED
        if self.evidence == EVIDENCE_NO_TRACE:
            return EXIT_ERROR
        return EXIT_RAN_NOT_VERIFIED
