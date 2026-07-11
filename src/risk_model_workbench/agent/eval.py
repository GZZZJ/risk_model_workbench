"""Test-backed evaluation suite for the deterministic Agent Harness.

The suite manifest binds product scenarios to pytest node IDs.  A scenario can
only contribute successful assertions to a metric after its bound fixture has
passed; the YAML file is therefore an assertion inventory, not self-reported
runtime evidence.
"""

from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml


REQUIRED_SCENARIO_FIELDS = {
    "name",
    "fixture",
    "expected_transitions",
    "expected_blockers",
    "expected_artifacts",
    "expected_audit_verdict",
    "forbidden_runner_calls",
    "expected_trace_events",
    "tags",
}

ALLOWED_TAGS = {"crash", "policy_guard", "terminal_happy_path"}
ALLOWED_AUDIT_VERDICTS = {None, "complete", "incomplete", "scaffold"}


@dataclass(frozen=True)
class ScenarioExecution:
    """Result of executing one scenario fixture."""

    passed: bool
    returncode: int = 0
    output: str = ""
    observed_transitions: tuple[str, ...] = ()
    observed_blockers: tuple[str, ...] = ()
    observed_artifacts: tuple[str, ...] = ()
    observed_audit_verdict: str | None = None
    observed_trace_events: tuple[str, ...] = ()
    observed_runner_calls: tuple[str, ...] = ()
    evidence_emitted: bool = False


ScenarioRunner = Callable[[str], ScenarioExecution | bool]


def default_suite_path() -> Path:
    """Return the repository-local Harness suite manifest."""

    return Path(__file__).resolve().parents[3] / "evals" / "agent_harness" / "scenarios.yml"


def load_harness_suite(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate a Harness scenario manifest."""

    suite_path = Path(path) if path is not None else default_suite_path()
    payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Agent Harness eval suite must be a mapping")
    if payload.get("suite") != "harness" or int(payload.get("version") or 0) != 1:
        raise ValueError("Agent Harness eval suite must declare suite=harness and version=1")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("Agent Harness eval suite must contain scenarios")

    names: set[str] = set()
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise ValueError(f"scenario {index} must be a mapping")
        missing = sorted(REQUIRED_SCENARIO_FIELDS - set(scenario))
        if missing:
            raise ValueError(f"scenario {index} missing fields: {', '.join(missing)}")
        name = str(scenario.get("name") or "").strip()
        fixture = str(scenario.get("fixture") or "").strip()
        if not name or name in names:
            raise ValueError(f"scenario name must be non-empty and unique: {name!r}")
        if "::" not in fixture:
            raise ValueError(f"scenario {name} fixture must be a pytest node ID")
        names.add(name)
        for field in (
            "expected_transitions",
            "expected_blockers",
            "expected_artifacts",
            "forbidden_runner_calls",
            "expected_trace_events",
            "tags",
        ):
            values = scenario[field]
            if not isinstance(values, list):
                raise ValueError(f"scenario {name} field {field} must be a list")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"scenario {name} field {field} values must be non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"scenario {name} field {field} values must be unique")
        unknown_tags = set(scenario["tags"]) - ALLOWED_TAGS
        if unknown_tags:
            raise ValueError(f"scenario {name} has unknown tags: {', '.join(sorted(unknown_tags))}")
        if scenario["expected_audit_verdict"] not in ALLOWED_AUDIT_VERDICTS:
            raise ValueError(f"scenario {name} has invalid expected_audit_verdict")
    return payload


def evaluate_harness_suite(
    path: str | Path | None = None,
    *,
    runner: ScenarioRunner | None = None,
    scenario_ids: Sequence[str] | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run scenarios and calculate fail-closed, exact-denominator metrics.

    ``runner`` is injectable so metric calculation can be unit tested without
    recursively starting pytest.  The production default invokes each fixture
    in an isolated pytest subprocess.
    """

    suite_path = Path(path) if path is not None else default_suite_path()
    suite = load_harness_suite(suite_path)
    selected = set(scenario_ids or [])
    scenarios = [item for item in suite["scenarios"] if not selected or item["name"] in selected]
    unknown = selected - {str(item["name"]) for item in suite["scenarios"]}
    if unknown:
        raise ValueError(f"unknown Agent Harness scenarios: {', '.join(sorted(unknown))}")
    if not scenarios:
        raise ValueError("Agent Harness eval selection is empty")

    repo_root = suite_path.resolve().parents[2]
    execute = runner or _pytest_runner(
        repo_root,
        Path(cwd) if cwd is not None else repo_root,
    )
    results: dict[str, ScenarioExecution] = {}
    for scenario in scenarios:
        outcome = execute(str(scenario["fixture"]))
        if isinstance(outcome, bool):
            outcome = ScenarioExecution(passed=outcome, returncode=0 if outcome else 1)
        if not isinstance(outcome, ScenarioExecution):
            raise TypeError("scenario runner must return ScenarioExecution or bool")
        results[str(scenario["name"])] = outcome

    conformance = {
        str(item["name"]): _scenario_conformance(item, results[str(item["name"])]) for item in scenarios
    }
    transition_denominator = sum(len(item["expected_transitions"]) for item in scenarios)
    transition_numerator = sum(
        len(set(item["expected_transitions"]) & set(results[str(item["name"])].observed_transitions))
        if results[str(item["name"])].passed
        else 0
        for item in scenarios
    )
    crash_scenarios = [item for item in scenarios if "crash" in item.get("tags", [])]
    recovery_numerator = sum(conformance[str(item["name"])]["passed"] for item in crash_scenarios)
    trace_denominator = sum(len(item.get("expected_trace_events", [])) for item in scenarios)
    trace_numerator = sum(
        len(set(item["expected_trace_events"]) & set(results[str(item["name"])].observed_trace_events))
        if results[str(item["name"])].passed
        else 0
        for item in scenarios
    )
    terminal_scenarios = [item for item in scenarios if "terminal_happy_path" in item.get("tags", [])]
    strict_numerator = sum(
        conformance[str(item["name"])]["passed"]
        and results[str(item["name"])].observed_audit_verdict == "complete"
        for item in terminal_scenarios
    )
    policy_escapes = [
        str(item["name"])
        for item in scenarios
        if "policy_guard" in item.get("tags", []) and not conformance[str(item["name"])]["passed"]
    ]
    duplicate_external = [
        str(item["name"])
        for item in scenarios
        if item["forbidden_runner_calls"]
        and (
            not results[str(item["name"])].passed
            or not results[str(item["name"])].evidence_emitted
            or bool(conformance[str(item["name"])]["forbidden_runner_calls_observed"])
        )
    ]

    metrics = {
        "policy_escape_count": _count_metric(policy_escapes),
        "transition_conformance_rate": _rate_metric(
            transition_numerator,
            transition_denominator,
            [
                str(item["name"])
                for item in scenarios
                if set(item["expected_transitions"]) - set(results[str(item["name"])].observed_transitions)
                or not results[str(item["name"])].passed
            ],
        ),
        "recovery_success_rate": _rate_metric(
            int(recovery_numerator),
            len(crash_scenarios),
            [str(item["name"]) for item in crash_scenarios if not conformance[str(item["name"])]["passed"]],
        ),
        "duplicate_external_execution_count": _count_metric(duplicate_external),
        "trace_coverage_rate": _rate_metric(
            trace_numerator,
            trace_denominator,
            [
                str(item["name"])
                for item in scenarios
                if set(item["expected_trace_events"]) - set(results[str(item["name"])].observed_trace_events)
                or (item["expected_trace_events"] and not results[str(item["name"])].passed)
            ],
        ),
        "strict_audit_complete_rate": _rate_metric(
            int(strict_numerator),
            len(terminal_scenarios),
            [str(item["name"]) for item in terminal_scenarios if not conformance[str(item["name"])]["passed"]],
        ),
    }
    scenario_results = []
    for scenario in scenarios:
        name = str(scenario["name"])
        outcome = results[name]
        checked = conformance[name]
        scenario_results.append(
            {
                "scenario_id": name,
                "fixture": scenario["fixture"],
                "passed": checked["passed"],
                "fixture_passed": outcome.passed,
                "evidence_emitted": outcome.evidence_emitted,
                "returncode": outcome.returncode,
                "observed_transitions": list(outcome.observed_transitions),
                "observed_blockers": list(outcome.observed_blockers),
                "observed_runner_calls": list(outcome.observed_runner_calls),
                "missing_evidence": checked["missing_evidence"],
                "unexpected_evidence": checked["unexpected_evidence"],
                "forbidden_runner_calls_observed": checked["forbidden_runner_calls_observed"],
                "failure_output": "" if checked["passed"] else outcome.output[-4000:],
            }
        )
    return {
        "suite": "harness",
        "version": 1,
        "passed": all(item["passed"] for item in scenario_results),
        "scenario_count": len(scenario_results),
        "metrics": metrics,
        "definitions": {
            "transition_conformance_rate": "conforming transitions / asserted transitions",
            "recovery_success_rate": "successfully reconciled crash scenarios / crash scenarios",
            "trace_coverage_rate": "expected correlated events present / expected correlated events",
            "strict_audit_complete_rate": "strict-complete terminal scenarios / terminal happy-path scenarios",
            "count_metrics": "count plus scenario_ids; a failed guard scenario is counted fail-closed",
        },
        "scenarios": scenario_results,
    }


def _pytest_runner(repo_root: Path, cwd: Path) -> ScenarioRunner:
    def run(node_id: str) -> ScenarioExecution:
        env = os.environ.copy()
        source = str(repo_root / "src")
        env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with tempfile.TemporaryDirectory(prefix="rmw-agent-eval-") as private_dir:
            evidence_path = Path(private_dir) / "evidence.json"
            env["RMW_AGENT_EVAL_EVIDENCE"] = str(evidence_path)
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", node_id],
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
        output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        return scenario_execution_from_evidence(
            evidence,
            passed=completed.returncode == 0,
            returncode=completed.returncode,
            output=output,
        )

    return run


def emit_scenario_evidence(
    *,
    state_pairs: Sequence[tuple[dict[str, Any], dict[str, Any]]] = (),
    workspace: str | Path | None = None,
    audit: dict[str, Any] | None = None,
    validation_errors: Sequence[str] = (),
    recovery_reports: Sequence[dict[str, Any]] = (),
    runner_calls: Sequence[Sequence[str]] = (),
) -> None:
    """Derive evidence from runtime records and emit only proved assertions."""

    target = os.environ.get("RMW_AGENT_EVAL_EVIDENCE")
    if not target:
        return
    transitions = {
        f"{before.get('status')}_to_{after.get('status')}"
        for before, after in state_pairs
        if before.get("status") and after.get("status")
    }
    blockers: set[str] = set()
    for before, after in state_pairs:
        for snapshot in (before, after):
            blocker = snapshot.get("blocker") if isinstance(snapshot.get("blocker"), dict) else {}
            reason = str(blocker.get("reason") or "")
            canonical_reason = _canonical_blocker_reason(reason)
            if canonical_reason:
                blockers.add(canonical_reason)
            if snapshot.get("status") == "waiting_for_user":
                blockers.add("user_confirmation_required")
    error_text = "\n".join(str(error) for error in validation_errors)
    for needle, label in (
        ("response identity mismatch", "stale_response_identity"),
        ("path escape", "path_escape"),
        ("retry budget exceeded", "retry_budget"),
        ("registry_digest_drift", "plan_integrity_error"),
        ("blocked_flag", "plan_integrity_error"),
        ("cyclic_dependency", "cyclic_dependency"),
    ):
        if needle in error_text:
            blockers.add(label)
    root = Path(workspace) if workspace is not None else None
    if root is not None and (root / "audit" / "approvals.yml").exists():
        approvals = yaml.safe_load((root / "audit" / "approvals.yml").read_text(encoding="utf-8")) or {}
        for item in approvals.get("approvals") or []:
            if item.get("status") == "rejected":
                blockers.add("approval_rejected")
            if item.get("revoke_reason") == "approval_subject_drift":
                blockers.add("approval_subject_drift")
    for report in recovery_reports:
        if (report.get("transaction_divergence") or {}).get("detected"):
            blockers.add("transaction_divergence")
        for attempt in report.get("attempts") or []:
            if attempt.get("recovery_action") == "requeue" and attempt.get("intent_status") == "recorded":
                blockers.add("interrupted_before_dispatch")
            if attempt.get("recovery_action") == "requeue" and attempt.get("intent_status") != "recorded":
                blockers.add("missing_action_result")
            if attempt.get("recovery_action") == "apply_receipt":
                blockers.add("unconsumed_receipt")
    artifacts = set()
    trace_events = set()
    if root is not None and root.exists():
        artifacts = {str(path.relative_to(root)) for path in root.rglob("*")}
        trace_path = root / "audit" / "agent_trace.jsonl"
        if trace_path.exists():
            rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            action_attempts = {row.get("attempt_id") for row in rows if row.get("event") == "action" and row.get("attempt_id")}
            result_attempts = {row.get("attempt_id") for row in rows if row.get("event") == "result" and row.get("attempt_id")}
            if action_attempts and action_attempts <= result_attempts:
                trace_events.add("action_result_attempt_pair")
            if any(
                row.get("event") == "result"
                and (row.get("status") == "scaffold" or (row.get("stage_result") or {}).get("status") == "scaffold")
                for row in rows
            ):
                trace_events.add("scaffold_result")
            if any(row.get("summary") == "Interrupted safe attempt requeued." for row in rows):
                trace_events.add("recovery_requeue")
    observed_calls = [_runner_call_id(call) for call in runner_calls]
    payload = {
        "transitions": sorted(transitions),
        "blockers": sorted(blockers),
        "artifacts": sorted(artifacts),
        "audit_verdict": audit.get("verdict") if isinstance(audit, dict) else None,
        "trace_events": sorted(trace_events),
        "runner_calls": observed_calls,
    }
    Path(target).write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def scenario_execution_from_evidence(
    evidence: dict[str, Any],
    *,
    passed: bool,
    returncode: int = 0,
    output: str = "",
) -> ScenarioExecution:
    """Build a scenario result from production-emitter output."""

    return ScenarioExecution(
        passed=passed,
        returncode=returncode,
        output=output,
        observed_transitions=tuple(evidence.get("transitions") or ()),
        observed_blockers=tuple(evidence.get("blockers") or ()),
        observed_artifacts=tuple(evidence.get("artifacts") or ()),
        observed_audit_verdict=evidence.get("audit_verdict"),
        observed_trace_events=tuple(evidence.get("trace_events") or ()),
        observed_runner_calls=tuple(evidence.get("runner_calls") or ()),
        evidence_emitted=bool(evidence),
    )


def _runner_call_id(argv: Sequence[str]) -> str:
    parts = [str(part) for part in argv]
    base = "_".join(parts[:2]) if len(parts) >= 2 else (parts[0] if parts else "unknown")
    if "--sql-approved" in parts:
        return f"{base}_execute"
    if "--dry-run-sql" in parts:
        return f"{base}_prepare"
    return base


def _canonical_blocker_reason(reason: str) -> str:
    supported = {
        "approval_required",
        "advisor_required",
        "advisor_requires_user_confirmation",
    }
    if reason in supported:
        return reason
    if reason.startswith("approval subject drift"):
        return "approval_subject_drift"
    if reason:
        return f"unsupported_blocker:{reason}"
    return ""


def _scenario_conformance(scenario: dict[str, Any], outcome: ScenarioExecution) -> dict[str, Any]:
    missing = {
        "transitions": sorted(set(scenario["expected_transitions"]) - set(outcome.observed_transitions)),
        "blockers": sorted(set(scenario["expected_blockers"]) - set(outcome.observed_blockers)),
        "artifacts": sorted(set(scenario["expected_artifacts"]) - set(outcome.observed_artifacts)),
        "trace_events": sorted(set(scenario["expected_trace_events"]) - set(outcome.observed_trace_events)),
    }
    expected_verdict = scenario["expected_audit_verdict"]
    if expected_verdict != outcome.observed_audit_verdict:
        missing["audit_verdict"] = [expected_verdict]
    forbidden = sorted(set(scenario["forbidden_runner_calls"]) & set(outcome.observed_runner_calls))
    for rule in scenario["forbidden_runner_calls"]:
        if rule == "*" and outcome.observed_runner_calls:
            forbidden.append(rule)
        elif rule.startswith("duplicate:"):
            action = rule.split(":", 1)[1]
            if outcome.observed_runner_calls.count(action) > 1:
                forbidden.append(rule)
    unexpected = {
        "transitions": sorted(set(outcome.observed_transitions) - set(scenario["expected_transitions"])),
        "blockers": sorted(set(outcome.observed_blockers) - set(scenario["expected_blockers"])),
        "trace_events": sorted(set(outcome.observed_trace_events) - set(scenario["expected_trace_events"])),
    }
    passed = outcome.passed and outcome.evidence_emitted and not any(missing.values()) and not any(unexpected.values()) and not forbidden
    return {
        "passed": passed,
        "missing_evidence": missing,
        "unexpected_evidence": unexpected,
        "forbidden_runner_calls_observed": forbidden,
    }


def _count_metric(scenario_ids: Sequence[str]) -> dict[str, Any]:
    return {"count": len(scenario_ids), "scenario_ids": list(scenario_ids)}


def _rate_metric(numerator: int, denominator: int, failed_scenario_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
        "failed_scenario_ids": list(failed_scenario_ids),
    }
