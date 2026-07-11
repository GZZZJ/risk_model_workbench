import copy
import json
from pathlib import Path

import pytest
import yaml

import risk_model_workbench.cli as cli_module
from risk_model_workbench.agent.eval import (
    ScenarioExecution,
    emit_scenario_evidence,
    evaluate_harness_suite,
    load_harness_suite,
    scenario_execution_from_evidence,
)
from risk_model_workbench.cli import main


SUITE = Path(__file__).resolve().parents[2] / "evals" / "agent_harness" / "scenarios.yml"


def test_harness_eval_schema_and_exact_denominators():
    suite = load_harness_suite(SUITE)
    report = evaluate_harness_suite(SUITE, runner=_evidence_runner(suite))

    scenarios = suite["scenarios"]
    metrics = report["metrics"]
    assert report["passed"] is True
    assert report["scenario_count"] == 18
    assert metrics["transition_conformance_rate"]["denominator"] == sum(
        len(item["expected_transitions"]) for item in scenarios
    )
    assert metrics["recovery_success_rate"] == {
        "numerator": 4,
        "denominator": 4,
        "rate": 1.0,
        "failed_scenario_ids": [],
    }
    assert metrics["trace_coverage_rate"]["denominator"] == sum(
        len(item.get("expected_trace_events", [])) for item in scenarios
    )
    assert metrics["strict_audit_complete_rate"] == {
        "numerator": 1,
        "denominator": 1,
        "rate": 1.0,
        "failed_scenario_ids": [],
    }
    assert metrics["policy_escape_count"] == {"count": 0, "scenario_ids": []}
    assert metrics["duplicate_external_execution_count"] == {"count": 0, "scenario_ids": []}


def test_harness_eval_failure_metrics_name_scenarios_and_keep_denominators():
    suite = load_harness_suite(SUITE)
    passing = _evidence_runner(suite)

    def runner(fixture: str):
        if "approval_drift" in fixture:
            return ScenarioExecution(False, 1, "synthetic failure")
        return passing(fixture)

    report = evaluate_harness_suite(SUITE, runner=runner)
    metrics = report["metrics"]

    assert report["passed"] is False
    assert metrics["policy_escape_count"] == {"count": 1, "scenario_ids": ["sql_drift"]}
    assert metrics["duplicate_external_execution_count"] == {"count": 1, "scenario_ids": ["sql_drift"]}
    assert metrics["transition_conformance_rate"]["denominator"] > metrics["transition_conformance_rate"]["numerator"]
    assert metrics["transition_conformance_rate"]["failed_scenario_ids"] == ["sql_drift"]
    scenario = next(item for item in report["scenarios"] if item["scenario_id"] == "sql_drift")
    assert scenario["failure_output"] == "synthetic failure"


def test_bogus_expected_transition_and_trace_reduce_rates_and_fail_suite(tmp_path):
    original = load_harness_suite(SUITE)
    mutated = copy.deepcopy(original)
    target = mutated["scenarios"][0]
    target["expected_transitions"].append("bogus_transition_never_observed")
    target["expected_trace_events"].append("bogus_trace_never_observed")
    mutated_path = tmp_path / "evals" / "agent_harness" / "scenarios.yml"
    mutated_path.parent.mkdir(parents=True)
    mutated_path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")

    report = evaluate_harness_suite(mutated_path, runner=_evidence_runner(original))

    assert report["passed"] is False
    transition = report["metrics"]["transition_conformance_rate"]
    trace = report["metrics"]["trace_coverage_rate"]
    assert transition["numerator"] + 1 == transition["denominator"]
    assert trace["numerator"] + 1 == trace["denominator"]
    local = next(item for item in report["scenarios"] if item["scenario_id"] == "local_happy_path")
    assert local["missing_evidence"]["transitions"] == ["bogus_transition_never_observed"]
    assert local["missing_evidence"]["trace_events"] == ["bogus_trace_never_observed"]


def test_production_emitter_does_not_hide_unexpected_runtime_transition(tmp_path, monkeypatch):
    suite = load_harness_suite(SUITE)
    passing = _evidence_runner(suite)
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("RMW_AGENT_EVAL_EVIDENCE", str(evidence_path))
    emit_scenario_evidence(state_pairs=[({"status": "draft"}, {"status": "blocked"})])
    emitted = scenario_execution_from_evidence(json.loads(evidence_path.read_text(encoding="utf-8")), passed=True)

    def runner(fixture):
        if "local_happy_path" in fixture:
            return emitted
        return passing(fixture)

    report = evaluate_harness_suite(SUITE, runner=runner)
    local = next(item for item in report["scenarios"] if item["scenario_id"] == "local_happy_path")
    assert report["passed"] is False
    assert emitted.observed_transitions == ("draft_to_blocked",)
    assert local["unexpected_evidence"]["transitions"] == ["draft_to_blocked"]


def test_production_emitter_forbidden_runner_call_fails_and_is_named(tmp_path, monkeypatch):
    suite = load_harness_suite(SUITE)
    passing = _evidence_runner(suite)
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("RMW_AGENT_EVAL_EVIDENCE", str(evidence_path))
    emit_scenario_evidence(runner_calls=[["feature", "refine", "--sql-approved"]])
    emitted = scenario_execution_from_evidence(json.loads(evidence_path.read_text(encoding="utf-8")), passed=True)

    def runner(fixture):
        if "approval_remains_blocked" in fixture:
            return emitted
        return passing(fixture)

    report = evaluate_harness_suite(SUITE, runner=runner)
    metric = report["metrics"]["duplicate_external_execution_count"]
    assert report["passed"] is False
    assert metric == {"count": 1, "scenario_ids": ["sql_reject"]}
    rejected = next(item for item in report["scenarios"] if item["scenario_id"] == "sql_reject")
    assert rejected["forbidden_runner_calls_observed"] == ["feature_refine_execute"]


def test_production_emitter_unknown_blocker_fails_closed(tmp_path, monkeypatch):
    suite = load_harness_suite(SUITE)
    passing = _evidence_runner(suite)
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setenv("RMW_AGENT_EVAL_EVIDENCE", str(evidence_path))
    emit_scenario_evidence(
        state_pairs=[
            (
                {"status": "blocked", "blocker": {"reason": "new_security_blocker"}},
                {"status": "blocked", "blocker": {"reason": "new_security_blocker"}},
            )
        ]
    )
    emitted = scenario_execution_from_evidence(json.loads(evidence_path.read_text(encoding="utf-8")), passed=True)

    def runner(fixture):
        if "plan_validation_rejects_registry" in fixture:
            return emitted
        return passing(fixture)

    report = evaluate_harness_suite(SUITE, runner=runner)
    scenario = next(item for item in report["scenarios"] if item["scenario_id"] == "plan_tamper")
    assert report["passed"] is False
    assert emitted.observed_blockers == ("unsupported_blocker:new_security_blocker",)
    assert scenario["unexpected_evidence"]["blockers"] == ["unsupported_blocker:new_security_blocker"]


def test_agent_eval_cli_outputs_json_and_returns_gate_status(monkeypatch, capsys):
    report = {
        "suite": "harness",
        "version": 1,
        "passed": True,
        "scenario_count": 1,
        "metrics": {},
        "scenarios": [],
    }
    monkeypatch.setattr(cli_module, "evaluate_harness_suite", lambda: report)

    assert main(["agent", "eval", "--suite", "harness", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == report

    monkeypatch.setattr(cli_module, "evaluate_harness_suite", lambda: {**report, "passed": False})
    assert main(["agent", "eval", "--suite", "harness", "--json"]) == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tags", ["unknown"], "unknown tags"),
        ("expected_audit_verdict", "maybe", "invalid expected_audit_verdict"),
        ("expected_transitions", [""], "non-empty strings"),
        ("expected_trace_events", ["same", "same"], "must be unique"),
    ],
)
def test_suite_schema_rejects_invalid_tags_verdict_and_list_values(tmp_path, field, value, message):
    payload = copy.deepcopy(load_harness_suite(SUITE))
    payload["scenarios"][0][field] = value
    path = tmp_path / "scenarios.yml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_harness_suite(path)


def _evidence_runner(suite):
    by_fixture = {item["fixture"]: item for item in suite["scenarios"]}

    def run(fixture):
        scenario = by_fixture[fixture]
        return ScenarioExecution(
            passed=True,
            observed_transitions=tuple(scenario["expected_transitions"]),
            observed_blockers=tuple(scenario["expected_blockers"]),
            observed_artifacts=tuple(scenario["expected_artifacts"]),
            observed_audit_verdict=scenario["expected_audit_verdict"],
            observed_trace_events=tuple(scenario["expected_trace_events"]),
            evidence_emitted=True,
        )

    return run
