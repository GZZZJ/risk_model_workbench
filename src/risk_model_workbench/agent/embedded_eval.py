"""Deterministic product gates for the embedded reasoning layer."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from pydantic import ValidationError

from risk_model_workbench.agent.context_pack import build_context_pack, context_pack_hash
from risk_model_workbench.agent.model_gateway import (
    AgentModelConfig,
    FakeModelGateway,
    ModelUnavailableError,
    StructuredOutputError,
)
from risk_model_workbench.agent.reasoning_contracts import EmbeddedAdvisorAnswer, StructuredReasoningRequest
from risk_model_workbench.modeling.llm_tuning import resolve_tuning_config, validate_tuning_plan


def evaluate_embedded_suite() -> dict[str, Any]:
    cases: list[tuple[str, Callable[[], dict[str, Any]]]] = [
        ("valid_structured_answer", _valid_structured_answer),
        ("malformed_answer_fails_closed", _malformed_answer_fails_closed),
        ("tampered_context_rejected", _tampered_context_rejected),
        ("row_level_context_rejected", _row_level_context_rejected),
        ("missing_model_config_fails_closed", _missing_model_config_fails_closed),
        ("tuning_bounds_enforced", _tuning_bounds_enforced),
    ]
    results = []
    for case_id, function in cases:
        try:
            evidence = function()
            passed = bool(evidence.get("passed"))
            error = ""
        except Exception as exc:  # evaluator itself also fails closed
            evidence = {}
            passed = False
            error = f"{type(exc).__name__}: {str(exc)[:500]}"
        results.append({"case_id": case_id, "passed": passed, "evidence": evidence, "error": error})
    passed_count = sum(item["passed"] for item in results)
    return {
        "suite": "embedded",
        "passed": passed_count == len(results),
        "case_count": len(results),
        "metrics": {
            "contract_case_pass_rate": {
                "numerator": passed_count,
                "denominator": len(results),
                "rate": passed_count / len(results) if results else None,
                "failed_case_ids": [item["case_id"] for item in results if not item["passed"]],
            },
            "unsafe_accept_count": {
                "count": sum(
                    item["evidence"].get("unsafe_accepted", 0)
                    for item in results
                    if isinstance(item.get("evidence"), dict)
                ),
            },
        },
        "cases": results,
    }


def _valid_structured_answer() -> dict[str, Any]:
    request = _request()
    gateway = FakeModelGateway(
        [{"type": "failure_diagnosis", "status": "answered", "decision": "stop", "summary": "Stop safely."}]
    )
    answer, metadata = gateway.generate_structured(request, EmbeddedAdvisorAnswer)
    return {"passed": answer.decision == "stop" and metadata.context_hash == request.context_hash}


def _malformed_answer_fails_closed() -> dict[str, Any]:
    try:
        FakeModelGateway([{"type": "failure_diagnosis", "decision": "continue"}]).generate_structured(
            _request(), EmbeddedAdvisorAnswer
        )
    except StructuredOutputError:
        return {"passed": True, "unsafe_accepted": 0}
    return {"passed": False, "unsafe_accepted": 1}


def _tampered_context_rejected() -> dict[str, Any]:
    pack = _context_pack()
    digest = pack["context_hash"]
    pack["project"] = "tampered"
    try:
        StructuredReasoningRequest(
            request_id="eval_request",
            request_type="failure_diagnosis_required",
            expected_response_type="failure_diagnosis",
            question="Diagnose.",
            context_hash=digest,
            context_pack=pack,
        )
    except ValidationError:
        return {"passed": True, "unsafe_accepted": 0}
    return {"passed": False, "unsafe_accepted": 1}


def _row_level_context_rejected() -> dict[str, Any]:
    with TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        source = workspace / "modeling" / "main" / "metrics.json"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('{"rows":[{"customer_id":"12345678901","score":0.5}]}\n', encoding="utf-8")
        pack = build_context_pack(
            workspace,
            project="eval",
            version_id="v1",
            task_id="task",
            attempt_id="attempt",
            request_type="failure_diagnosis_required",
            paths=["modeling/main/metrics.json"],
            constraints=[],
            allowed_tools=[],
            output_contract={"type": "failure_diagnosis"},
        )
        entry = pack["files"][0]
        return {
            "passed": entry.get("status") == "rejected" and entry.get("reason") == "row_level_content",
            "unsafe_accepted": int(entry.get("status") == "included"),
        }


def _missing_model_config_fails_closed() -> dict[str, Any]:
    try:
        AgentModelConfig.from_env({})
    except ModelUnavailableError:
        return {"passed": True, "unsafe_accepted": 0}
    return {"passed": False, "unsafe_accepted": 1}


def _tuning_bounds_enforced() -> dict[str, Any]:
    config = resolve_tuning_config(
        {"training": {"mode": "llm_guided_tune", "tuning": {"candidates_per_round": 3}}}
    )
    plan = {
        "algorithm": "lightgbm",
        "round": 1,
        "experiment": "main",
        "diagnosis": {
            "state": "healthy",
            "summary": "eval",
            "evidence": ["baseline_only"],
            "recommended_direction": ["bounded_local_exploration"],
            "confidence": 0.8,
        },
        "decision": "continue",
        "candidates": [
            {"name": "unsafe", "params": {"learning_rate": 9.0}, "reason": "out of bounds"}
        ],
    }
    try:
        validate_tuning_plan(
            plan,
            config,
            advisor_type="embedded_eval",
            expected_experiment="main",
            expected_round=1,
            expected_algorithm="lightgbm",
        )
    except ValueError:
        return {"passed": True, "unsafe_accepted": 0}
    return {"passed": False, "unsafe_accepted": 1}


def _request() -> StructuredReasoningRequest:
    pack = _context_pack()
    return StructuredReasoningRequest(
        request_id="eval_request",
        request_type="failure_diagnosis_required",
        expected_response_type="failure_diagnosis",
        question="Diagnose.",
        context_hash=pack["context_hash"],
        context_pack=pack,
    )


def _context_pack() -> dict[str, Any]:
    pack = {
        "version": 1,
        "project": "eval",
        "version_id": "v1",
        "task_id": "task",
        "attempt_id": "attempt",
        "request_type": "failure_diagnosis_required",
        "files": [],
        "constraints": [],
        "allowed_tools": [],
        "output_contract": {"type": "failure_diagnosis"},
        "provenance": [],
        "limits": {},
        "truncation": {},
    }
    pack["context_hash"] = context_pack_hash(pack)
    return pack
