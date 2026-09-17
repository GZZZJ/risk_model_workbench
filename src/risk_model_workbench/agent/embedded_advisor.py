"""Resolve durable Advisor requests with the embedded model gateway."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.agent.advisor import (
    accept_advisor_response,
    load_advisor_context_pack,
    load_advisor_request,
)
from risk_model_workbench.agent.model_gateway import ModelGateway
from risk_model_workbench.agent.reasoning_contracts import EmbeddedAdvisorAnswer, StructuredReasoningRequest
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.modeling.llm_tuning import resolve_tuning_config, validate_tuning_plan


class EmbeddedAdvisorError(RuntimeError):
    """Raised when a model answer cannot pass the existing Advisor contract."""


def answer_advisor_request(
    workspace: str | Path,
    request_id: str,
    gateway: ModelGateway,
) -> dict[str, Any]:
    """Generate, persist, validate, and accept one pending Advisor answer.

    The function deliberately submits through ``accept_advisor_response`` so an
    embedded model has no more authority than an external Advisor response had.
    """
    workspace_path = Path(workspace)
    request = load_advisor_request(workspace_path, request_id)
    if request.get("status") != "pending":
        raise EmbeddedAdvisorError(f"advisor request is not pending: {request.get('status')}")
    context_pack = load_advisor_context_pack(workspace_path, request)
    expected_type = str((request.get("expected_response") or {}).get("type") or "")
    reasoning_request = StructuredReasoningRequest(
        request_id=request_id,
        request_type=str(request.get("type") or ""),
        expected_response_type=expected_type,
        question=str(request.get("question") or ""),
        context_hash=str(request.get("context_hash") or ""),
        context_pack=context_pack,
        constraints=[str(item) for item in request.get("constraints") or []],
    )
    answer, metadata = gateway.generate_structured(reasoning_request, EmbeddedAdvisorAnswer)
    if answer.type != expected_type:
        raise EmbeddedAdvisorError(f"embedded response type mismatch: expected {expected_type}, got {answer.type}")

    output_files: list[str] = []
    if answer.tuning_plan is not None:
        output_files.append(_persist_tuning_plan(workspace_path, request, answer.tuning_plan.model_dump(mode="json")))

    response = {
        "version": 1,
        "request_id": request_id,
        "type": answer.type,
        "status": answer.status,
        "decision": answer.decision,
        "summary": answer.summary,
        "request_identity": {
            "task_id": request.get("task_id"),
            "attempt_id": request.get("attempt_id"),
            "invocation_hash": request.get("invocation_hash"),
            "context_hash": request.get("context_hash"),
            "round": request.get("round"),
        },
        "output_files": output_files,
        "risk_notes": answer.risk_notes,
        "requires_user_confirmation": answer.requires_user_confirmation,
        "generated_by": "embedded_agent",
    }
    store = WorkspaceStore(workspace_path)
    invocation_relative = Path("audit") / "model_invocations" / f"{request_id}.json"
    invocation_evidence = {
        "version": 1,
        "request_id": request_id,
        "request_hash": request.get("request_hash", ""),
        "context_hash": request.get("context_hash", ""),
        "response_type": answer.type,
        "decision": answer.decision,
        "model": metadata.model_dump(mode="json"),
    }
    if not store.create_once(invocation_relative, invocation_evidence):
        existing = json.loads((workspace_path / invocation_relative).read_text(encoding="utf-8"))
        if existing != invocation_evidence:
            raise EmbeddedAdvisorError(f"model invocation evidence collision: {request_id}")
    response["model_invocation"] = str(invocation_relative)

    draft_relative = Path("audit") / "embedded_advisor_drafts" / f"{request_id}.response.json"
    draft_path = store.atomic_write(draft_relative, response)
    accepted = accept_advisor_response(workspace_path, draft_path)
    if not accepted.get("accepted"):
        raise EmbeddedAdvisorError("embedded Advisor response rejected: " + "; ".join(accepted.get("errors") or []))
    return {
        "request_id": request_id,
        "accepted": True,
        "response_path": accepted.get("response_path", ""),
        "model_invocation": str(workspace_path / invocation_relative),
        "decision": answer.decision,
        "requires_user_confirmation": answer.requires_user_confirmation,
        "output_files": output_files,
    }


def _persist_tuning_plan(workspace: Path, request: dict[str, Any], plan: dict[str, Any]) -> str:
    experiment = _experiment_from_command(list(request.get("command") or []))
    if not experiment:
        raise EmbeddedAdvisorError("tuning request does not bind an experiment")
    expected_round = int(request.get("round", 0) or 0)
    expected_algorithm = _algorithm_from_tuning_context(workspace, experiment, expected_round)
    payload = dict(plan)
    payload["round"] = expected_round
    payload["experiment"] = experiment
    tuning_cfg = _load_tuning_config(workspace, algorithm=expected_algorithm)
    try:
        normalized = validate_tuning_plan(
            payload,
            tuning_cfg,
            advisor_type="embedded_langgraph",
            expected_experiment=experiment,
            expected_round=expected_round,
            expected_algorithm=expected_algorithm,
            allowed_evidence=_diagnosis_evidence_from_tuning_context(workspace, experiment, expected_round),
        )
    except ValueError as exc:
        raise EmbeddedAdvisorError(f"embedded tuning plan violates bounds: {exc}") from exc
    relative = Path("modeling") / experiment / f"llm_tuning_plan_round_{expected_round}.json"
    store = WorkspaceStore(workspace)
    if not store.create_once(relative, normalized):
        existing = json.loads((workspace / relative).read_text(encoding="utf-8"))
        if existing != normalized:
            raise EmbeddedAdvisorError(f"tuning plan already exists with different content: {relative}")
    return relative.as_posix()


def _load_tuning_config(workspace: Path, *, algorithm: str = "lightgbm") -> dict[str, Any]:
    path = workspace / "configs_runtime" / "train.yaml"
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(payload, dict):
            return resolve_tuning_config(payload, algorithm=algorithm)
    return resolve_tuning_config({"training": {"mode": "llm_guided_tune"}}, algorithm=algorithm)


def _algorithm_from_tuning_context(workspace: Path, experiment: str, round_index: int) -> str:
    path = workspace / "modeling" / experiment / f"tuning_context_round_{round_index}.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            algorithm = str(payload.get("algorithm") or "")
            if algorithm:
                return algorithm
        except (OSError, json.JSONDecodeError):
            pass
    return "lightgbm"


def _diagnosis_evidence_from_tuning_context(workspace: Path, experiment: str, round_index: int) -> set[str] | None:
    path = workspace / "modeling" / experiment / f"tuning_context_round_{round_index}.json"
    if not path.exists():
        return None
    try:
        evidence = ((json.loads(path.read_text(encoding="utf-8")).get("deterministic_diagnosis") or {}).get("allowed_evidence"))
    except (OSError, json.JSONDecodeError):
        return None
    return {str(item) for item in evidence} if isinstance(evidence, list) else None


def _experiment_from_command(command: list[str]) -> str:
    for index, value in enumerate(command):
        if value == "--experiment" and index + 1 < len(command):
            return str(command[index + 1])
    return ""
