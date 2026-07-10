"""Standard Advisor request/response file protocol for RMW Agent."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ADVISOR_PROTOCOL_VERSION = 1
REQUEST_TYPES = {
    "tuning_plan_required",
    "failure_diagnosis_required",
    "data_gap_decision_required",
    "product_decision_required",
}
RESPONSE_TYPES = {"tuning_plan", "failure_diagnosis", "data_gap_decision", "product_decision"}
REQUEST_TO_RESPONSE = {
    "tuning_plan_required": "tuning_plan",
    "failure_diagnosis_required": "failure_diagnosis",
    "data_gap_decision_required": "data_gap_decision",
    "product_decision_required": "product_decision",
}


def advisor_requests_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "advisor_requests"


def advisor_responses_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "advisor_responses"


def create_advisor_request(
    workspace: str | Path,
    *,
    project_dir: str | Path,
    version_id: str,
    task: dict[str, Any],
    reason: str,
    message: str = "",
    request_type: str | None = None,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    task_id = str(task.get("task_id") or "task")
    resolved_type = request_type or infer_request_type(task, reason)
    expected_type = REQUEST_TO_RESPONSE[resolved_type]
    command = list((task.get("command") or {}).get("args") or [])
    digest = _command_hash(command)
    request_id = _request_id(task_id, resolved_type, digest)
    path = advisor_requests_dir(workspace_path) / f"{request_id}.json"
    request = {
        "version": ADVISOR_PROTOCOL_VERSION,
        "request_id": request_id,
        "type": resolved_type,
        "status": "pending",
        "project": str(project_dir),
        "version_id": version_id,
        "task_id": task_id,
        "action_id": task.get("action_id", ""),
        "tool_name": task.get("tool_name", ""),
        "reason": reason,
        "message": message,
        "question": _question_for(resolved_type, task, message),
        "command": command,
        "command_hash": digest,
        "context_files": _context_files_for(resolved_type, task),
        "expected_response": {"type": expected_type},
        "constraints": _constraints_for(resolved_type),
        "created_at": _now(),
        "path": str(path.relative_to(workspace_path)),
        "accepted_response": "",
        "answered_at": "",
    }
    errors = validate_advisor_request(request)
    if errors:
        raise ValueError("; ".join(errors))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return request


def infer_request_type(task: dict[str, Any], reason: str) -> str:
    action_id = str(task.get("action_id") or "")
    if reason == "advisor_required" and action_id == "train_baseline":
        return "tuning_plan_required"
    if reason in {"data_missing", "dependency_missing", "artifact_contract_failed", "scaffold_only"}:
        return "data_gap_decision_required"
    if reason in {"unknown", "transient_io"}:
        return "failure_diagnosis_required"
    return "failure_diagnosis_required"


def list_advisor_requests(workspace: str | Path) -> list[dict[str, Any]]:
    rows = []
    directory = advisor_requests_dir(workspace)
    if not directory.exists():
        return []
    for path in sorted(directory.glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def load_advisor_request(workspace: str | Path, request_id: str) -> dict[str, Any]:
    path = advisor_requests_dir(workspace) / f"{request_id}.json"
    if not path.exists():
        raise KeyError(f"unknown advisor request: {request_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def advisor_request_is_answered(workspace: str | Path, request_id: str) -> bool:
    try:
        request = load_advisor_request(workspace, request_id)
    except KeyError:
        return False
    response_path = str(request.get("accepted_response") or "")
    return bool(request.get("status") == "answered" and response_path and (Path(workspace) / response_path).exists())


def validate_advisor_request(request: dict[str, Any]) -> list[str]:
    required = [
        "version",
        "request_id",
        "type",
        "status",
        "project",
        "version_id",
        "task_id",
        "action_id",
        "tool_name",
        "reason",
        "question",
        "command",
        "command_hash",
        "expected_response",
        "created_at",
    ]
    errors = _missing_errors(request, required)
    if request.get("type") not in REQUEST_TYPES:
        errors.append(f"unknown advisor request type: {request.get('type')}")
    if request.get("status") not in {"pending", "answered", "rejected"}:
        errors.append(f"unknown advisor request status: {request.get('status')}")
    expected = request.get("expected_response") if isinstance(request.get("expected_response"), dict) else {}
    if expected.get("type") not in RESPONSE_TYPES:
        errors.append("expected_response.type is required")
    if not isinstance(request.get("command"), list):
        errors.append("command must be an argv-style list")
    return errors


def validate_advisor_response(workspace: str | Path, response: dict[str, Any]) -> list[str]:
    required = ["version", "request_id", "type", "status", "decision", "summary"]
    errors = _missing_errors(response, required)
    if response.get("type") not in RESPONSE_TYPES:
        errors.append(f"unknown advisor response type: {response.get('type')}")
    if response.get("status") not in {"answered", "rejected"}:
        errors.append(f"unknown advisor response status: {response.get('status')}")
    if response.get("decision") not in {"continue", "retry", "stop", "needs_user_confirmation"}:
        errors.append(f"unknown advisor response decision: {response.get('decision')}")
    request_id = str(response.get("request_id") or "")
    try:
        request = load_advisor_request(workspace, request_id)
    except KeyError:
        errors.append(f"unknown advisor request: {request_id}")
        request = {}
    expected_type = ((request.get("expected_response") or {}) if isinstance(request.get("expected_response"), dict) else {}).get("type")
    if expected_type and response.get("type") != expected_type:
        errors.append(f"response type mismatch: expected {expected_type}, got {response.get('type')}")
    for output in response.get("output_files") or []:
        if Path(str(output)).is_absolute():
            errors.append(f"output_files must be workspace-relative: {output}")
    return errors


def accept_advisor_response(workspace: str | Path, response_path: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace)
    source = Path(response_path)
    if not source.is_absolute():
        source = workspace_path / source
    response = json.loads(source.read_text(encoding="utf-8"))
    errors = validate_advisor_response(workspace_path, response)
    if errors:
        return {"accepted": False, "errors": errors}
    request_id = str(response["request_id"])
    request = load_advisor_request(workspace_path, request_id)
    target = advisor_responses_dir(workspace_path) / f"{request_id}.response.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    request["status"] = "answered" if response.get("status") == "answered" else "rejected"
    request["answered_at"] = _now()
    request["accepted_response"] = str(target.relative_to(workspace_path))
    _save_advisor_request(workspace_path, request)
    return {
        "accepted": request["status"] == "answered",
        "request_id": request_id,
        "response_path": str(target),
        "errors": [],
    }


def _save_advisor_request(workspace: Path, request: dict[str, Any]) -> Path:
    path = advisor_requests_dir(workspace) / f"{request['request_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _question_for(request_type: str, task: dict[str, Any], message: str) -> str:
    if request_type == "tuning_plan_required":
        return "Create a bounded tuning plan for the current training task from the emitted tuning context."
    if request_type == "data_gap_decision_required":
        return "Decide whether the Agent should stop, wait for missing data, or continue with explicit scaffold evidence."
    if request_type == "product_decision_required":
        return "Recommend a product decision and identify risks or user confirmation points."
    return f"Diagnose the task failure and recommend the next safe action. {message}".strip()


def _context_files_for(request_type: str, task: dict[str, Any]) -> list[str]:
    if request_type == "tuning_plan_required":
        return [
            "modeling/*/tuning_context_round_*.json",
            "configs_runtime/train.yaml",
            "feature_selection/final_features.txt",
        ]
    return [
        "audit/agent_state.yml",
        "agent_plan.yml",
        "version_state.yml",
        "audit/artifact_manifest.json",
    ]


def _constraints_for(request_type: str) -> list[str]:
    common = [
        "Do not bypass SQL/DP approval gates.",
        "Do not treat scaffold or imported evidence as locally complete.",
        "Return only workspace-relative output files.",
    ]
    if request_type == "tuning_plan_required":
        return [
            *common,
            "Do not optimize directly against OOT as the primary tuning target.",
            "Use 3-5 bounded candidates.",
            "Explain each candidate briefly.",
        ]
    return common


def _request_id(task_id: str, request_type: str, digest: str) -> str:
    safe_task = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in task_id).strip("_") or "task"
    return f"advisor_{safe_task}_{request_type}_{digest[:8]}"


def _command_hash(args: list[str]) -> str:
    payload = json.dumps(args, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _missing_errors(payload: dict[str, Any], required: list[str]) -> list[str]:
    errors = []
    for key in required:
        value = payload.get(key)
        if key not in payload or value is None or value == "":
            errors.append(f"missing required field: {key}")
    return errors


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
