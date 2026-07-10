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
REQUEST_STATUSES = {"pending", "answered", "rejected", "consumed", "waiting_for_user"}
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


def advisor_contexts_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "advisor_contexts"


def advisor_consumptions_dir(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "advisor_consumptions"


def create_advisor_request(
    workspace: str | Path,
    *,
    project_dir: str | Path,
    version_id: str,
    task: dict[str, Any],
    reason: str,
    message: str = "",
    request_type: str | None = None,
    attempt_id: str = "",
    invocation_hash: str = "",
    round_index: int | None = None,
    retry_index: int = 0,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    task_id = str(task.get("task_id") or "task")
    resolved_type = request_type or infer_request_type(task, reason)
    expected_type = REQUEST_TO_RESPONSE[resolved_type]
    command = list((task.get("command") or {}).get("args") or [])
    command_digest = _command_hash(command)
    resolved_attempt_id = str(attempt_id or task.get("attempt_id") or f"attempt_{task_id}")
    resolved_invocation_hash = str(invocation_hash or task.get("invocation_hash") or command_digest)
    resolved_round = _round_for(workspace_path, resolved_type, task, round_index)
    context_files = _context_files_for(resolved_type, task)
    context_entries = build_context_manifest(workspace_path, context_files)
    context_hash = context_manifest_hash(context_entries)
    identity = {
        "task_id": task_id,
        "attempt_id": resolved_attempt_id,
        "request_type": resolved_type,
        "invocation_hash": resolved_invocation_hash,
        "context_hash": context_hash,
        "round": resolved_round,
        "retry_index": int(retry_index),
    }
    request_hash = canonical_hash(identity)
    request_id = _request_id(task_id, resolved_type, request_hash)
    path = advisor_requests_dir(workspace_path) / f"{request_id}.json"
    context_path = advisor_contexts_dir(workspace_path) / f"{request_id}.json"
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
        "attempt_id": resolved_attempt_id,
        "invocation_hash": resolved_invocation_hash,
        "context_hash": context_hash,
        "context_manifest": str(context_path.relative_to(workspace_path)),
        "round": resolved_round,
        "request_hash": request_hash,
        "request_retry_index": int(retry_index),
        "reason": reason,
        "message": message,
        "question": _question_for(resolved_type, task, message),
        "command": command,
        "command_hash": command_digest,
        "context_files": context_files,
        "expected_response": {"type": expected_type},
        "constraints": _constraints_for(resolved_type),
        "retry_budget": {"max_retries": 2},
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
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(
        json.dumps(
            {
                "version": ADVISOR_PROTOCOL_VERSION,
                "request_id": request_id,
                "context_hash": context_hash,
                "patterns": context_files,
                "files": context_entries,
                "created_at": _now(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return request


def create_replacement_advisor_request(workspace: str | Path, request: dict[str, Any], *, reason: str) -> dict[str, Any]:
    workspace_path = Path(workspace)
    retry_index = int(request.get("request_retry_index", 0) or 0) + 1
    identity = {
        "task_id": str(request.get("task_id") or ""),
        "attempt_id": str(request.get("attempt_id") or ""),
        "request_type": str(request.get("type") or ""),
        "invocation_hash": str(request.get("invocation_hash") or ""),
        "context_hash": str(request.get("context_hash") or ""),
        "round": int(request.get("round", 0) or 0),
        "retry_index": retry_index,
    }
    context_entries = build_context_manifest(workspace_path, list(request.get("context_files") or []))
    identity["context_hash"] = context_manifest_hash(context_entries)
    request_hash = canonical_hash(identity)
    request_id = _request_id(str(request.get("task_id") or "task"), str(request.get("type") or "advisor"), request_hash)
    replacement = dict(request)
    replacement.update(
        {
            "request_id": request_id,
            "status": "pending",
            "reason": reason,
            "context_hash": identity["context_hash"],
            "request_hash": request_hash,
            "request_retry_index": retry_index,
            "path": str(Path("audit") / "advisor_requests" / f"{request_id}.json"),
            "context_manifest": str(Path("audit") / "advisor_contexts" / f"{request_id}.json"),
            "accepted_response": "",
            "answered_at": "",
            "replacement_of": request.get("request_id", ""),
            "created_at": _now(),
        }
    )
    _save_advisor_request(workspace_path, replacement)
    context_path = workspace_path / replacement["context_manifest"]
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(
        json.dumps(
            {
                "version": ADVISOR_PROTOCOL_VERSION,
                "request_id": replacement["request_id"],
                "context_hash": replacement["context_hash"],
                "patterns": replacement.get("context_files") or [],
                "files": context_entries,
                "created_at": _now(),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return replacement


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
        "attempt_id",
        "invocation_hash",
        "context_hash",
        "context_manifest",
        "round",
        "request_hash",
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
    if request.get("status") not in REQUEST_STATUSES:
        errors.append(f"unknown advisor request status: {request.get('status')}")
    expected = request.get("expected_response") if isinstance(request.get("expected_response"), dict) else {}
    if expected.get("type") not in RESPONSE_TYPES:
        errors.append("expected_response.type is required")
    if not isinstance(request.get("command"), list):
        errors.append("command must be an argv-style list")
    return errors


def validate_advisor_response(workspace: str | Path, response: dict[str, Any]) -> list[str]:
    required = ["version", "request_id", "type", "status", "decision", "summary", "request_identity"]
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
    identity = response.get("request_identity") if isinstance(response.get("request_identity"), dict) else {}
    for field in ["task_id", "attempt_id", "invocation_hash", "context_hash", "round"]:
        expected = request.get(field)
        actual = identity.get(field)
        if field == "round":
            try:
                expected = int(expected or 0)
                actual = int(actual or 0)
            except (TypeError, ValueError):
                errors.append("response identity mismatch: round")
                continue
        if request and actual != expected:
            errors.append(f"response identity mismatch: {field}")
    output_files = response.get("output_files") or []
    if not isinstance(output_files, list):
        errors.append("output_files must be a list")
        output_files = []
    for output in output_files:
        errors.extend(_validate_workspace_output_path(Path(workspace), str(output)))
    return errors


def accept_advisor_response(workspace: str | Path, response_path: str | Path) -> dict[str, Any]:
    workspace_path = Path(workspace)
    source = Path(response_path)
    if not source.is_absolute():
        source = workspace_path / source
    response = json.loads(source.read_text(encoding="utf-8"))
    errors = validate_advisor_response(workspace_path, response)
    if errors:
        return {"accepted": False, "stored": False, "errors": errors}
    request_id = str(response["request_id"])
    request = load_advisor_request(workspace_path, request_id)
    if request.get("status") != "pending":
        return {
            "accepted": False,
            "stored": False,
            "request_id": request_id,
            "errors": [f"advisor request is not pending: {request.get('status')}"],
        }
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
        "stored": True,
        "request_id": request_id,
        "response_path": str(target),
        "errors": [],
    }


def load_accepted_advisor_response(workspace: str | Path, request: dict[str, Any]) -> dict[str, Any]:
    response_path = str(request.get("accepted_response") or "")
    if not response_path:
        raise FileNotFoundError(f"advisor response not accepted: {request.get('request_id')}")
    return json.loads((Path(workspace) / response_path).read_text(encoding="utf-8"))


def current_context_hash_for_request(workspace: str | Path, request: dict[str, Any]) -> str:
    return context_manifest_hash(build_context_manifest(Path(workspace), list(request.get("context_files") or [])))


def request_identity(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(request.get("task_id") or ""),
        "attempt_id": str(request.get("attempt_id") or ""),
        "invocation_hash": str(request.get("invocation_hash") or ""),
        "context_hash": str(request.get("context_hash") or ""),
        "round": int(request.get("round", 0) or 0),
    }


def _save_advisor_request(workspace: Path, request: dict[str, Any]) -> Path:
    path = advisor_requests_dir(workspace) / f"{request['request_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def build_context_manifest(workspace: Path, patterns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    workspace_resolved = workspace.resolve()
    for pattern in patterns:
        matches = sorted(path for path in workspace.glob(pattern) if path.is_file())
        if not matches and not _has_glob(pattern):
            matches = [workspace / pattern]
        if not matches:
            rows.append({"path": pattern, "sha256": "", "size": 0, "missing": True})
            continue
        for path in matches:
            resolved = path.resolve()
            if resolved != workspace_resolved and workspace_resolved not in resolved.parents:
                rows.append({"path": pattern, "sha256": "", "size": 0, "missing": True, "error": "path_escape"})
                continue
            if not resolved.exists():
                rows.append({"path": str(path.relative_to(workspace)), "sha256": "", "size": 0, "missing": True})
                continue
            rows.append(
                {
                    "path": str(resolved.relative_to(workspace_resolved)),
                    "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
                    "size": resolved.stat().st_size,
                    "missing": False,
                }
            )
    return sorted(rows, key=lambda item: str(item.get("path") or ""))


def context_manifest_hash(entries: list[dict[str, Any]]) -> str:
    return canonical_hash(entries)


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
    return f"advisor_{safe_task}_{request_type}_{digest[:12]}"


def _round_for(workspace: Path, request_type: str, task: dict[str, Any], round_index: int | None) -> int:
    if round_index is not None:
        return int(round_index)
    for key in ("round", "advisor_round"):
        if task.get(key) not in {None, ""}:
            return int(task[key])
    if request_type == "tuning_plan_required":
        rounds: list[int] = []
        for path in workspace.glob("modeling/*/tuning_context_round_*.json"):
            try:
                rounds.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return max(rounds) if rounds else 1
    return 0


def _validate_workspace_output_path(workspace: Path, output: str) -> list[str]:
    relative = Path(output)
    if relative.is_absolute():
        return [f"output_files must be workspace-relative: {output}"]
    if ".." in relative.parts:
        return [f"output_files path escape: {output}"]
    workspace_resolved = workspace.resolve()
    path = (workspace / relative).resolve()
    if path != workspace_resolved and workspace_resolved not in path.parents:
        return [f"output_files path escape: {output}"]
    if not path.exists():
        return [f"missing output file: {output}"]
    if path.is_symlink():
        target = path.resolve()
        if target != workspace_resolved and workspace_resolved not in target.parents:
            return [f"output_files path escape: {output}"]
    return []


def _has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in "*?[")


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
