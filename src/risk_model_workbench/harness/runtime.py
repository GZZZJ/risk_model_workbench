"""Runtime helpers for recording harnessed stage action execution."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from risk_model_workbench.harness.actions import ActionSpec, automatic_recovery_allowed, get_action_spec
from risk_model_workbench.harness.errors import (
    DATA_MISSING,
    DEPENDENCY_MISSING,
    EXTERNAL_OUTCOME_UNKNOWN,
    FAILURE_CODES,
    SCAFFOLD_ONLY,
    SQL_APPROVAL_REQUIRED,
    TRANSIENT_IO,
    UNKNOWN,
    DuplicateActionResultError,
    InvalidActionResultError,
    MissingActionResultError,
    get_failure_class,
)
from risk_model_workbench.registry import load_artifact_manifest, register_artifact as registry_register_artifact
from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.state import (
    load_run_state,
    mark_stage_done,
    mark_stage_failed,
    mark_stage_started,
    register_artifact as state_register_artifact,
    save_run_state,
)


T = TypeVar("T")


@dataclass(frozen=True)
class ActionResult:
    schema_version: int = 1
    attempt_id: str = ""
    task_id: str = ""
    action_id: str = ""
    invocation_hash: str = ""
    project: str = ""
    version_id: str = ""
    status: str = ""
    failure_code: str = ""
    next_required_action: str = "none"
    retryable: bool = False
    scaffold: bool = False
    message: str = ""
    created_at: str = ""
    retry_count: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionAttempt:
    workspace: Path
    attempt_id: str
    task_id: str
    action_id: str
    invocation_hash: str
    project: str
    version_id: str
    approval_id: str = ""
    approval_subject_hash: str = ""
    approval_consumption_receipt: str = ""
    parent_operation_id: str = ""


_CURRENT_ATTEMPT: ContextVar[ActionAttempt | None] = ContextVar("rmw_action_attempt", default=None)


@contextmanager
def action_attempt(attempt: ActionAttempt) -> Iterator[None]:
    token = _CURRENT_ATTEMPT.set(attempt)
    try:
        yield
    finally:
        _CURRENT_ATTEMPT.reset(token)


def current_action_attempt() -> ActionAttempt | None:
    """Return the active Agent attempt without granting callers mutation authority."""
    return _CURRENT_ATTEMPT.get()


def action_result_path(workspace: str | Path, attempt_id: str) -> Path:
    _validate_attempt_id(attempt_id)
    return WorkspaceStore(workspace).path(Path("audit") / "action_results" / f"{attempt_id}.json")


def write_action_result(workspace: str | Path, result: ActionResult) -> Path:
    payload = result.to_dict()
    _validate_action_result_payload(payload)
    relative = Path("audit") / "action_results" / f"{result.attempt_id}.json"
    store = WorkspaceStore(workspace)
    if not store.create_once(relative, payload):
        raise DuplicateActionResultError(f"ActionResult already exists for attempt {result.attempt_id}")
    path = Path(workspace) / relative
    registry_register_artifact(
        workspace,
        path,
        stage="agent_runtime",
        kind="audit",
        source="generated",
        description=f"Attempt-scoped ActionResult for {result.task_id}",
    )
    return path


def load_action_result(
    workspace: str | Path,
    attempt_id: str,
    *,
    task_id: str = "",
    action_id: str = "",
    invocation_hash: str = "",
    project: str = "",
    version_id: str = "",
) -> ActionResult:
    path = action_result_path(workspace, attempt_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise MissingActionResultError(f"missing ActionResult for attempt {attempt_id}") from exc
    except (OSError, UnicodeError) as exc:
        raise InvalidActionResultError(f"unreadable ActionResult for attempt {attempt_id}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise InvalidActionResultError(f"malformed ActionResult for attempt {attempt_id}: {exc}") from exc
    _validate_action_result_payload(payload)
    expected = {
        "attempt_id": attempt_id,
        "task_id": task_id,
        "action_id": action_id,
        "invocation_hash": invocation_hash,
        "project": project,
        "version_id": version_id,
    }
    mismatches = [name for name, value in expected.items() if value and payload.get(name) != value]
    if mismatches:
        raise InvalidActionResultError("ActionResult subject mismatch: " + ",".join(mismatches))
    return ActionResult(**payload)


def _validate_action_result_payload(payload: object) -> None:
    if not isinstance(payload, dict):
        raise InvalidActionResultError("ActionResult must be an object")
    required = {
        "schema_version": int,
        "attempt_id": str,
        "task_id": str,
        "action_id": str,
        "invocation_hash": str,
        "project": str,
        "version_id": str,
        "status": str,
    }
    allowed = set(ActionResult.__dataclass_fields__)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise InvalidActionResultError("unknown ActionResult fields: " + ",".join(unknown))
    for name, expected_type in required.items():
        value = payload.get(name)
        valid_type = type(value) is int if expected_type is int else isinstance(value, expected_type)
        if not valid_type or (expected_type is str and not value.strip()):
            raise InvalidActionResultError(f"invalid ActionResult field: {name}")
    if payload["schema_version"] != 1:
        raise InvalidActionResultError("unsupported ActionResult schema_version")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", payload["attempt_id"]):
        raise InvalidActionResultError("invalid ActionResult attempt_id")
    if payload["status"] not in {"done", "scaffold", "failed", "review_ready"}:
        raise InvalidActionResultError(f"invalid ActionResult status: {payload['status']}")
    if payload.get("next_required_action", "none") not in {"none", "approval", "advisor", "user", "reconciliation"}:
        raise InvalidActionResultError("invalid ActionResult next_required_action")
    optional_types = {
        "failure_code": str,
        "retryable": bool,
        "scaffold": bool,
        "message": str,
        "created_at": str,
        "retry_count": int,
        "artifacts": list,
    }
    for name, expected_type in optional_types.items():
        value = payload.get(name)
        valid_type = type(value) is expected_type if expected_type in {bool, int} else isinstance(value, expected_type)
        if not valid_type:
            raise InvalidActionResultError(f"invalid ActionResult field: {name}")
    if not payload["created_at"].strip():
        raise InvalidActionResultError("invalid ActionResult field: created_at")
    if any(not isinstance(item, dict) for item in payload["artifacts"]):
        raise InvalidActionResultError("invalid ActionResult artifacts")
    if payload.get("decision") is not None and not isinstance(payload.get("decision"), dict):
        raise InvalidActionResultError("invalid ActionResult decision")
    if bool(payload["scaffold"]) != (payload["status"] == "scaffold"):
        raise InvalidActionResultError("ActionResult scaffold flag conflicts with status")


def register_action_artifact(
    run_path: str | Path,
    action_id: str,
    artifact: str | Path,
    *,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
    storage_class: str | None = None,
    contract_role: str = "required",
    retention_reason: str = "",
    regeneration: str = "",
    external_reference: str = "",
    integrity_mode: str = "content",
) -> dict[str, Any]:
    """Register an artifact through the action harness."""
    spec = _require_stage_action(action_id)
    entry = state_register_artifact(
        run_path,
        str(spec.stage),
        artifact,
        kind=kind,
        source=source,
        description=description,
        storage_class=storage_class,
        contract_role=contract_role,
        retention_reason=retention_reason,
        regeneration=regeneration,
        external_reference=external_reference,
        integrity_mode=integrity_mode,
    )
    state = load_run_state(run_path)
    stage_state = state.setdefault("stages", {}).setdefault(str(spec.stage), {"status": "pending", "artifacts": []})
    stage_state["action"] = _action_metadata(spec)
    if isinstance(stage_state.get("last_result"), dict):
        stage_state["last_result"]["artifacts"] = _stage_artifact_results(run_path, state, str(spec.stage), spec)
    save_run_state(run_path, state)
    return _artifact_result(entry, spec)


def stage_action_started(run_path: str | Path, action_id: str) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    state = mark_stage_started(run_path, str(spec.stage))
    return _write_stage_action_metadata(run_path, state, spec, result=None)


def stage_action_done(
    run_path: str | Path,
    action_id: str,
    *,
    scaffold: bool = False,
    message: str = "",
    failure_code: str = "",
    retry_count: int = 0,
) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    state = mark_stage_done(run_path, str(spec.stage), scaffold=scaffold)
    result = ActionResult(
        status="scaffold" if scaffold else "done",
        scaffold=scaffold,
        failure_code=_normalize_failure_code(failure_code or (SCAFFOLD_ONLY if scaffold else "")),
        message=message,
        retry_count=retry_count,
        artifacts=_stage_artifact_results(run_path, state, str(spec.stage), spec),
        decision=_latest_stage_decision(state, str(spec.stage)),
    )
    result = _bind_current_attempt(result, spec)
    state = _write_stage_action_metadata(run_path, state, spec, result=result)
    _emit_action_progress(run_path, spec, result)
    return state


def stage_action_failed(
    run_path: str | Path,
    action_id: str,
    reason: str,
    *,
    failure_code: str = "",
    retry_count: int = 0,
) -> dict[str, Any]:
    spec = _require_stage_action(action_id)
    normalized = _normalize_failure_code(failure_code or classify_exception_message(reason))
    state = mark_stage_failed(run_path, str(spec.stage), reason)
    result = ActionResult(
        status="failed",
        failure_code=normalized,
        next_required_action="reconciliation" if normalized == EXTERNAL_OUTCOME_UNKNOWN else "none",
        message=reason,
        retry_count=retry_count,
        artifacts=_stage_artifact_results(run_path, state, str(spec.stage), spec),
        decision=_latest_stage_decision(state, str(spec.stage)),
    )
    result = _bind_current_attempt(result, spec)
    state = _write_stage_action_metadata(run_path, state, spec, result=result)
    _emit_action_progress(run_path, spec, result)
    return state


def classify_exception(exc: BaseException) -> str:
    if type(exc).__name__ == "ExternalOutcomeUnknown":
        return EXTERNAL_OUTCOME_UNKNOWN
    if isinstance(exc, (FileNotFoundError, KeyError, ValueError)):
        return DATA_MISSING
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return DEPENDENCY_MISSING
    if isinstance(exc, (TimeoutError, OSError)):
        return TRANSIENT_IO
    return classify_exception_message(str(exc))


def classify_exception_message(message: str) -> str:
    lowered = message.lower()
    if "external operation outcome is unknown" in lowered or "external_outcome_unknown" in lowered:
        return EXTERNAL_OUTCOME_UNKNOWN
    if "approval" in lowered or "approve" in lowered or "sql_review_required" in lowered:
        return SQL_APPROVAL_REQUIRED
    if "dependency" in lowered or "no module named" in lowered or "import" in lowered:
        return DEPENDENCY_MISSING
    if "not available" in lowered or "missing" in lowered or "not found" in lowered or "does not exist" in lowered:
        return DATA_MISSING
    if "timeout" in lowered or "temporar" in lowered or "transient" in lowered:
        return TRANSIENT_IO
    if "scaffold" in lowered:
        return SCAFFOLD_ONLY
    return UNKNOWN


def should_retry_failure(
    action_id: str,
    failure_code: str,
    *,
    attempt: int,
    max_attempts: int = 3,
    execution_semantics: str = "read_only",
) -> bool:
    spec = get_action_spec(action_id)
    code = _normalize_failure_code(failure_code)
    if attempt >= max_attempts:
        return False
    if not automatic_recovery_allowed(execution_semantics):
        return False
    if spec.retry_policy == "never":
        return False
    return get_failure_class(code).retryable


def run_with_retry(action_id: str, operation: Callable[[], T], *, max_attempts: int = 3) -> tuple[T, int]:
    """Run a safe operation according to the action retry policy.

    The helper never retries unknown or non-retryable failures, and write-stage
    actions keep ``retry_policy=never`` unless their ActionSpec explicitly says
    otherwise.
    """
    attempt = 1
    while True:
        try:
            return operation(), attempt - 1
        except Exception as exc:
            failure_code = classify_exception(exc)
            if not should_retry_failure(action_id, failure_code, attempt=attempt, max_attempts=max_attempts):
                raise
            attempt += 1


def _write_stage_action_metadata(
    run_path: str | Path,
    state: dict[str, Any],
    spec: ActionSpec,
    *,
    result: ActionResult | None,
) -> dict[str, Any]:
    stage_state = state.setdefault("stages", {}).setdefault(str(spec.stage), {"status": "pending", "artifacts": []})
    stage_state["action"] = _action_metadata(spec)
    if result is not None:
        stage_state["last_result"] = result.to_dict()
        if result.failure_code:
            stage_state["failure_code"] = result.failure_code
        elif stage_state.get("failure_code"):
            stage_state.pop("failure_code", None)
    save_run_state(run_path, state)
    return state


def _require_stage_action(action_id: str) -> ActionSpec:
    spec = get_action_spec(action_id)
    if spec.kind != "stage" or not spec.stage:
        raise ValueError(f"action is not a stage action: {action_id}")
    return spec


def _action_metadata(spec: ActionSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "kind": spec.kind,
        "approval_required": spec.approval_required,
        "approval_type": spec.approval_type,
        "retry_policy": spec.retry_policy,
        "expected_inputs": list(spec.inputs),
        "expected_outputs": list(spec.outputs),
        "artifact_rules": list(spec.artifact_rules),
    }


def _stage_artifact_results(
    run_path: str | Path,
    state: dict[str, Any],
    stage: str,
    spec: ActionSpec,
) -> list[dict[str, Any]]:
    stage_state = (state.get("stages") or {}).get(stage) or {}
    paths = [str(path) for path in stage_state.get("artifacts", [])]
    if not paths:
        return []
    manifest = load_artifact_manifest(run_path)
    manifest_by_path = {
        (str(item.get("stage")), str(item.get("path"))): item
        for item in manifest.get("artifacts", [])
        if item.get("path")
    }
    results = []
    for artifact_path in paths:
        entry = manifest_by_path.get((stage, artifact_path), {"path": artifact_path, "stage": stage})
        results.append(_artifact_result(entry, spec))
    return results


def _artifact_result(entry: dict[str, Any], spec: ActionSpec) -> dict[str, Any]:
    path = str(entry.get("path") or "")
    rule = _matching_artifact_rule(path, spec.artifact_rules)
    return {
        "path": path,
        "kind": entry.get("kind", "file"),
        "source": entry.get("source", ""),
        "exists": bool(entry.get("exists", False)),
        "description": entry.get("description", ""),
        "artifact_rule": rule,
        "rule_matched": bool(rule),
    }


def _matching_artifact_rule(path: str, rules: tuple[str, ...]) -> str:
    for rule in rules:
        if path == rule or fnmatch(path, rule):
            return rule
    return ""


def _latest_stage_decision(state: dict[str, Any], stage: str) -> dict[str, Any] | None:
    for item in reversed(list(state.get("decisions") or [])):
        if item.get("stage") == stage:
            return dict(item)
    return None


def _emit_action_progress(run_path: str | Path, spec: ActionSpec, result: ActionResult) -> None:
    try:
        from risk_model_workbench.progress import emit_progress, stage_label

        metrics = {
            "action_id": spec.id,
            "failure_code": result.failure_code,
            "retry_count": result.retry_count,
            "artifact_count": len(result.artifacts),
        }
        if result.decision:
            metrics["decision"] = result.decision.get("decision", "")
        emit_progress(
            run_path,
            str(spec.stage),
            step="action_failed" if result.status == "failed" else "action_done",
            status=result.status,
            message=result.message or f"{stage_label(str(spec.stage))}{'失败' if result.status == 'failed' else '完成'}",
            percent=None if result.status == "failed" else 100,
            metrics=metrics,
            level="error" if result.status == "failed" else "info",
            emit_terminal=False,
        )
    except Exception:
        return


def _normalize_failure_code(code: str) -> str:
    if not code:
        return ""
    return code if code in FAILURE_CODES else UNKNOWN


def _validate_attempt_id(attempt_id: str) -> None:
    if not isinstance(attempt_id, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", attempt_id):
        raise InvalidActionResultError("invalid ActionResult attempt_id")


def _bind_current_attempt(result: ActionResult, spec: ActionSpec) -> ActionResult:
    attempt = _CURRENT_ATTEMPT.get()
    if attempt is None:
        return result
    if attempt.action_id != spec.id:
        raise InvalidActionResultError(
            f"active attempt action mismatch: expected {attempt.action_id}, got {spec.id}"
        )
    next_required_action = result.next_required_action
    if result.failure_code == SQL_APPROVAL_REQUIRED:
        next_required_action = "approval"
    elif result.failure_code == "advisor_required":
        next_required_action = "advisor"
    bound = replace(
        result,
        schema_version=1,
        attempt_id=attempt.attempt_id,
        task_id=attempt.task_id,
        action_id=attempt.action_id,
        invocation_hash=attempt.invocation_hash,
        project=attempt.project,
        version_id=attempt.version_id,
        next_required_action=next_required_action,
        created_at=result.created_at or datetime.now().isoformat(timespec="seconds"),
    )
    write_action_result(attempt.workspace, bound)
    return bound
