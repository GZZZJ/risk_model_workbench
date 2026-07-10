"""Approval ledger for high-risk Agent actions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from risk_model_workbench.agent.workspace_store import WorkspaceStore, tracked_payload


APPROVALS_VERSION = 1
APPROVAL_STATUSES = {"pending", "approved", "rejected", "revoked", "consumed"}


class ApprovalBindingError(ValueError):
    """The externally submitted SQL cannot be proven to match one consumed approval."""


def approvals_path(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "approvals.yml"


def command_hash(args: list[str]) -> str:
    payload = json.dumps(list(args), ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approval_id_for(args: list[str]) -> str:
    return f"approval_{command_hash(args)[:12]}"


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_approval_subject(
    workspace: str | Path,
    *,
    project: str,
    version_id: str,
    task_id: str,
    invocation_hash: str,
    operation_id: str,
) -> dict[str, Any]:
    workspace_path = Path(workspace)
    sql_manifest = workspace_path / "queries" / "sql_evidence_manifest.json"
    manifest_hash = hashlib.sha256(sql_manifest.read_bytes()).hexdigest() if sql_manifest.exists() else canonical_hash({})
    sql_files: list[dict[str, str]] = []
    if sql_manifest.exists():
        manifest = json.loads(sql_manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("entries"), list):
            raise ValueError("invalid SQL evidence manifest")
        seen_paths: set[str] = set()
        expected_stage = operation_id.removesuffix("_execute").removesuffix("_prepare")
        for entry in manifest.get("entries", []) or []:
            if not isinstance(entry, dict):
                raise ValueError("invalid SQL evidence entry")
            if str(entry.get("stage") or "") != expected_stage:
                continue
            relative = str(entry.get("path") or "")
            if relative in seen_paths:
                raise ValueError(f"duplicate SQL evidence path: {relative}")
            seen_paths.add(relative)
            path = (workspace_path / relative).resolve()
            if not relative or workspace_path.resolve() not in path.parents or not path.is_file():
                raise ValueError(f"invalid SQL evidence path: {relative}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            recorded_digest = str(entry.get("sql_sha256") or "")
            if recorded_digest and recorded_digest != digest:
                raise ValueError(f"SQL evidence hash mismatch: {relative}")
            sql_files.append({"path": relative, "sha256": digest})
    config_rows: list[dict[str, str]] = []
    for directory in ("configs_runtime", "configs_snapshot"):
        root = workspace_path / directory
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            config_rows.append(
                {
                    "path": str(path.relative_to(workspace_path)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    if not config_rows:
        raise ValueError("approval subject requires a config snapshot")
    return {
        "project": project,
        "version_id": version_id,
        "task_id": task_id,
        "invocation_hash": invocation_hash,
        "config_snapshot_hash": canonical_hash(config_rows),
        "sql_evidence_manifest_hash": manifest_hash,
        "sql_files": sorted(sql_files, key=lambda item: item["path"]),
        "operation_id": operation_id,
    }


def subject_hash(subject: dict[str, Any]) -> str:
    _validate_subject(subject)
    return canonical_hash(subject)


def load_approvals(workspace: str | Path) -> dict[str, Any]:
    path = approvals_path(workspace)
    if not path.exists():
        payload = tracked_payload(WorkspaceStore(workspace).read_yaml("audit/approvals.yml"))
        payload.update({"version": APPROVALS_VERSION, "approvals": []})
        return payload
    payload = tracked_payload(WorkspaceStore(workspace).read_yaml("audit/approvals.yml"))
    payload.setdefault("version", APPROVALS_VERSION)
    payload.setdefault("approvals", [])
    return payload


def save_approvals(workspace: str | Path, payload: dict[str, Any]) -> Path:
    path = approvals_path(workspace)
    payload["version"] = APPROVALS_VERSION
    payload.setdefault("approvals", [])
    payload["updated_at"] = _now()
    store = WorkspaceStore(workspace)
    expected_revision = getattr(payload, "store_revision", store.read_yaml("audit/approvals.yml").revision)
    revision = store.write_yaml("audit/approvals.yml", dict(payload), expected_revision)
    if hasattr(payload, "store_revision"):
        payload.store_revision = revision
    return path


def ensure_approval_request(workspace: str | Path, task: dict[str, Any], *, reason: str) -> dict[str, Any]:
    args = list((task.get("command") or {}).get("args") or [])
    approval_id = approval_id_for(args)
    digest = command_hash(args)
    payload = load_approvals(workspace)
    approvals = list(payload.get("approvals") or [])
    for item in approvals:
        if item.get("approval_id") == approval_id:
            return item
    item = {
        "approval_id": approval_id,
        "status": "pending",
        "task_id": task.get("task_id", ""),
        "tool_name": task.get("tool_name", ""),
        "permission": task.get("permission", ""),
        "reason": reason,
        "command": args,
        "command_hash": digest,
        "created_at": _now(),
        "approved_by": "",
        "approved_at": "",
        "note": "",
    }
    approvals.append(item)
    payload["approvals"] = approvals
    save_approvals(workspace, payload)
    return item


def ensure_subject_approval(
    workspace: str | Path,
    subject: dict[str, Any],
    *,
    reason: str,
) -> dict[str, Any]:
    digest = subject_hash(subject)
    approval_id_base = f"approval_{digest[:16]}"
    payload = load_approvals(workspace)
    approvals = list(payload.get("approvals") or [])
    for item in approvals:
        if item.get("subject_hash") == digest and item.get("status") != "revoked":
            return dict(item)
    prior_revisions = sum(1 for item in approvals if item.get("subject_hash") == digest)
    approval_id = approval_id_base if prior_revisions == 0 else f"{approval_id_base}_r{prior_revisions + 1}"
    now = _now()
    for item in approvals:
        same_operation = (
            item.get("task_id") == subject["task_id"]
            and item.get("operation_id") == subject["operation_id"]
        )
        if same_operation and item.get("status") in {"pending", "approved"} and item.get("subject_hash") != digest:
            item["status"] = "revoked"
            item["revoked_at"] = now
            item["revoke_reason"] = "approval_subject_drift"
    item = {
        "approval_id": approval_id,
        "status": "pending",
        "task_id": subject["task_id"],
        "operation_id": subject["operation_id"],
        "subject": subject,
        "subject_hash": digest,
        "reason": reason,
        "created_at": now,
        "approved_by": "",
        "approved_at": "",
        "rejected_by": "",
        "rejected_at": "",
        "consumed_at": "",
        "note": "",
    }
    approvals.append(item)
    payload["approvals"] = approvals
    save_approvals(workspace, payload)
    return dict(item)


def approval_for_command(workspace: str | Path, args: list[str]) -> dict[str, Any] | None:
    digest = command_hash(args)
    for item in load_approvals(workspace).get("approvals", []) or []:
        if item.get("command_hash") == digest:
            return dict(item)
    return None


def is_command_approved(workspace: str | Path, args: list[str]) -> bool:
    approval = approval_for_command(workspace, args)
    return bool(approval and approval.get("status") == "approved")


def approve_request(workspace: str | Path, approval_id: str, *, approved_by: str, note: str = "") -> dict[str, Any]:
    payload = load_approvals(workspace)
    approvals = list(payload.get("approvals") or [])
    for item in approvals:
        if item.get("approval_id") == approval_id:
            if item.get("status") != "pending":
                raise ValueError(f"approval is not pending: {approval_id}")
            item["status"] = "approved"
            item["approved_by"] = approved_by
            item["approved_at"] = _now()
            item["note"] = note
            payload["approvals"] = approvals
            save_approvals(workspace, payload)
            return item
    raise KeyError(f"unknown approval_id: {approval_id}")


def reject_request(
    workspace: str | Path,
    approval_id: str,
    *,
    rejected_by: str,
    note: str = "",
) -> dict[str, Any]:
    payload = load_approvals(workspace)
    for item in payload.get("approvals", []) or []:
        if item.get("approval_id") != approval_id:
            continue
        if item.get("status") not in {"pending", "approved"}:
            raise ValueError(f"approval cannot be rejected from {item.get('status')}: {approval_id}")
        item["status"] = "rejected"
        item["rejected_by"] = rejected_by
        item["rejected_at"] = _now()
        item["note"] = note
        save_approvals(workspace, payload)
        return dict(item)
    raise KeyError(f"unknown approval_id: {approval_id}")


def revoke_approval(workspace: str | Path, approval_id: str, *, reason: str) -> dict[str, Any]:
    payload = load_approvals(workspace)
    for item in payload.get("approvals", []) or []:
        if item.get("approval_id") != approval_id:
            continue
        if item.get("status") not in {"pending", "approved"}:
            return dict(item)
        item["status"] = "revoked"
        item["revoked_at"] = _now()
        item["revoke_reason"] = reason
        save_approvals(workspace, payload)
        return dict(item)
    raise KeyError(f"unknown approval_id: {approval_id}")


def consume_approval(
    workspace: str | Path,
    approval_id: str,
    subject: dict[str, Any],
    *,
    consumed_by: str,
) -> dict[str, Any]:
    payload = load_approvals(workspace)
    approval = next(
        (item for item in payload.get("approvals", []) or [] if item.get("approval_id") == approval_id),
        None,
    )
    if approval is None:
        raise KeyError(f"unknown approval_id: {approval_id}")
    if approval.get("status") == "consumed":
        raise ValueError(f"approval already consumed: {approval_id}")
    if approval.get("status") != "approved":
        raise ValueError(f"approval is not approved: {approval_id}")
    current_hash = subject_hash(subject)
    if approval.get("subject_hash") != current_hash:
        approval["status"] = "revoked"
        approval["revoked_at"] = _now()
        approval["revoke_reason"] = "approval_subject_drift"
        save_approvals(workspace, payload)
        raise ValueError(f"approval subject drift: {approval_id}")
    receipt = {
        "receipt_id": f"approval_consumption_{approval_id}",
        "approval_id": approval_id,
        "blocker_id": approval_id,
        "subject_hash": current_hash,
        "consumed": True,
        "consumed_by": consumed_by,
        "consumed_at": _now(),
    }
    relative = Path("audit") / "approval_consumptions" / f"{approval_id}.json"
    if not WorkspaceStore(workspace).create_once(relative, receipt):
        raise ValueError(f"approval already consumed: {approval_id}")
    approval["status"] = "consumed"
    approval["consumed_at"] = receipt["consumed_at"]
    approval["consumption_receipt"] = str(relative)
    save_approvals(workspace, payload)
    return receipt


def approval_by_id(workspace: str | Path, approval_id: str) -> dict[str, Any] | None:
    for item in load_approvals(workspace).get("approvals", []) or []:
        if item.get("approval_id") == approval_id:
            return dict(item)
    return None


def is_subject_approval_consumed(workspace: str | Path, subject: dict[str, Any]) -> bool:
    digest = subject_hash(subject)
    return any(
        item.get("subject_hash") == digest and item.get("status") == "consumed"
        for item in load_approvals(workspace).get("approvals", []) or []
    )


def consumed_subject_hash_for_operation(workspace: str | Path, operation_id: str) -> str:
    for item in reversed(load_approvals(workspace).get("approvals", []) or []):
        if item.get("operation_id") == operation_id and item.get("status") == "consumed":
            return str(item.get("subject_hash") or "")
    return ""


def approved_sql_for_consumed_operation(
    workspace: str | Path,
    *,
    approval_id: str,
    subject_hash_value: str,
    consumption_receipt: str,
    parent_operation_id: str,
    candidate_sql: str,
) -> str:
    """Return the immutable approved SQL text matching a generated candidate.

    Validation happens immediately before the external client call. The returned
    bytes come from the version-scoped evidence file, not from regenerated SQL.
    """
    workspace_path = Path(workspace).resolve()
    approval = approval_by_id(workspace_path, approval_id)
    if approval is None or approval.get("status") != "consumed":
        raise ApprovalBindingError("consumed approval record missing")
    if approval.get("subject_hash") != subject_hash_value:
        raise ApprovalBindingError("consumed approval subject mismatch")
    subject = approval.get("subject") if isinstance(approval.get("subject"), dict) else {}
    try:
        _validate_subject(subject)
    except ValueError as exc:
        raise ApprovalBindingError(str(exc)) from exc
    if subject_hash(subject) != subject_hash_value:
        raise ApprovalBindingError("consumed approval subject drift")
    if subject.get("operation_id") != parent_operation_id:
        raise ApprovalBindingError("consumed approval operation mismatch")
    if str(approval.get("consumption_receipt") or "") != str(consumption_receipt):
        raise ApprovalBindingError("consumed approval receipt reference mismatch")

    receipt_path = (workspace_path / consumption_receipt).resolve()
    if workspace_path not in receipt_path.parents or not receipt_path.is_file():
        raise ApprovalBindingError("approval consumption receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        receipt.get("approval_id") != approval_id
        or receipt.get("subject_hash") != subject_hash_value
        or receipt.get("consumed") is not True
    ):
        raise ApprovalBindingError("approval consumption receipt mismatch")

    canonical_candidate = _canonical_sql(candidate_sql)
    for item in subject.get("sql_files", []) or []:
        if not isinstance(item, dict):
            continue
        path = (workspace_path / str(item.get("path") or "")).resolve()
        if workspace_path not in path.parents or not path.is_file():
            raise ApprovalBindingError("approved SQL evidence missing")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.get("sha256"):
            raise ApprovalBindingError("approved SQL evidence drift")
        approved_text = raw.decode("utf-8")
        if _canonical_sql(approved_text) == canonical_candidate:
            return approved_text
    raise ApprovalBindingError("candidate SQL is not present in the consumed approval subject")


def _canonical_sql(value: str) -> str:
    return " ".join(value.split())


def _validate_subject(subject: dict[str, Any]) -> None:
    required = {
        "project",
        "version_id",
        "task_id",
        "invocation_hash",
        "config_snapshot_hash",
        "sql_evidence_manifest_hash",
        "sql_files",
        "operation_id",
    }
    if set(subject) != required:
        raise ValueError("invalid approval subject fields")
    for name in required - {"sql_files"}:
        if not isinstance(subject.get(name), str) or not str(subject.get(name) or "").strip():
            raise ValueError(f"invalid approval subject field: {name}")
    if not isinstance(subject.get("sql_files"), list) or not subject["sql_files"]:
        raise ValueError("approval subject requires SQL files")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
