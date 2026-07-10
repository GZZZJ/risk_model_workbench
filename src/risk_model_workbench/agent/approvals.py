"""Approval ledger for high-risk Agent actions."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


APPROVALS_VERSION = 1


def approvals_path(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "approvals.yml"


def command_hash(args: list[str]) -> str:
    payload = json.dumps(list(args), ensure_ascii=False, sort_keys=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approval_id_for(args: list[str]) -> str:
    return f"approval_{command_hash(args)[:12]}"


def load_approvals(workspace: str | Path) -> dict[str, Any]:
    path = approvals_path(workspace)
    if not path.exists():
        return {"version": APPROVALS_VERSION, "approvals": []}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    payload.setdefault("version", APPROVALS_VERSION)
    payload.setdefault("approvals", [])
    return payload


def save_approvals(workspace: str | Path, payload: dict[str, Any]) -> Path:
    path = approvals_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["version"] = APPROVALS_VERSION
    payload.setdefault("approvals", [])
    payload["updated_at"] = _now()
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
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
            item["status"] = "approved"
            item["approved_by"] = approved_by
            item["approved_at"] = _now()
            item["note"] = note
            payload["approvals"] = approvals
            save_approvals(workspace, payload)
            return item
    raise KeyError(f"unknown approval_id: {approval_id}")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
