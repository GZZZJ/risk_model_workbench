"""Append-only trace log for Agent decisions and actions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


BLOCKED_KEYS = {"hidden_reasoning", "reasoning", "thought", "chain_of_thought", "cot"}


def trace_path(workspace: str | Path) -> Path:
    return Path(workspace) / "audit" / "agent_trace.jsonl"


def append_trace(workspace: str | Path, event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    item = _sanitize(payload or {})
    item["timestamp"] = _now()
    item["event"] = event
    path = trace_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False, default=str))
        handle.write("\n")
    return item


def load_recent_trace(workspace: str | Path, *, limit: int = 20) -> list[dict[str, Any]]:
    path = trace_path(workspace)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize(val) for key, val in value.items() if str(key) not in BLOCKED_KEYS}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    return value


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
