"""Progress event helpers for long-running workbench commands."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:  # pragma: no cover - fcntl is unavailable on Windows.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


STAGE_LABELS = {
    "validate_config": "配置校验",
    "sample_check": "样本检查",
    "feature_metadata": "特征元数据",
    "feature_prescreen": "特征初筛",
    "d01_d02_screening": "特征初筛",
    "build_wide_sql": "宽表 SQL 生成",
    "feature_refine": "特征精筛",
    "train_baseline": "模型训练",
    "evaluate": "模型评估",
    "compare": "冠军挑战者对比",
    "report": "报告生成",
}

STATUS_LABELS = {
    "started": "开始",
    "running": "进行中",
    "done": "完成",
    "failed": "失败",
    "waiting_for_approval": "等待审批",
    "skipped": "跳过",
    "scaffold": "脚手架完成",
}


def stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, stage)


def progress_events_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "audit" / "progress_events.jsonl"


def progress_summary_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "audit" / "progress_summary.json"


def load_progress_events(run_dir: str | Path, *, tail: int | None = None) -> list[dict[str, Any]]:
    path = progress_events_path(run_dir)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if tail is not None:
        lines = lines[-tail:]
    events = []
    for line in lines:
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def load_progress_summary(run_dir: str | Path) -> dict[str, Any]:
    path = progress_summary_path(run_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return ""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}小时{minutes}分{secs}秒"
    if minutes:
        return f"{minutes}分{secs}秒"
    return f"{secs}秒"


def format_progress_event(event: dict[str, Any]) -> str:
    stage = stage_label(str(event.get("stage") or ""))
    percent = event.get("percent")
    if percent is None:
        percent_text = "--"
    else:
        percent_text = f"{float(percent):.0f}%"
    message = event.get("message") or STATUS_LABELS.get(str(event.get("status") or ""), "")
    parts = [f"[RMW] {stage} {percent_text}", str(message)]
    elapsed = format_elapsed(event.get("elapsed_seconds"))
    if elapsed:
        parts.append(f"已用 {elapsed}")
    return " | ".join(parts)


def format_progress_report(
    *,
    run_state: dict[str, Any],
    summary: dict[str, Any] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> str:
    summary = summary or {}
    events = events or []
    current_stage = run_state.get("current_stage") or summary.get("stage") or ""
    lines = [
        f"version_id: {run_state.get('version_id', '')}",
        f"run_id: {run_state.get('run_id', '')}",
        f"status: {run_state.get('status', '')}",
        f"current_stage: {stage_label(str(current_stage))}",
    ]
    if summary:
        latest = summary.get("latest_event", {})
        percent = latest.get("percent")
        percent_text = "" if percent is None else f"{float(percent):.0f}%"
        lines.extend(
            [
                f"progress: {percent_text}",
                f"message: {latest.get('message', '')}",
                f"heartbeat_at: {latest.get('timestamp', '')}",
            ]
        )
    stage_progress = (run_state.get("stages") or {}).get(current_stage, {}).get("progress", {})
    if stage_progress and not summary:
        percent = stage_progress.get("percent")
        percent_text = "" if percent is None else f"{float(percent):.0f}%"
        lines.extend(
            [
                f"progress: {percent_text}",
                f"message: {stage_progress.get('message', '')}",
                f"heartbeat_at: {stage_progress.get('heartbeat_at', '')}",
            ]
        )
    if events:
        lines.extend(["", "recent_events:"])
        for event in events:
            lines.append(f"  - {format_progress_event(event)}")
    return "\n".join(lines) + "\n"


class ProgressReporter:
    """Append machine-readable progress and render Chinese terminal lines."""

    def __init__(self, run_dir: str | Path | None, stage: str, *, emit_terminal: bool = True) -> None:
        self.run_dir = Path(run_dir).resolve() if run_dir else None
        self.stage = stage
        self.emit_terminal = emit_terminal
        self._created_at = time.time()

    @property
    def enabled(self) -> bool:
        return self.run_dir is not None

    def emit(
        self,
        *,
        step: str,
        status: str = "running",
        message: str,
        current: int | None = None,
        total: int | None = None,
        percent: float | None = None,
        metrics: dict[str, Any] | None = None,
        level: str = "info",
        emit_terminal: bool | None = None,
    ) -> dict[str, Any]:
        event = build_event(
            run_dir=self.run_dir,
            stage=self.stage,
            step=step,
            status=status,
            message=message,
            current=current,
            total=total,
            percent=percent,
            elapsed_seconds=self._elapsed_seconds(),
            metrics=metrics or {},
            level=level,
        )
        should_print = self.emit_terminal if emit_terminal is None else emit_terminal
        if should_print:
            print(format_progress_event(event), flush=True)
        if self.run_dir:
            append_progress_event(self.run_dir, event)
        return event

    def _elapsed_seconds(self) -> float:
        return round(time.time() - self._created_at, 3)


def build_event(
    *,
    run_dir: Path | None,
    stage: str,
    step: str,
    status: str,
    message: str,
    current: int | None,
    total: int | None,
    percent: float | None,
    elapsed_seconds: float | int | None,
    metrics: dict[str, Any],
    level: str,
) -> dict[str, Any]:
    if percent is None and current is not None and total:
        percent = current / total * 100
    if percent is not None:
        percent = round(min(100.0, max(0.0, float(percent))), 2)
    timestamp = datetime.now().isoformat(timespec="seconds")
    return {
        "timestamp": timestamp,
        "version_id": _version_id(run_dir),
        "run_id": _run_id(run_dir),
        "stage": stage,
        "step": step,
        "status": status,
        "message": message,
        "current": current,
        "total": total,
        "percent": percent,
        "elapsed_seconds": elapsed_seconds,
        "eta_seconds": _eta_seconds(current=current, total=total, elapsed_seconds=elapsed_seconds),
        "metrics": metrics,
        "level": level,
        "pid": os.getpid(),
    }


def append_progress_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    run_path = Path(run_dir)
    path = progress_events_path(run_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _progress_lock(run_path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
        _write_progress_summary(run_path, event)
        _update_run_state_progress(run_path, event)


@contextmanager
def _progress_lock(run_dir: Path):
    lock_path = run_dir / "audit" / ".progress.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def emit_progress(
    run_dir: str | Path | None,
    stage: str,
    *,
    step: str,
    status: str = "running",
    message: str,
    current: int | None = None,
    total: int | None = None,
    percent: float | None = None,
    metrics: dict[str, Any] | None = None,
    level: str = "info",
    emit_terminal: bool = True,
) -> dict[str, Any]:
    return ProgressReporter(run_dir, stage, emit_terminal=emit_terminal).emit(
        step=step,
        status=status,
        message=message,
        current=current,
        total=total,
        percent=percent,
        metrics=metrics,
        level=level,
    )


def emit_stage_started(run_dir: str | Path, stage: str) -> None:
    emit_progress(
        run_dir,
        stage,
        step="stage_started",
        status="started",
        message=f"{stage_label(stage)}开始执行",
        percent=0,
    )


def emit_stage_done(run_dir: str | Path, stage: str, *, scaffold: bool = False) -> None:
    status = "scaffold" if scaffold else "done"
    suffix = "完成，产物为脚手架或占位结果" if scaffold else "完成"
    emit_progress(
        run_dir,
        stage,
        step="stage_done",
        status=status,
        message=f"{stage_label(stage)}{suffix}",
        percent=100,
    )


def emit_stage_failed(run_dir: str | Path, stage: str, reason: str) -> None:
    emit_progress(
        run_dir,
        stage,
        step="stage_failed",
        status="failed",
        message=f"{stage_label(stage)}失败：{reason}",
        percent=None,
        level="error",
    )


def _write_progress_summary(run_dir: Path, event: dict[str, Any]) -> None:
    summary_path = progress_summary_path(run_dir)
    existing = load_progress_summary(run_dir)
    by_stage = existing.get("stages", {}) if isinstance(existing.get("stages"), dict) else {}
    stage_summary = dict(by_stage.get(event["stage"], {}))
    stage_summary.update(
        {
            "stage": event["stage"],
            "stage_label": stage_label(event["stage"]),
            "latest_event": event,
            "updated_at": event["timestamp"],
        }
    )
    by_stage[event["stage"]] = stage_summary
    payload = {
        "version": 1,
        "version_id": event.get("version_id", ""),
        "run_id": event.get("run_id", ""),
        "updated_at": event["timestamp"],
        "stage": event["stage"],
        "stage_label": stage_label(event["stage"]),
        "latest_event": event,
        "stages": by_stage,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _update_run_state_progress(run_dir: Path, event: dict[str, Any]) -> None:
    path = run_dir / "version_state.yml"
    if not path.exists():
        path = run_dir / "run_state.yml"
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        state = yaml.safe_load(handle) or {}
    stage_state = state.setdefault("stages", {}).setdefault(event["stage"], {"status": "pending", "artifacts": []})
    stage_state["progress"] = {
        "step": event["step"],
        "status": event["status"],
        "message": event["message"],
        "current": event["current"],
        "total": event["total"],
        "percent": event["percent"],
        "heartbeat_at": event["timestamp"],
        "elapsed_seconds": event["elapsed_seconds"],
        "last_event_status": event["status"],
    }
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(state, handle, allow_unicode=True, sort_keys=False)


def _run_id(run_dir: Path | None) -> str:
    if not run_dir:
        return ""
    for state_path in [run_dir / "version_state.yml", run_dir / "run_state.yml"]:
        if not state_path.exists():
            continue
        try:
            state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
            if state.get("run_id"):
                return str(state["run_id"])
        except yaml.YAMLError:
            continue
    return run_dir.name


def _version_id(run_dir: Path | None) -> str:
    if not run_dir:
        return ""
    state_path = run_dir / "version_state.yml"
    if state_path.exists():
        try:
            state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
            if state.get("version_id"):
                return str(state["version_id"])
        except yaml.YAMLError:
            pass
    return ""


def _eta_seconds(*, current: int | None, total: int | None, elapsed_seconds: float | int | None) -> float | None:
    if not current or not total or current <= 0 or current >= total or elapsed_seconds is None:
        return None
    remaining = (float(elapsed_seconds) / current) * (total - current)
    return round(max(0.0, remaining), 3)
