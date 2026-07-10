import json
import threading
import time
from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.progress import (
    ProgressReporter,
    format_progress_report,
    load_progress_events,
    load_progress_summary,
)
from risk_model_workbench.state import create_run_state, save_run_state


def test_progress_reporter_writes_chinese_message_and_run_state(tmp_path):
    run_path = tmp_path / "project" / "runs" / "run1"
    state = create_run_state(tmp_path / "project", run_id="run1", workflow="full_modeling")
    save_run_state(run_path, state)

    reporter = ProgressReporter(run_path, "feature_prescreen", emit_terminal=False)
    reporter.emit(
        step="quality_screen_done",
        message="表 9/50：质量初筛完成，保留 86/120 个变量",
        current=9,
        total=50,
        metrics={"table": "demo.table", "d01_remain": 86},
    )

    events = load_progress_events(run_path)
    assert events[-1]["message"] == "表 9/50：质量初筛完成，保留 86/120 个变量"
    assert events[-1]["percent"] == 18.0
    assert events[-1]["metrics"]["d01_remain"] == 86

    summary = load_progress_summary(run_path)
    assert summary["stage_label"] == "特征初筛"
    assert summary["latest_event"]["message"].startswith("表 9/50")

    updated = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    progress = updated["stages"]["feature_prescreen"]["progress"]
    assert progress["message"] == "表 9/50：质量初筛完成，保留 86/120 个变量"
    assert progress["percent"] == 18.0


def test_run_status_progress_outputs_chinese_summary(tmp_path, capsys):
    project = tmp_path / "project"
    run_path = project / "runs" / "run1"
    state = create_run_state(project, run_id="run1", workflow="full_modeling")
    state["current_stage"] = "feature_refine"
    save_run_state(run_path, state)
    ProgressReporter(run_path, "feature_refine", emit_terminal=False).emit(
        step="global_corr_done",
        message="全局相关性筛选完成，保留 732 个，剔除 118 个",
        percent=45,
    )

    assert main(["run", "status", "--project", str(project), "--run-id", "run1", "--progress"]) == 0
    output = capsys.readouterr().out
    assert "current_stage: 特征精筛" in output
    assert "progress: 45%" in output
    assert "全局相关性筛选完成" in output


def test_run_watch_once_outputs_recent_events(tmp_path, capsys):
    project = tmp_path / "project"
    run_path = project / "runs" / "run1"
    state = create_run_state(project, run_id="run1", workflow="full_modeling")
    save_run_state(run_path, state)
    ProgressReporter(run_path, "feature_metadata", emit_terminal=False).emit(
        step="table_metadata_done",
        message="表 1/3：元数据完成，候选字段 20 个",
        current=1,
        total=3,
    )

    assert main(["run", "watch", "--project", str(project), "--run-id", "run1", "--once"]) == 0
    output = capsys.readouterr().out
    assert "特征元数据" in output
    assert "表 1/3：元数据完成" in output


def test_progress_heartbeat_emits_repeated_event(monkeypatch, tmp_path):
    from risk_model_workbench import progress as progress_module

    run_path = tmp_path / "project" / "runs" / "run1"
    reporter = ProgressReporter(run_path, "feature_refine", emit_terminal=False)
    heartbeat_emitted = threading.Event()
    original_append = progress_module.append_progress_event

    def tracked_append(run_dir, event, **kwargs):
        result = original_append(run_dir, event)
        heartbeat_emitted.set()
        return result

    monkeypatch.setattr(progress_module, "append_progress_event", tracked_append)

    with reporter.heartbeat(
        step="long_filter_heartbeat",
        message="稳定性筛选仍在计算",
        percent=34,
        metrics={"input_features": 1200},
        interval_seconds=0.001,
    ):
        assert heartbeat_emitted.wait(timeout=1)

    events = load_progress_events(run_path)
    assert any(event["step"] == "long_filter_heartbeat" for event in events)
    assert events[-1]["message"] == "稳定性筛选仍在计算"
    assert events[-1]["metrics"]["input_features"] == 1200


def test_waiting_for_approval_has_distinct_status_note():
    report = format_progress_report(
        run_state={"version_id": "v1", "status": "running", "current_stage": "feature_refine"},
        summary={
            "latest_event": {
                "timestamp": "2000-01-01T00:00:00",
                "status": "waiting_for_approval",
                "message": "特征精筛 SQL 等待人工审批",
                "percent": 10,
            }
        },
    )

    assert "正在等待人工审批" in report
    assert "进程已停止" not in report


def test_progress_report_keeps_feature_funnel_after_stage_done(tmp_path):
    run_path = tmp_path / "project" / "runs" / "run1"
    state = create_run_state(tmp_path / "project", run_id="run1", workflow="full_modeling")
    state["current_stage"] = "feature_refine"
    save_run_state(run_path, state)
    reporter = ProgressReporter(run_path, "feature_refine", emit_terminal=False)
    reporter.emit(
        step="write_outputs",
        status="done",
        message="特征精筛产物写入完成",
        percent=100,
        metrics={
            "feature_funnel": {
                "initial": 2837,
                "available_after_preprocess": 2563,
                "after_quality_filter": 1800,
                "after_stability_filter": 1400,
                "after_global_correlation": 900,
                "after_random_importance": 620,
                "after_null_importance": 520,
                "final": 500,
            }
        },
    )
    reporter.emit(step="action_done", status="done", message="特征精筛完成", percent=100)

    report = format_progress_report(
        run_state=yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8")),
        summary=load_progress_summary(run_path),
        events=load_progress_events(run_path, tail=1),
    )

    assert "feature_funnel:" in report
    assert "初始 2837" in report
    assert "全局相关性去重 900" in report
    assert "最终 500" in report


def test_feature_funnel_is_cleared_when_stage_restarts(tmp_path):
    run_path = tmp_path / "project" / "runs" / "run1"
    state = create_run_state(tmp_path / "project", run_id="run1", workflow="full_modeling")
    state["current_stage"] = "feature_refine"
    save_run_state(run_path, state)
    reporter = ProgressReporter(run_path, "feature_refine", emit_terminal=False)
    reporter.emit(
        step="write_outputs",
        status="done",
        message="特征精筛产物写入完成",
        percent=100,
        metrics={"feature_funnel": {"initial": 100, "final": 20}},
    )
    reporter.emit(
        step="stage_started",
        status="started",
        message="特征精筛开始执行",
        percent=0,
    )

    summary = load_progress_summary(run_path)
    report = format_progress_report(
        run_state=state,
        summary=summary,
        events=load_progress_events(run_path),
    )

    assert "feature_funnel" not in summary["stages"]["feature_refine"]
    assert "feature_funnel:" not in report


def test_progress_summary_and_state_use_atomic_replace(monkeypatch, tmp_path):
    from risk_model_workbench import progress as progress_module

    run_path = tmp_path / "project" / "runs" / "run1"
    state = create_run_state(tmp_path / "project", run_id="run1", workflow="full_modeling")
    save_run_state(run_path, state)
    original_replace = progress_module.os.replace
    replaced_targets = []

    def tracked_replace(source, target):
        replaced_targets.append(Path(target))
        original_replace(source, target)

    monkeypatch.setattr(progress_module.os, "replace", tracked_replace)
    ProgressReporter(run_path, "feature_refine", emit_terminal=False).emit(
        step="quality_start",
        message="基础质量筛选开始",
        percent=30,
    )

    assert run_path / "audit" / "progress_summary.json" in replaced_targets
    assert run_path / "run_state.yml" in replaced_targets


def test_heartbeat_exit_is_bounded_and_cancels_late_event(monkeypatch, tmp_path):
    from risk_model_workbench import progress as progress_module

    run_path = tmp_path / "project" / "runs" / "run1"
    reporter = ProgressReporter(run_path, "feature_refine", emit_terminal=False)
    append_started = threading.Event()
    release_append = threading.Event()
    original_append = progress_module.append_progress_event

    def blocked_append(run_dir, event, *, cancel_event=None):
        append_started.set()
        release_append.wait(timeout=2)
        return original_append(run_dir, event, cancel_event=cancel_event)

    monkeypatch.setattr(progress_module, "append_progress_event", blocked_append)
    heartbeat = reporter.heartbeat(
        step="blocked_heartbeat",
        message="相关矩阵仍在计算",
        percent=38,
        interval_seconds=0.001,
        shutdown_timeout_seconds=0.01,
    )
    heartbeat.__enter__()
    assert append_started.wait(timeout=1)
    started_at = time.monotonic()
    heartbeat.__exit__(None, None, None)
    elapsed = time.monotonic() - started_at

    assert elapsed < 0.5
    release_append.set()
    assert heartbeat._thread is not None
    heartbeat._thread.join(timeout=1)
    assert not heartbeat._thread.is_alive()
    assert load_progress_events(run_path) == []


def test_sample_check_emits_stage_progress(tmp_path, capsys):
    project = _make_project(tmp_path)
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "run1"]) == 0
    capsys.readouterr()

    assert main(["sample", "check", "--project", str(project), "--run-id", "run1"]) == 0
    output = capsys.readouterr().out
    assert "[RMW] 样本检查" in output
    assert "样本检查完成" in output

    events_path = project / "runs" / "run1" / "audit" / "progress_events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["stage"] == "sample_check" and event["status"] == "scaffold" for event in events)


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: demo_project",
                "  display_name: Demo Project",
                "data:",
                "  source_table: demo.sample",
                "  id_columns:",
                "    - uid",
                "  target_column: label",
                "  time_column: event_time",
                "  period_column: ds",
                "segments:",
                "  - name: all",
                "    display_name: All",
                "    filter: null",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return project
