import json
from pathlib import Path

import yaml

from risk_model_workbench.agent.advisor import accept_advisor_response, load_advisor_request
from risk_model_workbench.agent.executor import run_agent
from risk_model_workbench.agent.eval import emit_scenario_evidence
from risk_model_workbench.agent.state import load_agent_state
from risk_model_workbench.harness.runtime import current_action_attempt, stage_action_done, stage_action_failed
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.state import register_artifact
from risk_model_workbench.cli import main


VERSION_ID = "demo_model_v1_20260710"


def test_agent_local_happy_path_closes_strict_audit_after_advisor_pause(tmp_path):
    project = _make_project(tmp_path)
    request_path = _write_request(project)
    assert (
        main(
            [
                "agent",
                "start",
                "--project",
                str(project),
                "--request",
                str(request_path),
                "--version-id",
                VERSION_ID,
                "--workflow",
                "full_modeling",
            ]
        )
        == 0
    )
    workspace = project / "versions" / VERSION_ID
    initial = load_agent_state(workspace)

    runner = _SyntheticRunner(workspace)
    first = run_agent(project, VERSION_ID, runner=runner)
    assert first["status"] == "waiting_for_advisor"

    request_id = first["blocker"]["advisor_request_id"]
    request = load_advisor_request(workspace, request_id)
    response_path = _write_advisor_response(workspace, request)
    assert accept_advisor_response(workspace, response_path)["accepted"] is True

    final = run_agent(project, VERSION_ID, runner=runner)
    audit = audit_run(project, VERSION_ID)

    assert final["status"] == "done"
    assert audit["verdict"] == "complete"
    assert any(item["stage"] == "agent_runtime" and item["verdict"] == "complete" for item in audit["stages"])
    trace = [json.loads(line) for line in (workspace / "audit" / "agent_trace.jsonl").read_text(encoding="utf-8").splitlines()]
    action_attempts = {item["attempt_id"] for item in trace if item.get("event") == "action" and item.get("attempt_id")}
    result_attempts = {item["attempt_id"] for item in trace if item.get("event") == "result" and item.get("attempt_id")}
    assert action_attempts
    assert action_attempts <= result_attempts
    emit_scenario_evidence(
        state_pairs=[(initial, first), (first, final)],
        workspace=workspace,
        audit=audit,
    )


class _SyntheticRunner:
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.train_attempts = 0

    def __call__(self, _argv: list[str]) -> int:
        attempt = current_action_attempt()
        assert attempt is not None
        action_id = attempt.action_id
        if action_id == "train_baseline" and self.train_attempts == 0:
            self.train_attempts += 1
            _write_train_context(self.workspace)
            stage_action_failed(self.workspace, "train_baseline", "host agent plan required", failure_code="advisor_required")
            return 1
        _write_stage_artifacts(self.workspace, action_id)
        stage_action_done(self.workspace, action_id)
        return 0


def _write_stage_artifacts(workspace: Path, action_id: str) -> None:
    mapping = {
        "sample_check": [
            ("sample_check/sample_summary.json", '{"status":"done"}\n'),
            ("sample_check/sample_check_report.md", "# Sample Check\n\nstatus: done\n"),
        ],
        "feature_metadata": [
            ("feature_metadata/columns.csv", "name,type\nx1,float\n"),
        ],
        "feature_prescreen": [
            ("feature_selection/prescreen_run_summary.json", '{"status":"done"}\n'),
            ("feature_selection/prescreen_final_remain_features.json", '{"features":["x1"]}\n'),
        ],
        "build_wide_sql": [
            ("feature_selection/wide_sql_summary.json", '{"status":"done"}\n'),
            ("feature_selection/wide_table_execution.json", '{"status":"done"}\n'),
        ],
        "feature_refine": [
            ("feature_selection/stage_summary.json", '{"status":"done"}\n'),
            ("feature_selection/final_features.txt", "x1\n"),
        ],
        "train_baseline": [
            ("modeling/baseline_all/train_metrics.json", '{"status":"done","valid_auc":0.7}\n'),
        ],
        "evaluate": [
            ("evaluation/evaluation_summary.json", '{"status":"done"}\n'),
        ],
        "compare": [
            ("evaluation/champion_challenger.json", '{"status":"done"}\n'),
        ],
        "report": [
            ("reports/model_report.md", "# Model Report\n"),
            ("reports/model_report.html", "<h1>Model Report</h1>\n"),
            ("reports/model_card.md", "# Model Card\n"),
            ("reports/executive_summary.md", "# Executive Summary\n"),
        ],
    }
    for relative, content in mapping[action_id]:
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        register_artifact(workspace, action_id, relative)


def _write_train_context(workspace: Path) -> None:
    _write_stage_artifacts(workspace, "feature_refine")
    model_dir = workspace / "modeling" / "baseline_all"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "tuning_context_round_1.json").write_text(
        json.dumps({"round": 1, "experiment": "baseline_all"}),
        encoding="utf-8",
    )


def _write_advisor_response(workspace: Path, request: dict) -> Path:
    plan_path = workspace / "modeling" / "baseline_all" / "llm_tuning_plan_round_1.json"
    plan_path.write_text(
        json.dumps(
            {
                "experiment": "baseline_all",
                "round": 1,
                "diagnosis": "bounded synthetic candidate",
                "candidates": [
                    {
                        "name": "regularized",
                        "params": {"learning_rate": 0.03, "num_leaves": 31, "max_depth": 5},
                        "reason": "synthetic local path",
                    }
                ],
                "stop": False,
            }
        ),
        encoding="utf-8",
    )
    response = {
        "version": 1,
        "request_id": request["request_id"],
        "type": request["expected_response"]["type"],
        "status": "answered",
        "decision": "continue",
        "summary": "Continue with bounded plan.",
        "request_identity": {
            "task_id": request["task_id"],
            "attempt_id": request["attempt_id"],
            "invocation_hash": request["invocation_hash"],
            "context_hash": request["context_hash"],
            "round": request["round"],
        },
        "output_files": ["modeling/baseline_all/llm_tuning_plan_round_1.json"],
        "risk_notes": [],
        "requires_user_confirmation": False,
    }
    response_path = workspace / "advisor_response.json"
    response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    return response_path


def _write_request(project: Path) -> Path:
    request_path = project / "request.md"
    request_path.write_text(
        "---\n"
        + yaml.safe_dump(
            {
                "request_id": "agent-local-happy-path",
                "project": "demo_project",
                "workflow": "full_modeling",
                "target_column": "label",
                "id_columns": ["uid"],
                "split_column": "ds",
                "data_source_mode": "local_feather",
                "sample_location": "data/raw/sample.feather",
                "sample_checks": ["sample_check_001"],
                "feature_selection": {"rounds": ["metadata", "prescreen", "refine"]},
                "experiments": [{"name": "baseline_all"}],
                "evaluation": {"metrics": ["auc"], "champions": []},
                "reports": {"outputs": ["model_report.md"]},
            },
            allow_unicode=True,
            sort_keys=False,
        )
        + "---\n",
        encoding="utf-8",
    )
    return request_path


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs", "versions", "data/raw"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project", "project_key": "demo_model"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (project / "data" / "raw" / "sample.feather").write_bytes(b"synthetic")
    return project
