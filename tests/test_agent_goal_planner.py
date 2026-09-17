from pathlib import Path

import yaml

from risk_model_workbench.agent.goal_planner import create_request_draft_from_objective
from risk_model_workbench.agent.model_gateway import FakeModelGateway
from risk_model_workbench.request import parse_model_request


def test_goal_planner_uses_model_for_intent_but_project_for_data_contract(tmp_path):
    project = _project(tmp_path)
    gateway = FakeModelGateway(
        [
            {
                "objective_summary": "Audit the current sample and produce an evidence report.",
                "workflow": "sample_audit",
                "experiment_name": "baseline_lgbm",
                "method": "lightgbm",
                "training_mode": "llm_guided_tune",
                "metrics": ["auc", "ks"],
                "report_outputs": ["model_report.md"],
                "assumptions": ["Use the project-defined target and split fields."],
                "requires_user_confirmation": True,
            }
        ]
    )

    result = create_request_draft_from_objective(
        project,
        objective="检查样本并生成报告；不要使用项目外的标签。",
        request_id="goal-draft-001",
        gateway=gateway,
    )
    request = parse_model_request(result["request_path"])

    assert result["requires_user_confirmation"] is True
    assert result["validation"]["status"] == "ok"
    assert request["metadata"]["target_column"] == "label"
    assert request["metadata"]["id_columns"] == ["uid"]
    assert request["metadata"]["split_column"] == "ds"
    assert request["metadata"]["workflow"] == "sample_audit"
    assert (project / "requests" / "audit" / "goal-draft-001.model_invocation.json").exists()


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "reports", "requests", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo", "project_key": "demo"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
