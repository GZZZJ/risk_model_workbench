from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.project_state import audit_run


def test_unmanaged_version_does_not_require_agent_runtime(tmp_path):
    project = _make_project(tmp_path)
    assert main(["version", "init", "--project", str(project), "--workflow", "sample_audit", "--version-id", "demo_model_v1_20260710"]) == 0

    audit = audit_run(project, "demo_model_v1_20260710", stage="validate_config")

    assert audit["verdict"] == "complete"
    assert "agent_runtime" not in {item["stage"] for item in audit["stages"]}


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs", "versions"]:
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
    return project
