from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.run_evidence import load_run_evidence


def test_version_init_writes_version_workspace(tmp_path):
    project = _make_project(tmp_path)

    assert (
        main(
            [
                "version",
                "init",
                "--project",
                str(project),
                "--workflow",
                "full_modeling",
                "--version-id",
                "demo_model_v1_20260701",
            ]
        )
        == 0
    )

    version_dir = project / "versions" / "demo_model_v1_20260701"
    state = yaml.safe_load((version_dir / "version_state.yml").read_text(encoding="utf-8"))
    index = yaml.safe_load((project / "versions" / "index.yml").read_text(encoding="utf-8"))

    assert state["version_id"] == "demo_model_v1_20260701"
    assert not (version_dir / "run_state.yml").exists()
    assert (version_dir / "audit" / "artifact_manifest.json").exists()
    assert index["active_version_id"] == "demo_model_v1_20260701"


def test_migrate_run_to_version_keeps_legacy_run_readable(tmp_path):
    project = _make_project(tmp_path)

    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "legacy_run"]) == 0
    assert (
        main(
            [
                "version",
                "migrate-run",
                "--project",
                str(project),
                "--run-id",
                "legacy_run",
                "--version-id",
                "demo_model_v2_20260701",
            ]
        )
        == 0
    )

    version_dir = project / "versions" / "demo_model_v2_20260701"
    state = yaml.safe_load((version_dir / "version_state.yml").read_text(encoding="utf-8"))
    evidence = load_run_evidence(project, "legacy_run")

    assert state["lineage"]["legacy_run_id"] == "legacy_run"
    assert evidence.run_id == "demo_model_v2_20260701"
    assert evidence.run_path == version_dir
    assert main(["run", "audit", "--project", str(project), "--run-id", "legacy_run", "--stage", "validate_config"]) == 0


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "runs", "reports", "docs"]:
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
