import json
from pathlib import Path

import yaml

import risk_model_workbench.rules as rules_module
from risk_model_workbench.cli import main
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.state import mark_stage_done, register_artifact


def test_workflow_validate_stage_contracts(tmp_path):
    valid = tmp_path / "valid.yml"
    valid.write_text(
        "\n".join(
            [
                "name: demo",
                "stages:",
                "  - validate_config",
                "stage_contracts:",
                "  validate_config:",
                "    required_artifacts:",
                "      - configs_snapshot/*",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["workflow", "validate", "--workflow", str(valid)]) == 0

    unknown_stage = tmp_path / "unknown_stage.yml"
    unknown_stage.write_text(
        "\n".join(
            [
                "name: demo",
                "stages:",
                "  - validate_config",
                "stage_contracts:",
                "  missing:",
                "    required_artifacts:",
                "      - configs_snapshot/*",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["workflow", "validate", "--workflow", str(unknown_stage)]) == 1

    empty_pattern = tmp_path / "empty_pattern.yml"
    empty_pattern.write_text(
        "\n".join(
            [
                "name: demo",
                "stages:",
                "  - validate_config",
                "stage_contracts:",
                "  validate_config:",
                "    required_artifacts:",
                "      - ''",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["workflow", "validate", "--workflow", str(empty_pattern)]) == 1


def test_audit_contract_strict_json_and_evidence_sources(tmp_path, capsys):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "complete_run")
    _register_sample_artifacts(run_path)
    mark_stage_done(run_path, "sample_check")

    audit = audit_run(project, "complete_run", stage="sample_check")
    assert audit["verdict"] == "complete"
    assert audit["source_of_truth"] == [
        "runs/complete_run/run_state.yml",
        "runs/complete_run/audit/artifact_manifest.json",
        "workflows/full_modeling.yml",
    ]
    assert main(["run", "audit", "--project", str(project), "--run-id", "complete_run", "--stage", "sample_check", "--strict"]) == 0

    capsys.readouterr()
    assert main(["run", "audit", "--project", str(project), "--run-id", "complete_run", "--stage", "sample_check", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "complete"
    assert payload["stages"][0]["issues"] == []
    assert "workflows/full_modeling.yml" in payload["source_of_truth"]


def test_audit_blocks_missing_scaffold_and_imported_evidence(tmp_path):
    project = _make_project(tmp_path)

    missing_run = _init_run(project, "missing_run")
    mark_stage_done(missing_run, "sample_check")
    audit = audit_run(project, "missing_run", stage="sample_check")
    assert audit["verdict"] == "incomplete"
    assert audit["stages"][0]["failure_code"] == "artifact_contract_failed"
    assert "data_missing" in audit["stages"][0]["failure_codes"]
    assert main(["run", "audit", "--project", str(project), "--run-id", "missing_run", "--stage", "sample_check", "--strict"]) == 1

    scaffold_run = _init_run(project, "scaffold_run")
    _register_sample_artifacts(scaffold_run, source="scaffold")
    mark_stage_done(scaffold_run, "sample_check")
    audit = audit_run(project, "scaffold_run", stage="sample_check")
    assert audit["verdict"] == "scaffold"
    assert audit["stages"][0]["failure_code"] == "scaffold_only"
    assert main(["run", "audit", "--project", str(project), "--run-id", "scaffold_run", "--stage", "sample_check", "--strict"]) == 1

    imported_run = _init_run(project, "imported_run")
    _register_sample_artifacts(imported_run, source="imported")
    mark_stage_done(imported_run, "sample_check")
    audit = audit_run(project, "imported_run", stage="sample_check")
    assert audit["verdict"] == "imported"
    assert audit["stages"][0]["failure_code"] == "unknown"
    assert main(["run", "audit", "--project", str(project), "--run-id", "imported_run", "--stage", "sample_check", "--strict"]) == 1


def test_audit_requires_tuning_evidence_when_llm_tuning_enabled(tmp_path):
    project = _make_project(tmp_path)
    run_path = _init_run(project, "tuning_missing")
    train_cfg = run_path / "configs_runtime" / "train.yaml"
    train_cfg.parent.mkdir(parents=True, exist_ok=True)
    train_cfg.write_text("training:\n  mode: llm_guided_tune\n", encoding="utf-8")
    model_dir = run_path / "modeling" / "main_lgbm"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "train_metrics.json").write_text('{"status": "done"}\n', encoding="utf-8")
    register_artifact(run_path, "train_baseline", "modeling/main_lgbm/train_metrics.json")
    mark_stage_done(run_path, "train_baseline")

    audit = audit_run(project, "tuning_missing", stage="train_baseline")

    assert audit["verdict"] == "incomplete"
    assert any("llm tuning artifact missing" in issue for issue in audit["stages"][0]["issues"])


def test_lesson_promote_and_rules_list_are_idempotent(tmp_path, monkeypatch, capsys):
    project = _make_project(tmp_path)
    rules_path = tmp_path / "workbench_rules.yml"
    monkeypatch.setattr(rules_module, "RULES_PATH", rules_path)

    assert (
        main(
            [
                "lesson",
                "add",
                "--project",
                str(project),
                "--title",
                "SQL approval gate",
                "--kind",
                "guardrail",
                "--body",
                "Dry-run SQL must be reviewed before DP pulls.",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "lesson",
                "promote",
                "--project",
                str(project),
                "--title",
                "SQL approval gate",
                "--target",
                "guardrail",
                "--rule-id",
                "sql-approval-gate",
                "--note",
                "first note",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "lesson",
                "promote",
                "--project",
                str(project),
                "--title",
                "SQL approval gate",
                "--target",
                "guardrail",
                "--rule-id",
                "sql-approval-gate",
                "--note",
                "updated note",
            ]
        )
        == 0
    )

    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
    assert len(rules["rules"]) == 1
    assert rules["rules"][0]["status"] == "proposed"
    assert rules["rules"][0]["note"] == "updated note"

    capsys.readouterr()
    assert main(["rules", "list", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rules"][0]["id"] == "sql-approval-gate"

    assert main(["project", "status", "--project", str(project)]) == 0
    status_text = capsys.readouterr().out
    assert "proposed: 1" in status_text
    assert "unenforced_guardrails: 1" in status_text


def _init_run(project: Path, run_id: str) -> Path:
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", run_id]) == 0
    return project / "runs" / run_id


def _register_sample_artifacts(run_path: Path, *, source: str = "generated") -> None:
    summary = run_path / "sample_check" / "sample_summary.json"
    report = run_path / "sample_check" / "sample_check_report.md"
    summary.write_text('{"status": "done"}\n', encoding="utf-8")
    report.write_text("# Sample Check\n\nstatus: done\n", encoding="utf-8")
    register_artifact(run_path, "sample_check", "sample_check/sample_summary.json", source=source)
    register_artifact(run_path, "sample_check", "sample_check/sample_check_report.md", source=source)


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
