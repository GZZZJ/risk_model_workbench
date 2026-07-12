from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.paths import project_config_path, resolve_project_path, stage_config_path
from risk_model_workbench.project.create import create_project


def test_project_config_path_prefers_yml():
    project = resolve_project_path("projects/2026-05-fujie-gcard-v1")
    assert project_config_path(project).name == "project.yml"


def test_stage_config_path_prefers_yaml_and_accepts_legacy_only(tmp_path):
    configs = tmp_path / "configs"
    configs.mkdir()
    legacy = configs / "train.yml"
    legacy.write_text("training: {rounds: 10}\n", encoding="utf-8")
    assert stage_config_path(tmp_path, "train") == legacy

    canonical = configs / "train.yaml"
    canonical.write_text("training:\n  rounds: 10\n", encoding="utf-8")
    assert stage_config_path(tmp_path, "train") == canonical


def test_create_project_generates_legacy_mirrors_from_canonical_templates(tmp_path):
    project = create_project(
        tmp_path,
        name="mirror-demo",
        display_name="Mirror Demo",
        scenario="demo",
    )

    assert (project / "project.yaml").read_text(encoding="utf-8") == (
        project / "project.yml"
    ).read_text(encoding="utf-8")
    canonical_configs = sorted((project / "configs").glob("*.yaml"))
    assert canonical_configs
    for canonical in canonical_configs:
        mirror = canonical.with_suffix(".yml")
        assert mirror.read_text(encoding="utf-8") == canonical.read_text(encoding="utf-8")


def test_template_keeps_synchronized_legacy_mirrors():
    template = Path(__file__).resolve().parents[1] / "templates" / "project"
    assert (template / "project.yaml").read_text(encoding="utf-8") == (
        template / "project.yml"
    ).read_text(encoding="utf-8")
    for canonical in sorted((template / "configs").glob("*.yaml")):
        assert canonical.with_suffix(".yml").read_text(encoding="utf-8") == canonical.read_text(
            encoding="utf-8"
        )


def _minimal_project(tmp_path: Path, project_yaml: str) -> Path:
    project = tmp_path / "project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(project_yaml, encoding="utf-8")
    return project


def test_project_validate_reports_project_variant_conflict(tmp_path, capsys):
    payload = (
        "project: {name: demo}\n"
        "data: {source_table: t, id_columns: [id], target_column: y, time_column: dt, period_column: ds}\n"
        "segments: [{name: all}]\n"
    )
    project = _minimal_project(tmp_path, payload)
    (project / "project.yaml").write_text(payload.replace("name: demo", "name: other"), encoding="utf-8")

    assert main(["project", "validate", "--project", str(project)]) == 1
    output = capsys.readouterr().out
    assert str(project / "project.yml") in output
    assert str(project / "project.yaml") in output


def test_project_validate_reports_stage_variant_conflict(tmp_path, capsys):
    project = _minimal_project(
        tmp_path,
        "project: {name: demo}\n"
        "data: {source_table: t, id_columns: [id], target_column: y, time_column: dt, period_column: ds}\n"
        "segments: [{name: all}]\n",
    )
    (project / "configs" / "train.yaml").write_text("training: {rounds: 10}\n", encoding="utf-8")
    (project / "configs" / "train.yml").write_text("training: {rounds: 20}\n", encoding="utf-8")

    assert main(["project", "validate", "--project", str(project)]) == 1
    output = capsys.readouterr().out
    assert str(project / "configs" / "train.yaml") in output
    assert str(project / "configs" / "train.yml") in output


def test_project_validate_reports_strict_yaml_format_error(tmp_path, capsys):
    project = _minimal_project(tmp_path, "project: one\nproject: two\n")

    assert main(["project", "validate", "--project", str(project)]) == 1
    output = capsys.readouterr().out
    assert "duplicate key" in output
    assert str(project / "project.yml") in output


def test_project_status_returns_one_for_config_conflict(tmp_path, capsys):
    payload = "project: {name: demo}\ndata: {}\nsegments: []\n"
    project = _minimal_project(tmp_path, payload)
    (project / "project.yaml").write_text(payload.replace("demo", "other"), encoding="utf-8")

    assert main(["project", "status", "--project", str(project)]) == 1
    output = capsys.readouterr().out
    assert "project status failed" in output
    assert str(project / "project.yml") in output
    assert str(project / "project.yaml") in output


def test_version_migration_returns_one_for_config_format_error(tmp_path, capsys):
    payload = "project: {name: demo}\ndata: {}\nsegments: []\n"
    project = _minimal_project(tmp_path, payload)
    (project / "project.yaml").write_text("project: one\nproject: two\n", encoding="utf-8")

    assert main(
        ["version", "migrate-run", "--project", str(project), "--run-id", "legacy-run"]
    ) == 1
    output = capsys.readouterr().out
    assert "version migration failed" in output
    assert str(project / "project.yml") in output
    assert str(project / "project.yaml") in output


def test_active_project_config_mirrors_are_semantically_equal():
    project = resolve_project_path("projects/2026-05-fujie-gcard-v1")
    assert yaml.safe_load((project / "project.yml").read_text(encoding="utf-8")) == yaml.safe_load(
        (project / "project.yaml").read_text(encoding="utf-8")
    )
    for canonical in sorted((project / "configs").glob("*.yaml")):
        mirror = canonical.with_suffix(".yml")
        assert mirror.exists()
        assert yaml.safe_load(canonical.read_text(encoding="utf-8")) == yaml.safe_load(
            mirror.read_text(encoding="utf-8")
        )
