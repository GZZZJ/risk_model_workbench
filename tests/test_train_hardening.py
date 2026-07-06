import json
from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.modeling.llm_tuning import HostAgentTuningPlanRequired


def _make_train_project(tmp_path: Path, *, input_exists: bool = True, feature_exists: bool = True) -> tuple[Path, Path]:
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "data").mkdir()
    (project / "runs" / "modeling_feature_set").mkdir(parents=True)
    project.joinpath("project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: train-hardening",
                "  display_name: Train Hardening",
                "data:",
                "  source_table: mart.base",
                "  id_columns: [uid]",
                "  target_column: target",
                "  split_column: final_flag",
                "  time_column: apply_time",
                "  period_column: apply_month",
                "split:",
                "  source_column: final_flag",
                "  ins_values: [DEV]",
                "  oos_values: [DEV-OOS]",
                "  oot_values: [OOT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project.joinpath("configs", "train.yaml").write_text(
        "\n".join(
            [
                "training:",
                "  default_algorithm: logistic_regression",
                "  train_values: [DEV]",
                "  valid_values: [DEV-OOS]",
                "  feature_list_path: runs/modeling_feature_set/feature_list.txt",
                "input:",
                "  feather_path: data/train.feather",
                "  label_column: target",
                "  split_column: final_flag",
                "  base_columns: [uid, final_flag, target]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if input_exists:
        (project / "data" / "train.feather").write_text("placeholder", encoding="utf-8")
    if feature_exists:
        (project / "runs" / "modeling_feature_set" / "feature_list.txt").write_text("f1\nf2\n", encoding="utf-8")
    assert main(["run", "init", "--project", str(project), "--workflow", "full_modeling", "--run-id", "r1"]) == 0
    return project, project / "runs" / "r1"


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_train_plan_only_writes_plan_without_model(tmp_path):
    project, run_path = _make_train_project(tmp_path)

    rc = main(["train", "--project", str(project), "--run-id", "r1", "--experiment", "baseline", "--plan-only"])

    model_dir = run_path / "modeling" / "baseline"
    assert rc == 0
    assert (model_dir / "train_plan.json").exists()
    assert (model_dir / "train_plan.md").exists()
    assert not (model_dir / "model.pkl").exists()
    assert _read_json(model_dir / "train_plan.json")["will_train"] is True
    state = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    assert state["stages"]["train_baseline"]["status"] == "scaffold"


def test_train_input_missing_still_scaffolds_with_status_and_summary(tmp_path):
    project, run_path = _make_train_project(tmp_path, input_exists=False)

    rc = main(["train", "--project", str(project), "--run-id", "r1", "--experiment", "baseline"])

    model_dir = run_path / "modeling" / "baseline"
    assert rc == 0
    assert _read_json(model_dir / "train_metrics.json")["status"] == "scaffold"
    assert _read_json(model_dir / "training_status.json")["status"] == "scaffold"
    assert (model_dir / "training_summary.md").exists()
    state = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    assert state["stages"]["train_baseline"]["status"] == "scaffold"


def test_real_training_exception_marks_failed_not_scaffold(tmp_path, monkeypatch):
    project, run_path = _make_train_project(tmp_path)

    def fail_training(**kwargs):
        raise RuntimeError("boom during fit")

    import risk_model_workbench.modeling.train_xgb as train_xgb

    monkeypatch.setattr(train_xgb, "train_tabular_from_feather", fail_training)
    rc = main(["train", "--project", str(project), "--run-id", "r1", "--experiment", "baseline"])

    model_dir = run_path / "modeling" / "baseline"
    assert rc == 1
    assert _read_json(model_dir / "train_metrics.json")["status"] == "failed"
    status = _read_json(model_dir / "training_status.json")
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    state = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    assert state["stages"]["train_baseline"]["status"] == "failed"
    assert state["decisions"][-1]["decision"] == "failed"


def test_advisor_required_writes_status_and_returns_two(tmp_path, monkeypatch):
    project, run_path = _make_train_project(tmp_path)

    def require_advisor(**kwargs):
        raise HostAgentTuningPlanRequired(plan_path="plan.json", context_path="context.json")

    import risk_model_workbench.modeling.train_xgb as train_xgb

    monkeypatch.setattr(train_xgb, "train_tabular_from_feather", require_advisor)
    rc = main(["train", "--project", str(project), "--run-id", "r1", "--experiment", "baseline"])

    model_dir = run_path / "modeling" / "baseline"
    assert rc == 2
    assert _read_json(model_dir / "train_metrics.json")["status"] == "advisor_required"
    assert _read_json(model_dir / "training_status.json")["status"] == "advisor_required"
    state = yaml.safe_load((run_path / "run_state.yml").read_text(encoding="utf-8"))
    assert state["stages"]["train_baseline"]["status"] == "failed"
    assert state["stages"]["train_baseline"]["failure_code"] == "advisor_required"


def test_successful_training_writes_status_summary_and_registers_artifacts(tmp_path, monkeypatch):
    project, run_path = _make_train_project(tmp_path)

    def fake_training(output_dir, **kwargs):
        output_dir = Path(output_dir)
        metrics = {
            "train_auc": 0.8,
            "valid_auc": 0.7,
            "train_ks": 0.4,
            "valid_ks": 0.3,
            "auc_gap": 0.1,
            "best_iteration": 12,
        }
        (output_dir / "metrics_train_valid.json").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        (output_dir / "actual_feature_list.txt").write_text("f1\nf2\n", encoding="utf-8")
        (output_dir / "feature_importance.csv").write_text("feature,gain\nf1,1\n", encoding="utf-8")
        (output_dir / "run_config.json").write_text(
            json.dumps({"training_mode": "single_train", "candidate_feature_count": 2, "actual_feature_count": 2, "params": {"max_iter": 10}}) + "\n",
            encoding="utf-8",
        )
        return metrics

    import risk_model_workbench.modeling.train_xgb as train_xgb

    monkeypatch.setattr(train_xgb, "train_tabular_from_feather", fake_training)
    rc = main(["train", "--project", str(project), "--run-id", "r1", "--experiment", "baseline"])

    model_dir = run_path / "modeling" / "baseline"
    assert rc == 0
    assert _read_json(model_dir / "training_status.json")["status"] == "done"
    summary = (model_dir / "training_summary.md").read_text(encoding="utf-8")
    assert "Valid AUC" in summary
    manifest = (run_path / "audit" / "artifact_manifest.json").read_text(encoding="utf-8")
    assert "modeling/baseline/training_status.json" in manifest
    assert "modeling/baseline/training_summary.md" in manifest


def test_runtime_code_has_no_external_encrypted_skill_dependency():
    repo = Path(__file__).resolve().parents[1]
    forbidden = ["xgb" + "-lgb-trainer-enc", "." + "claude/skills", "boot" + "strap.py"]
    roots = [repo / "src", repo / "tests", repo / "templates", repo / "workflows"]
    files = [repo / "pyproject.toml"]
    for root in roots:
        files.extend(path for path in root.rglob("*") if path.is_file() and path.suffix in {".py", ".yml", ".yaml", ".toml", ".md"})

    offenders = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            if token in text:
                offenders.append(str(path.relative_to(repo)))

    assert offenders == []
