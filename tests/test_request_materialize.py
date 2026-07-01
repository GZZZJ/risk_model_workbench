import json
from pathlib import Path

import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.request.materialize import materialize_request_runtime_configs


def _write_project(project_dir: Path) -> None:
    (project_dir / "configs").mkdir(parents=True)
    (project_dir / "project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: pytest-project",
                "  display_name: Pytest Project",
                "  scenario: pytest",
                "segments:",
                "  - name: e2e3",
                "    filter: blue_customer_flag in ['E2', 'E3']",
                "data:",
                "  source_table: mart.base",
                "  id_columns: [uid]",
                "  target_column: old_target",
                "  split_column: old_split",
                "  time_column: old_time",
                "split:",
                "  source_column: old_split",
                "  ins_values: [DEV]",
                "  oos_values: [OOS]",
                "  oot_values: [OOT]",
                "champions:",
                "  score_columns: [old_score]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "feature_select.yaml").write_text(
        "\n".join(
            [
                "feature_select:",
                "  wide_table:",
                "    output_table: mart.wide",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "refine_features.yaml").write_text(
        "feature_refine:\n  output_dir: runs/refine\n  preprocessing:\n    drop_constant: true\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "train.yaml").write_text(
        "training:\n  default_algorithm: lightgbm\ninput: {}\nlightgbm: {}\npreprocessing: {}\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "evaluate.yaml").write_text("evaluation:\n  score_columns: [model_score]\n", encoding="utf-8")
    (project_dir / "configs" / "report.yaml").write_text("report:\n  sections: []\n", encoding="utf-8")


def _request_doc() -> dict:
    return {
        "path": "/tmp/request.md",
        "metadata": {
            "request_id": "req-1",
            "project": "pytest-project",
            "workflow": "full_modeling",
            "data_source_mode": "local_feather",
            "target_column": "target",
            "id_columns": ["uid", "sample_date"],
            "time_column": "apply_time",
            "period_column": "apply_month",
            "split_column": "final_flag",
            "sample_location": "data/raw/model.feather",
            "splits": {
                "dev": {"values": ["DEV"]},
                "oos": {"values": ["DEV-OOS"]},
                "oot": {"values": ["OOT"]},
            },
            "step_params": {
                "missing_rate_filter": {"threshold": 0.88},
                "constant_value_filter": {"max_unique_values": 2},
                "iv_filter": {"min_iv": 0.01},
                "psi_filter": {"max_psi": 0.2},
                "correlation_dedup": {"method": "spearman", "max_abs_corr": 0.75},
                "baseline_importance_filter": {"keep_top_n": 300},
                "scale_pos_weight": {"mode": "negative_over_positive"},
            },
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {
                    "max_rounds": 2,
                    "candidates_per_round": 3,
                    "max_trials": 6,
                },
            },
            "experiments": [
                {"name": "baseline_all", "method": "xgboost", "segment": "all"},
                {"name": "baseline_e2e3", "method": "logistic_regression", "segment": "e2e3"},
            ],
            "evaluation": {
                "metrics": ["auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"],
                "champions": ["score_v1", "score_v2"],
                "comparison_dimensions": ["final_flag", "apply_month"],
                "risk_profile_dimensions": ["blue_customer_flag", "zc_level"],
            },
            "reports": {
                "sections": ["sample_overview", "model_performance", "risk_profile"],
                "outputs": ["model_report.md", "model_report.html", "model_card.md", "executive_summary.md"],
            },
        },
        "body": "",
    }


def test_materialize_request_runtime_configs_maps_builder_fields(tmp_path):
    project_dir = tmp_path / "project"
    run_dir = project_dir / "runs" / "run1"
    _write_project(project_dir)

    paths = materialize_request_runtime_configs(request_doc=_request_doc(), project_dir=project_dir, run_dir=run_dir)

    assert set(paths) >= {"project.yml", "sample.yaml", "feature_select.yaml", "refine_features.yaml", "train.yaml", "evaluate.yaml", "report.yaml"}
    runtime_project = yaml.safe_load((run_dir / "configs_runtime" / "project.yml").read_text(encoding="utf-8"))
    feature_select = yaml.safe_load((run_dir / "configs_runtime" / "feature_select.yaml").read_text(encoding="utf-8"))
    refine = yaml.safe_load((run_dir / "configs_runtime" / "refine_features.yaml").read_text(encoding="utf-8"))
    train = yaml.safe_load((run_dir / "configs_runtime" / "train.yaml").read_text(encoding="utf-8"))
    evaluate = yaml.safe_load((run_dir / "configs_runtime" / "evaluate.yaml").read_text(encoding="utf-8"))
    report = yaml.safe_load((run_dir / "configs_runtime" / "report.yaml").read_text(encoding="utf-8"))

    assert runtime_project["data"]["raw_path"] == "data/raw/model.feather"
    assert runtime_project["data"].get("source_table") is None
    assert runtime_project["request"]["data_source_mode"] == "local_feather"
    assert runtime_project["request"]["sample_location"] == "data/raw/model.feather"
    request_runtime = yaml.safe_load((run_dir / "configs_runtime" / "request_runtime.yaml").read_text(encoding="utf-8"))
    assert request_runtime["data_source_mode"] == "local_feather"
    assert request_runtime["sample_location"] == "data/raw/model.feather"
    assert feature_select["feature_select"]["prescreen"]["output_dir"] == str(run_dir / "feature_selection" / "prescreen")
    assert feature_select["feature_select"]["wide_table"]["remain_features_path"] == str(
        run_dir / "feature_selection" / "prescreen" / "results" / "prescreen_final_remain_features.json"
    )
    assert feature_select["feature_select"]["wide_table"]["feature_map_output"] == str(run_dir / "feature_selection" / "prescreen_wide_feature_map.csv")
    assert refine["feature_refine"]["output_dir"] == str(run_dir / "feature_selection" / "refine")
    assert refine["feature_refine"]["dp_feather"]["data_dir"] == str(run_dir / "data" / "dp_feather" / "feature_refine")
    assert refine["feature_refine"]["input"]["local_feather_path"] == "data/raw/model.feather"
    assert runtime_project["split"]["oos_values"] == ["DEV-OOS"]
    assert refine["feature_refine"]["preprocessing"]["min_non_null_rate"] == 0.12
    assert refine["feature_refine"]["preprocessing"]["max_unique_values"] == 2
    assert refine["feature_refine"]["global_corr"]["threshold"] == 0.75
    assert refine["feature_refine"]["target_feature_count"] == 300
    assert train["training"]["valid_values"] == ["DEV-OOS"]
    assert train["training"]["experiments"][1]["algorithm"] == "logistic_regression"
    assert train["training"]["experiments"][1]["segment_filter"] == "blue_customer_flag in ['E2', 'E3']"
    assert train["training"]["runtime_step_params"]["scale_pos_weight"]["mode"] == "negative_over_positive"
    assert train["training"]["mode"] == "llm_guided_tune"
    assert train["training"]["tuning"]["max_rounds"] == 2
    assert train["training"]["tuning"]["candidates_per_round"] == 3
    assert evaluate["metrics"] == ["auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"]
    assert evaluate["evaluation"]["score_columns"] == ["model_score", "score_v1", "score_v2"]
    assert "zc_level" in evaluate["evaluation"]["segment_columns"]
    assert report["report"]["output_formats"] == ["markdown", "html"]


def test_materialize_remote_table_mode_overrides_project_raw_path(tmp_path):
    project_dir = tmp_path / "project"
    run_dir = project_dir / "runs" / "run1"
    _write_project(project_dir)
    project_yml = project_dir / "project.yml"
    project_yml.write_text(project_yml.read_text(encoding="utf-8") + "  raw_path: data/raw/stale.feather\n", encoding="utf-8")
    request_doc = _request_doc()
    request_doc["metadata"]["data_source_mode"] = "remote_table"
    request_doc["metadata"]["sample_location"] = "mart.request_sample"

    materialize_request_runtime_configs(request_doc=request_doc, project_dir=project_dir, run_dir=run_dir)

    runtime_project = yaml.safe_load((run_dir / "configs_runtime" / "project.yml").read_text(encoding="utf-8"))
    feature_select = yaml.safe_load((run_dir / "configs_runtime" / "feature_select.yaml").read_text(encoding="utf-8"))
    refine = yaml.safe_load((run_dir / "configs_runtime" / "refine_features.yaml").read_text(encoding="utf-8"))
    train = yaml.safe_load((run_dir / "configs_runtime" / "train.yaml").read_text(encoding="utf-8"))
    assert runtime_project["data"]["source_table"] == "mart.request_sample"
    assert runtime_project["data"].get("raw_path") is None
    assert runtime_project["request"]["data_source_mode"] == "remote_table"
    assert feature_select["feature_select"]["wide_table"]["base_table"] == "mart.request_sample"
    assert feature_select["feature_select"]["wide_table"]["sql_output"] == str(run_dir / "queries" / "06_build_prescreen_wide_table.sql")
    assert refine["feature_refine"]["input"]["feature_map"] == str(run_dir / "feature_selection" / "prescreen_wide_feature_map.csv")
    assert refine["feature_refine"]["input"]["wide_table"] == "mart.wide"
    assert "feather_path" not in train["input"]


def test_materialize_refine_only_remote_table_uses_request_table_as_wide_input(tmp_path):
    project_dir = tmp_path / "project"
    run_dir = project_dir / "runs" / "run1"
    _write_project(project_dir)
    request_doc = _request_doc()
    request_doc["metadata"]["data_source_mode"] = "remote_table"
    request_doc["metadata"]["sample_location"] = "pdm_risk.large_wide_table"
    request_doc["metadata"]["feature_selection"] = {"rounds": ["refine"]}

    materialize_request_runtime_configs(request_doc=request_doc, project_dir=project_dir, run_dir=run_dir)

    refine = yaml.safe_load((run_dir / "configs_runtime" / "refine_features.yaml").read_text(encoding="utf-8"))
    assert refine["feature_refine"]["input"]["wide_table"] == "pdm_risk.large_wide_table"
    assert refine["feature_refine"]["input"].get("feature_map") is None


def test_run_init_materializes_and_registers_runtime_configs(tmp_path):
    project_dir = tmp_path / "project"
    _write_project(project_dir)
    request_path = project_dir / "request.md"
    request_path.write_text("---\n" + yaml.safe_dump(_request_doc()["metadata"], allow_unicode=True, sort_keys=False) + "---\n", encoding="utf-8")

    assert main(["run", "init", "--project", str(project_dir), "--workflow", "full_modeling", "--run-id", "run1", "--request", str(request_path)]) == 0

    run_dir = project_dir / "runs" / "run1"
    assert (run_dir / "configs_runtime" / "train.yaml").exists()
    manifest = json.loads((run_dir / "audit" / "artifact_manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {item["path"] for item in manifest["artifacts"]}
    assert "configs_runtime" in manifest_paths
    assert "configs_runtime/train.yaml" in manifest_paths
