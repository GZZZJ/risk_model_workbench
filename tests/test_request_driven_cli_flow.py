from pathlib import Path

import pytest
import yaml

from risk_model_workbench.cli import main


def test_request_driven_synthetic_flow_runs_local_outputs(tmp_path):
    pytest.importorskip("pyarrow")
    pytest.importorskip("sklearn")
    import pandas as pd

    project_dir = tmp_path / "project"
    (project_dir / "configs").mkdir(parents=True)
    (project_dir / "data" / "raw").mkdir(parents=True)
    (project_dir / "project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: pytest-project",
                "  display_name: Pytest Project",
                "  scenario: pytest",
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
                "  oot_values: [OOT]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "train.yaml").write_text(
        "\n".join(
            [
                "training:",
                "  default_algorithm: logistic_regression",
                "  random_seed: 0",
                "input:",
                "  base_columns: [uid, apply_time, apply_month, final_flag, target, blue_customer_flag, zc_level]",
                "preprocessing:",
                "  drop_constant: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "feature_select.yaml").write_text("feature_select:\n  wide_table:\n    output_table: mart.wide\n", encoding="utf-8")
    (project_dir / "configs" / "refine_features.yaml").write_text("feature_refine:\n  output_dir: runs/refine\n", encoding="utf-8")
    (project_dir / "configs" / "evaluate.yaml").write_text("evaluation:\n  score_columns: [model_score]\n", encoding="utf-8")
    (project_dir / "configs" / "report.yaml").write_text("report:\n  outputs: [model_report.md]\n", encoding="utf-8")

    rows = []
    for idx in range(120):
        if idx < 60:
            split = "DEV"
        elif idx < 90:
            split = "DEV-OOS"
        else:
            split = "OOT"
        target = 1 if idx % 5 in {0, 1} else 0
        rows.append(
            {
                "uid": idx,
                "apply_time": f"2026-0{1 + (idx // 40)}-01",
                "apply_month": f"2026-0{1 + (idx // 40)}",
                "final_flag": split,
                "target": target,
                "blue_customer_flag": "E2" if idx % 2 else "B2",
                "zc_level": "A" if idx % 3 else "B",
                "feat_a": target + idx / 1000,
                "feat_b": (idx % 7) / 10,
                "score_v1": target * 0.6 + (idx % 10) / 100,
            }
        )
    pd.DataFrame(rows).to_feather(project_dir / "data" / "raw" / "model.feather")

    request_meta = {
        "request_id": "synthetic-request",
        "project": "pytest-project",
        "workflow": "full_modeling",
        "target_column": "target",
        "id_columns": ["uid"],
        "time_column": "apply_time",
        "period_column": "apply_month",
        "split_column": "final_flag",
        "sample_location": "data/raw/model.feather",
        "experiments": [{"name": "logit_all", "method": "logistic_regression", "segment": "all"}],
        "evaluation": {
            "metrics": ["auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"],
            "champions": ["score_v1"],
            "comparison_dimensions": ["final_flag"],
            "risk_profile_dimensions": ["blue_customer_flag", "zc_level"],
        },
        "reports": {
            "model_display_name": "Pytest Challenger",
            "score_labels": {"score_v1": "Champion V1"},
            "sections": ["model_performance", "risk_profile"],
            "outputs": ["model_report.md", "model_report.html"],
            "targets": [
                {
                    "name": "tuned_logit",
                    "experiment": "logit_all",
                    "output_dir": "reports_tuned_logit",
                }
            ],
        },
    }
    request_path = project_dir / "request.md"
    request_path.write_text("---\n" + yaml.safe_dump(request_meta, allow_unicode=True, sort_keys=False) + "---\n", encoding="utf-8")

    assert main(["run", "init", "--project", str(project_dir), "--workflow", "full_modeling", "--run-id", "run1", "--request", str(request_path)]) == 0
    run_dir = project_dir / "runs" / "run1"
    (run_dir / "feature_selection").mkdir(exist_ok=True)
    (run_dir / "feature_selection" / "final_features.txt").write_text("feat_a\nfeat_b\n", encoding="utf-8")

    assert main(["sample", "check", "--project", str(project_dir), "--run-id", "run1"]) == 0
    assert main(["train", "--project", str(project_dir), "--run-id", "run1", "--experiment", "logit_all"]) == 0
    assert main(["evaluate", "--project", str(project_dir), "--run-id", "run1"]) == 0
    assert main(["compare", "--project", str(project_dir), "--run-id", "run1"]) == 0
    assert main(["report", "--project", str(project_dir), "--run-id", "run1"]) == 0

    assert (run_dir / "modeling" / "logit_all" / "model.pkl").exists()
    assert (run_dir / "evaluation" / "dimension_metrics.csv").exists()
    assert (run_dir / "evaluation" / "champion_challenger.json").exists()
    assert (run_dir / "reports" / "model_report.html").exists()
    assert (run_dir / "reports_tuned_logit" / "model_report.html").exists()
    tuned_html = (run_dir / "reports_tuned_logit" / "model_report.html").read_text(encoding="utf-8")
    assert "Pytest Challenger" in tuned_html
    scope = yaml.safe_load((run_dir / "reports_tuned_logit" / "report_scope.json").read_text(encoding="utf-8"))
    assert scope["target"] == "tuned_logit"
    assert scope["train_dir"] == "modeling/logit_all"
