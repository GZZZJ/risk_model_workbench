"""Local-feather by-design handling for feature_prescreen and build_wide_sql.

In local_feather mode the wide table pre-exists as the feather file, so
``feature_prescreen`` and ``build_wide_sql`` must finish as ``done`` (not
``scaffold``/``failed``) and emit the contract-accepted placeholder artifacts,
so the audit verdict converges to ``complete``.
"""

from pathlib import Path

import pytest
import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.project_state import audit_run


def _build_local_feather_project(tmp_path: Path, run_id: str) -> tuple[Path, Path]:
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
                "  oos_values: [DEV-OOS]",
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
                "  base_columns: [uid, apply_time, apply_month, final_flag, target]",
                "preprocessing:",
                "  drop_constant: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "feature_select.yaml").write_text(
        "feature_select:\n  wide_table:\n    output_table: mart.wide\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "refine_features.yaml").write_text(
        "feature_refine:\n  output_dir: runs/refine\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "evaluate.yaml").write_text(
        "evaluation:\n  score_columns: [model_score]\n",
        encoding="utf-8",
    )
    (project_dir / "configs" / "report.yaml").write_text(
        "report:\n  outputs: [model_report.md]\n",
        encoding="utf-8",
    )

    rows = []
    for idx in range(120):
        split = "DEV" if idx < 70 else "OOT"
        target = 1 if idx % 5 in {0, 1} else 0
        rows.append(
            {
                "uid": idx,
                "apply_time": f"2026-0{1 + (idx // 40)}-01",
                "apply_month": f"2026-0{1 + (idx // 40)}",
                "final_flag": split,
                "target": target,
                "feat_a": target + idx / 1000,
                "feat_b": (idx % 7) / 10,
            }
        )
    feather_path = project_dir / "data" / "raw" / "model.feather"
    pd.DataFrame(rows).to_feather(feather_path)

    request_meta = {
        "request_id": "lf-by-design",
        "project": "pytest-project",
        "workflow": "full_modeling",
        "target_column": "target",
        "id_columns": ["uid"],
        "time_column": "apply_time",
        "period_column": "apply_month",
        "split_column": "final_flag",
        "data_source_mode": "local_feather",
        "sample_location": str(feather_path),
        "experiments": [{"name": "logit_all", "method": "logistic_regression", "segment": "all"}],
        "evaluation": {"metrics": ["auc", "ks"], "champions": []},
        "reports": {"sections": ["model_performance"], "outputs": ["model_report.md"]},
    }
    request_path = project_dir / "request.md"
    request_path.write_text(
        "---\n" + yaml.safe_dump(request_meta, allow_unicode=True, sort_keys=False) + "---\n",
        encoding="utf-8",
    )

    assert (
        main(
            [
                "run",
                "init",
                "--project",
                str(project_dir),
                "--workflow",
                "full_modeling",
                "--run-id",
                run_id,
                "--request",
                str(request_path),
            ]
        )
        == 0
    )
    return project_dir, project_dir / "runs" / run_id


def _stage_status(run_dir: Path, stage: str) -> str:
    state = yaml.safe_load((run_dir / "run_state.yml").read_text(encoding="utf-8"))
    return (state.get("stages") or {}).get(stage, {}).get("status")


def test_build_wide_sql_local_feather_is_done_by_design(tmp_path):
    project_dir, run_dir = _build_local_feather_project(tmp_path, run_id="lfwidesql")

    # Local short-circuit must not depend on a prescreen remain-features artifact.
    assert main(["build-wide-sql", "--project", str(project_dir), "--run-id", "lfwidesql"]) == 0

    assert _stage_status(run_dir, "build_wide_sql") == "done"
    assert (run_dir / "feature_selection" / "wide_table_skipped.json").exists()

    audit = audit_run(project_dir, "lfwidesql", stage="build_wide_sql")
    assert audit["stages"][0]["verdict"] == "complete"


def test_feature_prescreen_local_feather_is_done_by_design(tmp_path):
    project_dir, run_dir = _build_local_feather_project(tmp_path, run_id="lfprescreen")

    assert main(["feature", "prescreen", "--project", str(project_dir), "--run-id", "lfprescreen"]) == 0

    assert _stage_status(run_dir, "feature_prescreen") == "done"
    for name in (
        "data_source_contract.json",
        "resource_plan.json",
        "sampling_plan.json",
        "batch_plan.json",
    ):
        assert (run_dir / "feature_selection" / name).exists(), name

    audit = audit_run(project_dir, "lfprescreen", stage="feature_prescreen")
    assert audit["stages"][0]["verdict"] == "complete"
