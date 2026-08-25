"""Regression guards for the feature-screening summary table.

Covers: refine stage_summary key mapping, d01/d02 rendering when present,
local-feather row ordering (monotonic funnel), and the precedence trap where
feature_screening_process.json shadows stage_summary.
"""

import json
from pathlib import Path

from risk_model_workbench.reporting.excel_report import _screening_steps_frame


def _stage_summary_local() -> dict:
    return {
        "data_source_mode": "local_feather",
        "initial_features": 2837,
        "available_features": 2563,
        "d01_kept_features": 2400,
        "d02_kept_features": 2400,
        "after_global_corr": 1852,
        "after_d03_random_importance": 1852,
        "after_d04_null_importance": 1028,
        "final_features": 500,
        "d03_mode": "feature_select_v2",
    }


def _stage_summary_remote_no_d01_d02() -> dict:
    return {
        "data_source_mode": "remote_table",
        "initial_features": 2837,
        "available_features": 2563,
        "after_global_corr": 1852,
        "after_d03_random_importance": 1852,
        "after_d04_null_importance": 1028,
        "final_features": 500,
        "d03_mode": "feature_select_v2",
    }


def test_screening_steps_renders_d01_d02_counts_in_local_feather(tmp_path):
    feature_dir = tmp_path / "fs"
    feature_dir.mkdir()

    frame = _screening_steps_frame(_stage_summary_local(), feature_dir)
    by_method = dict(zip(frame["筛选方法"], frame["剩余变量个数"]))

    assert by_method["原始候选变量总数"] == 2837
    assert by_method["Feather观察样本可用特征"] == 2563
    # d01/d02 now computed locally → real counts, with honest "全表" labels
    assert by_method["分表基础预筛：缺失率、相关性、IV（全表，local feather）"] == 2400
    assert by_method["稳定性审查：DEV首月Base月度PSI（仅证据，local feather）"] == 2400
    assert by_method["选取Top500特征入模"] == 500


def test_screening_steps_local_feather_funnel_is_monotonic(tmp_path):
    feature_dir = tmp_path / "fs"
    feature_dir.mkdir()

    frame = _screening_steps_frame(_stage_summary_local(), feature_dir)
    counts = [c for c in frame["剩余变量个数"] if isinstance(c, int)]
    assert counts == sorted(counts, reverse=True), f"funnel not monotone: {counts}"


def test_screening_steps_missing_d01_d02_keys_show_na_remote(tmp_path):
    """Remote stage_summary without d01_kept_features/d02_kept_features → N/A (honest)."""
    feature_dir = tmp_path / "fs"
    feature_dir.mkdir()

    frame = _screening_steps_frame(_stage_summary_remote_no_d01_d02(), feature_dir)
    by_method = dict(zip(frame["筛选方法"], frame["剩余变量个数"]))
    # remote legacy label, key absent → N/A
    assert by_method["分表基础预筛：缺失率、相关性、IV"] == "N/A"
    assert by_method["稳定性审查：DEV首月Base月度PSI（仅证据）"] == "N/A"


def test_screening_steps_shadowed_by_feature_screening_process(tmp_path):
    """Precedence trap guard: feature_screening_process.json shadows stage_summary."""
    feature_dir = tmp_path / "fs"
    feature_dir.mkdir()
    (feature_dir / "feature_screening_process.json").write_text(
        json.dumps({"screening_rows": [{"step": 1, "method": "shadow", "remaining_features": 99}]}),
        encoding="utf-8",
    )

    frame = _screening_steps_frame(_stage_summary_local(), feature_dir)
    # process.json branch wins over stage_summary
    assert 99 in list(frame["剩余变量个数"])
    assert 2400 not in list(frame["剩余变量个数"])
