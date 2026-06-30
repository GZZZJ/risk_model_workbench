"""Regression guard for the feature-screening summary table field mapping.

The fallback path of ``_screening_steps_frame`` reads ``stage_summary.json`` (produced
by feature_refine). Its keys must match what refine actually writes, otherwise the
report's "剩余变量个数" column shows N/A for steps that genuinely have counts.
"""

from pathlib import Path

from risk_model_workbench.reporting.excel_report import _screening_steps_frame


def _refine_stage_summary() -> dict:
    return {
        "initial_features": 2837,
        "available_features": 2563,
        "after_global_corr": 1852,
        "after_d03_random_importance": 1852,
        "after_d04_null_importance": 1028,
        "final_features": 500,
        "d03_mode": "feature_select_v2",
    }


def test_screening_steps_frame_maps_refine_stage_summary_keys(tmp_path):
    feature_dir = tmp_path / "feature_selection"
    feature_dir.mkdir()  # no feature_screening_process.json → fallback to stage_summary

    frame = _screening_steps_frame(_refine_stage_summary(), feature_dir)
    by_method = dict(zip(frame["筛选方法"], frame["剩余变量个数"]))

    # Refine-backed steps must show real counts, not N/A.
    assert by_method["原始候选变量总数"] == 2837
    assert by_method["Feather观察样本可用特征"] == 2563
    assert by_method["全局相关性去重：按单变量AUC保留更强特征"] == 1852
    assert by_method["空标签重要性筛选：保留显著高于空标签分布的特征"] == 1028
    assert by_method["最终训练特征"] == 500


def test_screening_steps_frame_prescreen_steps_absent_in_local_feather(tmp_path):
    """d01 (分表预筛) / d02 (稳定性 PSI) are prescreen-side; in local_feather mode
    prescreen runs by-design without them, so they honestly stay N/A — not a bug."""
    feature_dir = tmp_path / "feature_selection"
    feature_dir.mkdir()

    frame = _screening_steps_frame(_refine_stage_summary(), feature_dir)
    by_method = dict(zip(frame["筛选方法"], frame["剩余变量个数"]))

    assert by_method["分表基础预筛：缺失率、相关性、IV"] == "N/A"
    assert by_method["稳定性筛选：DEV vs OOT PSI"] == "N/A"
