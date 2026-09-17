import json

import numpy as np
import pandas as pd

from risk_model_workbench.feature_refine import DatasetParts, d02_local_psi
from risk_model_workbench.feature_selection.leakage_check import review_feature_leakage
from risk_model_workbench.feature_selection.risk_review import (
    MISSING_BUCKET,
    badrate_evidence,
    build_feature_decision_ledger,
    generate_feature_risk_review,
    importance_stability_evidence,
    monthly_badrate_evidence,
    monthly_psi_evidence,
)


def test_base_month_bins_are_reused_for_later_months_and_missing_is_bucketed():
    base = np.arange(100, dtype=float)
    compare = np.arange(100, 200, dtype=float)
    frame = pd.DataFrame({"feature_a": np.r_[base, compare, np.nan]})
    months = pd.Series(["2025-01"] * 100 + ["2025-02"] * 101)

    detail, _, specs, _ = monthly_psi_evidence(frame, months, ["feature_a"], min_base_samples=10)

    assert detail.loc[0, "base_month"] == "2025-01"
    assert detail.loc[0, "compare_month"] == "2025-02"
    assert detail.loc[0, "psi"] >= 0.25
    assert max(edge for edge in specs["feature_a"].edges if np.isfinite(edge)) < 100
    bins, _ = badrate_evidence(frame, pd.Series(([0, 1] * 100) + [0]), ["feature_a"], specs, min_bin_samples=1)
    assert MISSING_BUCKET in set(bins["bin"])


def test_oot_does_not_participate_in_feature_selection_psi():
    train_x = pd.DataFrame({"x": np.r_[np.arange(50), np.arange(50)]})
    train_y = pd.Series([0, 1] * 50)
    months = pd.Series(["2025-01"] * 50 + ["2025-02"] * 50)
    stable_oot = DatasetParts(train_x, train_y, pd.DataFrame({"x": np.arange(100)}), train_y)
    shifted_oot = DatasetParts(train_x, train_y, pd.DataFrame({"x": np.arange(100) + 10000}), train_y)
    cfg = {"local_d02": {"enabled": True, "warning_threshold": 0.1, "fail_threshold": 0.25, "min_base_samples": 10}}
    cfg["_runtime_dev_months"] = months

    kept_a, detail_a = d02_local_psi(stable_oot, ["x"], cfg)
    kept_b, detail_b = d02_local_psi(shifted_oot, ["x"], cfg)

    assert kept_a == kept_b == ["x"]
    pd.testing.assert_frame_equal(detail_a, detail_b)


def test_remote_prescreen_d02_is_also_dev_monthly_and_never_calls_vendor_oot_psi():
    from risk_model_workbench.batch_feature_select import run_d02

    frame = pd.DataFrame(
        {
            "split": ["DEV"] * 8 + ["OOT"] * 4,
            "ds": ["2025-01-01"] * 4 + ["2025-02-01"] * 4 + ["2025-03-01"] * 4,
            "x": [1, 2, 3, 4, 1, 2, 8, 9, 1000, 2000, 3000, 4000],
        }
    )

    result, max_psi, dropped = run_d02(
        frame,
        ["x"],
        split_col="split",
        train_value="DEV",
        valid_value="OOT",
        psi_threshold=0.1,
        batch_psi_func=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("vendor PSI must not run")),
    )

    assert result["mode"] == "dev_first_natural_month_base"
    assert result["oot_used"] is False
    assert set(max_psi) == {"x"}
    assert dropped == []


def test_small_base_month_generates_explicit_warning():
    frame = pd.DataFrame({"x": [1, 2, 3, 4]})
    months = pd.Series(["2025-01", "2025-01", "2025-02", "2025-02"])

    _, summary, _, warnings = monthly_psi_evidence(frame, months, ["x"], min_base_samples=10)

    assert warnings == ["base_month_sample_too_small:2025-01:2<10"]
    assert "base_month_sample_too_small" in summary.loc[0, "psi_warning"]


def test_monthly_badrate_risk_order_reversal_is_flagged_with_reused_bins():
    values = np.tile([1.0, 2.0, 3.0], 40)
    frame = pd.DataFrame({"x": np.r_[values, values]})
    months = pd.Series(["2025-01"] * len(values) + ["2025-02"] * len(values))
    base_y = np.tile([0, 0, 1], 40)
    reversed_y = np.tile([1, 0, 0], 40)
    _, _, specs, _ = monthly_psi_evidence(frame, months, ["x"], n_bins=3, min_base_samples=10)

    detail, summary = monthly_badrate_evidence(frame, pd.Series(np.r_[base_y, reversed_y]), months, ["x"], specs, min_month_bin_samples=1)

    assert bool(detail.loc[detail["month"] == "2025-02", "risk_order_reversal"].all())
    assert summary.loc[0, "relationship_stability"] == "unstable"
    assert set(detail["monthly_relationship_stability_flag"]) == {"unstable"}


def test_importance_stability_uses_multiple_runs_and_selection_rate():
    detail = pd.DataFrame(
        [
            {"round": 0, "feature": "stable", "gain": 10.0},
            {"round": 0, "feature": "unstable", "gain": 1.0},
            {"round": 1, "feature": "stable", "gain": 9.0},
        ]
    )

    result = importance_stability_evidence(["stable", "unstable"], detail, pd.DataFrame(), top_n=1, min_selection_rate=0.75)
    by_feature = result.set_index("feature")

    assert by_feature.loc["stable", "importance_status"] == "stable_importance"
    assert by_feature.loc["stable", "top_n_selection_rate"] == 1.0
    assert by_feature.loc["unstable", "importance_status"] == "unstable_importance"
    assert by_feature.loc["unstable", "top_n_selection_rate"] == 0.0


def test_available_after_prediction_is_deterministic_leakage_failure():
    metadata = [
        {
            "feature_name": "balance",
            "observation_time": "2025-01-01",
            "available_time": "2025-01-02",
            "label_window_start": "2025-01-01",
            "derivation_logic": "daily snapshot",
        }
    ]

    result = review_feature_leakage(["balance"], metadata)

    assert result.loc[0, "leakage_status"] == "failed"
    assert "available_time_after_prediction_time" in result.loc[0, "leakage_reason"]


def test_name_signal_is_warning_only_and_never_automatic_failure():
    metadata = [
        {
            "feature_name": "future_balance",
            "observation_time": "2025-01-01",
            "available_time": "2025-01-01",
            "label_window_start": "2025-02-01",
            "derivation_logic": "snapshot named future_balance from legacy source",
        }
    ]

    result = review_feature_leakage(["future_balance"], metadata)

    assert result.loc[0, "leakage_status"] == "warning"
    assert result.loc[0, "availability_check"] == "passed"
    assert "rule_based_signal:future" in result.loc[0, "leakage_reason"]


def _ledger(*, rank: float = 100, pattern: str = "irregular", selected: bool = True) -> pd.DataFrame:
    return build_feature_decision_ledger(
        ["x"],
        selected_features=["x"] if selected else [],
        preprocess_stats=pd.DataFrame([{"feature": "x", "non_null_rate": 1.0, "drop_reason": ""}]),
        d01_detail=pd.DataFrame([{"feature": "x", "iv": 0.1, "drop_reason": "kept"}]),
        corr_drops=pd.DataFrame(),
        leakage=pd.DataFrame([{"feature_name": "x", "leakage_status": "passed", "leakage_reason": "passed"}]),
        psi_summary=pd.DataFrame([{"feature": "x", "monthly_psi_max": 0.03, "monthly_psi_mean": 0.02, "psi_status": "stable"}]),
        patterns=pd.DataFrame([{"feature": "x", "badrate_pattern": pattern, "sparse_bin_flag": False}]),
        relationship=pd.DataFrame([{"feature": "x", "relationship_stability": "stable"}]),
        importance=pd.DataFrame(
            [{"feature": "x", "importance_median": 10.0, "rank_median": rank, "rank_std": 1.0, "top_n_selection_rate": 1.0, "importance_status": "stable_importance"}]
        ),
    )


def test_non_monotonic_badrate_does_not_automatically_drop_feature():
    ledger = _ledger(rank=100, pattern="irregular")

    assert ledger.loc[0, "final_status"] == "selected"
    assert not bool(ledger.loc[0, "review_required"])


def test_high_importance_and_irregular_badrate_requires_review_without_drop():
    ledger = _ledger(rank=1, pattern="irregular")

    assert ledger.loc[0, "final_status"] == "review_required"
    assert bool(ledger.loc[0, "review_required"])
    assert "high_importance_badrate_review" in ledger.loc[0, "decision_reason"]


def test_high_psi_requires_review_but_does_not_drop_selected_feature():
    rebuilt = build_feature_decision_ledger(
        ["x"],
        selected_features=["x"],
        preprocess_stats=pd.DataFrame([{"feature": "x", "non_null_rate": 1.0, "drop_reason": ""}]),
        d01_detail=pd.DataFrame([{"feature": "x", "iv": 0.1, "drop_reason": "kept"}]),
        corr_drops=pd.DataFrame(),
        leakage=pd.DataFrame([{"feature_name": "x", "leakage_status": "passed", "leakage_reason": "passed"}]),
        psi_summary=pd.DataFrame([{"feature": "x", "monthly_psi_max": 0.3, "monthly_psi_mean": 0.2, "psi_status": "high_drift"}]),
        patterns=pd.DataFrame([{"feature": "x", "badrate_pattern": "monotonic_increasing", "sparse_bin_flag": False}]),
        relationship=pd.DataFrame(),
        importance=pd.DataFrame([{"feature": "x", "rank_median": 100, "importance_status": "stable_importance"}]),
    )

    assert rebuilt.loc[0, "final_status"] == "review_required"
    assert "monthly_psi_high_drift" in rebuilt.loc[0, "decision_reason"]


def test_missing_metadata_is_unknown_not_passed():
    result = review_feature_leakage(["plain_feature"], None)

    assert result.loc[0, "leakage_status"] == "unknown_metadata"


def test_decision_ledger_has_one_final_record_for_every_initial_feature():
    features = ["a", "b", "c"]
    ledger = build_feature_decision_ledger(
        features,
        selected_features=["a"],
        preprocess_stats=pd.DataFrame(
            [
                {"feature": "a", "non_null_rate": 1.0, "drop_reason": ""},
                {"feature": "b", "non_null_rate": 0.0, "drop_reason": "low_non_null_rate"},
                {"feature": "c", "non_null_rate": 1.0, "drop_reason": ""},
            ]
        ),
        d01_detail=pd.DataFrame([{"feature": "a", "iv": 0.2, "drop_reason": "kept"}, {"feature": "c", "iv": 0.0, "drop_reason": "low_iv"}]),
        corr_drops=pd.DataFrame(),
        leakage=review_feature_leakage(features, None),
        psi_summary=pd.DataFrame(),
        patterns=pd.DataFrame(),
        relationship=pd.DataFrame(),
        importance=pd.DataFrame(),
    )

    assert ledger["feature"].tolist() == features
    assert len(ledger) == len(features)
    assert ledger["final_status"].notna().all()
    assert ledger["decision_reason"].str.len().gt(0).all()


def test_generator_writes_machine_human_and_ledger_artifacts(tmp_path):
    raw = pd.DataFrame(
        {
            "split": ["DEV"] * 8 + ["OOT"] * 2,
            "label": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "month": ["2025-01"] * 4 + ["2025-02"] * 4 + ["2025-03"] * 2,
            "x": [1, 2, 3, 4, 1, 2, 7, 8, 100, 200],
        }
    )
    x = raw[["x"]].copy()
    cfg = {
        "input": {"split_column": "split", "label_column": "label", "train_value": "DEV", "valid_value": "OOT", "feature_map": None},
        "feature_risk_review": {"enabled": True, "month_column": "month", "psi": {"min_base_samples": 2}, "badrate": {"min_bin_samples": 1}, "importance": {"top_n": 1}},
    }

    final, artifacts, summary = generate_feature_risk_review(
        project_dir=tmp_path,
        output_dir=tmp_path / "feature_selection",
        raw_df=raw,
        feature_frame=x,
        config=cfg,
        initial_features=["x"],
        available_features=["x"],
        selected_features=["x"],
        preprocess_stats=pd.DataFrame([{"feature": "x", "non_null_rate": 1.0, "drop_reason": ""}]),
        d01_detail=pd.DataFrame([{"feature": "x", "iv": 0.2, "drop_reason": "kept"}]),
        corr_drops=pd.DataFrame(),
        d03_features=["x"],
        d03_detail=pd.DataFrame(),
        d04_features=["x"],
        d05_importance=pd.DataFrame([{"feature": "x", "gain": 1.0, "rank": 1}]),
    )

    assert final == ["x"]
    assert len(artifacts) == 9
    assert summary["oot_used_for_feature_psi"] is False
    assert (tmp_path / "feature_selection" / "feature_decision_ledger.csv").exists()
    payload = json.loads((tmp_path / "feature_selection" / "feature_risk_review.json").read_text())
    assert payload["schema_version"] == 1
    assert (tmp_path / "feature_selection" / "feature_risk_review.md").exists()
