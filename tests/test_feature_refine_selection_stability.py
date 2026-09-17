import numpy as np
import pandas as pd

from risk_model_workbench import feature_refine
from risk_model_workbench.feature_refine import (
    DatasetParts,
    d03_random_importance,
    d04_null_importance,
    d05_top_importance,
    make_selection_dataset_parts,
)
from risk_model_workbench.feature_selection.risk_review import (
    build_feature_decision_ledger,
    importance_stability_evidence,
)


class _ImportanceModel:
    def __init__(self, features, gains):
        self.features = list(features)
        self.gains = gains

    def feature_importance(self, importance_type):
        if importance_type == "split":
            return [max(0, int(self.gains.get(feature, 0))) for feature in self.features]
        return [float(self.gains.get(feature, 0.0)) for feature in self.features]


def _base_cfg():
    return {
        "random_seed": 0,
        "input": {"label_column": "label", "split_column": "split", "train_value": "DEV", "valid_value": "OOT"},
        "selection_split": {"mode": "dev_temporal", "month_column": "month", "valid_months": 1},
        "d03_random_importance": {
            "enabled": True,
            "mode": "feature_select_v2",
            "bagging_rounds": 1,
            "bagging_fraction": 1.0,
            "thresholds": None,
            "importance_types": ["split", "gain"],
        },
        "d04_null_importance": {"enabled": True, "real_rounds": 1, "null_rounds": 1, "score_threshold": 0.9},
        "d05_baseline_importance": {"enabled": True, "keep_top_n": 1, "seeds": [0, 17, 42, 73, 101]},
        "feature_risk_review": {"force_keep_features": []},
        "lightgbm": {},
    }


def _selection_frame(oot_value: float) -> pd.DataFrame:
    rows = []
    for month, split in [("2025-01", "DEV"), ("2025-02", "DEV"), ("2025-03", "DEV")]:
        for index in range(4):
            rows.append({"split": split, "label": index % 2, "month": month, "strong": index + 1, "weak": 4 - index})
    for index in range(4):
        rows.append({"split": "OOT", "label": index % 2, "month": "2025-04", "strong": oot_value, "weak": -oot_value})
    return pd.DataFrame(rows)


def test_selection_temporal_split_uses_last_dev_month_and_never_oot():
    raw = _selection_frame(999999.0)
    parts, details = make_selection_dataset_parts(raw, raw[["strong", "weak"]], _base_cfg())

    assert details["mode"] == "dev_temporal"
    assert details["train_months"] == ["2025-01", "2025-02"]
    assert details["valid_months"] == ["2025-03"]
    assert details["oot_used"] is False
    assert parts.valid_x["strong"].max() < 100


def test_oot_changes_do_not_change_d03_d04_d05_or_final_features(monkeypatch):
    cfg = _base_cfg()
    seen_valid_max = []

    def fake_d03_train(train_x, train_y, features, config, seed):
        assert train_x["strong"].abs().max() < 100
        gains = {feature: 30.0 if feature == "strong" else 15.0 if feature == "weak" else 10.0 for feature in features}
        return _ImportanceModel(features, gains), 0.7

    def fake_train(parts, features, config, seed):
        seen_valid_max.append(float(parts.valid_x.abs().max().max()))
        gains = {"strong": 20.0, "weak": 10.0}
        return _ImportanceModel(features, gains), 0.7

    monkeypatch.setattr(feature_refine, "train_feature_select_v2_model", fake_d03_train)
    monkeypatch.setattr(feature_refine, "train_lgbm", fake_train)

    def run(oot_value):
        raw = _selection_frame(oot_value)
        parts, _ = make_selection_dataset_parts(raw, raw[["strong", "weak"]], cfg)
        d03_features, d03 = d03_random_importance(parts, ["strong", "weak"], cfg)
        d04_features, d04 = d04_null_importance(parts, d03_features, cfg)
        final_features, d05, _ = d05_top_importance(parts, d04_features, cfg)
        return d03, d04, d05, final_features

    result_a = run(1000.0)
    result_b = run(-1000000.0)

    pd.testing.assert_frame_equal(result_a[0].reset_index(drop=True), result_b[0].reset_index(drop=True))
    pd.testing.assert_frame_equal(result_a[1].reset_index(drop=True), result_b[1].reset_index(drop=True))
    pd.testing.assert_frame_equal(result_a[2].reset_index(drop=True), result_b[2].reset_index(drop=True))
    assert result_a[3] == result_b[3]
    assert max(seen_valid_max) < 100


def test_d05_multi_seed_stability_is_deterministic_and_does_not_promote_one_off_rank_one(monkeypatch):
    parts = DatasetParts(
        train_x=pd.DataFrame({"one_off": [1, 2, 3, 4], "stable": [2, 3, 4, 5], "other": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"one_off": [2, 3], "stable": [3, 4], "other": [3, 2]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = _base_cfg()

    def fake_train(parts, features, config, seed):
        gains = {"one_off": 100.0, "stable": 10.0, "other": 1.0} if seed == 0 else {"one_off": 0.0, "stable": 100.0, "other": 10.0}
        return _ImportanceModel(features, gains), 0.7

    monkeypatch.setattr(feature_refine, "train_lgbm", fake_train)
    first = d05_top_importance(parts, ["one_off", "stable", "other"], cfg)
    second = d05_top_importance(parts, ["one_off", "stable", "other"], cfg)

    assert first[0] == second[0] == ["stable"]
    pd.testing.assert_frame_equal(first[1], second[1])
    evidence = first[1].set_index("feature")
    assert evidence.loc["one_off", "top_n_selection_rate"] == 0.2
    assert evidence.loc["stable", "top_n_selection_rate"] == 0.8
    assert evidence.loc["stable", "final_rank"] == 1
    assert evidence.loc["one_off", "final_rank"] > 1


def test_force_keep_is_retained_and_unstable_evidence_is_visible_in_ledger(monkeypatch):
    parts = DatasetParts(
        train_x=pd.DataFrame({"force": [1, 2, 3, 4], "stable": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"force": [2, 3], "stable": [3, 2]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = _base_cfg()
    cfg["feature_risk_review"]["force_keep_features"] = ["force"]
    cfg["d05_baseline_importance"].update({"min_selection_rate": 0.8, "max_rank_std": 0.1})

    def fake_train(parts, features, config, seed):
        gains = {"force": 100.0, "stable": 1.0} if seed == 0 else {"force": 0.0, "stable": 100.0}
        return _ImportanceModel(features, gains), 0.7

    monkeypatch.setattr(feature_refine, "train_lgbm", fake_train)
    selected, d05, _ = d05_top_importance(parts, ["force", "stable"], cfg)
    assert "force" in selected
    assert d05.set_index("feature").loc["force", "importance_status"] == "unstable_importance"

    importance = importance_stability_evidence(["force", "stable"], pd.DataFrame(), d05)
    ledger = build_feature_decision_ledger(
        ["force", "stable"], selected_features=selected, force_keep_features=["force"],
        preprocess_stats=pd.DataFrame([{"feature": feature, "non_null_rate": 1.0, "drop_reason": ""} for feature in ["force", "stable"]]),
        d01_detail=pd.DataFrame([{"feature": feature, "iv": 0.1, "drop_reason": "kept"} for feature in ["force", "stable"]]),
        corr_drops=pd.DataFrame(), leakage=pd.DataFrame([{"feature_name": feature, "leakage_status": "passed", "leakage_reason": "passed"} for feature in ["force", "stable"]]),
        psi_summary=pd.DataFrame(), patterns=pd.DataFrame(), relationship=pd.DataFrame(), importance=importance,
    ).set_index("feature")
    assert ledger.loc["force", "final_status"] == "force_kept"
    assert "force_kept" in ledger.loc["force", "decision_reason"]
    assert "importance_unstable_review" in ledger.loc["force", "decision_reason"]


def test_risk_review_prefers_formal_d05_stability_evidence_over_d03_detail():
    d03 = pd.DataFrame([
        {"round": 0, "feature": "x", "gain": 100.0},
        {"round": 1, "feature": "x", "gain": 100.0},
    ])
    d05 = pd.DataFrame([
        {
            "feature": "x", "importance_median": 2.0, "importance_mean": 2.0, "importance_std": 0.0,
            "rank_median": 7.0, "rank_mean": 7.0, "rank_std": 0.0, "top_n_selection_rate": 0.8,
            "runs": 5, "final_rank": 3, "selected": True, "importance_status": "stable_importance",
        }
    ])

    result = importance_stability_evidence(["x"], d03, d05)
    assert result.loc[0, "importance_median"] == 2.0
    assert result.loc[0, "rank_median"] == 7.0
    assert result.loc[0, "importance_runs"] == 5
