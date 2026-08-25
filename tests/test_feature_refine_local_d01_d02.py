"""Local-feather d01 and DEV-monthly d02 compute tests.

These pin the refine-stage local prescreen so the report's d01/d02 rows show real
counts instead of N/A in local_feather mode. Pure functions on synthetic DataFrames —
no 23-min full refine run.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyarrow")


def _make_parts(n: int = 4000, seed: int = 0):
    """DatasetParts with an informative feature, a pure-noise feature, and a feature
    highly correlated to the informative one. OOT shifts are intentionally irrelevant to d02."""
    from risk_model_workbench.feature_refine import DatasetParts

    rng = np.random.RandomState(seed)
    f_info = rng.normal(size=n)
    f_noise = rng.normal(size=n)
    f_corr = f_info * 0.95 + rng.normal(scale=0.05, size=n)
    label = (f_info + rng.normal(scale=0.3, size=n) > 0.5).astype(int)

    train_x = pd.DataFrame({"f_info": f_info, "f_noise": f_noise, "f_corr": f_corr})
    train_y = pd.Series(label, name="y")

    # OOT: f_info/f_corr same distribution (low PSI); f_noise shifted (high PSI)
    f_info_o = rng.normal(size=n)
    f_noise_o = rng.normal(size=n) + 5.0
    f_corr_o = f_info_o * 0.95 + rng.normal(scale=0.05, size=n)
    valid_x = pd.DataFrame({"f_info": f_info_o, "f_noise": f_noise_o, "f_corr": f_corr_o})
    valid_y = pd.Series((f_info_o > 0.5).astype(int), name="y")
    return DatasetParts(train_x, train_y, valid_x, valid_y)


def test_d01_drops_low_iv_and_keeps_informative():
    from risk_model_workbench.feature_refine import d01_local_prescreen

    parts = _make_parts()
    cfg = {"local_d01": {"enabled": True, "iv": 0.02, "corr": 0.95, "n_bins": 10}}
    kept, detail = d01_local_prescreen(parts, ["f_info", "f_noise", "f_corr"], cfg)

    assert "f_info" in kept
    assert "f_noise" not in kept  # pure noise → IV ≈ 0 → dropped
    assert isinstance(detail, pd.DataFrame)
    reasons = dict(zip(detail["feature"], detail["drop_reason"]))
    assert reasons["f_noise"] == "low_iv"


def test_d01_corr_filter_keeps_higher_iv_feature():
    from risk_model_workbench.feature_refine import d01_local_prescreen

    parts = _make_parts()
    # iv threshold 0 → every feature survives IV; corr decides between f_info and f_corr
    cfg = {"local_d01": {"enabled": True, "iv": 0.0, "corr": 0.8, "n_bins": 10}}
    kept, detail = d01_local_prescreen(parts, ["f_info", "f_corr"], cfg)

    assert "f_info" in kept  # higher IV survives
    reasons = dict(zip(detail["feature"], detail["drop_reason"]))
    assert any(v == "high_corr" for v in reasons.values())


def test_d02_psi_is_dev_only_and_does_not_drop_features():
    from risk_model_workbench.feature_refine import d02_local_psi

    parts = _make_parts()
    cfg = {"local_d02": {"enabled": True, "warning_threshold": 0.1, "fail_threshold": 0.25, "min_base_samples": 10}}
    months = pd.Series(["2025-01"] * 2000 + ["2025-02"] * 2000)
    cfg["_runtime_dev_months"] = months
    kept, detail = d02_local_psi(parts, ["f_info", "f_noise"], cfg)

    assert kept == ["f_info", "f_noise"]
    assert isinstance(detail, pd.DataFrame)
    assert "max_psi" in detail.columns
    assert set(detail["drop_reason"]) == {"kept"}


def test_d02_high_psi_still_returns_all_features():
    from risk_model_workbench.feature_refine import d02_local_psi

    parts = _make_parts()
    cfg = {"local_d02": {"enabled": True, "warning_threshold": 0.0001, "fail_threshold": 0.0002}}
    months = pd.Series(["2025-01"] * 2000 + ["2025-02"] * 2000)
    cfg["_runtime_dev_months"] = months
    kept, detail = d02_local_psi(parts, ["f_noise"], cfg)
    assert kept == ["f_noise"]
    assert isinstance(detail, pd.DataFrame)


def test_d01_disabled_returns_all():
    from risk_model_workbench.feature_refine import d01_local_prescreen

    parts = _make_parts()
    cfg = {"local_d01": {"enabled": False, "iv": 0.02, "corr": 0.8}}
    kept, detail = d01_local_prescreen(parts, ["f_info", "f_noise", "f_corr"], cfg)
    assert set(kept) == {"f_info", "f_noise", "f_corr"}


def test_d01_reports_iv_and_correlation_substeps(monkeypatch, tmp_path):
    from risk_model_workbench import feature_refine
    from risk_model_workbench.progress import ProgressReporter, load_progress_events

    def fake_iv_filter(dev, features, target, threshold, n_bins):
        assert target == "_target_"
        return ["f_noise"], {"f_info": 0.8, "f_noise": 0.0, "f_corr": 0.7}

    def fake_corr_filter(dev, features, iv_dict, threshold):
        assert features == ["f_info", "f_corr"]
        return ["f_corr"]

    monkeypatch.setattr(
        feature_refine,
        "_load_vendor_feature_select",
        lambda: (fake_iv_filter, fake_corr_filter, object()),
    )
    reporter = ProgressReporter(tmp_path / "version", "feature_refine", emit_terminal=False)

    kept, _ = feature_refine.d01_local_prescreen(
        _make_parts(n=20),
        ["f_info", "f_noise", "f_corr"],
        {"local_d01": {"enabled": True, "iv": 0.02, "corr": 0.8, "n_bins": 10}},
        progress=reporter,
    )

    assert kept == ["f_info"]
    events = load_progress_events(tmp_path / "version")
    steps = [event["step"] for event in events]
    assert steps == [
        "d01_prepare_start",
        "d01_prepare_done",
        "d01_iv_start",
        "d01_iv_done",
        "d01_corr_start",
        "d01_corr_done",
    ]
    assert events[3]["metrics"]["kept"] == 2
    assert events[-1]["metrics"]["kept"] == 1


def test_d02_reports_psi_substep(tmp_path):
    from risk_model_workbench import feature_refine
    from risk_model_workbench.progress import ProgressReporter, load_progress_events

    reporter = ProgressReporter(tmp_path / "version", "feature_refine", emit_terminal=False)

    kept, _ = feature_refine.d02_local_psi(
        _make_parts(n=20),
        ["f_info", "f_noise"],
        {"local_d02": {"enabled": True, "warning_threshold": 0.1, "fail_threshold": 0.25, "min_base_samples": 2}},
        progress=reporter,
    )

    assert kept == ["f_info", "f_noise"]
    events = load_progress_events(tmp_path / "version")
    assert [event["step"] for event in events] == ["d02_psi_start", "d02_psi_done"]
    assert events[-1]["metrics"] == {"input_features": 2, "kept": 2, "dropped": 0}


def test_global_correlation_reports_score_matrix_and_scan_progress(tmp_path):
    from risk_model_workbench.feature_refine import global_corr_select
    from risk_model_workbench.progress import ProgressReporter, load_progress_events

    parts = _make_parts(n=20)
    reporter = ProgressReporter(tmp_path / "version", "feature_refine", emit_terminal=False)

    kept, _ = global_corr_select(
        parts.train_x,
        parts.train_y,
        {"global_corr": {"enabled": True, "threshold": 0.8}},
        progress=reporter,
    )

    assert kept
    events = load_progress_events(tmp_path / "version")
    steps = [event["step"] for event in events]
    assert "global_corr_score_progress" in steps
    assert "global_corr_matrix_start" in steps
    assert "global_corr_matrix_done" in steps
    assert "global_corr_scan_progress" in steps
    assert events[-1]["metrics"]["processed_features"] == 3
