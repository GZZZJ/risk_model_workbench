"""Local-feather d01 (missing/corr/IV) and d02 (DEV-vs-OOT PSI) compute tests.

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
    highly correlated to the informative one. OOT valid_x shifts f_noise so its PSI is high."""
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


def test_d02_psi_drops_shifted_feature_keeps_stable():
    from risk_model_workbench.feature_refine import d02_local_psi

    parts = _make_parts()
    cfg = {"local_d02": {"enabled": True, "psi": 0.2}}
    kept, detail = d02_local_psi(parts, ["f_info", "f_noise"], cfg)

    assert "f_info" in kept      # stable DEV→OOT → low PSI
    assert "f_noise" not in kept  # OOT shifted by +5 → high PSI → dropped
    assert isinstance(detail, pd.DataFrame)
    assert "max_psi" in detail.columns
    psi_by_feature = dict(zip(detail["feature"], detail["max_psi"]))
    assert psi_by_feature["f_noise"] > psi_by_feature["f_info"]


def test_d02_returns_empty_kept_when_all_unstable():
    from risk_model_workbench.feature_refine import d02_local_psi

    parts = _make_parts()
    cfg = {"local_d02": {"enabled": True, "psi": 0.0001}}  # impossible threshold → all dropped
    kept, detail = d02_local_psi(parts, ["f_noise"], cfg)
    assert kept == []
    assert isinstance(detail, pd.DataFrame)


def test_d01_disabled_returns_all():
    from risk_model_workbench.feature_refine import d01_local_prescreen

    parts = _make_parts()
    cfg = {"local_d01": {"enabled": False, "iv": 0.02, "corr": 0.8}}
    kept, detail = d01_local_prescreen(parts, ["f_info", "f_noise", "f_corr"], cfg)
    assert set(kept) == {"f_info", "f_noise", "f_corr"}
