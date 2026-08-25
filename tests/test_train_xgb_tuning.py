import sys

import pandas as pd
import pytest

from risk_model_workbench.modeling.train_xgb import _make_model


def _config():
    return {
        "training": {"random_seed": 7},
        "xgboost": {
            "learning_rate": 0.05,
            "n_estimators": 120,
            "max_depth": 3,
            "early_stopping_rounds": 30,
        },
    }


def test_xgboost_tuning_passes_supported_params_and_constructor_early_stopping():
    model, backend = _make_model(
        "xgboost", _config(), pd.Series([0, 1, 0, 1]), tuning=True,
        trial_params={"max_depth": 4, "gamma": .2, "early_stopping_rounds": 40},
    )
    assert backend == "xgboost"
    assert model.get_params()["max_depth"] == 4
    assert model.get_params()["gamma"] == pytest.approx(.2)
    assert model.get_params()["early_stopping_rounds"] == 40


def test_xgboost_llm_tuning_never_falls_back_to_sklearn(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)
    with pytest.raises(RuntimeError, match="cannot use sklearn fallback"):
        _make_model("xgboost", _config(), pd.Series([0, 1]), tuning=True)
