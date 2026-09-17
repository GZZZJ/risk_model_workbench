import json

import pytest

from risk_model_workbench.modeling.llm_tuning import (
    DIAGNOSIS_STATES,
    build_tuning_context,
    deterministic_stop_reason,
    resolve_tuning_config,
    select_best_trial,
    suggest_lgb_candidates,
    validate_tuning_plan,
)
from risk_model_workbench.agent.reasoning_contracts import TuningPlan


def _config(*, algorithm="lightgbm", candidates=2, **tuning):
    return resolve_tuning_config(
        {"training": {"mode": "llm_guided_tune", "tuning": {"candidates_per_round": candidates, **tuning}}},
        algorithm=algorithm,
    )


def _diagnosis(*, state="healthy", evidence=None):
    return {
        "state": state,
        "summary": "deterministic metric evidence reviewed",
        "evidence": evidence or ["baseline_only"],
        "recommended_direction": ["bounded_local_exploration"],
        "confidence": 0.8,
    }


def _plan(*, algorithm="lightgbm", candidates=None, diagnosis=None, decision="continue", stop_reason=None):
    return {
        "algorithm": algorithm,
        "experiment": "main_model",
        "round": 1,
        "diagnosis": diagnosis or _diagnosis(),
        "decision": decision,
        "stop_reason": stop_reason,
        "candidates": candidates if candidates is not None else [
            {"name": "candidate_1", "params": {"learning_rate": 0.03}, "reason": "bounded trial"},
        ],
    }


def test_heuristic_plan_supports_two_runtime_candidates():
    cfg = _config(candidates=2, advisor={"mode": "heuristic"})
    context = build_tuning_context(
        round_index=1, experiment="main_model", algorithm="lightgbm",
        base_params={"learning_rate": 0.05, "num_leaves": 31, "max_depth": 5},
        training_summary={"train_samples": 1000, "valid_samples": 300, "train_bad_rate": .1, "valid_bad_rate": .11, "kept_features": 20},
        trial_history=[{"trial_id": 0, "round": 0, "params": {}, "train_auc": .75, "valid_auc": .72, "train_ks": .4, "valid_ks": .35, "auc_gap": .03}],
        tuning_cfg=cfg,
    )
    plan = suggest_lgb_candidates(context, cfg)
    assert plan["algorithm"] == "lightgbm"
    assert len(plan["candidates"]) == 2
    assert plan["diagnosis"]["state"] in DIAGNOSIS_STATES


@pytest.mark.parametrize(
    ("algorithm", "params", "message"),
    [
        ("lightgbm", {"learning_rate": 9}, "outside allowed bounds"),
        ("xgboost", {"learning_rate": 9}, "outside allowed bounds"),
        ("lightgbm", {"gamma": 1}, "unsupported"),
        ("xgboost", {"num_leaves": 31}, "unsupported"),
    ],
)
def test_algorithm_specific_parameter_validation(algorithm, params, message):
    cfg = _config(algorithm=algorithm)
    with pytest.raises(ValueError, match=message):
        validate_tuning_plan(
            _plan(algorithm=algorithm, candidates=[{"name": "bad", "params": params, "reason": "bad"}]),
            cfg, advisor_type="test", expected_experiment="main_model", expected_round=1,
            expected_algorithm=algorithm, allowed_evidence={"baseline_only"},
        )


def test_algorithm_mismatch_and_invalid_diagnosis_are_rejected():
    cfg = _config()
    with pytest.raises(ValueError, match="algorithm mismatch"):
        validate_tuning_plan(_plan(algorithm="xgboost"), cfg, advisor_type="test", expected_algorithm="lightgbm")
    with pytest.raises(ValueError, match="diagnosis state"):
        validate_tuning_plan(
            _plan(diagnosis=_diagnosis(state="invented_state")), cfg, advisor_type="test",
            expected_algorithm="lightgbm", allowed_evidence={"baseline_only"},
        )


def test_context_is_deterministic_and_row_free():
    cfg = _config()
    history = [{"trial_id": 0, "round": 0, "params": {"max_depth": 5}, "train_auc": .78, "valid_auc": .72, "train_ks": .45, "valid_ks": .35, "auc_gap": .06}]
    context = build_tuning_context(
        round_index=1, experiment="main_model", algorithm="lightgbm",
        base_params={"learning_rate": .05, "max_depth": 5},
        training_summary={"train_samples": 100, "valid_samples": 50, "train_bad_rate": .2, "valid_bad_rate": .25, "kept_features": 8},
        trial_history=history, tuning_cfg=cfg,
        feature_review_summary={"selected_feature_count": 8, "review_required_count": 1, "raw_dataframe": "must_not_be_present"},
    )
    assert context["dataset_summary"] == {"train_sample_count": 100, "valid_sample_count": 50, "train_bad_rate": .2, "valid_bad_rate": .25, "feature_count": 8}
    assert context["current_model_diagnostics"]["auc_gap"] == pytest.approx(.06)
    assert "max_depth increased" not in context["experiment_trajectory"]["parameter_directions_explored"]
    assert "raw_dataframe" not in json.dumps(context)
    assert "train_valid_auc_gap_above_guardrail" in context["deterministic_diagnosis"]["allowed_evidence"]


def test_oot_is_not_a_tuning_context_input_or_selection_metric():
    cfg = _config()
    common = dict(round_index=1, experiment="main_model", algorithm="lightgbm", base_params={"learning_rate": .05}, training_summary={"train_samples": 10, "valid_samples": 5, "train_bad_rate": .2, "valid_bad_rate": .2, "kept_features": 1}, trial_history=[{"trial_id": 0, "round": 0, "params": {}, "train_auc": .7, "valid_auc": .68, "train_ks": .3, "valid_ks": .28, "auc_gap": .02}], tuning_cfg=cfg)
    first, second = build_tuning_context(**common), build_tuning_context(**common)
    assert first == second
    champion_a, select_a = select_best_trial(common["trial_history"], cfg)
    champion_b, select_b = select_best_trial(common["trial_history"], cfg)
    assert champion_a == champion_b and select_a == select_b


def test_plateau_and_all_guardrail_failure_are_deterministic():
    cfg = _config(stop_rules={"stop_if_no_improvement_rounds": 2})
    history = [{"trial_id": 0, "valid_ks": .3}, {"trial_id": 1, "valid_ks": .3}]
    assert deterministic_stop_reason(trial_history=history, tuning_cfg=cfg, no_improvement_rounds=2) == "plateau"
    best, selection = select_best_trial(
        [{"trial_id": 0, "valid_ks": .4, "valid_auc": .7, "auc_gap": .08}, {"trial_id": 1, "valid_ks": .5, "valid_auc": .72, "auc_gap": .05}],
        _config(guardrails={"max_train_valid_auc_gap": .03}),
    )
    assert best is None
    assert selection["guardrail_status"] == "all_trials_failed_guardrail"
    assert selection["numerical_best_trial_id"] == 1


def test_baseline_can_win_when_candidates_fail_guardrail():
    best, selection = select_best_trial(
        [{"trial_id": 0, "valid_ks": .4, "valid_auc": .7, "auc_gap": .02}, {"trial_id": 1, "valid_ks": .5, "valid_auc": .72, "auc_gap": .08}],
        _config(),
    )
    assert best["trial_id"] == 0
    assert selection["guardrail_status"] == "passed"


def test_structured_advisor_stop_is_validated():
    cfg = _config()
    plan = validate_tuning_plan(
        _plan(decision="stop", stop_reason="advisor_recommends_stop", candidates=[]),
        cfg, advisor_type="test", expected_experiment="main_model", expected_round=1,
        expected_algorithm="lightgbm", allowed_evidence={"baseline_only"},
    )
    assert plan["stop"] is True
    assert plan["stop_reason"] == "advisor_recommends_stop"


def test_pydantic_contract_permits_two_candidates_but_runtime_still_binds_count():
    payload = _plan(candidates=[
        {"name": "one", "params": {"learning_rate": .03}, "reason": "one"},
        {"name": "two", "params": {"learning_rate": .02}, "reason": "two"},
    ])
    assert len(TuningPlan.model_validate(payload).candidates) == 2
    with pytest.raises(ValueError, match="candidate count"):
        validate_tuning_plan(
            payload, _config(candidates=1), advisor_type="test", expected_algorithm="lightgbm",
            allowed_evidence={"baseline_only"},
        )
