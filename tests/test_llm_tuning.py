import json

import pytest

from risk_model_workbench.modeling.llm_tuning import (
    HostAgentTuningPlanRequired,
    build_tuning_context,
    resolve_tuning_config,
    select_best_trial,
    suggest_lgb_candidates,
)


def test_heuristic_advisor_returns_bounded_candidates():
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"max_rounds": 1, "candidates_per_round": 2, "max_trials": 2},
            }
        }
    )
    context = build_tuning_context(
        round_index=1,
        experiment="main_lgbm",
        algorithm="lightgbm",
        base_params={"learning_rate": 0.05, "num_leaves": 31, "max_depth": 5},
        training_summary={"train_samples": 1000, "valid_samples": 300, "kept_features": 20},
        trial_history=[
            {
                "trial_id": 0,
                "candidate_name": "baseline",
                "params": {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 5},
                "valid_auc": 0.72,
                "valid_ks": 0.35,
                "auc_gap": 0.01,
            }
        ],
        tuning_cfg=tuning_cfg,
    )

    plan = suggest_lgb_candidates(context, tuning_cfg)

    assert plan["advisor_type"] == "host_agent_unavailable_local_heuristic_fallback"
    assert len(plan["candidates"]) == 2
    for candidate in plan["candidates"]:
        params = candidate["params"]
        assert 0.005 <= params["learning_rate"] <= 0.2
        assert 7 <= params["num_leaves"] <= 255
        assert 3 <= params["max_depth"] <= 12


def test_select_best_trial_prefers_guardrail_passing_candidate():
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"guardrails": {"max_train_valid_auc_gap": 0.03}},
            }
        }
    )
    trials = [
        {"trial_id": 0, "valid_ks": 0.75, "valid_auc": 0.90, "auc_gap": 0.08},
        {"trial_id": 1, "valid_ks": 0.73, "valid_auc": 0.89, "auc_gap": 0.02},
    ]

    best, selection = select_best_trial(trials, tuning_cfg)

    assert best["trial_id"] == 1
    assert selection["guardrail_status"] == "passed"


def test_host_agent_plan_file_is_used(tmp_path):
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"advisor": {"mode": "host_agent", "fallback_to_heuristic": False}},
            }
        }
    )
    context = {"round": 1, "trial_history": [], "base_params": {}}
    plan_path = tmp_path / "llm_tuning_plan_round_1.json"
    plan_path.write_text(
        json.dumps(
            {
                "round": 1,
                "diagnosis": "increase capacity",
                "candidates": [
                    {
                        "name": "agent_candidate",
                        "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 7},
                        "reason": "host agent diagnosis",
                    }
                ],
                "stop": False,
            }
        ),
        encoding="utf-8",
    )

    plan = suggest_lgb_candidates(context, tuning_cfg, plan_path=plan_path)

    assert plan["advisor_type"] == "host_agent_plan_file"
    assert plan["candidates"][0]["name"] == "agent_candidate"


def test_host_agent_mode_can_require_plan_file(tmp_path):
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"advisor": {"mode": "host_agent", "fallback_to_heuristic": False}},
            }
        }
    )

    with pytest.raises(HostAgentTuningPlanRequired):
        suggest_lgb_candidates({"round": 1}, tuning_cfg, plan_path=tmp_path / "missing_plan.json")
