import json
import sys

import pytest

from risk_model_workbench.modeling.llm_tuning import (
    HostAgentTuningPlanRequired,
    build_tuning_context,
    resolve_tuning_config,
    select_best_trial,
    suggest_lgb_candidates,
    validate_tuning_plan,
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


def test_host_agent_plan_file_rejects_wrong_experiment(tmp_path):
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"advisor": {"mode": "host_agent", "fallback_to_heuristic": False}},
            }
        }
    )
    context = {"round": 1, "experiment": "main_lgbm", "trial_history": [], "base_params": {}}
    plan_path = tmp_path / "llm_tuning_plan_round_1.json"
    plan_path.write_text(
        json.dumps(
            {
                "experiment": "other_lgbm",
                "round": 1,
                "diagnosis": "wrong experiment",
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

    with pytest.raises(ValueError, match="experiment mismatch"):
        suggest_lgb_candidates(context, tuning_cfg, plan_path=plan_path)


def test_external_advisor_command_rejects_wrong_round(tmp_path):
    script = tmp_path / "advisor_command.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({"
        "'experiment': 'main_lgbm', 'round': 2, 'diagnosis': 'wrong round', "
        "'candidates': [{'name': 'candidate', 'params': {'learning_rate': 0.03}, 'reason': 'ok'}], "
        "'stop': False"
        "}))\n",
        encoding="utf-8",
    )
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {
                    "advisor": {
                        "command": f"{sys.executable} {script}",
                        "fallback_to_heuristic": False,
                    }
                },
            }
        }
    )

    with pytest.raises(ValueError, match="round mismatch"):
        suggest_lgb_candidates({"round": 1, "experiment": "main_lgbm"}, tuning_cfg)


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


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        ({"experiment": "other", "round": 1, "candidates": [], "stop": False}, "experiment mismatch"),
        ({"experiment": "main_lgbm", "round": 2, "candidates": [], "stop": False}, "round mismatch"),
        ({"experiment": "main_lgbm", "round": 1, "candidates": [], "stop": False}, "candidate count"),
        (
            {
                "experiment": "main_lgbm",
                "round": 1,
                "candidates": [
                    {"name": f"candidate_{idx}", "params": {"learning_rate": 0.03}, "reason": "ok"}
                    for idx in range(10)
                ],
                "stop": False,
            },
            "candidate count",
        ),
        (
            {
                "experiment": "main_lgbm",
                "round": 1,
                "candidates": [
                    {"name": "bad_bounds", "params": {"learning_rate": 9.0}, "reason": "bad"}
                ],
                "stop": False,
            },
            "outside allowed bounds",
        ),
    ],
)
def test_validate_tuning_plan_rejects_stale_or_unbounded_plan(plan, message):
    tuning_cfg = resolve_tuning_config(
        {
            "training": {
                "mode": "llm_guided_tune",
                "tuning": {"candidates_per_round": 4},
            }
        }
    )

    with pytest.raises(ValueError, match=message):
        validate_tuning_plan(
            plan,
            tuning_cfg,
            advisor_type="host_agent_response",
            expected_experiment="main_lgbm",
            expected_round=1,
        )
