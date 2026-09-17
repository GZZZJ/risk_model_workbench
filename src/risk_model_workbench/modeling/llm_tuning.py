"""Algorithm-aware, auditable LLM-guided tuning helpers."""
from __future__ import annotations

import json
import shlex
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

DIAGNOSIS_STATES = {"overfit", "underfit", "healthy", "plateau", "unstable", "insufficient_evidence"}
STOP_REASONS = {"target_reached", "plateau", "trial_budget_exhausted", "advisor_recommends_stop", "all_trials_failed_guardrail", "insufficient_evidence", "manual_stop"}

@dataclass(frozen=True)
class AlgorithmTuningSpec:
    algorithm: str
    numeric_bounds: dict[str, tuple[float, float]]
    integer_bounds: dict[str, tuple[int, int]]
    control_params: frozenset[str]
    private_params: frozenset[str] = frozenset()

LGB_NUMERIC_BOUNDS = {"learning_rate": (0.005, 0.2), "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0), "reg_alpha": (0.0, 20.0), "reg_lambda": (0.0, 50.0), "min_gain_to_split": (0.0, 5.0)}
LGB_INTEGER_BOUNDS = {"num_leaves": (7, 255), "max_depth": (3, 12), "min_child_samples": (20, 1000), "bagging_freq": (0, 10), "max_bin": (63, 511), "num_boost_round": (100, 3000), "early_stopping_rounds": (20, 300)}
XGB_NUMERIC_BOUNDS = {"learning_rate": (0.005, 0.3), "min_child_weight": (0.0, 100.0), "subsample": (0.5, 1.0), "colsample_bytree": (0.5, 1.0), "gamma": (0.0, 20.0), "reg_alpha": (0.0, 20.0), "reg_lambda": (0.0, 50.0)}
XGB_INTEGER_BOUNDS = {"max_depth": (2, 12), "n_estimators": (100, 3000), "early_stopping_rounds": (20, 300)}
ALGORITHM_TUNING_SPECS = {
    "lightgbm": AlgorithmTuningSpec("lightgbm", LGB_NUMERIC_BOUNDS, LGB_INTEGER_BOUNDS, frozenset({"num_boost_round", "early_stopping_rounds"}), frozenset({"seed", "feature_fraction_seed", "bagging_seed", "verbose"})),
    "xgboost": AlgorithmTuningSpec("xgboost", XGB_NUMERIC_BOUNDS, XGB_INTEGER_BOUNDS, frozenset({"n_estimators", "early_stopping_rounds"}), frozenset({"random_state", "n_jobs", "eval_metric", "scale_pos_weight"})),
}
TRAIN_CONTROL_PARAMS = ALGORITHM_TUNING_SPECS["lightgbm"].control_params
DEFAULT_TUNING = {"max_rounds": 2, "candidates_per_round": 4, "max_trials": 8, "max_model_calls": 2, "objective": {"primary_metric": "valid_ks", "secondary_metric": "valid_auc"}, "guardrails": {"max_train_valid_auc_gap": 0.03}, "stop_rules": {"min_primary_improvement": 0.001, "stop_if_no_improvement_rounds": 1}, "advisor": {"mode": "embedded_agent", "fallback_to_heuristic": False}}

class AdvisorTuningPlanRequired(RuntimeError):
    def __init__(self, *, plan_path: str | Path | None, context_path: str | Path | None = None) -> None:
        self.plan_path, self.context_path = str(plan_path or ""), str(context_path or "")
        detail = f"embedded advisor tuning plan required: write {self.plan_path}"
        super().__init__(detail + (f" from {self.context_path}" if self.context_path else ""))

class TuningChampionUnavailable(RuntimeError):
    failure_code = "all_trials_failed_guardrail"
    def __init__(self, selection: dict[str, Any]) -> None:
        self.selection = selection
        super().__init__("all tuning trials, including baseline, failed hard guardrails; no champion was produced")

HostAgentTuningPlanRequired = AdvisorTuningPlanRequired

def normalize_algorithm(value: Any) -> str:
    value = str(value or "lightgbm").strip().lower().replace("-", "_")
    return {"lgb": "lightgbm", "lgbm": "lightgbm", "xgb": "xgboost"}.get(value, value)

def tuning_spec(algorithm: str) -> AlgorithmTuningSpec:
    algorithm = normalize_algorithm(algorithm)
    if algorithm not in ALGORITHM_TUNING_SPECS:
        raise ValueError(f"LLM-guided tuning is unsupported for algorithm: {algorithm}")
    return ALGORITHM_TUNING_SPECS[algorithm]

def llm_guided_tuning_enabled(config: dict[str, Any]) -> bool:
    training = config.get("training") or {}
    tuning = training.get("tuning") if isinstance(training.get("tuning"), dict) else {}
    return str(training.get("mode") or tuning.get("mode") or "").strip().lower() in {"llm_guided_tune", "llm-guided-tune", "llm_guided"}

def resolve_tuning_config(config: dict[str, Any], *, algorithm: str = "lightgbm") -> dict[str, Any]:
    spec = tuning_spec(algorithm)
    training = config.get("training") or {}
    raw = training.get("tuning") if isinstance(training.get("tuning"), dict) else {}
    resolved = _deep_merge(DEFAULT_TUNING, raw)
    resolved.update(mode="llm_guided_tune", algorithm=spec.algorithm)
    resolved["max_rounds"] = max(0, int(resolved["max_rounds"]))
    resolved["candidates_per_round"] = min(5, max(1, int(resolved["candidates_per_round"])))
    resolved["max_trials"] = max(1, int(resolved["max_trials"]))
    resolved["max_model_calls"] = max(0, int(resolved.get("max_model_calls", resolved["max_rounds"])))
    resolved["param_bounds"] = _merge_param_bounds(raw.get("param_bounds") if isinstance(raw.get("param_bounds"), dict) else {}, spec.algorithm)
    resolved["advisor"] = resolved["advisor"] if isinstance(resolved.get("advisor"), dict) else {}
    resolved["stop_rules"] = resolved["stop_rules"] if isinstance(resolved.get("stop_rules"), dict) else {}
    return resolved

def public_params(params: dict[str, Any], algorithm: str) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key not in tuning_spec(algorithm).private_params}

def public_lgb_params(params: dict[str, Any]) -> dict[str, Any]:
    return public_params(params, "lightgbm")

def sanitize_candidate_params(params: dict[str, Any], tuning_cfg: dict[str, Any], *, algorithm: str | None = None) -> dict[str, Any]:
    spec = tuning_spec(algorithm or tuning_cfg.get("algorithm") or "lightgbm")
    bounds = tuning_cfg.get("param_bounds") or _merge_param_bounds({}, spec.algorithm)
    result: dict[str, Any] = {}
    for key, value in (params or {}).items():
        if key in spec.integer_bounds:
            low, high = bounds["integer"].get(key, spec.integer_bounds[key])
            result[key] = int(max(low, min(high, round(float(value)))))
        elif key in spec.numeric_bounds:
            low, high = bounds["numeric"].get(key, spec.numeric_bounds[key])
            result[key] = float(max(low, min(high, float(value))))
    return result

def build_tuning_context(*, round_index: int, experiment: str, algorithm: str, base_params: dict[str, Any], training_summary: dict[str, Any], trial_history: list[dict[str, Any]], tuning_cfg: dict[str, Any], feature_review_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    algorithm = tuning_spec(algorithm).algorithm
    objective = tuning_cfg.get("objective") or {}
    primary, secondary = str(objective.get("primary_metric") or "valid_ks"), str(objective.get("secondary_metric") or "valid_auc")
    history = [_public_trial(item) for item in trial_history]
    dataset = {"train_sample_count": int(training_summary.get("train_samples", 0) or 0), "valid_sample_count": int(training_summary.get("valid_samples", 0) or 0), "train_bad_rate": _number(training_summary.get("train_bad_rate")), "valid_bad_rate": _number(training_summary.get("valid_bad_rate")), "feature_count": int(training_summary.get("kept_features", training_summary.get("feature_count", 0)) or 0)}
    diagnostics = _current_diagnostics(history, primary, secondary)
    trajectory = _trajectory(history, base_params, primary, secondary)
    deterministic = _deterministic_diagnosis(diagnostics, trajectory, tuning_cfg)
    remaining_trials = max(0, int(tuning_cfg["max_trials"]) - max(0, len(history) - 1))
    return {"schema_version": 2, "round": round_index, "experiment": experiment, "task_type": "binary_classification", "algorithm": algorithm, "base_params": public_params(base_params, algorithm), "dataset_summary": dataset, "training_summary": {**training_summary, **dataset}, "current_model_diagnostics": diagnostics, "experiment_trajectory": trajectory, "deterministic_diagnosis": deterministic, "trial_history": history, "feature_review_summary": _safe_feature_summary(feature_review_summary), "budget": {"remaining_trials": remaining_trials, "remaining_rounds": max(0, int(tuning_cfg["max_rounds"]) - round_index + 1), "remaining_model_calls": max(0, int(tuning_cfg.get("max_model_calls", tuning_cfg["max_rounds"])) - round_index + 1)}, "constraints": {"max_candidates": int(tuning_cfg["candidates_per_round"]), "remaining_trials": remaining_trials, "param_bounds": tuning_cfg["param_bounds"], "objective": objective, "guardrails": tuning_cfg.get("guardrails"), "diagnosis_evidence_catalog": deterministic["allowed_evidence"]}}

def select_best_trial(trials: list[dict[str, Any]], tuning_cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not trials:
        raise ValueError("cannot select best trial from empty trial history")
    objective, guardrails = tuning_cfg.get("objective") or {}, tuning_cfg.get("guardrails") or {}
    primary, secondary = str(objective.get("primary_metric") or "valid_ks"), str(objective.get("secondary_metric") or "valid_auc")
    max_gap = float(guardrails.get("max_train_valid_auc_gap", 0.03))
    ranked = []
    for trial in trials:
        gap, p, s = float(trial.get("auc_gap", 0.0) or 0.0), float(trial.get(primary, 0.0) or 0.0), float(trial.get(secondary, 0.0) or 0.0)
        ranked.append({"trial": trial, "selection_score": p + s * .05, "primary_value": p, "secondary_value": s, "auc_gap": gap, "guardrail_passed": gap <= max_gap})
    ranked.sort(key=lambda row: (row["selection_score"], row["primary_value"], row["secondary_value"]), reverse=True)
    numerical, eligible = ranked[0], [row for row in ranked if row["guardrail_passed"]]
    champion = eligible[0] if eligible else None
    selection = {"best_trial_id": champion["trial"].get("trial_id") if champion else None, "numerical_best_trial_id": numerical["trial"].get("trial_id"), "selection_score": champion["selection_score"] if champion else None, "primary_metric": primary, "primary_value": champion["primary_value"] if champion else None, "secondary_metric": secondary, "secondary_value": champion["secondary_value"] if champion else None, "auc_gap": champion["auc_gap"] if champion else None, "guardrail_passed": bool(champion), "guardrail_status": "passed" if champion else "all_trials_failed_guardrail", "failure_reason": "" if champion else "all_trials_failed_guardrail", "failed_guardrails": [] if champion else ["max_train_valid_auc_gap"], "candidate_count": len(trials)}
    return (champion["trial"] if champion else None), selection

def write_selection_reason(path: str | Path, best_trial: dict[str, Any] | None, selection: dict[str, Any]) -> None:
    lines = ["# Tuning Selection Reason", "", f"- Champion: {selection.get('best_trial_id')}", f"- Numerical best candidate: {selection.get('numerical_best_trial_id')}", f"- Guardrail status: {selection.get('guardrail_status')}"]
    if best_trial is None:
        lines += [f"- No champion reason: {selection.get('failure_reason')}", f"- Failed guardrails: {', '.join(selection.get('failed_guardrails') or [])}", ""]
    else:
        lines += [f"- Primary metric: {selection['primary_metric']}={selection['primary_value']:.6f}", f"- Secondary metric: {selection['secondary_metric']}={selection['secondary_value']:.6f}", f"- AUC gap: {selection['auc_gap']:.6f}", f"- Candidate: {best_trial.get('candidate_name')}", f"- Reason: {best_trial.get('reason', '')}", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")

def suggest_candidates(context: dict[str, Any], tuning_cfg: dict[str, Any], *, plan_path: str | Path | None = None, context_path: str | Path | None = None) -> dict[str, Any]:
    advisor = tuning_cfg.get("advisor") or {}
    command = str(advisor.get("command") or "").strip()
    if command:
        try:
            return _suggest_with_command(command, context, tuning_cfg)
        except Exception as exc:
            if advisor.get("fallback_to_heuristic", True) is False:
                raise
            result = _heuristic_plan(context, tuning_cfg)
            result["fallback_reason"] = str(exc)
            return result
    mode = str(advisor.get("mode") or advisor.get("type") or "embedded_agent").strip().lower()
    if mode in {"embedded_agent", "advisor", "host_agent", "agent_in_loop", "codex", "claudecode", "claude_code", "claude-code"}:
        if plan_path and Path(plan_path).exists():
            return validate_tuning_plan(json.loads(Path(plan_path).read_text(encoding="utf-8")), tuning_cfg, advisor_type="embedded_agent_plan_file", expected_experiment=str(context.get("experiment") or "") or None, expected_round=int(context.get("round", 0) or 0) or None, expected_algorithm=str(context.get("algorithm") or "") or None, allowed_evidence=_allowed_evidence(context))
        if advisor.get("fallback_to_heuristic", True) is False:
            raise AdvisorTuningPlanRequired(plan_path=plan_path, context_path=context_path)
        result = _heuristic_plan(context, tuning_cfg)
        result.update(advisor_type="embedded_agent_unavailable_local_heuristic_fallback", fallback_reason="embedded Advisor plan file not found; used local heuristic fallback")
        return result
    return _heuristic_plan(context, tuning_cfg)

def suggest_lgb_candidates(context: dict[str, Any], tuning_cfg: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return suggest_candidates(context, tuning_cfg, **kwargs)

def validate_tuning_plan(plan: dict[str, Any], tuning_cfg: dict[str, Any], *, advisor_type: str, expected_experiment: str | None = None, expected_round: int | None = None, expected_algorithm: str | None = None, allowed_evidence: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("advisor output must be a JSON object")
    if not plan.get("algorithm"):
        raise ValueError("advisor output requires algorithm")
    algorithm = normalize_algorithm(plan["algorithm"])
    expected = normalize_algorithm(expected_algorithm or tuning_cfg.get("algorithm") or "")
    if algorithm != expected:
        raise ValueError(f"algorithm mismatch: expected {expected}, got {algorithm}")
    if expected_experiment and plan.get("experiment") and str(plan["experiment"]) != expected_experiment:
        raise ValueError(f"experiment mismatch: expected {expected_experiment}, got {plan['experiment']}")
    round_index = int(plan.get("round", 0) or 0)
    if expected_round is not None and round_index != expected_round:
        raise ValueError(f"round mismatch: expected {expected_round}, got {round_index}")
    decision = str(plan.get("decision") or ("stop" if plan.get("stop") else "continue"))
    if decision not in {"continue", "stop"}:
        raise ValueError("tuning decision must be continue or stop")
    diagnosis = _validate_diagnosis(plan.get("diagnosis"), allowed_evidence)
    stop_reason = str(plan.get("stop_reason") or "")
    if decision == "stop" and stop_reason not in {"advisor_recommends_stop", "insufficient_evidence", "manual_stop"}:
        raise ValueError("advisor stop requires an allowed stop_reason")
    if decision == "continue" and stop_reason:
        raise ValueError("continue plan must not include stop_reason")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("advisor output requires candidates list")
    if decision == "continue" and not 1 <= len(candidates) <= int(tuning_cfg["candidates_per_round"]):
        raise ValueError(f"candidate count must be between 1 and {tuning_cfg['candidates_per_round']}")
    if decision == "stop" and candidates:
        raise ValueError("stop plan must not contain candidates")
    result = []
    for index, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate {index} must be an object")
        params = candidate.get("params") if isinstance(candidate.get("params"), dict) else {}
        _validate_candidate_params(params, tuning_cfg, algorithm=algorithm, index=index)
        sanitized = sanitize_candidate_params(params, tuning_cfg, algorithm=algorithm)
        if not sanitized:
            raise ValueError(f"candidate {index} has no executable params")
        result.append({"name": str(candidate.get("name") or f"candidate_{index}"), "params": sanitized, "reason": str(candidate.get("reason") or "")})
    return {"algorithm": algorithm, "round": round_index, "experiment": str(plan.get("experiment") or expected_experiment or ""), "advisor_type": advisor_type, "diagnosis": diagnosis, "decision": decision, "stop_reason": stop_reason, "candidates": result, "stop": decision == "stop"}

def deterministic_stop_reason(*, trial_history: list[dict[str, Any]], tuning_cfg: dict[str, Any], no_improvement_rounds: int) -> str:
    if len(trial_history) - 1 >= int(tuning_cfg["max_trials"]):
        return "trial_budget_exhausted"
    rules, primary = tuning_cfg.get("stop_rules") or {}, str((tuning_cfg.get("objective") or {}).get("primary_metric") or "valid_ks")
    target = rules.get("target_primary_metric")
    if target is not None and max(float(row.get(primary, 0) or 0) for row in trial_history) >= float(target):
        return "target_reached"
    return "plateau" if no_improvement_rounds >= int(rules.get("stop_if_no_improvement_rounds", 1)) else ""

def run_tuning_loop(*, output_dir: Path, experiment: str, algorithm: str, base_params: dict[str, Any], training_summary: dict[str, Any], tuning_cfg: dict[str, Any], run_trial: Callable[[int, str, dict[str, Any], str, str, dict[str, Any]], dict[str, Any]], feature_review_summary: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any], str]:
    algorithm, history, decisions = tuning_spec(algorithm).algorithm, [], []
    baseline = run_trial(0, "baseline", {}, "configured baseline parameters", "baseline", {})
    baseline["trial_id"] = 0
    history.append(baseline)
    no_improvement, stop_reason = 0, ""
    for round_index in range(1, int(tuning_cfg["max_rounds"]) + 1):
        pre_stop = deterministic_stop_reason(trial_history=history, tuning_cfg=tuning_cfg, no_improvement_rounds=no_improvement)
        if pre_stop:
            stop_reason = pre_stop; decisions.append({"round": round_index, "decision": "stop", "reason": pre_stop, "source": "deterministic"}); break
        context = build_tuning_context(round_index=round_index, experiment=experiment, algorithm=algorithm, base_params=base_params, training_summary=training_summary, trial_history=history, tuning_cfg=tuning_cfg, feature_review_summary=feature_review_summary)
        context_path, plan_path = output_dir / f"tuning_context_round_{round_index}.json", output_dir / f"llm_tuning_plan_round_{round_index}.json"
        _write_json(output_dir / "tuning_context.json", context); _write_json(context_path, context)
        plan = suggest_candidates(context, tuning_cfg, plan_path=plan_path, context_path=context_path)
        _write_json(plan_path, plan); _write_json(output_dir / f"tuning_diagnosis_round_{round_index}.json", {"round": round_index, "algorithm": algorithm, "diagnosis": plan["diagnosis"], "deterministic_evidence": context["deterministic_diagnosis"]})
        decisions.append({"round": round_index, "decision": plan["decision"], "reason": plan["stop_reason"], "source": plan["advisor_type"], "diagnosis": plan["diagnosis"]})
        if plan["decision"] == "stop":
            stop_reason = plan["stop_reason"]; break
        primary = str((tuning_cfg.get("objective") or {}).get("primary_metric") or "valid_ks")
        before = max(float(row.get(primary, 0) or 0) for row in history)
        for candidate in plan["candidates"]:
            if len(history) - 1 >= int(tuning_cfg["max_trials"]): break
            record = run_trial(round_index, candidate["name"], candidate["params"], candidate["reason"], plan["advisor_type"], plan["diagnosis"])
            record["trial_id"] = len(history)
            history.append(record)
        after = max(float(row.get(primary, 0) or 0) for row in history)
        threshold = float((tuning_cfg.get("stop_rules") or {}).get("min_primary_improvement", (tuning_cfg.get("stop_rules") or {}).get("min_ks_improvement", .001)))
        no_improvement = no_improvement + 1 if after - before < threshold else 0
        decisions.append({"round": round_index, "decision": "feedback", "improvement": after - before, "threshold": threshold, "no_improvement_rounds": no_improvement})
    if not stop_reason and int(tuning_cfg["max_rounds"]) == 0: stop_reason = "insufficient_evidence"
    champion, selection = select_best_trial(history, tuning_cfg)
    if champion is None: stop_reason = "all_trials_failed_guardrail"
    trajectory = _trajectory([_public_trial(row) for row in history], base_params, str((tuning_cfg.get("objective") or {}).get("primary_metric") or "valid_ks"), str((tuning_cfg.get("objective") or {}).get("secondary_metric") or "valid_auc"))
    _write_json(output_dir / "tuning_trajectory.json", trajectory); _write_trial_csv(output_dir / "tuning_trials.csv", history)
    summary = {"mode": "llm_guided_tune", "algorithm": algorithm, "trial_count": len(history), "selection": selection, "stop_reason": stop_reason, "decisions": decisions, "best_trial_id": selection["best_trial_id"], "best_params": champion.get("params") if champion else None}
    _write_json(output_dir / "tuning_summary.json", summary); write_selection_reason(output_dir / "selection_reason.md", champion, selection)
    (output_dir / "llm_tuning_decisions.md").write_text("# LLM Tuning Decisions\n\n" + "\n".join(f"- round {row.get('round')}: {row.get('decision')} {row.get('reason', '')}" for row in decisions) + "\n", encoding="utf-8")
    return history, champion, summary, stop_reason

def _suggest_with_command(command: str, context: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    proc = subprocess.run(shlex.split(command), input=json.dumps(context, ensure_ascii=False), capture_output=True, text=True, timeout=int((tuning_cfg.get("advisor") or {}).get("timeout_seconds", 120)), check=False)
    if proc.returncode: raise RuntimeError(f"advisor command failed with code {proc.returncode}: {proc.stderr.strip()}")
    return validate_tuning_plan(json.loads(proc.stdout), tuning_cfg, advisor_type="external_llm_command", expected_experiment=str(context.get("experiment") or "") or None, expected_round=int(context.get("round", 0) or 0) or None, expected_algorithm=str(context.get("algorithm") or "") or None, allowed_evidence=_allowed_evidence(context))

def _heuristic_plan(context: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    algorithm = tuning_spec(context.get("algorithm") or tuning_cfg.get("algorithm") or "lightgbm").algorithm
    diagnosis = _heuristic_diagnosis(context)
    raw_base = (_best_public_trial(context.get("trial_history") or []) or {}).get("params") or context.get("base_params") or {}
    spec = tuning_spec(algorithm)
    base = {key: value for key, value in raw_base.items() if key in spec.numeric_bounds or key in spec.integer_bounds}
    if algorithm == "lightgbm":
        rows = [_merge(base, learning_rate=.03, num_leaves=31, max_depth=5, min_child_samples=180, subsample=.8, colsample_bytree=.75, reg_alpha=.3, reg_lambda=4, num_boost_round=1200, early_stopping_rounds=80), _merge(base, learning_rate=.025, num_leaves=15, max_depth=4, min_child_samples=250, subsample=.78, colsample_bytree=.7, reg_alpha=.8, reg_lambda=8, num_boost_round=1500, early_stopping_rounds=100), _merge(base, learning_rate=.04, num_leaves=63, max_depth=7, min_child_samples=120, subsample=.85, colsample_bytree=.8, reg_alpha=.1, reg_lambda=2, num_boost_round=900, early_stopping_rounds=70)]
    else:
        rows = [_merge(base, learning_rate=.03, max_depth=4, min_child_weight=5, subsample=.8, colsample_bytree=.75, gamma=.2, reg_alpha=.3, reg_lambda=4, n_estimators=1200, early_stopping_rounds=80), _merge(base, learning_rate=.025, max_depth=3, min_child_weight=10, subsample=.78, colsample_bytree=.7, gamma=.5, reg_alpha=.8, reg_lambda=8, n_estimators=1500, early_stopping_rounds=100), _merge(base, learning_rate=.05, max_depth=5, min_child_weight=2, subsample=.85, colsample_bytree=.8, gamma=0, reg_alpha=.1, reg_lambda=2, n_estimators=900, early_stopping_rounds=70)]
    candidates = [{"name": f"heuristic_{algorithm}_r{context.get('round', 1)}_{index}", "params": params, "reason": f"deterministic {diagnosis['state']} evidence; bounded {algorithm} exploration"} for index, params in enumerate(rows[:int(tuning_cfg["candidates_per_round"])], 1)]
    return validate_tuning_plan({"algorithm": algorithm, "round": int(context.get("round", 1)), "experiment": str(context.get("experiment") or ""), "diagnosis": diagnosis, "decision": "continue", "candidates": candidates}, tuning_cfg, advisor_type="local_heuristic_fallback", expected_experiment=str(context.get("experiment") or "") or None, expected_round=int(context.get("round", 0) or 0) or None, expected_algorithm=algorithm, allowed_evidence=_allowed_evidence(context))

def _validate_candidate_params(params: dict[str, Any], tuning_cfg: dict[str, Any], *, algorithm: str, index: int) -> None:
    spec, bounds = tuning_spec(algorithm), tuning_cfg.get("param_bounds") or _merge_param_bounds({}, algorithm)
    for key, value in params.items():
        if key in spec.integer_bounds: low, high = bounds["integer"].get(key, spec.integer_bounds[key])
        elif key in spec.numeric_bounds: low, high = bounds["numeric"].get(key, spec.numeric_bounds[key])
        else: raise ValueError(f"candidate {index} param {key} is unsupported for {algorithm}")
        if isinstance(value, bool): raise ValueError(f"candidate {index} param {key} has invalid type")
        try: numeric = float(value)
        except (TypeError, ValueError) as exc: raise ValueError(f"candidate {index} param {key} has invalid type") from exc
        if numeric < low or numeric > high: raise ValueError(f"candidate {index} param {key} outside allowed bounds")

def _validate_diagnosis(value: Any, allowed_evidence: set[str] | None) -> dict[str, Any]:
    if not isinstance(value, dict): raise ValueError("diagnosis must be a structured object")
    state, evidence, directions = str(value.get("state") or ""), value.get("evidence"), value.get("recommended_direction")
    if state not in DIAGNOSIS_STATES: raise ValueError("diagnosis state is invalid")
    if not str(value.get("summary") or "").strip() or not isinstance(evidence, list) or not evidence or not isinstance(directions, list): raise ValueError("diagnosis requires summary, evidence, and recommended_direction")
    evidence = [str(item) for item in evidence]
    if any(not item.replace("_", "").isalnum() for item in evidence): raise ValueError("diagnosis evidence must be deterministic evidence codes")
    if allowed_evidence is not None and not set(evidence).issubset(allowed_evidence): raise ValueError("diagnosis evidence is not present in deterministic context")
    try: confidence = float(value.get("confidence"))
    except (TypeError, ValueError) as exc: raise ValueError("diagnosis confidence must be numeric") from exc
    if not 0 <= confidence <= 1: raise ValueError("diagnosis confidence must be between 0 and 1")
    return {"state": state, "summary": str(value["summary"]), "evidence": evidence, "recommended_direction": [str(item) for item in directions], "confidence": confidence}

def _current_diagnostics(history: list[dict[str, Any]], primary: str, secondary: str) -> dict[str, Any]:
    best = _best_public_trial(history) or {}
    tr_auc, va_auc, tr_ks, va_ks = _number(best.get("train_auc")), _number(best.get("valid_auc")), _number(best.get("train_ks")), _number(best.get("valid_ks"))
    return {"train_auc": tr_auc, "valid_auc": va_auc, "auc_gap": _gap(tr_auc, va_auc), "train_ks": tr_ks, "valid_ks": va_ks, "ks_gap": _gap(tr_ks, va_ks), "best_iteration": best.get("best_iteration"), "current_primary_metric": primary, "current_primary_value": _number(best.get(primary)), "current_secondary_metric": secondary, "current_secondary_value": _number(best.get(secondary))}

def _trajectory(history: list[dict[str, Any]], base_params: dict[str, Any], primary: str, secondary: str) -> dict[str, Any]:
    baseline, current = (history[0] if history else {}), (_best_public_trial(history) or {})
    directions = set()
    for row in history[1:]:
        for key, value in (row.get("params") or {}).items():
            if key in base_params:
                try: delta = float(value) - float(base_params[key])
                except (TypeError, ValueError): continue
                if delta: directions.add(f"{key} {'increased' if delta > 0 else 'decreased'}")
    baseline_value, current_value = _number(baseline.get(primary)), _number(current.get(primary))
    return {"baseline_metric": baseline_value, "current_best_metric": current_value, "improvement_vs_baseline": _gap(current_value, baseline_value), "improvement_vs_previous_round": _last_round_improvement(history, primary), "no_improvement_round_count": _no_improvement_rounds(history, primary), "trial_count": len(history), "parameter_directions_explored": sorted(directions), "primary_metric": primary, "secondary_metric": secondary}

def _deterministic_diagnosis(diagnostics: dict[str, Any], trajectory: dict[str, Any], tuning_cfg: dict[str, Any]) -> dict[str, Any]:
    max_gap, codes = float((tuning_cfg.get("guardrails") or {}).get("max_train_valid_auc_gap", .03)), []
    if diagnostics["valid_auc"] is None: codes.append("metrics_incomplete")
    if (diagnostics["auc_gap"] or 0) > max_gap: codes.append("train_valid_auc_gap_above_guardrail")
    if (diagnostics["ks_gap"] or 0) > max_gap: codes.append("train_valid_ks_gap_above_guardrail")
    if trajectory["trial_count"] <= 1: codes.append("baseline_only")
    if trajectory["no_improvement_round_count"] > 0: codes.append("no_primary_metric_improvement")
    if not codes: codes.append("within_configured_guardrails")
    state = "overfit" if "train_valid_auc_gap_above_guardrail" in codes else "plateau" if "no_primary_metric_improvement" in codes else "insufficient_evidence" if "metrics_incomplete" in codes else "healthy"
    return {"allowed_evidence": codes, "suggested_state": state, "metric_guardrails": {"max_train_valid_auc_gap": max_gap}}

def _heuristic_diagnosis(context: dict[str, Any]) -> dict[str, Any]:
    deterministic = context.get("deterministic_diagnosis") or {}
    state, evidence = str(deterministic.get("suggested_state") or "insufficient_evidence"), list(deterministic.get("allowed_evidence") or ["metrics_incomplete"])
    direction = {"overfit": ["reduce_capacity", "increase_regularization"], "plateau": ["explore_regularized_capacity_tradeoff"], "healthy": ["bounded_local_exploration"], "insufficient_evidence": ["collect_more_valid_evidence"]}.get(state, ["bounded_local_exploration"])
    return {"state": state, "summary": f"Deterministic evidence indicates {state}.", "evidence": evidence, "recommended_direction": direction, "confidence": .8}

def load_feature_review_summary(workspace: Path) -> dict[str, Any] | None:
    path = workspace / "feature_selection" / "feature_risk_review.json"
    if not path.exists(): return None
    try: payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
    return _safe_feature_summary({"selected_feature_count": summary.get("selected_feature_count", summary.get("selected_features")), "review_required_count": summary.get("review_required_count"), "high_drift_count": summary.get("high_drift_count"), "unknown_metadata_count": summary.get("unknown_metadata_count"), "importance_stability_summary": summary.get("importance_stability_summary")})

def _safe_feature_summary(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(value, dict): return None
    return {key: value[key] for key in {"selected_feature_count", "review_required_count", "high_drift_count", "unknown_metadata_count", "importance_stability_summary"} if key in value}
def _allowed_evidence(context: dict[str, Any]) -> set[str] | None:
    evidence = ((context.get("deterministic_diagnosis") or {}).get("allowed_evidence"))
    return {str(item) for item in evidence} if isinstance(evidence, list) else None
def _merge_param_bounds(override: dict[str, Any], algorithm: str) -> dict[str, dict[str, tuple[float, float] | tuple[int, int]]]:
    spec, numeric, integer = tuning_spec(algorithm), deepcopy(tuning_spec(algorithm).numeric_bounds), deepcopy(tuning_spec(algorithm).integer_bounds)
    for key, value in override.items():
        if isinstance(value, (list, tuple)) and len(value) == 2:
            if key in numeric: numeric[key] = (float(value[0]), float(value[1]))
            elif key in integer: integer[key] = (int(value[0]), int(value[1]))
    return {"numeric": numeric, "integer": integer}
def _write_json(path: Path, payload: dict[str, Any]) -> None: path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
def _write_trial_csv(path: Path, history: list[dict[str, Any]]) -> None:
    import pandas as pd
    rows = []
    for trial in history:
        row = {key: value for key, value in trial.items() if not key.startswith("_") and key != "params"}
        row.update({f"param_{key}": value for key, value in (trial.get("params") or {}).items()}); rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items(): result[key] = _deep_merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result
def _merge(base: dict[str, Any], **values: Any) -> dict[str, Any]: result = dict(base); result.update(values); return result
def _public_trial(trial: dict[str, Any]) -> dict[str, Any]: return {key: value for key, value in trial.items() if not key.startswith("_")}
def _best_public_trial(history: list[dict[str, Any]]) -> dict[str, Any] | None: return sorted([_public_trial(row) for row in history], key=lambda row: (float(row.get("valid_ks", 0) or 0), float(row.get("valid_auc", 0) or 0)), reverse=True)[0] if history else None
def _number(value: Any) -> float | None:
    try: return float(value) if value is not None else None
    except (TypeError, ValueError): return None
def _gap(left: float | None, right: float | None) -> float | None: return left - right if left is not None and right is not None else None
def _no_improvement_rounds(history: list[dict[str, Any]], primary: str) -> int:
    rounds: dict[int, float] = {}
    for row in history: rounds[int(row.get("round", 0) or 0)] = max(rounds.get(int(row.get("round", 0) or 0), float("-inf")), float(row.get(primary, 0) or 0))
    count = 0
    for before, after in zip([rounds[key] for key in sorted(rounds)], [rounds[key] for key in sorted(rounds)][1:]): count = count + 1 if after <= before else 0
    return count
def _last_round_improvement(history: list[dict[str, Any]], primary: str) -> float | None:
    if not history: return None
    rounds = sorted({int(row.get("round", 0) or 0) for row in history})
    if len(rounds) < 2: return None
    value = lambda r: max(float(row.get(primary, 0) or 0) for row in history if int(row.get("round", 0) or 0) == r)
    return value(rounds[-1]) - value(rounds[-2])
