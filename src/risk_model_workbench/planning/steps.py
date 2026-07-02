"""Scenario profile and stage-step registry for execution planning."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.request.training import effective_training_config, llm_guided_tuning_enabled


KNOWN_STAGES = {
    "validate_config",
    "sample_check",
    "feature_metadata",
    "feature_prescreen",
    "build_wide_sql",
    "feature_refine",
    "train_baseline",
    "evaluate",
    "compare",
    "report",
}


STAGE_ALIASES = {
    "d01_d02_screening": "feature_prescreen",
}


STEP_ALIASES = {
    "d01_d02_batch_screening": "feature_quality_prescreen",
}


STEP_REGISTRY: dict[str, dict[str, Any]] = {
    "field_contract": {
        "id": "field_contract",
        "stage": "sample_check",
        "description": "Confirm required id, time, split, target, and feature fields.",
        "default_params": {},
        "source_reference": "AIAgent domain SOP: field definition checks",
        "implementation_status": "implemented",
    },
    "key_uniqueness": {
        "id": "key_uniqueness",
        "stage": "sample_check",
        "description": "Check sample primary-key uniqueness and duplicate columns.",
        "default_params": {},
        "source_reference": "AIAgent domain SOP: duplicate and uniqueness checks",
        "implementation_status": "implemented",
    },
    "monthly_label_distribution": {
        "id": "monthly_label_distribution",
        "stage": "sample_check",
        "description": "Profile monthly sample size and target distribution.",
        "default_params": {"min_samples_per_month": 50},
        "source_reference": "AIAgent domain SOP: monthly sample statistics",
        "implementation_status": "implemented",
    },
    "segment_distribution": {
        "id": "segment_distribution",
        "stage": "sample_check",
        "description": "Profile configured business segment distribution.",
        "default_params": {},
        "source_reference": "Fujie GCard sample audit artifacts",
        "implementation_status": "implemented",
    },
    "account_status_distribution": {
        "id": "account_status_distribution",
        "stage": "sample_check",
        "description": "Profile in-loan account status buckets such as M0/M1/M2.",
        "default_params": {},
        "source_reference": "AIAgent DZ in-loan modeling SOP",
        "implementation_status": "implemented",
    },
    "channel_distribution": {
        "id": "channel_distribution",
        "stage": "sample_check",
        "description": "Profile acquisition channel sample size and positive rate.",
        "default_params": {},
        "source_reference": "AIAgent Hk acquisition modeling SOP",
        "implementation_status": "implemented",
    },
    "dual_target_split": {
        "id": "dual_target_split",
        "stage": "sample_check",
        "description": "Declare acquisition quality/conversion target separation.",
        "default_params": {},
        "source_reference": "AIAgent Hk acquisition dual-engine SOP",
        "implementation_status": "implemented",
    },
    "credit_product_coverage": {
        "id": "credit_product_coverage",
        "stage": "sample_check",
        "description": "Profile third-party credit product coverage and label availability.",
        "default_params": {},
        "source_reference": "AIAgent credit product evaluation SOP",
        "implementation_status": "implemented",
    },
    "feature_metadata_export": {
        "id": "feature_metadata_export",
        "stage": "feature_metadata",
        "description": "Export feature table and column metadata.",
        "default_params": {},
        "source_reference": "RMW feature metadata flow",
        "implementation_status": "implemented",
    },
    "feature_quality_prescreen": {
        "id": "feature_quality_prescreen",
        "stage": "feature_prescreen",
        "description": "Run coarse feature quality and stability prescreening or SQL dry run.",
        "default_params": {"require_sql_approval": True},
        "source_reference": "RMW feature prescreening flow",
        "implementation_status": "implemented",
    },
    "wide_sql_generation": {
        "id": "wide_sql_generation",
        "stage": "build_wide_sql",
        "description": "Generate wide-table SQL and feature mapping from remaining features.",
        "default_params": {},
        "source_reference": "RMW wide SQL generation flow",
        "implementation_status": "implemented",
    },
    "sql_review_gate": {
        "id": "sql_review_gate",
        "stage": "build_wide_sql",
        "description": "Review SQL for leakage, join explosion, null handling, and unsafe logic.",
        "default_params": {"block_on_high_risk": True},
        "source_reference": "AIAgent code review SOP",
        "implementation_status": "implemented",
    },
    "feature_availability_filter": {
        "id": "feature_availability_filter",
        "stage": "feature_refine",
        "description": "Keep only executable model features and remove id, label, split, score, random, and other non-feature columns.",
        "default_params": {},
        "source_reference": "Fujie GCard feature availability and base-column exclusion flow",
        "implementation_status": "implemented",
    },
    "missing_rate_filter": {
        "id": "missing_rate_filter",
        "stage": "feature_refine",
        "description": "Drop features above the configured missing-rate threshold.",
        "default_params": {"threshold": 0.9},
        "source_reference": "AIAgent three-domain feature selection defaults",
        "implementation_status": "implemented",
    },
    "constant_value_filter": {
        "id": "constant_value_filter",
        "stage": "feature_refine",
        "description": "Drop constant or near-constant features using the configured unique-value threshold.",
        "default_params": {"max_unique_values": 1},
        "source_reference": "Fujie GCard constant-value filtering flow",
        "implementation_status": "implemented",
    },
    "iv_filter": {
        "id": "iv_filter",
        "stage": "feature_refine",
        "description": "Drop features below the configured IV threshold.",
        "default_params": {"min_iv": 0.005},
        "source_reference": "AIAgent three-domain feature selection defaults",
        "implementation_status": "implemented",
    },
    "psi_filter": {
        "id": "psi_filter",
        "stage": "feature_refine",
        "description": "Drop unstable features above the configured PSI threshold.",
        "default_params": {"max_psi": 0.2},
        "source_reference": "AIAgent three-domain feature selection defaults",
        "implementation_status": "implemented",
    },
    "correlation_dedup": {
        "id": "correlation_dedup",
        "stage": "feature_refine",
        "description": "Deduplicate highly correlated features, preferring stronger IV.",
        "default_params": {"method": "spearman", "max_abs_corr": 0.8},
        "source_reference": "AIAgent DZ in-loan modeling SOP",
        "implementation_status": "implemented",
    },
    "random_noise_importance": {
        "id": "random_noise_importance",
        "stage": "feature_refine",
        "description": "Filter features using feature-select-v2 compatible random-column importance checks.",
        "default_params": {"mode": "feature_select_v2", "bagging_rounds": 5, "bagging_fraction": 0.5, "thresholds": 0.95, "importance_types": ["split", "gain"]},
        "source_reference": "feature-select-v2 D03 random importance flow",
        "implementation_status": "implemented",
    },
    "null_importance_filter": {
        "id": "null_importance_filter",
        "stage": "feature_refine",
        "description": "Filter features whose real importance does not beat shuffled-label null importance.",
        "default_params": {"null_rounds": 20, "null_percentile": 75, "score_threshold": 1.0},
        "source_reference": "Fujie GCard D04 null importance flow",
        "implementation_status": "implemented",
    },
    "baseline_importance_filter": {
        "id": "baseline_importance_filter",
        "stage": "feature_refine",
        "description": "Train a baseline model and keep important features by LightGBM gain ranking.",
        "default_params": {"importance_type": "gain", "keep_top_n": 500},
        "source_reference": "Fujie GCard D05 baseline importance flow",
        "implementation_status": "implemented",
    },
    "gain_importance_filter": {
        "id": "gain_importance_filter",
        "stage": "feature_refine",
        "description": "Compatibility alias for low-gain tail filtering in older requests.",
        "default_params": {"tail_fraction": 0.1, "max_auc_drop": 0.005},
        "source_reference": "AIAgent three-domain feature selection defaults",
        "implementation_status": "implemented",
    },
    "lightgbm_binary_training": {
        "id": "lightgbm_binary_training",
        "stage": "train_baseline",
        "description": "Train a standard binary LightGBM model.",
        "default_params": {"early_stopping_rounds": 50, "max_auc_gap": 0.02},
        "source_reference": "AIAgent three-domain model training defaults",
        "implementation_status": "implemented",
    },
    "llm_guided_tuning": {
        "id": "llm_guided_tuning",
        "stage": "train_baseline",
        "description": "Run bounded host-agent LightGBM candidate tuning and select the final trial by metric guardrails.",
        "default_params": {"max_rounds": 2, "candidates_per_round": 4, "max_trials": 8},
        "source_reference": "RMW host-agent tuning flow",
        "implementation_status": "implemented",
    },
    "scale_pos_weight": {
        "id": "scale_pos_weight",
        "stage": "train_baseline",
        "description": "Configure positive-class weighting for imbalanced labels.",
        "default_params": {"mode": "negative_over_positive"},
        "source_reference": "AIAgent DZ in-loan modeling SOP",
        "implementation_status": "implemented",
    },
    "hier_ranknet_training": {
        "id": "hier_ranknet_training",
        "stage": "train_baseline",
        "description": "Train acquisition conversion HierRankNet multi-objective model.",
        "default_params": {},
        "source_reference": "AIAgent Hk HierRankNet skill",
        "implementation_status": "implemented",
    },
    "teacher_student_distillation": {
        "id": "teacher_student_distillation",
        "stage": "train_baseline",
        "description": "Train acquisition quality teacher-student distillation chain.",
        "default_params": {},
        "source_reference": "AIAgent Hk quality distillation skill",
        "implementation_status": "implemented",
    },
    "auc_ks": {
        "id": "auc_ks",
        "stage": "evaluate",
        "description": "Evaluate AUC and KS by split.",
        "default_params": {},
        "source_reference": "RMW evaluation flow",
        "implementation_status": "implemented",
    },
    "decile_lift": {
        "id": "decile_lift",
        "stage": "evaluate",
        "description": "Evaluate decile lift and risk ordering.",
        "default_params": {"bins": 10},
        "source_reference": "RMW evaluation flow",
        "implementation_status": "implemented",
    },
    "monthly_stability": {
        "id": "monthly_stability",
        "stage": "evaluate",
        "description": "Evaluate monthly AUC/KS stability.",
        "default_params": {"min_samples": 50},
        "source_reference": "RMW evaluation flow",
        "implementation_status": "implemented",
    },
    "score_psi": {
        "id": "score_psi",
        "stage": "evaluate",
        "description": "Evaluate monthly score distribution PSI.",
        "default_params": {"bins": 10, "warn_psi": 0.2},
        "source_reference": "RMW evaluation flow",
        "implementation_status": "implemented",
    },
    "score_psi_bin_detail": {
        "id": "score_psi_bin_detail",
        "stage": "evaluate",
        "description": "Per-bin PSI component (base/current score distribution) by month — the bin-level detail behind score_psi.",
        "default_params": {"bins": 10},
        "source_reference": "RMW evaluation flow (stability.compute_score_psi)",
        "implementation_status": "implemented",
    },
    "segment_metrics": {
        "id": "segment_metrics",
        "stage": "evaluate",
        "description": "Evaluate model performance by configured business segments.",
        "default_params": {},
        "source_reference": "RMW evaluation flow",
        "implementation_status": "implemented",
    },
    "intent_zc_cross_risk": {
        "id": "intent_zc_cross_risk",
        "stage": "evaluate",
        "description": "Evaluate intent by qualification cross-risk distributions.",
        "default_params": {},
        "source_reference": "Fujie GCard evaluation outputs",
        "implementation_status": "implemented",
    },
    "intent_risk_segmented": {
        "id": "intent_risk_segmented",
        "stage": "evaluate",
        "description": "Intent x qualification risk matrices per business segment (e2e3/b2), in addition to the cohort matrix.",
        "default_params": {"segments": ["e2e3", "b2"]},
        "source_reference": "RMW evaluation flow (_write_intent_risk_outputs)",
        "implementation_status": "implemented",
    },
    "cross_gain_matrix": {
        "id": "cross_gain_matrix",
        "stage": "evaluate",
        "description": "Evaluate 10x10 cross-gain against a baseline score.",
        "default_params": {"bins": 10},
        "source_reference": "AIAgent DQ feature gain and evaluation SOPs",
        "implementation_status": "implemented",
    },
    "roll_rate_analysis": {
        "id": "roll_rate_analysis",
        "stage": "evaluate",
        "description": "Evaluate in-loan account roll-rate by score band.",
        "default_params": {},
        "source_reference": "AIAgent DZ in-loan modeling SOP",
        "implementation_status": "implemented",
    },
    "channel_metrics": {
        "id": "channel_metrics",
        "stage": "evaluate",
        "description": "Evaluate acquisition model performance by channel.",
        "default_params": {},
        "source_reference": "AIAgent Hk acquisition modeling SOP",
        "implementation_status": "implemented",
    },
    "dual_model_synergy": {
        "id": "dual_model_synergy",
        "stage": "evaluate",
        "description": "Evaluate acquisition quality and conversion model synergy.",
        "default_params": {},
        "source_reference": "AIAgent Hk acquisition modeling SOP",
        "implementation_status": "implemented",
    },
    "sub_funnel_metrics": {
        "id": "sub_funnel_metrics",
        "stage": "evaluate",
        "description": "Evaluate acquisition conversion sub-funnel ordering.",
        "default_params": {},
        "source_reference": "AIAgent Hk HierRankNet skill",
        "implementation_status": "implemented",
    },
    "credit_product_standalone_eval": {
        "id": "credit_product_standalone_eval",
        "stage": "evaluate",
        "description": "Evaluate third-party credit product standalone performance.",
        "default_params": {},
        "source_reference": "AIAgent credit product evaluation SOP",
        "implementation_status": "implemented",
    },
    "credit_product_fusion_eval": {
        "id": "credit_product_fusion_eval",
        "stage": "evaluate",
        "description": "Evaluate third-party credit product fusion uplift.",
        "default_params": {},
        "source_reference": "AIAgent credit product evaluation SOP",
        "implementation_status": "implemented",
    },
    "feature_gain_summary": {
        "id": "feature_gain_summary",
        "stage": "evaluate",
        "description": "Summarize new-feature uplift and contribution.",
        "default_params": {},
        "source_reference": "AIAgent feature gain evaluation SOP",
        "implementation_status": "implemented",
    },
    "champion_challenger": {
        "id": "champion_challenger",
        "stage": "compare",
        "description": "Compare current model score with configured champion scores.",
        "default_params": {},
        "source_reference": "RMW compare flow",
        "implementation_status": "implemented",
    },
    "model_report": {
        "id": "model_report",
        "stage": "report",
        "description": "Generate model report, model card, and executive summary.",
        "default_params": {},
        "source_reference": "RMW report flow",
        "implementation_status": "implemented",
    },
    "model_recovery_report": {
        "id": "model_recovery_report",
        "stage": "report",
        "description": "Generate model recovery monitoring report.",
        "default_params": {},
        "source_reference": "AIAgent model recovery SOP",
        "implementation_status": "implemented",
    },
    "credit_product_report": {
        "id": "credit_product_report",
        "stage": "report",
        "description": "Generate third-party credit product evaluation report.",
        "default_params": {},
        "source_reference": "AIAgent credit product evaluation SOP",
        "implementation_status": "implemented",
    },
}


PROFILE_STAGE_STEPS: dict[str, dict[str, list[str]]] = {
    "acquisition": {
        "sample_check": [
            "field_contract",
            "key_uniqueness",
            "monthly_label_distribution",
            "channel_distribution",
            "dual_target_split",
        ],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training", "teacher_student_distillation", "hier_ranknet_training"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "channel_metrics",
            "sub_funnel_metrics",
            "dual_model_synergy",
        ],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "preloan_credit_card": {
        "sample_check": ["field_contract", "key_uniqueness", "monthly_label_distribution"],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training"],
        "evaluate": ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "cross_gain_matrix"],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "inloan_behavior_card": {
        "sample_check": [
            "field_contract",
            "key_uniqueness",
            "monthly_label_distribution",
            "account_status_distribution",
        ],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "correlation_dedup",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training", "scale_pos_weight"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "cross_gain_matrix",
            "roll_rate_analysis",
        ],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "inloan_operation": {
        "sample_check": ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "correlation_dedup",
            "random_noise_importance",
            "null_importance_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "segment_metrics",
            "cross_gain_matrix",
            "feature_gain_summary",
        ],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "acquisition_quality": {
        "sample_check": [
            "field_contract",
            "key_uniqueness",
            "monthly_label_distribution",
            "channel_distribution",
            "dual_target_split",
        ],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training", "teacher_student_distillation"],
        "evaluate": ["auc_ks", "decile_lift", "monthly_stability", "score_psi", "channel_metrics", "dual_model_synergy"],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "acquisition_conversion": {
        "sample_check": [
            "field_contract",
            "key_uniqueness",
            "monthly_label_distribution",
            "channel_distribution",
            "dual_target_split",
        ],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training", "hier_ranknet_training"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "channel_metrics",
            "sub_funnel_metrics",
            "dual_model_synergy",
        ],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "feature_gain_eval": {
        "sample_check": ["field_contract", "key_uniqueness", "monthly_label_distribution"],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "psi_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training"],
        "evaluate": ["auc_ks", "decile_lift", "monthly_stability", "cross_gain_matrix", "feature_gain_summary"],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
    "credit_product_eval": {
        "sample_check": ["field_contract", "key_uniqueness", "monthly_label_distribution", "credit_product_coverage"],
        "train_baseline": ["lightgbm_binary_training"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "credit_product_standalone_eval",
            "credit_product_fusion_eval",
        ],
        "compare": ["champion_challenger"],
        "report": ["credit_product_report"],
    },
    "fujie_gcard_main_lgbm": {
        "sample_check": ["field_contract", "key_uniqueness", "monthly_label_distribution", "segment_distribution"],
        "feature_metadata": ["feature_metadata_export"],
        "feature_prescreen": ["feature_quality_prescreen"],
        "build_wide_sql": ["wide_sql_generation", "sql_review_gate"],
        "feature_refine": [
            "feature_availability_filter",
            "missing_rate_filter",
            "constant_value_filter",
            "iv_filter",
            "correlation_dedup",
            "random_noise_importance",
            "null_importance_filter",
            "baseline_importance_filter",
        ],
        "train_baseline": ["lightgbm_binary_training"],
        "evaluate": [
            "auc_ks",
            "decile_lift",
            "monthly_stability",
            "score_psi",
            "segment_metrics",
            "intent_zc_cross_risk",
        ],
        "compare": ["champion_challenger"],
        "report": ["model_report"],
    },
}


BUSINESS_DOMAIN_PROFILE_DEFAULTS = {
    "acquisition": "acquisition",
    "preloan": "preloan_credit_card",
    "inloan_risk": "inloan_behavior_card",
    "inloan_operation": "inloan_operation",
}


PROFILE_STEP_PARAMS: dict[str, dict[str, dict[str, Any]]] = {
    "inloan_behavior_card": {
        "psi_filter": {"max_psi": 0.25},
        "monthly_stability": {"max_ks_std": 0.03},
    },
    "fujie_gcard_main_lgbm": {
        "feature_quality_prescreen": {"require_sql_approval": True},
        "sql_review_gate": {"block_on_high_risk": True},
    },
}


TEMPLATE_PROFILE_DEFAULTS = {
    "fujie-gcard": "fujie_gcard_main_lgbm",
    "generic": "preloan_credit_card",
}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item.get("name") if isinstance(item, dict) else item) for item in value]
    return [str(value)]


def _normalize_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _normalize_step_ids(steps: list[str]) -> list[str]:
    return [STEP_ALIASES.get(step, step) for step in steps]


def _load_project_yaml(project_path: str | Path | None) -> dict[str, Any]:
    if project_path is None:
        return {}
    project_dir = Path(project_path)
    for name in ["project.yml", "project.yaml"]:
        path = project_dir / name
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
    return {}


def infer_scenario_profile(metadata: dict[str, Any], project_path: str | Path | None = None) -> str:
    """Infer the scenario profile from request metadata or project template."""
    business_domain = metadata.get("business_domain")
    if business_domain and str(business_domain) not in BUSINESS_DOMAIN_PROFILE_DEFAULTS:
        raise ValueError(f"unknown business_domain: {business_domain}")

    explicit = metadata.get("scenario_profile")
    if explicit:
        return str(explicit)

    if business_domain:
        return BUSINESS_DOMAIN_PROFILE_DEFAULTS[str(business_domain)]

    project_config = _load_project_yaml(project_path)
    template = project_config.get("project", {}).get("template")
    return TEMPLATE_PROFILE_DEFAULTS.get(str(template), "preloan_credit_card")


def resolve_step_configuration(
    metadata: dict[str, Any],
    project_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve profile defaults plus request-level step overrides."""
    project_config = _load_project_yaml(project_path)
    profile = infer_scenario_profile(metadata, project_path)
    if profile not in PROFILE_STAGE_STEPS:
        raise ValueError(f"unknown scenario_profile: {profile}")

    stage_steps = deepcopy(PROFILE_STAGE_STEPS[profile])
    overrides = metadata.get("stage_steps") or {}
    if not isinstance(overrides, dict):
        raise ValueError("stage_steps must be a mapping")
    for stage, raw_steps in overrides.items():
        stage_name = _normalize_stage(str(stage))
        if stage_name not in KNOWN_STAGES:
            raise ValueError(f"unknown stage in stage_steps: {stage_name}")
        stage_steps[stage_name] = _normalize_step_ids(_as_list(raw_steps))

    effective_training = effective_training_config(metadata, project_config)
    if llm_guided_tuning_enabled(effective_training):
        train_steps = stage_steps.setdefault("train_baseline", [])
        if "llm_guided_tuning" not in train_steps:
            train_steps.append("llm_guided_tuning")

    unknown_steps = sorted({step for steps in stage_steps.values() for step in steps if step not in STEP_REGISTRY})
    if unknown_steps:
        raise ValueError(f"unknown step id: {', '.join(unknown_steps)}")

    request_params = metadata.get("step_params") or {}
    if not isinstance(request_params, dict):
        raise ValueError("step_params must be a mapping")
    request_params = {STEP_ALIASES.get(str(step), str(step)): params for step, params in request_params.items()}
    unknown_param_steps = sorted(set(request_params) - set(STEP_REGISTRY))
    if unknown_param_steps:
        raise ValueError(f"unknown step id in step_params: {', '.join(unknown_param_steps)}")

    used_steps = sorted({step for steps in stage_steps.values() for step in steps})
    step_params: dict[str, dict[str, Any]] = {}
    for step_id in used_steps:
        params = deepcopy(STEP_REGISTRY[step_id].get("default_params", {}))
        params.update(deepcopy(PROFILE_STEP_PARAMS.get(profile, {}).get(step_id, {})))
        if step_id == "llm_guided_tuning":
            tuning_params = effective_training.get("tuning") if isinstance(effective_training.get("tuning"), dict) else {}
            params.update(deepcopy(tuning_params))
        if step_id in request_params:
            if not isinstance(request_params[step_id], dict):
                raise ValueError(f"step_params.{step_id} must be a mapping")
            params.update(deepcopy(request_params[step_id]))
        step_params[step_id] = params

    resolved_steps = [deepcopy(STEP_REGISTRY[step_id]) for step_id in used_steps]
    planned_steps = [step for step in resolved_steps if step["implementation_status"] == "planned"]
    implemented_steps = [step for step in resolved_steps if step["implementation_status"] == "implemented"]
    return {
        "scenario_profile": profile,
        "stage_steps": stage_steps,
        "step_params": step_params,
        "resolved_steps": resolved_steps,
        "planned_steps": planned_steps,
        "implemented_steps": implemented_steps,
    }


def implemented_step_ids_for_stage(step_config: dict[str, Any], stage: str) -> list[str]:
    """Return implemented step ids for a stage from a resolved step config."""
    return [
        step_id
        for step_id in step_config.get("stage_steps", {}).get(stage, [])
        if STEP_REGISTRY[step_id]["implementation_status"] == "implemented"
    ]


def step_params_for(step_config: dict[str, Any], step_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Return parameter mapping for selected step ids."""
    params = step_config.get("step_params", {})
    return {step_id: deepcopy(params.get(step_id, {})) for step_id in step_ids}
