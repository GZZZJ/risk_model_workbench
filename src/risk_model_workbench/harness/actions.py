"""Declarative action specs for rmw workflow and utility commands."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from risk_model_workbench.harness.command_metadata import display_template
from risk_model_workbench.harness.errors import (
    ADVISOR_REQUIRED,
    ARTIFACT_CONTRACT_FAILED,
    DATA_MISSING,
    DEPENDENCY_MISSING,
    EXTERNAL_OUTCOME_UNKNOWN,
    SCAFFOLD_ONLY,
    SQL_APPROVAL_REQUIRED,
    TRANSIENT_IO,
    UNKNOWN,
)


@dataclass(frozen=True)
class ActionSpec:
    id: str
    command: str
    description: str
    kind: str
    stage: str | None = None
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    approval_required: bool = False
    approval_type: str = "none"
    retry_policy: str = "never"
    failure_codes: tuple[str, ...] = ()
    artifact_rules: tuple[str, ...] = ()
    mutates_run_state: bool = False
    mutates_manifest: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


COMMON_STAGE_FAILURES = (
    DATA_MISSING,
    ARTIFACT_CONTRACT_FAILED,
    SCAFFOLD_ONLY,
    DEPENDENCY_MISSING,
    TRANSIENT_IO,
    UNKNOWN,
)

# ToolSpec owns the execution_semantics value.  The harness keeps the retry
# predicate here so normal retries and crash recovery cannot drift apart.
SAFE_RECOVERY_SEMANTICS = frozenset({"read_only", "idempotent_write"})


def automatic_recovery_allowed(execution_semantics: str) -> bool:
    return execution_semantics in SAFE_RECOVERY_SEMANTICS


def declared_action_ids() -> frozenset[str]:
    """Return action identities that application handlers may implement."""
    return frozenset(spec.id for spec in ACTION_SPECS)


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(
        id="validate_config",
        command=display_template("validate_config"),
        description="Initialize a run and snapshot project configuration.",
        kind="stage",
        stage="validate_config",
        inputs=("project.yml", "configs/*.yml", "optional model_request.md", "optional execution_plan.yml"),
        outputs=("run_state.yml", "configs_snapshot/", "audit/artifact_manifest.json"),
        failure_codes=(DATA_MISSING, ARTIFACT_CONTRACT_FAILED, UNKNOWN),
        artifact_rules=("configs_snapshot", "configs_snapshot/*"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="sample_check",
        command=display_template("sample_check"),
        description="Profile sample fields, label distribution, splits, and configured segments.",
        kind="stage",
        stage="sample_check",
        inputs=("project.yml", "configs_snapshot/*"),
        outputs=("sample_check/sample_summary.json", "sample_check/sample_check_report.md"),
        failure_codes=COMMON_STAGE_FAILURES,
        artifact_rules=(
            "sample_check/sample_summary.json",
            "sample_check/sample_check_report.md",
            "sample_check/sample_split_summary.csv",
            "sample_check/label_distribution.csv",
            "sample_check/segment_distribution.csv",
        ),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="feature_metadata",
        command=display_template("feature_metadata"),
        description="Export feature table and column metadata for downstream screening.",
        kind="stage",
        stage="feature_metadata",
        inputs=("project.yml", "optional tables file"),
        outputs=("feature_metadata/*",),
        failure_codes=COMMON_STAGE_FAILURES,
        artifact_rules=("feature_metadata/*",),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="feature_prescreen",
        command=display_template("feature_prescreen"),
        description="Run or dry-run coarse feature prescreening before wide-table refinement.",
        kind="stage",
        stage="feature_prescreen",
        inputs=("project.yml", "feature metadata", "SQL review approval for DP pull"),
        outputs=("feature_selection/prescreen_*", "feature_selection/*_run_summary.json", "feature_selection/*_final_remain_features.json"),
        approval_required=True,
        approval_type="sql_review",
        retry_policy="never",
        failure_codes=(SQL_APPROVAL_REQUIRED, EXTERNAL_OUTCOME_UNKNOWN, *COMMON_STAGE_FAILURES),
        artifact_rules=("feature_selection/prescreen_*", "feature_selection/*_run_summary.json", "feature_selection/*_final_remain_features.json"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="build_wide_sql",
        command=display_template("build_wide_sql"),
        description="Generate wide-table SQL and feature mapping from remaining features.",
        kind="stage",
        stage="build_wide_sql",
        inputs=("feature prescreen remaining features", "project.yml"),
        outputs=("queries/*.sql", "feature_selection/wide_sql_summary.json", "feature_selection/wide_table_execution.json"),
        approval_required=True,
        approval_type="sql_review",
        failure_codes=(SQL_APPROVAL_REQUIRED, EXTERNAL_OUTCOME_UNKNOWN, *COMMON_STAGE_FAILURES),
        artifact_rules=("queries/*.sql", "feature_selection/*wide*summary.json", "feature_selection/wide_table_execution.json", "feature_selection/*feature_map.csv"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="feature_refine",
        command=display_template("feature_refine"),
        description="Filter executable model features and produce final feature lists.",
        kind="stage",
        stage="feature_refine",
        inputs=("training data feather", "wide SQL output", "SQL review approval for DP pull"),
        outputs=("feature_selection/final_features.txt", "feature_selection/final_500_features.txt", "feature_selection/stage_summary.json"),
        approval_required=True,
        approval_type="sql_review",
        retry_policy="never",
        failure_codes=(SQL_APPROVAL_REQUIRED, EXTERNAL_OUTCOME_UNKNOWN, *COMMON_STAGE_FAILURES),
        artifact_rules=("feature_selection/final_features.txt", "feature_selection/final_500_features.txt", "feature_selection/*stage_summary.json"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="train_baseline",
        command=display_template("train_baseline"),
        description="Train the baseline model or create an explicit scaffold when real training inputs are absent.",
        kind="stage",
        stage="train_baseline",
        inputs=("training data feather", "feature_selection/final_features.txt", "project.yml"),
        outputs=(
            "modeling/*/metrics_train_valid.json",
            "modeling/*/training_status.json",
            "modeling/*/training_summary.md",
            "modeling/*/train_plan.json",
            "modeling/*/train_plan.md",
            "modeling/*/actual_feature_list.txt",
            "modeling/*/feature_importance.csv",
            "modeling/*/tuning_summary.json",
            "modeling/*/tuning_trials.csv",
            "modeling/*/best_params.json",
        ),
        failure_codes=(*COMMON_STAGE_FAILURES, ADVISOR_REQUIRED),
        artifact_rules=(
            "modeling/*/metrics_train_valid.json",
            "modeling/*/actual_feature_list.txt",
            "modeling/*/feature_importance.csv",
            "modeling/*/train_metrics.json",
            "modeling/*/training_status.json",
            "modeling/*/training_summary.md",
            "modeling/*/train_plan.json",
            "modeling/*/train_plan.md",
            "modeling/*/tuning_summary.json",
            "modeling/*/tuning_trials.csv",
            "modeling/*/best_params.json",
            "modeling/*/llm_tuning_decisions.md",
        ),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="evaluate",
        command=display_template("evaluate"),
        description="Evaluate model scores and generate evaluation summaries.",
        kind="stage",
        stage="evaluate",
        inputs=("model scores feather", "project.yml"),
        outputs=("evaluation/evaluation_summary.json",),
        failure_codes=COMMON_STAGE_FAILURES,
        artifact_rules=("evaluation/evaluation_summary.json",),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="compare",
        command=display_template("compare"),
        description="Compare current model evidence with configured champion or benchmark scores.",
        kind="stage",
        stage="compare",
        inputs=("evaluation/evaluation_summary.json", "optional champion score"),
        outputs=("evaluation/champion_challenger.json", "evaluation/benchmark_uplift.csv"),
        failure_codes=COMMON_STAGE_FAILURES,
        artifact_rules=("evaluation/champion_challenger.json", "evaluation/benchmark_uplift.csv"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="report",
        command=display_template("report"),
        description="Generate Markdown, HTML, model card, executive summary, and optional Excel report.",
        kind="stage",
        stage="report",
        inputs=("run_state.yml", "audit/artifact_manifest.json", "modeling artifacts", "evaluation artifacts"),
        outputs=("reports/model_report.xlsx", "reports/model_report.md", "reports/model_report.html", "reports/model_card.md", "reports/executive_summary.md"),
        failure_codes=COMMON_STAGE_FAILURES,
        artifact_rules=("reports/model_report.xlsx", "reports/model_report.md", "reports/model_report.html", "reports/model_card.md", "reports/executive_summary.md"),
        mutates_run_state=True,
        mutates_manifest=True,
    ),
    ActionSpec(
        id="project_status",
        command=display_template("project_status"),
        description="Summarize project continuity state and active run status.",
        kind="utility",
        inputs=("project_state.yml", "optional run_state.yml", "optional artifact_manifest.json"),
        outputs=("stdout",),
        retry_policy="transient_io",
        failure_codes=(DATA_MISSING, UNKNOWN),
    ),
    ActionSpec(
        id="run_status",
        command=display_template("run_status"),
        description="Show the run state source of truth.",
        kind="utility",
        inputs=("run_state.yml",),
        outputs=("stdout",),
        retry_policy="transient_io",
        failure_codes=(DATA_MISSING, UNKNOWN),
    ),
    ActionSpec(
        id="run_audit",
        command=display_template("run_audit"),
        description="Audit run or stage closure readiness against run state, manifest, and workflow contracts.",
        kind="audit",
        inputs=("run_state.yml", "audit/artifact_manifest.json", "workflow contract"),
        outputs=("stdout", "optional JSON"),
        retry_policy="transient_io",
        failure_codes=(DATA_MISSING, ARTIFACT_CONTRACT_FAILED, SCAFFOLD_ONLY, UNKNOWN),
    ),
    ActionSpec(
        id="workflow_validate",
        command=display_template("workflow_validate"),
        description="Validate workflow shape and stage contract syntax.",
        kind="audit",
        inputs=("workflow YAML",),
        outputs=("stdout",),
        retry_policy="transient_io",
        failure_codes=(ARTIFACT_CONTRACT_FAILED, DATA_MISSING, UNKNOWN),
    ),
    ActionSpec(
        id="rules_list",
        command=display_template("rules_list"),
        description="List promoted workbench rules and guardrails.",
        kind="utility",
        inputs=("docs/workbench_rules.yml",),
        outputs=("stdout", "optional JSON"),
        retry_policy="transient_io",
        failure_codes=(DATA_MISSING, UNKNOWN),
    ),
)

ACTION_REGISTRY: dict[str, ActionSpec] = {spec.id: spec for spec in ACTION_SPECS}
ACTION_ALIASES = {
    "d01_d02_screening": "feature_prescreen",
}


def list_action_specs(*, kind: str | None = None) -> tuple[ActionSpec, ...]:
    specs = ACTION_SPECS
    if kind:
        specs = tuple(spec for spec in specs if spec.kind == kind)
    return tuple(sorted(specs, key=lambda spec: spec.id))


def get_action_spec(action_id: str) -> ActionSpec:
    action_id = ACTION_ALIASES.get(action_id, action_id)
    try:
        return ACTION_REGISTRY[action_id]
    except KeyError as exc:
        raise KeyError(f"unknown action: {action_id}") from exc


def action_to_dict(spec: ActionSpec) -> dict[str, object]:
    return spec.to_dict()


def format_action_list(specs: tuple[ActionSpec, ...]) -> str:
    lines = ["Action ID              Stage             Kind     Approval     Retry       Command"]
    lines.append("-" * 96)
    for spec in specs:
        stage = spec.stage or "-"
        approval = spec.approval_type if spec.approval_required else "none"
        lines.append(
            f"{spec.id:<22} {stage:<17} {spec.kind:<8} {approval:<12} {spec.retry_policy:<11} {spec.command}"
        )
    return "\n".join(lines) + "\n"


def format_action_detail(spec: ActionSpec) -> str:
    payload = spec.to_dict()
    lines = [
        f"action: {payload['id']}",
        f"kind: {payload['kind']}",
        f"stage: {payload['stage'] or '-'}",
        f"command: {payload['command']}",
        f"description: {payload['description']}",
        f"approval: {payload['approval_type'] if payload['approval_required'] else 'none'}",
        f"retry_policy: {payload['retry_policy']}",
        f"mutates_run_state: {payload['mutates_run_state']}",
        f"mutates_manifest: {payload['mutates_manifest']}",
        "inputs:",
    ]
    lines.extend(f"- {item}" for item in spec.inputs)
    lines.append("outputs:")
    lines.extend(f"- {item}" for item in spec.outputs)
    lines.append("failure_codes:")
    lines.extend(f"- {item}" for item in spec.failure_codes)
    if spec.artifact_rules:
        lines.append("artifact_rules:")
        lines.extend(f"- {item}" for item in spec.artifact_rules)
    return "\n".join(lines) + "\n"
