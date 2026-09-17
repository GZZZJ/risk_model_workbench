"""Typed contracts for the embedded RMW reasoning layer.

The model is allowed to recommend a bounded decision.  It never receives an
executable callback and it never supplies the request identity used by the
Harness; the runtime binds that identity after validating the model output.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from risk_model_workbench.agent.context_pack import context_pack_hash


AdvisorRequestType = Literal[
    "tuning_plan_required",
    "failure_diagnosis_required",
    "data_gap_decision_required",
    "product_decision_required",
]
AdvisorResponseType = Literal[
    "tuning_plan",
    "failure_diagnosis",
    "data_gap_decision",
    "product_decision",
]
AdvisorDecision = Literal["continue", "retry", "stop", "needs_user_confirmation"]


class TuningCandidate(BaseModel):
    """One bounded algorithm-specific candidate proposed by the embedded Advisor."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    params: dict[str, int | float | str | bool] = Field(default_factory=dict)
    reason: str = Field(min_length=1, max_length=1000)


class ModelDiagnosis(BaseModel):
    """LLM interpretation of deterministic, precomputed tuning evidence."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["overfit", "underfit", "healthy", "plateau", "unstable", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=2000)
    evidence: list[str] = Field(min_length=1, max_length=20)
    recommended_direction: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = Field(ge=0.0, le=1.0)


class TuningPlan(BaseModel):
    """File payload consumed by the existing LLM-guided tuning workflow."""

    model_config = ConfigDict(extra="forbid")

    algorithm: Literal["lightgbm", "xgboost"]
    experiment: str | None = Field(default=None, max_length=80)
    round: int = Field(ge=1, le=20)
    diagnosis: ModelDiagnosis
    decision: Literal["continue", "stop"]
    candidates: list[TuningCandidate] = Field(default_factory=list, max_length=5)
    stop_reason: Literal["advisor_recommends_stop", "insufficient_evidence", "manual_stop"] | None = None

    @model_validator(mode="after")
    def validate_tuning_shape(self) -> "TuningPlan":
        if self.decision == "continue":
            if not self.candidates:
                raise ValueError("continuing tuning plan requires at least one candidate")
            if self.stop_reason is not None:
                raise ValueError("continuing tuning plan must not include stop_reason")
        elif self.candidates:
            raise ValueError("stop tuning plan must not include candidates")
        elif self.stop_reason is None:
            raise ValueError("stop tuning plan requires stop_reason")
        return self


class EmbeddedAdvisorAnswer(BaseModel):
    """Provider-neutral structured answer returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    type: AdvisorResponseType
    status: Literal["answered", "rejected"] = "answered"
    decision: AdvisorDecision
    summary: str = Field(min_length=1, max_length=4000)
    risk_notes: list[str] = Field(default_factory=list, max_length=20)
    requires_user_confirmation: bool = False
    tuning_plan: TuningPlan | None = None

    @model_validator(mode="after")
    def validate_decision_shape(self) -> "EmbeddedAdvisorAnswer":
        if self.type == "tuning_plan" and self.status == "answered" and self.decision in {"continue", "retry", "stop"}:
            if self.tuning_plan is None:
                raise ValueError("tuning_plan is required for a tuning decision")
            if self.decision == "stop" and self.tuning_plan.decision != "stop":
                raise ValueError("stop Advisor decision requires a stop tuning plan")
            if self.decision in {"continue", "retry"} and self.tuning_plan.decision != "continue":
                raise ValueError("continuing Advisor decision requires a continuing tuning plan")
        elif self.tuning_plan is not None:
            raise ValueError("tuning_plan is only allowed for tuning_plan responses")
        if self.decision == "needs_user_confirmation" and not self.requires_user_confirmation:
            raise ValueError("needs_user_confirmation decision requires the confirmation flag")
        if self.requires_user_confirmation and self.decision != "needs_user_confirmation":
            raise ValueError("confirmation flag requires needs_user_confirmation decision")
        return self


class GoalInterpretation(BaseModel):
    """Bounded model interpretation of a natural-language modeling objective."""

    model_config = ConfigDict(extra="forbid")

    objective_summary: str = Field(min_length=1, max_length=2000)
    workflow: str = Field(min_length=1, max_length=80)
    experiment_name: str = Field(
        default="main_lgbm",
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    method: Literal[
        "lightgbm",
        "xgboost",
        "logistic_regression",
        "custom",
        "hier_ranknet",
        "teacher_student_distillation",
    ] = "lightgbm"
    training_mode: Literal["single_train", "llm_guided_tune"] = "llm_guided_tune"
    metrics: list[Literal["auc", "ks", "decile_lift", "ranking_inversion", "psi", "business_risk"]] = Field(
        default_factory=lambda: ["auc", "ks"], min_length=1, max_length=6
    )
    report_outputs: list[str] = Field(default_factory=lambda: ["model_report.md", "model_report.html"], min_length=1, max_length=6)
    scenario_profile: str | None = Field(default=None, max_length=120)
    assumptions: list[str] = Field(default_factory=list, max_length=20)
    requires_user_confirmation: bool = True

    @model_validator(mode="after")
    def validate_report_outputs(self) -> "GoalInterpretation":
        allowed = {"", ".md", ".markdown", ".html", ".xlsx", ".json"}
        for output in self.report_outputs:
            suffix = "." + output.rsplit(".", 1)[-1].lower() if "." in output else ""
            if suffix not in allowed or "/" in output or "\\" in output or ".." in output:
                raise ValueError(f"unsupported report output: {output}")
        return self


class GoalReasoningRequest(BaseModel):
    """Sanitized project contract plus one user objective."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    request_type: Literal["goal_interpretation"] = "goal_interpretation"
    expected_response_type: Literal["goal_plan"] = "goal_plan"
    question: str = Field(min_length=1, max_length=4000)
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_pack: dict[str, Any]
    constraints: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def validate_context_pack(self) -> "GoalReasoningRequest":
        if self.context_pack.get("context_hash") != self.context_hash:
            raise ValueError("context_hash does not match context pack")
        if context_pack_hash(self.context_pack) != self.context_hash:
            raise ValueError("context pack content hash mismatch")
        return self


class StructuredReasoningRequest(BaseModel):
    """Bound, sanitized input for one model decision."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    request_type: AdvisorRequestType
    expected_response_type: AdvisorResponseType
    question: str = Field(min_length=1, max_length=4000)
    context_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_pack: dict[str, Any]
    constraints: list[str] = Field(default_factory=list, max_length=100)
    max_context_bytes: int = Field(default=256 * 1024, ge=1024, le=1024 * 1024)

    @model_validator(mode="after")
    def validate_context_pack(self) -> "StructuredReasoningRequest":
        if self.context_pack.get("context_hash") != self.context_hash:
            raise ValueError("context_hash does not match context pack")
        if context_pack_hash(self.context_pack) != self.context_hash:
            raise ValueError("context pack content hash mismatch")
        serialized = json.dumps(
            self.context_pack,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        if len(serialized) > self.max_context_bytes:
            raise ValueError("context pack exceeds model context byte limit")
        files = self.context_pack.get("files")
        if not isinstance(files, list):
            raise ValueError("context pack files must be a list")
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("context pack file entries must be mappings")
            if item.get("status") == "included" and not isinstance(item.get("summary"), dict):
                raise ValueError("included context entry requires a deterministic summary")
        return self


class ModelUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelInvocationMetadata(BaseModel):
    """Non-sensitive evidence returned alongside a structured answer."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    attempts: int = Field(ge=1)
    latency_ms: int = Field(ge=0)
    request_id: str
    context_hash: str
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    response_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    usage: ModelUsage = Field(default_factory=ModelUsage)
