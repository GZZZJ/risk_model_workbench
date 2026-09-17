"""Small typed contracts; observations are facts, never tool routing instructions."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Hypothesis(Contract):
    claim: Literal["information_redundancy", "incremental_value", "other"] = "other"
    description: str = Field(min_length=1)
    status: Literal["untested", "supported", "rejected", "inconclusive"] = "untested"
    evidence: list[str] = Field(default_factory=list)


class Constraints(Contract):
    # Immutable policy in the runtime, not an LLM-editable switch.
    max_iterations: int = Field(default=2, ge=1, le=2)
    oot_for_tuning: bool = False
    new_external_source_requires_human: bool = True
    production_promotion_requires_human: bool = True
    label_change_requires_human: bool = True


class ExperimentRecord(Contract):
    action_id: str
    action_type: str
    outcome_id: str
    status: str
    summary: str
    training_experiments: int = 0


FORBIDDEN = {"next_step", "next_action", "recommended_next_action", "selected_actions", "steps", "workflow", "depends_on", "tasks", "sql", "python", "command"}


def reject_control_fields(value: JsonValue) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN.intersection(value)
        if forbidden:
            raise ValueError(f"control/code fields are forbidden: {sorted(forbidden)}")
        for child in value.values():
            reject_control_fields(child)
    elif isinstance(value, list):
        for child in value:
            reject_control_fields(child)
    elif isinstance(value, float):
        import math
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")


class ModelingState(Contract):
    schema_version: Literal["demo-1"] = "demo-1"
    session_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    iteration: int = Field(default=0, ge=0)
    goal: str = Field(min_length=1)
    business_context: dict[str, JsonValue]
    observations: dict[str, JsonValue] = Field(default_factory=dict)
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    experiment_history: list[ExperimentRecord] = Field(default_factory=list)
    constraints: Constraints = Field(default_factory=Constraints)
    status: Literal["running", "waiting_for_human", "budget_exhausted"] = "running"

    @model_validator(mode="after")
    def facts_only(self):
        reject_control_fields(self.observations)
        reject_control_fields(self.business_context)
        return self


class CandidateAction(Contract):
    action_type: str
    priority: int = Field(ge=1)
    reason: str


class SelectedAction(Contract):
    action_type: str
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def no_hidden_plan(self):
        reject_control_fields(self.parameters)
        return self


class DecisionContract(Contract):
    schema_version: Literal["demo-1"] = "demo-1"
    decision_id: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    state_summary: str = Field(min_length=1)
    primary_hypothesis: Hypothesis
    candidate_actions: list[CandidateAction] = Field(min_length=1)
    selected_action: SelectedAction
    reason: str = Field(min_length=1)
    expected_evidence: str = Field(min_length=1)
    open_questions: list[str] = Field(default_factory=list)
    retrieved_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_candidates(self):
        names = [c.action_type for c in self.candidate_actions]
        priorities = [c.priority for c in self.candidate_actions]
        if len(names) != len(set(names)) or len(priorities) != len(set(priorities)):
            raise ValueError("candidate actions and priorities must be unique")
        if self.selected_action.action_type not in names:
            raise ValueError("selected action must be one of the ranked candidates")
        return self


class ParameterSpec(Contract):
    kind: Literal["string", "boolean"] = "string"
    allowed_values: list[str] = Field(default_factory=list)
    required: bool = False


class SemanticActionSpec(Contract):
    action_type: str
    description: str
    category: Literal["diagnosis", "experiment", "meta"]
    allowed_parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    underlying_existing_tool: str | None = None
    implementation: Literal["REAL", "ADAPTER", "MOCK"]
    implementation_note: str
    guardrails: list[str] = Field(default_factory=list)
    training_experiments: int = Field(default=0, ge=0)


class ActionOutcome(Contract):
    schema_version: Literal["demo-1"] = "demo-1"
    outcome_id: str
    action_id: str
    action_type: str
    state_id_before: str
    execution_status: Literal["success", "denied", "waiting_for_human", "unavailable"]
    implementation: Literal["REAL", "ADAPTER", "MOCK"]
    summary: str
    observations: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    guardrail_codes: list[str] = Field(default_factory=list)
    training_experiments: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def facts_only(self):
        reject_control_fields(self.observations)
        return self


class TrajectoryStep(Contract):
    state_before: ModelingState
    decision: DecisionContract
    retrieval: dict[str, JsonValue]
    outcome: ActionOutcome
    state_after: ModelingState
    adapter: str
