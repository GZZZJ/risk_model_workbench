"""Facts-only deterministic update. LLM cannot overwrite policy or measured metrics."""
from .schemas import ActionOutcome, DecisionContract, ExperimentRecord, Hypothesis, ModelingState


def reduce_state(state: ModelingState, decision: DecisionContract, outcome: ActionOutcome) -> ModelingState:
    if decision.state_id != state.state_id or outcome.state_id_before != state.state_id:
        raise ValueError("state/decision/outcome identity mismatch")
    if decision.selected_action.action_type != outcome.action_type:
        raise ValueError("outcome action mismatch")
    updated = state.model_copy(deep=True)
    if outcome.execution_status == "success":
        updated.observations.update(outcome.observations)
    # A proposal starts untested, regardless of the adapter's claimed verdict.
    hypothesis = Hypothesis(description=decision.primary_hypothesis.description, claim=decision.primary_hypothesis.claim)
    if outcome.execution_status == "success":
        hypothesis.status = "inconclusive"
        hypothesis.evidence = outcome.evidence
        if outcome.action_type == "change_training_objective" and hypothesis.claim == "information_redundancy":
            hypothesis.status = "supported" if outcome.observations.get("correlation_status") == "improved_or_feasible" else "rejected"
        if outcome.action_type == "compare_challenger" and hypothesis.claim == "incremental_value":
            hypothesis.status = "inconclusive"  # framework evidence is not local combination measurement
    updated.hypotheses.append(hypothesis)
    updated.open_questions = list(decision.open_questions)
    updated.experiment_history.append(ExperimentRecord(action_id=outcome.action_id, action_type=outcome.action_type,
        outcome_id=outcome.outcome_id, status=outcome.execution_status, summary=outcome.summary,
        training_experiments=outcome.training_experiments))
    updated.iteration += 1
    updated.state_id = f"{state.session_id}_{updated.iteration}"
    if outcome.execution_status in {"denied", "waiting_for_human"} or outcome.action_type == "compare_challenger":
        updated.status = "waiting_for_human"
    elif updated.iteration >= updated.constraints.max_iterations:
        updated.status = "budget_exhausted"
    return ModelingState.model_validate(updated.model_dump(mode="json"))
