"""Non-negotiable gates. Decision text never grants permission."""
from .action_registry import validate_action
from .schemas import ModelingState, SelectedAction, SemanticActionSpec


def check_guardrails(state: ModelingState, action: SelectedAction, registry: dict[str, SemanticActionSpec]) -> list[str]:
    spec = validate_action(action, registry)
    codes = []
    split = action.parameters.get("optimization_split")
    if action.action_type == "change_training_objective" and split == "OOT":
        codes.append("OOT_FOR_HYPERPARAMETER_TUNING_DENIED")
    if action.action_type in {"change_label_definition", "retrain_with_new_label"} and state.constraints.label_change_requires_human:
        codes.append("LABEL_CHANGE_REQUIRES_HUMAN")
    if action.parameters.get("new_external_source") and state.constraints.new_external_source_requires_human:
        codes.append("NEW_EXTERNAL_DATA_SOURCE_REQUIRES_HUMAN")
    if action.parameters.get("production_promotion") and state.constraints.production_promotion_requires_human:
        codes.append("PRODUCTION_PROMOTION_REQUIRES_HUMAN")
    return codes
