"""Contract tests for the one auxiliary-model incremental-value demo."""
import json
import pytest
from risk_model_workbench.agentic.action_registry import build_registry
from risk_model_workbench.agentic.demo import main
from risk_model_workbench.agentic.demo_executor import DemoActionExecutor
from risk_model_workbench.agentic.demo_runtime import DemoRuntime
from risk_model_workbench.agentic.guardrails import check_guardrails
from risk_model_workbench.agentic.orchestrator import DemoFixtureOrchestratorAdapter
from risk_model_workbench.agentic.retrieval import EvidenceRetriever
from risk_model_workbench.agentic.scenarios import LABEL_SCENARIO, SCENARIO, SEGMENT_SCENARIO, initial_state
from risk_model_workbench.agentic.schemas import SelectedAction


def make_runtime(counterfactual=False):
    return DemoRuntime(DemoFixtureOrchestratorAdapter(), executor=DemoActionExecutor(counterfactual))


def test_two_round_real_knowledge_fixture_path():
    steps = make_runtime().run(initial_state(SCENARIO, "main"))
    assert [s.decision.selected_action.action_type for s in steps] == ["change_training_objective", "compare_challenger"]
    assert steps[0].decision.selected_action.parameters["objective"] == "covloss"
    assert "H_COVLOSS_OUTCOME" in steps[0].decision.retrieved_refs
    assert steps[1].state_after.status == "waiting_for_human"
    assert steps[0].outcome.implementation == "MOCK"
    assert "当前候选的精确重训指标" in steps[0].outcome.summary


def test_counterfactual_outcome_changes_second_action():
    normal = make_runtime().run(initial_state(SCENARIO, "normal"))
    counterfactual = make_runtime(True).run(initial_state(SCENARIO, "counter"))
    assert normal[0].decision.selected_action == make_runtime(True).run(initial_state(SCENARIO, "counter-first"))[0].decision.selected_action
    assert counterfactual[0].outcome.evidence == ["COUNTERFACTUAL MOCK"]
    assert counterfactual[1].decision.selected_action.action_type == "change_feature_space"
    assert normal[1].decision.selected_action.action_type == "compare_challenger"


def test_retrieval_keeps_decision_and_historical_experience_separate():
    retrieved = EvidenceRetriever().retrieve(initial_state(SCENARIO, "rag"))
    assert retrieved["decision_knowledge"] and retrieved["historical_experience"]
    assert all(row["kind"] == "decision_knowledge" and row["source_heading"] for row in retrieved["decision_knowledge"])
    assert any("squareloss" in row["fact_summary"] for row in retrieved["historical_experience"])


def test_historical_replay_cutoff_is_reserved_and_changes_retrieval():
    state = initial_state(SCENARIO, "replay")
    state.business_context["knowledge_mode"] = "historical_replay"
    state.business_context["as_of"] = "2020-01-01"
    retrieved = EvidenceRetriever().retrieve(state)
    assert not retrieved["decision_knowledge"] and not retrieved["historical_experience"]


@pytest.mark.parametrize("action,params,code", [
    ("change_training_objective", {"objective": "covloss", "optimization_split": "OOT"}, "OOT_FOR_HYPERPARAMETER_TUNING_DENIED"),
    ("change_label_definition", {"label_definition": "fpd15"}, "LABEL_CHANGE_REQUIRES_HUMAN"),
    ("change_sample_composition", {"strategy": "approved_differentiated_sample", "new_external_source": True}, "NEW_EXTERNAL_DATA_SOURCE_REQUIRES_HUMAN"),
    ("compare_challenger", {"production_promotion": True}, "PRODUCTION_PROMOTION_REQUIRES_HUMAN"),
])
def test_guardrails_are_deterministic(action, params, code):
    state = initial_state(SCENARIO, "guard")
    assert code in check_guardrails(state, SelectedAction(action_type=action, parameters=params), build_registry())


def test_cli_writes_two_round_trace(tmp_path):
    output = tmp_path / "demo"
    assert main(["--scenario", SCENARIO, "--output", str(output)]) == 0
    trace = (output / "demo_run.md").read_text()
    assert "change_training_objective" in trace and "compare_challenger" in trace
    current = json.loads((output / "current.json").read_text())
    assert current["status"] == "waiting_for_human"


def test_segment_specific_modeling_uses_negative_evidence_to_change_direction():
    steps = make_runtime().run(initial_state(SEGMENT_SCENARIO, "segment"))
    assert [step.decision.selected_action.action_type for step in steps] == ["segment_modeling", "rule_based_segment_plus_kmeans"]
    assert steps[0].outcome.observations["negative_evidence"] == ["KMeans alone hurts new_customer_segment"]
    assert steps[0].state_after.observations["segmentation_result"]["status"] == "partially_failed"
    assert steps[1].outcome.observations["segmentation_result"]["new_customer_ks_after"] == 0.2352


def test_label_definition_diagnosis_revisits_label_before_retrain_contract():
    steps = make_runtime().run(initial_state(LABEL_SCENARIO, "label"))
    assert [step.decision.selected_action.action_type for step in steps] == ["revisit_label_definition", "retrain_with_new_label"]
    assert steps[0].state_after.observations["label_issue"] == "confirmed"
    assert steps[1].outcome.execution_status == "denied"
    assert steps[1].outcome.guardrail_codes == ["LABEL_CHANGE_REQUIRES_HUMAN"]
