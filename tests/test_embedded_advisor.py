"""Embedded Advisor integration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.agent.advisor import create_advisor_request, load_advisor_request
from risk_model_workbench.agent.embedded_advisor import answer_advisor_request
from risk_model_workbench.agent.model_gateway import FakeModelGateway
from risk_model_workbench.cli import main


def test_embedded_advisor_answers_through_existing_validated_protocol(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260823"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = {
        "task_id": "train_main",
        "type": "train",
        "status": "pending",
        "depends_on": [],
        "command": {"executable": "rmw", "args": ["train", "--project", str(project), "--version-id", version_id]},
        "action_id": "sample_check",
        "tool_name": "sample_check",
    }
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=version_id,
        task=task,
        reason="data_missing",
        message="sample evidence is missing",
    )
    gateway = FakeModelGateway(
        [
            {
                "type": "data_gap_decision",
                "status": "answered",
                "decision": "stop",
                "summary": "Stop because required sample evidence is unavailable.",
                "risk_notes": ["A scaffold must not be treated as a trained model."],
            }
        ]
    )

    result = answer_advisor_request(workspace, request["request_id"], gateway)
    updated = load_advisor_request(workspace, request["request_id"])
    stored_response = json.loads((workspace / updated["accepted_response"]).read_text(encoding="utf-8"))

    assert result["accepted"] is True
    assert updated["status"] == "answered"
    assert stored_response["generated_by"] == "embedded_agent"
    assert stored_response["decision"] == "stop"
    assert (workspace / stored_response["model_invocation"]).exists()


def test_embedded_advisor_writes_bounded_tuning_plan_for_existing_trainer(tmp_path):
    project = _make_project(tmp_path)
    version_id = "demo_model_v2_20260823"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    context = workspace / "modeling" / "main_lgbm" / "tuning_context_round_1.json"
    context.parent.mkdir(parents=True, exist_ok=True)
    context.write_text('{"round":1,"experiment":"main_lgbm","trial_history":[]}\n', encoding="utf-8")
    task = {
        "task_id": "train_main",
        "type": "train",
        "status": "pending",
        "depends_on": [],
        "command": {
            "executable": "rmw",
            "args": ["train", "--project", str(project), "--version-id", version_id, "--experiment", "main_lgbm"],
        },
        "action_id": "train_baseline",
        "tool_name": "train_baseline",
    }
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=version_id,
        task=task,
        reason="advisor_required",
    )
    candidates = [
        {
            "name": "regularized_capacity",
            "params": {
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 120,
                "reg_lambda": 2.0,
                "num_boost_round": 1000,
                "early_stopping_rounds": 80,
            },
            "reason": "Bound model capacity and control overfitting.",
        },
        {
            "name": "conservative_depth",
            "params": {
                "learning_rate": 0.04,
                "num_leaves": 23,
                "max_depth": 5,
                "min_child_samples": 180,
                "reg_lambda": 4.0,
                "num_boost_round": 900,
                "early_stopping_rounds": 70,
            },
            "reason": "Test a lower-variance alternative.",
        },
        {
            "name": "balanced_sampling",
            "params": {
                "learning_rate": 0.025,
                "num_leaves": 47,
                "max_depth": 7,
                "min_child_samples": 100,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "bagging_freq": 3,
                "num_boost_round": 1200,
                "early_stopping_rounds": 90,
            },
            "reason": "Explore moderate capacity with bounded sampling.",
        },
    ]
    gateway = FakeModelGateway(
        [
            {
                "type": "tuning_plan",
                "status": "answered",
                "decision": "continue",
                "summary": "Run three bounded candidates against validation metrics.",
                "tuning_plan": {
                    "round": 1,
                    "diagnosis": "No trial history is available; start with bounded capacity variants.",
                    "candidates": candidates,
                    "stop": False,
                },
            }
        ]
    )

    result = answer_advisor_request(workspace, request["request_id"], gateway)
    plan_path = workspace / result["output_files"][0]
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan_path.name == "llm_tuning_plan_round_1.json"
    assert plan["advisor_type"] == "embedded_langgraph"
    assert plan["experiment"] == "main_lgbm"
    assert len(plan["candidates"]) == 3


@pytest.mark.parametrize(
    ("request_type", "response_type", "decision", "confirmation"),
    [
        ("failure_diagnosis_required", "failure_diagnosis", "retry", False),
        ("product_decision_required", "product_decision", "needs_user_confirmation", True),
    ],
)
def test_embedded_advisor_supports_diagnosis_and_product_decisions(
    tmp_path, request_type, response_type, decision, confirmation
):
    project = _make_project(tmp_path)
    version_id = f"demo_{response_type}_v1"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    workspace = project / "versions" / version_id
    task = {
        "task_id": "sample_check_1",
        "status": "pending",
        "depends_on": [],
        "command": {"executable": "rmw", "args": ["sample", "check", "--project", str(project), "--version-id", version_id]},
        "action_id": "sample_check",
        "tool_name": "sample_check",
    }
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=version_id,
        task=task,
        reason="unknown",
        request_type=request_type,
    )
    gateway = FakeModelGateway(
        [
            {
                "type": response_type,
                "status": "answered",
                "decision": decision,
                "summary": "Use the bounded next action and preserve all approval gates.",
                "requires_user_confirmation": confirmation,
            }
        ]
    )

    result = answer_advisor_request(workspace, request["request_id"], gateway)

    assert result["decision"] == decision
    assert result["requires_user_confirmation"] is confirmation


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project", "project_key": "demo_model"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
