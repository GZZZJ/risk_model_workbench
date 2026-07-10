import json
import os
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.agent.advisor import (
    accept_advisor_response,
    create_advisor_request,
    load_advisor_request,
    validate_advisor_response,
)
from risk_model_workbench.agent.advisor_reducer import consume_advisor_response
from risk_model_workbench.agent.plan import save_agent_plan
from risk_model_workbench.agent.state import init_agent_state, load_agent_state, pause_agent, save_agent_state
from risk_model_workbench.cli import main


VERSION_ID = "demo_model_v1_20260710"


def test_consume_continue_response_with_valid_output(tmp_path):
    project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="continue")

    accepted = accept_advisor_response(workspace, response_path)
    assert accepted["accepted"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)
    reloaded = load_agent_state(workspace)
    loaded_request = load_advisor_request(workspace, request["request_id"])
    receipt = _read_json(workspace / "audit" / "advisor_consumptions" / f"{request['request_id']}.json")

    assert result.consumed is True
    assert result.status == "running"
    assert reloaded["status"] == "running"
    assert reloaded["tasks"][0]["status"] == "pending"
    assert loaded_request["status"] == "consumed"
    assert receipt["decision"] == "continue"
    assert receipt["consumed"] is True


def test_consume_retry_is_bounded_by_request_budget(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    request["retry_budget"]["max_retries"] = 1
    _save_request(workspace, request)
    response_path = _write_response(workspace, request, decision="retry")
    assert accept_advisor_response(workspace, response_path)["accepted"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)
    assert result.consumed is True
    reloaded = load_agent_state(workspace)
    assert reloaded["tasks"][0]["advisor_retry_count"] == 1
    assert reloaded["tasks"][0]["status"] == "pending"

    request_2 = _clone_new_request(workspace, request, suffix="retry_budget")
    state_2 = pause_agent(
        workspace,
        status="waiting_for_advisor",
        reason="advisor_required",
        task_id="train_main",
        advisor_request=str(request_2["path"]),
        advisor_request_id=request_2["request_id"],
    )
    response_path_2 = _write_response(workspace, request_2, decision="retry")
    assert accept_advisor_response(workspace, response_path_2)["accepted"] is True

    rejected = consume_advisor_response(workspace, request_2["request_id"], state_2)
    assert rejected.consumed is False
    assert any("retry budget exceeded" in error for error in rejected.errors)


def test_consume_stop_moves_agent_to_terminal_stopped(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="stop", output_files=[])
    assert accept_advisor_response(workspace, response_path)["accepted"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)
    reloaded = load_agent_state(workspace)

    assert result.status == "stopped"
    assert reloaded["status"] == "stopped"
    assert reloaded["tasks"][0]["status"] == "stopped"


def test_consume_needs_user_confirmation_blocks_resume_until_confirmed(tmp_path, capsys):
    project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="needs_user_confirmation", output_files=[])
    assert accept_advisor_response(workspace, response_path)["accepted"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)
    assert result.status == "waiting_for_user"
    assert load_agent_state(workspace)["status"] == "waiting_for_user"

    assert main(["agent", "resume", "--project", str(project), "--version-id", VERSION_ID]) == 1
    assert "waiting for user confirmation" in capsys.readouterr().out

    assert (
        main(
            [
                "agent",
                "confirm",
                "--project",
                str(project),
                "--version-id",
                VERSION_ID,
                "--request-id",
                request["request_id"],
                "--confirmed-by",
                "tester",
            ]
        )
        == 0
    )
    assert load_agent_state(workspace)["status"] == "running"


def test_rejected_advisor_response_creates_replacement_request(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, status="rejected", decision="stop", output_files=[])
    accepted = accept_advisor_response(workspace, response_path)
    assert accepted["accepted"] is False
    assert accepted["stored"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)
    reloaded = load_agent_state(workspace)

    assert result.consumed is True
    assert result.status == "waiting_for_advisor"
    assert result.replacement_request_id
    assert result.replacement_request_id != request["request_id"]
    assert reloaded["blocker"]["advisor_request_id"] == result.replacement_request_id
    assert load_advisor_request(workspace, result.replacement_request_id)["status"] == "pending"


def test_duplicate_consumption_is_rejected(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="continue")
    assert accept_advisor_response(workspace, response_path)["accepted"] is True
    assert consume_advisor_response(workspace, request["request_id"], state).consumed is True

    duplicate = consume_advisor_response(workspace, request["request_id"], load_agent_state(workspace))
    assert duplicate.consumed is False
    assert any("already consumed" in error for error in duplicate.errors)

    overwritten = accept_advisor_response(workspace, response_path)
    assert overwritten["accepted"] is False
    assert any("not pending" in error for error in overwritten["errors"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("attempt_id", "attempt_stale"),
        ("invocation_hash", "stale_invocation"),
        ("context_hash", "stale_context"),
        ("round", 99),
    ],
)
def test_stale_response_identity_is_rejected(tmp_path, field, value):
    _project, workspace, request, _state = _paused_request(tmp_path)
    response = _response_payload(request, decision="continue")
    response["request_identity"][field] = value

    errors = validate_advisor_response(workspace, response)

    assert any("response identity mismatch" in error for error in errors)


def test_stale_current_context_is_rejected_at_consumption(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="continue")
    assert accept_advisor_response(workspace, response_path)["accepted"] is True
    (workspace / "modeling" / "main_lgbm" / "tuning_context_round_1.json").write_text(
        json.dumps({"round": 1, "changed": True}),
        encoding="utf-8",
    )

    result = consume_advisor_response(workspace, request["request_id"], state)

    assert result.consumed is False
    assert any("context hash mismatch" in error for error in result.errors)


def test_response_output_files_must_exist_and_stay_inside_workspace(tmp_path):
    _project, workspace, request, _state = _paused_request(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escape_link = workspace / "modeling" / "escape_link.json"
    escape_link.symlink_to(outside)

    assert any(
        "missing output file" in error
        for error in validate_advisor_response(workspace, _response_payload(request, output_files=["modeling/missing.json"]))
    )
    assert any(
        "workspace-relative" in error
        for error in validate_advisor_response(workspace, _response_payload(request, output_files=[str(outside.resolve())]))
    )
    assert any(
        "path escape" in error
        for error in validate_advisor_response(workspace, _response_payload(request, output_files=["../outside.txt"]))
    )
    assert any(
        "path escape" in error
        for error in validate_advisor_response(workspace, _response_payload(request, output_files=["modeling/escape_link.json"]))
    )


def test_tuning_plan_consumption_requires_output_plan_file(tmp_path):
    _project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="continue", output_files=[])
    assert accept_advisor_response(workspace, response_path)["accepted"] is True

    result = consume_advisor_response(workspace, request["request_id"], state)

    assert result.consumed is False
    assert any("requires llm_tuning_plan output" in error for error in result.errors)


def test_context_manifest_records_file_hash_size_and_missing_status(tmp_path):
    _project, workspace, request, _state = _paused_request(tmp_path)

    manifest = _read_json(workspace / request["context_manifest"])
    rows = {row["path"]: row for row in manifest["files"]}

    context_path = "modeling/main_lgbm/tuning_context_round_1.json"
    assert manifest["context_hash"] == request["context_hash"]
    assert rows[context_path]["missing"] is False
    assert rows[context_path]["size"] > 0
    assert len(rows[context_path]["sha256"]) == 64


def test_advisor_user_reject_cli_stops_waiting_confirmation(tmp_path, capsys):
    project, workspace, request, state = _paused_request(tmp_path)
    response_path = _write_response(workspace, request, decision="needs_user_confirmation", output_files=[])
    assert accept_advisor_response(workspace, response_path)["accepted"] is True
    assert consume_advisor_response(workspace, request["request_id"], state).status == "waiting_for_user"

    assert (
        main(
            [
                "agent",
                "reject",
                "--project",
                str(project),
                "--version-id",
                VERSION_ID,
                "--request-id",
                request["request_id"],
                "--reason",
                "not safe enough",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "advisor_rejection: accepted" in output
    assert load_agent_state(workspace)["status"] == "stopped"
    assert load_advisor_request(workspace, request["request_id"])["consumption_receipt"].endswith(".rejection.json")


def test_non_numeric_round_identity_is_rejected_without_crashing(tmp_path):
    _project, workspace, request, _state = _paused_request(tmp_path)
    response = _response_payload(request, decision="continue")
    response["request_identity"]["round"] = "not-a-number"

    errors = validate_advisor_response(workspace, response)

    assert any("response identity mismatch: round" in error for error in errors)


def _paused_request(tmp_path: Path):
    project = _make_project(tmp_path)
    workspace = _init_version(project)
    _write_context_files(workspace)
    task = _task("train_main")
    task["attempt_id"] = "attempt_train_main_1"
    task["invocation_hash"] = "invocation_train_main_1"
    plan = {
        "version": 2,
        "plan_id": "agent_plan",
        "plan_hash": "plan_hash",
        "registry_digest": "registry_digest",
        "version_id": VERSION_ID,
        "project": str(project),
        "workflow": "train_baseline",
        "tasks": [task],
    }
    save_agent_plan(workspace, plan)
    state = init_agent_state(workspace, project=str(project), version_id=VERSION_ID, agent_plan=plan)
    state["status"] = "running"
    state["tasks"][0]["attempt_id"] = task["attempt_id"]
    state["tasks"][0]["invocation_hash"] = task["invocation_hash"]
    save_agent_state(workspace, state)
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=VERSION_ID,
        task=task,
        reason="advisor_required",
        message="host agent plan required",
        attempt_id=task["attempt_id"],
        invocation_hash=task["invocation_hash"],
        round_index=1,
    )
    state = pause_agent(
        workspace,
        status="waiting_for_advisor",
        reason="advisor_required",
        task_id=task["task_id"],
        advisor_request=str(request["path"]),
        advisor_request_id=request["request_id"],
    )
    return project, workspace, request, state


def _write_context_files(workspace: Path) -> None:
    (workspace / "modeling" / "main_lgbm").mkdir(parents=True, exist_ok=True)
    (workspace / "configs_runtime").mkdir(parents=True, exist_ok=True)
    (workspace / "feature_selection").mkdir(parents=True, exist_ok=True)
    (workspace / "modeling" / "main_lgbm" / "tuning_context_round_1.json").write_text(
        json.dumps({"round": 1, "experiment": "main_lgbm"}),
        encoding="utf-8",
    )
    (workspace / "configs_runtime" / "train.yaml").write_text("training: {}\n", encoding="utf-8")
    (workspace / "feature_selection" / "final_features.txt").write_text("x1\n", encoding="utf-8")


def _write_response(
    workspace: Path,
    request: dict,
    *,
    status: str = "answered",
    decision: str,
    output_files: list[str] | None = None,
) -> Path:
    if output_files is None:
        plan_path = workspace / "modeling" / "main_lgbm" / "llm_tuning_plan_round_1.json"
        plan_path.write_text(
            json.dumps(
                {
                    "experiment": "main_lgbm",
                    "round": 1,
                    "diagnosis": "bounded plan",
                    "candidates": [
                        {
                            "name": "regularized_capacity",
                            "params": {"learning_rate": 0.03, "num_leaves": 63, "max_depth": 7},
                            "reason": "bounded candidate",
                        }
                    ],
                    "stop": False,
                }
            ),
            encoding="utf-8",
        )
        output_files = ["modeling/main_lgbm/llm_tuning_plan_round_1.json"]
    response_path = workspace / f"{request['request_id']}.response.source.json"
    response_path.write_text(
        json.dumps(
            _response_payload(request, status=status, decision=decision, output_files=output_files),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return response_path


def _response_payload(
    request: dict,
    *,
    status: str = "answered",
    decision: str = "continue",
    output_files: list[str] | None = None,
) -> dict:
    return {
        "version": 1,
        "request_id": request["request_id"],
        "type": request["expected_response"]["type"],
        "status": status,
        "decision": decision,
        "summary": "Advisor response.",
        "request_identity": {
            "task_id": request["task_id"],
            "attempt_id": request["attempt_id"],
            "invocation_hash": request["invocation_hash"],
            "context_hash": request["context_hash"],
            "round": request["round"],
        },
        "output_files": output_files or [],
        "risk_notes": [],
        "requires_user_confirmation": decision == "needs_user_confirmation",
    }


def _clone_new_request(workspace: Path, request: dict, *, suffix: str) -> dict:
    cloned = dict(request)
    cloned["request_id"] = f"{request['request_id']}_{suffix}"
    cloned["status"] = "pending"
    cloned["accepted_response"] = ""
    cloned["answered_at"] = ""
    cloned["path"] = str(Path("audit") / "advisor_requests" / f"{cloned['request_id']}.json")
    _save_request(workspace, cloned)
    return cloned


def _save_request(workspace: Path, request: dict) -> None:
    path = workspace / "audit" / "advisor_requests" / f"{request['request_id']}.json"
    path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _task(task_id: str) -> dict:
    return {
        "task_id": task_id,
        "type": "train",
        "status": "pending",
        "workspace": f"tasks/{task_id}",
        "depends_on": [],
        "command": {
            "executable": "rmw",
            "args": ["train", "--project", "p", "--version-id", VERSION_ID, "--experiment", "main_lgbm"],
        },
        "outputs": ["modeling/main_lgbm/tuning_summary.json"],
        "action_id": "train_baseline",
        "tool_name": "train_baseline",
        "permission": "writes_run",
        "requires_approval": False,
    }


def _init_version(project: Path) -> Path:
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", VERSION_ID]) == 0
    return project / "versions" / VERSION_ID


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
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
