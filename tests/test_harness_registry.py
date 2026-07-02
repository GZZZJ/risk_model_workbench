import json

from risk_model_workbench.cli import main
from risk_model_workbench.config import load_yaml
from risk_model_workbench.harness.actions import get_action_spec, list_action_specs
from risk_model_workbench.harness.errors import (
    SQL_APPROVAL_REQUIRED,
    TRANSIENT_IO,
    UNKNOWN,
    get_failure_class,
)
from risk_model_workbench.harness.tools import get_tool_spec, list_tool_specs, validate_tool_registry
from risk_model_workbench.paths import workflow_path


def test_action_registry_covers_full_modeling_stages():
    workflow = load_yaml(workflow_path("full_modeling"))
    stage_actions = {spec.stage for spec in list_action_specs(kind="stage")}

    assert set(workflow["stages"]).issubset(stage_actions)
    assert get_action_spec("feature_prescreen").approval_required is True
    assert get_action_spec("feature_prescreen").approval_type == "sql_review"
    assert SQL_APPROVAL_REQUIRED in get_action_spec("feature_prescreen").failure_codes
    assert get_action_spec("feature_refine").approval_required is True
    assert get_action_spec("train_baseline").mutates_manifest is True
    assert "modeling/*/tuning_summary.json" in get_action_spec("train_baseline").artifact_rules
    assert "reports/model_report.html" in get_action_spec("report").outputs
    assert "reports/model_report.html" in get_action_spec("report").artifact_rules


def test_tool_registry_declares_permissions_and_auditor_boundary():
    assert validate_tool_registry() == []

    dp_tool = get_tool_spec("feature_prescreen_pull")
    assert dp_tool.permission == "dp_sql_pull"
    assert dp_tool.requires_approval is True
    assert dp_tool.allowed_for_auditor is False
    wide_execute_tool = get_tool_spec("build_wide_sql_execute")
    assert wide_execute_tool.permission == "dp_sql_pull"
    assert wide_execute_tool.requires_approval is True

    audit_tool = get_tool_spec("run_audit")
    assert audit_tool.permission == "read_only"
    assert audit_tool.allowed_for_auditor is True

    for tool in list_tool_specs():
        get_action_spec(tool.action_id)


def test_failure_classes_keep_retry_boundary():
    assert get_failure_class(TRANSIENT_IO).retryable is True
    assert get_failure_class(SQL_APPROVAL_REQUIRED).retryable is False
    assert get_failure_class("not_registered").code == UNKNOWN


def test_action_and_tool_cli_json(capsys):
    capsys.readouterr()
    assert main(["action", "list", "--json"]) == 0
    actions = json.loads(capsys.readouterr().out)
    assert any(item["id"] == "sample_check" for item in actions)
    assert any(item["id"] == "feature_prescreen" and item["approval_type"] == "sql_review" for item in actions)

    assert main(["action", "show", "feature_refine", "--json"]) == 0
    action = json.loads(capsys.readouterr().out)
    assert action["approval_required"] is True
    assert action["mutates_manifest"] is True

    assert main(["tool", "list", "--permission", "read_only", "--json"]) == 0
    tools = json.loads(capsys.readouterr().out)
    assert all(item["permission"] == "read_only" for item in tools)
    assert any(item["name"] == "run_audit" and item["allowed_for_auditor"] is True for item in tools)

    assert main(["tool", "show", "feature_prescreen_pull", "--json"]) == 0
    tool = json.loads(capsys.readouterr().out)
    assert tool["permission"] == "dp_sql_pull"
    assert tool["requires_approval"] is True


def test_action_and_tool_cli_unknown_ids(capsys):
    assert main(["action", "show", "missing_action"]) == 1
    assert "unknown action: missing_action" in capsys.readouterr().out

    assert main(["tool", "show", "missing_tool"]) == 1
    assert "unknown tool: missing_tool" in capsys.readouterr().out
