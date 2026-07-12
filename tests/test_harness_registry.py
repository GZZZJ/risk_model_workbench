import hashlib
import inspect
import json
from dataclasses import fields, replace

from risk_model_workbench.cli import main
from risk_model_workbench.config import load_yaml
from risk_model_workbench.harness.actions import (
    ACTION_ALIASES,
    ActionSpec,
    format_action_detail,
    format_action_list,
    get_action_spec,
    list_action_specs,
)
from risk_model_workbench.harness.errors import (
    SQL_APPROVAL_REQUIRED,
    TRANSIENT_IO,
    UNKNOWN,
    get_failure_class,
)
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import (
    TOOL_ALIASES,
    ToolSpec,
    format_tool_detail,
    format_tool_list,
    get_tool_spec,
    list_tool_specs,
    registry_digest,
    validate_tool_registry,
)
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


def test_action_and_tool_public_behavior_snapshot():
    """Freeze the public registry shape while command metadata is deduplicated."""
    invocation_params = {
        "workflow_validate": {"workflow": "full_modeling"},
        "train_baseline": {
            "experiment": "main",
            "input_feather": "data.feather",
            "feature_list": "features.txt",
            "score_output": "score.feather",
            "input_dir": "inputs",
            "config": "train.yml",
            "plan_only": True,
        },
        "compare": {"champions": ["legacy_a", "legacy_b"]},
        "evaluate": {"scores_feather": "scores.feather", "output_dir": "evaluation"},
        "report": {"report_target": "reports/custom.md"},
    }
    actions = list_action_specs()
    tools = list_tool_specs()
    snapshot = {
        "action_signature": str(inspect.signature(ActionSpec)),
        "action_fields": [item.name for item in fields(ActionSpec)],
        "actions": [item.to_dict() for item in actions],
        "action_repr": [repr(item) for item in actions],
        "action_list": format_action_list(actions),
        "action_details": {item.id: format_action_detail(item) for item in actions},
        "action_aliases": {name: get_action_spec(name).id for name in ACTION_ALIASES},
        "action_equal_replace": [replace(item) == item for item in actions],
        "tool_signature": str(inspect.signature(ToolSpec)),
        "tool_fields": [item.name for item in fields(ToolSpec)],
        "tools": [item.to_dict() for item in tools],
        "tool_repr": [repr(item) for item in tools],
        "tool_list": format_tool_list(tools),
        "tool_details": {item.name: format_tool_detail(item) for item in tools},
        "tool_aliases": {name: get_tool_spec(name).name for name in TOOL_ALIASES},
        "tool_equal_replace": [replace(item) == item for item in tools],
        "tool_argv": {},
        "registry_digest": registry_digest(),
    }
    for item in tools:
        invocation = ActionInvocation(
            tool_name=item.name,
            params=invocation_params.get(item.name, {}),
            project="/project",
            version_id="version_1",
        )
        snapshot["tool_argv"][item.name] = item.render_argv(invocation)

    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == (
        "ecb9ba355a9f8c44724f9cc2ba85e1002760cbd714f2089147698aa339391a50"
    )


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


def test_command_metadata_is_the_single_source_for_actions_and_tools():
    from risk_model_workbench.harness.command_metadata import COMMAND_DECLARATIONS

    declarations = COMMAND_DECLARATIONS
    for action in list_action_specs():
        assert declarations[action.id].display_template == action.command
    for tool in list_tool_specs():
        assert declarations[tool.name].display_template == tool.command
        assert declarations[tool.name].argv_template


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


def test_workflow_list_excludes_internal_contract_registry(capsys):
    assert main(["workflow", "list"]) == 0
    assert capsys.readouterr().out == (
        "challenger_evaluation\n"
        "feature_selection\n"
        "full_modeling\n"
        "ranking_optimization\n"
        "report_generation\n"
        "sample_audit\n"
        "train_baseline\n"
    )
