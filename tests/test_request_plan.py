import re
from pathlib import Path

from risk_model_workbench.cli import main
from risk_model_workbench.planning.steps import KNOWN_STAGES, STEP_REGISTRY
from risk_model_workbench.planning import create_execution_plan
from risk_model_workbench.request import parse_model_request, validate_model_request


def _request_doc(**overrides):
    metadata = {
        "request_id": "pytest-request",
        "project": "pytest-project",
        "workflow": "full_modeling",
        "target_column": "target",
        "id_columns": ["uid", "sample_date"],
        "split_column": "final_flag",
        "experiments": [{"name": "baseline_all"}],
        "evaluation": {"metrics": ["auc", "ks"], "champions": []},
        "reports": {"outputs": ["model_report.md"]},
    }
    metadata.update(overrides)
    return {"path": "/tmp/pytest-request.md", "metadata": metadata, "body": ""}


def test_parse_and_validate_gcard_request_template():
    request_path = Path("projects/2026-05-fujie-gcard-v1/requests/model_request_template.md")
    request_doc = parse_model_request(request_path)
    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))
    assert result["status"] == "ok"
    assert request_doc["metadata"]["request_id"] == "2026-06-fujie-gcard-baseline"


def test_create_execution_plan_from_request():
    request_path = Path("projects/2026-05-fujie-gcard-v1/requests/model_request_template.md")
    request_doc = parse_model_request(request_path)
    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]
    assert "sample_check_profile" in task_ids
    assert "feature_prescreen" in task_ids
    assert "build_wide_sql" in task_ids
    assert "train_baseline_all" in task_ids
    assert task_ids[-1] == "report_final"
    assert task_ids.index("feature_prescreen") < task_ids.index("build_wide_sql") < task_ids.index("feature_refine")
    feature_refine = next(task for task in plan["tasks"] if task["task_id"] == "feature_refine")
    assert feature_refine["depends_on"] == ["build_wide_sql"]
    assert plan["scenario_profile"] == "fujie_gcard_main_lgbm"
    assert "sql_review_gate" in plan["stage_steps"]["build_wide_sql"]
    assert "feature_availability_filter" in plan["stage_steps"]["feature_refine"]
    assert "constant_value_filter" in plan["stage_steps"]["feature_refine"]
    assert "random_noise_importance" in plan["stage_steps"]["feature_refine"]
    assert "null_importance_filter" in plan["stage_steps"]["feature_refine"]
    assert "baseline_importance_filter" in plan["stage_steps"]["feature_refine"]
    assert "llm_guided_tuning" in plan["stage_steps"]["train_baseline"]
    train_task = next(task for task in plan["tasks"] if task["task_id"] == "train_baseline_all")
    assert "llm_guided_tuning" in train_task["step_ids"]
    assert "modeling/baseline_all/tuning_summary.json" in train_task["outputs"]
    assert plan["step_params"]["constant_value_filter"]["max_unique_values"] == 1
    assert "hier_ranknet_training" not in {step for steps in plan["stage_steps"].values() for step in steps}
    assert not plan["planned_steps"]
    report_task = next(task for task in plan["tasks"] if task["task_id"] == "report_final")
    assert "reports/model_report.html" in report_task["outputs"]


def test_refine_only_feature_rounds_do_not_force_build_wide_sql():
    request_doc = _request_doc(feature_selection={"rounds": ["refine"]})

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]
    feature_refine = next(task for task in plan["tasks"] if task["task_id"] == "feature_refine")

    assert "build_wide_sql" not in task_ids
    assert feature_refine["depends_on"] == ["sample_check_001"]


def test_profile_defaults_are_resolved_into_plan_metadata():
    request_doc = _request_doc(scenario_profile="inloan_behavior_card")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    sample_task = next(task for task in plan["tasks"] if task["type"] == "sample_check")

    assert plan["scenario_profile"] == "inloan_behavior_card"
    assert "account_status_distribution" in plan["stage_steps"]["sample_check"]
    assert "field_contract" in sample_task["step_ids"]
    assert plan["step_params"]["psi_filter"]["max_psi"] == 0.25


def test_business_domain_defaults_are_resolved_into_profile_metadata():
    acquisition_plan = create_execution_plan(_request_doc(business_domain="acquisition"), "projects/2026-05-fujie-gcard-v1")
    operation_plan = create_execution_plan(
        _request_doc(business_domain="inloan_operation"),
        "projects/2026-05-fujie-gcard-v1",
    )

    assert acquisition_plan["scenario_profile"] == "acquisition"
    assert "channel_distribution" in acquisition_plan["stage_steps"]["sample_check"]
    assert "hier_ranknet_training" in acquisition_plan["stage_steps"]["train_baseline"]
    assert not acquisition_plan["planned_steps"]
    assert operation_plan["scenario_profile"] == "inloan_operation"
    assert "segment_metrics" in operation_plan["stage_steps"]["evaluate"]
    assert "fujie_gcard_main_lgbm" != operation_plan["scenario_profile"]


def test_request_stage_steps_and_step_params_override_profile_defaults():
    request_doc = _request_doc(
        scenario_profile="inloan_behavior_card",
        stage_steps={"sample_check": ["field_contract"]},
        step_params={"psi_filter": {"max_psi": 0.3}},
    )

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    sample_task = next(task for task in plan["tasks"] if task["type"] == "sample_check")

    assert plan["stage_steps"]["sample_check"] == ["field_contract"]
    assert sample_task["step_ids"] == ["field_contract"]
    assert plan["step_params"]["psi_filter"]["max_psi"] == 0.3


def test_unknown_profile_and_step_fail_validation():
    unknown_profile = validate_model_request(_request_doc(scenario_profile="unknown_profile"))
    unknown_domain = validate_model_request(_request_doc(business_domain="unknown_domain"))
    unknown_step = validate_model_request(_request_doc(stage_steps={"sample_check": ["unknown_step"]}))
    unknown_workflow = validate_model_request(_request_doc(workflow="unknown_workflow"))

    assert unknown_profile["status"] == "failed"
    assert "unknown scenario_profile" in unknown_profile["errors"][0]
    assert unknown_domain["status"] == "failed"
    assert "unknown business_domain" in unknown_domain["errors"][0]
    assert unknown_step["status"] == "failed"
    assert "unknown step id" in unknown_step["errors"][0]
    assert unknown_workflow["status"] == "failed"
    assert "unknown workflow" in unknown_workflow["errors"][0]


def test_custom_training_requires_project_entrypoint():
    request_doc = _request_doc(experiments=[{"name": "custom_model", "method": "custom"}])

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "failed"
    assert any("custom training requires" in error for error in result["errors"])


def test_training_mode_validation_accepts_llm_guided_tune():
    result = validate_model_request(
        _request_doc(training={"mode": "llm_guided_tune", "tuning": {"max_rounds": 2, "candidates_per_round": 3}})
    )

    assert result["status"] == "ok"


def test_training_mode_validation_rejects_unknown_mode():
    result = validate_model_request(_request_doc(training={"mode": "unknown_tune"}))

    assert result["status"] == "failed"
    assert "unsupported training.mode" in result["errors"][0]


def test_project_default_tuning_can_be_disabled_with_reason():
    request_doc = _request_doc(training={"mode": "single_train"})
    missing_reason = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert missing_reason["status"] == "failed"
    assert "training.disable_tuning_reason is required" in missing_reason["errors"][0]

    request_doc["metadata"]["training"]["disable_tuning_reason"] = "本次只验证数据口径，不做参数搜索"
    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))
    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")

    assert result["status"] == "ok"
    assert "llm_guided_tuning" not in plan["stage_steps"]["train_baseline"]


def test_builder_visible_steps_have_executor_task_binding():
    request_doc = _request_doc(scenario_profile="acquisition_conversion")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    planned_ids = {step["id"] for step in plan["planned_steps"]}
    train_task = next(task for task in plan["tasks"] if task["type"] == "train")

    assert "hier_ranknet_training" not in planned_ids
    assert "hier_ranknet_training" in train_task["step_ids"]


def test_request_builder_stage_and_step_ids_are_known_to_planner():
    app_js = Path("tools/model_request_builder/app.js").read_text(encoding="utf-8")
    stage_ids = set(re.findall(r'\{\s*stage:\s*"([^"]+)"', app_js))
    step_label_block = re.search(r"const STEP_LABELS = \{(?P<body>.*?)\n\};", app_js, flags=re.S)
    assert step_label_block is not None
    step_ids = set(re.findall(r"^\s*([a-zA-Z0-9_]+):", step_label_block.group("body"), flags=re.M))

    assert stage_ids <= KNOWN_STAGES
    assert step_ids <= set(STEP_REGISTRY)


def test_request_builder_exports_data_source_mode_contract():
    html = Path("tools/model_request_builder/index.html").read_text(encoding="utf-8")
    app_js = Path("tools/model_request_builder/app.js").read_text(encoding="utf-8")

    assert 'name="data_source_mode"' in html
    assert 'value="remote_table"' in html
    assert 'value="local_feather"' in html
    assert 'data_source_mode: "remote_table"' in app_js
    assert "`data_source_mode: ${yamlScalar(state.data_source_mode)}`" in app_js


def test_request_builder_workflows_are_known_and_limit_planned_tasks():
    html = Path("tools/model_request_builder/index.html").read_text(encoding="utf-8")
    workflow_select = re.search(r'<select name="workflow">(?P<body>.*?)</select>', html, flags=re.S)
    assert workflow_select is not None
    workflow_options = set(re.findall(r'<option value="([^"]+)"', workflow_select.group("body")))

    expected = {"full_modeling", "feature_selection", "train_baseline", "challenger_evaluation"}
    assert workflow_options == expected

    feature_plan = create_execution_plan(
        _request_doc(
            workflow="feature_selection",
            scenario_profile="fujie_gcard_main_lgbm",
            feature_selection={"rounds": ["metadata", "prescreen", "refine"]},
        ),
        "projects/2026-05-fujie-gcard-v1",
    )
    train_plan = create_execution_plan(_request_doc(workflow="train_baseline"), "projects/2026-05-fujie-gcard-v1")
    challenger_plan = create_execution_plan(_request_doc(workflow="challenger_evaluation"), "projects/2026-05-fujie-gcard-v1")

    assert [task["task_id"] for task in feature_plan["tasks"]] == [
        "feature_metadata",
        "feature_prescreen",
        "build_wide_sql",
        "feature_refine",
    ]
    assert set(feature_plan["stage_steps"]) == {"feature_metadata", "feature_prescreen", "build_wide_sql", "feature_refine"}
    assert {task["type"] for task in train_plan["tasks"]} == {"train"}
    assert set(train_plan["stage_steps"]) == {"train_baseline"}
    assert [task["task_id"] for task in challenger_plan["tasks"]] == ["evaluate_final", "compare_final"]
    assert set(challenger_plan["stage_steps"]) == {"evaluate", "compare"}


def test_request_without_id_columns_can_use_project_contract():
    request_doc = _request_doc(id_columns=[])

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "ok"
    assert "request id_columns omitted; using project.yml data.id_columns" in result["warnings"]


def test_request_without_id_columns_fails_without_project_contract():
    request_doc = _request_doc(id_columns=[])

    result = validate_model_request(request_doc)

    assert result["status"] == "failed"
    assert "missing required field: id_columns" in result["errors"][-1]


def test_validate_explicit_local_feather_source_mode():
    request_doc = _request_doc(data_source_mode="local_feather", sample_location="data/raw/model.feather")

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "ok"


def test_validate_rejects_local_feather_mode_without_feather_path():
    request_doc = _request_doc(data_source_mode="local_feather", sample_location="mart.sample_table")

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "failed"
    assert "local_feather data_source_mode requires sample_location ending with .feather" in result["errors"]


def test_local_feather_defaults_to_refine_only_without_remote_feature_source():
    request_doc = _request_doc(data_source_mode="local_feather", sample_location="data/raw/model.feather")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]

    assert "feature_metadata" not in task_ids
    assert "feature_prescreen" not in task_ids
    assert "build_wide_sql" not in task_ids
    assert "feature_refine" in task_ids


def test_task_mode_label_is_plan_compatible():
    request_doc = _request_doc(task_mode="完整建模")

    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")

    assert plan["workflow"] == "full_modeling"
    assert plan["request_id"] == "pytest-request"


def test_experiment_description_derives_baseline_experiment():
    request_doc = _request_doc(experiments=[], experiment_description="先跑一个全客群 baseline。")

    validation = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))
    plan = create_execution_plan(request_doc, "projects/2026-05-fujie-gcard-v1")
    task_ids = [task["task_id"] for task in plan["tasks"]]

    assert validation["status"] == "ok"
    assert "train_baseline_from_description" in task_ids


def test_report_targets_are_validated():
    request_doc = _request_doc(
        reports={
            "outputs": ["model_report.md"],
            "score_labels": {"model_score": "G卡V8"},
            "targets": [
                {
                    "name": "tuned_main",
                    "experiment": "main_lgbm_tuned",
                    "train_dir": "modeling/main_lgbm_tuned",
                    "eval_dir": "evaluation_tuned/main_lgbm_tuned",
                    "output_dir": "reports_tuned_main_lgbm_tuned",
                }
            ],
        }
    )

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "ok"


def test_report_targets_reject_bad_shape():
    request_doc = _request_doc(reports={"outputs": ["model_report.md"], "targets": [{"eval_dir": 123}]})

    result = validate_model_request(request_doc, Path("projects/2026-05-fujie-gcard-v1"))

    assert result["status"] == "failed"
    assert "reports.targets[1].name is required" in result["errors"]
    assert "reports.targets[1].eval_dir must be a string" in result["errors"]


def test_cli_request_validate_and_plan_create(tmp_path):
    output = tmp_path / "execution_plan.yml"
    assert main(
        [
            "request",
            "validate",
            "--project",
            "projects/2026-05-fujie-gcard-v1",
            "--request",
            "projects/2026-05-fujie-gcard-v1/requests/model_request_template.md",
        ]
    ) == 0
    assert main(
        [
            "plan",
            "create",
            "--project",
            "projects/2026-05-fujie-gcard-v1",
            "--request",
            "projects/2026-05-fujie-gcard-v1/requests/model_request_template.md",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.exists()
