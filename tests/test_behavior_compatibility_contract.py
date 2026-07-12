from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

import risk_model_workbench
from risk_model_workbench.cli import main
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.state import create_run_state, load_run_state, save_run_state


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO_ROOT = Path(risk_model_workbench.__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "compatibility"
MUTATION_FIXTURE = FIXTURE_ROOT / "write_command_baseline.json"
READ_ONLY_FIXTURE = FIXTURE_ROOT / "read_only_cli_baseline.json"
VERSION_ID = "compat_model_v1_20260712"
RUN_ID = "compat_legacy_run"
REQUEST_ID = "compat-request"
REQUEST_VERSION_ID = "compat_request_v1_20260712"
_TIMESTAMP = re.compile(r"20\d\d-\d\d-\d\d(?:T| )[0-9:.+-]+")
_ACTION_RESULT_ID = re.compile(r"(cli_[a-z0-9_]+_)[0-9a-f]{32}(\.json)")
_TRANSACTION_ID = re.compile(r"txn_[0-9a-f]{32}")
_INSTALLED_LEGACY_CONTRACT_SCRIPT = r"""
import dataclasses
import importlib
import inspect
import json
import re
import sys

contract = json.load(open(sys.argv[1], encoding="utf-8"))


def normalized_signature(value):
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<unavailable>"
    return re.sub(r" at 0x[0-9a-fA-F]+", " at <ADDRESS>", signature)


for module_name, expected_rows in contract["legacy_public_api"].items():
    module = importlib.import_module(module_name)
    for expected in expected_rows:
        value = getattr(module, expected["name"])
        assert type(value).__name__ == expected["kind"], (
            module_name,
            expected["name"],
            type(value).__name__,
            expected["kind"],
        )
        if "signature" in expected:
            assert normalized_signature(value) == expected["signature"], (
                module_name,
                expected["name"],
                normalized_signature(value),
                expected["signature"],
            )

legacy_actions = importlib.import_module("jingying_model_agent.harness.actions")
legacy_tools = importlib.import_module("jingying_model_agent.harness.tools")
ActionSpec = legacy_actions.ActionSpec
ToolSpec = legacy_tools.ToolSpec
assert normalized_signature(ActionSpec) == contract["spec_contract"]["ActionSpec"]["signature"]
assert normalized_signature(ToolSpec) == contract["spec_contract"]["ToolSpec"]["signature"]
assert [field.name for field in dataclasses.fields(ActionSpec)] == [
    field["name"] for field in contract["spec_contract"]["ActionSpec"]["fields"]
]
assert [field.name for field in dataclasses.fields(ToolSpec)] == [
    field["name"] for field in contract["spec_contract"]["ToolSpec"]["fields"]
]
assert ActionSpec(
    id="compat",
    command="rmw doctor",
    description="compat",
    kind="utility",
).to_dict()["id"] == "compat"
assert ToolSpec(
    name="compat",
    action_id="compat",
    command="rmw doctor",
    permission="read_only",
    description="compat",
    render_argv=lambda invocation: ["doctor"],
).to_dict()["name"] == "compat"
print(
    json.dumps(
        {
            "legacy_module_count": len(contract["legacy_public_api"]),
            "legacy_ActionSpec": ActionSpec.__name__,
            "legacy_ToolSpec": ToolSpec.__name__,
        },
        sort_keys=True,
    )
)
"""


def _normalize_text(value: str, *, project: Path | None = None) -> str:
    normalized = value.replace(str(REPO_ROOT), "<REPO_ROOT>")
    normalized = normalized.replace(str(SOURCE_REPO_ROOT), "<REPO_ROOT>")
    if project is not None:
        normalized = normalized.replace(str(project.resolve()), "<PROJECT>")
        normalized = normalized.replace(str(project), "<PROJECT>")
    normalized = _TIMESTAMP.sub("<TIMESTAMP>", normalized)
    normalized = _ACTION_RESULT_ID.sub(r"\1<ID>\2", normalized)
    normalized = _TRANSACTION_ID.sub("txn_<ID>", normalized)
    return normalized


def _normalize_path(value: str) -> str:
    return _ACTION_RESULT_ID.sub(r"\1<ID>\2", value)


def _project_config() -> dict[str, Any]:
    return {
        "project": {
            "name": "compat_project",
            "display_name": "Compatibility Project",
            "project_key": "compat_model",
        },
        "data": {
            "source_table": "demo.sample",
            "raw_path": "data/sample.feather",
            "id_columns": ["uid"],
            "target_column": "label",
            "time_column": "event_time",
            "period_column": "split",
            "split_column": "split",
        },
        "segments": [{"name": "all", "display_name": "All", "filter": None}],
    }


def _request_metadata() -> dict[str, Any]:
    return {
        "request_id": REQUEST_ID,
        "project": "compat_project",
        "workflow": "full_modeling",
        "data_source_mode": "local_feather",
        "sample_location": "data/sample.feather",
        "target_column": "label",
        "id_columns": ["uid"],
        "time_column": "event_time",
        "period_column": "split",
        "split_column": "split",
        "splits": {
            "dev": {"values": ["DEV"]},
            "oot": {"values": ["OOT"]},
        },
        "feature_selection": {"rounds": ["refine"]},
        "experiments": [{"name": "baseline_all", "method": "lightgbm", "segment": "all"}],
        "evaluation": {"metrics": ["auc", "ks"], "champions": []},
        "reports": {"outputs": ["model_report.md"]},
    }


def _make_project(base: Path) -> Path:
    project = base / "project"
    for directory in ["configs", "data", "queries", "reports", "runs", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    project_config = _project_config()
    (project / "project.yml").write_text(
        yaml.safe_dump(project_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (project / "configs" / "project.yml").write_text(
        yaml.safe_dump(project_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (project / "configs" / "feature_select.yaml").write_text(
        yaml.safe_dump(
            {
                "feature_select": {
                    "runtime_request": {
                        "data_source_mode": "local_feather",
                        "sample_location": "data/sample.feather",
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    for name, payload in {
        "refine_features.yaml": {"feature_refine": {}},
        "train.yaml": {"training": {"default_algorithm": "lightgbm"}, "input": {}},
        "evaluate.yaml": {"evaluation": {"score_columns": ["model_score"]}},
        "report.yaml": {"report": {"outputs": ["model_report.md"]}},
    }.items():
        (project / "configs" / name).write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
    pd.DataFrame(
        {
            "uid": [1, 2, 2, 3],
            "label": [0, 1, 1, 0],
            "split": ["DEV", "DEV", "OOT", "OOT"],
            "event_time": ["2026-01-01", "2026-01-02", "2026-02-01", "2026-02-02"],
            "feature_x": [0.1, 0.2, 0.3, 0.4],
        }
    ).to_feather(project / "data" / "sample.feather")
    return project


def _write_request(project: Path) -> Path:
    request_path = project / "requests" / f"{REQUEST_ID}.md"
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        "---\n"
        + yaml.safe_dump(_request_metadata(), allow_unicode=True, sort_keys=False)
        + "---\n",
        encoding="utf-8",
    )
    return request_path


def _make_run_workspace(project: Path) -> Path:
    workspace = project / "runs" / RUN_ID
    (workspace / "audit").mkdir(parents=True)
    (workspace / "configs_runtime").mkdir()
    save_run_state(
        workspace,
        create_run_state(project, run_id=RUN_ID, workflow="full_modeling"),
    )
    (workspace / "audit" / "artifact_manifest.json").write_text(
        json.dumps({"version": 2, "artifacts": []}),
        encoding="utf-8",
    )
    for name in ["project.yml", "feature_select.yaml"]:
        source = project / "configs" / name
        (workspace / "configs_runtime" / name).write_text(
            source.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    return workspace


def _capture(argv: list[str]) -> dict[str, Any]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code = main(argv)
    return {"exit_code": exit_code, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def _write_command_snapshot(base: Path) -> dict[str, Any]:
    project = _make_project(base)
    request_path = _write_request(project)
    plan_path = project / "requests" / f"{REQUEST_ID}.execution_plan.yml"
    workspace = _make_run_workspace(project)
    commands = [
        [
            "request",
            "validate",
            "--request",
            str(request_path),
            "--project",
            str(project),
        ],
        [
            "plan",
            "create",
            "--project",
            str(project),
            "--request",
            str(request_path),
            "--output",
            str(plan_path),
        ],
        [
            "version",
            "init",
            "--project",
            str(project),
            "--workflow",
            "full_modeling",
            "--version-id",
            REQUEST_VERSION_ID,
            "--request",
            str(request_path),
            "--plan",
            str(plan_path),
        ],
        ["sample", "check", "--project", str(project), "--run-id", RUN_ID],
        ["feature", "metadata", "--project", str(project), "--run-id", RUN_ID],
        ["feature", "prescreen", "--project", str(project), "--run-id", RUN_ID],
        ["build-wide-sql", "--project", str(project), "--run-id", RUN_ID],
    ]
    results = [_capture(argv) for argv in commands]
    state = load_run_state(workspace)
    manifest = load_artifact_manifest(workspace)
    stages = ["sample_check", "feature_metadata", "feature_prescreen", "build_wide_sql"]
    manifest_rows = []
    for row in manifest.get("artifacts", []):
        if row.get("stage") not in {*stages, "agent_runtime"}:
            continue
        manifest_rows.append(
            {
                key: _normalize_path(str(row.get(key, "")))
                for key in [
                    "path",
                    "stage",
                    "kind",
                    "source",
                    "contract_role",
                    "storage_class",
                    "retention_reason",
                    "integrity_mode",
                    "description",
                ]
            }
        )
    artifact_files = sorted(
        _normalize_path(path.relative_to(workspace).as_posix())
        for root in ["sample_check", "feature_metadata", "feature_selection", "queries"]
        for path in (workspace / root).rglob("*")
        if path.is_file()
    )
    version_workspace = project / "versions" / REQUEST_VERSION_ID
    version_state = load_run_state(version_workspace)
    version_manifest = load_artifact_manifest(version_workspace)
    version_manifest_rows = [
        {
            key: _normalize_path(str(row.get(key, "")))
            for key in [
                "path",
                "stage",
                "kind",
                "source",
                "contract_role",
                "storage_class",
                "retention_reason",
                "integrity_mode",
                "description",
            ]
        }
        for row in version_manifest.get("artifacts", [])
        if row.get("stage") == "validate_config"
    ]
    execution_plan = yaml.safe_load(
        (version_workspace / "execution_plan.yml").read_text(encoding="utf-8")
    )
    runtime_project = yaml.safe_load(
        (version_workspace / "configs_runtime" / "project.yml").read_text(encoding="utf-8")
    )
    version_index = yaml.safe_load(
        (project / "versions" / "index.yml").read_text(encoding="utf-8")
    )
    project_state = yaml.safe_load((project / "project_state.yml").read_text(encoding="utf-8"))
    return {
        "commands": [
            {
                "argv": [part.replace(str(project), "<PROJECT>") for part in argv],
                "exit_code": result["exit_code"],
                "stdout": _normalize_text(result["stdout"], project=project),
                "stderr": _normalize_text(result["stderr"], project=project),
            }
            for argv, result in zip(commands, results)
        ],
        "stage_state": {
            stage: {
                key: state["stages"][stage].get(key)
                for key in ["status", "scaffold", "failure_code", "artifacts"]
                if key in state["stages"][stage]
            }
            for stage in stages
        },
        "state_decisions": [
            {
                "stage": row.get("stage"),
                "decision": row.get("decision"),
                "reason": _normalize_text(str(row.get("reason", "")), project=project),
            }
            for row in state.get("decisions", [])
        ],
        "decision_log": _normalize_text(
            (workspace / "audit" / "decision_log.md").read_text(encoding="utf-8"),
            project=project,
        ),
        "manifest": manifest_rows,
        "artifact_files": artifact_files,
        "request_plan_version_flow": {
            "version_state": {
                key: version_state.get(key)
                for key in [
                    "version_id",
                    "workflow",
                    "source_type",
                    "managed_by",
                    "status",
                    "current_stage",
                ]
            },
            "validate_config_stage": {
                key: version_state["stages"]["validate_config"].get(key)
                for key in ["status", "scaffold", "failure_code", "artifacts"]
                if key in version_state["stages"]["validate_config"]
            },
            "bindings": {
                "model_request_matches": (version_workspace / "model_request.md").read_text(
                    encoding="utf-8"
                )
                == request_path.read_text(encoding="utf-8"),
                "execution_plan": {
                    "request_id": execution_plan.get("request_id"),
                    "workflow": execution_plan.get("workflow"),
                    "project": _normalize_text(
                        str(execution_plan.get("project", "")),
                        project=project,
                    ),
                },
                "task_ids": [row.get("task_id") for row in execution_plan.get("tasks", [])],
                "runtime_config_files": sorted(
                    path.name
                    for path in (version_workspace / "configs_runtime").iterdir()
                    if path.is_file()
                ),
                "runtime_request": runtime_project.get("request", {}),
            },
            "manifest": version_manifest_rows,
            "artifact_files": sorted(
                path.relative_to(version_workspace).as_posix()
                for root in ["configs_snapshot", "configs_runtime"]
                for path in (version_workspace / root).rglob("*")
                if path.is_file()
            )
            + ["execution_plan.yml", "model_request.md"],
            "version_index": {
                "active_version_id": version_index.get("active_version_id"),
                "versions": [
                    {
                        key: item.get(key)
                        for key in [
                            "version_id",
                            "source_type",
                            "managed_by",
                            "status",
                            "workflow",
                            "path",
                        ]
                    }
                    for item in version_index.get("versions", [])
                ],
            },
            "project_state": {
                key: project_state.get(key)
                for key in ["active_version_id", "active_run_id", "status"]
            },
        },
    }


def _read_only_snapshot(base: Path) -> dict[str, Any]:
    project = _make_project(base)
    setup_commands = [
        [
            "version",
            "init",
            "--project",
            str(project),
            "--workflow",
            "sample_audit",
            "--version-id",
            VERSION_ID,
        ],
        [
            "run",
            "init",
            "--project",
            str(project),
            "--workflow",
            "sample_audit",
            "--run-id",
            RUN_ID,
        ],
    ]
    assert all(_capture(argv)["exit_code"] == 0 for argv in setup_commands)
    invalid_project = base / "invalid_project"
    invalid_project.mkdir()
    cases = {
        "doctor": ["doctor"],
        "action_list": ["action", "list", "--json"],
        "tool_list": ["tool", "list", "--json"],
        "rules_list": ["rules", "list", "--json"],
        "project_status": ["project", "status", "--project", str(project)],
        "project_validate_failure": ["project", "validate", "--project", str(invalid_project)],
        "version_list": ["version", "list", "--project", str(project), "--json"],
        "version_status": [
            "version",
            "status",
            "--project",
            str(project),
            "--version-id",
            VERSION_ID,
        ],
        "version_audit": [
            "version",
            "audit",
            "--project",
            str(project),
            "--version-id",
            VERSION_ID,
            "--json",
        ],
        "run_status": ["run", "status", "--project", str(project), "--run-id", RUN_ID],
        "run_audit": [
            "run",
            "audit",
            "--project",
            str(project),
            "--run-id",
            RUN_ID,
            "--json",
        ],
        "workflow_validate": ["workflow", "validate", "--workflow", "full_modeling"],
        "workflow_list": ["workflow", "list"],
    }
    snapshot = {}
    for name, argv in cases.items():
        result = _capture(argv)
        normalized_stdout = _normalize_text(result["stdout"], project=project)
        normalized_stderr = _normalize_text(result["stderr"], project=project)
        for invalid_path in [invalid_project.resolve(), invalid_project]:
            normalized_stdout = normalized_stdout.replace(str(invalid_path), "<INVALID_PROJECT>")
            normalized_stderr = normalized_stderr.replace(str(invalid_path), "<INVALID_PROJECT>")
        snapshot[name] = {
            "argv": [
                part.replace(str(project), "<PROJECT>").replace(str(invalid_project), "<INVALID_PROJECT>")
                for part in argv
            ],
            "exit_code": result["exit_code"],
            "stdout": normalized_stdout,
            "stderr": normalized_stderr,
        }
    return snapshot


def test_representative_write_commands_match_frozen_mutation_baseline(tmp_path):
    expected = json.loads(MUTATION_FIXTURE.read_text(encoding="utf-8"))
    assert _write_command_snapshot(tmp_path) == expected


def test_read_only_commands_match_pre_refactor_output_baseline(tmp_path):
    expected = json.loads(READ_ONLY_FIXTURE.read_text(encoding="utf-8"))
    assert _read_only_snapshot(tmp_path) == expected


def _venv_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _clean_subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    return environment


def _repo_build_inventory() -> set[str]:
    build_dir = REPO_ROOT / "build"
    if not build_dir.exists():
        return set()
    return {path.relative_to(REPO_ROOT).as_posix() for path in build_dir.rglob("*")}


def _build_python() -> str:
    candidates = [sys.executable, shutil.which("python"), shutil.which("python3")]
    for candidate in dict.fromkeys(value for value in candidates if value):
        result = subprocess.run(
            [str(candidate), "-c", "import setuptools.build_meta"],
            check=False,
            capture_output=True,
            text=True,
            env=_clean_subprocess_env(),
        )
        if result.returncode == 0:
            return str(candidate)
    raise AssertionError("wheel/editable compatibility test requires a Python with setuptools.build_meta")


def _installed_surface(environment: Path) -> dict[str, Any]:
    python = _venv_python(environment)
    bin_dir = python.parent
    scripts = ["rmw", "jm", "jingying-agent"]
    help_outputs = {}
    for script in scripts:
        executable = bin_dir / (f"{script}.exe" if os.name == "nt" else script)
        result = subprocess.run(
            [str(executable), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=_clean_subprocess_env(),
        )
        assert result.returncode == 0, result.stderr
        help_outputs[script] = result.stdout.replace(script, "<ENTRYPOINT>", 1)
    action_list = subprocess.run(
        [str(bin_dir / ("rmw.exe" if os.name == "nt" else "rmw")), "action", "list", "--json"],
        cwd=environment,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_subprocess_env(),
    )
    assert action_list.returncode == 0, action_list.stderr
    legacy_contract = subprocess.run(
        [
            str(python),
            "-c",
            _INSTALLED_LEGACY_CONTRACT_SCRIPT,
            str(FIXTURE_ROOT / "cli_surface.json"),
        ],
        cwd=environment,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_subprocess_env(),
    )
    assert legacy_contract.returncode == 0, legacy_contract.stderr
    return {
        "help": help_outputs,
        "action_list": action_list.stdout,
        "legacy_contract": legacy_contract.stdout,
    }


def _repo_script_agent_py_surface(python: Path, cwd: Path) -> str:
    result = subprocess.run(
        [
            str(python),
            str(REPO_ROOT / "agent.py"),
            "action",
            "list",
            "--json",
        ],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_subprocess_env(),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _assert_isolated_install_contract(tmp_path: Path) -> None:
    build_before = _repo_build_inventory()
    build_python = _build_python()
    isolated_source = tmp_path / "isolated-source"
    isolated_source.mkdir()
    shutil.copy2(REPO_ROOT / "pyproject.toml", isolated_source / "pyproject.toml")
    shutil.copytree(REPO_ROOT / "src", isolated_source / "src")
    wheel_dir = tmp_path / "dist"
    wheel_dir.mkdir()
    try:
        wheel_build = subprocess.run(
            [
                build_python,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(wheel_dir),
                str(isolated_source),
            ],
            cwd=isolated_source,
            check=False,
            capture_output=True,
            text=True,
            env=_clean_subprocess_env(),
        )
        assert wheel_build.returncode == 0, wheel_build.stderr
        wheel = next(wheel_dir.glob("risk_model_workbench-*.whl"))

        surfaces = {}
        environments = {}
        for mode in ["editable", "wheel"]:
            environment = tmp_path / mode
            create_environment = subprocess.run(
                [build_python, "-m", "venv", "--system-site-packages", str(environment)],
                check=False,
                capture_output=True,
                text=True,
                env=_clean_subprocess_env(),
            )
            assert create_environment.returncode == 0, create_environment.stderr
            source = str(isolated_source) if mode == "editable" else str(wheel)
            install = subprocess.run(
                [
                    str(_venv_python(environment)),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--no-build-isolation",
                    "--force-reinstall",
                    *(["--editable"] if mode == "editable" else []),
                    source,
                ],
                cwd=isolated_source,
                check=False,
                capture_output=True,
                text=True,
                env=_clean_subprocess_env(),
            )
            assert install.returncode == 0, install.stderr
            surfaces[mode] = _installed_surface(environment)
            environments[mode] = environment

        assert surfaces["editable"] == surfaces["wheel"]
        repo_script_surface = _repo_script_agent_py_surface(
            _venv_python(environments["wheel"]),
            tmp_path,
        )
        assert repo_script_surface == surfaces["wheel"]["action_list"]
    finally:
        assert _repo_build_inventory() == build_before


def test_editable_and_wheel_install_keep_all_compatibility_entrypoints(tmp_path):
    _assert_isolated_install_contract(tmp_path)


def test_isolated_install_contract_is_parallel_safe_and_idempotent(tmp_path):
    _assert_isolated_install_contract(tmp_path)
