from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import inspect
import json
import os
import pickletools
import re
import subprocess
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import risk_model_workbench
from risk_model_workbench.cli import build_parser, main as rmw_main
from risk_model_workbench.harness.actions import ActionSpec, list_action_specs
from risk_model_workbench.harness.invocation import ActionInvocation
from risk_model_workbench.harness.tools import ToolSpec, list_tool_specs


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPO_ROOT = Path(risk_model_workbench.__file__).resolve().parents[2]
FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "compatibility" / "cli_surface.json"
EXPECTED_CONSOLE_SCRIPTS = {
    "rmw": "risk_model_workbench.cli:main",
    "jm": "risk_model_workbench.cli:main",
    "jingying-agent": "risk_model_workbench.cli:main",
}
LEGACY_MODULES = (
    "jingying_agent",
    "jingying_agent.batch_feature_select",
    "jingying_agent.cli",
    "jingying_agent.config",
    "jingying_agent.dp_feather",
    "jingying_agent.feature_metadata",
    "jingying_agent.feature_refine",
    "jingying_agent.feature_screening",
    "jingying_agent.manifest",
    "jingying_agent.project",
    "jingying_agent.wide_sql",
    "jingying_model_agent",
    "jingying_model_agent.cli",
    "jingying_model_agent.config",
    "jingying_model_agent.project",
    "jingying_model_agent.harness.actions",
    "jingying_model_agent.harness.tools",
)
READ_ONLY_CASES = (
    ("action", "list", "--json"),
    ("tool", "list", "--json"),
    ("rules", "list", "--json"),
    ("workflow", "validate", "--workflow", "full_modeling"),
)
CURRENT_PICKLE_MODULES = {
    "collections",
    "lightgbm.basic",
    "numpy",
    "numpy._core.multiarray",
    "numpy.core.multiarray",
}
LEGACY_PICKLE_PREFIXES = ("jingying_agent", "jingying_model_agent")
_MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__qualname__))
    return f"{module}.{qualname}"


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        normalized = value.replace(str(REPO_ROOT), "<REPO_ROOT>")
        return _MEMORY_ADDRESS.sub(" at <ADDRESS>", normalized)
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_normalize(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if callable(value):
        return f"<callable:{_callable_name(value)}>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    normalized = repr(value).replace(str(REPO_ROOT), "<REPO_ROOT>")
    return _MEMORY_ADDRESS.sub(" at <ADDRESS>", normalized)


def _command_help(action: argparse._SubParsersAction[argparse.ArgumentParser]) -> dict[str, str | None]:
    return {choice.dest: _normalize(choice.help) for choice in action._choices_actions}


def _argument_surface(action: argparse.Action) -> dict[str, Any]:
    return {
        "action": type(action).__name__,
        "choices": _normalize(action.choices),
        "const": _normalize(action.const),
        "default": _normalize(action.default),
        "dest": action.dest,
        "help": _normalize(action.help),
        "metavar": _normalize(action.metavar),
        "nargs": _normalize(action.nargs),
        "option_strings": list(action.option_strings),
        "required": action.required,
        "type": _callable_name(action.type) if action.type is not None else None,
    }


def _parser_surface(parser: argparse.ArgumentParser, command_path: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    subparser_action = next(
        (action for action in parser._actions if isinstance(action, argparse._SubParsersAction)),
        None,
    )
    arguments = [
        _argument_surface(action)
        for action in parser._actions
        if not isinstance(action, argparse._SubParsersAction)
    ]
    groups = [
        {
            "required": group.required,
            "destinations": [action.dest for action in group._group_actions],
        }
        for group in parser._mutually_exclusive_groups
    ]
    row: dict[str, Any] = {
        "allow_abbrev": parser.allow_abbrev,
        "arguments": arguments,
        "argument_groups": [
            {
                "title": group.title,
                "description": _normalize(group.description),
                "destinations": [action.dest for action in group._group_actions],
            }
            for group in parser._action_groups
        ],
        "command_path": list(command_path),
        "description": _normalize(parser.description),
        "epilog": _normalize(parser.epilog),
        "format_help": _normalize(parser.format_help()),
        "formatter_class": _callable_name(parser.formatter_class),
        "mutually_exclusive_groups": groups,
        "prog": _normalize(parser.prog),
        "usage": _normalize(parser.usage),
    }
    rows = [row]
    if subparser_action is not None:
        help_by_command = _command_help(subparser_action)
        row["subcommands"] = [
            {"name": name, "help": help_by_command.get(name)}
            for name in subparser_action.choices
        ]
        for name, child in subparser_action.choices.items():
            rows.extend(_parser_surface(child, (*command_path, name)))
    return rows


def _tracked_model_paths() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.pkl"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return tuple(REPO_ROOT / value for value in sorted(set(result.stdout.splitlines())) if value)


def _pickle_global_modules(path: Path) -> tuple[set[str], int]:
    """Inspect GLOBAL opcodes without importing or unpickling the artifact."""
    modules: set[str] = set()
    memo: dict[int, str | None] = {}
    next_memo_index = 0
    current_top: str | None = None
    global_operands: list[str | None] = []
    unresolved_stack_globals = 0
    for opcode, argument, _ in pickletools.genops(path.read_bytes()):
        if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
            current_top = str(argument)
            global_operands.append(current_top)
            global_operands = global_operands[-2:]
        elif opcode.name == "MEMOIZE":
            memo[next_memo_index] = current_top
            next_memo_index += 1
        elif opcode.name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[int(argument)] = current_top
        elif opcode.name in {"BINGET", "LONG_BINGET", "GET"}:
            current_top = memo.get(int(argument))
            global_operands.append(current_top)
            global_operands = global_operands[-2:]
        elif opcode.name == "GLOBAL":
            modules.add(str(argument).split()[0])
            current_top = None
            global_operands.clear()
        elif opcode.name == "STACK_GLOBAL":
            if len(global_operands) == 2 and all(isinstance(item, str) for item in global_operands):
                modules.add(str(global_operands[0]))
            else:
                unresolved_stack_globals += 1
            current_top = None
            global_operands.clear()
        else:
            current_top = None
            global_operands.clear()
    return modules, unresolved_stack_globals


def _pickle_module_inventory() -> dict[str, list[str]]:
    inventory: dict[str, list[str]] = {}
    for path in _tracked_model_paths():
        modules, unresolved = _pickle_global_modules(path)
        assert unresolved == 0, f"could not statically resolve all STACK_GLOBAL opcodes in {path}"
        inventory[path.relative_to(REPO_ROOT).as_posix()] = sorted(modules)
    return inventory


def _dataclass_field_contract(spec_type: type[Any]) -> list[dict[str, Any]]:
    rows = []
    for field in dataclasses.fields(spec_type):
        default = "<MISSING>" if field.default is dataclasses.MISSING else _normalize(field.default)
        default_factory = (
            "<MISSING>"
            if field.default_factory is dataclasses.MISSING
            else _normalize(field.default_factory)
        )
        rows.append(
            {
                "name": field.name,
                "type": str(field.type),
                "default": default,
                "default_factory": default_factory,
                "init": field.init,
                "repr": field.repr,
                "compare": field.compare,
                "kw_only": field.kw_only,
            }
        )
    return rows


def _spec_contract() -> dict[str, Any]:
    actions = []
    for spec in list_action_specs():
        actions.append(
            {
                "id": spec.id,
                "attributes": {
                    field.name: _normalize(getattr(spec, field.name))
                    for field in dataclasses.fields(spec)
                },
                "repr": repr(spec),
            }
        )
    tools = []
    for spec in list_tool_specs():
        invocation = ActionInvocation(
            tool_name=spec.name,
            params={},
            project="<project>",
            version_id="<version_id>",
        )
        tools.append(
            {
                "name": spec.name,
                "attributes": {
                    field.name: (
                        callable(getattr(spec, field.name))
                        if field.name == "render_argv"
                        else _normalize(getattr(spec, field.name))
                    )
                    for field in dataclasses.fields(spec)
                },
                "command_template": spec.command_template,
                "rendered_argv": spec.render_argv(invocation) if spec.render_argv else None,
                "repr": repr(spec),
            }
        )
    return {
        "ActionSpec": {
            "signature": str(inspect.signature(ActionSpec)),
            "fields": _dataclass_field_contract(ActionSpec),
            "instances": actions,
        },
        "ToolSpec": {
            "signature": str(inspect.signature(ToolSpec)),
            "fields": _dataclass_field_contract(ToolSpec),
            "instances": tools,
        },
    }


def _legacy_public_api() -> dict[str, list[dict[str, Any]]]:
    script = (
        "import importlib, inspect, json, re, sys; "
        "inventory={}; "
        "modules=json.loads(sys.argv[1]); "
        "norm=lambda value: re.sub(r' at 0x[0-9a-fA-F]+', ' at <ADDRESS>', value); "
        "exec(\"for module_name in modules:\\n"
        " module=importlib.import_module(module_name)\\n"
        " exported=getattr(module, '__all__', None)\\n"
        " names=sorted(exported if exported else (name for name in vars(module) if not name.startswith('_')))\\n"
        " rows=[]\\n"
        " for name in names:\\n"
        "  value=getattr(module, name)\\n"
        "  row={'name': name, 'kind': type(value).__name__}\\n"
        "  if callable(value):\\n"
        "   try: row['signature']=norm(str(inspect.signature(value)))\\n"
        "   except (TypeError, ValueError): row['signature']='<unavailable>'\\n"
        "  rows.append(row)\\n"
        " inventory[module_name]=rows\"); "
        "print(json.dumps(inventory, ensure_ascii=False, sort_keys=True))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SOURCE_REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(LEGACY_MODULES)],
        cwd=SOURCE_REPO_ROOT / "src",
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return json.loads(result.stdout)


def _build_contract_parser() -> argparse.ArgumentParser:
    with patch.object(sys, "argv", ["rmw"]):
        return build_parser()


def _compatibility_snapshot() -> dict[str, Any]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    with patch.dict(os.environ, {"COLUMNS": "80", "LINES": "24"}):
        cli_parsers = _parser_surface(_build_contract_parser())
    return {
        "schema_version": 1,
        "console_scripts": {
            name: pyproject["project"]["scripts"].get(name)
            for name in EXPECTED_CONSOLE_SCRIPTS
        },
        "cli_parsers": cli_parsers,
        "action_specs": [_normalize(spec.to_dict()) for spec in list_action_specs()],
        "tool_specs": [_normalize(spec.to_dict()) for spec in list_tool_specs()],
        "spec_contract": _spec_contract(),
        "legacy_public_api": _legacy_public_api(),
        "pickle_module_inventory": _pickle_module_inventory(),
    }


def test_normalized_cli_and_registry_surface_matches_baseline():
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    actual = _compatibility_snapshot()

    serialized = json.dumps(actual, ensure_ascii=False, sort_keys=True)
    assert str(REPO_ROOT) not in serialized
    expected_legacy_api = expected.pop("legacy_public_api")
    actual_legacy_api = actual.pop("legacy_public_api")
    assert actual == expected
    for module_name, expected_symbols in expected_legacy_api.items():
        actual_symbols = {row["name"]: row for row in actual_legacy_api[module_name]}
        for expected_symbol in expected_symbols:
            assert actual_symbols.get(expected_symbol["name"]) == expected_symbol


def test_console_script_declarations_remain_compatible():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert {name: scripts.get(name) for name in EXPECTED_CONSOLE_SCRIPTS} == EXPECTED_CONSOLE_SCRIPTS


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_legacy_module_paths_remain_importable(module_name: str):
    assert importlib.import_module(module_name) is not None


def test_action_and_tool_specs_keep_public_dataclass_semantics():
    for spec in list_action_specs():
        clone = replace(spec)
        assert clone == spec
        assert repr(clone) == repr(spec)
        assert spec.to_dict()["id"] == spec.id

    for spec in list_tool_specs():
        clone = replace(spec)
        assert clone == spec
        assert repr(clone) == repr(spec)
        assert spec.to_dict()["name"] == spec.name
        assert spec.to_dict()["command_template"] == spec.command_template


@pytest.mark.parametrize("argv", READ_ONLY_CASES)
def test_jm_read_only_surface_matches_rmw(argv: tuple[str, ...], capsys: pytest.CaptureFixture[str]):
    from jingying_model_agent.cli import main as jm_main

    rmw_status = rmw_main(list(argv))
    rmw_capture = capsys.readouterr()
    jm_status = jm_main(list(argv))
    jm_capture = capsys.readouterr()

    assert (jm_status, jm_capture.out, jm_capture.err) == (
        rmw_status,
        rmw_capture.out,
        rmw_capture.err,
    )


def test_committed_model_pickles_only_reference_reviewed_import_paths():
    inventory = _pickle_module_inventory()
    assert inventory, "expected committed model.pkl artifacts to be inventoried"

    modules = {module for referenced in inventory.values() for module in referenced}
    assert "lightgbm.basic" in modules

    # If a historical model starts naming a legacy project package, its import
    # path becomes a permanent compatibility contract and must retain a shim.
    legacy_modules = {
        module
        for module in modules
        if module.startswith(LEGACY_PICKLE_PREFIXES)
    }
    assert modules - legacy_modules <= CURRENT_PICKLE_MODULES
    for module in legacy_modules:
        assert importlib.util.find_spec(module) is not None, f"pickle module {module} requires a forwarding shim"
