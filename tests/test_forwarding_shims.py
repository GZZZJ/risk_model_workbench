from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT_SHIM_DIR = REPO_ROOT / "jingying_agent"

ROOT_MODULE_TARGETS = {
    "batch_feature_select": "risk_model_workbench.batch_feature_select",
    "cli": "risk_model_workbench.cli",
    "config": "risk_model_workbench.config",
    "dp_feather": "risk_model_workbench.dp_feather",
    "feature_metadata": "risk_model_workbench.feature_metadata",
    "feature_refine": "risk_model_workbench.feature_refine",
    "feature_screening": "risk_model_workbench.feature_screening",
    "manifest": "risk_model_workbench.manifest",
    "project": "risk_model_workbench.project",
    "wide_sql": "risk_model_workbench.wide_sql",
}

_ADJACENT_SOURCE_SCRIPT = textwrap.dedent(
    """
    import importlib.util
    import pathlib
    import sys

    case = sys.argv[1]
    repo_root = pathlib.Path(sys.argv[2])
    other_source = sys.argv[3]
    adjacent_source = sys.argv[4]
    sys.path[:] = [
        str(repo_root),
        other_source,
        adjacent_source,
        *[
            entry
            for entry in sys.path
            if entry not in {str(repo_root), other_source, adjacent_source, ""}
        ],
    ]

    if case == "__init__":
        import jingying_agent
    else:
        path = repo_root / "jingying_agent" / f"{case}.py"
        spec = importlib.util.spec_from_file_location(f"_forward_{case}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    import risk_model_workbench

    canonical = (
        risk_model_workbench
        if case == "__init__"
        else importlib.import_module(f"risk_model_workbench.{case}")
    )
    print(pathlib.Path(canonical.__file__).resolve())
    """
)


def _load_file(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


@pytest.mark.parametrize(("legacy_name", "target_name"), ROOT_MODULE_TARGETS.items())
def test_root_legacy_files_forward_public_objects_by_identity(legacy_name: str, target_name: str):
    target = __import__(target_name, fromlist=["*"])
    shim = _load_file(ROOT_SHIM_DIR / f"{legacy_name}.py", f"_compat_{legacy_name}")

    target_public = {
        name for name in vars(target) if not name.startswith("_")
    }
    assert target_public <= set(vars(shim))
    for name in target_public:
        assert getattr(shim, name) is getattr(target, name)


@pytest.mark.parametrize(("legacy_name", "target_name"), ROOT_MODULE_TARGETS.items())
def test_checkout_legacy_imports_preserve_identity(
    legacy_name: str,
    target_name: str,
):
    legacy = __import__(f"jingying_agent.{legacy_name}", fromlist=["*"])
    target = __import__(target_name, fromlist=["*"])

    for name in vars(target):
        if not name.startswith("_"):
            assert getattr(legacy, name) is getattr(target, name)


@pytest.mark.parametrize("module_name", ["__init__", *ROOT_MODULE_TARGETS])
def test_root_forwarders_prioritize_their_adjacent_source_tree(
    module_name: str,
    tmp_path: Path,
):
    other_source = tmp_path / "other-worktree" / "src"
    fake_package = other_source / "risk_model_workbench"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        'ORIGIN = "other-worktree"\n',
        encoding="utf-8",
    )
    adjacent_source = REPO_ROOT / "src"
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME"}
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _ADJACENT_SOURCE_SCRIPT,
            module_name,
            str(REPO_ROOT),
            str(other_source),
            str(adjacent_source),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    expected = adjacent_source / "risk_model_workbench"
    expected = (
        expected / "__init__.py"
        if module_name == "__init__"
        else expected / module_name / "__init__.py"
        if module_name == "project"
        else expected / f"{module_name}.py"
    )
    assert Path(result.stdout.strip()).resolve() == expected.resolve()


@pytest.mark.parametrize("module_name", ["batch_feature_select", "feature_metadata", "feature_refine"])
def test_former_root_scripts_forward_help_without_an_installed_package(module_name: str):
    result = subprocess.run(
        [sys.executable, str(ROOT_SHIM_DIR / f"{module_name}.py"), "--help"],
        cwd=REPO_ROOT.parent,
        check=False,
        capture_output=True,
        text=True,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_shadowed_project_module_forwards_complete_create_api_by_identity():
    from risk_model_workbench.project import create

    shim = _load_file(
        REPO_ROOT / "src" / "risk_model_workbench" / "project.py",
        "_risk_model_workbench_project_compat",
    )

    for name in [
        "REPO_ROOT",
        "TEMPLATE_ROOT",
        "ProjectContext",
        "default_context",
        "render_text",
        "create_project",
    ]:
        assert getattr(shim, name) is getattr(create, name)


def test_project_package_exposes_the_same_complete_create_api():
    from risk_model_workbench import project
    from risk_model_workbench.project import create

    for name in [
        "REPO_ROOT",
        "TEMPLATE_ROOT",
        "ProjectContext",
        "default_context",
        "render_text",
        "create_project",
    ]:
        assert getattr(project, name) is getattr(create, name)
