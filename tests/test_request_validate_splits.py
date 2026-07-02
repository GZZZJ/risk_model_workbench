"""End-to-end tests for the split-consistency gate (time-out pollution defense).

Covers all three CLI gates (request validate / plan create / version|run init)
plus the materialize-level fail-fast and the --skip-split-check escape hatch.
The invariant under test: a time-out (OOT) label must never silently reach the
in-time validation set (valid_values) used by early stopping / tuning.
"""

from pathlib import Path

import pytest
import yaml

from risk_model_workbench.cli import main
from risk_model_workbench.request.materialize import materialize_request_runtime_configs
from risk_model_workbench.request.splits import SplitValidationError
from risk_model_workbench.request.validate import validate_model_request


# ---------- fixtures ----------

def _meta(oos_values, oot_values, dev_values=None):
    return {
        "request_id": "split-check-req",
        "project": "pytest-project",
        "workflow": "full_modeling",
        "target_column": "target",
        "split_column": "final_flag",
        "id_columns": ["uid"],
        "evaluation": {"metrics": ["auc"]},
        "reports": {"outputs": ["model_report.md"]},
        "experiments": [{"name": "main", "method": "logistic_regression", "segment": "all"}],
        "splits": {
            "dev": {"values": dev_values or ["DEV"]},
            "oos": {"values": oos_values},
            "oot": {"values": oot_values},
        },
    }


def _write_min_project(tmp_path, *, oos_values, oot_values, dev_values=None, extra_meta=None):
    """Build a minimal project + request.md with caller-controlled splits."""
    project_dir = tmp_path / "project"
    (project_dir / "configs").mkdir(parents=True)
    project_dir.joinpath("project.yml").write_text(
        "\n".join(
            [
                "project:",
                "  name: pytest-project",
                "  display_name: Pytest Project",
                "  scenario: pytest",
                "data:",
                "  source_table: mart.base",
                "  id_columns: [uid]",
                "  target_column: target",
                "  split_column: final_flag",
                "  time_column: apply_time",
                "  period_column: apply_month",
                "split:",
                "  source_column: final_flag",
                "  ins_values: [DEV]",
                "  oos_values: [DEV-OOS]",
                "  oot_values: [OOT, OOT-OOS]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    project_dir.joinpath("configs", "train.yaml").write_text(
        "training:\n  default_algorithm: logistic_regression\ninput:\n  base_columns: [uid, final_flag, target]\n",
        encoding="utf-8",
    )
    meta = _meta(oos_values, oot_values, dev_values=dev_values)
    if extra_meta:
        meta.update(extra_meta)
    request_path = tmp_path / "request.md"
    request_path.write_text("---\n" + yaml.safe_dump(meta, allow_unicode=True) + "---\n\n# request\n", encoding="utf-8")
    return project_dir, request_path


# ---------- unit: validate_model_request ----------

def test_validate_returns_split_errors_for_intersection():
    # oos overlaps oot: OOT-OOS leaks into the in-time validation set
    result = validate_model_request({"metadata": _meta(["OOT-OOS"], ["OOT", "OOT-OOS"])}, None)
    assert result["split_errors"], "expected split_errors for oos∩oot"
    assert any("相交" in e for e in result["split_errors"])
    assert result["status"] == "failed"


def test_validate_clean_config_has_no_split_errors():
    result = validate_model_request({"metadata": _meta(["DEV-OOS"], ["OOT", "OOT-OOS"])}, None)
    assert result["split_errors"] == []
    assert result["status"] == "ok"


def test_validate_dead_training_valid_values_is_warned():
    meta = _meta(["DEV-OOS"], ["OOT"])
    meta["training"] = {"valid_values": ["DEV-OOS"]}  # dead field, silently overridden
    result = validate_model_request({"metadata": meta}, None)
    assert result["split_errors"] == []
    assert any("training.valid_values" in w for w in result["warnings"])


# ---------- unit: materialize (producer-side defense) ----------

def test_materialize_strict_raises_on_intersection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SplitValidationError):
        materialize_request_runtime_configs(
            request_doc={"metadata": _meta(["OOT"], ["OOT", "OOT-OOS"])},
            project_dir=tmp_path,
            run_dir=run_dir,
            strict=True,
        )


def test_materialize_non_strict_passes_intersection(tmp_path):
    # escape hatch: strict=False must not raise, even on a polluting config
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    materialize_request_runtime_configs(
        request_doc={"metadata": _meta(["OOT"], ["OOT", "OOT-OOS"])},
        project_dir=tmp_path,
        run_dir=run_dir,
        strict=False,
    )
    # runtime config still produced
    assert (run_dir / "configs_runtime").is_dir()


def test_materialize_oos_empty_never_falls_back_to_oot(tmp_path):
    # the original bug: empty oos silently fell back to OOT. Now it must resolve
    # to the safe in-time default (DEV-OOS), never OOT, even under non-strict.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    meta = _meta([""], ["OOT"])  # explicit dirty oos -> resolves to default
    materialize_request_runtime_configs(
        request_doc={"metadata": meta},
        project_dir=tmp_path,
        run_dir=run_dir,
        strict=False,
    )
    train_cfg = yaml.safe_load((run_dir / "configs_runtime" / "train.yaml").read_text(encoding="utf-8"))
    assert train_cfg["training"]["valid_values"] == ["DEV-OOS"]


# ---------- CLI gate 1: rmw request validate ----------

def test_cli_request_validate_blocks_intersection(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    rc = main(["request", "validate", "--request", str(request_path), "--project", str(project_dir)])
    assert rc == 1


def test_cli_request_validate_skip_split_check_passes(tmp_path, capsys):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    rc = main([
        "request", "validate",
        "--request", str(request_path), "--project", str(project_dir),
        "--skip-split-check",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    assert "suppressed by --skip-split-check" in out


def test_cli_request_validate_passes_clean(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["DEV-OOS"], oot_values=["OOT", "OOT-OOS"])
    rc = main(["request", "validate", "--request", str(request_path), "--project", str(project_dir)])
    assert rc == 0


# ---------- CLI gate 2: rmw plan create ----------

def test_cli_plan_create_blocks_intersection(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    rc = main(["plan", "create", "--project", str(project_dir), "--request", str(request_path)])
    assert rc == 1


# ---------- CLI gate 3: rmw version init (parse->materialize passthrough closed) ----------

def test_cli_version_init_blocks_intersection_no_materialize(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    version_id = "v_test_intersect"
    rc = main([
        "version", "init",
        "--project", str(project_dir),
        "--workflow", "full_modeling",
        "--version-id", version_id,
        "--request", str(request_path),
    ])
    assert rc == 1
    # gate fires before materialize, so no runtime config is produced
    runtime_dir = project_dir / "versions" / version_id / "configs_runtime"
    produced = list(runtime_dir.glob("*.y*ml")) if runtime_dir.exists() else []
    assert produced == [], f"expected no runtime config, found {[p.name for p in produced]}"


def test_cli_version_init_skip_split_check_materializes(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    version_id = "v_test_skip"
    rc = main([
        "version", "init",
        "--project", str(project_dir),
        "--workflow", "full_modeling",
        "--version-id", version_id,
        "--request", str(request_path),
        "--skip-split-check",
    ])
    assert rc == 0
    runtime_dir = project_dir / "versions" / version_id / "configs_runtime"
    assert runtime_dir.exists() and list(runtime_dir.glob("*.y*ml")), "expected runtime configs under escape hatch"


# ---------- materialize: strict default + defense-in-depth cleansing ----------

def test_materialize_default_strict_raises_on_intersection(tmp_path):
    # omit strict= entirely; direct (non-CLI) callers must get the strict default
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(SplitValidationError):
        materialize_request_runtime_configs(
            request_doc={"metadata": _meta(["OOT"], ["OOT", "OOT-OOS"])},
            project_dir=tmp_path,
            run_dir=run_dir,
        )


def test_materialize_non_strict_cleans_valid_values(tmp_path):
    # escape hatch skips the block, but the persisted valid_values must still be
    # cleansed of time-out labels (never ['OOT']).
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    materialize_request_runtime_configs(
        request_doc={"metadata": _meta(["OOT"], ["OOT", "OOT-OOS"])},
        project_dir=tmp_path,
        run_dir=run_dir,
        strict=False,
    )
    train_cfg = yaml.safe_load((run_dir / "configs_runtime" / "train.yaml").read_text(encoding="utf-8"))
    assert train_cfg["training"]["valid_values"] == ["DEV-OOS"]


def test_materialize_overrides_legacy_train_yaml_valid_values_oot(tmp_path):
    # legacy configs/train.yaml hardcodes valid_values:[OOT]; materialize must
    # override it with the cleansed in-time value, not inherit the pollution.
    project_dir = tmp_path / "proj"
    (project_dir / "configs").mkdir(parents=True)
    project_dir.joinpath("project.yml").write_text(
        "project:\n  name: p\ndata:\n  id_columns: [uid]\n  target_column: target\n  split_column: final_flag\n"
        "split:\n  ins_values: [DEV]\n  oos_values: [DEV-OOS]\n  oot_values: [OOT]\n",
        encoding="utf-8",
    )
    project_dir.joinpath("configs", "train.yaml").write_text("training:\n  valid_values: [OOT]\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    materialize_request_runtime_configs(
        request_doc={"metadata": _meta(["DEV-OOS"], ["OOT"])},
        project_dir=project_dir,
        run_dir=run_dir,
        strict=False,
    )
    train_cfg = yaml.safe_load((run_dir / "configs_runtime" / "train.yaml").read_text(encoding="utf-8"))
    assert train_cfg["training"]["valid_values"] == ["DEV-OOS"]


# ---------- CLI gate 3b: run init (as_version=False) ----------

def test_cli_run_init_blocks_intersection(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    rc = main(["run", "init", "--project", str(project_dir), "--workflow", "full_modeling", "--request", str(request_path), "--run-id", "test_run"])
    assert rc == 1
    runtime_dir = project_dir / "runs" / "test_run" / "configs_runtime"
    produced = list(runtime_dir.glob("*.y*ml")) if runtime_dir.exists() else []
    assert produced == []


def test_cli_run_init_skip_split_check_materializes(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    rc = main([
        "run", "init", "--project", str(project_dir), "--workflow", "full_modeling",
        "--request", str(request_path), "--run-id", "test_run_skip", "--skip-split-check",
    ])
    assert rc == 0
    runtime_dir = project_dir / "runs" / "test_run_skip" / "configs_runtime"
    assert runtime_dir.exists() and list(runtime_dir.glob("*.y*ml"))


# ---------- CLI gate: plan create --skip-split-check ----------

def test_cli_plan_create_skip_split_check_passes(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    out = tmp_path / "plan.yml"
    rc = main(["plan", "create", "--project", str(project_dir), "--request", str(request_path), "--output", str(out), "--skip-split-check"])
    assert rc == 0


# ---------- init failure leaves no orphan workspace ----------

def test_cli_version_init_failure_leaves_no_orphan(tmp_path):
    project_dir, request_path = _write_min_project(tmp_path, oos_values=["OOT"], oot_values=["OOT", "OOT-OOS"])
    version_id = "v_orphan"
    rc = main([
        "version", "init", "--project", str(project_dir), "--workflow", "full_modeling",
        "--version-id", version_id, "--request", str(request_path),
    ])
    assert rc == 1
    version_path = project_dir / "versions" / version_id
    assert not version_path.exists(), "validation failure must not leave an orphan workspace"


# ---------- CLI gate 4: train-time guard (critical bypass fix) ----------

def _write_train_workspace(tmp_path, *, valid_values):
    project_dir = tmp_path / "project"
    (project_dir / "configs").mkdir(parents=True)
    project_dir.joinpath("project.yml").write_text(
        "project:\n  name: pytest-project\n  display_name: Pytest Project\n  scenario: pytest\n"
        "data:\n  source_table: mart.base\n  id_columns: [uid]\n  target_column: target\n"
        "  split_column: final_flag\n  time_column: apply_time\n  period_column: apply_month\n"
        "split:\n  source_column: final_flag\n  ins_values: [DEV]\n  oos_values: [DEV-OOS]\n  oot_values: [OOT, OOT-OOS]\n",
        encoding="utf-8",
    )
    project_dir.joinpath("configs", "train.yaml").write_text(
        f"training:\n  valid_values: {valid_values}\n  train_values: [DEV]\ninput:\n  base_columns: [uid, final_flag, target]\n",
        encoding="utf-8",
    )
    # Initialize a proper run workspace (with run_state.yml) via the CLI itself,
    # so cmd_train's stage machinery can load state. No --request -> no split
    # gate, no materialization; the legacy configs/train.yaml is what train reads.
    main(["run", "init", "--project", str(project_dir), "--workflow", "full_modeling", "--run-id", "r1"])
    return project_dir, project_dir / "runs" / "r1"


def test_cli_train_blocks_oot_valid_values(tmp_path):
    project_dir, run_path = _write_train_workspace(tmp_path, valid_values='["OOT"]')
    rc = main(["train", "--project", str(project_dir), "--run-id", "r1", "--experiment", "baseline"])
    assert rc == 1
    assert (run_path / "modeling" / "baseline" / "train_metrics.json").exists()


def test_cli_train_blocks_scalar_oot_valid_values(tmp_path):
    # YAML scalar `valid_values: OOT` (no brackets) must also be caught after
    # guard input normalization — regressions that iterate the string char-by-char
    # would silently bypass OOT detection.
    project_dir, run_path = _write_train_workspace(tmp_path, valid_values="OOT")
    rc = main(["train", "--project", str(project_dir), "--run-id", "r1", "--experiment", "baseline"])
    assert rc == 1


def test_cli_train_skip_split_check_passes_guard(tmp_path, capsys):
    project_dir, run_path = _write_train_workspace(tmp_path, valid_values='["OOT"]')
    # no input feather -> train will scaffold-fail later, but the split guard must
    # only warn (not block) under --skip-split-check. Assert the warning surfaced.
    main(["train", "--project", str(project_dir), "--run-id", "r1", "--experiment", "baseline", "--skip-split-check"])
    out = capsys.readouterr().out
    assert "suppressed by --skip-split-check" in out
