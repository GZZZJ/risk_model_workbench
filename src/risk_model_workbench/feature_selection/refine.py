"""Feature prescreen/refinement domain API."""

from pathlib import Path

from risk_model_workbench.feature_refine import *  # noqa: F401,F403


def execute_prescreen_action(
    *,
    project_dir: Path,
    workspace: Path,
    config_path: Path | None,
    prepare: bool,
    execute: bool,
    tables: list[str],
    max_tables: int | None,
) -> int:
    from risk_model_workbench.batch_feature_select import run_prescreen_service

    return run_prescreen_service(
        project_dir=project_dir,
        run_dir=workspace,
        stage="feature_prescreen",
        config=str(config_path) if config_path is not None and config_path.exists() else "configs/feature_select.yaml",
        dry_run_sql=prepare,
        sql_approved=execute,
        tables=tables,
        max_tables=max_tables,
    )


def execute_refine_action(
    *,
    project_dir: Path,
    workspace: Path,
    config_path: Path | None,
    prepare: bool,
    execute: bool,
    sample_max_rows: int | None,
) -> int:
    from risk_model_workbench.feature_refine import run_refine_service

    return run_refine_service(
        project_dir=project_dir,
        run_dir=workspace,
        config=str(config_path) if config_path is not None and config_path.exists() else "configs/refine_features.yaml",
        dry_run_sql=prepare,
        sql_approved=execute,
        sample_max_rows=sample_max_rows,
    )


__all__ = ["execute_prescreen_action", "execute_refine_action"]
