"""Feature metadata domain API."""

from pathlib import Path

from risk_model_workbench.feature_metadata import *  # noqa: F401,F403


def execute_metadata_action(
    *, project_dir: Path, workspace: Path, config_path: Path | None, tables_file: str | None
) -> int:
    """Execute metadata collection from typed paths."""
    from risk_model_workbench.feature_metadata import run_metadata_service

    return run_metadata_service(
        project_dir=project_dir,
        run_dir=workspace,
        config=str(config_path) if config_path is not None and config_path.exists() else None,
        tables_file=tables_file or "configs/feature_tables.txt",
    )


__all__ = ["execute_metadata_action"]
