"""Internal single source of display and argv templates for harness commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandDeclaration:
    display_template: str
    argv_template: tuple[str, ...]


COMMAND_DECLARATIONS: dict[str, CommandDeclaration] = {
    "validate_config": CommandDeclaration(
        "rmw run init --project <project> --workflow <workflow>",
        ("run", "init", "--project", "{project}", "--workflow", "{workflow}"),
    ),
    "project_status": CommandDeclaration(
        "rmw project status --project <project>",
        ("project", "status", "--project", "{project}"),
    ),
    "run_status": CommandDeclaration(
        "rmw run status --project <project> --run-id <run_id>",
        ("version", "status", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "run_audit": CommandDeclaration(
        "rmw run audit --project <project> --run-id <run_id>",
        ("version", "audit", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "workflow_validate": CommandDeclaration(
        "rmw workflow validate --workflow <workflow>",
        ("workflow", "validate", "--workflow", "{workflow}"),
    ),
    "rules_list": CommandDeclaration("rmw rules list", ("rules", "list")),
    "sample_check": CommandDeclaration(
        "rmw sample check --project <project> --run-id <run_id>",
        ("sample", "check", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_metadata": CommandDeclaration(
        "rmw feature metadata --project <project> --run-id <run_id>",
        ("feature", "metadata", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_prescreen": CommandDeclaration(
        "rmw feature prescreen --project <project> --run-id <run_id>",
        ("feature", "prescreen", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_prescreen_local": CommandDeclaration(
        "rmw feature prescreen --project <project> --run-id <run_id>",
        ("feature", "prescreen", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_prescreen_prepare": CommandDeclaration(
        "rmw feature prescreen --project <project> --run-id <run_id> --dry-run-sql",
        (
            "feature",
            "prescreen",
            "--project",
            "{project}",
            "--version-id",
            "{version_id}",
            "--dry-run-sql",
        ),
    ),
    "feature_prescreen_execute": CommandDeclaration(
        "rmw feature prescreen --project <project> --run-id <run_id> --sql-approved",
        (
            "feature",
            "prescreen",
            "--project",
            "{project}",
            "--version-id",
            "{version_id}",
            "--sql-approved",
        ),
    ),
    "build_wide_sql": CommandDeclaration(
        "rmw build-wide-sql --project <project> --run-id <run_id>",
        ("build-wide-sql", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "build_wide_sql_local": CommandDeclaration(
        "rmw build-wide-sql --project <project> --run-id <run_id>",
        ("build-wide-sql", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "build_wide_sql_prepare": CommandDeclaration(
        "rmw build-wide-sql --project <project> --run-id <run_id>",
        ("build-wide-sql", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "build_wide_sql_execute": CommandDeclaration(
        "rmw build-wide-sql --project <project> --run-id <run_id> --execute --sql-approved",
        (
            "build-wide-sql",
            "--project",
            "{project}",
            "--version-id",
            "{version_id}",
            "--execute",
            "--sql-approved",
        ),
    ),
    "feature_refine": CommandDeclaration(
        "rmw feature refine --project <project> --run-id <run_id>",
        ("feature", "refine", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_refine_local": CommandDeclaration(
        "rmw feature refine --project <project> --run-id <run_id>",
        ("feature", "refine", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "feature_refine_prepare": CommandDeclaration(
        "rmw feature refine --project <project> --run-id <run_id> --dry-run-sql",
        ("feature", "refine", "--project", "{project}", "--version-id", "{version_id}", "--dry-run-sql"),
    ),
    "feature_refine_execute": CommandDeclaration(
        "rmw feature refine --project <project> --run-id <run_id> --sql-approved",
        ("feature", "refine", "--project", "{project}", "--version-id", "{version_id}", "--sql-approved"),
    ),
    "train_baseline": CommandDeclaration(
        "rmw train --project <project> --run-id <run_id> --experiment <name>",
        ("train", "--project", "{project}", "--version-id", "{version_id}", "--experiment", "{experiment}"),
    ),
    "evaluate": CommandDeclaration(
        "rmw evaluate --project <project> --run-id <run_id>",
        ("evaluate", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "compare": CommandDeclaration(
        "rmw compare --project <project> --run-id <run_id>",
        ("compare", "--project", "{project}", "--version-id", "{version_id}"),
    ),
    "report": CommandDeclaration(
        "rmw report --project <project> --run-id <run_id>",
        ("report", "--project", "{project}", "--version-id", "{version_id}"),
    ),
}


def display_template(command_id: str) -> str:
    return COMMAND_DECLARATIONS[command_id].display_template


def argv_template(command_id: str) -> tuple[str, ...]:
    return COMMAND_DECLARATIONS[command_id].argv_template
