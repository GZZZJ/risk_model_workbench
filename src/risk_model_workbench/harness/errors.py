"""Standard failure classes for harnessed workbench actions."""

from __future__ import annotations

from dataclasses import dataclass


SQL_APPROVAL_REQUIRED = "sql_approval_required"
DATA_MISSING = "data_missing"
ARTIFACT_CONTRACT_FAILED = "artifact_contract_failed"
SCAFFOLD_ONLY = "scaffold_only"
DEPENDENCY_MISSING = "dependency_missing"
TRANSIENT_IO = "transient_io"
ADVISOR_REQUIRED = "advisor_required"
UNKNOWN = "unknown"
MISSING_ACTION_RESULT = "missing_action_result"
INVALID_ACTION_RESULT = "invalid_action_result"
EXTERNAL_OUTCOME_UNKNOWN = "external_outcome_unknown"


class ActionResultError(RuntimeError):
    """Base error for immutable attempt result receipts."""


class DuplicateActionResultError(ActionResultError):
    """Raised when a second semantic result targets the same attempt."""


class InvalidActionResultError(ActionResultError):
    """Raised when a result is malformed or bound to another attempt."""


class MissingActionResultError(ActionResultError):
    """Raised when a completed process emitted no semantic receipt."""


class WorkspaceLockedError(RuntimeError):
    """Raised when another local runner already owns the workspace."""


@dataclass(frozen=True)
class FailureClass:
    code: str
    description: str
    retryable: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "description": self.description,
            "retryable": self.retryable,
        }


FAILURE_CLASSES: tuple[FailureClass, ...] = (
    FailureClass(
        SQL_APPROVAL_REQUIRED,
        "A DP or SQL action requires explicit SQL approval before execution.",
    ),
    FailureClass(
        DATA_MISSING,
        "Required local input data, feature lists, scores, or configuration are missing.",
    ),
    FailureClass(
        ARTIFACT_CONTRACT_FAILED,
        "Registered artifacts do not satisfy the workflow stage contract.",
    ),
    FailureClass(
        SCAFFOLD_ONLY,
        "Only scaffold or placeholder evidence exists for the action output.",
    ),
    FailureClass(
        DEPENDENCY_MISSING,
        "An optional runtime dependency required by the action is unavailable.",
    ),
    FailureClass(
        TRANSIENT_IO,
        "A safe local IO operation failed transiently and may be retried.",
        retryable=True,
    ),
    FailureClass(
        ADVISOR_REQUIRED,
        "Host-agent tuning needs an explicit advisor plan before training can continue.",
    ),
    FailureClass(
        MISSING_ACTION_RESULT,
        "The process returned without an ActionResult for the active attempt.",
    ),
    FailureClass(
        INVALID_ACTION_RESULT,
        "The ActionResult does not match the active task, invocation, or workspace.",
    ),
    FailureClass(
        EXTERNAL_OUTCOME_UNKNOWN,
        "An external operation may have been submitted, but its final outcome is not proven locally.",
    ),
    FailureClass(
        UNKNOWN,
        "Fallback for unmapped failures; never retry automatically.",
    ),
)

FAILURE_CODES: tuple[str, ...] = tuple(item.code for item in FAILURE_CLASSES)
RETRYABLE_FAILURE_CODES: tuple[str, ...] = tuple(item.code for item in FAILURE_CLASSES if item.retryable)
NON_RETRYABLE_FAILURE_CODES: tuple[str, ...] = tuple(item.code for item in FAILURE_CLASSES if not item.retryable)


def get_failure_class(code: str) -> FailureClass:
    for item in FAILURE_CLASSES:
        if item.code == code:
            return item
    return get_failure_class(UNKNOWN)


def list_failure_classes() -> tuple[FailureClass, ...]:
    return FAILURE_CLASSES
