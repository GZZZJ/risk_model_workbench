"""DP query caching helpers for local feather datasets."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from risk_model_workbench.agent.workspace_store import WorkspaceStore


class ExternalOutcomeUnknown(RuntimeError):
    """Raised when an external operation outcome cannot be proven locally."""


def write_external_operation_intent(
    workspace: str | Path,
    *,
    operation_id: str,
    subject_hash: str,
    sql_sha256: str,
    parent_operation_id: str = "",
    approval_id: str = "",
    consumption_receipt: str = "",
    attempt_id: str = "",
) -> Path:
    payload = {
        "operation_id": operation_id,
        "subject_hash": subject_hash,
        "sql_sha256": sql_sha256,
        "parent_operation_id": parent_operation_id or operation_id,
        "approval_id": approval_id,
        "consumption_receipt": consumption_receipt,
        "attempt_id": attempt_id,
        "status": "intent_recorded",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    relative = Path("audit") / "external_operations" / f"{operation_id}.intent.json"
    if not WorkspaceStore(workspace).create_once(relative, payload):
        raise ValueError(f"external operation intent already exists: {operation_id}")
    return Path(workspace) / relative


def write_external_operation_receipt(
    workspace: str | Path,
    *,
    operation_id: str,
    status: str,
) -> Path:
    intent = Path(workspace) / "audit" / "external_operations" / f"{operation_id}.intent.json"
    if not intent.exists() or status not in {"succeeded", "failed"}:
        raise ExternalOutcomeUnknown(f"external operation outcome is unknown: {operation_id}")
    intent_payload = json.loads(intent.read_text(encoding="utf-8"))
    payload = {
        "operation_id": operation_id,
        "parent_operation_id": intent_payload.get("parent_operation_id", operation_id),
        "approval_id": intent_payload.get("approval_id", ""),
        "subject_hash": intent_payload.get("subject_hash", ""),
        "attempt_id": intent_payload.get("attempt_id", ""),
        "status": status,
        "intent_path": str(intent.relative_to(workspace)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    relative = Path("audit") / "external_operations" / f"{operation_id}.receipt.json"
    if not WorkspaceStore(workspace).create_once(relative, payload):
        raise ValueError(f"external operation receipt already exists: {operation_id}")
    return Path(workspace) / relative


def external_operation_context_for_current_attempt() -> dict[str, str] | None:
    from risk_model_workbench.harness.runtime import current_action_attempt

    attempt = current_action_attempt()
    if attempt is None:
        return None
    return {
        "workspace": str(attempt.workspace),
        "attempt_id": attempt.attempt_id,
        "task_id": attempt.task_id,
        "approval_id": attempt.approval_id,
        "subject_hash": attempt.approval_subject_hash,
        "consumption_receipt": attempt.approval_consumption_receipt,
        "parent_operation_id": attempt.parent_operation_id,
    }


def _attempt_external_operation(
    dataset_id: str,
    external_operation_context: dict[str, str] | None = None,
) -> dict[str, str] | None:
    context = external_operation_context or external_operation_context_for_current_attempt()
    if context is None:
        return None
    parent_operation_id = str(context.get("parent_operation_id") or context.get("task_id") or "")
    safe_parent = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent_operation_id).strip("._-") or "operation"
    return {
        **context,
        "parent_operation_id": parent_operation_id,
        "operation_id": f"{safe_parent}__{sha256_text(dataset_id)[:12]}",
    }


def _approved_sql(context: dict[str, str], candidate_sql: str) -> str:
    from risk_model_workbench.agent.approvals import approved_sql_for_consumed_operation

    required = ("workspace", "approval_id", "subject_hash", "consumption_receipt", "parent_operation_id")
    missing = [name for name in required if not str(context.get(name) or "")]
    if missing:
        raise ValueError("external operation approval context missing: " + ",".join(missing))
    return approved_sql_for_consumed_operation(
        context["workspace"],
        approval_id=context["approval_id"],
        subject_hash_value=context["subject_hash"],
        consumption_receipt=context["consumption_receipt"],
        parent_operation_id=context["parent_operation_id"],
        candidate_sql=candidate_sql,
    )


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def relative_display(path: Path, base_dir: Path) -> str:
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def default_dataset_paths(
    project_dir: Path,
    *,
    dataset_id: str,
    data_dir: str | Path = "data/local/dp_feather",
    metadata_dir: str | Path = "data/profile/dp_feather_datasets",
) -> tuple[Path, Path]:
    feather_path = resolve_project_path(project_dir, data_dir) / f"{dataset_id}.feather"
    metadata_path = resolve_project_path(project_dir, metadata_dir) / f"{dataset_id}.json"
    return feather_path, metadata_path


def write_dataset_metadata(
    *,
    project_dir: Path,
    metadata_path: Path,
    feather_path: Path,
    dataset_id: str,
    description: str,
    sql: str,
    status: str,
    row_count: int | None = None,
    column_count: int | None = None,
    columns: list[str] | None = None,
    note: str = "",
    source: str = "dp_query",
    source_path: str | Path | None = None,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "description": description,
        "status": status,
        "created_or_updated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": {
            "feather_path": relative_display(feather_path, project_dir),
            "gitignored": True,
        },
        "source": source,
        "dimensions": {
            "rows": row_count,
            "columns": column_count,
        },
        "sql": sql,
        "sql_sha256": sha256_text(sql),
    }
    if columns is not None:
        payload["columns"] = columns
    if source_path is not None:
        resolved_source = resolve_project_path(project_dir, source_path)
        payload["storage"]["source_path"] = relative_display(resolved_source, project_dir)
    if note:
        payload["note"] = note
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resolve_approved_local_feather(project_dir: Path, approved_local_feather_path: str | Path | None) -> Path | None:
    if not approved_local_feather_path:
        return None
    path = resolve_project_path(project_dir, approved_local_feather_path)
    if path.suffix.lower() != ".feather":
        raise ValueError("approved local feather path must end with .feather")
    if not path.exists():
        raise FileNotFoundError(f"approved local feather path does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"approved local feather path is not a file: {path}")
    return path


def print_sql_review(
    *,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql: str,
) -> None:
    print("=" * 80)
    print("DP SQL REVIEW REQUIRED")
    print("=" * 80)
    print(f"dataset_id: {dataset_id}")
    print(f"description: {description}")
    print(f"feather_path: {feather_path}")
    print(f"metadata_path: {metadata_path}")
    print("-" * 80)
    print(sql.rstrip())
    print("-" * 80)


def print_sql_execution_review(
    *,
    operation_id: str,
    description: str,
    metadata_path: Path,
    sql: str,
) -> None:
    print("=" * 80)
    print("DP SQL REVIEW REQUIRED")
    print("=" * 80)
    print(f"operation_id: {operation_id}")
    print(f"description: {description}")
    print(f"metadata_path: {metadata_path}")
    print("-" * 80)
    print(sql.rstrip())
    print("-" * 80)


def require_sql_approval(
    *,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql: str,
    sql_approved: bool,
) -> None:
    print_sql_review(
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        sql=sql,
    )
    if sql_approved:
        print("[SQL] Approval flag received; running DP query.")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Refusing to query DP without SQL approval. Review the SQL above, then rerun with --sql-approved."
        )
    answer = input("Type APPROVE to run this DP query: ").strip()
    if answer != "APPROVE":
        raise RuntimeError("DP query cancelled before execution.")


def require_sql_execution_approval(
    *,
    operation_id: str,
    description: str,
    metadata_path: Path,
    sql: str,
    sql_approved: bool,
) -> None:
    print_sql_execution_review(
        operation_id=operation_id,
        description=description,
        metadata_path=metadata_path,
        sql=sql,
    )
    if sql_approved:
        print("[SQL] Approval flag received; executing DP SQL.")
        return
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Refusing to execute DP SQL without approval. Review the SQL above, then rerun with --sql-approved."
        )
    answer = input("Type APPROVE to execute this DP SQL: ").strip()
    if answer != "APPROVE":
        raise RuntimeError("DP SQL execution cancelled before execution.")


def execute_dp_sql(
    *,
    project_dir: Path,
    sql: str,
    operation_id: str,
    description: str,
    metadata_path: Path,
    sql_approved: bool = False,
    progress: Any | None = None,
    audit_workspace: str | Path | None = None,
    approval_subject_hash: str = "",
) -> dict[str, Any]:
    """Execute a reviewed DP SQL statement, including DDL statements."""
    if progress and not sql_approved:
        progress.emit(
            step="sql_review",
            status="waiting_for_approval",
            message=f"DP SQL 需要审批：{operation_id}",
            metrics={"operation_id": operation_id, "metadata_path": relative_display(metadata_path, project_dir)},
        )
    require_sql_execution_approval(
        operation_id=operation_id,
        description=description,
        metadata_path=metadata_path,
        sql=sql,
        sql_approved=sql_approved,
    )

    if progress:
        progress.emit(step="dp_sql_execute", message=f"开始执行 DP SQL：{operation_id}", metrics={"operation_id": operation_id})
    from tmlpatch.database import TMLSQLClient

    context = external_operation_context_for_current_attempt()
    submitted_sql = sql
    if context is not None:
        if context.get("parent_operation_id") != operation_id:
            raise ValueError("external operation id does not match consumed approval")
        submitted_sql = _approved_sql(context, sql)
        audit_workspace = context["workspace"]
        approval_subject_hash = context["subject_hash"]
    if audit_workspace is not None:
        write_external_operation_intent(
            audit_workspace,
            operation_id=operation_id,
            subject_hash=approval_subject_hash or sha256_text(f"explicit_cli_approval:{sql}"),
            sql_sha256=sha256_text(submitted_sql),
            parent_operation_id=operation_id,
            approval_id=str((context or {}).get("approval_id") or ""),
            consumption_receipt=str((context or {}).get("consumption_receipt") or ""),
            attempt_id=str((context or {}).get("attempt_id") or ""),
        )
    client = None
    execution_method = "client.sql"
    try:
        client = TMLSQLClient()
        result = client.sql(submitted_sql)
        if hasattr(result, "execute"):
            result.execute()
            execution_method = "client.sql.execute"
        elif hasattr(result, "to_pandas"):
            result.to_pandas()
            execution_method = "client.sql.to_pandas"
        if audit_workspace is not None:
            write_external_operation_receipt(audit_workspace, operation_id=operation_id, status="succeeded")
    except Exception as exc:
        if audit_workspace is not None:
            raise ExternalOutcomeUnknown(f"external operation outcome is unknown: {operation_id}") from exc
        raise
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception:
                pass

    executed_at = datetime.now().isoformat(timespec="seconds")
    if progress:
        progress.emit(
            step="dp_sql_execute_done",
            message=f"DP SQL 执行完成：{operation_id}",
            metrics={"operation_id": operation_id, "execution_method": execution_method},
        )
    return {
        "operation_id": operation_id,
        "status": "executed",
        "executed_at": executed_at,
        "execution_method": execution_method,
        "sql_sha256": sha256_text(submitted_sql),
    }


def fetch_dp_query_to_feather(
    *,
    project_dir: Path,
    sql: str,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    sql_approved: bool = False,
    progress: Any | None = None,
    external_operation_context: dict[str, str] | None = None,
) -> Path:
    """Run a reviewed DP query, write feather locally, and record metadata."""
    if progress and not sql_approved:
        progress.emit(
            step="sql_review",
            status="waiting_for_approval",
            message=f"DP SQL 需要审批：{dataset_id}",
            metrics={"dataset_id": dataset_id, "metadata_path": relative_display(metadata_path, project_dir)},
        )
    require_sql_approval(
        dataset_id=dataset_id,
        description=description,
        feather_path=feather_path,
        metadata_path=metadata_path,
        sql=sql,
        sql_approved=sql_approved,
    )

    from tmlpatch.database import TMLSQLClient

    if progress:
        progress.emit(step="dp_query", message=f"开始执行 DP 查询：{dataset_id}", metrics={"dataset_id": dataset_id})
    operation = _attempt_external_operation(dataset_id, external_operation_context) if sql_approved else None
    submitted_sql = sql
    if operation is not None:
        submitted_sql = _approved_sql(operation, sql)
        write_external_operation_intent(
            operation["workspace"],
            operation_id=operation["operation_id"],
            subject_hash=operation["subject_hash"],
            sql_sha256=sha256_text(submitted_sql),
            parent_operation_id=operation["parent_operation_id"],
            approval_id=operation["approval_id"],
            consumption_receipt=operation["consumption_receipt"],
            attempt_id=operation.get("attempt_id", ""),
        )
    client = None
    try:
        client = TMLSQLClient()
        df = client.sql(submitted_sql).to_pandas()
        if operation is not None:
            write_external_operation_receipt(
                operation["workspace"], operation_id=operation["operation_id"], status="succeeded"
            )
    except Exception as exc:
        if operation is not None:
            raise ExternalOutcomeUnknown(
                f"external operation outcome is unknown: {operation['operation_id']}"
            ) from exc
        raise
    finally:
        if client is not None:
            try:
                client.stop()
            except Exception:
                pass

    feather_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_feather(feather_path)
    if progress:
        progress.emit(
            step="dp_query_done",
            message=f"DP 查询完成并写入 feather：{dataset_id}，{len(df)} 行 {len(df.columns)} 列",
            metrics={
                "dataset_id": dataset_id,
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "feather_path": relative_display(feather_path, project_dir),
            },
        )
    write_dataset_metadata(
        project_dir=project_dir,
        metadata_path=metadata_path,
        feather_path=feather_path,
        dataset_id=dataset_id,
        description=description,
        sql=sql,
        status="ready",
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
    )
    return feather_path


def load_or_fetch_dp_feather(
    *,
    project_dir: Path,
    sql: str,
    dataset_id: str,
    description: str,
    feather_path: Path,
    metadata_path: Path,
    refresh: bool = False,
    sql_approved: bool = False,
    approved_local_feather_path: str | Path | None = None,
    progress: Any | None = None,
    external_operation_context: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return a local feather-backed DP dataset, fetching only after SQL approval."""
    approved_path = _resolve_approved_local_feather(project_dir, approved_local_feather_path)
    if approved_path is not None:
        if progress:
            progress.emit(
                step="read_approved_local_feather",
                message=f"正在读取已确认的本地 feather：{dataset_id}",
                metrics={"dataset_id": dataset_id, "feather_path": relative_display(approved_path, project_dir)},
            )
        df = pd.read_feather(approved_path)
        write_dataset_metadata(
            project_dir=project_dir,
            metadata_path=metadata_path,
            feather_path=approved_path,
            dataset_id=dataset_id,
            description=description,
            sql=sql,
            status="ready",
            row_count=int(len(df)),
            column_count=int(len(df.columns)),
            columns=[str(column) for column in df.columns],
            source="approved_local_feather",
            source_path=approved_path,
            note="Read from a user-approved local feather path; no remote DP pull was executed.",
        )
        if progress:
            progress.emit(
                step="read_approved_local_feather_done",
                message=f"本地 feather 读取完成：{dataset_id}，{len(df)} 行 {len(df.columns)} 列",
                metrics={"dataset_id": dataset_id, "rows": int(len(df)), "columns": int(len(df.columns))},
            )
        return df

    if refresh or not feather_path.exists() or not metadata_path.exists():
        write_dataset_metadata(
            project_dir=project_dir,
            metadata_path=metadata_path,
            feather_path=feather_path,
            dataset_id=dataset_id,
            description=description,
            sql=sql,
            status="sql_review_required" if refresh or not feather_path.exists() else "ready",
            note="Run with --sql-approved only after the displayed SQL has been reviewed.",
        )
    if refresh or not feather_path.exists():
        fetch_dp_query_to_feather(
            project_dir=project_dir,
            sql=sql,
            dataset_id=dataset_id,
            description=description,
            feather_path=feather_path,
            metadata_path=metadata_path,
            sql_approved=sql_approved,
            progress=progress,
            external_operation_context=external_operation_context,
        )
    if progress:
        progress.emit(
            step="read_feather",
            message=f"正在读取本地 feather：{dataset_id}",
            metrics={"dataset_id": dataset_id, "feather_path": relative_display(feather_path, project_dir)},
        )
    df = pd.read_feather(feather_path)
    if progress:
        progress.emit(
            step="read_feather_done",
            message=f"本地 feather 读取完成：{dataset_id}，{len(df)} 行 {len(df.columns)} 列",
            metrics={"dataset_id": dataset_id, "rows": int(len(df)), "columns": int(len(df.columns))},
        )
    write_dataset_metadata(
        project_dir=project_dir,
        metadata_path=metadata_path,
        feather_path=feather_path,
        dataset_id=dataset_id,
        description=description,
        sql=sql,
        status="ready",
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=[str(column) for column in df.columns],
    )
    return df
