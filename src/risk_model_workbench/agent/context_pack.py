"""Deterministic, bounded context snapshots for an external Host-Agent.

Context packs contain only locally generated summaries of explicitly allowed
workspace files.  They never follow symlinks or expose raw datasets, model
binaries, credentials, row-level identifiers, or secret values.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit

import yaml

from risk_model_workbench.agent.workspace_store import WorkspaceStore


CONTEXT_PACK_VERSION = 1
DEFAULT_MAX_FILE_BYTES = 64 * 1024
DEFAULT_MAX_TOTAL_BYTES = 256 * 1024
DEFAULT_MAX_ENTRIES = 128
DEFAULT_MAX_PACK_BYTES = 256 * 1024

_PROHIBITED_SUFFIXES = {
    ".arrow",
    ".avro",
    ".bin",
    ".ckpt",
    ".db",
    ".feather",
    ".joblib",
    ".model",
    ".npy",
    ".npz",
    ".onnx",
    ".orc",
    ".parquet",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".csv",
    ".sqlite",
    ".tsv",
    ".xls",
    ".xlsx",
}
_CREDENTIAL_NAMES = {
    ".env",
    ".netrc",
    "credentials",
    "credentials.json",
    "credentials.yml",
    "credentials.yaml",
    "secrets.json",
    "secrets.yml",
    "secrets.yaml",
}
_SENSITIVE_KEYS = re.compile(
    r"(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"authorization|cookie|session|phone|mobile|email|identity|id[_-]?number|"
    r"customer[_-]?id|user[_-]?id|member[_-]?id|device[_-]?id|card[_-]?no|uid)$",
    re.IGNORECASE,
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@[^\s]+", re.IGNORECASE),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|pk|rk|ghp|github_pat)-?[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"(?i)\b(?:password|passwd|pwd|secret|token|api[_-]?key)\s*[:=]\s*\S+"),
)
_SAMPLE_ID_VALUE = re.compile(r"^(?:[A-Z]{1,4}\d{6,}|\d{11,}|[0-9a-f]{24,})$", re.IGNORECASE)
_STRUCTURED_SUFFIXES = {".json", ".yml", ".yaml"}
_TEXT_SUFFIXES = {".md", ".txt"}
_SAFE_OPERATIONAL_IDS = {
    "action_id",
    "attempt_id",
    "blocker_id",
    "experiment_id",
    "feature_id",
    "metric_id",
    "model_id",
    "operation_id",
    "plan_id",
    "request_id",
    "run_id",
    "segment_id",
    "stage_id",
    "task_id",
    "trial_id",
    "version_id",
}
_AGGREGATE_SUFFIXES = (
    "_count",
    "_rate",
    "_sum",
    "_avg",
    "_mean",
    "_min",
    "_max",
    "_pct",
    "_ratio",
    "_share",
    "_total",
    "_median",
    "_std",
)
_AGGREGATE_DIMENSIONS = {
    "bucket",
    "cohort",
    "date",
    "decile",
    "group",
    "month",
    "metric_name",
    "period",
    "quantile",
    "segment",
    "segment_id",
    "week",
    "year",
}
_AGGREGATE_METRICS = {
    "accuracy",
    "auc",
    "f1",
    "gini",
    "iv",
    "ks",
    "lift",
    "logloss",
    "mae",
    "precision",
    "psi",
    "recall",
    "rmse",
    "value",
}
_TUNING_PARAMETER_KEYS = {
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "bagging_freq",
    "num_boost_round",
    "early_stopping_rounds",
}
_SAMPLE_TOKEN_PATTERNS = (
    re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?![0-9a-f])", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:[A-Z]{1,4}\d{6,}|\d{11,}|[0-9a-f]{24,})(?![A-Za-z0-9])", re.IGNORECASE),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
)


class RowLevelContentError(ValueError):
    """Raised when an otherwise allowed file contains raw row/sample records."""


def build_context_pack(
    workspace: str | Path,
    *,
    project: str,
    version_id: str,
    task_id: str,
    attempt_id: str,
    request_type: str,
    paths: Iterable[str],
    constraints: Iterable[str],
    allowed_tools: Iterable[str],
    output_contract: dict[str, Any],
    provenance: Iterable[str] | None = None,
    approved_text_reports: Iterable[str] | None = None,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_pack_bytes: int = DEFAULT_MAX_PACK_BYTES,
) -> dict[str, Any]:
    """Build a pure, deterministic pack from one version workspace."""
    if min(max_file_bytes, max_total_bytes, max_entries, max_pack_bytes) <= 0:
        raise ValueError("context pack byte limits must be positive")
    root = Path(workspace).resolve()
    approved = {_normalise_input_path(item) for item in (approved_text_reports or [])}
    requested_paths = sorted({_normalise_input_path(item) for item in paths})
    selected_paths = requested_paths[:max_entries]
    rows: list[dict[str, Any]] = []
    total = 0
    for source in selected_paths:
        row, consumed = _summarise_source(
            root,
            source,
            approved_text_reports=approved,
            remaining_bytes=max_total_bytes - total,
            max_file_bytes=max_file_bytes,
        )
        rows.append(row)
        total += consumed

    pack: dict[str, Any] = {
        "version": CONTEXT_PACK_VERSION,
        "project": Path(str(project)).name if str(project) else "",
        "version_id": str(version_id),
        "task_id": str(task_id),
        "attempt_id": str(attempt_id),
        "request_type": str(request_type),
        "files": rows,
        "constraints": sorted({str(item) for item in constraints}),
        "allowed_tools": sorted({str(item) for item in allowed_tools if str(item)}),
        "output_contract": _sanitize(deepcopy(output_contract)),
        "provenance": sorted(
            {_normalise_input_path(item) for item in (provenance if provenance is not None else requested_paths)}
        ),
        "limits": {
            "max_file_bytes": int(max_file_bytes),
            "max_total_bytes": int(max_total_bytes),
            "max_entries": int(max_entries),
            "max_pack_bytes": int(max_pack_bytes),
        },
        "truncation": {
            "requested_entries": len(requested_paths),
            "recorded_entries": len(rows),
            "omitted_entries": len(requested_paths) - len(rows),
        },
    }
    _enforce_serialized_limit(pack, max_pack_bytes)
    pack["context_hash"] = context_pack_hash(pack)
    if _serialized_size(pack) > max_pack_bytes:
        raise ValueError("context pack serialized bytes exceed max_pack_bytes")
    return pack


def context_pack_hash(pack: dict[str, Any]) -> str:
    payload = deepcopy(pack)
    payload.pop("context_hash", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def persist_context_pack(workspace: str | Path, pack: dict[str, Any]) -> str:
    """Publish a content-addressed pack exactly once and verify collisions."""
    expected = context_pack_hash(pack)
    if pack.get("context_hash") != expected:
        raise ValueError("context pack hash mismatch")
    relative = str(PurePosixPath("audit") / "context_packs" / f"{expected}.json")
    store = WorkspaceStore(workspace)
    created = store.create_once(relative, pack)
    if not created and load_context_pack(workspace, relative) != pack:
        raise ValueError(f"context pack identity collision: {expected}")
    return relative


def load_context_pack(workspace: str | Path, relative_path: str) -> dict[str, Any]:
    normalised = _normalise_input_path(relative_path)
    if Path(normalised).is_absolute() or ".." in PurePosixPath(normalised).parts:
        raise ValueError("context pack path must be workspace-relative")
    root = Path(workspace).resolve()
    path = root / normalised
    if _contains_symlink(root, path):
        raise ValueError("context pack may not be a symlink")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    if not isinstance(payload, dict):
        raise ValueError("context pack must be a JSON object")
    if payload.get("context_hash") != context_pack_hash(payload):
        raise ValueError("context pack hash mismatch")
    return payload


def context_pack_capabilities() -> dict[str, Any]:
    return {
        "version": CONTEXT_PACK_VERSION,
        "max_file_bytes": DEFAULT_MAX_FILE_BYTES,
        "max_total_bytes": DEFAULT_MAX_TOTAL_BYTES,
        "max_entries": DEFAULT_MAX_ENTRIES,
        "max_pack_bytes": DEFAULT_MAX_PACK_BYTES,
        "structured_types": sorted(_STRUCTURED_SUFFIXES),
        "approved_text_types": sorted(_TEXT_SUFFIXES),
        "prohibited_types": sorted(_PROHIBITED_SUFFIXES),
        "symlinks": "rejected",
        "summaries": "deterministic_local",
    }


def _summarise_source(
    root: Path,
    source: str,
    *,
    approved_text_reports: set[str],
    remaining_bytes: int,
    max_file_bytes: int,
) -> tuple[dict[str, Any], int]:
    base = {"path": source, "sha256": "", "size": 0, "summary": {}, "missing": False}
    path_error = _path_error(source)
    if path_error:
        return {**base, "status": "rejected", "reason": path_error}, 0
    candidate = root / source
    if _contains_symlink(root, candidate):
        return {**base, "status": "rejected", "reason": "symlink"}, 0
    if not candidate.exists():
        return {**base, "status": "missing", "reason": "not_found", "missing": True}, 0
    if not candidate.is_file():
        return {**base, "status": "rejected", "reason": "not_file"}, 0
    policy_error = _policy_error(source, approved_text_reports)
    if policy_error:
        return {**base, "status": "rejected", "reason": policy_error}, 0
    size = candidate.stat().st_size
    if size > max_file_bytes:
        return {**base, "size": size, "status": "rejected", "reason": "file_bytes_exceeded"}, 0
    if size > remaining_bytes:
        return {**base, "size": size, "status": "rejected", "reason": "total_bytes_exceeded"}, 0
    raw = candidate.read_bytes()
    try:
        summary = _deterministic_summary(source, raw)
    except RowLevelContentError:
        return {
            **base,
            "size": len(raw),
            "status": "rejected",
            "reason": "row_level_content",
        }, 0
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError, TypeError, ValueError):
        return {
            **base,
            "size": len(raw),
            "status": "rejected",
            "reason": "invalid_structured_content",
        }, 0
    return {
        **base,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "summary": summary,
        "status": "included",
    }, len(raw)


def _path_error(source: str) -> str:
    candidate = Path(source)
    if candidate.is_absolute():
        return "absolute_path"
    parts = PurePosixPath(source).parts
    if ".." in parts:
        return "path_escape"
    if not source or source in {".", "./"}:
        return "invalid_path"
    return ""


def _contains_symlink(root: Path, candidate: Path) -> bool:
    current = root
    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _policy_error(source: str, approved_text_reports: set[str]) -> str:
    path = PurePosixPath(source)
    name = path.name.lower()
    suffix = path.suffix.lower()
    lowered_parts = {part.lower() for part in path.parts}
    suffixes = {suffix.lower() for suffix in path.suffixes}
    if (
        name in _CREDENTIAL_NAMES
        or name.startswith(".env.")
        or any(marker in name for marker in ("credential", "secret", "service_account"))
        or suffixes.intersection({".pem", ".key"})
        or any(
            part in {".ssh", ".aws", ".config", "credential", "credentials", "secret", "secrets"}
            or "credential" in part
            or "secret" in part
            or part == ".env"
            or part.startswith(".env.")
            for part in lowered_parts
        )
    ):
        return "credential_file"
    if suffixes.intersection(_PROHIBITED_SUFFIXES):
        return "prohibited_file_type"
    if suffix in _TEXT_SUFFIXES:
        if source.startswith("reports/") and source not in approved_text_reports:
            return "text_report_not_approved"
        if source.startswith("feature_selection/") and suffix == ".txt":
            return ""
        if source not in approved_text_reports:
            return "text_report_not_approved"
        return ""
    if suffix not in _STRUCTURED_SUFFIXES:
        return "file_type_not_allowed"
    if source in {
        "version_state.yml",
        "agent_plan.yml",
        "execution_plan.yml",
        "audit/agent_state.yml",
        "audit/artifact_manifest.json",
    }:
        return ""
    if not path.parts:
        return "path_not_allowed"
    root = path.parts[0]
    if root in {"configs_runtime", "configs_snapshot"}:
        return ""
    allowed_markers = {
        "modeling": ("metric", "summary", "tuning_context", "config", "trial", "metadata"),
        "feature_selection": ("metric", "summary", "config", "selection", "stability", "correlation", "iv"),
        "evaluation": ("metric", "summary", "comparison", "config", "threshold", "stability"),
    }
    if root in allowed_markers and any(marker in name for marker in allowed_markers[root]):
        return ""
    return "path_not_allowed"


def _deterministic_summary(source: str, raw: bytes) -> dict[str, Any]:
    suffix = PurePosixPath(source).suffix.lower()
    text = raw.decode("utf-8", errors="strict")
    if suffix == ".json":
        payload = json.loads(text)
        if _contains_row_level_content(payload):
            raise RowLevelContentError("row-level JSON content")
        if source == "audit/artifact_manifest.json":
            return {"kind": "artifact_manifest_summary", "content": _manifest_summary(payload)}
        return {"kind": "json", "content": _sanitize(payload)}
    if suffix in {".yml", ".yaml"}:
        payload = yaml.safe_load(text)
        if _contains_row_level_content(payload):
            raise RowLevelContentError("row-level YAML content")
        return {"kind": "yaml", "content": _sanitize(payload)}
    return {"kind": "text", "content": _sanitize_text(text)}


def _sanitize(value: Any, *, depth: int = 0) -> Any:
    if depth >= 8:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for index, key in enumerate(sorted(value, key=lambda item: str(item))):
            if index >= 100:
                result["[TRUNCATED]"] = len(value) - index
                break
            rendered_key = str(key)
            normalised_key = _normalise_key(rendered_key)
            if normalised_key == "args" and isinstance(value[key], (list, tuple)):
                result[rendered_key] = {"redacted": True, "count": len(value[key])}
            else:
                result[rendered_key] = "[REDACTED]" if _is_sensitive_key(rendered_key) else _sanitize(value[key], depth=depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        items = [_sanitize(item, depth=depth + 1) for item in value[:100]]
        if len(value) > 100:
            items.append(f"[TRUNCATED {len(value) - 100} ITEMS]")
        return items
    if isinstance(value, (set, frozenset)):
        items = [_sanitize(item, depth=depth + 1) for item in value]
        return sorted(items, key=_canonical_sort_key)
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return "[NON_FINITE:NaN]"
        if math.isinf(value):
            return "[NON_FINITE:+Infinity]" if value > 0 else "[NON_FINITE:-Infinity]"
        return value
    if value is None or isinstance(value, (bool, int)):
        return value
    raise TypeError(f"unsupported context value type: {type(value).__name__}")


def _sanitize_text(value: str) -> str:
    text = value[:8192]
    for pattern in _SECRET_VALUE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    for pattern in _SAMPLE_TOKEN_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    if _SAMPLE_ID_VALUE.fullmatch(text.strip()):
        return "[REDACTED]"
    if len(value) > 8192:
        text += "\n[TRUNCATED]"
    return text


def _is_sensitive_key(key: str) -> bool:
    lowered = _normalise_key(key)
    if lowered in _SAFE_OPERATIONAL_IDS:
        return False
    if _is_aggregate_safe_key(lowered):
        return False
    tokens = set(lowered.split("_"))
    sensitive_tokens = {
        "account",
        "acct",
        "auth",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "loan",
        "order",
        "password",
        "passwd",
        "private",
        "pwd",
        "secret",
        "session",
        "token",
    }
    return bool(
        _SENSITIVE_KEYS.search(lowered)
        or tokens.intersection(sensitive_tokens)
        or "dsn" in tokens
        or ({"database"}.issubset(tokens) and bool(tokens.intersection({"url", "uri"})))
        or {"connection", "string"}.issubset(tokens)
        or lowered == "name"
        or lowered.endswith("_no")
        or lowered.endswith("_number")
        or lowered == "id"
        or lowered.endswith("_id")
    )


def _manifest_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"artifact_count": 0, "artifacts": []}
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
    rows = []
    for item in artifacts[:200]:
        if not isinstance(item, dict):
            continue
        rows.append(
            _sanitize(
                {
                    key: item.get(key)
                    for key in ("path", "kind", "source", "exists", "description", "stage")
                    if key in item
                }
            )
        )
    return {
        "artifact_count": len(artifacts),
        "artifacts": rows,
        "truncated": len(artifacts) > len(rows),
    }


def _normalise_input_path(value: str) -> str:
    raw = str(value).replace("\\", "/")
    if "\x00" in raw:
        return "[INVALID_NUL_PATH]"
    return PurePosixPath(raw).as_posix()


def _canonical_sort_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _serialized_size(pack: dict[str, Any]) -> int:
    return len(
        (json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    )


def _contains_row_level_content(value: Any, *, top_level: bool = True) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalised = _normalise_key(str(key))
            if isinstance(child, list) and any(isinstance(item, dict) for item in child):
                records = [item for item in child if isinstance(item, dict)]
                if not _is_allowed_record_container(normalised, records):
                    return True
                if any(_contains_row_level_content(record, top_level=False) for record in records):
                    return True
                # The named container and every mapping were validated above.
                # Do not feed the same list into the default-deny list branch.
                continue
            if normalised in {
                "row",
                "record",
                "sample",
                "observation",
                "prediction",
                "example",
                "case",
                "detail",
                "result",
            } and isinstance(child, dict):
                return True
            if normalised == "data" and isinstance(child, dict) and not _is_aggregate_mapping(child):
                return True
            if _contains_row_level_content(child, top_level=False):
                return True
        return False
    if isinstance(value, (list, tuple)):
        records = [item for item in value if isinstance(item, dict)]
        # A list of mappings is row-shaped unless its parent explicitly proved
        # that it is one of the bounded aggregate/config record contracts.
        if records:
            return True
        return any(_contains_row_level_content(item, top_level=False) for item in value)
    return False


def _is_allowed_record_container(key: str, records: list[dict[Any, Any]]) -> bool:
    """Fail closed for mapping lists; admit only named, validated contracts."""
    if not records:
        return True
    if key in {"metrics", "aggregate", "aggregates", "summary", "summaries", "totals"} or key.endswith(
        "_metrics"
    ):
        return all(_is_aggregate_mapping(record) for record in records)
    if key == "tasks":
        return all(_is_task_record(record) for record in records)
    if key in {"candidates", "items"}:
        return all(_is_candidate_record(record) for record in records)
    if key in {"trials", "trial_history"}:
        return all(_is_trial_record(record) for record in records)
    return False


def _is_task_record(record: dict[Any, Any]) -> bool:
    normalised = _normalised_mapping(record)
    if normalised is None:
        return False
    keys = set(normalised)
    allowed = {
        "task_id",
        "action_id",
        "type",
        "status",
        "tool_name",
        "attempt_id",
        "depends_on",
        "command",
        "outputs",
        "invocation",
        "invocation_hash",
        "derived_metadata",
        "reason",
        "message",
    }
    command = normalised.get("command")
    invocation = normalised.get("invocation")
    metadata = normalised.get("derived_metadata")
    return (
        bool(keys.intersection({"task_id", "action_id"}))
        and keys <= allowed
        and not _record_has_sample_identifier(record)
        and (command is None or _is_command_mapping(command))
        and (invocation is None or _is_invocation_mapping(invocation))
        and (metadata is None or _is_derived_metadata_mapping(metadata))
    )


def _is_candidate_record(record: dict[Any, Any]) -> bool:
    normalised = _normalised_mapping(record)
    if normalised is None:
        return False
    keys = set(normalised)
    identifying = _TUNING_PARAMETER_KEYS | {"name", "candidate_name"}
    allowed = identifying | {"params", "reason", "status"}
    params = normalised.get("params")
    return (
        bool(keys.intersection(identifying))
        and keys <= allowed
        and (params is None or _is_tuning_params_mapping(params))
        and not _record_has_sample_identifier(record)
    )


def _is_trial_record(record: dict[Any, Any]) -> bool:
    normalised = _normalised_mapping(record)
    if normalised is None:
        return False
    keys = set(normalised)
    allowed = {
        "trial_id",
        "trial_name",
        "candidate_name",
        "advisor_type",
        "status",
        "params",
        "metrics",
        "train_auc",
        "valid_auc",
        "oot_auc",
        "train_ks",
        "valid_ks",
        "oot_ks",
        "train_samples",
        "valid_samples",
        "train_bad_rate",
        "valid_bad_rate",
        "best_iteration",
        "train_time_seconds",
        "auc_gap",
        "round",
        "trial_index",
        "reason",
    }
    params = normalised.get("params")
    metrics = normalised.get("metrics")
    return (
        bool(keys.intersection({"trial_id", "trial_name", "candidate_name"}))
        and keys <= allowed
        and not _record_has_sample_identifier(record)
        and (params is None or _is_tuning_params_mapping(params))
        and (metrics is None or isinstance(metrics, dict) and _is_aggregate_mapping(metrics))
    )


def _normalised_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, Any] = {}
    for key, child in value.items():
        normalised = _normalise_key(str(key))
        if not normalised or normalised in result:
            return None
        result[normalised] = child
    return result


def _is_command_mapping(value: Any) -> bool:
    mapping = _normalised_mapping(value)
    if mapping is None or not set(mapping) <= {"executable", "args"}:
        return False
    executable = mapping.get("executable")
    if executable is not None and executable != "rmw":
        return False
    args = mapping.get("args", [])
    return isinstance(args, list) and all(
        not isinstance(item, (dict, list, tuple)) and not _looks_like_structured_payload(item)
        for item in args
    )


def _is_invocation_mapping(value: Any) -> bool:
    mapping = _normalised_mapping(value)
    if mapping is None or not set(mapping) <= {"tool_name", "params", "project", "version_id"}:
        return False
    tool_name = mapping.get("tool_name")
    if not isinstance(tool_name, str):
        return False
    params = mapping.get("params")
    return params is None or _is_action_params_mapping(params, tool_name=tool_name)


def _is_action_params_mapping(value: Any, *, tool_name: str) -> bool:
    mapping = _normalised_mapping(value)
    if mapping is None:
        return False
    try:
        from risk_model_workbench.harness.tools import get_tool_spec

        schema = get_tool_spec(tool_name).params_schema
    except (KeyError, ValueError):
        return False
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    if schema.get("additionalProperties") is False and not set(mapping) <= set(properties):
        return False
    if any(name not in mapping for name in required):
        return False
    return all(_value_matches_schema(mapping[name], properties.get(name, {})) for name in mapping)


def _is_derived_metadata_mapping(value: Any) -> bool:
    mapping = _normalised_mapping(value)
    allowed = {
        "action_id",
        "permission",
        "requires_approval",
        "allowed_for_auditor",
        "execution_semantics",
        "approval_type",
    }
    return mapping is not None and set(mapping) <= allowed and all(
        not isinstance(child, (dict, list, tuple)) for child in mapping.values()
    )


def _is_tuning_params_mapping(value: Any) -> bool:
    mapping = _normalised_mapping(value)
    return mapping is not None and set(mapping) <= _TUNING_PARAMETER_KEYS and all(
        not isinstance(child, (dict, list, tuple)) for child in mapping.values()
    )


def _looks_like_structured_payload(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if stripped.startswith(("{", "[")) or ("\n" in stripped and ":" in stripped):
        return True
    parsed_url = urlsplit(stripped)
    if parsed_url.scheme and parsed_url.netloc:
        return False
    if len(re.findall(r"(?<![A-Za-z0-9_.-])[A-Za-z_][A-Za-z0-9_.-]*\s*=", stripped)) >= 2:
        return True
    if ":" in stripped:
        try:
            parsed = yaml.safe_load(stripped)
        except yaml.YAMLError:
            return True
        if isinstance(parsed, (dict, list)):
            return True
    return False


def _value_matches_schema(value: Any, schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    expected = schema.get("type")
    if expected == "string":
        return isinstance(value, str) and not _looks_like_structured_payload(value)
    if expected == "array":
        if not isinstance(value, list):
            return False
        item_schema = schema.get("items", {})
        return all(_value_matches_schema(item, item_schema) for item in value)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict) and not _contains_row_level_content(value)
    return False


def _record_has_sample_identifier(record: dict[Any, Any]) -> bool:
    return any(_is_sample_identifier_key(str(key)) for key in record)


def _is_sample_identifier_key(key: str) -> bool:
    lowered = _normalise_key(key)
    if lowered in _SAFE_OPERATIONAL_IDS:
        return False
    if _is_aggregate_safe_key(lowered):
        return False
    tokens = set(lowered.split("_"))
    return bool(
        lowered in {"id", "name", "account", "acct", "loan", "order", "uid"}
        or lowered.endswith("_id")
        or lowered.endswith("_no")
        or lowered.endswith("_number")
        or tokens.intersection({"account", "acct", "customer", "member", "user", "loan", "order"})
    )


def _normalise_key(key: str) -> str:
    snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", snake.lower()).strip("_")


def _is_aggregate_safe_key(lowered: str) -> bool:
    return bool(
        lowered in _AGGREGATE_DIMENSIONS
        or lowered in _AGGREGATE_METRICS
        or lowered.endswith(_AGGREGATE_SUFFIXES)
        or lowered.endswith("_metrics")
    )


def _is_aggregate_mapping(value: dict[Any, Any]) -> bool:
    if not value:
        return True
    aggregate_containers = {"aggregate", "aggregates", "metrics", "summary", "totals"}
    for key, child in value.items():
        lowered = _normalise_key(str(key))
        if _is_aggregate_safe_key(lowered):
            if isinstance(child, dict) and not _is_aggregate_mapping(child):
                return False
            if isinstance(child, list) and not all(
                not isinstance(item, dict) or _is_aggregate_mapping(item) for item in child
            ):
                return False
            continue
        if lowered in aggregate_containers:
            if isinstance(child, dict) and _is_aggregate_mapping(child):
                continue
            if isinstance(child, list) and all(
                not isinstance(item, dict) or _is_aggregate_mapping(item) for item in child
            ):
                continue
        return False
    return True


def _enforce_serialized_limit(pack: dict[str, Any], max_pack_bytes: int) -> None:
    pack["context_hash"] = "0" * 64
    if _serialized_size(pack) <= max_pack_bytes:
        pack.pop("context_hash", None)
        return
    for row in reversed(pack["files"]):
        if row.get("summary"):
            kind = row["summary"].get("kind") if isinstance(row["summary"], dict) else "summary"
            row["summary"] = {"kind": kind, "truncated": True}
            row["summary_truncated"] = True
            if _serialized_size(pack) <= max_pack_bytes:
                pack.pop("context_hash", None)
                return
    while pack["files"] and _serialized_size(pack) > max_pack_bytes:
        pack["files"].pop()
        pack["truncation"]["recorded_entries"] -= 1
        pack["truncation"]["omitted_entries"] += 1
    if _serialized_size(pack) > max_pack_bytes:
        raise ValueError("max_pack_bytes is too small for context pack metadata")
    pack.pop("context_hash", None)
