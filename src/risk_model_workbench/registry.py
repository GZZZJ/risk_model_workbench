"""Artifact registry helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
from typing import Any
from uuid import uuid4

from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.manifest import describe_file


STORAGE_CLASSES = {"repository", "workspace_only", "local_only", "external", "legacy_workspace"}
CONTRACT_ROLES = {"required", "optional"}


def manifest_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "audit" / "artifact_manifest.json"


def load_artifact_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = manifest_path(run_dir)
    if not path.exists():
        return {"version": 1, "artifacts": []}
    return WorkspaceStore(run_dir).read_json("audit/artifact_manifest.json").payload


def save_artifact_manifest(
    run_dir: str | Path,
    manifest: dict[str, Any],
    *,
    transaction_id: str = "",
) -> Path:
    path = manifest_path(run_dir)
    manifest.setdefault("version", 1)
    manifest.setdefault("artifacts", [])
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["transaction_id"] = transaction_id or _new_unpaired_transaction_id()
    WorkspaceStore(run_dir).update_json("audit/artifact_manifest.json", lambda _current: dict(manifest))
    return path


def register_artifact(
    run_dir: str | Path,
    artifact: str | Path,
    *,
    stage: str,
    kind: str = "file",
    source: str = "generated",
    description: str = "",
    transaction_id: str = "",
    storage_class: str | None = None,
    contract_role: str = "required",
    retention_reason: str = "",
    regeneration: str = "",
    external_reference: str = "",
    integrity_mode: str = "content",
) -> dict[str, Any]:
    """Register an artifact relative to the run directory when possible."""
    run_path = Path(run_dir).resolve()
    artifact_path = Path(artifact)
    if not artifact_path.is_absolute():
        artifact_path = run_path / artifact_path
    artifact_path = artifact_path.resolve()
    try:
        artifact_path.relative_to(run_path)
    except ValueError as exc:
        raise ValueError("artifact path must stay inside the version workspace") from exc

    if contract_role not in CONTRACT_ROLES:
        raise ValueError(f"unknown artifact contract_role: {contract_role}")
    if storage_class is not None and storage_class not in STORAGE_CLASSES - {"legacy_workspace"}:
        raise ValueError(f"unknown artifact storage_class: {storage_class}")
    if integrity_mode not in {"content", "mutable"}:
        raise ValueError(f"unknown artifact integrity_mode: {integrity_mode}")
    ignored = _git_path_is_ignored(run_path, artifact_path)
    if ignored and storage_class is None:
        raise ValueError("ignored artifact requires an explicit storage_class")
    resolved_storage = storage_class or "workspace_only"
    if resolved_storage == "local_only" and contract_role == "required":
        raise ValueError("local_only artifact cannot have required contract_role")
    if resolved_storage == "local_only" and not retention_reason:
        raise ValueError("local_only artifact requires retention_reason")
    if resolved_storage == "external" and not external_reference:
        raise ValueError("external artifact requires external_reference")
    repository_tracked = resolved_storage == "repository" and _git_path_is_tracked(run_path, artifact_path)
    if resolved_storage == "repository" and not repository_tracked:
        resolved_storage = "workspace_only"

    if artifact_path.exists() and artifact_path.is_file():
        entry = describe_file(artifact_path, run_path)
    else:
        try:
            display_path = artifact_path.relative_to(run_path)
        except ValueError:
            display_path = artifact_path
        entry = {"path": str(display_path), "exists": artifact_path.exists()}

    entry.update(
        {
            "stage": stage,
            "kind": kind,
            "source": source,
            "description": description,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "storage_class": resolved_storage,
            "contract_role": contract_role,
            "retention_reason": retention_reason or _default_retention_reason(resolved_storage),
            "integrity_mode": integrity_mode,
        }
    )
    if regeneration:
        entry["regeneration"] = regeneration
    if external_reference:
        entry["external_reference"] = external_reference
    if resolved_storage == "repository":
        entry["repository_tracked"] = repository_tracked
    if integrity_mode == "mutable" and not _valid_mutable_integrity(entry, resolved_storage):
        raise ValueError("mutable integrity is limited to approved workspace audit artifacts")

    resolved_transaction_id = transaction_id or _new_unpaired_transaction_id()

    def update_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
        manifest.setdefault("version", 1)
        artifacts = [
            item
            for item in manifest.get("artifacts", [])
            if not (item.get("path") == entry["path"] and item.get("stage") == stage)
        ]
        artifacts.append(entry)
        manifest["artifacts"] = artifacts
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["transaction_id"] = resolved_transaction_id
        return manifest

    WorkspaceStore(run_path).update_json("audit/artifact_manifest.json", update_manifest)
    return entry


def audit_artifact_availability(run_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Separate local execution availability from clean-clone reproducibility."""
    run_path = Path(run_dir).resolve()
    summary = {
        "repository_present": 0,
        "workspace_only_present": 0,
        "local_only_present": 0,
        "local_only_missing": 0,
        "external_references": 0,
        "legacy_workspace": 0,
        "required_missing": 0,
    }
    warnings: list[str] = []
    issues: list[str] = []
    entries: list[dict[str, Any]] = []
    raw_artifacts = manifest.get("artifacts", []) or []
    latest_registration_by_path: dict[str, str] = {}
    for raw in raw_artifacts:
        artifact_path = str(raw.get("path") or "")
        registered_at = str(raw.get("registered_at") or "")
        if registered_at >= latest_registration_by_path.get(artifact_path, ""):
            latest_registration_by_path[artifact_path] = registered_at
    for raw in raw_artifacts:
        entry = dict(raw)
        storage = str(entry.get("storage_class") or "legacy_workspace")
        role = str(entry.get("contract_role") or "required")
        entry["storage_class"] = storage
        entry["contract_role"] = role
        path, path_error = _artifact_absolute_path(run_path, entry)
        present = path is not None and _artifact_is_present(path, entry)
        if path_error:
            issues.append(path_error)
        if role not in CONTRACT_ROLES:
            issues.append(f"unknown contract_role for {entry.get('path')}: {role}")
        metadata_errors = _retention_metadata_errors(entry, storage)
        issues.extend(metadata_errors)
        is_latest_registration = str(entry.get("registered_at") or "") == latest_registration_by_path.get(
            str(entry.get("path") or ""), ""
        )
        if present and path is not None and is_latest_registration:
            issues.extend(_integrity_errors(path, entry))
        if storage == "legacy_workspace":
            summary["legacy_workspace"] += 1
            warnings.append(f"legacy artifact has no retention metadata: {entry.get('path')}")
        elif storage == "repository":
            tracked = path is not None and _git_path_is_tracked(run_path, path)
            if present and tracked:
                summary["repository_present"] += 1
            elif present:
                summary["workspace_only_present"] += 1
                warnings.append(f"repository artifact is not tracked; treated as workspace_only: {entry.get('path')}")
            elif role == "required":
                summary["required_missing"] += 1
        elif storage == "workspace_only":
            tracked = path is not None and _git_path_is_tracked(run_path, path)
            if present and tracked:
                summary["repository_present"] += 1
            elif present:
                summary["workspace_only_present"] += 1
            elif role == "required":
                summary["required_missing"] += 1
        elif storage == "local_only":
            if role == "required":
                issues.append(f"required artifact cannot be local_only: {entry.get('path')}")
                if not present:
                    summary["required_missing"] += 1
            elif present:
                summary["local_only_present"] += 1
            else:
                summary["local_only_missing"] += 1
                warnings.append(f"optional local_only artifact unavailable: {entry.get('path')}")
        elif storage == "external":
            summary["external_references"] += 1
            if role == "required":
                issues.append(f"external artifact is not local reproduction evidence: {entry.get('path')}")
        else:
            issues.append(f"unknown storage_class for {entry.get('path')}: {storage}")
        if role == "required" and not present and storage not in {"external", "local_only"}:
            issue = f"required artifact unavailable: {entry.get('path')}"
            if issue not in issues:
                issues.append(issue)
        entries.append(entry)

    execution_verdict = "incomplete" if summary["required_missing"] or issues else "complete"
    if execution_verdict == "incomplete":
        reproducibility_verdict = "incomplete"
    elif summary["workspace_only_present"] or summary["legacy_workspace"]:
        reproducibility_verdict = "workspace_dependent"
    elif summary["local_only_missing"] or summary["external_references"]:
        reproducibility_verdict = "complete_with_warnings"
    else:
        reproducibility_verdict = "complete"
    return {
        "availability_summary": summary,
        "execution_verdict": execution_verdict,
        "reproducibility_verdict": reproducibility_verdict,
        "warnings": warnings,
        "issues": issues,
        "artifacts": entries,
    }


def _artifact_absolute_path(run_path: Path, entry: dict[str, Any]) -> tuple[Path | None, str]:
    path = Path(str(entry.get("path") or ""))
    candidate = (path if path.is_absolute() else run_path / path).resolve()
    try:
        candidate.relative_to(run_path)
    except ValueError:
        return None, f"artifact path escapes version workspace: {entry.get('path')}"
    return candidate, ""


def _artifact_is_present(path: Path, entry: dict[str, Any]) -> bool:
    if entry.get("kind") == "directory":
        return path.is_dir()
    return path.is_file()


def _retention_metadata_errors(entry: dict[str, Any], storage: str) -> list[str]:
    if storage == "legacy_workspace":
        return []
    errors: list[str] = []
    if storage not in STORAGE_CLASSES:
        errors.append(f"unknown storage_class for {entry.get('path')}: {storage}")
        return errors
    if not str(entry.get("retention_reason") or "").strip():
        errors.append(f"retention_reason missing: {entry.get('path')}")
    integrity_mode = str(entry.get("integrity_mode") or "content")
    if integrity_mode not in {"content", "mutable"}:
        errors.append(f"unknown integrity_mode for {entry.get('path')}: {integrity_mode}")
    elif integrity_mode == "mutable" and not _valid_mutable_integrity(entry, storage):
        errors.append(f"invalid mutable integrity scope: {entry.get('path')}")
    if storage in {"repository", "workspace_only", "local_only"} and entry.get("kind") != "directory":
        sha256 = str(entry.get("sha256") or "")
        if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
            errors.append(f"artifact sha256 invalid: {entry.get('path')}")
        size = entry.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            errors.append(f"artifact size_bytes invalid: {entry.get('path')}")
    if storage == "local_only":
        if not (entry.get("regeneration") or entry.get("source")):
            errors.append(f"local_only provenance missing: {entry.get('path')}")
    if storage == "external" and not entry.get("external_reference"):
        errors.append(f"external_reference missing: {entry.get('path')}")
    return errors


def _integrity_errors(path: Path, entry: dict[str, Any]) -> list[str]:
    if not path.is_file():
        return []
    storage = str(entry.get("storage_class") or "legacy_workspace")
    if entry.get("integrity_mode") == "mutable" and _valid_mutable_integrity(entry, storage):
        return []
    errors: list[str] = []
    expected_size = entry.get("size_bytes")
    if isinstance(expected_size, int) and path.stat().st_size != expected_size:
        errors.append(f"artifact size mismatch: {entry.get('path')}")
    expected_hash = str(entry.get("sha256") or "")
    if expected_hash:
        from risk_model_workbench.manifest import sha256_file

        if sha256_file(path) != expected_hash:
            errors.append(f"artifact sha256 mismatch: {entry.get('path')}")
    return errors


def _valid_mutable_integrity(entry: dict[str, Any], storage: str) -> bool:
    return bool(
        storage == "workspace_only"
        and entry.get("kind") == "audit"
        and str(entry.get("path") or "") in {"audit/agent_state.yml", "audit/agent_trace.jsonl"}
    )


def _git_root(path: Path) -> Path | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        return None
    return Path(completed.stdout.strip()).resolve()


def _git_path_is_tracked(run_path: Path, artifact_path: Path) -> bool:
    root = _git_root(run_path)
    if root is None:
        return False
    try:
        relative = artifact_path.resolve().relative_to(root)
    except ValueError:
        return False
    return subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", str(relative)],
        capture_output=True,
        check=False,
    ).returncode == 0


def _git_path_is_ignored(run_path: Path, artifact_path: Path) -> bool:
    root = _git_root(run_path)
    if root is None:
        return False
    try:
        relative = artifact_path.resolve().relative_to(root)
    except ValueError:
        return False
    return subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", "--", str(relative)],
        capture_output=True,
        check=False,
    ).returncode == 0


def _default_retention_reason(storage_class: str) -> str:
    return {
        "repository": "tracked repository artifact",
        "workspace_only": "generated in the current workspace",
        "local_only": "intentionally retained outside version control",
        "external": "retained by an external system",
    }.get(storage_class, "legacy workspace artifact")


def _new_unpaired_transaction_id() -> str:
    """Identify a manifest mutation that has no matching version-state write."""
    return f"txn_unpaired_{uuid4().hex}"
