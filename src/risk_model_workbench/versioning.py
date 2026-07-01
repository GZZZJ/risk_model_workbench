"""Version workspace helpers.

Versions are the user-facing modeling workspaces. Legacy runs remain readable
for compatibility and can be migrated into versions.
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from risk_model_workbench.paths import project_config_path
from risk_model_workbench.state import (
    load_run_state,
    run_dir,
    save_version_state,
    version_dir,
)


VERSION_INDEX_VERSION = 1
VERSION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[0-9]{8}$|^[a-z][a-z0-9_]*$")
STRICT_VERSION_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_v[0-9]+_[0-9]{8}$")
SOURCE_TYPES = {"workbench", "imported", "manual"}


def versions_dir(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / "versions"


def version_index_path(project_dir: str | Path) -> Path:
    return versions_dir(project_dir) / "index.yml"


def load_version_index(project_dir: str | Path) -> dict[str, Any]:
    path = version_index_path(project_dir)
    if not path.exists():
        return {"version": VERSION_INDEX_VERSION, "versions": []}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    payload.setdefault("version", VERSION_INDEX_VERSION)
    payload.setdefault("versions", [])
    return payload


def save_version_index(project_dir: str | Path, index: dict[str, Any]) -> Path:
    path = version_index_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    index["version"] = VERSION_INDEX_VERSION
    index.setdefault("versions", [])
    index["updated_at"] = _now()
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(index, handle, allow_unicode=True, sort_keys=False)
    return path


def validate_version_id(version_id: str, *, strict: bool = False) -> None:
    pattern = STRICT_VERSION_ID_PATTERN if strict else VERSION_ID_PATTERN
    if not pattern.fullmatch(version_id):
        expected = "<project_key>_v<version_number>_<yyyymmdd>" if strict else "lowercase snake_case"
        raise ValueError(f"invalid version_id: {version_id}; expected {expected}")


def upsert_version_index(project_dir: str | Path, entry: dict[str, Any], *, active: bool = False) -> Path:
    index = load_version_index(project_dir)
    version_id = str(entry["version_id"])
    versions = [item for item in index.get("versions", []) if item.get("version_id") != version_id]
    versions.append(entry)
    versions.sort(key=lambda item: str(item.get("version_id", "")))
    index["versions"] = versions
    if active or not index.get("active_version_id"):
        index["active_version_id"] = version_id
    return save_version_index(project_dir, index)


def list_versions(project_dir: str | Path) -> list[dict[str, Any]]:
    return list(load_version_index(project_dir).get("versions", []) or [])


def version_exists(project_dir: str | Path, version_id: str) -> bool:
    return (version_dir(project_dir, version_id) / "version_state.yml").exists()


def resolve_workspace_dir(
    project_dir: str | Path,
    *,
    version_id: str | None = None,
    run_id: str | None = None,
) -> Path:
    project_path = Path(project_dir)
    if version_id:
        return version_dir(project_path, version_id)
    if not run_id:
        raise ValueError("version_id or run_id is required")
    direct_version = version_dir(project_path, run_id)
    if (direct_version / "version_state.yml").exists():
        return direct_version
    lineage_version = find_version_by_legacy_run_id(project_path, run_id)
    if lineage_version:
        return version_dir(project_path, lineage_version)
    return run_dir(project_path, run_id)


def find_version_by_legacy_run_id(project_dir: str | Path, run_id: str) -> str | None:
    for entry in list_versions(project_dir):
        if str(entry.get("legacy_run_id") or "") == run_id:
            return str(entry.get("version_id"))
    for state_path in versions_dir(project_dir).glob("*/version_state.yml"):
        try:
            state = yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            continue
        lineage = state.get("lineage") if isinstance(state.get("lineage"), dict) else {}
        if str(lineage.get("legacy_run_id") or "") == run_id:
            return str(state.get("version_id") or state_path.parent.name)
    return None


def standard_run_dirs(project_dir: str | Path) -> list[Path]:
    runs_path = Path(project_dir) / "runs"
    if not runs_path.exists():
        return []
    candidates = []
    for item in sorted(runs_path.iterdir()):
        if not item.is_dir():
            continue
        if (item / "run_state.yml").exists() and (item / "audit" / "artifact_manifest.json").exists():
            candidates.append(item)
    return candidates


def migrate_run_to_version(
    project_dir: str | Path,
    *,
    run_id: str,
    version_id: str,
    source_type: str | None = None,
    display_name: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    validate_version_id(version_id)
    if source_type is not None and source_type not in SOURCE_TYPES:
        raise ValueError(f"invalid source_type: {source_type}")

    project_path = Path(project_dir)
    source = run_dir(project_path, run_id)
    if not (source / "run_state.yml").exists():
        raise FileNotFoundError(f"legacy run_state.yml not found: {source / 'run_state.yml'}")
    if not (source / "audit" / "artifact_manifest.json").exists():
        raise FileNotFoundError(f"legacy artifact_manifest.json not found: {source / 'audit' / 'artifact_manifest.json'}")

    target = version_dir(project_path, version_id)
    if target.exists():
        existing = _load_version_state(target)
        lineage = existing.get("lineage") if isinstance(existing.get("lineage"), dict) else {}
        if not force and lineage.get("legacy_run_id") == run_id:
            entry = _index_entry(project_path, target, existing, display_name=display_name)
            upsert_version_index(project_path, entry)
            return {"status": "exists", "version_id": version_id, "path": str(target), "entry": entry}
        if not force:
            raise FileExistsError(f"version already exists: {target}")
        shutil.rmtree(target)

    shutil.copytree(source, target)
    legacy_state_path = target / "run_state.yml"
    legacy_state = yaml.safe_load(legacy_state_path.read_text(encoding="utf-8")) or {}
    inferred_source_type = source_type or ("imported" if str(legacy_state.get("status", "")).startswith("imported") or str(legacy_state.get("workflow", "")).startswith("imported") else "workbench")
    version_state = _run_state_to_version_state(project_path, version_id, run_id, legacy_state, inferred_source_type)
    save_version_state(target, version_state)
    legacy_state_path.unlink()

    entry = _index_entry(project_path, target, version_state, display_name=display_name)
    upsert_version_index(project_path, entry, active=version_state.get("status") in {"done", "released"})
    return {"status": "migrated", "version_id": version_id, "path": str(target), "entry": entry}


def suggest_version_id(project_dir: str | Path, run_id: str) -> str:
    project_key = _project_key(Path(project_dir))
    if run_id == "2026-06-30-gcard-main-lgbm-full":
        return f"{project_key}_v7_20260630"
    date = _date_from_run_id(run_id)
    slug = _slugify(run_id)
    return f"{project_key}_{slug}_{date}" if date else f"{project_key}_{slug}"


def _run_state_to_version_state(
    project_dir: Path,
    version_id: str,
    run_id: str,
    run_state: dict[str, Any],
    source_type: str,
) -> dict[str, Any]:
    state = dict(run_state)
    state.pop("run_id", None)
    state["version_id"] = version_id
    state["project"] = str(project_dir.resolve())
    state["source_type"] = source_type
    state["lineage"] = {
        "legacy_run_id": run_id,
        "migrated_from": f"runs/{run_id}",
        "migrated_at": _now(),
    }
    return state


def _index_entry(project_dir: Path, version_path: Path, state: dict[str, Any], *, display_name: str | None = None) -> dict[str, Any]:
    version_id = str(state.get("version_id") or version_path.name)
    lineage = state.get("lineage") if isinstance(state.get("lineage"), dict) else {}
    entry = {
        "version_id": version_id,
        "display_name": display_name or version_id,
        "source_type": state.get("source_type", "workbench"),
        "status": state.get("status", ""),
        "workflow": state.get("workflow", ""),
        "path": str(version_path.relative_to(project_dir)),
        "created_at": state.get("created_at", ""),
        "updated_at": state.get("updated_at", ""),
    }
    if lineage.get("legacy_run_id"):
        entry["legacy_run_id"] = lineage["legacy_run_id"]
    return entry


def _load_version_state(path: Path) -> dict[str, Any]:
    state_path = path / "version_state.yml"
    if not state_path.exists():
        return {}
    return yaml.safe_load(state_path.read_text(encoding="utf-8")) or {}


def _project_key(project_dir: Path) -> str:
    config_path = project_config_path(project_dir)
    if config_path.exists():
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            config = {}
        project = config.get("project") if isinstance(config.get("project"), dict) else {}
        explicit = project.get("project_key") or project.get("model_key")
        if explicit:
            return _slugify(str(explicit))
        display_name = str(project.get("display_name") or "")
        if "复借" in display_name and "G" in display_name.upper():
            return "fujie_gcard"
    return _slugify(project_dir.name)


def _date_from_run_id(run_id: str) -> str:
    match = re.search(r"(20[0-9]{2})[-_]?([01][0-9])[-_]?([0-3][0-9])", run_id)
    if not match:
        return ""
    return "".join(match.groups())


def _slugify(value: str) -> str:
    text = value.strip().lower().replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "version"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
