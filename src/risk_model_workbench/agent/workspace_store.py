"""Minimum atomic storage primitives for one local Agent workspace."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml

from risk_model_workbench.harness.errors import WorkspaceLockedError


_REVISION_KEY = "_workspace_revision"


class RevisionConflictError(RuntimeError):
    """Raised when a compare-and-swap write observes a stale revision."""


@dataclass(frozen=True)
class VersionedDocument:
    """A YAML payload paired with the revision observed on disk."""

    payload: dict[str, Any]
    revision: int


class VersionedPayload(dict[str, Any]):
    """A normal mapping that remembers the revision it was loaded from."""

    def __init__(self, payload: dict[str, Any], revision: int):
        super().__init__(payload)
        self.store_revision = revision


yaml.SafeDumper.add_representer(VersionedPayload, yaml.representer.SafeRepresenter.represent_dict)


def tracked_payload(document: VersionedDocument) -> VersionedPayload:
    return VersionedPayload(document.payload, document.revision)


class WorkspaceStore:
    """P0 single-workspace atomic writes, exclusive creates, and runner lock."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    def atomic_write(self, relative_path: str | Path, payload: dict[str, Any]) -> Path:
        target = self._target(relative_path)
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._atomic_write_text(target, content)
        return target

    def read_yaml(self, relative_path: str | Path) -> VersionedDocument:
        """Read one YAML mapping and its embedded CAS revision.

        Documents written before the P1 store migration have revision zero and
        are upgraded on their next successful write.
        """
        target = self._target(relative_path)
        if not target.exists():
            return VersionedDocument(payload={}, revision=0)
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"workspace YAML document must be a mapping: {relative_path}")
        payload = dict(raw)
        try:
            revision = int(payload.pop(_REVISION_KEY, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid workspace revision: {relative_path}") from exc
        if revision < 0:
            raise ValueError(f"invalid workspace revision: {relative_path}")
        return VersionedDocument(payload=payload, revision=revision)

    def write_yaml(
        self,
        relative_path: str | Path,
        payload: dict[str, Any],
        expected_revision: int,
    ) -> int:
        """Atomically replace YAML if ``expected_revision`` still matches."""
        target = self._target(relative_path)
        if not isinstance(payload, dict):
            raise TypeError("workspace YAML payload must be a mapping")
        clean_payload = dict(payload)
        clean_payload.pop(_REVISION_KEY, None)
        with self._document_lock(target):
            current = self.read_yaml(relative_path)
            if current.revision != int(expected_revision):
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current.revision}: {relative_path}"
                )
            next_revision = current.revision + 1
            document = clean_payload
            document[_REVISION_KEY] = next_revision
            content = yaml.safe_dump(document, allow_unicode=True, sort_keys=False)
            self._atomic_write_text(target, content)
            return next_revision

    def read_json(self, relative_path: str | Path) -> VersionedDocument:
        """Read a legacy or revisioned JSON mapping."""
        target = self._target(relative_path)
        if not target.exists():
            return VersionedDocument(payload={}, revision=0)
        raw = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"workspace JSON document must be a mapping: {relative_path}")
        payload = dict(raw)
        try:
            revision = int(payload.pop(_REVISION_KEY, 0) or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid workspace revision: {relative_path}") from exc
        if revision < 0:
            raise ValueError(f"invalid workspace revision: {relative_path}")
        return VersionedDocument(payload=payload, revision=revision)

    def write_json(
        self,
        relative_path: str | Path,
        payload: dict[str, Any],
        expected_revision: int,
    ) -> int:
        """Atomically replace JSON if ``expected_revision`` still matches."""
        target = self._target(relative_path)
        with self._document_lock(target):
            current = self.read_json(relative_path)
            if current.revision != int(expected_revision):
                raise RevisionConflictError(
                    f"expected revision {expected_revision}, found {current.revision}: {relative_path}"
                )
            return self._write_revisioned_json(target, payload, current.revision + 1)

    def update_json(
        self,
        relative_path: str | Path,
        updater: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> VersionedDocument:
        """Run one locked JSON read-modify-write without losing peer updates."""
        target = self._target(relative_path)
        with self._document_lock(target):
            current = self.read_json(relative_path)
            updated = updater(dict(current.payload))
            if not isinstance(updated, dict):
                raise TypeError("workspace JSON updater must return a mapping")
            revision = current.revision + 1
            self._write_revisioned_json(target, updated, revision)
            clean = dict(updated)
            clean.pop(_REVISION_KEY, None)
            return VersionedDocument(payload=clean, revision=revision)

    def append_jsonl(self, relative_path: str | Path, event: dict[str, Any]) -> None:
        """Append one complete, durable JSON event under a per-document lock."""
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":")) + "\n"
        with self._document_lock(target):
            created = not target.exists()
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                encoded = line.encode("utf-8")
                written = 0
                while written < len(encoded):
                    count = os.write(descriptor, encoded[written:])
                    if count <= 0:
                        raise OSError(f"short append while writing {relative_path}")
                    written += count
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if created:
                _fsync_directory(target.parent)

    def create_once(self, relative_path: str | Path, payload: dict[str, Any]) -> bool:
        """Publish complete JSON once; the final path is never partially visible."""
        target = self._target(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                return False
            except OSError:
                # hard links not supported on this filesystem; fall back to rename
                os.rename(temporary_path, target)
            _fsync_directory(target.parent)
            return True
        finally:
            temporary_path.unlink(missing_ok=True)

    def path(self, relative_path: str | Path) -> Path:
        """Resolve one workspace-contained path without creating it."""
        return self._target(relative_path)

    @contextmanager
    def runner_lock(self) -> Iterator[Path]:
        lock = self._target("audit/agent_runner.lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WorkspaceLockedError(f"agent workspace already has an active runner: {lock}") from exc
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
            os.fsync(fd)
            _fsync_directory(lock.parent)
            yield lock
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def lock(self) -> Iterator[Path]:
        """Public P1 name for the workspace single-runner lock."""
        return self.runner_lock()

    @contextmanager
    def _document_lock(self, target: Path) -> Iterator[None]:
        target.parent.mkdir(parents=True, exist_ok=True)
        lock_path = target.parent / f".{target.name}.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _atomic_write_text(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        temporary_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            _fsync_directory(target.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _write_revisioned_json(self, target: Path, payload: dict[str, Any], revision: int) -> int:
        document = dict(payload)
        document.pop(_REVISION_KEY, None)
        document[_REVISION_KEY] = revision
        content = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
        self._atomic_write_text(target, content)
        return revision

    def _target(self, relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            target = (self.workspace / candidate).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            raise ValueError(f"workspace path escape: {relative_path}")
        return target


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
