"""Minimum atomic storage primitives for one local Agent workspace."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from risk_model_workbench.harness.errors import WorkspaceLockedError


class WorkspaceStore:
    """P0 single-workspace atomic writes, exclusive creates, and runner lock."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()

    def atomic_write(self, relative_path: str | Path, payload: dict[str, Any]) -> Path:
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
            os.replace(temporary_path, target)
            _fsync_directory(target.parent)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return target

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
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise WorkspaceLockedError(f"agent workspace already has an active runner: {lock}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_directory(lock.parent)
            yield lock
        finally:
            lock.unlink(missing_ok=True)
            _fsync_directory(lock.parent)

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
