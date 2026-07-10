"""Minimum atomic Agent workspace store contract tests."""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from risk_model_workbench.agent.workspace_store import WorkspaceStore
from risk_model_workbench.harness.errors import WorkspaceLockedError


def test_create_once_rejects_a_second_consumer(tmp_path):
    store = WorkspaceStore(tmp_path)
    target = "audit/advisor_consumptions/request-001.json"

    assert store.create_once(target, {"request_id": "request-001"}) is True
    assert store.create_once(target, {"request_id": "request-001"}) is False


def test_concurrent_consumers_have_exactly_one_winner(tmp_path):
    target = "audit/approval_consumptions/approval-001.json"

    def consume(_index):
        return WorkspaceStore(tmp_path).create_once(target, {"approval_id": "approval-001"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(consume, range(8)))

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 7
    assert json.loads((tmp_path / target).read_text(encoding="utf-8")) == {"approval_id": "approval-001"}


def test_create_once_publishes_only_after_complete_temp_file(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path)
    target = tmp_path / "audit" / "receipt.json"
    real_link = os.link

    def inspect_before_publish(source, destination):
        assert Path(destination) == target
        assert target.exists() is False
        assert json.loads(Path(source).read_text(encoding="utf-8")) == {"complete": True}
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", inspect_before_publish)
    assert store.create_once("audit/receipt.json", {"complete": True}) is True


def test_create_once_interruption_leaves_no_target_or_temp(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path)
    target = tmp_path / "audit" / "receipt.json"

    def fail_link(_source, _destination):
        raise OSError("injected publish failure")

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        store.create_once("audit/receipt.json", {"complete": True})

    assert target.exists() is False
    assert list(target.parent.glob(".receipt.json.*.tmp")) == []


def test_atomic_write_uses_replace_and_fsyncs_file_and_parent(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path)
    replace_calls = []
    fsync_calls = []
    real_replace = os.replace
    real_fsync = os.fsync

    def tracked_replace(source, target):
        replace_calls.append((source, target))
        return real_replace(source, target)

    def tracked_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "replace", tracked_replace)
    monkeypatch.setattr(os, "fsync", tracked_fsync)

    path = store.atomic_write("audit/state.json", {"revision": 1})

    assert path.exists()
    assert len(replace_calls) == 1
    assert len(fsync_calls) >= 2


def test_interrupted_replace_preserves_previous_valid_document(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path)
    path = store.atomic_write("audit/state.json", {"revision": 1})
    previous = path.read_text(encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.atomic_write("audit/state.json", {"revision": 2})

    assert path.read_text(encoding="utf-8") == previous
    assert list(path.parent.glob(".state.json.*.tmp")) == []


def test_runner_lock_is_exclusive_and_released(tmp_path):
    first = WorkspaceStore(tmp_path)
    second = WorkspaceStore(tmp_path)

    with first.runner_lock():
        with pytest.raises(WorkspaceLockedError):
            with second.runner_lock():
                pass
    with second.runner_lock():
        pass


def test_workspace_path_escape_is_rejected(tmp_path):
    store = WorkspaceStore(tmp_path / "workspace")
    with pytest.raises(ValueError, match="path escape"):
        store.atomic_write("../outside.json", {"unsafe": True})
