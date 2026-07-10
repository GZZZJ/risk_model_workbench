"""Minimum atomic Agent workspace store contract tests."""

import os
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from risk_model_workbench.agent.workspace_store import RevisionConflictError, WorkspaceStore
from risk_model_workbench.agent.approvals import load_approvals, save_approvals
from risk_model_workbench.agent.state import load_agent_state, save_agent_state
from risk_model_workbench.harness.errors import WorkspaceLockedError
from risk_model_workbench.harness.runtime import ActionResult, write_action_result
from risk_model_workbench.registry import load_artifact_manifest
from risk_model_workbench.registry import register_artifact as registry_register_artifact
from risk_model_workbench.state import create_version_state, load_run_state, register_artifact, save_version_state


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


def test_versioned_yaml_compare_and_swap(tmp_path):
    store = WorkspaceStore(tmp_path)

    revision = store.write_yaml("audit/agent_state.yml", {"status": "draft"}, expected_revision=0)
    document = store.read_yaml("audit/agent_state.yml")

    assert revision == 1
    assert document.revision == 1
    assert document.payload == {"status": "draft"}
    assert store.write_yaml(
        "audit/agent_state.yml",
        {"status": "running"},
        expected_revision=document.revision,
    ) == 2


def test_read_yaml_rejects_falsey_non_mapping(tmp_path):
    (tmp_path / "bad.yml").write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        WorkspaceStore(tmp_path).read_yaml("bad.yml")


def test_versioned_yaml_stale_writer_cannot_lose_an_update(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.write_yaml("audit/approvals.yml", {"approvals": []}, expected_revision=0)
    first = store.read_yaml("audit/approvals.yml")
    second = store.read_yaml("audit/approvals.yml")

    store.write_yaml("audit/approvals.yml", {"approvals": [{"approval_id": "one"}]}, first.revision)

    with pytest.raises(RevisionConflictError, match="expected revision 1, found 2"):
        store.write_yaml("audit/approvals.yml", {"approvals": [{"approval_id": "two"}]}, second.revision)
    assert store.read_yaml("audit/approvals.yml").payload["approvals"] == [{"approval_id": "one"}]


def test_versioned_yaml_interrupted_replace_preserves_previous_document(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path)
    store.write_yaml("audit/agent_state.yml", {"status": "draft"}, expected_revision=0)
    previous = (tmp_path / "audit" / "agent_state.yml").read_text(encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected yaml replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected yaml"):
        store.write_yaml("audit/agent_state.yml", {"status": "running"}, expected_revision=1)

    assert (tmp_path / "audit" / "agent_state.yml").read_text(encoding="utf-8") == previous
    assert store.read_yaml("audit/agent_state.yml").revision == 1


def test_concurrent_yaml_writers_have_one_cas_winner(tmp_path):
    store = WorkspaceStore(tmp_path)
    store.write_yaml("audit/agent_state.yml", {"winner": "none"}, expected_revision=0)

    def write(name):
        try:
            WorkspaceStore(tmp_path).write_yaml(
                "audit/agent_state.yml",
                {"winner": name},
                expected_revision=1,
            )
            return "written"
        except RevisionConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(write, [f"writer-{index}" for index in range(8)]))

    assert outcomes.count("written") == 1
    assert outcomes.count("conflict") == 7
    assert store.read_yaml("audit/agent_state.yml").revision == 2


def test_append_jsonl_serializes_complete_events_and_fsyncs(tmp_path, monkeypatch):
    fsync_calls = []
    real_fsync = os.fsync

    def tracked_fsync(descriptor):
        fsync_calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", tracked_fsync)

    def append(index):
        WorkspaceStore(tmp_path).append_jsonl("audit/agent_trace.jsonl", {"sequence": index})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append, range(40)))

    rows = [json.loads(line) for line in (tmp_path / "audit" / "agent_trace.jsonl").read_text().splitlines()]
    assert sorted(row["sequence"] for row in rows) == list(range(40))
    assert len(fsync_calls) >= 40


def test_lock_alias_uses_single_runner_lock(tmp_path):
    first = WorkspaceStore(tmp_path)
    second = WorkspaceStore(tmp_path)

    with first.lock():
        with pytest.raises(WorkspaceLockedError):
            with second.lock():
                pass


def test_agent_state_and_approval_documents_reject_stale_saves(tmp_path):
    save_agent_state(tmp_path, {"version": 2, "status": "draft", "tasks": []})
    first_state = load_agent_state(tmp_path)
    stale_state = load_agent_state(tmp_path)
    first_state["status"] = "running"
    save_agent_state(tmp_path, first_state)
    stale_state["status"] = "failed"
    with pytest.raises(RevisionConflictError):
        save_agent_state(tmp_path, stale_state)

    approvals = load_approvals(tmp_path)
    approvals["approvals"] = [{"approval_id": "approval-one"}]
    save_approvals(tmp_path, approvals)
    first_approvals = load_approvals(tmp_path)
    stale_approvals = load_approvals(tmp_path)
    first_approvals["approvals"].append({"approval_id": "approval-two"})
    save_approvals(tmp_path, first_approvals)
    stale_approvals["approvals"].append({"approval_id": "approval-stale"})
    with pytest.raises(RevisionConflictError):
        save_approvals(tmp_path, stale_approvals)


def test_artifact_registration_binds_manifest_and_state_transaction(tmp_path):
    state = create_version_state(tmp_path.parent, version_id=tmp_path.name, workflow="full_modeling")
    save_version_state(tmp_path, state)
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}\n", encoding="utf-8")

    register_artifact(tmp_path, "validate_config", artifact)

    current_state = load_run_state(tmp_path)
    manifest = load_artifact_manifest(tmp_path)
    assert current_state["transaction_id"].startswith("txn_")
    assert manifest["transaction_id"] == current_state["transaction_id"]


def test_concurrent_manifest_registration_preserves_every_entry(tmp_path):
    (tmp_path / "audit").mkdir()

    def register(index):
        artifact = tmp_path / f"artifact-{index}.json"
        artifact.write_text("{}\n", encoding="utf-8")
        registry_register_artifact(tmp_path, artifact, stage="agent_runtime")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(register, range(40)))

    manifest = load_artifact_manifest(tmp_path)
    assert {item["path"] for item in manifest["artifacts"]} == {
        f"artifact-{index}.json" for index in range(40)
    }


def test_direct_manifest_mutation_breaks_old_state_transaction_binding(tmp_path):
    state = create_version_state(tmp_path.parent, version_id=tmp_path.name, workflow="full_modeling")
    save_version_state(tmp_path, state)
    paired = tmp_path / "paired.json"
    paired.write_text("{}\n", encoding="utf-8")
    register_artifact(tmp_path, "validate_config", paired)
    bound_state = load_run_state(tmp_path)
    bound_transaction = bound_state["transaction_id"]
    assert load_artifact_manifest(tmp_path)["transaction_id"] == bound_transaction

    write_action_result(
        tmp_path,
        ActionResult(
            attempt_id="attempt-direct-manifest",
            task_id="sample_check_001",
            action_id="sample_check",
            invocation_hash="a" * 64,
            project=str(tmp_path.parent),
            version_id=tmp_path.name,
            status="done",
            created_at=datetime.now().isoformat(timespec="seconds"),
        ),
    )

    assert load_run_state(tmp_path)["transaction_id"] == bound_transaction
    assert load_artifact_manifest(tmp_path)["transaction_id"] != bound_transaction
