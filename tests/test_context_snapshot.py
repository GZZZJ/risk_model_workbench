import json

import pytest

from risk_model_workbench.agent.context_pack import build_context_pack, persist_context_pack
from risk_model_workbench.context_snapshot import attach_context_pack_reference


def test_context_snapshot_context_pack_reference_is_verified_and_deterministic(tmp_path):
    snapshot = {"version": 1, "generated_at": "volatile", "sources": ["version_state.yml"]}
    (tmp_path / "version_state.yml").write_text("status: running\n", encoding="utf-8")
    pack = build_context_pack(
        tmp_path,
        project="demo",
        version_id="v1",
        task_id="task",
        attempt_id="attempt",
        request_type="failure_diagnosis_required",
        paths=["version_state.yml"],
        constraints=[],
        allowed_tools=[],
        output_contract={"type": "failure_diagnosis"},
    )
    relative = persist_context_pack(tmp_path, pack)
    first = attach_context_pack_reference(snapshot, tmp_path, relative, pack["context_hash"])
    second = attach_context_pack_reference(snapshot, tmp_path, relative, pack["context_hash"])

    assert first == second
    assert first["host_agent_context"] == {
        "context_pack": relative,
        "context_hash": pack["context_hash"],
    }
    assert "volatile" in json.dumps(first)


def test_context_snapshot_rejects_invalid_context_pack_references(tmp_path):
    for path, digest in [
        ("/tmp/pack.json", "a" * 64),
        ("../pack.json", "a" * 64),
        ("audit/other/pack.json", "a" * 64),
        ("audit/context_packs/not-a-hash.json", "not-a-hash"),
        (f"audit/context_packs/{'a' * 64}.json", "a" * 64),
    ]:
        with pytest.raises((ValueError, FileNotFoundError)):
            attach_context_pack_reference({}, tmp_path, path, digest)
