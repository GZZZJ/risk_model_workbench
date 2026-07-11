from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from risk_model_workbench.registry import (
    audit_artifact_availability,
    load_artifact_manifest,
    register_artifact,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    workspace = repo / "projects" / "demo" / "versions" / "demo_v1"
    workspace.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.test")
    _git(repo, "config", "user.name", "Test")
    (repo / ".gitignore").write_text("*.feather\n**/cache/\n", encoding="utf-8")
    return repo, workspace


def test_ignored_artifact_requires_explicit_storage_class(tmp_path):
    repo, workspace = _repo(tmp_path)
    score = workspace / "evaluation" / "scores.feather"
    score.parent.mkdir(parents=True)
    score.write_bytes(b"local scores")

    with pytest.raises(ValueError, match="explicit storage_class"):
        register_artifact(workspace, score, stage="evaluate")

    entry = register_artifact(
        workspace,
        score,
        stage="evaluate",
        storage_class="local_only",
        contract_role="optional",
        retention_reason="large scored dataset",
        regeneration="rmw evaluate --project projects/demo --version-id demo_v1",
    )
    assert entry["storage_class"] == "local_only"
    assert entry["sha256"]
    assert entry["size_bytes"] == len(b"local scores")


def test_duplicate_stage_registrations_use_latest_content_snapshot(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    shared = workspace / "feature_selection" / "resource_plan.json"
    shared.parent.mkdir(parents=True)
    shared.write_text('{"stage":"prescreen"}', encoding="utf-8")
    first = register_artifact(workspace, shared, stage="feature_prescreen")
    shared.write_text('{"stage":"refine"}', encoding="utf-8")
    second = register_artifact(workspace, shared, stage="feature_refine")
    manifest = load_artifact_manifest(workspace)
    for entry in manifest["artifacts"]:
        if entry["stage"] == "feature_prescreen":
            entry["registered_at"] = "2026-07-11T10:00:00"
        elif entry["stage"] == "feature_refine":
            entry["registered_at"] = "2026-07-11T10:01:00"

    result = audit_artifact_availability(workspace, manifest)

    assert first["sha256"] != second["sha256"]
    assert not any("resource_plan.json" in issue for issue in result["issues"])


def test_untracked_artifact_cannot_claim_repository_retention(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    report = workspace / "reports" / "model.md"
    report.parent.mkdir(parents=True)
    report.write_text("report", encoding="utf-8")

    entry = register_artifact(
        workspace,
        report,
        stage="report",
        storage_class="repository",
        retention_reason="intended report",
    )

    assert entry["storage_class"] == "workspace_only"


def test_repository_aware_audit_promotes_tracked_workspace_artifact(tmp_path):
    repo, workspace = _repo(tmp_path)
    report = workspace / "reports" / "model.md"
    report.parent.mkdir(parents=True)
    report.write_text("report", encoding="utf-8")
    register_artifact(workspace, report, stage="report")
    _git(repo, "add", str(report.relative_to(repo)))

    result = audit_artifact_availability(workspace, load_artifact_manifest(workspace))

    assert result["availability_summary"]["repository_present"] == 1
    assert result["availability_summary"]["workspace_only_present"] == 0
    assert result["reproducibility_verdict"] == "complete"


def test_clean_archive_keeps_required_closure_and_warns_for_optional_local_only(tmp_path):
    repo, workspace = _repo(tmp_path)
    report = workspace / "reports" / "model.md"
    report.parent.mkdir(parents=True)
    report.write_text("registered report", encoding="utf-8")
    _git(repo, "add", ".gitignore", str(report.relative_to(repo)))
    _git(repo, "commit", "-m", "add report")

    register_artifact(
        workspace,
        report,
        stage="report",
        storage_class="repository",
        contract_role="required",
        retention_reason="required committed report",
    )
    score = workspace / "evaluation" / "scores.feather"
    score.parent.mkdir(parents=True)
    score.write_bytes(b"scores")
    register_artifact(
        workspace,
        score,
        stage="evaluate",
        storage_class="local_only",
        contract_role="optional",
        retention_reason="large local scores",
        regeneration="rmw evaluate --project projects/demo --version-id demo_v1",
    )
    manifest_path = workspace / "audit" / "artifact_manifest.json"
    _git(repo, "add", str(manifest_path.relative_to(repo)))
    _git(repo, "commit", "-m", "add retention manifest")

    archive = tmp_path / "repo.tar"
    subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", "-o", str(archive), "HEAD"], check=True)
    clone = tmp_path / "clean"
    clone.mkdir()
    with tarfile.open(archive) as handle:
        handle.extractall(clone, filter="data")
    _git(clone, "init")
    _git(clone, "config", "user.email", "test@example.test")
    _git(clone, "config", "user.name", "Test")
    _git(clone, "add", ".")
    _git(clone, "commit", "-m", "materialize clean archive")
    clean_workspace = clone / workspace.relative_to(repo)
    availability = audit_artifact_availability(clean_workspace, load_artifact_manifest(clean_workspace))

    assert availability["execution_verdict"] == "complete"
    assert availability["reproducibility_verdict"] == "complete_with_warnings"
    assert availability["availability_summary"]["repository_present"] == 1
    assert availability["availability_summary"]["local_only_missing"] == 1
    assert any("optional local_only" in warning for warning in availability["warnings"])

    (clean_workspace / "reports" / "model.md").unlink()
    missing = audit_artifact_availability(clean_workspace, load_artifact_manifest(clean_workspace))
    assert missing["execution_verdict"] == "incomplete"
    assert missing["availability_summary"]["required_missing"] == 1


def test_legacy_manifest_is_readable_and_not_repository_retained(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    legacy = workspace / "reports" / "legacy.md"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("legacy", encoding="utf-8")
    manifest = {"version": 1, "artifacts": [{"path": "reports/legacy.md", "stage": "report"}]}

    availability = audit_artifact_availability(workspace, manifest)

    assert availability["execution_verdict"] == "complete"
    assert availability["reproducibility_verdict"] == "workspace_dependent"
    assert availability["availability_summary"]["legacy_workspace"] == 1
    assert availability["availability_summary"]["repository_present"] == 0


def test_local_only_cannot_satisfy_required_contract(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    artifact = workspace / "scores.feather"
    artifact.write_bytes(b"scores")
    with pytest.raises(ValueError, match="cannot have required"):
        register_artifact(
            workspace,
            artifact,
            stage="evaluate",
            storage_class="local_only",
            contract_role="required",
            retention_reason="invalid",
        )


def test_registration_and_audit_reject_workspace_escape_and_wrong_kind(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the version workspace"):
        register_artifact(workspace, outside, stage="report")

    directory = workspace / "reports"
    directory.mkdir()
    manifest = {
        "artifacts": [
            {"path": str(outside), "storage_class": "workspace_only", "contract_role": "required", "retention_reason": "bad"},
            {"path": "reports", "kind": "file", "storage_class": "workspace_only", "contract_role": "required", "retention_reason": "bad"},
        ]
    }
    availability = audit_artifact_availability(workspace, manifest)
    assert availability["execution_verdict"] == "incomplete"
    assert availability["availability_summary"]["required_missing"] == 2
    assert any("escapes" in issue for issue in availability["issues"])


def test_malformed_retention_metadata_and_unknown_role_fail_closed(tmp_path):
    _repo_path, workspace = _repo(tmp_path)
    manifest = {
        "artifacts": [
            {"path": "scores.feather", "storage_class": "local_only", "contract_role": "optional"},
            {"path": "optional.txt", "storage_class": "workspace_only", "contract_role": "banana", "retention_reason": "bad role"},
            {"path": "external", "storage_class": "external", "contract_role": "optional", "retention_reason": "remote"},
        ]
    }
    availability = audit_artifact_availability(workspace, manifest)
    assert availability["execution_verdict"] == "incomplete"
    issue_text = "\n".join(availability["issues"])
    assert "artifact sha256 invalid" in issue_text
    assert "artifact size_bytes invalid" in issue_text
    assert "retention_reason missing" in issue_text
    assert "unknown contract_role" in issue_text
    assert "external_reference missing" in issue_text


def test_repository_requires_live_git_index_and_matching_integrity(tmp_path):
    repo, workspace = _repo(tmp_path)
    report = workspace / "reports" / "model.md"
    report.parent.mkdir(parents=True)
    report.write_text("original", encoding="utf-8")
    _git(repo, "add", ".gitignore", str(report.relative_to(repo)))
    _git(repo, "commit", "-m", "track report")
    entry = register_artifact(
        workspace,
        report,
        stage="report",
        storage_class="repository",
        retention_reason="tracked report",
    )
    manifest = {"artifacts": [entry]}
    assert audit_artifact_availability(workspace, manifest)["reproducibility_verdict"] == "complete"

    report.write_text("tampered", encoding="utf-8")
    tampered = audit_artifact_availability(workspace, manifest)
    assert tampered["execution_verdict"] == "incomplete"
    assert any("mismatch" in issue for issue in tampered["issues"])

    archive_workspace = tmp_path / "archive" / "versions" / "v1"
    archive_workspace.mkdir(parents=True)
    copied = archive_workspace / "model.md"
    copied.write_text("original", encoding="utf-8")
    forged = dict(entry, path="model.md", repository_tracked=True)
    no_git = audit_artifact_availability(archive_workspace, {"artifacts": [forged]})
    assert no_git["availability_summary"]["repository_present"] == 0
    assert no_git["reproducibility_verdict"] == "workspace_dependent"


def test_local_file_retention_requires_valid_hash_and_non_negative_size(tmp_path):
    repo, workspace = _repo(tmp_path)
    report = workspace / "report.md"
    report.write_text("report", encoding="utf-8")
    _git(repo, "add", ".gitignore", str(report.relative_to(repo)))
    _git(repo, "commit", "-m", "track report")
    manifest = {
        "artifacts": [
            {"path": "report.md", "storage_class": "repository", "contract_role": "required", "retention_reason": "tracked"},
            {"path": "missing.feather", "storage_class": "local_only", "contract_role": "optional", "retention_reason": "local", "sha256": "x", "size_bytes": -1, "source": "generated"},
        ]
    }
    availability = audit_artifact_availability(workspace, manifest)
    assert availability["execution_verdict"] == "incomplete"
    text = "\n".join(availability["issues"])
    assert text.count("artifact sha256 invalid") == 2
    assert text.count("artifact size_bytes invalid") == 2


def test_mutable_integrity_cannot_be_forged_for_general_artifacts(tmp_path):
    repo, workspace = _repo(tmp_path)
    report = workspace / "report.md"
    report.write_text("report", encoding="utf-8")
    _git(repo, "add", ".gitignore", str(report.relative_to(repo)))
    _git(repo, "commit", "-m", "track report")
    forged = {
        "path": "report.md",
        "kind": "audit",
        "storage_class": "repository",
        "contract_role": "required",
        "retention_reason": "forged mutable",
        "integrity_mode": "mutable",
        "sha256": "0" * 64,
        "size_bytes": 0,
    }
    availability = audit_artifact_availability(workspace, {"artifacts": [forged]})
    assert availability["execution_verdict"] == "incomplete"
    text = "\n".join(availability["issues"])
    assert "invalid mutable integrity scope" in text
    assert "mismatch" in text

    unknown = dict(forged, integrity_mode="anything")
    unknown_result = audit_artifact_availability(workspace, {"artifacts": [unknown]})
    assert unknown_result["execution_verdict"] == "incomplete"
    assert any("unknown integrity_mode" in issue for issue in unknown_result["issues"])

    other_log = workspace / "audit" / "other.log"
    other_log.parent.mkdir(exist_ok=True)
    other_log.write_text("mutable", encoding="utf-8")
    with pytest.raises(ValueError, match="approved workspace audit"):
        register_artifact(
            workspace,
            other_log,
            stage="agent_runtime",
            kind="audit",
            storage_class="workspace_only",
            integrity_mode="mutable",
            retention_reason="not allowlisted",
        )
