import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from risk_model_workbench.agent.advisor import (
    create_advisor_request,
    current_context_hash_for_request,
    load_advisor_context_pack,
)
from risk_model_workbench.agent.context_pack import (
    DEFAULT_MAX_FILE_BYTES,
    DEFAULT_MAX_PACK_BYTES,
    build_context_pack,
    load_context_pack,
    persist_context_pack,
)
from risk_model_workbench.cli import main
from risk_model_workbench.context_snapshot import build_context_snapshot


def _build(workspace: Path, paths: list[str], **kwargs):
    return build_context_pack(
        workspace,
        project="demo",
        version_id="demo_v1",
        task_id="train_main",
        attempt_id="attempt_1",
        request_type="tuning_plan_required",
        paths=paths,
        constraints=["bounded"],
        allowed_tools=["train_baseline"],
        output_contract={"type": "tuning_plan"},
        **kwargs,
    )


def test_context_pack_is_deterministic_sorted_and_content_addressed(tmp_path):
    (tmp_path / "version_state.yml").write_text("status: running\n", encoding="utf-8")
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "agent_state.yml").write_text("status: waiting_for_advisor\n", encoding="utf-8")

    first = _build(tmp_path, ["version_state.yml", "audit/agent_state.yml"])
    second = _build(tmp_path, ["audit/agent_state.yml", "version_state.yml", "version_state.yml"])

    assert first == second
    assert [item["path"] for item in first["files"]] == ["audit/agent_state.yml", "version_state.yml"]
    assert all(len(item["sha256"]) == 64 for item in first["files"])
    assert len(first["context_hash"]) == 64
    assert first["version"] == 1

    (tmp_path / "version_state.yml").write_text("status: done\n", encoding="utf-8")
    changed = _build(tmp_path, ["version_state.yml", "audit/agent_state.yml"])
    assert changed["context_hash"] != first["context_hash"]


def test_yaml_sets_are_deterministic_across_python_hash_seeds(tmp_path):
    (tmp_path / "configs_runtime").mkdir()
    (tmp_path / "configs_runtime" / "train.yaml").write_text(
        "tags: !!set\n  zebra: null\n  alpha: null\n  middle: null\n",
        encoding="utf-8",
    )
    code = (
        "from risk_model_workbench.agent.context_pack import build_context_pack; "
        f"p=build_context_pack({str(tmp_path)!r},project='demo',version_id='v',task_id='t',"
        "attempt_id='a',request_type='x',paths=['configs_runtime/train.yaml'],"
        "constraints=[],allowed_tools=[],output_contract={}); print(p['context_hash'])"
    )
    hashes = []
    for seed in ["1", "2", "99"]:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        hashes.append(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())

    assert len(set(hashes)) == 1


def test_context_pack_records_missing_and_rejects_unsafe_inputs(tmp_path):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "artifact_manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "data.parquet").write_bytes(b"raw")
    (tmp_path / "model.pkl").write_bytes(b"pickle")
    (tmp_path / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (tmp_path / "credentials.json").write_text("{}", encoding="utf-8")
    (tmp_path / "modeling").mkdir()
    (tmp_path / "modeling" / "sample.json").write_text("{}", encoding="utf-8")
    (tmp_path / "linked.yml").symlink_to(tmp_path / "audit" / "artifact_manifest.json")

    pack = _build(
        tmp_path,
        [
            "missing.yml",
            "data.parquet",
            "model.pkl",
            ".env",
            "credentials.json",
            "modeling/sample.json",
            "linked.yml",
            "../escape.yml",
            str((tmp_path / "audit" / "artifact_manifest.json").resolve()),
        ],
    )
    by_path = {item["path"]: item for item in pack["files"]}

    assert by_path["missing.yml"]["status"] == "missing"
    assert by_path["data.parquet"]["status"] == "rejected"
    assert by_path["model.pkl"]["status"] == "rejected"
    assert by_path[".env"]["reason"] == "credential_file"
    assert by_path["credentials.json"]["reason"] == "credential_file"
    assert by_path["modeling/sample.json"]["reason"] == "path_not_allowed"
    assert by_path["linked.yml"]["reason"] == "symlink"
    assert by_path["../escape.yml"]["reason"] == "path_escape"
    absolute = str((tmp_path / "audit" / "artifact_manifest.json").resolve())
    assert by_path[absolute]["reason"] == "absolute_path"
    assert all(item["sha256"] == "" for item in pack["files"])


def test_context_pack_rejects_disguised_suffixes_and_credential_path_segments(tmp_path):
    (tmp_path / "configs_runtime" / "credentials").mkdir(parents=True)
    candidates = {
        "configs_runtime/.env.yaml": "safe: false\n",
        "configs_runtime/data.parquet.yaml": "safe: false\n",
        "configs_runtime/model.pkl.json": "{}",
        "configs_runtime/credentials/train.yaml": "safe: false\n",
    }
    for relative, content in candidates.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    pack = _build(tmp_path, list(candidates))
    rows = {row["path"]: row for row in pack["files"]}

    assert all(rows[path]["status"] == "rejected" for path in candidates)
    assert rows["configs_runtime/.env.yaml"]["reason"] == "credential_file"
    assert rows["configs_runtime/data.parquet.yaml"]["reason"] == "prohibited_file_type"
    assert rows["configs_runtime/model.pkl.json"]["reason"] == "prohibited_file_type"
    assert rows["configs_runtime/credentials/train.yaml"]["reason"] == "credential_file"


def test_context_pack_redacts_sensitive_keys_identifiers_and_secret_content(tmp_path):
    (tmp_path / "configs_runtime").mkdir()
    (tmp_path / "configs_runtime" / "train.yaml").write_text(
        yaml.safe_dump(
            {
                "learning_rate": 0.03,
                "password": "dont-print-me",
                "nested": {"api_key": "sk-live-secret", "customer_id": "C12345678"},
                "notes": "Bearer abcdefghijklmnopqrstuvwxyz",
            }
        ),
        encoding="utf-8",
    )

    pack = _build(tmp_path, ["configs_runtime/train.yaml"])
    rendered = json.dumps(pack["files"][0]["summary"], ensure_ascii=False)

    assert pack["files"][0]["status"] == "included"
    assert "learning_rate" in rendered
    assert "dont-print-me" not in rendered
    assert "sk-live-secret" not in rendered
    assert "C12345678" not in rendered
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
    assert rendered.count("[REDACTED]") >= 3


def test_context_pack_redacts_nested_container_keys_and_secret_value_formats(tmp_path):
    (tmp_path / "configs_runtime").mkdir()
    (tmp_path / "configs_runtime" / "train.yaml").write_text(
        yaml.safe_dump(
            {
                "credentials": {"username": "alice", "value": "nested-secret"},
                "secret_value": "secret-value",
                "database_url": "postgresql://alice:db-password@example.test/risk",
                "clientSecretValue": "camel-secret",
                "aws_hint": "AKIAIOSFODNN7EXAMPLE",
                "trace": "550e8400-e29b-41d4-a716-446655440000",
                "safe": {"learning_rate": 0.03},
            }
        ),
        encoding="utf-8",
    )

    pack = _build(tmp_path, ["configs_runtime/train.yaml"])
    rendered = json.dumps(pack["files"][0]["summary"], ensure_ascii=False)

    for leaked in [
        "alice",
        "nested-secret",
        "secret-value",
        "db-password",
        "camel-secret",
        "AKIAIOSFODNN7EXAMPLE",
        "550e8400-e29b-41d4-a716-446655440000",
        "446655440000",
    ]:
        assert leaked not in rendered
    assert "learning_rate" in rendered


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        (
            "configs_runtime/train.yaml",
            {"rows": [{"name": "Alice", "acct": "A10001", "loan_no": "L90001", "x1": 1.2}]},
        ),
        (
            "evaluation/metrics.json",
            {"auc": 0.72, "sample": [{"name": "Bob", "account": "B20002", "order_no": "O80002"}]},
        ),
        (
            "evaluation/metrics.json",
            {"items": [{"name": "Carol", "account_id": "C30003", "score": 0.8}]},
        ),
    ],
)
def test_context_pack_rejects_row_level_record_containers(tmp_path, relative, payload):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    pack = _build(tmp_path, [relative])
    rendered = json.dumps(pack, ensure_ascii=False)

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"
    for leaked in ["Alice", "A10001", "L90001", "Bob", "B20002", "O80002", "Carol", "C30003"]:
        assert leaked not in rendered


@pytest.mark.parametrize(
    "payload",
    [
        {"row": {"age": 31, "score": 0.72}},
        {"record": {"age": 32, "score": 0.73}},
        {"sample": {"age": 33, "score": 0.74}},
        {"data": {"age": 34, "score": 0.75}},
        [{"age": 35, "score": 0.76}],
    ],
)
def test_context_pack_fails_closed_on_row_shapes_without_identifiers(tmp_path, payload):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


@pytest.mark.parametrize(
    ("container", "record"),
    [
        ("observations", {"age": 30, "score": 0.71}),
        ("predictions", {"score": 0.72, "label": 1}),
        ("examples", {"age": 31, "score": 0.73}),
        ("cases", {"age": 32, "score": 0.74}),
        ("details", {"age": 33, "score": 0.75}),
        ("results", {"score": 0.76, "label": 0}),
        ("unknown_container", {"age": 34, "score": 0.77}),
    ],
)
def test_unknown_nested_record_lists_fail_closed_by_structure(tmp_path, container, record):
    (tmp_path / "evaluation").mkdir()
    payload = {"outer": {container: [record]}}
    (tmp_path / "evaluation" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


@pytest.mark.parametrize(
    ("container", "record"),
    [
        ("candidates", {"candidate_name": "safe", "learning_rate": 0.1, "unknown": [{"age": 30}]}),
        ("tasks", {"task_id": "task_1", "type": "train", "unknown": [{"age": 30}]}),
        ("trials", {"trial_id": "trial_1", "status": "done", "payload": {"observations": [{"age": 30}]}}),
    ],
)
def test_allowed_record_containers_still_scan_nested_records(tmp_path, container, record):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps({container: [record]}), encoding="utf-8"
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


@pytest.mark.parametrize(
    ("container", "record"),
    [
        ("tasks", {"task_id": "task_1", "age": 41, "score": 0.91, "label": 1}),
        ("candidates", {"candidate_name": "safe", "age": 41, "score": 0.91, "label": 1}),
        ("items", {"candidate_name": "safe", "age": 41, "score": 0.91, "label": 1}),
        ("trials", {"trial_id": "trial_1", "age": 41, "score": 0.91, "label": 1}),
    ],
)
def test_whitelisted_record_contracts_reject_unknown_scalar_fields(tmp_path, container, record):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps({container: [record]}), encoding="utf-8"
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


@pytest.mark.parametrize(
    ("container", "record"),
    [
        ("tasks", {"task_id": "task_1", "invocation": {"age": 45, "score": 0.95, "label": 1}}),
        ("candidates", {"candidate_name": "safe", "Params": {"age": 46, "score": 0.96}}),
        ("trials", {"trial_id": "trial_1", "params": {"age": 47, "score": 0.97}}),
        ("trials", {"trial_id": "trial_1", "metrics": {"age": 47, "score": 0.97}}),
    ],
)
def test_whitelisted_record_contracts_reject_nested_schema_spoof(tmp_path, container, record):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps({container: [record]}), encoding="utf-8"
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


@pytest.mark.parametrize(
    "record",
    [
        {
            "task_id": "task_1",
            "invocation": {
                "tool_name": "sample_check",
                "project": "projects/demo",
                "version_id": "demo_v1",
                "params": {"raw_payload": {"income": 123456, "bad_flag": 1}},
            },
        },
        {
            "task_id": "task_1",
            "command": {"executable": "rmw", "args": ["train", '{"income":234567,"bad_flag":1}']},
        },
        {
            "task_id": "task_1",
            "command": {
                "executable": "rmw",
                "args": ["train", "income: 456789, bad_flag: 1, risk_value: 0.97"],
            },
        },
        {
            "task_id": "task_1",
            "command": {
                "executable": "rmw",
                "args": ["train", "income=567890,bad_flag=1,risk_value=.98"],
            },
        },
        {
            "task_id": "task_1",
            "derived_metadata": {
                "action_id": "sample_check",
                "artifact_contract": {"raw_payload": {"income": 345678, "bad_flag": 1}},
            },
        },
    ],
)
def test_task_child_contracts_reject_opaque_or_unknown_payloads(tmp_path, record):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps({"tasks": [record]}), encoding="utf-8"
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


def test_command_args_are_redacted_without_rejecting_multi_query_urls(tmp_path):
    (tmp_path / "evaluation").mkdir()
    url = "https://example.test/a?x=1&y=2"
    payload = {
        "tasks": [
            {
                "task_id": "task_1",
                "command": {"executable": "rmw", "args": ["report", url]},
            }
        ]
    }
    (tmp_path / "evaluation" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    pack = _build(tmp_path, ["evaluation/metrics.json"])
    rendered = json.dumps(pack["files"][0]["summary"], ensure_ascii=False)

    assert pack["files"][0]["status"] == "included"
    assert url not in rendered
    assert '"redacted": true' in rendered


@pytest.mark.parametrize(
    "container",
    ["observation", "prediction", "example", "case", "detail", "result"],
)
def test_singular_raw_record_containers_fail_closed(tmp_path, container):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps({container: {"age": 31, "score": 0.72}}), encoding="utf-8"
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "row_level_content"


def test_context_pack_keeps_aggregate_metrics_but_redacts_identifier_keys(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(
        json.dumps(
            {
                "sample_count": 1000,
                "metrics": [{"metric_name": "auc", "value": 0.72}],
                "monthly_metrics": [{"period": "2026-01", "auc": 0.71}],
                "owner": {"account": "A10001", "name": "Alice", "model_name": "main_lgbm"},
                "items": [{"candidate_name": "regularized", "learning_rate": 0.03}],
            }
        ),
        encoding="utf-8",
    )

    pack = _build(tmp_path, ["evaluation/metrics.json"])
    rendered = json.dumps(pack["files"][0]["summary"], ensure_ascii=False)

    assert pack["files"][0]["status"] == "included"
    assert "sample_count" in rendered
    assert "metric_name" in rendered
    assert "main_lgbm" in rendered
    assert "regularized" in rendered
    assert "A10001" not in rendered
    assert "Alice" not in rendered


def test_context_pack_allows_monthly_and_segment_aggregate_records(tmp_path):
    (tmp_path / "evaluation").mkdir()
    payload = {
        "monthly_metrics": [
            {
                "period": "2026-01",
                "customer_count": 1000,
                "loan_count": 800,
                "order_count": 900,
                "bad_rate": 0.12,
            }
        ],
        "segment_metrics": [
            {"segment_id": "new_customer", "account_count": 400, "auc": 0.73}
        ],
        "data": {"sample_count": 1000, "score_mean": 0.45, "bad_rate": 0.12},
    }
    (tmp_path / "evaluation" / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    pack = _build(tmp_path, ["evaluation/metrics.json"])
    summary = pack["files"][0]["summary"]
    rendered = json.dumps(summary, ensure_ascii=False)

    assert pack["files"][0]["status"] == "included"
    for field in [
        "customer_count",
        "loan_count",
        "order_count",
        "bad_rate",
        "segment_id",
        "account_count",
        "score_mean",
    ]:
        assert field in rendered
        assert f'"{field}": "[REDACTED]"' not in rendered
    assert "new_customer" in rendered


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        (
            "agent_plan.yml",
            {
                "tasks": [
                    {
                        "task_id": "train_main",
                        "type": "train",
                        "status": "pending",
                        "command": {"args": ["train", "--experiment", "main_lgbm"]},
                        "outputs": ["modeling/main_lgbm/metrics.json"],
                    }
                ]
            },
        ),
        (
            "configs_runtime/train.yaml",
            {
                "training": {
                    "candidates": [
                        {"learning_rate": 0.03, "num_leaves": 31, "reg_lambda": 2.0}
                    ]
                }
            },
        ),
        (
            "modeling/main_lgbm/tuning_context_round_1.json",
            {
                "trials": [
                    {
                        "trial_id": "trial_1",
                        "status": "done",
                        "params": {"learning_rate": 0.03, "num_leaves": 31},
                        "valid_auc": 0.73,
                    }
                ]
            },
        ),
    ],
)
def test_explicit_bounded_plan_and_config_record_structures_are_allowed(tmp_path, relative, payload):
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".json":
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    pack = _build(tmp_path, [relative])

    assert pack["files"][0]["status"] == "included"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("broken.yaml", "private_key: |\n  -----BEGIN PRIVATE KEY-----\n bad-indent\n"),
        ("broken.json", '{"database_url": "postgresql://u:password@host/db"'),
    ],
)
def test_invalid_structured_content_is_rejected_without_raw_fallback(tmp_path, name, content):
    (tmp_path / "configs_runtime").mkdir()
    (tmp_path / "configs_runtime" / name).write_text(content, encoding="utf-8")

    pack = _build(tmp_path, [f"configs_runtime/{name}"])
    rendered = json.dumps(pack, ensure_ascii=False)

    assert pack["files"][0]["status"] == "rejected"
    assert pack["files"][0]["reason"] == "invalid_structured_content"
    assert "PRIVATE KEY" not in rendered
    assert "password@host" not in rendered


def test_context_pack_enforces_per_file_total_limits_and_text_approval(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.json").write_text(json.dumps({"auc": 0.72, "padding": "x" * 40}), encoding="utf-8")
    (tmp_path / "modeling").mkdir()
    (tmp_path / "modeling" / "metrics.json").write_text(json.dumps({"ks": 0.31, "padding": "y" * 40}), encoding="utf-8")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "summary.md").write_text("approved summary", encoding="utf-8")

    bounded = _build(
        tmp_path,
        ["evaluation/metrics.json", "modeling/metrics.json"],
        max_file_bytes=128,
        max_total_bytes=70,
    )
    assert sum(item["size"] for item in bounded["files"] if item["status"] == "included") <= 70
    assert any(item.get("reason") == "total_bytes_exceeded" for item in bounded["files"])

    oversized = tmp_path / "configs_runtime"
    oversized.mkdir()
    (oversized / "train.yaml").write_text("x" * (DEFAULT_MAX_FILE_BYTES + 1), encoding="utf-8")
    rejected = _build(tmp_path, ["configs_runtime/train.yaml"])
    assert rejected["files"][0]["reason"] == "file_bytes_exceeded"

    unapproved = _build(tmp_path, ["reports/summary.md"])
    approved = _build(tmp_path, ["reports/summary.md"], approved_text_reports={"reports/summary.md"})
    assert unapproved["files"][0]["reason"] == "text_report_not_approved"
    assert approved["files"][0]["status"] == "included"


def test_context_pack_bounds_entry_count_and_final_serialized_bytes(tmp_path):
    paths = [f"missing-{index:05d}.yml" for index in range(10_000)]

    pack = _build(tmp_path, paths)
    encoded = (json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

    assert len(pack["files"]) <= 128
    assert pack["truncation"] == {
        "requested_entries": 10_000,
        "recorded_entries": len(pack["files"]),
        "omitted_entries": 10_000 - len(pack["files"]),
    }
    assert len(encoded) <= DEFAULT_MAX_PACK_BYTES


def test_non_finite_floats_are_canonical_json_and_persist_idempotently(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "evaluation" / "metrics.yaml").write_text(
        "metrics:\n  nan_value: .nan\n  positive_inf: .inf\n  negative_inf: -.inf\n",
        encoding="utf-8",
    )

    first = _build(tmp_path, ["evaluation/metrics.yaml"])
    second = _build(tmp_path, ["evaluation/metrics.yaml"])
    encoded = json.dumps(first, ensure_ascii=False, sort_keys=True, allow_nan=False)

    assert first == second
    assert "[NON_FINITE:NaN]" in encoded
    assert "[NON_FINITE:+Infinity]" in encoded
    assert "[NON_FINITE:-Infinity]" in encoded
    relative = persist_context_pack(tmp_path, first)
    assert persist_context_pack(tmp_path, second) == relative
    assert load_context_pack(tmp_path, relative) == first


def test_persisted_context_pack_is_immutable_and_tamper_evident(tmp_path):
    (tmp_path / "version_state.yml").write_text("status: running\n", encoding="utf-8")
    pack = _build(tmp_path, ["version_state.yml"])
    relative = persist_context_pack(tmp_path, pack)

    assert relative == f"audit/context_packs/{pack['context_hash']}.json"
    assert load_context_pack(tmp_path, relative) == pack

    stored = tmp_path / relative
    tampered = json.loads(stored.read_text(encoding="utf-8"))
    tampered["task_id"] = "other"
    stored.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="context pack hash mismatch"):
        load_context_pack(tmp_path, relative)


def test_advisor_request_binds_immutable_context_pack_and_cli_exposes_it(tmp_path, capsys):
    project = _make_project(tmp_path)
    version_id = "demo_model_v1_20260709"
    assert main(["version", "init", "--project", str(project), "--workflow", "train_baseline", "--version-id", version_id]) == 0
    capsys.readouterr()
    workspace = project / "versions" / version_id
    tuning = workspace / "modeling" / "main_lgbm" / "tuning_context_round_1.json"
    tuning.parent.mkdir(parents=True)
    tuning.write_text(json.dumps({"round": 1, "metrics": {"auc": 0.7}}), encoding="utf-8")
    task = {
        "task_id": "train_main",
        "attempt_id": "attempt_1",
        "action_id": "train_baseline",
        "tool_name": "train_baseline",
        "command": {"args": ["train"]},
    }
    request = create_advisor_request(
        workspace,
        project_dir=project,
        version_id=version_id,
        task=task,
        reason="advisor_required",
    )

    assert request["context_pack"].startswith("audit/context_packs/")
    assert request["context_hash"] == load_advisor_context_pack(workspace, request)["context_hash"]
    assert request["context_files"]  # provenance only
    original_hash = current_context_hash_for_request(workspace, request)
    tuning.write_text(json.dumps({"round": 1, "metrics": {"auc": 0.9}}), encoding="utf-8")
    assert current_context_hash_for_request(workspace, request) == original_hash
    snapshot = build_context_snapshot(project, version_id)
    assert snapshot["host_agent_context"] == {
        "context_pack": request["context_pack"],
        "context_hash": request["context_hash"],
    }

    assert main([
        "agent", "advisor", "context", "--project", str(project), "--version-id", version_id,
        "--request-id", request["request_id"], "--json",
    ]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["context_hash"] == request["context_hash"]

    assert main(["agent", "capabilities", "--json"]) == 0
    capabilities = json.loads(capsys.readouterr().out)
    assert capabilities["context_pack"]["version"] == 1
    assert "train_baseline" in {tool["name"] for tool in capabilities["tools"]}

    assert main([
        "agent", "advisor", "context", "--project", str(project), "--version-id", version_id,
        "--request-id", "missing", "--json",
    ]) == 1
    error = json.loads(capsys.readouterr().out)
    assert error["ok"] is False
    assert error["error"]


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "demo_project"
    for directory in ["configs", "queries", "reports", "versions"]:
        (project / directory).mkdir(parents=True, exist_ok=True)
    (project / "project.yml").write_text(
        yaml.safe_dump(
            {
                "project": {"name": "demo_project", "display_name": "Demo Project", "project_key": "demo_model"},
                "data": {
                    "source_table": "demo.sample",
                    "id_columns": ["uid"],
                    "target_column": "label",
                    "time_column": "event_time",
                    "period_column": "ds",
                },
                "segments": [{"name": "all", "display_name": "All", "filter": None}],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return project
