from pathlib import Path

import pytest

from risk_model_workbench.config import (
    ConfigConflictError,
    ConfigFormatError,
    load_yaml,
    resolve_yaml_variant,
)


def test_load_yaml_project_config():
    data = load_yaml(Path("projects/2026-05-fujie-gcard-v1/project.yml"))
    assert data["data"]["target_column"] == "ftr_30d_ord_flag"


def test_load_yaml_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "duplicate.yml"
    path.write_text("project: one\nproject: two\n", encoding="utf-8")

    with pytest.raises(ConfigFormatError, match=r"duplicate key.*project"):
        load_yaml(path)


def test_load_yaml_rejects_multiple_documents(tmp_path):
    path = tmp_path / "multiple.yml"
    path.write_text("project: one\n---\nproject: two\n", encoding="utf-8")

    with pytest.raises(ConfigFormatError, match="exactly one YAML document"):
        load_yaml(path)


def test_load_yaml_rejects_non_mapping(tmp_path):
    path = tmp_path / "list.yml"
    path.write_text("- one\n- two\n", encoding="utf-8")

    with pytest.raises(ConfigFormatError, match="YAML mapping"):
        load_yaml(path)


def test_load_yaml_supports_empty_anchors_and_literal_environment_strings(tmp_path):
    empty = tmp_path / "empty.yml"
    empty.write_text("", encoding="utf-8")
    assert load_yaml(empty) == {}

    path = tmp_path / "valid.yml"
    path.write_text(
        "defaults: &defaults\n"
        "  source: ${SOURCE_TABLE}\n"
        "data:\n"
        "  <<: *defaults\n"
        "  target: label\n",
        encoding="utf-8",
    )

    assert load_yaml(path) == {
        "defaults": {"source": "${SOURCE_TABLE}"},
        "data": {"source": "${SOURCE_TABLE}", "target": "label"},
    }


def test_load_yaml_rejects_duplicate_merge_keys(tmp_path):
    path = tmp_path / "duplicate_merge.yml"
    path.write_text(
        "first: &first\n"
        "  source: one\n"
        "second: &second\n"
        "  target: label\n"
        "data:\n"
        "  <<: *first\n"
        "  <<: *second\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigFormatError, match=r"duplicate key.*merge"):
        load_yaml(path)


def test_resolve_yaml_variant_handles_single_and_equivalent_variants(tmp_path):
    canonical = tmp_path / "train.yaml"
    legacy = tmp_path / "train.yml"

    legacy.write_text("training:\n  rounds: 10\n", encoding="utf-8")
    assert resolve_yaml_variant(canonical, legacy) == legacy

    canonical.write_text("training: {rounds: 10}\n", encoding="utf-8")
    assert resolve_yaml_variant(canonical, legacy) == canonical

    legacy.unlink()
    assert resolve_yaml_variant(canonical, legacy) == canonical


def test_resolve_yaml_variant_rejects_divergent_variants_with_both_paths(tmp_path):
    canonical = tmp_path / "train.yaml"
    legacy = tmp_path / "train.yml"
    canonical.write_text("training: {rounds: 10}\n", encoding="utf-8")
    legacy.write_text("training: {rounds: 20}\n", encoding="utf-8")

    with pytest.raises(ConfigConflictError) as exc_info:
        resolve_yaml_variant(canonical, legacy)

    message = str(exc_info.value)
    assert str(canonical) in message
    assert str(legacy) in message


@pytest.mark.parametrize("broken_variant", ["canonical", "legacy"])
def test_resolve_yaml_variant_format_error_names_both_existing_variants(tmp_path, broken_variant):
    canonical = tmp_path / "train.yaml"
    legacy = tmp_path / "train.yml"
    canonical.write_text("training: {rounds: 10}\n", encoding="utf-8")
    legacy.write_text("training:\n  rounds: 10\n", encoding="utf-8")
    target = canonical if broken_variant == "canonical" else legacy
    target.write_text("training: one\ntraining: two\n", encoding="utf-8")

    with pytest.raises(ConfigFormatError) as exc_info:
        resolve_yaml_variant(canonical, legacy)

    message = str(exc_info.value)
    assert str(canonical) in message
    assert str(legacy) in message
