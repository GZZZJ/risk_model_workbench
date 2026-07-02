"""Tests for the shared split resolver / consistency checker.

Covers the request-first → project → default resolution priority and every
consistency rule (time-out pollution, empty splits, dead-field warning, etc.).
These are the rules both validate_model_request and materialize enforce.
"""

from risk_model_workbench.request.splits import (
    DEFAULT_OOS_VALUES,
    SplitValidationError,
    check_split_consistency,
    resolve_split_values,
)


# ---------- resolve_split_values: priority chain ----------

def test_resolve_prefers_request_splits_over_project_and_default():
    metadata = {"splits": {"dev": {"values": ["DEVX"]}, "oos": {"values": ["OOSX"]}, "oot": {"values": ["OOTX"]}}}
    project_config = {"split": {"ins_values": ["DEV"], "oos_values": ["DEV-OOS"], "oot_values": ["OOT"]}}
    resolved = resolve_split_values(metadata, project_config)
    assert resolved["dev"]["values"] == ["DEVX"]
    assert resolved["dev"]["source"] == "request.splits"
    assert resolved["oos"]["values"] == ["OOSX"]
    assert resolved["oot"]["values"] == ["OOTX"]


def test_resolve_falls_back_to_project_split_when_request_omits_splits():
    metadata = {}
    project_config = {"split": {"ins_values": ["DEV"], "oos_values": ["DEV-OOS"], "oot_values": ["OOT", "OOT-OOS"]}}
    resolved = resolve_split_values(metadata, project_config)
    assert resolved["dev"]["values"] == ["DEV"]
    assert resolved["dev"]["source"] == "project.split"
    assert resolved["oos"]["values"] == ["DEV-OOS"]
    assert resolved["oot"]["values"] == ["OOT", "OOT-OOS"]


def test_resolve_uses_hardcoded_defaults_when_neither_request_nor_project_configures():
    resolved = resolve_split_values({}, {})
    assert resolved["dev"]["values"] == ["DEV"]
    assert resolved["oos"]["values"] == DEFAULT_OOS_VALUES
    assert resolved["oot"]["values"] == ["OOT"]
    assert all(resolved[k]["source"] == "default" for k in ("dev", "oos", "oot"))


def test_resolve_filters_dirty_empty_string_values_and_falls_back():
    # splits.oos.values: [""] is a truthy list but has no real labels — must fall through
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": [""]}, "oot": {"values": ["OOT"]}}}
    project_config = {"split": {"oos_values": ["DEV-OOS"]}}
    resolved = resolve_split_values(metadata, project_config)
    assert resolved["oos"]["values"] == ["DEV-OOS"]
    assert resolved["oos"]["source"] == "project.split"


# ---------- check_split_consistency: errors (blocking) ----------

def test_check_clean_config_has_no_errors():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": ["OOT", "OOT-OOS"]}}}
    result = check_split_consistency(metadata, {})
    assert result["errors"] == []


def test_check_oos_oot_intersection_is_error():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["OOT-OOS"]}, "oot": {"values": ["OOT", "OOT-OOS"]}}}
    result = check_split_consistency(metadata, {})
    assert any("相交" in e for e in result["errors"])
    assert any("OOT-OOS" in e for e in result["errors"])


def test_check_oos_contains_pure_oot_is_error():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["OOT"]}, "oot": {"values": ["OOT", "OOT-OOS"]}}}
    result = check_split_consistency(metadata, {})
    assert any("相交" in e for e in result["errors"])


def test_check_empty_dev_is_error():
    metadata = {"splits": {"dev": {"values": [""]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": ["OOT"]}}}
    result = check_split_consistency(metadata, {})
    assert any("splits.dev" in e and "为空或无效" in e for e in result["errors"])


# ---------- check_split_consistency: warnings (non-blocking) ----------

def test_check_dead_training_valid_values_field_is_warning():
    metadata = {
        "splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": ["OOT"]}},
        "training": {"valid_values": ["DEV-OOS"]},
    }
    result = check_split_consistency(metadata, {})
    assert result["errors"] == []
    assert any("training.valid_values" in w for w in result["warnings"])


def test_check_dev_oos_overlap_is_warning_not_error():
    metadata = {"splits": {"dev": {"values": ["DEV", "DEV-OOS"]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": ["OOT"]}}}
    result = check_split_consistency(metadata, {})
    assert result["errors"] == []
    assert any("dev" in w and "oos" in w and "相交" in w for w in result["warnings"])


def test_check_fully_defaulted_splits_is_warning():
    result = check_split_consistency({}, {})
    assert any("默认" in w for w in result["warnings"])


def test_check_resolved_returned_for_materializer_reuse():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": ["OOT"]}}}
    result = check_split_consistency(metadata, {})
    assert result["resolved"]["oos"]["values"] == ["DEV-OOS"]


# ---------- SplitValidationError is a ValueError subclass ----------

def test_split_validation_error_is_value_error():
    assert issubclass(SplitValidationError, ValueError)


# ---------- project.yml fallback path (bypasses request layer) ----------

def test_check_project_split_intersection_is_error():
    # request has no splits, but project.split has oos∩oot -> must still fire
    project_config = {"split": {"ins_values": ["DEV"], "oos_values": ["OOT"], "oot_values": ["OOT", "OOT-OOS"]}}
    result = check_split_consistency({}, project_config)
    assert result["errors"]
    assert any("相交" in e and "OOT" in e for e in result["errors"])


def test_resolve_project_split_used_as_source():
    project_config = {"split": {"ins_values": ["DEV"], "oos_values": ["DEV-OOS"], "oot_values": ["OOT"]}}
    resolved = resolve_split_values({}, project_config)
    assert resolved["oos"]["source"] == "project.split"


# ---------- multi-value partial intersection ----------

def test_check_partial_oos_intersection_reports_overlap():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["DEV-OOS", "OOT"]}, "oot": {"values": ["OOT", "OOT-OOS"]}}}
    result = check_split_consistency(metadata, {})
    assert result["errors"]
    overlap = [e for e in result["errors"] if "相交" in e]
    assert overlap
    assert any("OOT" in e for e in overlap)


# ---------- dirty oos / oot (explicit but empty) ----------

def test_check_explicit_dirty_oos_is_error():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": [""]}, "oot": {"values": ["OOT"]}}}
    result = check_split_consistency(metadata, {})
    assert any("splits.oos" in e and "为空或无效" in e for e in result["errors"])


def test_check_explicit_dirty_oot_is_error():
    metadata = {"splits": {"dev": {"values": ["DEV"]}, "oos": {"values": ["DEV-OOS"]}, "oot": {"values": [""]}}}
    result = check_split_consistency(metadata, {})
    assert any("splits.oot" in e and "为空或无效" in e for e in result["errors"])


# ---------- scalar value compatibility (regression guard) ----------

def test_resolve_scalar_value_is_wrapped_as_single_element_list():
    # project.yml written as `oos_values: DEV-OOS` (scalar) must be honored,
    # not silently dropped — matches legacy materialize._string_list behavior.
    project_config = {"split": {"ins_values": "DEV", "oos_values": "DEV-OOS", "oot_values": "OOT"}}
    resolved = resolve_split_values({}, project_config)
    assert resolved["dev"]["values"] == ["DEV"]
    assert resolved["oos"]["values"] == ["DEV-OOS"]
    assert resolved["oot"]["values"] == ["OOT"]


def test_resolve_scalar_request_splits_values_wrapped():
    # request written as `splits.oos.values: DEV-OOS` (scalar) must be honored
    # and attributed to request.splits, not silently dropped.
    resolved = resolve_split_values(
        {"splits": {"dev": {"values": "DEV"}, "oos": {"values": "DEV-OOS"}, "oot": {"values": "OOT"}}},
        {},
    )
    assert resolved["oos"]["values"] == ["DEV-OOS"]
    assert resolved["oos"]["source"] == "request.splits"
