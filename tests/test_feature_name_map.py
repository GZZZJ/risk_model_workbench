"""Tests for feature Chinese-name (feature_comment) injection into report tables."""

from pathlib import Path

import pandas as pd

from risk_model_workbench.reporting.excel_report import (
    _gcard_top_features_frame,
    _load_feature_name_map,
)


def _write_feature_columns(project_dir: Path) -> None:
    meta_dir = project_dir / "data" / "profile" / "feature_metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    # UTF-8 with BOM, mirroring the real feature_columns.csv
    (meta_dir / "feature_columns.csv").write_text(
        "﻿table_index,full_table_name,feature_name,feature_type,feature_comment,ordinal\n"
        "1,t,f1,string,中文名1,1\n"
        "1,t,f2,string,,2\n"
        "1,t,f3,string,中文名3,3\n",
        encoding="utf-8",
    )


def test_load_feature_name_map_reads_feature_comment(tmp_path):
    _write_feature_columns(tmp_path)
    name_map = _load_feature_name_map(tmp_path)
    assert name_map == {"f1": "中文名1", "f3": "中文名3"}  # empty comment skipped


def test_load_feature_name_map_degrades_when_missing(tmp_path):
    assert _load_feature_name_map(None) == {}
    assert _load_feature_name_map(tmp_path / "does-not-exist") == {}


def test_gcard_top_features_frame_uses_name_map_with_english_fallback():
    importance = pd.DataFrame({"feature": ["f1", "f2", "f3"], "gain": [3.0, 2.0, 1.0], "split": [10, 8, 5]})
    name_map = {"f1": "中文名1", "f3": "中文名3"}
    frame = _gcard_top_features_frame(importance, name_map=name_map)
    desc = dict(zip(frame["varname"], frame["desc"]))
    assert desc["f1"] == "中文名1"
    assert desc["f3"] == "中文名3"
    assert desc["f2"] == "f2"  # not in dict → English fallback


def test_gcard_top_features_frame_without_name_map_keeps_english():
    importance = pd.DataFrame({"feature": ["f1"], "gain": [1.0], "split": [5]})
    frame = _gcard_top_features_frame(importance, name_map=None)
    assert frame["desc"].iloc[0] == "f1"
