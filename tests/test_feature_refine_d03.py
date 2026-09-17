import pandas as pd

from risk_model_workbench import feature_refine
from risk_model_workbench.feature_refine import DatasetParts, d03_random_importance, select_feature_select_v2_drops
from risk_model_workbench.progress import ProgressReporter, load_progress_events


def test_feature_select_v2_drop_rules_split_gain_zero_and_tail():
    importance = pd.DataFrame(
        {
            "feature": ["strong", "medium", "weak", "zero", "random_col"],
            "split": [10, 5, 1, 0, 2],
            "gain": [100, 30, 6, 0, 5],
        }
    )

    drops, detail = select_feature_select_v2_drops(
        importance,
        "random_col",
        thresholds=0.8,
        importance_types=["split", "gain"],
    )

    assert drops["random"] == {"weak"}
    assert drops["zero"] == {"zero"}
    assert drops["thresholds"] == {"medium", "random_col"}
    survives = set(detail.loc[detail["survives"], "feature"])
    assert survives == {"strong"}
    assert "random_col" not in set(detail["feature"])


def test_feature_select_v2_threshold_cumsum_includes_random_column():
    importance = pd.DataFrame(
        {
            "feature": ["strong", "medium", "random_col"],
            "split": [10, 5, 1],
            "gain": [80, 15, 5],
        }
    )

    drops, detail = select_feature_select_v2_drops(
        importance,
        "random_col",
        thresholds=0.8,
        importance_types=["gain"],
    )

    assert drops["thresholds"] == {"medium", "random_col"}
    assert set(detail.loc[detail["dropped"], "feature"]) == {"medium"}


def test_d03_feature_select_v2_uses_union_of_bagging_drops(monkeypatch):
    class FakeModel:
        def __init__(self, features):
            self.features = features

        def feature_importance(self, importance_type):
            values = {
                "strong": {"split": 10, "gain": 100},
                "weak": {"split": 1, "gain": 6},
                "random_col": {"split": 2, "gain": 5},
            }
            return [values[feature][importance_type] for feature in self.features]

    def fake_train(train_x, train_y, features, cfg, seed):
        return FakeModel(features), 0.5

    monkeypatch.setattr(feature_refine, "train_feature_select_v2_model", fake_train)
    parts = DatasetParts(
        train_x=pd.DataFrame({"strong": [1, 2, 3, 4], "weak": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"strong": [1, 2], "weak": [2, 1]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = {
        "random_seed": 0,
        "d03_random_importance": {
            "enabled": True,
            "mode": "feature_select_v2",
            "bagging_rounds": 2,
            "bagging_fraction": 1.0,
            "thresholds": None,
            "importance_types": ["split", "gain"],
        },
    }

    kept, detail = d03_random_importance(parts, ["strong", "weak"], cfg)

    assert kept == ["strong"]
    assert set(detail["feature"]) == {"strong", "weak"}
    assert set(detail.loc[detail["dropped"], "feature"]) == {"weak"}


def test_d03_progress_uses_user_friendly_labels(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, features):
            self.features = features

        def feature_importance(self, importance_type):
            values = {
                "strong": {"split": 10, "gain": 100},
                "weak": {"split": 1, "gain": 6},
                "random_col": {"split": 2, "gain": 5},
            }
            return [values[feature][importance_type] for feature in self.features]

    def fake_train(train_x, train_y, features, cfg, seed):
        return FakeModel(features), 0.5

    monkeypatch.setattr(feature_refine, "train_feature_select_v2_model", fake_train)
    parts = DatasetParts(
        train_x=pd.DataFrame({"strong": [1, 2, 3, 4], "weak": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"strong": [1, 2], "weak": [2, 1]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = {
        "random_seed": 0,
        "d03_random_importance": {
            "enabled": True,
            "mode": "feature_select_v2",
            "bagging_rounds": 1,
            "bagging_fraction": 1.0,
            "iter_rounds": 1,
            "thresholds": None,
            "importance_types": ["split", "gain"],
        },
    }
    reporter = ProgressReporter(tmp_path / "run1", "feature_refine", emit_terminal=False)

    kept, _ = d03_random_importance(parts, ["strong", "weak"], cfg, progress=reporter)

    assert kept == ["strong"]
    events = load_progress_events(tmp_path / "run1")
    messages = [event["message"] for event in events]
    assert any("随机重要性筛选" in message for message in messages)
    assert not any("D03" in message for message in messages)
    assert any(event["step"] == "d03_v2_iteration_start" for event in events)


def test_d03_noise_survival_emits_start_progress_without_internal_label(monkeypatch, tmp_path):
    class FakeModel:
        def __init__(self, features):
            self.features = features

        def feature_importance(self, importance_type):
            values = []
            for feature in self.features:
                if feature == "strong":
                    values.append(10)
                elif feature.startswith("__random_noise"):
                    values.append(1)
                else:
                    values.append(0)
            return values

    def fake_train(parts, features, cfg, seed):
        return FakeModel(features), 0.7

    monkeypatch.setattr(feature_refine, "train_lgbm", fake_train)
    parts = DatasetParts(
        train_x=pd.DataFrame({"strong": [1, 2, 3, 4], "weak": [4, 3, 2, 1]}),
        train_y=pd.Series([0, 1, 0, 1]),
        valid_x=pd.DataFrame({"strong": [1, 2], "weak": [2, 1]}),
        valid_y=pd.Series([0, 1]),
    )
    cfg = {
        "random_seed": 0,
        "d03_random_importance": {
            "enabled": True,
            "mode": "noise_survival",
            "rounds": 1,
            "random_feature_count": 1,
            "min_survival_rate": 1.0,
        },
    }
    reporter = ProgressReporter(tmp_path / "run1", "feature_refine", emit_terminal=False)

    kept, _ = d03_random_importance(parts, ["strong", "weak"], cfg, progress=reporter)

    assert kept == ["strong"]
    events = load_progress_events(tmp_path / "run1")
    messages = [event["message"] for event in events]
    assert any(event["step"] == "d03_round_start" for event in events)
    assert any("随机重要性筛选第 1/1 轮开始" in message for message in messages)
    assert not any("D03" in message for message in messages)


def test_refine_main_emits_friendly_progress_funnel(monkeypatch, tmp_path, capsys):
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    feature_map = project / "feature_map.csv"
    feature_map.write_text(
        "output_feature\nf1\nf2\nf3\n",
        encoding="utf-8",
    )
    (config_dir / "refine_features.yaml").write_text(
        "\n".join(
            [
                "feature_refine:",
                "  output_dir: feature_selection/refine",
                "  random_seed: 0",
                "  selection_split:",
                "    mode: explicit",
                "    train_values: [train]",
                "    valid_values: [valid]",
                "  input:",
                "    wide_table: demo.wide",
                "    feature_map: feature_map.csv",
                "    base_columns: [id, label, split]",
                "    id_columns: [id]",
                "    label_column: label",
                "    split_column: split",
                "    train_value: train",
                "    valid_value: valid",
                "  sampling: {}",
                "  preprocessing:",
                "    missing_sentinels: []",
                "    min_non_null_rate: 0.0",
                "    drop_constant: false",
                "    max_unique_values: 1",
                "  global_corr:",
                "    enabled: true",
                "    threshold: 0.8",
                "  local_d01:",
                "    enabled: true",
                "  local_d02:",
                "    enabled: true",
                "  d03_random_importance:",
                "    enabled: true",
                "  d04_null_importance:",
                "    enabled: true",
                "  d05_baseline_importance:",
                "    enabled: true",
                "    keep_top_n: 1",
                "  lightgbm: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    raw_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "label": [0, 1, 0, 1],
            "split": ["train", "train", "valid", "valid"],
            "f1": [0.1, 0.9, 0.2, 0.8],
            "f2": [1.0, 2.0, 1.5, 2.5],
            "f3": [5.0, 4.0, 3.0, 2.0],
        }
    )

    monkeypatch.setattr(feature_refine, "load_or_fetch_dp_feather", lambda **kwargs: raw_df)
    monkeypatch.setattr(
        feature_refine,
        "d01_local_prescreen",
        lambda parts, features, cfg, progress=None: (["f1", "f2"], pd.DataFrame({"feature": features, "drop_reason": ["kept", "kept", "low_iv"]})),
    )
    monkeypatch.setattr(
        feature_refine,
        "d02_local_psi",
        lambda parts, features, cfg, progress=None: (["f1"], pd.DataFrame({"feature": features, "drop_reason": ["kept", "high_psi"]})),
    )
    monkeypatch.setattr(feature_refine, "global_corr_select", lambda train_x, train_y, cfg, progress=None: (["f1"], pd.DataFrame()))
    monkeypatch.setattr(feature_refine, "d03_random_importance", lambda parts, features, cfg, progress=None: (["f1"], pd.DataFrame({"feature": ["f1"]})))
    monkeypatch.setattr(feature_refine, "d04_null_importance", lambda parts, features, cfg, progress=None: (["f1"], pd.DataFrame({"feature": ["f1"], "survives": [True]})))
    monkeypatch.setattr(
        feature_refine,
        "d05_top_importance",
        lambda parts, features, cfg, progress=None: (
            ["f1"],
            pd.DataFrame({"feature": ["f1"], "gain": [1.0], "split": [1], "rank": [1]}),
            0.75,
        ),
    )
    run_dir = project / "versions" / "v1"

    assert feature_refine.main(["--project-dir", str(project), "--run-dir", str(run_dir)]) == 0
    terminal_output = capsys.readouterr().out

    events = load_progress_events(run_dir)
    messages = [event["message"] for event in events]
    assert any("基础质量筛选开始" in message for message in messages)
    assert any("稳定性筛选完成" in message for message in messages)
    assert any("全局相关性去重开始" in message for message in messages)
    assert not any("D03" in message or "D04" in message or "D05" in message for message in messages)
    assert "after_d01" not in terminal_output
    assert "after_d02" not in terminal_output
    assert "after_d03" not in terminal_output
    assert "D03" not in terminal_output
    funnel = events[-1]["metrics"]["feature_funnel"]
    assert funnel == {
        "initial": 3,
        "available_after_preprocess": 3,
        "after_quality_filter": 2,
        "after_stability_filter": 1,
        "after_global_correlation": 1,
        "after_random_importance": 1,
        "after_null_importance": 1,
        "final": 1,
    }
