"""Feature screening process summaries for model project reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path, stage_config_path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _read_feature_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _read_top_features(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "rank": int(float(row.get("rank") or len(rows) + 1)),
                    "feature": row.get("feature", ""),
                    "gain": float(row.get("gain") or 0),
                    "split": int(float(row.get("split") or 0)),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _describe_d03(d03: dict[str, Any]) -> str:
    mode = str(d03.get("mode", "feature_select_v2"))
    if mode in {"feature_select_v2", "feature_select_v2_compatible", "v2"}:
        importance_types = "/".join(d03.get("importance_types", ["split", "gain"]))
        thresholds = d03.get("thresholds", d03.get("d03_thresholds", 0.95))
        threshold_text = "不启用累计重要性尾部阈值" if thresholds is None else f"累计重要性阈值 {float(thresholds):.2f}"
        return (
            "随机数重要性筛选（feature-select-v2兼容）："
            f"{int(d03.get('bagging_rounds', d03.get('d03_bagging_round', 5)))}轮bagging，"
            f"采样比例 {float(d03.get('bagging_fraction', d03.get('d03_bagging_fraction', 0.5))):.2f}，"
            f"单随机列，对比 {importance_types}，{threshold_text}"
        )
    return (
        "随机噪声重要性筛选："
        f"{int(d03.get('rounds', 3))}轮，{int(d03.get('random_feature_count', 5))}个随机噪声特征，"
        f"存活率 >= {float(d03.get('min_survival_rate', 0.5)):.2f}"
    )


def build_feature_screening_summary(project_dir: str | Path) -> dict[str, Any]:
    """Build a source-backed summary of the current completed screening flow."""
    project_path = Path(project_dir).resolve()
    project_config = load_yaml(project_config_path(project_path))
    feature_select_config = load_yaml(stage_config_path(project_path, "feature_select")).get("feature_select", {})
    refine_config = load_yaml(stage_config_path(project_path, "refine_features"))["feature_refine"]

    prescreen_summary_path = _first_existing(
        [
            project_path / "runs" / "feature_prescreen" / "results" / "prescreen_run_summary.json",
            project_path / "runs" / "d01_d02_batch_select" / "results" / "d01_d02_run_summary.json",
        ]
    )
    prescreen = _read_json(prescreen_summary_path)
    preferred_refine_dir = project_path / "runs" / "feature_refine_feather"
    configured_refine_dir = project_path / refine_config.get("output_dir", "runs/feature_refine_wide")
    refine_dir = configured_refine_dir if (configured_refine_dir / "stage_summary.json").exists() else preferred_refine_dir
    refine_summary_path = refine_dir / "stage_summary.json"
    refine_summary = _read_json(refine_summary_path)
    final_features_path = refine_dir / "final_500_features.txt"
    d05_importance_path = refine_dir / "d05_baseline_importance.csv"

    final_count = _read_feature_count(final_features_path)
    thresholds = feature_select_config.get("thresholds", {})
    prescreen_config = feature_select_config.get("prescreen", {}) or feature_select_config.get("d01_d02", {})
    train_value = prescreen_config.get("train_value", "DEV")
    quality_thresholds = (
        f"缺失率 < {float(thresholds.get('empty', 0.95)):.2f}，"
        f"相关性 < {float(thresholds.get('corr', 0.80)):.2f}，"
        f"IV >= {float(thresholds.get('iv', 0.005)):.3f}"
    )
    psi_threshold = f"{train_value}首个自然月为Base，warning阈值 {float(thresholds.get('psi', 0.10)):.2f}；仅生成证据"
    global_corr = refine_config["global_corr"]
    d03 = dict(refine_config["d03_random_importance"])
    if refine_summary.get("d03_mode"):
        d03["mode"] = refine_summary["d03_mode"]
    elif refine_dir.name == "feature_refine_feather":
        d03["mode"] = "noise_survival"
    d04 = refine_config["d04_null_importance"]
    d05 = refine_config["d05_baseline_importance"]

    initial_feature_count = int(prescreen["input_features"])
    initial_table_count = int(prescreen.get("tables", 0)) or len(feature_select_config.get("bigtable", {}).get("tables", []))
    quality_remain = int(prescreen.get("quality_remain", prescreen.get("d01_remain", 0)))
    prescreen_source = str(prescreen_summary_path.relative_to(project_path))
    screening_rows = [
        {
            "step": "初始",
            "method": f"初始候选变量总数：{initial_table_count}张特征表，共{initial_feature_count:,}个字段级候选变量",
            "remaining_features": initial_feature_count,
            "source": prescreen_source,
        },
        {
            "step": 1,
            "method": f"特征初筛-质量规则：{quality_thresholds}",
            "remaining_features": quality_remain,
            "source": prescreen_source,
        },
        {
            "step": 2,
            "method": f"特征初筛-稳定性规则：{psi_threshold}",
            "remaining_features": int(prescreen["final_remain"]),
            "source": prescreen_source,
        },
        {
            "step": 3,
            "method": (
                f"Feather观察样本可用特征：{int(refine_summary.get('total_rows', refine_summary.get('raw_rows', 0))):,}行，"
                "过滤缺失率过低和常量字段"
            ),
            "remaining_features": int(refine_summary["available_features"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 4,
            "method": f"全局相关性去重：相关性阈值 {float(global_corr['threshold']):.2f}，按单变量AUC保留更强特征",
            "remaining_features": int(refine_summary["after_global_corr"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 5,
            "method": _describe_d03(d03),
            "remaining_features": int(refine_summary["after_d03_random_importance"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 6,
            "method": (
                "空标签重要性筛选："
                f"{int(d04['null_rounds'])}轮空标签，空标签重要性{int(d04['null_percentile'])}分位，"
                f"score >= {float(d04['score_threshold']):.2f}"
            ),
            "remaining_features": int(refine_summary["after_d04_null_importance"]),
            "source": str(refine_summary_path.relative_to(project_path)),
        },
        {
            "step": 7,
            "method": (
                "基线模型重要性筛选：LightGBM gain importance，"
                f"保留前{int(d05['keep_top_n'])}个，valid AUC={float(refine_summary['d05_valid_auc']):.4f}"
            ),
            "remaining_features": final_count,
            "source": str(final_features_path.relative_to(project_path)),
        },
    ]

    return {
        "project": {
            "name": project_config["project"]["name"],
            "display_name": project_config["project"]["display_name"],
            "scenario": project_config["project"]["scenario"],
        },
        "basis": {
            "primary_refine_source": refine_dir.name,
            "feather_path": refine_summary.get("feather_path", ""),
            "initial_feature_count": initial_feature_count,
            "sample_rows": int(refine_summary.get("total_rows", refine_summary.get("raw_rows", 0))),
            "train_samples": int(refine_summary.get("train_samples", 0)),
            "valid_samples": int(refine_summary.get("valid_samples", 0)),
            "d05_valid_auc": float(refine_summary["d05_valid_auc"]),
            "stage_summary": str(refine_summary_path.relative_to(project_path)),
        },
        "feature_select_v2_alignment": {
            "status": (
                "d03_feature_select_v2_compatible"
                if str(d03.get("mode", "feature_select_v2")) in {"feature_select_v2", "feature_select_v2_compatible", "v2"}
                else "concept_aligned_not_exact_reimplementation"
            ),
            "summary": _describe_d03(d03),
            "local_reference_paths": [
                "my-skills/develop/feature-select-v2",
                "vendor/feature-select-v2",
            ],
            "current_project_config": "configs/refine_features.yaml",
        },
        "screening_rows": screening_rows,
        "top_features": _read_top_features(d05_importance_path),
    }


def write_feature_screening_summary(project_dir: str | Path, output_path: str | Path) -> Path:
    project_path = Path(project_dir).resolve()
    path = Path(output_path)
    resolved = path if path.is_absolute() else project_path / path
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(build_feature_screening_summary(project_path), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return resolved
