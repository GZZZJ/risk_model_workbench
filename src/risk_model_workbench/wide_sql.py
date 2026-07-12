"""Build MaxCompute SQL for joining selected feature tables into one wide table."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path as resolve_project_config_path, stage_config_path


DEFAULT_BASE_COLUMNS = [
    "uid",
    "sample_date",
    "sample_month",
    "target",
    "final_flag",
]

DEFAULT_JOIN_KEYS = ["uid", "mdl_dte"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_]+$")
_BARE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


@dataclass(frozen=True)
class WideFeatureRecord:
    table_index: int
    table_name: str
    table_alias: str
    source_feature: str
    output_feature: str
    source_feature_count: int


def resolve_project_path(project_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def require_identifier(value: str, *, kind: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(f"Unsupported {kind} identifier: {value!r}")
    return value


def sql_identifier(value: str) -> str:
    value = require_identifier(value, kind="SQL")
    return value if _BARE_IDENTIFIER_RE.fullmatch(value) else f"`{value}`"


def qualified(alias: str, column: str) -> str:
    return f"{alias}.{sql_identifier(column)}"


def require_table_name(value: str, *, kind: str) -> str:
    if not _TABLE_RE.fullmatch(value):
        raise ValueError(f"Unsupported {kind} table name: {value!r}")
    return value


def table_alias(index: int) -> str:
    return f"t{index + 1}"


def table_short_name(table_name: str) -> str:
    name = table_name.split(".")[-1]
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")
    return name or "feature_table"


def load_remain_features(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in {path}")

    remain: dict[str, list[str]] = {}
    for table_name, features in payload.items():
        require_table_name(table_name, kind="feature")
        if not isinstance(features, list):
            raise ValueError(f"Expected feature list for {table_name}")
        clean_features = []
        for feature in features:
            if not isinstance(feature, str):
                raise ValueError(f"Feature name must be string in {table_name}: {feature!r}")
            clean_features.append(require_identifier(feature, kind="feature"))
        if clean_features:
            remain[table_name] = clean_features
    return remain


def make_feature_records(table_features: dict[str, list[str]]) -> list[WideFeatureRecord]:
    name_counts = Counter(feature for features in table_features.values() for feature in features)
    records: list[WideFeatureRecord] = []
    used_outputs: set[str] = set()

    for table_index, (table_name, features) in enumerate(table_features.items(), start=1):
        alias = table_alias(table_index)
        short_name = table_short_name(table_name)
        for feature in features:
            if name_counts[feature] == 1:
                output_feature = feature
            else:
                output_feature = f"{short_name}__{feature}"
                if len(output_feature) > 120:
                    output_feature = f"f{table_index:02d}__{feature}"

            candidate = output_feature
            suffix = 2
            while candidate in used_outputs:
                candidate = f"{output_feature}__{suffix}"
                suffix += 1
            used_outputs.add(candidate)
            records.append(
                WideFeatureRecord(
                    table_index=table_index,
                    table_name=table_name,
                    table_alias=alias,
                    source_feature=feature,
                    output_feature=require_identifier(candidate, kind="output feature"),
                    source_feature_count=name_counts[feature],
                )
            )

    return records


def indent(lines: list[str], prefix: str = "  ") -> list[str]:
    return [prefix + line for line in lines]


def comma_lines(items: list[str], prefix: str = "  ") -> list[str]:
    return [f"{prefix}{item}{',' if index < len(items) - 1 else ''}" for index, item in enumerate(items)]


def base_select_lines(base_alias: str, base_columns: list[str]) -> list[str]:
    return [qualified(base_alias, column) for column in base_columns]


def feature_select_lines(records: list[WideFeatureRecord]) -> list[str]:
    lines = []
    for record in records:
        expr = qualified(record.table_alias, record.source_feature)
        if record.output_feature != record.source_feature:
            expr += f" as {sql_identifier(record.output_feature)}"
        lines.append(expr)
    return lines


def subquery_select(columns: list[str], table_name: str, where_clause: str | None) -> list[str]:
    lines = ["select"]
    lines.extend(comma_lines([sql_identifier(column) for column in columns]))
    lines.append(f"from {require_table_name(table_name, kind='source')}")
    if where_clause:
        lines.append(f"where {where_clause}")
    return lines


def build_wide_table_sql(
    *,
    base_table: str,
    output_table: str,
    base_columns: list[str],
    join_keys: list[str],
    table_features: dict[str, list[str]],
    base_where: str | None = None,
    feature_where: str | None = None,
) -> tuple[str, list[WideFeatureRecord]]:
    require_table_name(base_table, kind="base")
    require_table_name(output_table, kind="output")
    join_keys = [require_identifier(key, kind="join key") for key in join_keys]
    base_columns = [require_identifier(column, kind="base column") for column in base_columns]

    for key in join_keys:
        if key not in base_columns:
            base_columns.insert(0, key)

    records = make_feature_records(table_features)
    records_by_table: dict[str, list[WideFeatureRecord]] = {}
    for record in records:
        records_by_table.setdefault(record.table_name, []).append(record)

    selected_columns = base_select_lines("t1", base_columns) + feature_select_lines(records)
    lines = [
        "-- Generated by risk_model_workbench.",
        f"-- feature_tables={len(table_features)}, features={len(records)}",
        f"create table if not exists {output_table} as",
        "select",
    ]
    lines.extend(comma_lines(selected_columns))

    base_subquery = subquery_select(base_columns, base_table, base_where)
    lines.append("from (")
    lines.extend(indent(base_subquery))
    lines.append(") t1")

    for table_name, table_records in records_by_table.items():
        alias = table_records[0].table_alias
        feature_columns = [record.source_feature for record in table_records]
        subquery_columns = join_keys + [feature for feature in feature_columns if feature not in join_keys]
        lines.append("left join (")
        lines.extend(indent(subquery_select(subquery_columns, table_name, feature_where)))
        lines.append(f") {alias}")
        join_expr = " and ".join(f"{qualified('t1', key)} = {qualified(alias, key)}" for key in join_keys)
        lines.append(f"  on {join_expr}")

    lines.append(";")
    return "\n".join(lines) + "\n", records


def write_feature_map(path: Path, records: list[WideFeatureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_feature",
                "source_feature",
                "source_table",
                "table_index",
                "table_alias",
                "source_feature_count",
                "is_duplicate_source_name",
                "is_output_renamed",
                "rename_reason",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "output_feature": record.output_feature,
                    "source_feature": record.source_feature,
                    "source_table": record.table_name,
                    "table_index": record.table_index,
                    "table_alias": record.table_alias,
                    "source_feature_count": record.source_feature_count,
                    "is_duplicate_source_name": record.source_feature_count > 1,
                    "is_output_renamed": record.output_feature != record.source_feature,
                    "rename_reason": "duplicate_source_name"
                    if record.source_feature_count > 1 and record.output_feature != record.source_feature
                    else "",
                }
            )


def write_summary(
    path: Path,
    *,
    sql_path: Path,
    feature_map_path: Path,
    output_table: str,
    records: list[WideFeatureRecord],
) -> None:
    duplicate_sources = sum(1 for record in records if record.output_feature != record.source_feature)
    summary = {
        "sql_path": str(sql_path),
        "feature_map_path": str(feature_map_path),
        "output_table": output_table,
        "feature_tables": len({record.table_name for record in records}),
        "features": len(records),
        "renamed_duplicate_features": duplicate_sources,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_wide_defaults(project_dir: Path, *, config_path: Path | None = None, project_config_path: Path | None = None) -> dict[str, Any]:
    project_config = load_yaml(project_config_path or resolve_project_config_path(project_dir))
    feature_config = load_yaml(config_path or stage_config_path(project_dir, "feature_select")).get("feature_select", {})
    wide_config = feature_config.get("wide_table", {}) or {}

    return {
        "base_table": wide_config.get("base_table") or project_config.get("data", {}).get("source_table"),
        "output_table": wide_config.get("output_table"),
        "join_keys": wide_config.get("join_keys") or project_config.get("data", {}).get("id_columns") or DEFAULT_JOIN_KEYS,
        "base_columns": wide_config.get("base_columns") or DEFAULT_BASE_COLUMNS,
        "base_where": wide_config.get("base_where"),
        "feature_where": wide_config.get("feature_where"),
    }


def generate_wide_sql(
    *,
    project_dir: Path,
    remain_features_path: Path,
    sql_output_path: Path,
    feature_map_path: Path,
    summary_path: Path,
    base_table: str | None = None,
    output_table: str | None = None,
    base_where: str | None = None,
    feature_where: str | None = None,
    config_path: Path | None = None,
    project_config_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    defaults = project_wide_defaults(project_dir, config_path=config_path, project_config_path=project_config_path)
    resolved_base_table = base_table or defaults["base_table"]
    resolved_output_table = output_table or defaults["output_table"]
    if not resolved_base_table:
        raise ValueError("base_table is required. Set feature_select.wide_table.base_table or pass --base-table.")
    if not resolved_output_table:
        raise ValueError("output_table is required. Set feature_select.wide_table.output_table or pass --output-table.")

    table_features = load_remain_features(remain_features_path)
    sql, records = build_wide_table_sql(
        base_table=resolved_base_table,
        output_table=resolved_output_table,
        base_columns=list(defaults["base_columns"]),
        join_keys=list(defaults["join_keys"]),
        table_features=table_features,
        base_where=base_where if base_where is not None else defaults["base_where"],
        feature_where=feature_where if feature_where is not None else defaults["feature_where"],
    )

    sql_output_path.parent.mkdir(parents=True, exist_ok=True)
    sql_output_path.write_text(sql, encoding="utf-8")
    write_feature_map(feature_map_path, records)
    write_summary(
        summary_path,
        sql_path=sql_output_path,
        feature_map_path=feature_map_path,
        output_table=resolved_output_table,
        records=records,
    )
    return sql_output_path, feature_map_path, summary_path
