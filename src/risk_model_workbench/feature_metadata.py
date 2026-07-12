from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from risk_model_workbench.config import load_yaml
from risk_model_workbench.paths import project_config_path, stage_config_path
from risk_model_workbench.progress import ProgressReporter

DEFAULT_BASE_SAMPLE_COLUMNS = {
    "sample_row_num",
    "uid",
    "sample_date",
    "sample_month",
    "target",
    "final_flag",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Dataphin feature table metadata.")
    parser.add_argument(
        "--project-dir",
        default=str(Path.cwd()),
        help="Project workspace directory.",
    )
    parser.add_argument(
        "--tables-file",
        default="configs/feature_tables.txt",
        help="Feature table list, relative to project-dir unless absolute.",
    )
    parser.add_argument("--config", default=None, help="Feature selection config path.")
    parser.add_argument("--project-config", default=None, help="Project config path.")
    parser.add_argument("--run-dir", default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _resolve(project_dir: Path, value: str | None, default: Path) -> Path:
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else project_dir / path


def load_metadata_options(project_dir: Path, tables_file_arg: str, *, config: str | None = None, project_config: str | None = None) -> dict:
    project_path = _resolve(project_dir, project_config, project_config_path(project_dir))
    feature_path = _resolve(project_dir, config, stage_config_path(project_dir, "feature_select"))
    project_config = load_yaml(project_path)
    feature_config = load_yaml(feature_path).get("feature_select", {})
    metadata_config = feature_config.get("metadata", {}) or {}
    data_config = project_config.get("data", {}) or {}
    split_config = project_config.get("split", {}) or {}

    explicit_sample_columns = set(metadata_config.get("sample_columns", []) or [])
    configured_excludes = set(explicit_sample_columns)
    configured_excludes.update(data_config.get("id_columns", []) or [])
    for key in ["time_column", "period_column", "target_column"]:
        if data_config.get(key):
            configured_excludes.add(data_config[key])
    if split_config.get("source_column"):
        configured_excludes.add(split_config["source_column"])
    configured_excludes.update(data_config.get("base_columns", []) or [])
    configured_excludes.update(metadata_config.get("exclude_columns", []) or [])
    if not configured_excludes:
        configured_excludes.update(DEFAULT_BASE_SAMPLE_COLUMNS)

    return {
        "project_display_name": project_config.get("project", {}).get("display_name", project_dir.name),
        "tables_file": metadata_config.get("tables_file") or tables_file_arg,
        "output_dir": metadata_config.get("output_dir", "data/profile/feature_metadata"),
        "docs_path": metadata_config.get("docs_path", "docs/特征表清单.md"),
        "sample_columns": configured_excludes,
    }


def read_tables(path: Path) -> tuple[str, list[str]]:
    tables: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tables.append(line)
    if len(tables) < 2:
        raise ValueError(f"Expected sample table plus feature tables in {path}")
    return tables[0], tables[1:]


def split_table(full_name: str) -> tuple[str, str]:
    if "." not in full_name:
        raise ValueError(f"Table name must be project.table: {full_name}")
    project, table = full_name.split(".", 1)
    return project, table


def meta_data(raw: dict) -> dict:
    data = raw.get("data", raw)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected metadata payload: {raw}")
    return data


def normalize_columns(columns: list) -> list[dict]:
    normalized = []
    for idx, column in enumerate(columns, start=1):
        if isinstance(column, dict):
            name = column.get("name") or column.get("columnName") or ""
            dtype = column.get("type") or column.get("dataType") or ""
            comment = column.get("comment") or column.get("description") or ""
        else:
            name = str(column)
            dtype = ""
            comment = ""
        normalized.append(
            {
                "ordinal": idx,
                "name": name,
                "type": dtype,
                "comment": comment,
            }
        )
    return normalized


class _TMLMetaClient:
    """Adapter that uses TMLSQLClient + DESC to provide get_table_meta()."""

    def __init__(self):
        from tmlpatch.database import TMLSQLClient

        self._client = TMLSQLClient()

    def get_table_meta(self, full_table_name: str) -> dict:
        sql = f"DESC {full_table_name}"
        df = self._client.sql(sql).to_pandas()
        return self._parse_desc(df, full_table_name)

    @staticmethod
    def _parse_desc(df, full_table_name: str) -> dict:
        lines: list[str] = []
        for _, row in df.iterrows():
            val = row["result"]
            if isinstance(val, list):
                lines.append(str(val[0]) if val else "")
            else:
                lines.append(str(val))

        columns: list[dict] = []
        table_des = ""
        project_name = ""

        in_columns = False
        past_header_sep = False
        for line in lines:
            line_s = line.strip()

            # Extract table-level metadata from header lines
            if "|" in line_s and not in_columns:
                parts = [p.strip() for p in line_s.split("|")]
                for part in parts:
                    if part.startswith("TableComment:"):
                        table_des = part.split(":", 1)[1].strip()
                    elif part.startswith("Project:"):
                        project_name = part.split(":", 1)[1].strip()

            # Detect start of column definitions
            if "| Field" in line_s or "|Field" in line_s:
                in_columns = True
                past_header_sep = False
                continue

            if not in_columns:
                continue

            # Skip the separator line between header and data rows
            if not past_header_sep and line_s.startswith("+--"):
                past_header_sep = True
                continue

            if not past_header_sep:
                continue

            # End of column definitions (next separator after data)
            if line_s.startswith("+--"):
                break

            if not line_s.startswith("|"):
                break

            # Parse: | name | type | label | comment |
            parts = [p.strip() for p in line_s.split("|")]
            parts = [p for p in parts if p]  # remove empty strings from leading/trailing |

            if len(parts) >= 2:
                col_name = parts[0]
                col_type = parts[1]
                col_comment = parts[3] if len(parts) > 3 else ""
                if col_comment == "null":
                    col_comment = ""
                columns.append(
                    {
                        "name": col_name,
                        "type": col_type,
                        "comment": col_comment,
                    }
                )

        project_hierarchy = full_table_name.rsplit(".", 1)[0] if "." in full_table_name else ""
        return {
            "columns": columns,
            "des": table_des,
            "name": full_table_name,
            "projectName": project_name or project_hierarchy,
            "partitioned": False,
            "partitions": [],
            "tableType": "",
        }


class _DpMcpMetaClient:
    """Meta client via sh_dp_mcp HTTP MCP (``get_dp_table_meta``).

    No tmlpatch dependency; returns the same ``{columns, des, name}`` shape as
    ``_TMLMetaClient._parse_desc`` so ``main()`` consumes it unchanged.
    """

    def __init__(self, url: str, api_key: str, timeout: int = 30):
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

    def get_table_meta(self, full_table_name: str) -> dict:
        project, table = split_table(full_table_name)
        payload = json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "id": 1,
                "params": {
                    "name": "get_dp_table_meta",
                    "arguments": {"projectName": project, "tableName": table},
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url,
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-CJJ-MCP-API-KEY": self._api_key},
        )
        with urllib.request.urlopen(request, timeout=self._timeout) as response:
            raw = response.read().decode("utf-8")
        envelope = json.loads(raw)
        result = envelope.get("result") or {}
        if result.get("isError"):
            raise RuntimeError(f"sh_dp_mcp returned an error result for {full_table_name}")
        text_blobs = result.get("content") or []
        if not text_blobs:
            raise RuntimeError(f"sh_dp_mcp returned no content for {full_table_name}")
        data = json.loads(text_blobs[0].get("text", "{}")).get("data") or {}
        columns = [
            {
                "name": column.get("name", ""),
                "type": column.get("type", ""),
                "comment": column.get("comment") or "",
            }
            for column in data.get("columns", [])
        ]
        return {
            "columns": columns,
            "des": data.get("des") or "",
            "name": data.get("name") or full_table_name,
        }


class _FallbackMetaClient:
    """Try the primary meta client; on failure, fall back per-table."""

    def __init__(self, primary, fallback):
        self._primary = primary
        self._fallback = fallback

    def get_table_meta(self, full_table_name: str) -> dict:
        try:
            return self._primary.get_table_meta(full_table_name)
        except Exception as exc:
            if self._fallback is None:
                raise
            print(
                f"[feature_metadata] MCP meta client failed for {full_table_name} ({exc}); "
                "falling back to TMLSQLClient for this table.",
                file=sys.stderr,
            )
            return self._fallback.get_table_meta(full_table_name)


def _resolve_dp_mcp_config():
    """Return ``(url, api_key)`` for sh_dp_mcp from env or ``~/.claude.json``, else ``None``.

    Priority: ``SH_DP_MCP_URL`` + ``SH_DP_MCP_API_KEY`` env vars, then
    ``~/.claude.json`` ``mcpServers.sh_dp_mcp`` (auto-probe a Claude-configured env).
    """
    url = os.environ.get("SH_DP_MCP_URL")
    api_key = os.environ.get("SH_DP_MCP_API_KEY")
    if url and api_key:
        return url, api_key
    claude_cfg = Path.home() / ".claude.json"
    if claude_cfg.exists():
        try:
            cfg = json.loads(claude_cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        mcp = (cfg.get("mcpServers") or {}).get("sh_dp_mcp") or {}
        url = url or mcp.get("url")
        headers = mcp.get("headers") or {}
        api_key = api_key or headers.get("X-CJJ-MCP-API-KEY")
    if url and api_key:
        return url, api_key
    return None


def _legacy_meta_client():
    """Return dp_cli or TMLSQLClient meta client; raise ImportError if neither available."""
    try:
        from dp_cli import create_clients

        dp, _ = create_clients()
        return dp
    except ImportError:
        pass
    return _TMLMetaClient()  # raises ImportError if tmlpatch missing


def load_dp_client():
    mcp_cfg = _resolve_dp_mcp_config()
    if mcp_cfg:
        try:
            legacy = _legacy_meta_client()
        except ImportError:
            legacy = None  # MCP path only; no legacy fallback available
        print("[feature_metadata] meta source: sh_dp_mcp (MCP)", file=sys.stderr)
        if legacy is not None:
            return _FallbackMetaClient(_DpMcpMetaClient(*mcp_cfg), legacy)
        return _DpMcpMetaClient(*mcp_cfg)
    try:
        client = _legacy_meta_client()
    except ImportError as exc:
        raise SystemExit(
            "Neither sh_dp_mcp nor dp_cli/TMLSQLClient is available. "
            "Set SH_DP_MCP_URL + SH_DP_MCP_API_KEY (or configure ~/.claude.json "
            "mcpServers.sh_dp_mcp), or install tmlpatch/dp_cli."
        ) from exc
    print("[feature_metadata] meta source: TMLSQLClient", file=sys.stderr)
    return client


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    payload: dict,
    summary_rows: list[dict],
    sample_columns: list[dict],
    *,
    project_display_name: str,
) -> None:
    lines = [
        f"# {project_display_name}特征表清单",
        "",
        f"- 生成时间: {payload['generated_at']}",
        f"- 样本表: `{payload['sample_table']}`",
        f"- 特征表数量: {payload['feature_table_count']}",
        f"- 候选特征字段数: {payload['feature_column_count']}",
        f"- 失败表数量: {len(payload['failed_tables'])}",
        "",
        "## 样本字段",
        "",
        "| 字段 | 类型 | 注释 |",
        "| --- | --- | --- |",
    ]
    for col in sample_columns:
        lines.append(f"| `{col['name']}` | {col['type']} | {col['comment']} |")

    lines.extend(
        [
            "",
            "## 特征表汇总",
            "",
            "| 序号 | 表名 | 字段数 | 候选特征数 | 描述 |",
            "| ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['table_index']} | `{row['full_table_name']}` | "
            f"{row['column_count']} | {row['feature_column_count']} | {row['description']} |"
        )

    if payload["failed_tables"]:
        lines.extend(["", "## 失败表", ""])
        for item in payload["failed_tables"]:
            lines.append(f"- `{item['table']}`: {item['error']}")

    lines.extend(
        [
            "",
            "## 产物文件",
            "",
            "- `data/profile/feature_metadata/feature_tables_meta.json`: 完整表元数据",
            "- `data/profile/feature_metadata/feature_table_summary.csv`: 表级字段统计",
            "- `data/profile/feature_metadata/feature_columns.csv`: 字段级候选特征清单",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_metadata_service(
    *,
    project_dir: str | Path,
    tables_file: str = "configs/feature_tables.txt",
    config: str | None = None,
    project_config: str | None = None,
    run_dir: str | Path | None = None,
) -> int:
    """Typed metadata export service; independent of argparse and CLI entrypoints."""
    project_dir = Path(project_dir).resolve()
    reporter = ProgressReporter(run_dir, "feature_metadata") if run_dir else None
    options = load_metadata_options(project_dir, tables_file, config=config, project_config=project_config)
    tables_file = Path(options["tables_file"])
    if not tables_file.is_absolute():
        tables_file = project_dir / tables_file

    output_dir = Path(options["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir
    docs_path = Path(options["docs_path"])
    if not docs_path.is_absolute():
        docs_path = project_dir / docs_path

    sample_table, feature_tables = read_tables(tables_file)
    if reporter:
        reporter.emit(
            step="load_tables",
            message=f"读取特征表清单完成，共 {len(feature_tables)} 张特征表",
            current=0,
            total=len(feature_tables),
            percent=5,
        )
    dp = load_dp_client()

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_meta = {
        "generated_at": generated_at,
        "sample_table": sample_table,
        "feature_tables": {},
        "failed_tables": [],
    }
    summary_rows: list[dict] = []
    feature_rows: list[dict] = []

    print(f"[INFO] sample table: {sample_table}")
    if reporter:
        reporter.emit(step="sample_table", message=f"正在读取样本表元数据：{sample_table}", percent=8)
    sample_raw = dp.get_table_meta(sample_table)
    sample_meta = meta_data(sample_raw)
    sample_columns = normalize_columns(sample_meta.get("columns", sample_meta.get("fields", [])))
    all_meta["sample_meta"] = sample_meta
    sample_column_names = {col["name"] for col in sample_columns} | set(options["sample_columns"])

    success_count = 0
    for idx, full_table_name in enumerate(feature_tables, start=1):
        print(f"[INFO] ({idx}/{len(feature_tables)}) {full_table_name}", flush=True)
        if reporter:
            reporter.emit(
                step="table_metadata",
                message=f"表 {idx}/{len(feature_tables)}：正在读取元数据 {full_table_name}",
                current=idx - 1,
                total=len(feature_tables),
                metrics={"table": full_table_name, "success": success_count, "failed": len(all_meta["failed_tables"])},
            )
        try:
            raw = dp.get_table_meta(full_table_name)
            data = meta_data(raw)
            columns = normalize_columns(data.get("columns", data.get("fields", [])))
            feature_columns = [col for col in columns if col["name"] not in sample_column_names]

            all_meta["feature_tables"][full_table_name] = data
            summary_rows.append(
                {
                    "table_index": idx,
                    "full_table_name": full_table_name,
                    "project_name": split_table(full_table_name)[0],
                    "table_name": split_table(full_table_name)[1],
                    "description": data.get("des", ""),
                    "partitioned": data.get("partitioned", False),
                    "column_count": len(columns),
                    "feature_column_count": len(feature_columns),
                }
            )
            for col in feature_columns:
                feature_rows.append(
                    {
                        "table_index": idx,
                        "full_table_name": full_table_name,
                        "feature_name": col["name"],
                        "feature_type": col["type"],
                        "feature_comment": col["comment"],
                        "ordinal": col["ordinal"],
                    }
                )
            success_count += 1
            if reporter:
                reporter.emit(
                    step="table_metadata_done",
                    message=(
                        f"表 {idx}/{len(feature_tables)}：元数据完成，候选字段 {len(feature_columns)} 个，"
                        f"成功 {success_count} 张，失败 {len(all_meta['failed_tables'])} 张"
                    ),
                    current=idx,
                    total=len(feature_tables),
                    metrics={
                        "table": full_table_name,
                        "feature_columns": len(feature_columns),
                        "success": success_count,
                        "failed": len(all_meta["failed_tables"]),
                    },
                )
        except Exception as exc:  # noqa: BLE001 - keep export progressing across bad tables.
            all_meta["failed_tables"].append({"table": full_table_name, "error": str(exc)})
            print(f"[WARN] failed: {full_table_name}: {exc}", file=sys.stderr, flush=True)
            if reporter:
                reporter.emit(
                    step="table_metadata_failed",
                    status="failed",
                    message=f"表 {idx}/{len(feature_tables)}：元数据读取失败 {full_table_name}：{exc}",
                    current=idx,
                    total=len(feature_tables),
                    metrics={"table": full_table_name, "success": success_count, "failed": len(all_meta["failed_tables"])},
                    level="warning",
                )

    all_meta["feature_table_count"] = len(feature_tables)
    all_meta["feature_column_count"] = len(feature_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_tables_meta.json").write_text(
        json.dumps(all_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(
        output_dir / "feature_table_summary.csv",
        summary_rows,
        [
            "table_index",
            "full_table_name",
            "project_name",
            "table_name",
            "description",
            "partitioned",
            "column_count",
            "feature_column_count",
        ],
    )
    write_csv(
        output_dir / "feature_columns.csv",
        feature_rows,
        ["table_index", "full_table_name", "feature_name", "feature_type", "feature_comment", "ordinal"],
    )
    write_markdown(
        docs_path,
        all_meta,
        summary_rows,
        sample_columns,
        project_display_name=options["project_display_name"],
    )

    print(f"[OK] tables: {len(summary_rows)}/{len(feature_tables)}")
    print(f"[OK] feature columns: {len(feature_rows)}")
    print(f"[OK] output: {output_dir}")
    print(f"[OK] docs: {docs_path}")
    if reporter:
        reporter.emit(
            step="write_outputs",
            status="done" if not all_meta["failed_tables"] else "failed",
            message=(
                f"特征元数据导出完成：成功 {len(summary_rows)}/{len(feature_tables)} 张表，"
                f"候选字段 {len(feature_rows)} 个"
            ),
            current=len(feature_tables),
            total=len(feature_tables),
            percent=100,
            metrics={"success": len(summary_rows), "failed": len(all_meta["failed_tables"]), "feature_columns": len(feature_rows)},
        )
    if all_meta["failed_tables"]:
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_metadata_service(
        project_dir=args.project_dir,
        tables_file=args.tables_file,
        config=args.config,
        project_config=args.project_config,
        run_dir=args.run_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
