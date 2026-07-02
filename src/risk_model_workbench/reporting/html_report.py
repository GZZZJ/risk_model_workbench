"""Standalone HTML renderer for model report Markdown."""

from __future__ import annotations

import re
import time
from html import escape
from pathlib import Path
from typing import Any


DEFAULT_REPORT_TITLE = "Model Report"


def _markdown_to_simple_html(markdown: str) -> str:
    """Compatibility wrapper for callers that still use the old renderer name."""
    return render_model_report_html(markdown)


def render_model_report_html(
    markdown: str,
    *,
    title: str | None = None,
    run_config: dict[str, Any] | None = None,
    eval_dir: Path | None = None,
    include_gcard_summary: bool = False,
    run_id: str | None = None,
    default_title: str = DEFAULT_REPORT_TITLE,
    score_labels: dict[str, str] | None = None,
) -> str:
    """Render the model report with the same dashboard layout used by GCard runs."""
    run_config = run_config or {}
    score_labels = score_labels or {}
    report_title = title or _markdown_title(markdown) or default_title
    generated_at = _markdown_generated_date(markdown) or time.strftime("%Y-%m-%d")
    sidebar_meta = _sidebar_meta(generated_at, include_gcard_summary=include_gcard_summary, score_labels=score_labels)
    hero_meta = _hero_meta(generated_at=generated_at, run_id=run_id, run_config=run_config)
    body = _markdown_body_to_report_html(
        markdown,
        include_gcard_summary=include_gcard_summary,
        run_config=run_config,
        eval_dir=eval_dir,
        score_labels=score_labels,
    )
    nav_items = _report_nav_items(markdown, include_gcard_summary=include_gcard_summary)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{escape(report_title)}</title>",
            "<style>",
            _report_html_style(),
            "</style>",
            "</head><body>",
            '<aside class="sidebar" aria-label="报告导航">',
            f'  <div class="sidebar-title">{escape(report_title)}</div>',
            f'  <div class="sidebar-meta">{sidebar_meta}</div>',
            '  <nav class="nav-list">',
            *[
                f'    <a class="nav-item" href="#{item_id}"><span class="nav-index">{idx}</span><span>{escape(label)}</span></a>'
                for idx, (item_id, label) in enumerate(nav_items, start=1)
            ],
            "  </nav>",
            '  <div class="nav-note">表格已自适应页面宽度；提升类指标按正负变化着色。</div>',
            "</aside>",
            '<main class="report-shell">',
            '<header class="report-hero">',
            "  <p class=\"report-eyebrow\">风险场景 AI 建模工作台 · 模型文档</p>",
            f"  <h1>{escape(report_title)}</h1>",
            '  <div class="hero-meta">',
            *[f"    <span>{_inline_markdown_to_html(item)}</span>" for item in hero_meta],
            "  </div>",
            "</header>",
            '<div class="report-body">',
            body,
            "</div>",
            "</main>",
            "<script>",
            _report_html_script(),
            "</script>",
            "</body></html>",
        ]
    )


def _markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _markdown_generated_date(markdown: str) -> str | None:
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("生成日期："):
            return stripped.split("：", 1)[1].strip()
    return None


def _sidebar_meta(generated_at: str, *, include_gcard_summary: bool, score_labels: dict[str, str]) -> str:
    if include_gcard_summary:
        compare_label = score_labels.get("gcard_v6", "G卡V6")
        return f"{escape(generated_at)}<br>model_score vs {escape(compare_label)}"
    return escape(generated_at)


def _hero_meta(*, generated_at: str, run_id: str | None, run_config: dict[str, Any]) -> list[str]:
    items = [f"生成日期：{generated_at}"]
    if run_id:
        items.append(f"Run：{run_id}")
    label_column = run_config.get("label_column")
    if label_column:
        items.append(f"标签：`{label_column}`")
    algorithm = run_config.get("algorithm")
    if algorithm:
        items.append(f"算法：{algorithm}")
    return items


def _report_nav_items(markdown: str, *, include_gcard_summary: bool) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for line in markdown.splitlines():
        if not line.startswith("## "):
            continue
        title = _strip_heading_order_prefix(line[3:].strip())
        if include_gcard_summary and title.startswith("Summary"):
            title = "总结"
        label = _nav_label(title)
        item_id = _section_id_for_title(title)
        if item_id and (item_id, label) not in items:
            items.append((item_id, label))
    return items


def _nav_label(title: str) -> str:
    if "Summary" in title or "总结" in title:
        return "总结"
    if "模型描述" in title:
        return "模型描述"
    if "变量筛选" in title:
        return "变量筛选"
    if "核心效果" in title:
        return "核心对比"
    if "模型效果" in title:
        return "模型效果"
    if "模型稳定性" in title:
        return "模型稳定性"
    if "重要变量" in title:
        return "重要变量"
    if "Top变量WOE" in title:
        return "Top变量WOE"
    if "待补充" in title:
        return "待补充事项"
    return title


def _section_id_for_title(title: str) -> str:
    if "Summary" in title or "总结" in title:
        return "summary"
    if "模型描述" in title:
        return "model-description"
    if "变量筛选" in title:
        return "feature-selection"
    if "核心效果" in title:
        return "core-comparison"
    if "模型效果" in title:
        return "model-performance"
    if "模型稳定性" in title:
        return "model-stability"
    if "重要变量" in title:
        return "important-features"
    if "Top变量WOE" in title:
        return "top-woe"
    if "待补充" in title:
        return "missing-results"
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "section"


def _markdown_body_to_report_html(
    markdown: str,
    *,
    include_gcard_summary: bool,
    run_config: dict[str, Any],
    eval_dir: Path | None,
    score_labels: dict[str, str],
) -> str:
    html_lines: list[str] = []
    in_ul = False
    in_table = False
    in_image_grid = False
    skip_summary = False

    def close_ul() -> None:
        nonlocal in_ul
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False

    def close_table() -> None:
        nonlocal in_table
        if in_table:
            html_lines.append("</table>")
            in_table = False

    def close_image_grid() -> None:
        nonlocal in_image_grid
        if in_image_grid:
            html_lines.append("</div>")
            in_image_grid = False

    for line in markdown.splitlines():
        if skip_summary and not line.startswith("## "):
            continue
        if skip_summary and line.startswith("## "):
            skip_summary = False

        stripped = line.strip()
        if line.startswith("# ") or stripped.startswith("生成日期："):
            close_ul()
            close_table()
            close_image_grid()
            continue
        if include_gcard_summary and line.startswith("## ") and line[3:].strip().startswith("Summary"):
            close_ul()
            close_table()
            close_image_grid()
            html_lines.append("<h2>总结</h2>")
            html_lines.append(_render_gcard_summary_grid(eval_dir=eval_dir, run_config=run_config, score_labels=score_labels))
            skip_summary = True
            continue
        if line.startswith("## "):
            close_ul()
            close_table()
            close_image_grid()
            title = _strip_heading_order_prefix(line[3:].strip())
            html_lines.append(f"<h2>{_inline_markdown_to_html(title)}</h2>")
            continue
        if line.startswith("### "):
            close_ul()
            close_table()
            close_image_grid()
            title = _strip_heading_order_prefix(line[4:].strip())
            html_lines.append(f'<h3 class="section-subtitle">{_inline_markdown_to_html(title)}</h3>')
            continue
        if line.startswith("- "):
            close_table()
            close_image_grid()
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_inline_markdown_to_html(line[2:].strip())}</li>")
            continue
        if line.startswith("| ") and line.endswith(" |"):
            close_ul()
            close_image_grid()
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if _is_markdown_table_separator(cells):
                continue
            if not in_table:
                html_lines.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            html_lines.append("<tr>" + "".join(f"<{tag}>{_inline_markdown_to_html(cell)}</{tag}>" for cell in cells) + "</tr>")
            continue
        if stripped.startswith('<figure class="report-image') and stripped.endswith("</figure>"):
            close_ul()
            close_table()
            if not in_image_grid:
                html_lines.append('<div class="report-image-grid">')
                in_image_grid = True
            html_lines.append(stripped)
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            close_ul()
            close_table()
            if not in_image_grid:
                html_lines.append('<div class="report-image-grid">')
                in_image_grid = True
            alt_text = image_match.group(1).strip()
            image_src = image_match.group(2).strip()
            html_lines.append(
                '<figure class="report-image">'
                f'<img src="{escape(image_src, quote=True)}" alt="{escape(alt_text, quote=True)}">'
                f"<figcaption>{_inline_markdown_to_html(alt_text)}</figcaption>"
                "</figure>"
            )
            continue
        if line.startswith("> "):
            close_ul()
            close_table()
            close_image_grid()
            html_lines.append(f"<blockquote>{_inline_markdown_to_html(line[2:].strip())}</blockquote>")
            continue

        close_ul()
        close_table()
        if stripped:
            close_image_grid()
            html_lines.append(f"<p>{_inline_markdown_to_html(stripped)}</p>")

    close_ul()
    close_table()
    close_image_grid()
    return "\n".join(html_lines)


def _is_markdown_table_separator(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def _inline_markdown_to_html(text: str) -> str:
    escaped = escape(str(text))
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return escaped


def _strip_heading_order_prefix(title: str) -> str:
    """Remove display-only ordered heading prefixes before HTML section numbering."""
    return re.sub(
        r"^\s*(?:[一二三四五六七八九十百]+|[0-9]+|[A-Za-z])(?:、|\.|．)\s*",
        "",
        title,
    ).strip()


def _render_gcard_summary_grid(*, eval_dir: Path | None, run_config: dict[str, Any], score_labels: dict[str, str]) -> str:
    if eval_dir is None:
        return (
            '<div class="summary-grid">'
            '<div class="summary-card"><span class="summary-label">报告状态</span>'
            "<p>当前报告未绑定评估目录，Summary 卡片等待评估产物补齐。</p></div></div>"
        )
    compare_score = "gcard_v6"
    compare_label = score_labels.get(compare_score, compare_score)
    model_label = score_labels.get("model_score", "本轮模型")
    overall = _read_csv(eval_dir / "overall_metrics.csv")
    segment = _read_csv(eval_dir / "segment_metrics.csv")
    psi = _read_csv(eval_dir / "score_psi_by_month.csv")

    return "\n".join(
        [
            '<div class="summary-grid">',
            _summary_card_html("迭代效果", _gcard_iteration_summary_table(overall=overall, compare_score=compare_score)),
            _summary_card_html("高分段表现", _gcard_top_decile_summary_table(eval_dir=eval_dir, compare_score=compare_score), css_class="green"),
            _summary_card_html("分客群切片", _gcard_segment_slice_summary(segment=segment, compare_score=compare_score), css_class="orange"),
            _summary_card_html(
                "最终结论",
                _gcard_final_conclusion_summary(
                    overall=overall,
                    compare_score=compare_score,
                    model_label=model_label,
                    compare_label=compare_label,
                ),
                css_class="purple",
            ),
            "</div>",
        ]
    )


def _summary_card_html(label: str, inner_html: str, *, css_class: str = "") -> str:
    class_attr = f"summary-label {css_class}".strip()
    return "\n".join(
        [
            '  <div class="summary-card">',
            f'    <span class="{class_attr}">{escape(label)}</span>',
            inner_html,
            "  </div>",
        ]
    )


def _gcard_iteration_summary_table(*, overall: Any, compare_score: str) -> str:
    rows = []
    if overall is not None and not overall.empty:
        for split in ["OOT-OOS", "OOT", "DEV-OOS"]:
            row = _row_by_value(overall, "final_flag", split)
            if not row:
                continue
            rows.append(
                "<tr>"
                f"<td>{escape(split)}</td>"
                f"<td>{_metric_arrow(row.get(f'{compare_score}_ks'), row.get('model_score_ks'))}</td>"
                f"<td>{_fmt_signed_pp(_delta(row.get('model_score_ks'), row.get(f'{compare_score}_ks')))}</td>"
                f"<td>{_metric_arrow(row.get(f'{compare_score}_auc'), row.get('model_score_auc'))}</td>"
                f"<td>{_fmt_signed_pp(_delta(row.get('model_score_auc'), row.get(f'{compare_score}_auc')))}</td>"
                "</tr>"
            )
    if not rows:
        rows.append('<tr><td colspan="5">暂无可用 overall_metrics 对比结果。</td></tr>')
    return "<table><tr><th>评估口径</th><th>KS 旧→新</th><th>ΔKS</th><th>AUC 旧→新</th><th>ΔAUC</th></tr>" + "".join(rows) + "</table>"


def _gcard_top_decile_summary_table(*, eval_dir: Path, compare_score: str) -> str:
    rows = []
    for segment_name, segment_key in [("全客群", "all"), ("老户次新", "e2e3"), ("流失户", "b2")]:
        model = _gcard_top_decile_stat(eval_dir=eval_dir, segment_key=segment_key, score_column="model_score")
        compare = _gcard_top_decile_stat(eval_dir=eval_dir, segment_key=segment_key, score_column=compare_score)
        if model is None and compare is None:
            continue
        model_rate = model.get("bad_rate") if model else None
        compare_rate = compare.get("bad_rate") if compare else None
        rows.append(
            "<tr>"
            f"<td>{escape(segment_name)}</td>"
            f"<td>{_metric_arrow(compare_rate, model_rate, formatter=_fmt_percent_metric)}</td>"
            f"<td>{_fmt_signed_pp(_delta(model_rate, compare_rate))}</td>"
            "</tr>"
        )
    if not rows:
        rows.append('<tr><td colspan="3">暂无可用 decile lift 高分段结果。</td></tr>')
    return "<table><tr><th>客群</th><th>高分10%发起率 旧→新</th><th>提升</th></tr>" + "".join(rows) + "</table>"


def _gcard_segment_slice_summary(*, segment: Any, compare_score: str) -> str:
    items = []
    if segment is not None and not segment.empty:
        for segment_name in ["次新", "老户", "流失户"]:
            if not {"segment", "final_flag"}.issubset(segment.columns):
                continue
            subset = segment[(segment["segment"] == segment_name) & (segment["final_flag"] == "OOT-OOS")]
            if subset.empty:
                continue
            row = subset.iloc[0].to_dict()
            delta = _delta(row.get("model_score_ks"), row.get(f"{compare_score}_ks"))
            direction = "提升" if delta is not None and delta >= 0 else "低"
            items.append(
                f"<li>OOT-OOS {escape(segment_name)} KS "
                f"{_metric_arrow(row.get(f'{compare_score}_ks'), row.get('model_score_ks'))}，"
                f"{direction} <strong>{_fmt_signed_pp(delta).replace('+', '')}</strong>。</li>"
            )
    if not items:
        items.append("<li>暂无可用 OOT-OOS 分客群切片结果。</li>")
    items.append("<li>分客群指标是切片效果，不代表已训练分客群专属模型。</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _gcard_stability_boundary_summary(
    *,
    psi: Any,
    run_config: dict[str, Any],
    compare_score: str,
    compare_label: str,
    model_label: str,
) -> str:
    items = []
    model_psi = _latest_psi(psi, "model_score")
    compare_psi = _latest_psi(psi, compare_score)
    if model_psi is not None or compare_psi is not None:
        items.append(
            f"<li>最新月 PSI：{escape(model_label)} <strong>{_fmt_metric(model_psi)}</strong>，"
            f"{escape(compare_label)} <strong>{_fmt_metric(compare_psi)}</strong>。</li>"
        )
    label = run_config.get("label_column")
    if label:
        items.append(f"<li>主口径：30天发起标签 <code>{escape(str(label))}</code>，关注 AUC、KS、sloping、PSI。</li>")
    items.append("<li>MOB/金额风险、变量分箱明细和业务字典仍以后文待补充说明为准。</li>")
    return "<ul>" + "".join(items) + "</ul>"


def _gcard_final_conclusion_summary(
    *,
    overall: Any,
    compare_score: str,
    model_label: str,
    compare_label: str,
) -> str:
    oot_oos = _row_by_value(overall, "final_flag", "OOT-OOS")
    dev_oos = _row_by_value(overall, "final_flag", "DEV-OOS")
    ks_delta = _delta(oot_oos.get("model_score_ks"), oot_oos.get(f"{compare_score}_ks")) if oot_oos else None
    auc_delta = _delta(oot_oos.get("model_score_auc"), oot_oos.get(f"{compare_score}_auc")) if oot_oos else None
    dev_ks_delta = _delta(dev_oos.get("model_score_ks"), dev_oos.get(f"{compare_score}_ks")) if dev_oos else None

    if ks_delta is None:
        items = [
            f"{escape(model_label)} 与 {escape(compare_label)} 缺少 OOT-OOS KS 对比证据。",
            "<strong>不建议基于当前报告单独做上线结论。</strong>",
            "需先补齐核心评估指标，再结合策略收益、稳定性和上线成本评审。",
        ]
    elif ks_delta >= 0.01 and (auc_delta is None or auc_delta >= -0.001) and (dev_ks_delta is None or dev_ks_delta >= -0.002):
        items = [
            f"{escape(model_label)} 相较 {escape(compare_label)} 在 OOT-OOS KS 提升 {_fmt_signed_pp(ks_delta)}，效果提升较明显。",
            "<strong>可作为候选版本进入上线评审或灰度验证。</strong>",
            "仍需结合 PSI、分客群表现、策略收益和上线成本确认最终上线方案。",
        ]
    elif ks_delta >= 0.005:
        items = [
            f"{escape(model_label)} 相较 {escape(compare_label)} 在 OOT-OOS KS 提升 {_fmt_signed_pp(ks_delta)}，有一定增益但强度有限。",
            "<strong>不建议仅凭当前模型效果直接单独上线。</strong>",
            "建议纳入版本方案评审，并结合稳定性、分客群收益和策略成本做取舍。",
        ]
    else:
        items = [
            f"{escape(model_label)} 模型效果相较 {escape(compare_label)} 提升不明显。",
            "<strong>不建议将该模型单独作为独立版本上线。</strong>",
            "建议先结合策略收益、稳定性和上线成本复核，再决定是否纳入后续统一版本方案。",
        ]
    return (
        "<ul>"
        + "".join(f"<li>{item}</li>" for item in items)
        + "</ul>"
    )


def _latest_psi(psi: Any, score_column: str) -> Any:
    if psi is None or psi.empty or "psi" not in psi.columns:
        return None
    subset = psi.copy()
    if "score_column" in subset.columns:
        subset = subset[subset["score_column"] == score_column].copy()
    if subset.empty:
        return None
    sort_col = "month" if "month" in subset.columns else "mdl_month" if "mdl_month" in subset.columns else None
    if sort_col:
        subset = subset.sort_values(sort_col)
    return subset.iloc[-1].get("psi")


def _delta(new_value: Any, old_value: Any) -> float | None:
    new_float = _to_float(new_value)
    old_float = _to_float(old_value)
    if new_float is None or old_float is None:
        return None
    return new_float - old_float


def _metric_arrow(old_value: Any, new_value: Any, *, formatter: Any | None = None) -> str:
    value_formatter = formatter or _fmt_metric
    return f"{value_formatter(old_value)}→{value_formatter(new_value)}"


def _fmt_signed_pp(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric * 100:+.1f}pp"


def _fmt_percent_metric(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "N/A"
    return f"{numeric:.1%}"


def _report_html_style() -> str:
    return _read_report_asset("model_report.css")


def _report_html_script() -> str:
    return _read_report_asset("model_report.js")


def _read_report_asset(name: str) -> str:
    from importlib.resources import files

    return files(__package__).joinpath("assets", name).read_text(encoding="utf-8").strip()


def _gcard_top_decile_stat(*, eval_dir: Path, segment_key: str, score_column: str) -> dict[str, Any] | None:
    import pandas as pd

    frame = _read_csv(eval_dir / f"decile_lift_{segment_key}_{score_column}.csv")
    if frame is None and score_column == "model_score":
        frame = _read_csv(eval_dir / f"decile_lift_{segment_key}.csv")
    if frame is None or frame.empty or "decile" not in frame.columns:
        return None
    top = frame[pd.to_numeric(frame["decile"], errors="coerce") == 10]
    if top.empty:
        return None
    row = top.iloc[0]
    return {"bad_rate": row.get("bad_rate"), "cum_lift": row.get("cum_lift")}


def _row_by_value(frame: Any, column: str, value: Any) -> dict[str, Any]:
    if frame is None or frame.empty or column not in frame.columns:
        return {}
    subset = frame[frame[column] == value]
    if subset.empty:
        return {}
    return subset.iloc[0].to_dict()


def _fmt_metric(value: Any) -> str:
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_csv(path: Path) -> Any:
    if not path.exists():
        return None
    import pandas as pd

    return pd.read_csv(path, encoding="utf-8-sig")
