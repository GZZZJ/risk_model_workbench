"""Read-only auditors for run evidence, SQL, reports, and configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from risk_model_workbench.config import ConfigError, load_yaml
from risk_model_workbench.paths import project_config_path
from risk_model_workbench.project_state import audit_run
from risk_model_workbench.run_evidence import load_run_evidence
from risk_model_workbench.state import run_dir


@dataclass(frozen=True)
class AuditorSpec:
    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


Finding = dict[str, str]
AuditorFunc = Callable[[Path, str], dict[str, Any]]


AUDITOR_SPECS: tuple[AuditorSpec, ...] = (
    AuditorSpec("sql_review", "Read generated SQL and flag obvious approval, join, leakage, and query risks."),
    AuditorSpec("artifact_consistency", "Compare run_state.yml, artifact_manifest.json, and workflow contracts."),
    AuditorSpec("report_gap_scan", "Scan report artifacts and missing-results notes for unresolved evidence gaps."),
    AuditorSpec("config_risk", "Review project and run configuration for missing modeling contract fields."),
)


def list_auditors() -> tuple[AuditorSpec, ...]:
    return AUDITOR_SPECS


def get_auditor(name: str) -> AuditorSpec:
    for spec in AUDITOR_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown auditor: {name}")


def run_auditor(name: str, project_dir: str | Path, run_id: str) -> dict[str, Any]:
    get_auditor(name)
    project_path = Path(project_dir)
    return _AUDITOR_FUNCS[name](project_path, run_id)


def format_auditor_list(specs: tuple[AuditorSpec, ...]) -> str:
    lines = ["Auditor               Description", "-" * 78]
    lines.extend(f"{spec.name:<21} {spec.description}" for spec in specs)
    return "\n".join(lines) + "\n"


def format_auditor_result(result: dict[str, Any]) -> str:
    lines = [
        f"auditor: {result.get('auditor')}",
        f"project: {result.get('project')}",
        f"run_id: {result.get('run_id')}",
        f"status: {result.get('status')}",
        "findings:",
    ]
    findings = result.get("findings") or []
    if not findings:
        lines.append("  - none")
    for finding in findings:
        source = f" ({finding.get('source_path')})" if finding.get("source_path") else ""
        lines.append(f"  - [{finding.get('severity')}] {finding.get('code')}: {finding.get('message')}{source}")
    return "\n".join(lines) + "\n"


def _base_result(name: str, project_path: Path, run_id: str, findings: list[Finding]) -> dict[str, Any]:
    severities = {item.get("severity") for item in findings}
    status = "fail" if "error" in severities else "warn" if "warning" in severities else "pass"
    return {
        "auditor": name,
        "project": str(project_path.resolve()),
        "run_id": run_id,
        "status": status,
        "read_only": True,
        "findings": findings,
    }


def _finding(severity: str, code: str, message: str, source_path: str = "") -> Finding:
    return {
        "severity": severity,
        "code": code,
        "message": message,
        "source_path": source_path,
    }


def _sql_review(project_path: Path, run_id: str) -> dict[str, Any]:
    selected_run = run_dir(project_path, run_id)
    findings: list[Finding] = []
    sql_files = sorted(selected_run.rglob("*.sql"))
    if not sql_files:
        findings.append(_finding("warning", "sql_missing", "No run-local SQL files found for review.", _display(selected_run)))
        return _base_result("sql_review", project_path, run_id, findings)

    for path in sql_files:
        rel = _display(path)
        text = _read_text(path).lower()
        if "select *" in text:
            findings.append(_finding("warning", "select_star", "SQL uses SELECT *; explicit columns are safer.", rel))
        if text.count(" join ") >= 5:
            findings.append(_finding("warning", "many_joins", "SQL contains five or more joins; review for join explosion.", rel))
        if any(token in text for token in ["target", "label", "is_bad", "bad_flag"]) and "where" not in text:
            findings.append(_finding("warning", "label_filter_absent", "SQL references label-like fields without an obvious WHERE clause.", rel))
        if "drop table" in text or "insert overwrite" in text:
            findings.append(_finding("error", "mutating_sql", "SQL appears to mutate tables and needs explicit review.", rel))

    approval_files = sorted(selected_run.rglob("*sql*metadata*.json")) + sorted(selected_run.rglob("*approval*.json"))
    if not approval_files:
        findings.append(_finding("warning", "approval_metadata_missing", "No SQL approval metadata file was found in the run.", _display(selected_run)))
    return _base_result("sql_review", project_path, run_id, findings)


def _artifact_consistency(project_path: Path, run_id: str) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        audit = audit_run(project_path, run_id)
    except Exception as exc:
        findings.append(_finding("error", "audit_failed", f"Run audit could not be loaded: {exc}", f"runs/{run_id}"))
        return _base_result("artifact_consistency", project_path, run_id, findings)

    for stage in audit.get("stages", []):
        verdict = stage.get("verdict")
        severity = "error" if verdict in {"open", "missing", "incomplete"} else "warning" if verdict in {"scaffold", "imported"} else "info"
        for issue in stage.get("issues", []):
            findings.append(_finding(severity, f"{verdict}_stage", f"{stage.get('stage')}: {issue}", f"runs/{run_id}"))
    return _base_result("artifact_consistency", project_path, run_id, findings)


def _report_gap_scan(project_path: Path, run_id: str) -> dict[str, Any]:
    try:
        evidence = load_run_evidence(project_path, run_id)
    except Exception as exc:
        findings = [_finding("error", "run_evidence_unreadable", f"Run evidence could not be loaded: {exc}", f"runs/{run_id}")]
        return _base_result("report_gap_scan", project_path, run_id, findings)

    selected_run = evidence.run_path
    findings: list[Finding] = []
    report_dir = selected_run / "reports"
    missing_doc = report_dir / "model_report_missing_results.md"
    if missing_doc.exists():
        for line in _read_text(missing_doc).splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "|")) and any(token in stripped.lower() for token in ["missing", "缺", "待补"]):
                findings.append(_finding("warning", "report_gap", stripped[:240], _display(missing_doc)))
    else:
        findings.append(_finding("warning", "missing_results_doc_absent", "No model_report_missing_results.md found.", _display(report_dir)))

    report_stage = (evidence.run_state.get("stages") or {}).get("report") or {}
    if report_stage.get("status") == "scaffold":
        findings.append(_finding("warning", "report_scaffold", "Report stage is scaffold and should not be treated as complete evidence.", f"runs/{run_id}/run_state.yml"))
    report_artifacts = evidence.manifest_by_stage.get("report", [])
    if not report_artifacts:
        findings.append(_finding("warning", "report_artifacts_missing", "No report artifacts are registered in the manifest.", f"runs/{run_id}/audit/artifact_manifest.json"))
    return _base_result("report_gap_scan", project_path, run_id, findings)


def _config_risk(project_path: Path, run_id: str) -> dict[str, Any]:
    findings: list[Finding] = []
    try:
        config_path = project_config_path(project_path)
        if not config_path.exists():
            findings.append(_finding("error", "project_config_missing", "project.yml is missing.", _display(config_path)))
            return _base_result("config_risk", project_path, run_id, findings)
        config = load_yaml(config_path)
    except (OSError, ConfigError) as exc:
        findings.append(_finding("error", "project_config_unreadable", f"project.yml could not be loaded: {exc}", _display(project_path)))
        return _base_result("config_risk", project_path, run_id, findings)

    data = config.get("data") if isinstance(config.get("data"), dict) else {}
    for key in ["source_table", "id_columns", "target_column", "time_column", "period_column"]:
        if not data.get(key):
            findings.append(_finding("error", "data_contract_missing", f"data.{key} is missing.", _display(config_path)))
    if not config.get("segments"):
        findings.append(_finding("warning", "segments_missing", "No business segments are configured.", _display(config_path)))

    selected_run = run_dir(project_path, run_id)
    snapshot = selected_run / "configs_snapshot" / config_path.name
    if not snapshot.exists():
        findings.append(_finding("warning", "config_snapshot_missing", "Run config snapshot is missing.", _display(snapshot)))
    return _base_result("config_risk", project_path, run_id, findings)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _display(path: Path) -> str:
    return str(path)


_AUDITOR_FUNCS: dict[str, AuditorFunc] = {
    "sql_review": _sql_review,
    "artifact_consistency": _artifact_consistency,
    "report_gap_scan": _report_gap_scan,
    "config_risk": _config_risk,
}
