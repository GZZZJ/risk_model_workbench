from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "projects" / "2026-05-fujie-gcard-v1"
START = "<!-- RMW_CURRENT_STATE:START -->"
END = "<!-- RMW_CURRENT_STATE:END -->"


def _state_block(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    body = text.split(START, 1)[1].split(END, 1)[0].strip()
    if body.startswith("```yaml"):
        body = body[len("```yaml") :].strip()
    if body.endswith("```"):
        body = body[:-3].strip()
    return yaml.safe_load(body)


def test_entrypoint_docs_match_project_and_active_version_state():
    project_state = yaml.safe_load((PROJECT / "project_state.yml").read_text(encoding="utf-8"))
    version_id = project_state["active_version_id"]
    version_state = yaml.safe_load(
        (PROJECT / "versions" / version_id / "version_state.yml").read_text(encoding="utf-8")
    )
    expected = {
        "project": "projects/2026-05-fujie-gcard-v1",
        "active_version_id": version_id,
        "objective": project_state["current_objective"],
        "workflow": version_state["workflow"],
        "status": version_state["status"],
    }

    for name in ["README.md", "AGENTS.md", "CLAUDE.md"]:
        assert _state_block(ROOT / name) == expected
