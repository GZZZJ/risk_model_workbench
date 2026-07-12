from pathlib import Path


SKILL_NAMES = (
    "risk-model-workbench",
    "rmw-model-training",
    "rmw-feature-selection",
    "rmw-model-evaluation",
    "rmw-report-generation",
)


def test_codex_and_claude_host_skills_do_not_drift():
    root = Path(__file__).resolve().parents[1]
    for skill_name in SKILL_NAMES:
        codex_skill = root / ".agents" / "skills" / skill_name / "SKILL.md"
        claude_skill = root / ".claude" / "skills" / skill_name / "SKILL.md"
        assert codex_skill.read_text(encoding="utf-8") == claude_skill.read_text(encoding="utf-8")
