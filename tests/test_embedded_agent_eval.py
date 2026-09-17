from risk_model_workbench.agent.embedded_eval import evaluate_embedded_suite
from risk_model_workbench.cli import main


def test_embedded_agent_eval_suite_passes():
    report = evaluate_embedded_suite()

    assert report["passed"] is True
    assert report["metrics"]["contract_case_pass_rate"]["rate"] == 1.0
    assert report["metrics"]["unsafe_accept_count"]["count"] == 0


def test_embedded_agent_eval_cli(capsys):
    assert main(["agent", "eval", "--suite", "embedded"]) == 0
    output = capsys.readouterr().out
    assert "suite: embedded" in output
    assert "passed: true" in output
