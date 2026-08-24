"""Embedded Agent model-gateway contract tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from risk_model_workbench.agent.context_pack import context_pack_hash
from risk_model_workbench.agent.model_gateway import (
    AgentModelConfig,
    FakeModelGateway,
    LangChainModelGateway,
    ModelUnavailableError,
    OPENCODE_GO_BASE_URL,
    OpenCodeGoCurlGateway,
    StructuredOutputError,
    _keychain_api_key_for_config,
    build_model_gateway,
)
from risk_model_workbench.agent.reasoning_contracts import EmbeddedAdvisorAnswer, StructuredReasoningRequest
from risk_model_workbench.cli import main


def _context_pack() -> dict:
    payload = {
        "version": 1,
        "project": "demo",
        "version_id": "v1",
        "task_id": "train_main",
        "attempt_id": "attempt_1",
        "request_type": "failure_diagnosis_required",
        "files": [],
        "constraints": ["do not bypass approval"],
        "allowed_tools": [],
        "output_contract": {},
        "provenance": [],
        "limits": {},
        "truncation": {},
    }
    payload["context_hash"] = context_pack_hash(payload)
    return payload


def _request() -> StructuredReasoningRequest:
    pack = _context_pack()
    return StructuredReasoningRequest(
        request_id="advisor_train_1",
        request_type="failure_diagnosis_required",
        expected_response_type="failure_diagnosis",
        question="Diagnose the bounded failure.",
        context_hash=pack["context_hash"],
        context_pack=pack,
        constraints=["do not bypass approval"],
    )


def test_model_config_requires_explicit_provider_and_model():
    with pytest.raises(ModelUnavailableError, match="RMW_AGENT_MODEL"):
        AgentModelConfig.from_env({})
    with pytest.raises(ModelUnavailableError, match="RMW_AGENT_PROVIDER"):
        AgentModelConfig.from_env({"RMW_AGENT_MODEL": "model"})

    config = AgentModelConfig.from_env(
        {
            "RMW_AGENT_PROVIDER": "openai",
            "RMW_AGENT_MODEL": "example-model",
            "RMW_AGENT_MAX_RETRIES": "1",
        }
    )
    assert config.provider == "openai"
    assert config.model == "example-model"
    assert config.max_retries == 1
    assert "api_key" not in config.model_dump()


def test_anthropic_provider_builds_standard_langchain_gateway(monkeypatch):
    pytest.importorskip("langchain_anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-not-a-real-key")

    gateway = build_model_gateway(
        AgentModelConfig(provider="anthropic", model="claude-sonnet-example")
    )

    assert isinstance(gateway, LangChainModelGateway)
    assert type(gateway._chat_model).__name__ == "ChatAnthropic"
    assert gateway.config.provider == "anthropic"


def test_opencode_keychain_fallback_is_restricted_to_official_base_url(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="stored-secret\n")

    monkeypatch.setattr("risk_model_workbench.agent.model_gateway.subprocess.run", fake_run)
    config = AgentModelConfig(provider="openai", model="deepseek-v4-flash", base_url=OPENCODE_GO_BASE_URL)

    secret = _keychain_api_key_for_config(config, environ={}, system_name="Darwin")

    assert secret is not None
    assert secret.get_secret_value() == "stored-secret"
    assert calls[0][0][-1] == "-w"

    untrusted = config.model_copy(update={"base_url": "https://example.test/v1"})
    assert _keychain_api_key_for_config(untrusted, environ={}, system_name="Darwin") is None
    assert len(calls) == 1


def test_provider_environment_key_takes_precedence_over_keychain(monkeypatch):
    def unexpected_run(*args, **kwargs):
        raise AssertionError("Keychain must not be read when OPENAI_API_KEY is set")

    monkeypatch.setattr("risk_model_workbench.agent.model_gateway.subprocess.run", unexpected_run)
    config = AgentModelConfig(provider="openai", model="deepseek-v4-flash", base_url=OPENCODE_GO_BASE_URL)

    assert _keychain_api_key_for_config(
        config,
        environ={"OPENAI_API_KEY": "environment-secret"},
        system_name="Darwin",
    ) is None


def test_opencode_builder_selects_curl_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    config = AgentModelConfig(provider="openai", model="deepseek-v4-flash", base_url=OPENCODE_GO_BASE_URL)

    gateway = build_model_gateway(config)

    assert isinstance(gateway, OpenCodeGoCurlGateway)


def test_opencode_curl_gateway_validates_tool_call_and_metadata(monkeypatch):
    calls = []
    response = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "tool_calls": [
                        {
                            "function": {
                                "name": "emit_response",
                                "arguments": json.dumps(
                                    {
                                        "type": "failure_diagnosis",
                                        "status": "answered",
                                        "decision": "stop",
                                        "summary": "Stop safely.",
                                    }
                                ),
                            }
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    def fake_runner(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(response) + "\n200", stderr="")

    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    config = AgentModelConfig(
        provider="openai",
        model="deepseek-v4-flash",
        base_url=OPENCODE_GO_BASE_URL,
        max_retries=0,
    )
    gateway = OpenCodeGoCurlGateway(config, runner=fake_runner)

    answer, metadata = gateway.generate_structured(_request(), EmbeddedAdvisorAnswer)

    assert answer.decision == "stop"
    assert metadata.usage.total_tokens == 15
    assert calls[0][0] == ["/usr/bin/curl", "--config", "-"]
    assert "environment-secret" not in calls[0][0]
    assert 'tool_choice = ' not in calls[0][1]["input"]


def test_model_status_cli_fails_closed_without_configuration(monkeypatch, capsys):
    monkeypatch.delenv("RMW_AGENT_MODEL", raising=False)
    monkeypatch.delenv("RMW_AGENT_PROVIDER", raising=False)

    assert main(["agent", "model", "status", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["runtime"] == "embedded_langgraph"
    assert payload["configured"] is False
    assert "RMW_AGENT_MODEL" in payload["error"]


def test_reasoning_request_rejects_tampered_context_pack():
    pack = _context_pack()
    pack["project"] = "tampered"
    with pytest.raises(ValueError, match="content hash mismatch"):
        StructuredReasoningRequest(
            request_id="advisor_train_1",
            request_type="failure_diagnosis_required",
            expected_response_type="failure_diagnosis",
            question="Diagnose.",
            context_hash=pack["context_hash"],
            context_pack=pack,
        )


def test_fake_gateway_returns_validated_answer_and_audit_metadata():
    gateway = FakeModelGateway(
        [
            {
                "type": "failure_diagnosis",
                "status": "answered",
                "decision": "stop",
                "summary": "Required evidence is missing.",
                "risk_notes": ["Do not fabricate a successful artifact."],
            }
        ]
    )
    answer, metadata = gateway.generate_structured(_request(), EmbeddedAdvisorAnswer)

    assert answer.decision == "stop"
    assert metadata.provider == "fake"
    assert metadata.context_hash == _request().context_hash
    assert len(metadata.prompt_hash) == 64
    assert len(metadata.response_hash) == 64
    assert len(gateway.requests) == 1


def test_fake_gateway_rejects_invalid_structured_answer():
    gateway = FakeModelGateway([{"type": "failure_diagnosis", "decision": "continue"}])
    with pytest.raises(StructuredOutputError, match="failed validation"):
        gateway.generate_structured(_request(), EmbeddedAdvisorAnswer)


class _StructuredRunnable:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages):
        assert len(messages) == 2
        assert json.loads(messages[1].content)["request_id"] == "advisor_train_1"
        return {"parsed": self.payload, "raw": None, "parsing_error": None}


class _ChatModelStub:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.structured_kwargs = {}

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        assert schema is EmbeddedAdvisorAnswer
        assert include_raw is True
        self.calls += 1
        self.structured_kwargs = kwargs
        return _StructuredRunnable(self.payload)


def test_langchain_gateway_uses_structured_output_without_provider_network():
    chat_model = _ChatModelStub(
        {
            "type": "failure_diagnosis",
            "status": "answered",
            "decision": "stop",
            "summary": "Stop safely.",
        }
    )
    gateway = LangChainModelGateway(
        AgentModelConfig(provider="test", model="test-model", max_retries=0),
        chat_model=chat_model,
    )

    answer, metadata = gateway.generate_structured(_request(), EmbeddedAdvisorAnswer)

    assert answer.summary == "Stop safely."
    assert metadata.provider == "test"
    assert chat_model.calls == 1
