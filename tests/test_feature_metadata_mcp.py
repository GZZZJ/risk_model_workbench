"""Tests for feature_metadata dual-source meta client (sh_dp_mcp MCP preferred, TMLSQLClient fallback)."""

import json
import urllib.error

import pytest

from risk_model_workbench import feature_metadata as fm
from risk_model_workbench.feature_metadata import (
    _DpMcpMetaClient,
    _FallbackMetaClient,
)


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._payload


def _envelope(columns, des="表描述", name="t", is_error=False):
    inner = {"code": 0, "data": {"columns": columns, "des": des, "name": name}}
    return json.dumps(
        {"result": {"isError": is_error, "content": [{"text": json.dumps(inner)}]}}
    ).encode()


def test_dp_mcp_meta_client_parses_columns(monkeypatch):
    columns = [
        {"comment": "近1m内提前还款次数", "name": "f1", "type": "string"},
        {"comment": "", "name": "f2", "type": "int"},
    ]
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResponse(_envelope(columns)),
    )
    client = _DpMcpMetaClient("http://mcp", "key")
    meta = client.get_table_meta("ads_app_off_feature.t")
    assert meta["des"] == "表描述"
    comments = {c["name"]: c["comment"] for c in meta["columns"]}
    assert comments["f1"] == "近1m内提前还款次数"
    assert comments["f2"] == ""  # empty comment preserved
    assert meta["name"] == "t"


def test_dp_mcp_meta_client_raises_on_http_error(monkeypatch):
    def boom(req, timeout=30):
        raise urllib.error.HTTPError("u", 401, "unauthorized", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", boom)
    client = _DpMcpMetaClient("http://mcp", "bad-key")
    with pytest.raises(urllib.error.HTTPError):
        client.get_table_meta("proj.t")


def test_dp_mcp_meta_client_raises_on_is_error(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=30: _FakeResponse(
            json.dumps({"result": {"isError": True, "content": [{"text": "{}"}]}}).encode()
        ),
    )
    client = _DpMcpMetaClient("http://mcp", "key")
    with pytest.raises(RuntimeError):
        client.get_table_meta("proj.t")


def test_fallback_meta_client_uses_primary_then_fallback():
    class FailingPrimary:
        def get_table_meta(self, full_name):
            raise RuntimeError("mcp down")

    class Legacy:
        def __init__(self):
            self.called = False

        def get_table_meta(self, full_name):
            self.called = True
            return {"columns": [], "des": "", "name": full_name}

    legacy = Legacy()
    client = _FallbackMetaClient(FailingPrimary(), legacy)
    meta = client.get_table_meta("proj.t")
    assert legacy.called
    assert meta["name"] == "proj.t"


def test_fallback_meta_client_reraises_when_no_fallback():
    class FailingPrimary:
        def get_table_meta(self, full_name):
            raise RuntimeError("mcp down")

    client = _FallbackMetaClient(FailingPrimary(), None)
    with pytest.raises(RuntimeError):
        client.get_table_meta("proj.t")


def test_resolve_dp_mcp_config_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("SH_DP_MCP_URL", "http://env")
    monkeypatch.setenv("SH_DP_MCP_API_KEY", "envkey")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no ~/.claude.json
    assert fm._resolve_dp_mcp_config() == ("http://env", "envkey")


def test_resolve_dp_mcp_config_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("SH_DP_MCP_URL", raising=False)
    monkeypatch.delenv("SH_DP_MCP_API_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)  # no ~/.claude.json
    assert fm._resolve_dp_mcp_config() is None


def test_resolve_dp_mcp_config_reads_claude_json(monkeypatch, tmp_path):
    monkeypatch.delenv("SH_DP_MCP_URL", raising=False)
    monkeypatch.delenv("SH_DP_MCP_API_KEY", raising=False)
    (tmp_path / ".claude.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "sh_dp_mcp": {
                        "url": "http://from-claude",
                        "headers": {"X-CJJ-MCP-API-KEY": "claudekey"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    assert fm._resolve_dp_mcp_config() == ("http://from-claude", "claudekey")


def test_load_dp_client_prefers_mcp_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("SH_DP_MCP_URL", "http://env")
    monkeypatch.setenv("SH_DP_MCP_API_KEY", "envkey")
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    legacy_sentinel = object()
    monkeypatch.setattr(fm, "_legacy_meta_client", lambda: legacy_sentinel)
    client = fm.load_dp_client()
    assert isinstance(client, _FallbackMetaClient)
    assert isinstance(client._primary, _DpMcpMetaClient)
    assert client._fallback is legacy_sentinel


def test_load_dp_client_legacy_when_no_mcp(monkeypatch, tmp_path):
    monkeypatch.delenv("SH_DP_MCP_URL", raising=False)
    monkeypatch.delenv("SH_DP_MCP_API_KEY", raising=False)
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    legacy_sentinel = object()
    monkeypatch.setattr(fm, "_legacy_meta_client", lambda: legacy_sentinel)
    assert fm.load_dp_client() is legacy_sentinel
