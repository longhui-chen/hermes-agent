import json

import pytest

from tools.clarify_tool import clarify_tool
from tools.connector_setup_intent import connector_setup_result, normalize_connector_setup


@pytest.mark.parametrize("intent", [
    {"resource_kind": "camera", "password": "test-secret"},
    {"resource_kind": "printer3d", "host": "192.168.1.2"},
    {"resource_kind": "custom_api"},
    {"resource_kind": "custom_mcp", "url": "https://user:password@example.test/mcp"},
    {"resource_kind": "custom_mcp", "url": "https://example.test/mcp?token=test-secret"},
    {"resource_kind": "custom_mcp", "url": "file:///etc/passwd"},
    {"resource_kind": "custom_mcp", "url": "https://example.test:bad/mcp"},
    {"resource_kind": "custom_mcp", "auth_kind": "mtls"},
    {"resource_kind": "unknown"},
])
def test_invalid_or_sensitive_intent_is_rejected(intent):
    with pytest.raises(ValueError):
        normalize_connector_setup(intent)


def test_retired_non_camera_setup_intents_are_rejected():
    for intent in (
        {"resource_kind": "custom_api", "template_id": "gitlab"},
        {"resource_kind": "custom_mcp", "url": "https://example.test/mcp", "auth_kind": "bearer"},
        {"resource_kind": "saas", "provider_id": "notion"},
        {"resource_kind": "tv"},
    ):
        with pytest.raises(ValueError, match="connector_setup_invalid"):
            normalize_connector_setup(intent)


def test_camera_proposals_remain_available():
    for proposal in ("live", "recording"):
        intent = {"resource_kind": "camera", proposal: {"camera_id": "cam-1"}}
        assert normalize_connector_setup(intent) == intent


@pytest.mark.parametrize("raw", ["private-token", '{"status":"submitted","secret":"private-token"}', '{"status":"private-token"}', 'x' * 4096])
def test_free_form_response_never_reaches_model(raw):
    assert json.loads(connector_setup_result(raw)) == {"status": "cancelled"}


def test_flow_legacy_callback_is_never_asked_for_a_secret():
    called = []
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera", "live": {"camera_id": "cam-1"}},
                          callback=lambda question, choices: called.append(question))
    assert "connector_setup_unavailable" in result
    assert not called


def test_flow_trusted_callback_returns_only_a_receipt_not_a_grant():
    intent = {"resource_kind": "camera", "live": {"camera_id": "cam-1"}}
    def callback(question, choices, *, connector_setup):
        assert question == "Connector setup"
        assert choices is None
        assert connector_setup == intent
        return json.dumps({"status": "submitted", "target_id": "camera-123"})

    result = json.loads(clarify_tool("ignored", connector_setup=intent, callback=callback))
    assert result["status"] == "submitted"
    assert "Verify" in result["next_step"]
    assert "user_response" not in result


def test_flow_provider_exception_is_not_echoed():
    def callback(question, choices, *, connector_setup):
        raise RuntimeError("provider echoed private-token")
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera", "live": {"camera_id": "cam-1"}}, callback=callback)
    assert "private-token" not in result
    assert "connector_setup_unavailable" in result


def test_submitted_receipt_is_explicitly_not_a_credential_exposure():
    result = json.loads(connector_setup_result('{"status":"submitted","target_id":"jira-test"}'))
    assert 'camera confirmation receipt contains no credentials' in result['next_step']
    assert 'not a grant' in result['next_step']


def test_legacy_imports_share_the_adapter_contract():
    from gateway.platforms import zet_agent_connector_setup_intent as adapter
    from tools import connector_setup_intent as legacy
    from tools.clarify_tool import CLARIFY_SCHEMA

    assert legacy.normalize_connector_setup is adapter.normalize_connector_setup
    assert legacy.connector_setup_result is adapter.connector_setup_result
    assert legacy.CONNECTOR_SETUP_SCHEMA is adapter.CONNECTOR_SETUP_SCHEMA
    assert CLARIFY_SCHEMA["parameters"]["properties"]["connector_setup"] is adapter.CONNECTOR_SETUP_SCHEMA
