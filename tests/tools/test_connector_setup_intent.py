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


def test_known_template_and_remote_mcp_are_credential_free():
    for intent in (
        {"resource_kind": "custom_api", "template_id": "gitlab"},
        {"resource_kind": "custom_mcp", "url": "https://example.test/mcp", "auth_kind": "bearer"},
        {"resource_kind": "saas", "provider_id": "notion"},
        {"resource_kind": "tv"},
    ):
        assert normalize_connector_setup(intent) == intent


@pytest.mark.parametrize("raw", ["private-token", '{"status":"submitted","secret":"private-token"}', '{"status":"private-token"}', 'x' * 4096])
def test_free_form_response_never_reaches_model(raw):
    assert json.loads(connector_setup_result(raw)) == {"status": "cancelled"}


def test_flow_legacy_callback_is_never_asked_for_a_secret():
    called = []
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera"},
                          callback=lambda question, choices: called.append(question))
    assert "connector_setup_unavailable" in result
    assert not called


def test_flow_trusted_callback_returns_only_a_receipt_not_a_grant():
    def callback(question, choices, *, connector_setup):
        assert question == "Connector setup"
        assert choices is None
        assert connector_setup == {"resource_kind": "camera"}
        return json.dumps({"status": "submitted", "target_id": "camera-123"})

    result = json.loads(clarify_tool("ignored", connector_setup={"resource_kind": "camera"}, callback=callback))
    assert result["status"] == "submitted"
    assert "Verify" in result["next_step"]
    assert "user_response" not in result


def test_flow_provider_exception_is_not_echoed():
    def callback(question, choices, *, connector_setup):
        raise RuntimeError("provider echoed private-token")
    result = clarify_tool("ignored", connector_setup={"resource_kind": "camera"}, callback=callback)
    assert "private-token" not in result
    assert "connector_setup_unavailable" in result


@pytest.mark.parametrize("variables", [
    {"token": "secret"}, {"base_url": "https://example.test?token=secret"},
    {"endpoint_path": "/items?key=secret"}, {"method": "TRACE"},
    {"base_url": "https://user:pass@example.test"}, {"header_name": "X-Key\r\nInjected"},
])
def test_private_or_invalid_template_variables_are_rejected(variables):
    with pytest.raises(ValueError):
        normalize_connector_setup({"resource_kind": "custom_api", "template_id": "rest-bearer-token", "variables": variables})


def test_known_public_configuration_survives_trusted_clarify_flow():
    intent = {"resource_kind": "custom_api", "template_id": "rest-bearer-token",
              "variables": {"base_url": "https://example.test", "endpoint_path": "/v1/items", "method": "GET"}}
    def callback(question, choices, *, connector_setup):
        assert connector_setup == intent
        return '{"status":"submitted","target_id":"created-api"}'
    assert json.loads(clarify_tool("ignored", connector_setup=intent, callback=callback))["status"] == "submitted"
