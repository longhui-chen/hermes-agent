"""Unit tests for the generic, owner-scoped Generated App data bridge."""

import io
import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from gateway.session_context import _VAR_MAP
from tests.tools._profile_scope import mux_profile_scope, request_fingerprint
from tools.app_data_tool import (
    APP_DATA_SCHEMA,
    _CAPABILITY_TIMEOUT,
    _MAX_ENVELOPE_BYTES,
    _MAX_PAYLOAD_BYTES,
    _MAX_RESPONSE_BYTES,
    _MAX_STRING_CHARS,
    _MUTATION_TIMEOUT,
    _OPERATION_RE,
    _READ_TIMEOUT,
    _SLUG_RE,
    _base_url,
    _check_app_data,
    _is_cron_session,
    _is_trusted_zet_agent_session,
    app_data_tool,
)


_BASE_URL = "http://127.0.0.1:18080/api/v1/internal/apps"
_SLUG = "project-workbench"
_READ_OPERATION = "records.list"
_MUTATION_OPERATION = "records.store"
_CAPABILITY_DIGEST = "a" * 64


def _scope(**overrides):
    values = {
        "ZET_APP_DATA_BASE_URL": _BASE_URL,
        "ZETTLAB_AGENT_ACTION_TOKEN": "profile-operation-token",
        "ZET_AGENT_ID": "profile-agent",
    }
    values.update(overrides)
    return values


def _capabilities(*operations):
    return {
        "version": 1,
        "capability_digest": _CAPABILITY_DIGEST,
        "operations": [
            {"name": name, "mode": mode}
            for name, mode in operations
        ],
    }


class _Response:
    def __init__(self, payload, status=200):
        self.status = status
        self.raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit=-1):
        return self.raw if limit < 0 else self.raw[:limit]


def _sequence(seen, *outcomes):
    remaining = iter(outcomes)

    def open_request(request, timeout=None, **_kwargs):
        seen.append((request, timeout))
        outcome = next(remaining)
        if isinstance(outcome, BaseException):
            raise outcome
        return _Response(outcome)

    return open_request


@pytest.fixture(autouse=True)
def _zet_agent_session_scope():
    platform_var = _VAR_MAP["HERMES_SESSION_PLATFORM"]
    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    platform_token = platform_var.set("zet_agent")
    cron_token = cron_var.set("")
    try:
        yield
    finally:
        cron_var.reset(cron_token)
        platform_var.reset(platform_token)


@contextmanager
def _cron_scope():
    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    token = cron_var.set("1")
    try:
        yield
    finally:
        cron_var.reset(token)


def test_availability_requires_profile_identity_token_and_exact_loopback_base(monkeypatch):
    complete = _scope()
    with mux_profile_scope(monkeypatch, complete):
        assert _check_app_data() is True

    for missing in complete:
        with mux_profile_scope(monkeypatch, {**complete, missing: ""}):
            assert _check_app_data() is False

    with mux_profile_scope(
        monkeypatch,
        _scope(ZET_APP_DATA_BASE_URL="http://example.com/api/v1/internal/apps"),
    ):
        assert _check_app_data() is False
    with mux_profile_scope(
        monkeypatch,
        _scope(ZET_APP_DATA_BASE_URL="http://127.0.0.1:18080/api/v1/internal/chat"),
    ):
        assert _check_app_data() is False
    assert getattr(_check_app_data, "_profile_scope_sensitive") is True
    assert getattr(_check_app_data, "_session_scope_sensitive") is True


def test_base_url_prefers_dedicated_app_data_url(monkeypatch):
    with mux_profile_scope(
        monkeypatch,
        _scope(
            ZET_APP_DATA_BASE_URL=_BASE_URL,
            ZET_APPHOST_BASE_URL=(
                "http://127.0.0.1:19090/api/v1/internal/apphost"
            ),
        ),
    ):
        assert _base_url() == _BASE_URL


@pytest.mark.parametrize(
    ("legacy_url", "expected"),
    [
        (_BASE_URL, _BASE_URL),
        (
            "http://127.0.0.1:18080/api/v1/internal/apphost",
            _BASE_URL,
        ),
        (
            "http://[::1]:18080/api/v1/internal/apphost/",
            "http://[::1]:18080/api/v1/internal/apps",
        ),
    ],
)
def test_base_url_uses_safe_legacy_apphost_compatibility(
    monkeypatch, legacy_url, expected
):
    with mux_profile_scope(
        monkeypatch,
        _scope(ZET_APP_DATA_BASE_URL="", ZET_APPHOST_BASE_URL=legacy_url),
    ):
        assert _base_url() == expected


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.com/api/v1/internal/apps",
        "https://127.0.0.1:18080/api/v1/internal/apps",
        "http://user@127.0.0.1:18080/api/v1/internal/apps",
        "http://127.0.0.1:bad/api/v1/internal/apps",
        "http://127.0.0.1:18080/api/v1/internal/apphost",
        "http://127.0.0.1:18080/api/v1/internal/apps/extra",
        "http://127.0.0.1:18080/api/v1/internal/apps?target=other",
        "http://127.0.0.1:18080/api/v1/internal/apps#other",
    ],
)
def test_explicit_invalid_app_data_url_fails_closed_without_legacy_fallback(
    monkeypatch, bad_url
):
    with mux_profile_scope(
        monkeypatch,
        _scope(
            ZET_APP_DATA_BASE_URL=bad_url,
            ZET_APPHOST_BASE_URL=_BASE_URL,
        ),
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _base_url() is None
        assert _check_app_data() is False
        output = json.loads(
            app_data_tool({"action": "capabilities", "slug": _SLUG})
        )

    assert output["error"]["code"] == "unsupported"
    assert output["status"] == 0
    urlopen.assert_not_called()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://example.com/api/v1/internal/apphost",
        "http://127.0.0.1:18080/api/v1/internal/chat",
        "http://127.0.0.1:18080/api/v1/internal/apphost?target=other",
        "http://127.0.0.1:18080/api/v1/internal/apphost#other",
    ],
)
def test_legacy_apphost_derivation_rejects_untrusted_or_ambiguous_urls(
    monkeypatch, bad_url
):
    with mux_profile_scope(
        monkeypatch,
        _scope(ZET_APP_DATA_BASE_URL="", ZET_APPHOST_BASE_URL=bad_url),
    ):
        assert _base_url() is None


@pytest.mark.parametrize("platform", ["", "api_server", "telegram", "cli"])
def test_non_zet_agent_platform_cannot_check_or_call_app_data(
    monkeypatch, platform
):
    platform_var = _VAR_MAP["HERMES_SESSION_PLATFORM"]
    token = platform_var.set(platform)
    try:
        with mux_profile_scope(monkeypatch, _scope()), patch(
            "tools.app_data_tool._urlopen"
        ) as urlopen:
            assert _is_trusted_zet_agent_session() is False
            assert _check_app_data() is False
            output = json.loads(app_data_tool({
                "action": "capabilities",
                "slug": _SLUG,
            }))
    finally:
        platform_var.reset(token)

    assert output["error"]["code"] == "platform_scope_denied"
    urlopen.assert_not_called()


def test_platform_scope_lookup_failure_denies_app_data(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "gateway.session_context.get_session_env",
        side_effect=RuntimeError("scope unavailable"),
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _is_trusted_zet_agent_session() is False
        assert _check_app_data() is False
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    # Cron scope lookup also fails closed first; either denial proves the
    # transport cannot be reached with an unresolved task-local identity.
    assert output["error"]["code"] in {
        "cron_scope_denied",
        "platform_scope_denied",
    }
    urlopen.assert_not_called()


def test_delegated_child_cannot_check_or_call_app_data(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "agent.delegation_context.is_delegated_child_context",
        return_value=True,
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _check_app_data() is False
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    assert output["ok"] is False
    assert output["error"]["code"] == "delegated_child_scope_denied"
    urlopen.assert_not_called()


def test_delegated_scope_lookup_failure_denies_app_data(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "agent.delegation_context.is_delegated_child_context",
        side_effect=RuntimeError("scope unavailable"),
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _check_app_data() is False
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    assert output["error"]["code"] == "delegated_child_scope_denied"
    urlopen.assert_not_called()


def test_cron_scope_lookup_failure_denies_app_data(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "gateway.session_context.get_session_env",
        side_effect=RuntimeError("scope unavailable"),
    ), patch("tools.app_data_tool._urlopen") as urlopen:
        assert _is_cron_session() is True
        assert _check_app_data() is False
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    assert output["ok"] is False
    assert output["error"]["code"] == "cron_scope_denied"
    urlopen.assert_not_called()


def test_schema_exposes_only_declared_operation_actions():
    params = APP_DATA_SCHEMA["parameters"]
    assert params["additionalProperties"] is False
    assert set(params["required"]) == {"action", "slug"}
    properties = params["properties"]
    assert properties["action"]["enum"] == ["capabilities", "invoke"]
    assert properties["slug"]["pattern"] == "^[a-z][a-z0-9-]{2,31}$"
    assert properties["operation"]["pattern"] == (
        "^[a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*){1,7}$"
    )
    assert properties["query"]["maxProperties"] == 16
    assert properties["query"]["propertyNames"]["pattern"] == (
        "^[a-z][a-z0-9_]{0,63}$"
    )
    for forbidden in (
        "agent_id",
        "token",
        "url",
        "method",
        "path",
        "prompt",
        "provider",
        "model",
        "delivery_target",
    ):
        assert forbidden not in properties


def test_runtime_rejects_additional_fields_before_transport(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        output = json.loads(
            app_data_tool(
                {
                    "action": "capabilities",
                    "slug": _SLUG,
                    "agent_id": "forged-agent",
                }
            )
        )

    assert output["error"]["code"] == "invalid_request"
    urlopen.assert_not_called()


@pytest.mark.parametrize(
    ("slug", "valid"),
    [
        ("abc", True),
        ("a" * 32, True),
        ("ab", False),
        ("a" * 33, False),
        ("App", False),
    ],
)
def test_slug_pattern_matches_local_server_contract(slug, valid):
    assert _SLUG_RE.pattern == r"^[a-z][a-z0-9-]{2,31}$"
    assert (_SLUG_RE.fullmatch(slug) is not None) is valid


@pytest.mark.parametrize(
    ("operation", "valid"),
    [
        ("records.list", True),
        ("a.b_c.d.e.f.g.h.i", True),
        ("records", False),
        ("a.b.c.d.e.f.g.h.i", False),
        ("records-list.run", False),
        ("Records.list", False),
    ],
)
def test_operation_name_pattern_matches_local_server_contract(operation, valid):
    assert _OPERATION_RE.pattern == (
        r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,7}$"
    )
    assert (_OPERATION_RE.fullmatch(operation) is not None) is valid


def test_capabilities_projects_only_version_name_and_mode(monkeypatch):
    seen = []
    response = {
        "version": 1,
        "operations": [
            {
                "name": _READ_OPERATION,
                "mode": "read",
                "description": "Future optional display metadata",
            },
            {
                "name": _MUTATION_OPERATION,
                "mode": "mutation",
                "input_schema_version": 2,
            },
        ],
        "capability_digest": _CAPABILITY_DIGEST,
        "owner_user_id": "must-not-project",
        "dispatcher_path": "/must-not-project",
    }
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(seen, response),
    ):
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    assert output == {
        "ok": True,
        "data": {
            "version": 1,
            "operations": [
                {"name": _READ_OPERATION, "mode": "read"},
                {"name": _MUTATION_OPERATION, "mode": "mutation"},
            ],
        },
        "untrusted_app_data": True,
    }
    request = seen[0][0]
    assert request.full_url == _BASE_URL + f"/{_SLUG}/capabilities"
    assert request.get_method() == "GET"
    assert request.data is None
    assert seen[0][1] == _CAPABILITY_TIMEOUT


def test_read_invoke_uses_declared_route_and_profile_token(monkeypatch):
    seen = []
    scope = _scope()
    with mux_profile_scope(monkeypatch, scope, poison_environ=True), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            seen,
            _capabilities((_READ_OPERATION, "read")),
            {"items": [{"id": "record-1"}]},
        ),
    ):
        output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
            "query": {"label": "weekly", "limit": "2"},
            "payload": {
                "record": {"source_url": "https://example.com/item"},
            },
        }))

    assert output["ok"] is True
    assert output["untrusted_app_data"] is True
    assert output["data"]["items"][0]["id"] == "record-1"
    capability_request, invoke_request = [item[0] for item in seen]
    assert [item[1] for item in seen] == [_CAPABILITY_TIMEOUT, _READ_TIMEOUT]
    assert capability_request.get_method() == "GET"
    assert invoke_request.full_url == _BASE_URL + f"/{_SLUG}/operations/{_READ_OPERATION}"
    assert invoke_request.get_method() == "POST"
    assert invoke_request.get_header("X-zettlab-agent-action-token") == scope[
        "ZETTLAB_AGENT_ACTION_TOKEN"
    ]
    assert json.loads(invoke_request.data) == {
        "capability_digest": _CAPABILITY_DIGEST,
        "query": {"label": "weekly", "limit": "2"},
        "payload": {
            "record": {"source_url": "https://example.com/item"},
        },
    }
    assert "profile-agent" not in invoke_request.data.decode()
    assert "stale-" not in request_fingerprint(invoke_request)


@pytest.mark.parametrize(
    "args,message",
    [
        ({"action": "capabilities"}, "slug"),
        ({"action": "capabilities", "slug": "ab"}, "slug"),
        ({"action": "capabilities", "slug": "a" * 33}, "slug"),
        ({"action": "capabilities", "slug": "../escape"}, "slug"),
        ({"action": "other", "slug": _SLUG}, "action"),
        (
            {"action": "capabilities", "slug": _SLUG, "operation": _READ_OPERATION},
            "不接受",
        ),
        ({"action": "invoke", "slug": _SLUG}, "operation"),
        (
            {"action": "invoke", "slug": _SLUG, "operation": "../escape"},
            "operation",
        ),
        (
            {"action": "invoke", "slug": _SLUG, "operation": "records"},
            "operation",
        ),
        (
            {"action": "invoke", "slug": _SLUG, "operation": "records-list.run"},
            "operation",
        ),
        (
            {
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "payload": {"nested": {"URL": "http://169.254.169.254"}},
            },
            "受保护字段",
        ),
        *(
            (
                {
                    "action": "invoke",
                    "slug": _SLUG,
                    "operation": _READ_OPERATION,
                    "payload": {"nested": {field: "forbidden"}},
                },
                "受保护字段",
            )
            for field in (
                "agentId",
                "access_key",
                "accessToken",
                "api_key",
                "apikey",
                "bearer",
                "credential",
                "deliveryTarget",
                "requestHeaders",
                "method",
                "filePath",
                "callback_urls",
                "endpoint_uris",
                "file_paths",
                "password",
                "prompt",
                "provider",
                "model",
                "secret",
                "secret_key",
                "skills",
                "source_urls",
            )
        ),
        (
            {
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "query": {"url": "http://169.254.169.254"},
            },
            "受保护字段",
        ),
        *(
            (
                {
                    "action": "invoke",
                    "slug": _SLUG,
                    "operation": _READ_OPERATION,
                    "query": {field: "forbidden"},
                },
                "受保护字段",
            )
            for field in (
                "access_key",
                "api_key",
                "apikey",
                "bearer",
                "credential",
                "callback_urls",
                "endpoint_uris",
                "file_paths",
                "password",
                "secret",
                "secret_key",
                "source_urls",
            )
        ),
        (
            {
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "query": {"limit": 2},
            },
            "query 值",
        ),
        (
            {
                "action": "invoke",
                "slug": _SLUG,
                "operation": _READ_OPERATION,
                "idempotency_key": "contains space",
            },
            "idempotency_key",
        ),
    ],
)
def test_invalid_generic_shapes_are_rejected_before_transport(monkeypatch, args, message):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        output = json.loads(app_data_tool(args))
    assert output["ok"] is False
    assert output["status"] == 0
    assert message in output["error"]["message"]
    urlopen.assert_not_called()


def test_payload_limit_is_separate_from_transport_envelope_limit(monkeypatch):
    payload_overhead = len(
        json.dumps(
            {"chunks": ["", ""]},
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    remaining = _MAX_PAYLOAD_BYTES - payload_overhead
    first_size = min(_MAX_STRING_CHARS, remaining)
    payload = {
        "chunks": ["a" * first_size, "b" * (remaining - first_size)]
    }
    assert len(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ) == _MAX_PAYLOAD_BYTES

    seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            seen,
            _capabilities((_READ_OPERATION, "read")),
            {"accepted": True},
        ),
    ):
        accepted = json.loads(
            app_data_tool(
                {
                    "action": "invoke",
                    "slug": _SLUG,
                    "operation": _READ_OPERATION,
                    "payload": payload,
                }
            )
        )
    assert accepted["ok"] is True
    assert len(seen[1][0].data) <= _MAX_ENVELOPE_BYTES

    oversized = {
        "chunks": [payload["chunks"][0], payload["chunks"][1] + "x"]
    }
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        rejected = json.loads(
            app_data_tool(
                {
                    "action": "invoke",
                    "slug": _SLUG,
                    "operation": _READ_OPERATION,
                    "payload": oversized,
                }
            )
        )
    assert rejected["error"]["code"] == "invalid_request"
    urlopen.assert_not_called()


def test_undeclared_operation_is_rejected_without_dispatch(monkeypatch):
    seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(seen, _capabilities((_READ_OPERATION, "read"))),
    ):
        output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": "records.unknown",
        }))

    assert output["error"]["code"] == "operation_not_declared"
    assert len(seen) == 1
    assert seen[0][0].get_method() == "GET"


def test_mutation_requires_key_then_approval_and_carries_exact_envelope(monkeypatch):
    missing_seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(missing_seen, _capabilities((_MUTATION_OPERATION, "mutation"))),
    ), patch("tools.app_data_tool._approval_result") as approval:
        missing = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": {"record_id": "record-1"},
        }))
    assert missing["error"]["code"] == "invalid_request"
    assert "idempotency_key" in missing["error"]["message"]
    assert len(missing_seen) == 1
    approval.assert_not_called()

    seen = []
    payload = {"record_id": "record-1", "state": "accepted"}
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._approval_result", return_value={"approved": True}
    ) as approval, patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            seen,
            _capabilities((_MUTATION_OPERATION, "mutation")),
            {"record_id": "record-1", "revision": 4},
        ),
    ):
        output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": payload,
            "idempotency_key": "record:record-1:3",
        }))

    approval_envelope = {
        "payload": payload,
        "idempotency_key": "record:record-1:3",
    }
    assert output["ok"] is True
    approval.assert_called_once_with(
        _SLUG,
        _MUTATION_OPERATION,
        approval_envelope,
    )
    assert json.loads(seen[-1][0].data) == {
        "capability_digest": _CAPABILITY_DIGEST,
        **approval_envelope,
    }


def test_only_capability_discovery_retries_and_invokes_are_not_retried(monkeypatch):
    capability_seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            capability_seen,
            TimeoutError("capability lookup lost"),
            _capabilities((_READ_OPERATION, "read")),
            {"items": []},
        ),
    ):
        capability_output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
        }))
    assert capability_output["ok"] is True
    assert [item[1] for item in capability_seen] == [
        _CAPABILITY_TIMEOUT,
        _CAPABILITY_TIMEOUT,
        _READ_TIMEOUT,
    ]

    read_seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            read_seen,
            _capabilities((_READ_OPERATION, "read")),
            TimeoutError("unknown read outcome"),
        ),
    ):
        read_output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
        }))
    assert read_output["ok"] is False
    assert len(read_seen) == 2
    assert [item[1] for item in read_seen] == [
        _CAPABILITY_TIMEOUT,
        _READ_TIMEOUT,
    ]

    mutation_seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._approval_result", return_value={"approved": True}
    ), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            mutation_seen,
            _capabilities((_MUTATION_OPERATION, "mutation")),
            TimeoutError("lost"),
        ),
    ):
        mutation_output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": {"record_id": "record-1"},
            "idempotency_key": "record:record-1",
        }))
    assert mutation_output["ok"] is False
    assert mutation_output["status"] is None
    assert len(mutation_seen) == 2
    assert [item[1] for item in mutation_seen] == [
        _CAPABILITY_TIMEOUT,
        _MUTATION_TIMEOUT,
    ]


def test_structured_http_error_is_bounded_and_never_leaks_credentials(monkeypatch):
    scope = _scope()
    seen = []

    def open_request(request, _timeout=None, **_kwargs):
        seen.append(request)
        if len(seen) == 1:
            return _Response(_capabilities((_MUTATION_OPERATION, "mutation")))
        body = json.dumps({"code": "stale_revision", "message": "revision changed"}).encode()
        raise urllib.error.HTTPError(request.full_url, 409, "Conflict", {}, io.BytesIO(body))

    with mux_profile_scope(monkeypatch, scope), patch(
        "tools.app_data_tool._approval_result", return_value={"approved": True}
    ), patch("tools.app_data_tool._urlopen", open_request):
        raw = app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _MUTATION_OPERATION,
            "payload": {"record_id": "record-1", "revision": 2},
            "idempotency_key": "record:record-1:2",
        })
    output = json.loads(raw)
    assert output == {
        "ok": False,
        "error": {"code": "stale_revision", "message": "revision changed"},
        "status": 409,
    }
    assert scope["ZETTLAB_AGENT_ACTION_TOKEN"] not in raw
    assert "127.0.0.1:18080" not in raw


@pytest.mark.parametrize(
    "capability_response",
    [
        {"version": 2, "operations": []},
        {"version": True, "operations": []},
        {"version": 1, "operations": []},
        {"version": 1, "operations": [], "capability_digest": "A" * 64},
        {"version": 1, "operations": [], "capability_digest": "a" * 63},
        {"version": 1, "operations": [_READ_OPERATION]},
        {"version": 1, "operations": [{"name": _READ_OPERATION}]},
        {"version": 1, "operations": [{"mode": "read"}]},
        _capabilities((_READ_OPERATION, "write")),
        _capabilities((_READ_OPERATION, [])),
        _capabilities((_READ_OPERATION, {})),
        _capabilities((_READ_OPERATION, "read"), (_READ_OPERATION, "read")),
    ],
)
def test_invalid_capability_declarations_fail_closed(monkeypatch, capability_response):
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        return_value=_Response(capability_response),
    ):
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))
    assert output["ok"] is False
    assert output["error"]["code"] == "invalid_response"


def test_oversized_response_fails_without_returning_partial_data(monkeypatch):
    seen = []
    with mux_profile_scope(monkeypatch, _scope()), patch(
        "tools.app_data_tool._urlopen",
        _sequence(
            seen,
            _capabilities((_READ_OPERATION, "read")),
            b"x" * (_MAX_RESPONSE_BYTES + 1),
        ),
    ):
        output = json.loads(app_data_tool({
            "action": "invoke",
            "slug": _SLUG,
            "operation": _READ_OPERATION,
        }))
    assert output["ok"] is False
    assert output["error"]["code"] == "response_too_large"


def test_cron_cannot_discover_or_call_app_data(monkeypatch):
    with mux_profile_scope(monkeypatch, _scope()), _cron_scope(), patch(
        "tools.app_data_tool._urlopen"
    ) as urlopen:
        assert _check_app_data() is False
        output = json.loads(app_data_tool({
            "action": "capabilities",
            "slug": _SLUG,
        }))

    assert output["ok"] is False
    assert output["error"]["code"] == "cron_scope_denied"
    urlopen.assert_not_called()
