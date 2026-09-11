"""Device-side reasoning override for the local AI proxy.

Since 2026-09-03 the cloud gateway behind ``/api/v1/ai-proxy/`` only returns
``reasoning_content`` when the request carries ``reasoning`` / ``thinking``.
``AIAgent._supports_reasoning_extra_body()`` returns False for a loopback
custom provider, so the request override is set in the zet_agent adapter.
These tests pin the pure decision function and the request assembly around it.
"""

import pytest

from gateway.platforms.zet_agent import (
    _device_reasoning_config,
    _device_reasoning_fast_path,
    _is_local_ai_proxy_base_url,
    _onboarding_deepseek_fast_path,
)

LOCAL_PROXY = "http://127.0.0.1:19090/api/v1/ai-proxy/v1"


# ---------- the four reasoning_config inputs ----------

def test_absent_config_defaults_to_medium():
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=None,
        request_overrides=None,
    )
    assert applied is True
    assert overrides == {"extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}}


def test_enabled_without_effort_defaults_to_medium():
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config={"enabled": True},
        request_overrides={"extra_body": {"existing": 1}},
    )
    assert applied is True
    assert overrides == {
        "extra_body": {"existing": 1, "reasoning": {"enabled": True, "effort": "medium"}}
    }


def test_explicit_effort_is_forwarded():
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides=None,
    )
    assert applied is True
    assert overrides["extra_body"]["reasoning"] == {"enabled": True, "effort": "high"}


def test_disabled_config_sends_thinking_disabled():
    # The App's reasoning_effort=none normalises to {"enabled": False} before
    # reaching here, so both disable shapes land on this branch. The gateway
    # accepts `thinking: disabled`, not `reasoning.enabled=false`.
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config={"enabled": False},
        request_overrides={"extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}},
    )
    assert applied is True
    assert overrides == {"extra_body": {"thinking": {"type": "disabled"}}}


# ---------- scope ----------

def test_onboarding_is_left_to_its_own_disable_fast_path():
    overrides, applied = _device_reasoning_fast_path(
        profile="onboarding",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=None,
        request_overrides={"extra_body": {"thinking": {"type": "disabled"}}},
    )
    assert applied is False
    assert overrides == {"extra_body": {"thinking": {"type": "disabled"}}}


@pytest.mark.parametrize(
    "provider,base_url",
    [
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("custom", "https://example.com/api/v1/ai-proxy/v1"),
        ("custom", "http://127.0.0.1:19090/v1"),
        ("zettlab", LOCAL_PROXY),
        ("custom", ""),
    ],
)
def test_other_routes_are_untouched(provider, base_url):
    original = {"extra_body": {"existing": 1}}
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider=provider,
        base_url=base_url,
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides=original,
    )
    assert applied is False
    assert overrides == original


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:19090/api/v1/ai-proxy/v1", True),
        ("http://localhost:19090/api/v1/ai-proxy/v1", True),
        ("HTTP://127.0.0.1:19090/API/V1/AI-PROXY/V1", True),
        ("https://cloud.example.com/api/v1/ai-proxy/v1", False),
        ("http://192.168.32.98:19090/api/v1/ai-proxy/v1", False),
        ("http://127.0.0.1:19090/v1", False),
        ("", False),
        # P1 3981215371: bracketed IPv6 loopback, with and without a port.
        ("http://[::1]:19090/api/v1/ai-proxy/v1", True),
        ("http://[::1]/api/v1/ai-proxy/v1", True),
        ("http://[0:0:0:0:0:0:0:1]:19090/api/v1/ai-proxy/v1", True),
        ("http://127.0.0.2:19090/api/v1/ai-proxy/v1", True),
        # A remote IPv6 host must not match even on the proxy path.
        ("http://[2001:db8::1]:19090/api/v1/ai-proxy/v1", False),
        # userinfo must not be able to spoof the host.
        ("http://127.0.0.1@evil.example.com/api/v1/ai-proxy/v1", False),
        # the path must really be the proxy path, not a query string.
        ("http://127.0.0.1:19090/v1?x=/api/v1/ai-proxy/", False),
        # non-http schemes are not the device proxy.
        ("ftp://127.0.0.1:19090/api/v1/ai-proxy/v1", False),
    ],
)
def test_local_proxy_matching_is_host_and_path(url, expected):
    assert _is_local_ai_proxy_base_url(url) is expected


# ---------- request assembly ----------

def test_assembly_order_onboarding_then_device():
    """Onboarding keeps `thinking: disabled`; a normal profile gets reasoning."""
    reasoning_config, overrides, onboarding = _onboarding_deepseek_fast_path(
        profile="onboarding",
        model="lite",
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides={},
    )
    assert onboarding is True
    overrides, applied = _device_reasoning_fast_path(
        profile="onboarding",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=reasoning_config,
        request_overrides=overrides,
    )
    assert applied is False
    assert overrides["extra_body"]["thinking"] == {"type": "disabled"}
    assert overrides["extra_body"]["reasoning_effort"] == "none"
    assert "reasoning" not in overrides["extra_body"]

    reasoning_config, overrides, onboarding = _onboarding_deepseek_fast_path(
        profile="main",
        model="pro",
        reasoning_config=None,
        request_overrides={},
    )
    assert onboarding is False
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=reasoning_config,
        request_overrides=overrides,
    )
    assert applied is True
    assert overrides["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}
    assert "thinking" not in overrides["extra_body"]


def test_openrouter_assembly_is_unchanged_end_to_end():
    reasoning_config, overrides, onboarding = _onboarding_deepseek_fast_path(
        profile="main",
        model="openrouter/some-model",
        reasoning_config={"enabled": True, "effort": "high"},
        request_overrides={"extra_body": {"provider": {"order": ["x"]}}},
    )
    assert onboarding is False
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        reasoning_config=reasoning_config,
        request_overrides=overrides,
    )
    assert applied is False
    assert overrides == {"extra_body": {"provider": {"order": ["x"]}}}


# ---------- P1 3981058281: resolve reasoning_config against the FINAL model ----------

def _per_model_loader(table):
    """Stand-in for GatewayRunner._load_reasoning_config's per-model behaviour."""
    return lambda model: table.get(model)


def test_config_is_reloaded_for_the_final_model():
    # config.yaml: lite disables reasoning, pro asks for high.  The runtime
    # loads the config for the default model (lite) before the session
    # /model override switches to pro.
    resolved = _device_reasoning_config(
        request_reasoning_config=None,
        load_for_model=_per_model_loader({"lite": {"enabled": False}, "pro": {"enabled": True, "effort": "high"}}),
        model="pro",
    )
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=resolved,
        request_overrides=None,
    )
    assert applied is True
    assert overrides == {"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}}


def test_switching_to_a_disabled_model_sends_thinking_disabled():
    resolved = _device_reasoning_config(
        request_reasoning_config=None,
        load_for_model=_per_model_loader({"lite": {"enabled": False}, "pro": {"enabled": True, "effort": "high"}}),
        model="lite",
    )
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=resolved,
        request_overrides=None,
    )
    assert applied is True
    assert overrides == {"extra_body": {"thinking": {"type": "disabled"}}}


def test_explicit_request_reasoning_still_wins_over_per_model_config():
    # The client stated model_options.reasoning for this turn; it must not be
    # overwritten by whatever config.yaml says about the resolved model.
    resolved = _device_reasoning_config(
        request_reasoning_config={"enabled": True, "effort": "low"},
        load_for_model=_per_model_loader({"pro": {"enabled": False}}),
        model="pro",
    )
    assert resolved == {"enabled": True, "effort": "low"}
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=resolved,
        request_overrides=None,
    )
    assert overrides == {"extra_body": {"reasoning": {"enabled": True, "effort": "low"}}}


def test_missing_per_model_config_falls_back_to_the_default_effort():
    resolved = _device_reasoning_config(
        request_reasoning_config=None,
        load_for_model=_per_model_loader({}),
        model="max",
    )
    assert resolved is None
    overrides, applied = _device_reasoning_fast_path(
        profile="main",
        provider="custom",
        base_url=LOCAL_PROXY,
        reasoning_config=resolved,
        request_overrides=None,
    )
    assert overrides == {"extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}}


def test_real_loader_honours_per_model_reasoning_effort():
    # Pins the assumption the fix rests on: the shared chokepoint really does
    # resolve per-model overrides, so passing the final model changes the answer.
    from hermes_constants import resolve_reasoning_config

    cfg = {"agent": {"reasoning_effort": "none", "reasoning_overrides": {"pro": "high"}}}
    assert resolve_reasoning_config(cfg, "pro") == {"enabled": True, "effort": "high"}
    assert resolve_reasoning_config(cfg, "lite") == {"enabled": False}
