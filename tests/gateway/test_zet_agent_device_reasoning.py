"""Device-side reasoning override for the local AI proxy.

Since 2026-09-03 the cloud gateway behind ``/api/v1/ai-proxy/`` only returns
``reasoning_content`` when the request carries ``reasoning`` / ``thinking``.
``AIAgent._supports_reasoning_extra_body()`` returns False for a loopback
custom provider, so the request override is set in the zet_agent adapter.
These tests pin the pure decision function and the request assembly around it.
"""

import pytest

from gateway.platforms.zet_agent import (
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
