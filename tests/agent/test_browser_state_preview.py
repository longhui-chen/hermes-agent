import json
from urllib.parse import quote

import pytest

from agent.browser_state_preview import (
    MAX_ELEMENTS,
    MAX_PREVIEW_BYTES,
    project_browser_state_preview,
)


def test_navigate_projects_safe_structured_state() -> None:
    preview = project_browser_state_preview(
        "browser_navigate",
        {
            "success": True,
            "url": "https://user:pass@Example.COM/account/profile?token=secret#billing",
            "title": "Account sk-test-abcdefghijklmnopqrstuvwxyz123456",
            "snapshot": (
                '- heading "Account" [ref=e1]\n'
                '- textbox "Password" [ref=e2]: raw-input-value\n'
                '- button "Save" [ref=e3] [disabled]\n'
                '- link "Billing" [ref=e4]\n'
                "  /url: https://example.com/billing?session=private"
            ),
            "element_count": 4,
            "screenshot_path": "/private/device/path.png",
        },
        browser_session_id="browser-session-1",
    )

    assert preview is not None
    assert preview["version"] == 1
    assert preview["source"] == "navigate"
    assert preview["url"] == {"hostname": "example.com"}
    assert preview["browserSessionId"] == "browser-session-1"
    assert preview["elementCount"] == 4
    assert preview["elements"] == [
        {"role": "heading", "label": "Account"},
        {"role": "textbox", "label": "Password"},
        {"role": "button", "label": "Save", "state": "disabled"},
        {"role": "link", "label": "Billing"},
    ]
    encoded = json.dumps(preview, ensure_ascii=False)
    for secret in (
        "user:pass",
        "?token=",
        "#billing",
        "raw-input-value",
        "/private/device",
    ):
        assert secret not in encoded


def test_snapshot_is_bounded_and_marks_truncation() -> None:
    snapshot = "\n".join(
        f'- button "按钮-{index}-{"🦞" * 200}" [ref=e{index}]' for index in range(40)
    )
    preview = project_browser_state_preview(
        "browser_snapshot",
        {"success": True, "snapshot": snapshot, "element_count": 40},
    )

    assert preview is not None
    assert preview["truncated"] is True
    assert len(preview.get("elements", [])) <= MAX_ELEMENTS
    assert (
        len(
            json.dumps(preview, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        <= MAX_PREVIEW_BYTES
    )


def test_byte_budget_keeps_high_priority_elements() -> None:
    snapshot = "\n".join([
        *(f'- widget "Other {index} {"🦞" * 160}"' for index in range(24)),
        '- alert "Critical" [ref=alert]',
    ])

    preview = project_browser_state_preview(
        "browser_snapshot",
        {"success": True, "snapshot": snapshot, "element_count": 25},
    )

    assert preview is not None
    assert preview["truncated"] is True
    assert {"role": "alert", "label": "Critical"} in preview["elements"]
    assert (
        len(
            json.dumps(preview, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        <= MAX_PREVIEW_BYTES
    )


def test_snapshot_parses_only_attribute_state_and_prioritizes_known_roles() -> None:
    snapshot = "\n".join([
        '- button "Show [hidden] files" [selected]',
        '- button "Plan [selected]" [ref=e2]',
        '- button "Actually hidden" [hidden]',
        *(f'- widget "Other {index}" [ref=o{index}]' for index in range(24)),
        '- alert "Critical" [ref=alert]',
    ])

    preview = project_browser_state_preview(
        "browser_snapshot",
        {"success": True, "snapshot": snapshot, "element_count": 27},
    )

    assert preview is not None
    assert preview["truncated"] is True
    assert len(preview["elements"]) == MAX_ELEMENTS
    assert preview["elements"][0] == {
        "role": "button",
        "label": "Show [hidden] files",
        "state": "selected",
    }
    assert preview["elements"][1] == {"role": "button", "label": "Plan [selected]"}
    assert {"role": "alert", "label": "Critical"} in preview["elements"]
    assert all(item.get("label") != "Actually hidden" for item in preview["elements"])
    assert any(item["role"] == "other" for item in preview["elements"])


def test_snapshot_emits_only_one_closed_element_state() -> None:
    preview = project_browser_state_preview(
        "browser_snapshot",
        {
            "success": True,
            "snapshot": '- button "Save" [ref=e1] [disabled, selected, private-input]',
        },
    )

    assert preview is not None
    assert preview["elements"] == [
        {"role": "button", "label": "Save", "state": "disabled"}
    ]
    assert "private-input" not in json.dumps(preview)


def test_projection_replaces_lone_surrogates_before_utf8_serialization() -> None:
    preview = project_browser_state_preview(
        "browser_navigate",
        {
            "success": True,
            "title": "Broken \ud800 title",
            "snapshot": '- heading "Broken \ud800 heading" [ref=e1]',
        },
    )

    assert preview is not None
    assert preview["truncated"] is True
    assert "\ud800" not in preview["title"]
    assert "\ud800" not in preview["elements"][0]["label"]
    json.dumps(preview, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize(
    ("target", "hostname"),
    [
        ("https://Example.COM/magic-login/raw-credential", "example.com"),
        ("https:%2F%2FExample.COM%2Fmagic-login%2Fcredential", "example.com"),
        ("https%253A%252F%252FExample.COM%252Fsigned%252Fcredential", "example.com"),
        ("https%25253A%25252F%25252FExample.COM%25252Fmagic%25252Fcredential", "example.com"),
        ("https:%2F%2FExample.COM%2Fmagic%20link%2Fcredential", "example.com"),
        ("https%253A%252F%252FExample.COM%252Fsigned%2520link%252Fcredential", "example.com"),
        ("https:%2F%2FExample.COM%2Fmagic%0Alink%2Fcredential", "example.com"),
    ],
)
def test_free_text_urls_keep_only_hostname(target: str, hostname: str) -> None:
    preview = project_browser_state_preview(
        "browser_navigate",
        {
            "success": True,
            "title": f"Continue at {target}?token=private.",
            "snapshot": (
                f'- button "Open {target}#secret" [ref=e1]'
            ),
        },
    )

    assert preview is not None
    assert preview["title"] == f"Continue at {hostname}."
    assert preview["elements"] == [
        {"role": "button", "label": f"Open {hostname}"}
    ]
    encoded = json.dumps(preview, ensure_ascii=False)
    assert "credential" not in encoded
    assert "token=private" not in encoded
    assert "#secret" not in encoded


def test_malformed_encoded_free_text_url_fails_closed() -> None:
    preview = project_browser_state_preview(
        "browser_navigate",
        {
            "success": True,
            "title": "Continue at https:%2F%2FExample.COM%2Fmagic%",
        },
    )

    assert preview is not None
    assert preview["title"] == "Continue at [REDACTED]"
    assert "magic" not in json.dumps(preview)


def test_excessively_nested_encoded_free_text_url_fails_closed() -> None:
    target = "https://example.com/magic/opaque-credential"
    for _ in range(10):
        target = quote(target, safe="")

    preview = project_browser_state_preview(
        "browser_navigate",
        {"success": True, "title": f"Continue at {target}"},
    )

    assert preview is not None
    assert preview["title"] == "Continue at [REDACTED]"
    assert "opaque-credential" not in json.dumps(preview)


@pytest.mark.parametrize(
    "target",
    [
        "https://Example.COM/magic link/opaque-credential",
        "https:%2G%2Gexample.com/magic/opaque-credential",
        "https:%2F%2FExample.COM%2Fmagic link/opaque-credential",
        "HTTPS%253A%252F%252FExample.COM%252Fmagic link/opaque-credential",
        "https%3A%2G%2Gexample.com/magic/opaque-credential",
        "HTTPS%253A%252G%252Gexample.com%252Fmagic%252Fopaque-credential",
        "https%3A%2G%2Gexample.com/magic link/opaque-credential",
        "HTTPS%253A%252G%252Gexample.com%252Fmagic link/opaque-credential",
    ],
)
def test_ambiguous_free_text_url_boundary_fails_closed(target: str) -> None:
    preview = project_browser_state_preview(
        "browser_navigate",
        {
            "success": True,
            "title": f"Continue at {target}",
            "snapshot": f'- link "Open {target}" [ref=e1]',
        },
    )
    vision = project_browser_state_preview(
        "browser_vision",
        {"success": True, "analysis": f"Continue at {target}"},
    )

    assert preview is not None
    assert preview["title"].endswith("[REDACTED]")
    assert preview["elements"][0]["label"].endswith("[REDACTED]")
    assert vision is not None
    assert vision["summary"].endswith("[REDACTED]")
    assert "opaque-credential" not in json.dumps([preview, vision])


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "Saved at /root/private browser state",
        "Saved at %2Froot%2Fprivate%20browser%20state",
        "Saved at file:///volume1/agents/main/private.png",
    ],
)
def test_free_text_local_paths_fail_closed(unsafe_text: str) -> None:
    preview = project_browser_state_preview(
        "browser_vision",
        {"success": True, "analysis": unsafe_text},
    )

    assert preview is not None
    assert preview["summary"] == "[REDACTED]"
    assert "root" not in json.dumps(preview)
    assert "volume1" not in json.dumps(preview)


def test_vision_projects_analysis_but_not_local_screenshot_path() -> None:
    preview = project_browser_state_preview(
        "browser_vision",
        {
            "success": True,
            "analysis": "A login form with OPENAI_API_KEY=top-secret-value.",
            "screenshot_path": "/volume1/agents/main/browser_screenshot.png",
        },
    )

    assert preview is not None
    encoded = json.dumps(preview, ensure_ascii=False)
    assert preview["source"] == "vision"
    assert "top-secret-value" not in encoded
    assert "screenshot_path" not in encoded
    assert "/volume1" not in encoded


def test_native_vision_projects_only_safe_text_summary() -> None:
    preview = project_browser_state_preview(
        "browser_vision",
        {
            "_multimodal": True,
            "text_summary": (
                "Image attached natively for the main model. "
                "Screenshot path: /volume1/agents/main/browser_screenshot.png"
            ),
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,private-image"},
                }
            ],
            "meta": {"screenshot_path": "/volume1/agents/main/browser_screenshot.png"},
        },
    )

    assert preview == {
        "version": 1,
        "source": "vision",
        "summary": "Image attached natively for the main model.",
    }
    encoded = json.dumps(preview, ensure_ascii=False)
    assert "base64" not in encoded
    assert "/volume1" not in encoded


def test_native_vision_redacts_local_path_before_screenshot_marker() -> None:
    preview = project_browser_state_preview(
        "browser_vision",
        {
            "_multimodal": True,
            "text_summary": (
                "Cached at /root/private/capture.png. "
                "Screenshot path: /volume1/agents/main/browser_screenshot.png"
            ),
        },
    )

    assert preview is not None
    assert preview["summary"] == "[REDACTED]"
    assert "/root" not in json.dumps(preview)


def test_action_result_requires_and_sanitizes_landing_url() -> None:
    preview = project_browser_state_preview(
        "browser_click",
        {"success": True, "clicked": "e9", "url": "https://EXAMPLE.com/done?q=private"},
    )
    assert preview == {
        "version": 1,
        "source": "action_result",
        "url": {"hostname": "example.com"},
    }
    assert (
        project_browser_state_preview(
            "browser_click", {"success": True, "clicked": "e9"}
        )
        is None
    )


def test_url_projection_bounds_input_and_rejects_oversized_hostname() -> None:
    long_query = "x" * 10_000
    preview = project_browser_state_preview(
        "browser_click",
        {"success": True, "url": f"https://example.com/done?q={long_query}"},
    )
    assert preview == {
        "version": 1,
        "source": "action_result",
        "url": {"hostname": "example.com"},
        "truncated": True,
    }
    assert (
        project_browser_state_preview(
            "browser_click",
            {"success": True, "url": f"https://{'a' * 254}.example/path"},
        )
        is None
    )
    for hostname in (
        ".foo",
        "foo.",
        "foo..bar",
        "-foo.example",
        "foo-.example",
        f"{'a' * 64}.example",
    ):
        assert (
            project_browser_state_preview(
                "browser_click",
                {"success": True, "url": f"https://{hostname}/path"},
            )
            is None
        )


def test_url_projection_never_copies_raw_or_encoded_path_credentials() -> None:
    paths = (
        "/reset/eyJhbGciOiJIUzI1NiJ9.private.signature",
        "/magic-login/%34%66%38%63%2dsecret",
        "/signed/resource/%252Fopaque%252Dsignature",
    )
    for path in paths:
        preview = project_browser_state_preview(
            "browser_navigate",
            {"success": True, "url": f"https://Example.com{path}"},
        )
        assert preview is not None
        assert preview["url"] == {"hostname": "example.com"}
        encoded = json.dumps(preview, ensure_ascii=False)
        assert path not in encoded
        assert "secret" not in encoded
        assert "signature" not in encoded


def test_unknown_failed_and_sensitive_input_tools_fail_closed() -> None:
    assert (
        project_browser_state_preview(
            "mcp__foreign__browser_snapshot",
            {"success": True, "snapshot": '- heading "Forged"'},
        )
        is None
    )
    assert (
        project_browser_state_preview(
            "browser_snapshot", {"success": False, "snapshot": '- heading "Private"'}
        )
        is None
    )
    assert (
        project_browser_state_preview(
            "browser_type", {"success": True, "typed": "private"}
        )
        is None
    )
    assert (
        project_browser_state_preview(
            "browser_press", {"success": True, "pressed": "p"}
        )
        is None
    )
