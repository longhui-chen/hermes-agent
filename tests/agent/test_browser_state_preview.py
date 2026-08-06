import json

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
