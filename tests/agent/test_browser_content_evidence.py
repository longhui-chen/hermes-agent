import json
from pathlib import Path

from agent.browser_content_evidence import (
    MAX_BLOCKS,
    MAX_BLOCK_CHARS,
    MAX_EVIDENCE_BYTES,
    project_browser_content_evidence,
)


FIXTURE = Path(__file__).parents[1] / "fixtures" / "browser_content_evidence_v1.json"


def test_browser_content_evidence_golden_vectors():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["version"] == 1
    for vector in fixture["vectors"]:
        assert (
            project_browser_content_evidence(
                vector["tool_name"],
                vector["output"],
                browser_session_id=vector.get("browser_session_id"),
            )
            == vector["expected"]
        ), vector["name"]


def test_browser_content_evidence_requires_success_and_provenance():
    snapshot = '- heading "Private"\n- text "Body"'
    assert (
        project_browser_content_evidence(
            "browser_snapshot", {"success": False, "snapshot": snapshot}
        )
        is None
    )
    assert (
        project_browser_content_evidence(
            "browser_snapshot", {"success": True, "snapshot": snapshot}
        )
        is None
    )
    assert (
        project_browser_content_evidence(
            "browser_snapshot",
            {
                "success": True,
                "snapshot": snapshot,
                "_browser_content_provenance": "vision_analysis",
            },
        )
        is None
    )
    assert (
        project_browser_content_evidence(
            "browser_click",
            {
                "success": True,
                "snapshot": snapshot,
                "_browser_content_provenance": "page_text",
            },
        )
        is None
    )


def test_browser_content_evidence_sanitizes_and_omits_input_values():
    evidence = project_browser_content_evidence(
        "browser_snapshot",
        {
            "success": True,
            "_browser_content_provenance": "page_text",
            "snapshot": (
                '- heading "Docs at https://example.com/reset/secret"\n'
                '- text "API key: sk-abcdefghijklmnop"\n'
                '- textbox "typed-password"\n'
                '- text "/Users/alice/private.txt"\n'
                '- text "Safe paragraph"'
            ),
        },
    )
    assert evidence is not None
    encoded = json.dumps(evidence)
    assert "reset/secret" not in encoded
    assert "sk-abcdefghijklmnop" not in encoded
    assert "typed-password" not in encoded
    assert "/Users/alice" not in encoded
    assert evidence["blocks"][-1] == {"kind": "paragraph", "text": "Safe paragraph"}
    assert "sanitization" in evidence["truncationReasons"]


def test_browser_content_evidence_prefers_semantic_role_when_link_appears_first():
    evidence = project_browser_content_evidence(
        "browser_snapshot",
        {
            "success": True,
            "_browser_content_provenance": "page_text",
            "snapshot": '- link "Quarterly report"\n- heading "Quarterly report"',
        },
    )
    assert evidence is not None
    assert evidence["blocks"] == [{"kind": "heading", "text": "Quarterly report"}]
    assert evidence["duplicateBlockCount"] == 1


def test_browser_content_evidence_enforces_block_and_byte_budgets():
    long_text = "🦞" * (MAX_BLOCK_CHARS + 100)
    snapshot = "\n".join(
        f'- text "{index}-{long_text}"' for index in range(MAX_BLOCKS + 10)
    )
    evidence = project_browser_content_evidence(
        "browser_snapshot",
        {
            "success": True,
            "_browser_content_provenance": "page_text",
            "snapshot": snapshot,
        },
    )
    assert evidence is not None
    assert len(evidence["blocks"]) <= MAX_BLOCKS
    assert all(len(block["text"]) <= MAX_BLOCK_CHARS for block in evidence["blocks"])
    assert (
        len(
            json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        <= MAX_EVIDENCE_BYTES
    )
    assert evidence["truncated"] is True
    assert "step_budget" in evidence["truncationReasons"]


def test_tool_completion_payload_projects_browser_content_evidence():
    from gateway.platforms.api_server import _tool_completion_payload

    payload = _tool_completion_payload(
        "call-1",
        "browser_snapshot",
        json.dumps({
            "success": True,
            "_browser_content_provenance": "page_text",
            "snapshot": '- heading "Account"\n- text "Current balance"',
        }),
    )
    assert payload["browserContentEvidence"] == {
        "version": 1,
        "provenance": "page_text",
        "blocks": [
            {"kind": "heading", "text": "Account"},
            {"kind": "paragraph", "text": "Current balance"},
        ],
        "originalBlockCount": 2,
    }


def test_browser_content_evidence_projection_failure_isolated(monkeypatch):
    from gateway.platforms import api_server

    monkeypatch.setattr(
        api_server,
        "project_browser_content_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    payload = api_server._tool_completion_payload(
        "call-1",
        "browser_snapshot",
        json.dumps({
            "success": True,
            "_browser_content_provenance": "page_text",
            "snapshot": '- heading "Still completed"',
        }),
    )
    assert payload["status"] == "completed"
    assert payload["outcome"] == "success"
    assert "browserContentEvidence" not in payload
