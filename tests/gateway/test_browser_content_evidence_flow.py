import json
from pathlib import Path

from gateway.platforms.api_server import _tool_completion_payload


FIXTURE = Path(__file__).parents[1] / "fixtures" / "browser_content_evidence_v1.json"


def test_model_facing_browser_result_flows_to_tool_progress_contract():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for index, vector in enumerate(fixture["vectors"]):
        payload = _tool_completion_payload(
            f"call-{index}",
            vector["tool_name"],
            json.dumps(vector["output"], ensure_ascii=False),
        )
        assert payload["status"] == "completed"
        expected = vector["expected"].copy()
        expected.pop("browserSessionId", None)
        assert payload["browserContentEvidence"] == expected, vector["name"]


def test_non_browser_and_failed_browser_results_keep_existing_completion_shape():
    non_browser = _tool_completion_payload(
        "call-terminal",
        "terminal",
        json.dumps({"success": True, "output": "plain text"}),
    )
    failed_browser = _tool_completion_payload(
        "call-browser",
        "browser_snapshot",
        json.dumps({
            "success": False,
            "error": "snapshot failed",
            "_browser_content_provenance": "page_text",
            "snapshot": '- text "must not persist"',
        }),
    )
    assert "browserContentEvidence" not in non_browser
    assert "browserContentEvidence" not in failed_browser
    assert non_browser["outcome"] == "success"
    assert failed_browser["outcome"] == "error"
