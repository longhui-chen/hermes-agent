from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_retired_business_transport_does_not_reappear() -> None:
    retired_markers = (
        "ZETTLAB_BUSINESS_EXECUTION_" + "TOKEN",
        "X-Zettlab-Business-Execution-" + "Token",
        "/authorization/" + "check",
    )
    for relative_root in ("agent", "gateway", "tools"):
        for path in (ROOT / relative_root).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in retired_markers:
                assert marker not in source, f"retired transport marker in {path}"


def test_video_and_hardware_bearer_storage_are_absent() -> None:
    session_context = (ROOT / "gateway/session_context.py").read_text(encoding="utf-8")
    assert "ZETTLAB_BUSINESS_EXECUTION_ACTION_VERSION" not in session_context
    # 52f94017af removed the session-scoped hardware bearer storage; camera and
    # other hardware helpers now authorize through owner context instead.
    assert "ZETTLAB_HARDWARE_EXECUTION_TOKEN" not in session_context
