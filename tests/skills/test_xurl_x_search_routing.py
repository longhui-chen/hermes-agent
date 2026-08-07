"""Behavioral contract for xurl / x_search routing guidance.

The fork does not bundle the xurl business skill, so the remaining contract
covers the x_search documentation surface where users are routed between the
public discovery tool and an externally installed account-action integration.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
X_SEARCH_DOC = REPO_ROOT / "website" / "docs" / "user-guide" / "features" / "x-search.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_any(text: str, *needles: str) -> bool:
    lowered = text.lower()
    return any(n.lower() in lowered for n in needles)


def test_x_search_doc_separates_discovery_from_account_actions():
    text = _read(X_SEARCH_DOC)
    lowered = text.lower()

    assert "x_search" in lowered
    assert "xurl" in lowered
    # Explicit comparison section or equivalent boundary language.
    assert _contains_any(text, "vs `xurl`", "vs xurl", "two different x surfaces")
    assert _contains_any(text, "read-only public", "public x discovery")
    assert _contains_any(
        text,
        "posting",
        "replying",
        "liking",
        "dm",
        "media upload",
        "deleting",
    )
    assert _contains_any(
        text,
        "authenticated",
        "exact or authenticated",
        "account actions",
        "state-changing",
    )
    # Write confirmation must come from xurl / X API, not x_search.
    assert _contains_any(
        text,
        "confirmed by `xurl`",
        "xurl` output",
        "x api response",
        "never evidence",
    )
    assert _contains_any(text, "switch to the `xurl`", "switch to `xurl`", "xurl skill")
