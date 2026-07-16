import time

import pytest

from portable_import_security import portable_credential_finding


@pytest.mark.parametrize(
    "value",
    [
        "Authorization=Bearer abcdefghijklmnop",
        "api key = abcdefghijklmnop",
        "Authorization:\u0020Bearer\u0020abcdefghijklmnop",
        r"{\"Authorization\":\u0020\"Bearer\u0020abcdefghijklmnop\"}",
    ],
    ids=[
        "authorization-equals",
        "spaced-api-key",
        "decoded-unicode-space",
        "literal-json-escapes",
    ],
)
def test_portable_credential_scanner_rejects_assignment_and_escape_forms(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "xoxb-123456789012-123456789012-abcdefghijklmnopqrstuvwxyzABCD",
        "AIza" + "A" * 35,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlX3ZhbHVl",
        "https://api.example.test/v1/items?access_token=abcdefghijklmnop",
    ],
    ids=["slack", "google-api", "jwt", "url-query"],
)
def test_portable_credential_scanner_rejects_high_confidence_bare_and_query_forms(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "Authorization=Bearer redacted",
        "api key = ${OPENAI_API_KEY}",
        r"{\"Authorization\":\u0020\"Bearer\u0020${ACCESS_TOKEN}\"}",
        "https://api.example.test/v1/items?access_token=${ACCESS_TOKEN}",
        "https://api.example.test/v1/items?api_key=redacted",
    ],
)
def test_portable_credential_scanner_keeps_placeholders(value):
    assert portable_credential_finding(value) is None


def test_portable_credential_scanner_bounds_long_non_matching_prose():
    value = ("ordinary words without credential assignments " * 2000)[:65_536]

    started_at = time.perf_counter()
    finding = portable_credential_finding(value)
    elapsed = time.perf_counter() - started_at

    assert finding is None
    assert elapsed < 2.0, f"credential scan took {elapsed:.3f}s"


def test_portable_credential_scanner_bounds_long_escape_run():
    value = "\\" * 65_536

    started_at = time.perf_counter()
    finding = portable_credential_finding(value)
    elapsed = time.perf_counter() - started_at

    assert finding is None
    assert elapsed < 2.0, f"credential scan took {elapsed:.3f}s"


def test_portable_credential_scanner_bounds_long_non_credential_query():
    value = ("https://example.test/items?mode=ordinary&" * 2000)[:65_536]

    started_at = time.perf_counter()
    finding = portable_credential_finding(value)
    elapsed = time.perf_counter() - started_at

    assert finding is None
    assert elapsed < 2.0, f"credential scan took {elapsed:.3f}s"
