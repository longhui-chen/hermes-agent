import time

import pytest

from portable_import_security import portable_credential_finding


@pytest.mark.parametrize(
    "value",
    [
        "Authorization=Bearer abcdefghijklmnop",
        "Authorization: Basic dXNlcjpwYXNzd29yZA==",
        "Authorization: Basic YTo=",
        '{"Authorization": "Basic dXNlcjpwYXNzd29yZA=="}',
        "api key = abcdefghijklmnop",
        "Authorization:\u0020Bearer\u0020abcdefghijklmnop",
        r"{\"Authorization\":\u0020\"Bearer\u0020abcdefghijklmnop\"}",
        "-----BEGIN PGP PRIVATE KEY BLOCK-----",
    ],
    ids=[
        "authorization-equals",
        "authorization-basic",
        "authorization-basic-short",
        "authorization-basic-json",
        "spaced-api-key",
        "decoded-unicode-space",
        "literal-json-escapes",
        "pgp-private-key-block",
    ],
)
def test_portable_credential_scanner_rejects_assignment_and_escape_forms(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7REAL",
        '{"AZURE_STORAGE_ACCOUNT_KEY":"abcdefghijklmnop"}',
        "AZURE_SUBSCRIPTION_KEY: abcdefghijklmnop",
    ],
    ids=["aws-access-key-id", "azure-storage-account-key", "azure-subscription-key"],
)
def test_portable_credential_scanner_rejects_vendor_assignment_vocabulary(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "public_key=abcdefghijklmnop",
        "account_id=1234567890123456",
        "subscription_key_hint=rotate-quarterly",
    ],
    ids=["public-key", "account-id", "subscription-key-hint"],
)
def test_portable_credential_scanner_allows_non_secret_assignment_neighbors(value):
    assert portable_credential_finding(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "xoxb-123456789012-123456789012-abcdefghijklmnopqrstuvwxyzABCD",
        "xapp-1-123456789012-abcdefghijklmnopqrstuvwxyzABCD",
        "AIza" + "A" * 35,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.c2lnbmF0dXJlX3ZhbHVl",
        "https://api.example.test/v1/items?access_token=abcdefghijklmnop",
        "https://bucket.s3.amazonaws.com/item?X-Amz-Signature=" + "a" * 64,
        "https://bucket.s3.amazonaws.com/item?X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20260717%2Fus-east-1%2Fs3%2Faws4_request",
    ],
    ids=["slack-bot", "slack-app", "google-api", "jwt", "url-query", "aws-signature", "aws-credential"],
)
def test_portable_credential_scanner_rejects_high_confidence_bare_and_query_forms(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "Authorization=Bearer redacted",
        "Authorization: Basic ${BASIC_AUTH}",
        "api key = ${OPENAI_API_KEY}",
        r"{\"Authorization\":\u0020\"Bearer\u0020${ACCESS_TOKEN}\"}",
        "xapp-your-app-token-here",
        "xapp-1-<SLACK_APP_TOKEN>",
        "https://api.example.test/v1/items?access_token=${ACCESS_TOKEN}",
        "https://api.example.test/v1/items?api_key=redacted",
        "https://bucket.s3.amazonaws.com/item?X-Amz-Signature=${AWS_SIGNATURE}",
        "https://bucket.s3.amazonaws.com/item?X-Amz-Credential=<AWS_CREDENTIAL>",
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
