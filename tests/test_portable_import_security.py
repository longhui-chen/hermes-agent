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
        r"Authorization\u005cu003a\u005cu0020Bearer\u005cu0020abcdefghijklmnop",
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
        "nested-literal-json-escapes",
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


@pytest.mark.parametrize(
    "value",
    [
        "client-key-data: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0tCk1JSUV2UUlCQURBTg==",
        "//registry.npmjs.org/:_auth=dXNlcjpwYXNzd29yZDEyMw==",
        "_auth=dXNlcjpwYXNzd29yZDEyMw==",
        '{"auths":{"registry.example.com":{"auth":"dXNlcjpwYXNzd29yZA=="}}}',
        '{"identitytoken":"dXNlcjpwYXNzd29yZDEyMzQ1"}',
        '{"registrytoken":"dXNlcjpwYXNzd29yZDEyMzQ1"}',
    ],
    ids=[
        "kubeconfig-client-key-data",
        "npmrc-scoped-registry-auth",
        "npmrc-bare-auth",
        "docker-config-auth",
        "docker-config-identitytoken",
        "docker-config-registrytoken",
    ],
)
def test_portable_credential_scanner_rejects_encoded_credential_field_shapes(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "auth: enabled",
        "the auth mode discussion text continues here",
        '{"auth": true}',
        "_auth=${NPM_AUTH_TOKEN}",
    ],
    ids=[
        "auth-enabled-prose",
        "auth-mode-discussion-prose",
        "auth-boolean-json",
        "npmrc-auth-placeholder",
    ],
)
def test_portable_credential_scanner_keeps_benign_auth_lookalikes(value):
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


def _nested_ascii_escape(value: str, depth: int) -> str:
    for _ in range(depth - 1):
        value = value.replace("\\", r"\u005c")
    return value


def test_portable_credential_scanner_detects_at_normalization_depth_limit():
    value = _nested_ascii_escape(
        r"Authorization\u003a\u0020Bearer\u0020abcdefghijklmnop", 8
    )
    assert portable_credential_finding(value) is not None


def test_portable_credential_scanner_fails_closed_beyond_depth_limit():
    value = _nested_ascii_escape(r"ordinary\u003a text", 9)
    assert portable_credential_finding(value) == "excessively nested escaped text"


def test_portable_credential_scanner_keeps_nested_near_miss():
    value = _nested_ascii_escape(r"Authorization\u003a\u0020Bearer\u0020redacted", 2)
    assert portable_credential_finding(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "clone https://alice:s3cr3t-pass@example.com/repo.git",
        "remote add origin https://token:x-oauth-basic@github.com/x.git",
        "DATABASE_URL=postgresql://admin:supersecret@db-host:5432/app",
        "redis://:password123@cache:6379/0",
    ],
)
def test_portable_credential_scanner_rejects_url_userinfo_passwords(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "https://user:${DB_PASS}@host/db",
        "connect to http://example.com:8080/health",
        "docs at https://example.com/a/b?ref=main",
        "https://alice@example.com/repo.git",
    ],
)
def test_portable_credential_scanner_allows_url_without_userinfo_password(value):
    assert portable_credential_finding(value) is None


@pytest.mark.parametrize(
    "value",
    [
        "client-key-data: |\n  LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t\n  bW9yZWJhc2U2NA==\n",
        "password: >-\n  supersecretvalue123\n",
        "api_key: |\n  AKIA1234567890ABCDEF\n",
    ],
)
def test_portable_credential_scanner_rejects_yaml_block_scalar_secrets(value):
    assert portable_credential_finding(value) is not None


@pytest.mark.parametrize(
    "value",
    [
        "description: |\n  This is a normal multi-line note.\n",
        "password: |\n  ${SECRET_VALUE}\n",
        "notes: >-\n  wrap this long sentence across lines\n",
    ],
)
def test_portable_credential_scanner_allows_benign_block_scalars(value):
    assert portable_credential_finding(value) is None
