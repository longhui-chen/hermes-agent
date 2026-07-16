"""Credential-free policy shared by portable import consumers."""

import re
from typing import Optional


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE | re.MULTILINE
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])authorization(?:\s*\\?[\"'])?\s*[:=]\s*"
    r"(?:\\?[\"']\s*)?bearer\s+([A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE | re.MULTILINE,
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s{,])[\"']?"
    r"((?:[a-z0-9]+[._-])*(?:api[._ -]?key|client[._ -]?secret|"
    r"secret[._ -]?access[._ -]?key|access[._ -]?key[._ -]?id|"
    r"account[._ -]?key|subscription[._ -]?key|access[._ -]?token|refresh[._ -]?token|"
    r"auth[._ -]?token|authorization|credentials?|secret|token|password|"
    r"passwd|cookie|private[._ -]?key))"
    r"[\"']?\s*[:=]\s*(\"[^\"\r\n]+\"|'[^'\r\n]+'|"
    r"[^\s,}\]\r\n#]+)",
    re.IGNORECASE | re.MULTILINE,
)
_HIGH_CONFIDENCE_BARE_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}|"
    r"xapp-\d+-[A-Za-z0-9-]{10,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AIza[A-Za-z0-9_-]{35}|"
    r"eyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b"
)
_URL_QUERY_CREDENTIAL_RE = re.compile(
    r"[?&](?:api[._-]?key|access[._-]?token|refresh[._-]?token|"
    r"auth[._-]?token|authorization|client[._-]?secret|secret|token|"
    r"password|passwd|cookie|credentials?|x-amz-signature|"
    r"x-amz-credential)=([^&#\s]+)",
    re.IGNORECASE,
)
_PLACEHOLDER_VALUE_RE = re.compile(
    r"(?:\$(?:\{[A-Z_][A-Z0-9_]*\}?|[A-Z_][A-Z0-9_]*)|<[^<>\r\n]+>|"
    r"env\.[A-Z_][A-Z0-9_]*|process\.env\.[A-Z_][A-Z0-9_]*|"
    r"your[-_](?:api[-_]?key|token|secret|password)(?:[-_]here)?|"
    r"example(?:[-_](?:token|key|secret|value))*|"
    r"redacted|changeme|sk-example|[x*_-]+)",
    re.IGNORECASE,
)
_ASCII_UNICODE_ESCAPE_RE = re.compile(
    r"(?<!\\)\\+u00([0-7][0-9a-f])", re.IGNORECASE
)
_ESCAPED_QUOTE_RE = re.compile(r"(?<!\\)\\+([\"'])")


def _credential_scan_text(value: str) -> str:
    """Decode only escaped ASCII syntax needed by the credential scanner.

    Imported transcript and memory entries can themselves contain serialized
    JSON. The outer request parser therefore leaves sequences such as
    ``\\u0020`` and ``\\\"`` in the final text value. Decoding the bounded ASCII
    form exposes credential separators without interpreting arbitrary Unicode
    or executing a general-purpose escape codec.
    """
    value = _ASCII_UNICODE_ESCAPE_RE.sub(
        lambda match: chr(int(match.group(1), 16)), value
    )
    return _ESCAPED_QUOTE_RE.sub(lambda match: match.group(1), value)


def _credential_value_looks_real(raw: str) -> bool:
    value = raw.strip().strip("\"'")
    return len(value) >= 6 and _PLACEHOLDER_VALUE_RE.fullmatch(value) is None


def _authorization_value_looks_real(raw: str) -> bool:
    value = raw.strip().strip("\"'")
    bearer_match = re.fullmatch(r"bearer\s+(.+)", value, re.IGNORECASE)
    if bearer_match:
        return _credential_value_looks_real(bearer_match.group(1))
    return _credential_value_looks_real(value)


def portable_credential_finding(value: str) -> Optional[str]:
    """Return the first high-confidence credential violation in bounded text."""
    value = _credential_scan_text(value)
    if _PRIVATE_KEY_RE.search(value):
        return "private key"
    if _HIGH_CONFIDENCE_BARE_TOKEN_RE.search(value):
        return "credential material"
    for match in _URL_QUERY_CREDENTIAL_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "URL query credential"
    for match in _AUTHORIZATION_BEARER_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "bearer credential"
    for match in _CREDENTIAL_ASSIGNMENT_RE.finditer(value):
        key = re.sub(r"[._ -]", "", match.group(1)).lower()
        raw_value = match.group(2)
        if key == "authorization" and raw_value.strip("\"'").lower() == "bearer":
            bearer_value = re.match(
                r"\s+([^\s,}\]\r\n#]+)", value[match.end() :]
            )
            if bearer_value:
                raw_value = f"Bearer {bearer_value.group(1)}"
        value_looks_real = (
            _authorization_value_looks_real(raw_value)
            if key == "authorization"
            else _credential_value_looks_real(raw_value)
        )
        if value_looks_real:
            return "credential assignment"
    return None


def reject_portable_credentials(value: str, *, field: str) -> None:
    """Reject input that is not credential-free under the V2.5 contract."""
    finding = portable_credential_finding(value)
    if finding:
        raise ValueError(f"{field} contains forbidden {finding}")
