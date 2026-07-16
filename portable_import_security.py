"""Credential-free policy shared by portable import consumers."""

import re
from typing import Optional


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.IGNORECASE | re.MULTILINE
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])authorization(?:\s*\\?[\"'])?\s*:\s*"
    r"(?:\\?[\"']\s*)?bearer\s+([A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE | re.MULTILINE,
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s{,])[\"']?"
    r"((?:[a-z0-9]+[._-])*(?:api[._-]?key|client[._-]?secret|"
    r"secret[._-]?access[._-]?key|access[._-]?token|refresh[._-]?token|"
    r"auth[._-]?token|credentials?|secret|token|password|private[._-]?key))"
    r"[\"']?\s*[:=]\s*(\"[^\"\r\n]+\"|'[^'\r\n]+'|"
    r"[^\s,}\]\r\n#]+)",
    re.IGNORECASE | re.MULTILINE,
)
_HIGH_CONFIDENCE_BARE_TOKEN_RE = re.compile(
    r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{20,})\b"
)
_PLACEHOLDER_VALUE_RE = re.compile(
    r"(?:\$(?:\{[A-Z_][A-Z0-9_]*\}?|[A-Z_][A-Z0-9_]*)|<[^<>\r\n]+>|"
    r"env\.[A-Z_][A-Z0-9_]*|process\.env\.[A-Z_][A-Z0-9_]*|"
    r"your[-_](?:api[-_]?key|token|secret|password)(?:[-_]here)?|"
    r"example(?:[-_](?:token|key|secret|value))*|"
    r"redacted|changeme|sk-example|[x*_-]+)",
    re.IGNORECASE,
)


def _credential_value_looks_real(raw: str) -> bool:
    value = raw.strip().strip("\"'")
    return len(value) >= 6 and _PLACEHOLDER_VALUE_RE.fullmatch(value) is None


def portable_credential_finding(value: str) -> Optional[str]:
    """Return the first high-confidence credential violation in bounded text."""
    if _PRIVATE_KEY_RE.search(value):
        return "private key"
    if _HIGH_CONFIDENCE_BARE_TOKEN_RE.search(value):
        return "credential material"
    for match in _AUTHORIZATION_BEARER_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "bearer credential"
    for match in _CREDENTIAL_ASSIGNMENT_RE.finditer(value):
        if _credential_value_looks_real(match.group(2)):
            return "credential assignment"
    return None


def reject_portable_credentials(value: str, *, field: str) -> None:
    """Reject input that is not credential-free under the V2.5 contract."""
    finding = portable_credential_finding(value)
    if finding:
        raise ValueError(f"{field} contains forbidden {finding}")
