"""Credential-free policy shared by portable import consumers."""

import re
from typing import Optional


_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: [A-Z0-9][A-Z0-9 ]*)?-----",
    re.IGNORECASE | re.MULTILINE,
)
_AUTHORIZATION_BEARER_RE = re.compile(
    r"(?<![A-Za-z0-9_-])authorization(?:\s*\\?[\"'])?\s*[:=]\s*"
    r"(?:\\?[\"']\s*)?bearer\s+([A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE | re.MULTILINE,
)
# Shared credential-field vocabulary, reused by the inline-assignment and the
# YAML block-scalar matchers so both stay in sync.
_CREDENTIAL_KEY_VOCAB = (
    r"(?:[a-z0-9]+[._-])*(?:api[._ -]?key|client[._ -]?key[._ -]?data|client[._ -]?secret|"
    r"secret[._ -]?access[._ -]?key|access[._ -]?key[._ -]?id|"
    r"account[._ -]?key|subscription[._ -]?key|access[._ -]?token|refresh[._ -]?token|"
    r"auth[._ -]?token|authorization|identitytoken|registrytoken|_?auth|credentials?|secret|token|password|"
    r"passwd|cookie|private[._ -]?key)"
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?:^|[\s{,:])[\"']?"
    r"(" + _CREDENTIAL_KEY_VOCAB + r")"
    r"[\"']?\s*[:=]\s*(\"[^\"\r\n]+\"|'[^'\r\n]+'|"
    r"[^\s,}\]\r\n#]+)",
    re.IGNORECASE | re.MULTILINE,
)
# YAML block scalars carry the value on the following lines, so the inline
# matcher above (which stops at the newline) only sees the `|`/`>` indicator
# and misses the real secret, e.g.
#   client-key-data: |
#     LS0tLS1CRUdJTi...
# Capture the whole block — including leading/interior blank lines a YAML
# scalar allows — up to the first dedented non-blank line, and check EVERY
# content line, so neither a blank first line nor a placeholder first line
# followed by the real key can slip through.
_CREDENTIAL_BLOCK_SCALAR_RE = re.compile(
    r"(?:^|[\s{,])[\"']?"
    r"(" + _CREDENTIAL_KEY_VOCAB + r")"
    r"[\"']?[ \t]*:[ \t]*"
    # Full block-scalar header: optional tag(s)/anchor(s) (e.g. !!str, &a),
    # the |/> indicator with optional chomp/indent, and an optional trailing
    # comment — all before the newline that starts the indented block.
    r"(?:(?:!!?[\w./+-]*|&[\w-]+)[ \t]+)*"
    # |/> then chomp(+/-) and indent(1-9) in EITHER order (|2-, |-2, >2+, ...)
    r"[|>][+\-0-9]*[ \t]*(?:\#[^\r\n]*)?\r?\n"
    r"((?:[ \t]*\r?\n|[ \t]+\S[^\r\n]*(?:\r?\n|$))+)",
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
# Credentials embedded in a URL's userinfo — scheme://user:password@host, as in
# git remotes (https://user:token@github.com) and DB DSNs
# (postgresql://admin:secret@host/db). The high-confidence bare-token scan
# already catches structured tokens anywhere (including the user field), so this
# only needs the password segment after the colon. A bare host:port (no '@') and
# a userless authority never match.
_URL_USERINFO_CREDENTIAL_RE = re.compile(
    r"(?:[a-z][a-z0-9+.\-]*)://[^\s/:@]*:([^\s/@]+)@",
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
_MAX_ESCAPE_NORMALIZATION_PASSES = 8


def _credential_scan_text(value: str) -> tuple[str, bool]:
    """Decode only escaped ASCII syntax needed by the credential scanner.

    Imported transcript and memory entries can themselves contain serialized
    JSON. The outer request parser therefore leaves sequences such as
    ``\\u0020`` and ``\\\"`` in the final text value. Decoding the bounded ASCII
    form exposes credential separators without interpreting arbitrary Unicode
    or executing a general-purpose escape codec.
    """
    for _ in range(_MAX_ESCAPE_NORMALIZATION_PASSES):
        normalized = _ASCII_UNICODE_ESCAPE_RE.sub(
            lambda match: chr(int(match.group(1), 16)), value
        )
        normalized = _ESCAPED_QUOTE_RE.sub(
            lambda match: match.group(1), normalized
        )
        if normalized == value:
            return normalized, True
        value = normalized

    # One bounded look-ahead distinguishes a value that became stable exactly
    # at the limit from deeper attacker-controlled nesting. Fail closed on the
    # latter instead of letting a still-obscured credential reach persistence.
    normalized = _ASCII_UNICODE_ESCAPE_RE.sub(
        lambda match: chr(int(match.group(1), 16)), value
    )
    normalized = _ESCAPED_QUOTE_RE.sub(lambda match: match.group(1), normalized)
    return value, normalized == value


def _credential_value_looks_real(raw: str) -> bool:
    value = raw.strip().strip("\"'")
    return len(value) >= 6 and _PLACEHOLDER_VALUE_RE.fullmatch(value) is None


def _authorization_value_looks_real(raw: str) -> bool:
    value = raw.strip().strip("\"'")
    scheme_match = re.fullmatch(
        r"[A-Za-z][A-Za-z0-9._~-]*\s+(.+)", value, re.IGNORECASE
    )
    if scheme_match:
        parameter = scheme_match.group(1).strip().strip("\"'")
        return bool(parameter) and _PLACEHOLDER_VALUE_RE.fullmatch(parameter) is None
    return _credential_value_looks_real(value)


_ENCODED_AUTH_VALUE_RE = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")


def _encoded_auth_value_looks_real(raw: str) -> bool:
    # The bare "auth" key also matches innocuous prose (e.g. "auth: enabled"),
    # so it requires the value to look like an encoded credential rather than
    # just being non-placeholder and non-trivial in length.
    value = raw.strip().strip("\"'")
    if len(value) < 12 or _PLACEHOLDER_VALUE_RE.fullmatch(value) is not None:
        return False
    if _ENCODED_AUTH_VALUE_RE.fullmatch(value) is None:
        return False
    return re.search(r"[0-9+/=]|[A-Z]", value) is not None


def portable_credential_finding(value: str) -> Optional[str]:
    """Return the first high-confidence credential violation in bounded text."""
    value, normalization_complete = _credential_scan_text(value)
    if not normalization_complete:
        return "excessively nested escaped text"
    if _PRIVATE_KEY_RE.search(value):
        return "private key"
    if _HIGH_CONFIDENCE_BARE_TOKEN_RE.search(value):
        return "credential material"
    for match in _URL_QUERY_CREDENTIAL_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "URL query credential"
    for match in _URL_USERINFO_CREDENTIAL_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "URL userinfo credential"
    for match in _CREDENTIAL_BLOCK_SCALAR_RE.finditer(value):
        # The regex greedily grabs every following indented line, but a YAML
        # block scalar ends when indentation dedents back to a sibling/parent
        # mapping key. Trim to the real block: the first non-blank line fixes
        # the content indent; stop at the first non-blank line shallower than
        # it. Without this, a nested `password: |` whose value is a placeholder
        # would swallow a following `username: admin` sibling and false-reject
        # a credential-free import.
        content_indent = None
        block_lines = []
        for line in match.group(2).splitlines():
            if not line.strip():
                block_lines.append(line)
                continue
            indent = len(line) - len(line.lstrip(" \t"))
            if content_indent is None:
                content_indent = indent
            elif indent < content_indent:
                break
            block_lines.append(line)

        lines = [ln.strip() for ln in block_lines if ln.strip()]
        # Check each physical line (a literal `|` block puts a full secret on
        # each line, e.g. base64) AND the space-joined value (a folded `>`
        # block splits one secret across short lines that only exceed the
        # threshold once folded).
        candidates = list(lines)
        if len(lines) > 1:
            candidates.append(" ".join(lines))
        for candidate in candidates:
            if (
                len(candidate) >= 6
                and _PLACEHOLDER_VALUE_RE.fullmatch(candidate) is None
            ):
                return "credential block scalar"
    for match in _AUTHORIZATION_BEARER_RE.finditer(value):
        if _credential_value_looks_real(match.group(1)):
            return "bearer credential"
    for match in _CREDENTIAL_ASSIGNMENT_RE.finditer(value):
        key = re.sub(r"[._ -]", "", match.group(1)).lower()
        raw_value = match.group(2)
        authorization_scheme = raw_value.strip("\"'")
        if key == "authorization" and re.fullmatch(
            r"[A-Za-z][A-Za-z0-9._~-]*", authorization_scheme
        ):
            authorization_value = re.match(
                r"\s+([^\s,}\]\r\n#]+)", value[match.end() :]
            )
            if authorization_value:
                raw_value = f"{authorization_scheme} {authorization_value.group(1)}"
        if key == "authorization":
            value_looks_real = _authorization_value_looks_real(raw_value)
        elif key == "auth":
            value_looks_real = _encoded_auth_value_looks_real(raw_value)
        else:
            value_looks_real = _credential_value_looks_real(raw_value)
        if value_looks_real:
            return "credential assignment"
    return None


def reject_portable_credentials(value: str, *, field: str) -> None:
    """Reject input that is not credential-free under the V2.5 contract."""
    finding = portable_credential_finding(value)
    if finding:
        raise ValueError(f"{field} contains forbidden {finding}")
