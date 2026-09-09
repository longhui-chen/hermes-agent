"""Reject credential solicitation through ordinary Memo clarification.

This is a refusal guard, never a prose-to-Connector router. Only validated
connector_setup metadata can select the trusted input path.
"""
import re

_SECRET = r"(?:密钥|令牌|凭据|密码|访问码|\b(?:PAT|token|password|secret|API[ _-]?key|access[ _-]?code)\b)"
_REQUEST = re.compile(
    r"(?:输入|填写|填入|粘贴|提供|提交|发送|贴上).{0,48}" + _SECRET
    + r"|\b(?:enter|paste|provide|submit|send|type)\b[^.!?\n]{0,80}" + _SECRET,
    re.IGNORECASE,
)


def requests_secret_input(question: str) -> bool:
    """Conservative guard for explicit secret requests, not secret detection."""
    return bool(_REQUEST.search(question) or re.search(
        r"安全连接卡|安全卡片|受保护输入|安全输入框|protected input|secure (?:connection )?(?:card|input)",
        question, re.IGNORECASE,
    ))
