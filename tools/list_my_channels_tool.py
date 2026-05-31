"""List My Channels Tool — read-only: which IM channels is THIS agent connected to.

IM channels (feishu/wechat/discord/...) live in zettlab-local-server's channel
table, NOT in this hermes process (per-agent gateway only runs zet_agent). This
tool calls back to local-server's loopback internal endpoint, which resolves the
agent identity from the action token and returns a narrow, credential-free list.
"""

import json
import os
import urllib.request
from urllib.parse import urlsplit, urlunsplit

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_CHANNELS_PATH = "/api/v1/internal/agent/channels"

LIST_MY_CHANNELS_SCHEMA = {
    "name": "list_my_channels",
    "description": (
        "List the IM channels (e.g. WeChat, Feishu, Discord) this agent is "
        "currently connected to. Read-only. Use when the user asks which "
        "messaging channels are linked / connected. Returns channel kind, name, "
        "and status — no credentials, no management details. NOTE: a channel "
        "being connected does NOT mean you can proactively message it; use "
        "send_channel_message, which may fail (no recent conversation / not a "
        "verified owner). Do not promise to send before calling it."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _resolve_channels_url():
    """Derive the local-server channels endpoint from ZET_CHAT_APPEND_URL.

    ZET_CHAT_APPEND_URL is injected by the registry and points at
    <base>/api/v1/internal/chat/append. We reuse its scheme+netloc and swap the
    path. Returns None when the env var is absent or malformed.
    """
    raw = os.environ.get("ZET_CHAT_APPEND_URL", "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, _CHANNELS_PATH, "", ""))


def _check_list_my_channels():
    """Expose the tool only inside a zet_agent subprocess: both the callback URL
    and the action token must be present."""
    return bool(
        os.environ.get("ZET_CHAT_APPEND_URL", "").strip()
        and os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "").strip()
    )


def list_my_channels_tool(args, **kw):
    # Tool handlers must return a STRING (json-encoded for structured data) —
    # same contract as send_message/clarify. run_agent's non-multimodal path
    # passes the return value straight into the tool message content via
    # maybe_persist_tool_result(content: str), so a raw dict reaches the model
    # provider as non-string content and gets rejected (deepseek-v4/MMGPT → 400).
    url = _resolve_channels_url()
    if not url:
        return json.dumps({"error": "local-server callback URL unavailable (ZET_CHAT_APPEND_URL unset)"}, ensure_ascii=False)
    token = os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "").strip()
    if not token:
        return json.dumps({"error": "agent action token unavailable (ZETTLAB_AGENT_ACTION_TOKEN unset)"}, ensure_ascii=False)
    try:
        req = urllib.request.Request(
            url,
            headers={_ACTION_TOKEN_HEADER: token, "Accept": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    except Exception as e:
        return json.dumps({"error": f"failed to fetch channels from local-server: {e}"}, ensure_ascii=False)

    # local-server envelope is {code, data:{installed_channels:[...]}}
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if isinstance(data, dict) and "installed_channels" in data:
        return json.dumps({"installed_channels": data["installed_channels"]}, ensure_ascii=False)
    return json.dumps({"error": f"unexpected response from local-server: {body[:200]}"}, ensure_ascii=False)


from tools.registry import registry  # noqa: E402

registry.register(
    name="list_my_channels",
    toolset="zettlab_channels",
    schema=LIST_MY_CHANNELS_SCHEMA,
    handler=list_my_channels_tool,
    check_fn=_check_list_my_channels,
    emoji="🔌",
)
