"""List My Channels Tool — read-only: which IM channels is THIS agent connected to.

IM channels (feishu/wechat/discord/...) live in zettlab-local-server's channel
table, NOT in this hermes process (per-agent gateway only runs zet_agent). This
tool calls back to local-server's loopback internal endpoint, which resolves the
agent identity from the action token and returns a narrow, credential-free list.
"""

import json
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import get_secret

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_CHANNELS_PATH = "/api/v1/internal/agent/channels"

LIST_MY_CHANNELS_SCHEMA = {
    "name": "list_my_channels",
    "description": (
        "List the IM channels this agent is currently connected to, plus "
        "available_kinds — the channel kinds this device can still connect "
        "(already filtered by the device's region, e.g. CN devices only get "
        "feishu/wecom/wechat). Read-only. Use when the user asks which "
        "messaging channels are linked, or before suggesting any channel. "
        "IMPORTANT: a kind absent from both installed_channels and "
        "available_kinds is NOT connectable on this device (region "
        "restriction) — never suggest or offer to connect it. Returns channel "
        "kind, name, and status — no credentials, no management details. "
        "NOTE: a channel being connected does NOT mean you can proactively "
        "message it; use send_channel_message, which may fail (no recent "
        "conversation / not a verified owner). Do not promise to send before "
        "calling it."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _resolve_channels_url():
    """Derive the local-server channels endpoint from ZET_CHAT_APPEND_URL.

    ZET_CHAT_APPEND_URL is injected by the registry and points at
    <base>/api/v1/internal/chat/append. We reuse its scheme+netloc and swap the
    path. Resolved via the profile secret scope (get_secret) so the shared
    multiplexing gateway — where the value lives in the profile ``.env``, not
    the process env — works too. Returns None when absent or malformed.
    """
    raw = str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
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
        str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    )


# Availability depends on the per-turn profile scope; the registry must not
# serve one profile's cached verdict to another.
_check_list_my_channels._profile_scope_sensitive = True  # type: ignore[attr-defined]


def list_my_channels_tool(args, **kw):
    # Tool handlers must return a STRING (json-encoded for structured data) —
    # same contract as send_message/clarify. run_agent's non-multimodal path
    # passes the return value straight into the tool message content via
    # maybe_persist_tool_result(content: str), so a raw dict reaches the model
    # provider as non-string content and gets rejected (deepseek-v4/MMGPT → 400).
    url = _resolve_channels_url()
    if not url:
        return json.dumps({"error": "local-server callback URL unavailable (ZET_CHAT_APPEND_URL unset)"}, ensure_ascii=False)
    token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
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

    # local-server envelope is {code, data:{installed_channels:[...], available_kinds:[...]}}
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if isinstance(data, dict) and "installed_channels" in data:
        out: dict = {"installed_channels": data["installed_channels"]}
        if isinstance(data.get("available_kinds"), list):
            # 区域感知的可连清单（新版 local-server 才有；老版本无此字段）。
            # 必须透传：governor 库存判定与主模型口径接地都靠它约束
            # "只推荐/只谈论本区域真实可连的渠道"。
            out["available_kinds"] = data["available_kinds"]
        return json.dumps(out, ensure_ascii=False)
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
