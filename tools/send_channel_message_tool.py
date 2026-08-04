"""Send Channel Message Tool — LLM-initiated outbound to a bound IM channel.

IM channels live in zettlab-local-server, not this hermes process. This tool
POSTs to local-server's loopback send endpoint, which resolves the agent from the
action token, validates target_ref ownership, enforces a verified-owner gate, and
sends. The LLM passes only a target_ref (from list_my_channels) + text — never a
channel_id, agent_id, or recipient. Returns a JSON string.
"""

import json
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import get_secret

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_SEND_PATH = "/api/v1/internal/agent/channels/send"

SEND_CHANNEL_MESSAGE_SCHEMA = {
    "name": "send_channel_message",
    "description": (
        "Send a message to one of THIS agent's connected IM channels (e.g. "
        "WeChat, Feishu). Call list_my_channels first to get a valid target_ref. "
        "The message goes only to the channel's verified owner (the recent "
        "contact); you cannot choose an arbitrary recipient. Returns message_id "
        "on success, or an error (e.g. no recent conversation / not a verified "
        "owner). Do not claim a message was sent unless you get a message_id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "target_ref": {
                "type": "string",
                "description": "The target_ref from list_my_channels, e.g. 'channel:wechat'. Do not invent one.",
            },
            "text": {"type": "string", "description": "The message text to send."},
        },
        "required": ["target_ref", "text"],
    },
}


def _resolve_send_url():
    # Resolved via the profile secret scope (get_secret) so the shared
    # multiplexing gateway — where the value lives in the profile ``.env``,
    # not the process env — works too.
    raw = str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
    if not raw:
        return None
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return None
    return urlunsplit((parts.scheme, parts.netloc, _SEND_PATH, "", ""))


def _check_send_channel_message():
    return bool(
        str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    )


# Availability depends on the per-turn profile scope; the registry must not
# serve one profile's cached verdict to another.
_check_send_channel_message._profile_scope_sensitive = True  # type: ignore[attr-defined]


def _post_channel_send(url, token, target_ref, text):
    """POST one (already-chunked) text. Returns (message_id, None) on success or
    (None, error_str) on failure."""
    payload = json.dumps({"target_ref": target_ref, "text": text, "source": "llm"}).encode("utf-8")
    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={_ACTION_TOKEN_HEADER: token, "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
    except Exception as e:
        return None, f"failed to send via local-server: {e}"
    if isinstance(parsed, dict) and parsed.get("code") == 200:
        return (parsed.get("data") or {}).get("message_id"), None
    detail = ""
    if isinstance(parsed, dict) and isinstance(parsed.get("data"), dict):
        detail = str(parsed["data"].get("detail", ""))
    return None, f"send failed: {detail or body[:200]}"


def send_channel_message_tool(args, **kw):
    args = args or {}
    target_ref = str(args.get("target_ref", "")).strip()
    text = str(args.get("text", "")).strip()
    if not target_ref or not text:
        return json.dumps({"error": "both target_ref and text are required"}, ensure_ascii=False)
    url = _resolve_send_url()
    if not url:
        return json.dumps({"error": "local-server callback URL unavailable (ZET_CHAT_APPEND_URL unset)"}, ensure_ascii=False)
    token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    if not token:
        return json.dumps({"error": "agent action token unavailable"}, ensure_ascii=False)

    # Chunk long text under local-server's per-message rune cap (mirrors native
    # send_message delivery); each chunk is a separate send.
    from tools.channel_text import chunk_channel_text
    chunks = chunk_channel_text(text)

    message_ids = []
    for idx, chunk in enumerate(chunks):
        mid, err = _post_channel_send(url, token, target_ref, chunk)
        if err:
            label = f" (part {idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            return json.dumps({"error": f"{err}{label}"}, ensure_ascii=False)
        message_ids.append(mid)

    if len(message_ids) > 1:
        return json.dumps({"message_id": message_ids[-1], "parts": len(message_ids)}, ensure_ascii=False)
    return json.dumps({"message_id": message_ids[0] if message_ids else None}, ensure_ascii=False)


from tools.registry import registry  # noqa: E402

registry.register(
    name="send_channel_message",
    toolset="zettlab_channels",
    schema=SEND_CHANNEL_MESSAGE_SCHEMA,
    handler=send_channel_message_tool,
    check_fn=_check_send_channel_message,
    emoji="📤",
)
