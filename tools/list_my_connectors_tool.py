"""List My Connectors Tool — read-only: which business-data connectors the
agent's user has authorized.

Connectors (gmail/notion/github/linear/...) live in zettlab-local-server's
connector state, NOT in this hermes process (per-agent gateway only runs
zet_agent). This tool calls back to local-server's loopback internal endpoint,
which resolves the agent identity from the action token and returns a narrow,
credential-free list of providers and their connection state.
"""

import json
import urllib.request
from urllib.parse import urlsplit, urlunsplit

from agent.secret_scope import get_secret

_ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
_CONNECTORS_PATH = "/api/v1/internal/agent/connectors"

LIST_MY_CONNECTORS_SCHEMA = {
    "name": "list_my_connectors",
    "description": (
        "List the business-data connectors (e.g. Gmail, Notion, GitHub, "
        "Linear) this agent's user has authorized or not yet authorized, with "
        "each connector's connection state. Read-only. Use when the user asks "
        "which data sources are connected / linked, or when deciding whether "
        "to recommend connecting a data source. Returns provider and state — "
        "no credentials, no management details. NOTE: a connector being "
        "connected does NOT mean you can immediately call that connector's "
        "tools in this session; do not promise connector-backed actions based "
        "on this list alone."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}


def _resolve_connectors_url():
    """Derive the local-server connectors endpoint from ZET_CHAT_APPEND_URL.

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
    return urlunsplit((parts.scheme, parts.netloc, _CONNECTORS_PATH, "", ""))


def _check_list_my_connectors():
    """Expose the tool only inside a zet_agent subprocess: both the callback URL
    and the action token must be present."""
    return bool(
        str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
        and str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    )


# Availability depends on the per-turn profile scope; the registry must not
# serve one profile's cached verdict to another.
_check_list_my_connectors._profile_scope_sensitive = True  # type: ignore[attr-defined]


def list_my_connectors_tool(args, **kw):
    # Tool handlers must return a STRING (json-encoded for structured data) —
    # same contract as send_message/clarify. run_agent's non-multimodal path
    # passes the return value straight into the tool message content via
    # maybe_persist_tool_result(content: str), so a raw dict reaches the model
    # provider as non-string content and gets rejected (deepseek-v4/MMGPT → 400).
    url = _resolve_connectors_url()
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
        return json.dumps({"error": f"failed to fetch connectors from local-server: {e}"}, ensure_ascii=False)

    # local-server envelope is {code, data:{connectors:[{provider, state, ...}]}}
    data = parsed.get("data") if isinstance(parsed, dict) else None
    if isinstance(data, dict) and "connectors" in data:
        return json.dumps({"connectors": data["connectors"]}, ensure_ascii=False)
    return json.dumps({"error": f"unexpected response from local-server: {body[:200]}"}, ensure_ascii=False)


from tools.registry import registry  # noqa: E402

registry.register(
    name="list_my_connectors",
    toolset="zettlab_connectors",
    schema=LIST_MY_CONNECTORS_SCHEMA,
    handler=list_my_connectors_tool,
    check_fn=_check_list_my_connectors,
    emoji="🔗",
)
