"""Read meeting summaries already stored by the local Zettlab device."""
import json, os, urllib.request, urllib.parse
from urllib.parse import urlsplit, urlunsplit
from tools.registry import registry

_TOKEN = "X-Zettlab-Agent-Action-Token"
_LIST = "/api/v1/internal/meetings"
_GET = "/api/v1/internal/meetings/"
SCHEMA = {"name":"device_meetings", "description":"Read stored device meetings. Use action list or get; meeting_id must come from list.", "parameters":{"type":"object","properties":{"action":{"type":"string","enum":["list","get"]},"meeting_id":{"type":"string"},"limit":{"type":"integer","minimum":1,"maximum":20},"offset":{"type":"integer","minimum":0}},"required":["action"],"additionalProperties":False}}

def _base():
    raw = os.environ.get("ZET_CHAT_APPEND_URL", "").strip(); p = urlsplit(raw)
    return urlunsplit((p.scheme,p.netloc,"","", "")) if p.scheme and p.netloc else None

def _available():
    return bool(_base() and os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "").strip())

def device_meetings_tool(args, **_kw):
    args = args or {}; action = str(args.get("action", "")).strip(); base = _base(); token = os.environ.get("ZETTLAB_AGENT_ACTION_TOKEN", "").strip()
    if not base or not token: return json.dumps({"error":"device meeting bridge unavailable"}, ensure_ascii=False)
    if action == "list":
        path = _LIST + "?limit=" + str(max(1,min(20,int(args.get("limit",20))))) + "&offset=" + str(max(0,int(args.get("offset",0))))
    elif action == "get" and str(args.get("meeting_id", "")).strip():
        path = _GET + urllib.parse.quote(str(args["meeting_id"]).strip(), safe="")
    else: return json.dumps({"error":"action requires list or get with meeting_id"}, ensure_ascii=False)
    try:
        req = urllib.request.Request(base + path, headers={_TOKEN:token,"Accept":"application/json"})
        with urllib.request.urlopen(req, timeout=8) as resp: body = resp.read(512*1024).decode("utf-8")
        return body
    except Exception as exc: return json.dumps({"error":"failed to read device meetings: " + str(exc)}, ensure_ascii=False)

registry.register(name="device_meetings", toolset="zettlab_skill_runtime", schema=SCHEMA, handler=device_meetings_tool, check_fn=_available, emoji="🗒️", max_result_size_chars=60000)
