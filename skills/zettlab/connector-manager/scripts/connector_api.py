#!/usr/bin/env python3
"""
Zettlab Connector Management CLI

Usage:
  connector_api.py list
  connector_api.py authorize <provider>
  connector_api.py revoke <connection_id>
  connector_api.py agent-policies
  connector_api.py set-policy <provider> <enabled>

Reads from environment:
  ZETTLAB_SERVER_URL   - e.g. https://api.zettlab.com
  ZETTLAB_USER_TOKEN   - Bearer token from login
  ZETTLAB_AGENT_ID     - Agent ID for policy operations
"""

import json
import os
import sys
import urllib.request
import urllib.error


def _env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"ERROR: {name} is not set. Add it to ~/.hermes/.env and restart the gateway.", file=sys.stderr)
        sys.exit(1)
    return value


def _request(method: str, path: str, body: dict | None = None) -> dict:
    server = _env("ZETTLAB_SERVER_URL").rstrip("/")
    token = _env("ZETTLAB_USER_TOKEN")
    url = f"{server}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/json")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        print(f"ERROR: HTTP {e.code} from {url}: {body_text}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach {url}: {e.reason}", file=sys.stderr)
        sys.exit(1)


def cmd_list() -> None:
    result = _request("GET", "/v1/api/connectors")
    providers = result.get("data", {}).get("providers", [])
    if not providers:
        print("No connectors configured.")
        return
    for p in sorted(providers, key=lambda x: x.get("sort_order", 99)):
        name = p["name"]
        connections = p.get("connections", [])
        if not connections:
            print(f"{name:<12} ✗ not connected")
        else:
            for conn in connections:
                alias = conn.get("account_alias", "")
                status = conn.get("status", "unknown")
                conn_id = conn.get("id", "")
                mark = "✓" if status == "active" else "!"
                print(f"{name:<12} {mark} {alias:<24} {status}  (id: {conn_id})")


def cmd_authorize(provider: str) -> None:
    result = _request("POST", f"/v1/api/connectors/{provider}/authorize", {
        "client_type": "web",
    })
    data = result.get("data", {})
    authorize_url = data.get("authorize_url", "")
    expires_at = data.get("expires_at", "")
    if not authorize_url:
        print("ERROR: No authorize_url in response", file=sys.stderr)
        sys.exit(1)
    print(authorize_url)
    if expires_at:
        print(f"(expires: {expires_at})", file=sys.stderr)


def cmd_revoke(connection_id: str) -> None:
    _request("DELETE", f"/v1/api/connectors/me/{connection_id}")
    print(f"Revoked connection {connection_id}")


def cmd_agent_policies() -> None:
    agent_id = _env("ZETTLAB_AGENT_ID")
    result = _request("GET", f"/v1/api/connectors/policies/{agent_id}")
    policies = result.get("data", {}).get("policies", [])
    if not policies:
        print(f"No connector policies for agent {agent_id}.")
        print("(New agents start with all connectors disabled by default.)")
        return
    for policy in policies:
        provider = policy.get("provider_id", "?")
        enabled = policy.get("enabled", False)
        alias = policy.get("default_account_alias", "")
        mark = "✓ enabled" if enabled else "✗ disabled"
        suffix = f"  (default: {alias})" if alias else ""
        print(f"{provider:<12} {mark}{suffix}")


def cmd_set_policy(provider: str, enabled_str: str) -> None:
    agent_id = _env("ZETTLAB_AGENT_ID")
    if enabled_str.lower() not in ("true", "false", "1", "0", "yes", "no"):
        print(f"ERROR: enabled must be true or false, got: {enabled_str}", file=sys.stderr)
        sys.exit(1)
    enabled = enabled_str.lower() in ("true", "1", "yes")
    _request("PUT", f"/v1/api/connectors/policies/{agent_id}/{provider}", {"enabled": enabled})
    action = "enabled" if enabled else "disabled"
    print(f"{provider} {action} for agent {agent_id}")


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)
    cmd = args[0]
    if cmd == "list":
        cmd_list()
    elif cmd == "authorize":
        if len(args) < 2:
            print("Usage: connector_api.py authorize <provider>", file=sys.stderr)
            sys.exit(1)
        cmd_authorize(args[1])
    elif cmd == "revoke":
        if len(args) < 2:
            print("Usage: connector_api.py revoke <connection_id>", file=sys.stderr)
            sys.exit(1)
        cmd_revoke(args[1])
    elif cmd == "agent-policies":
        cmd_agent_policies()
    elif cmd == "set-policy":
        if len(args) < 3:
            print("Usage: connector_api.py set-policy <provider> <true|false>", file=sys.stderr)
            sys.exit(1)
        cmd_set_policy(args[1], args[2])
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        print("Commands: list, authorize, revoke, agent-policies, set-policy", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
