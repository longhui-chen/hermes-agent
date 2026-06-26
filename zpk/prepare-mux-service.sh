#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
HERMES_HOME="$APP_BASE/data/hermes_home"
SECRET_DIR="$APP_BASE/data/secrets"
KEY_FILE="$SECRET_DIR/zet_agent.key"
ENV_FILE="$SECRET_DIR/hermes-agent-mux.env"

generate_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
        return
    fi
    "$HERMES_PYTHON" - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
}

write_mux_env() {
    mkdir -p "$SECRET_DIR"
    chmod 0700 "$SECRET_DIR"

    if [ ! -s "$KEY_FILE" ]; then
        generate_key > "$KEY_FILE.tmp.$$"
        chmod 0600 "$KEY_FILE.tmp.$$"
        mv "$KEY_FILE.tmp.$$" "$KEY_FILE"
    fi

    key="$(tr -d '\r\n' < "$KEY_FILE")"
    if [ -z "$key" ]; then
        echo "empty ZET_AGENT_KEY in $KEY_FILE" >&2
        exit 1
    fi

    {
        printf 'ZET_AGENT_KEY=%s\n' "$key"
        printf 'ZET_AGENT_ENABLED=true\n'
        printf 'ZET_AGENT_HOST=127.0.0.1\n'
        printf 'ZET_AGENT_PORT=7900\n'
    } > "$ENV_FILE.tmp.$$"
    chmod 0600 "$ENV_FILE.tmp.$$"
    mv "$ENV_FILE.tmp.$$" "$ENV_FILE"
}

enable_mux_config() {
    mkdir -p "$HERMES_HOME"
    "$HERMES_PYTHON" - "$HERMES_HOME/config.yaml" <<'PY'
import os
import sys
import tempfile

path = sys.argv[1]
if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
else:
    lines = []

def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))

gateway_idx = None
for i, line in enumerate(lines):
    stripped = line.strip()
    if stripped == "gateway:" and indent_of(line) == 0:
        gateway_idx = i
        break
    if stripped in {"gateway: {}", "gateway: null"} and indent_of(line) == 0:
        lines[i] = "gateway:\n"
        gateway_idx = i
        break

if gateway_idx is None:
    if lines and lines[-1].strip():
        lines.append("\n")
    lines.extend(["gateway:\n", "  multiplex_profiles: true\n"])
else:
    gateway_end = len(lines)
    for i in range(gateway_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped and indent_of(lines[i]) == 0 and not stripped.startswith("#"):
            gateway_end = i
            break

    updated = False
    for i in range(gateway_idx + 1, gateway_end):
        stripped = lines[i].strip()
        if stripped.startswith("multiplex_profiles:") and indent_of(lines[i]) == 2:
            comment = ""
            if "#" in lines[i]:
                comment = "  #" + lines[i].split("#", 1)[1].rstrip("\n")
            lines[i] = f"  multiplex_profiles: true{comment}\n"
            updated = True
            break
    if not updated:
        lines.insert(gateway_end, "  multiplex_profiles: true\n")

directory = os.path.dirname(path) or "."
fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=directory)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.writelines(lines)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
finally:
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass
PY
}

stop_legacy_per_profile_gateways() {
    pids="$(ps -eo pid=,args= | awk -v root="$APP_BASE" 'index($0, root) && index($0, " gateway run") && index($0, " -p ") {print $1}' || true)"
    [ -n "$pids" ] || return 0

    echo "Stopping legacy per-profile Hermes gateways: $pids"
    for pid in $pids; do
        [ "$pid" = "$$" ] && continue
        kill "$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in $pids; do
        [ "$pid" = "$$" ] && continue
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

if [ ! -x "$HERMES_PYTHON" ]; then
    echo "hermes python is not ready: $HERMES_PYTHON" >&2
    exit 127
fi

write_mux_env
enable_mux_config
stop_legacy_per_profile_gateways

echo "Hermes multiplex service prepared."
echo "  env: $ENV_FILE"
echo "  HERMES_HOME: $HERMES_HOME"
