#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
HERMES_BIN="$APP_ROOT/bin/hermes"
HERMES_HOME="$APP_BASE/data/hermes_home"
DATA_DIR="$APP_BASE/data"
SECRET_DIR="$APP_BASE/data/secrets"
OTA_DATA_TARGET="$(dirname "$(dirname "$APP_BASE")")/data/$(basename "$APP_BASE")"
VOLUME_DATA_TARGET="/volume1/subvol/apps/$(basename "$APP_BASE")/data"
LOCK_FILE="$SECRET_DIR/prepare-claw-service.lock"
KEY_FILE="$SECRET_DIR/zet_agent.key"
ENV_FILE="$SECRET_DIR/zettlab-claw.env"
DEFAULT_ZETTLAB_PRESETS_DIR="/volume1/subvol/agents/zettlab-presets/current"
LEGACY_ZETTLAB_PRESETS_DIR="/volume1/agents/zettlab-presets/current"

env_file_value() {
    local wanted="$1" name value
    while IFS= read -r -d '' name && IFS= read -r -d '' value; do
        [ -n "$name" ] || break
        [ "$name" = "$wanted" ] || continue
        printf '%s\n' "$value"
        return 0
    done < <(dump_persisted_env)
    return 1
}

presets_dir_is_trusted() {
    local candidate="$1" resolved current uid mode group other
    [ -n "$candidate" ] || return 1
    resolved="$(readlink -f "$candidate" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -d "$resolved" ] || return 1

    # zettlab-claw normally runs as root. Require the resolved version directory
    # to be root-owned and not group/world-writable before exposing it to Hermes.
    # Local non-root test/dev runs retain the write-bit rejection but cannot
    # assert device ownership.
    if [ "$(id -u)" -eq 0 ]; then
        current="$resolved"
        while :; do
            uid="$(stat -c '%u' "$current" 2>/dev/null || stat -f '%u' "$current" 2>/dev/null || true)"
            mode="$(stat -c '%a' "$current" 2>/dev/null || stat -f '%Lp' "$current" 2>/dev/null || true)"
            [ "$uid" = "0" ] && [ -n "$mode" ] || return 1
            group="${mode: -2:1}"
            other="${mode: -1}"
            (( (10#$group & 2) == 0 )) || return 1
            (( (10#$other & 2) == 0 )) || return 1
            [ "$current" = "/" ] && break
            current="$(dirname "$current")"
        done
    else
        mode="$(stat -c '%a' "$resolved" 2>/dev/null || stat -f '%Lp' "$resolved" 2>/dev/null || true)"
        [ -n "$mode" ] || return 1
        group="${mode: -2:1}"
        other="${mode: -1}"
        (( (10#$group & 2) == 0 )) || return 1
        (( (10#$other & 2) == 0 )) || return 1
    fi
    printf '%s\n' "$resolved"
}

detect_zettlab_presets_dir() {
    local existing candidate trusted
    existing="$(env_file_value ZETTLAB_PRESETS_DIR || true)"
    for candidate in "${ZETTLAB_PRESETS_DIR:-}" "$existing" "$DEFAULT_ZETTLAB_PRESETS_DIR" "$LEGACY_ZETTLAB_PRESETS_DIR"; do
        if trusted="$(presets_dir_is_trusted "$candidate")"; then
            printf '%s\n' "$trusted"
            return 0
        fi
    done
    return 1
}

trusted_data_symlink_target() {
    local resolved candidate expected="" uid mode group other
    resolved="$(readlink -f "$DATA_DIR" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -d "$resolved" ] || return 1
    for candidate in "$OTA_DATA_TARGET" "$VOLUME_DATA_TARGET"; do
        candidate="$(readlink -f "$candidate" 2>/dev/null || true)"
        if [ -n "$candidate" ] && [ "$resolved" = "$candidate" ]; then
            expected="$candidate"
            break
        fi
    done
    [ -n "$expected" ] || return 1

    uid="$(stat -c '%u' "$resolved" 2>/dev/null || stat -f '%u' "$resolved" 2>/dev/null || true)"
    mode="$(stat -c '%a' "$resolved" 2>/dev/null || stat -f '%Lp' "$resolved" 2>/dev/null || true)"
    [ "$uid" = "$(id -u)" ] && [ -n "$mode" ] || return 1
    group="${mode: -2:1}"
    other="${mode: -1}"
    (( (10#$group & 2) == 0 )) || return 1
    (( (10#$other & 2) == 0 )) || return 1
    printf '%s\n' "$resolved"
}

secure_state_directories() {
    local path resolved_path mode expected_mode uid
    for path in "$DATA_DIR" "$SECRET_DIR"; do
        if [ "$path" = "$DATA_DIR" ] && [ -L "$path" ]; then
            resolved_path="$(trusted_data_symlink_target || true)"
            if [ -z "$resolved_path" ]; then
                echo "refusing untrusted data symlink: $path" >&2
                exit 1
            fi
        elif [ -L "$path" ] || { [ -e "$path" ] && [ ! -d "$path" ]; }; then
            echo "refusing non-directory state path: $path" >&2
            exit 1
        else
            mkdir -p "$path"
            if [ -L "$path" ] || [ ! -d "$path" ]; then
                echo "state path changed while preparing it: $path" >&2
                exit 1
            fi
            resolved_path="$path"
        fi
        if [ "$(id -u)" -eq 0 ]; then
            uid="$(stat -c '%u' "$resolved_path" 2>/dev/null || stat -f '%u' "$resolved_path" 2>/dev/null || true)"
            if [ "$uid" != "0" ]; then
                echo "refusing non-root-owned state directory: $path" >&2
                exit 1
            fi
        fi
        if [ "$path" = "$SECRET_DIR" ]; then
            expected_mode=0700
        else
            mode="$(stat -c '%a' "$resolved_path" 2>/dev/null || stat -f '%Lp' "$resolved_path" 2>/dev/null || true)"
            [ -n "$mode" ] || {
                echo "cannot verify state directory mode: $path" >&2
                exit 1
            }
            printf -v expected_mode '%04o' "$(( (8#$mode | 0700) & 0755 ))"
        fi
        chmod "$expected_mode" "$resolved_path"
        mode="$(stat -c '%a' "$resolved_path" 2>/dev/null || stat -f '%Lp' "$resolved_path" 2>/dev/null || true)"
        [ -n "$mode" ] || {
            echo "cannot verify state directory mode: $path" >&2
            exit 1
        }
    done
}

acquire_prepare_lock() {
    if [ -L "$LOCK_FILE" ] || { [ -e "$LOCK_FILE" ] && [ ! -f "$LOCK_FILE" ]; }; then
        echo "refusing non-regular prepare lock file: $LOCK_FILE" >&2
        exit 1
    fi
    exec 9> "$LOCK_FILE"
    chmod 0600 "$LOCK_FILE"
    HERMES_PREPARE_LOCK_FD=9 "$HERMES_PYTHON" - <<'PY'
import fcntl
import os

fd = int(os.environ["HERMES_PREPARE_LOCK_FD"])
fcntl.flock(fd, fcntl.LOCK_EX)
PY
}

dump_persisted_env() {
    if [ ! -e "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ]; then
        printf '\000\000'
        return 0
    fi
    if [ -L "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
        echo "refusing non-regular environment file: $ENV_FILE" >&2
        return 1
    fi

    "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" "$ENV_FILE"
}

load_persisted_user_env() {
    local key value complete=0
    while IFS= read -r -d '' key && IFS= read -r -d '' value; do
        if [ -z "$key" ]; then
            complete=1
            break
        fi
        case "$key" in
            ZET_AGENT_KEY|ZET_AGENT_ENABLED|ZET_AGENT_HOST|ZET_AGENT_PORT|ZETTLAB_PRESETS_DIR)
                continue
                ;;
        esac
        if [ "${!key+x}" != "x" ]; then
            export "$key=$value"
        fi
    done < <(dump_persisted_env)
    if [ "$complete" != "1" ]; then
        echo "failed to parse environment file: $ENV_FILE" >&2
        exit 1
    fi
}

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

write_agent_env() {
    if [ -L "$KEY_FILE" ] || { [ -e "$KEY_FILE" ] && [ ! -f "$KEY_FILE" ]; }; then
        echo "refusing non-regular ZET_AGENT_KEY file: $KEY_FILE" >&2
        exit 1
    fi
    if [ -L "$ENV_FILE" ] || { [ -e "$ENV_FILE" ] && [ ! -f "$ENV_FILE" ]; }; then
        echo "refusing non-regular environment file: $ENV_FILE" >&2
        exit 1
    fi

    if [ ! -s "$KEY_FILE" ]; then
        generate_key > "$KEY_FILE.tmp.$$"
        chmod 0600 "$KEY_FILE.tmp.$$"
        mv "$KEY_FILE.tmp.$$" "$KEY_FILE"
    fi
    chmod 0600 "$KEY_FILE"

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
        if [ -n "$ZETTLAB_PRESETS_DIR" ]; then
            printf 'ZETTLAB_PRESETS_DIR=%s\n' "$ZETTLAB_PRESETS_DIR"
        fi

        # Preserve user-managed EnvironmentFile entries verbatim while replacing
        # complete logical assignments owned by this package. Physical-line
        # filtering is unsafe because systemd permits multiline quoted values.
        # Generated assignments come first so an accepted EOF-unclosed user value
        # cannot swallow package-owned fields appended after it.
        if [ -f "$ENV_FILE" ]; then
            "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" \
                --filter-excluding "$ENV_FILE" \
                ZET_AGENT_KEY ZET_AGENT_ENABLED ZET_AGENT_HOST \
                ZET_AGENT_PORT ZETTLAB_PRESETS_DIR
        fi
    } > "$ENV_FILE.tmp.$$"
    if ! "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" \
        "$ENV_FILE.tmp.$$" >/dev/null; then
        rm -f "$ENV_FILE.tmp.$$"
        echo "refusing invalid generated environment file" >&2
        return 1
    fi
    chmod 0600 "$ENV_FILE.tmp.$$"
    if [ -f "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ] \
        && cmp -s "$ENV_FILE.tmp.$$" "$ENV_FILE"; then
        chmod 0600 "$ENV_FILE"
        rm -f "$ENV_FILE.tmp.$$"
    else
        mv "$ENV_FILE.tmp.$$" "$ENV_FILE"
    fi
}

required_multiplex_config_key() {
    HERMES_HOME="$HERMES_HOME" "$HERMES_PYTHON" - "$HERMES_HOME/config.yaml" <<'PY'
import os
import json
import sys

from gateway.config import GatewayConfig
from utils import fast_safe_load

path = sys.argv[1]
config_exists = os.path.exists(path)

try:
    if config_exists:
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        config = fast_safe_load(source)
    else:
        config = {}
except Exception as exc:
    print(f"invalid Hermes config {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

if config is None:
    config = {}
if not isinstance(config, dict):
    print(f"invalid Hermes config {path}: expected a mapping", file=sys.stderr)
    raise SystemExit(1)

gateway_defaults = {}
gateway_json_path = os.path.join(os.path.dirname(path), "gateway.json")
try:
    with open(gateway_json_path, "r", encoding="utf-8") as f:
        gateway_defaults = json.load(f) or {}
except FileNotFoundError:
    pass
except Exception:
    # Match gateway startup: malformed legacy gateway.json is a warning/fallback,
    # while malformed primary config.yaml above fails closed.
    gateway_defaults = {}
if not isinstance(gateway_defaults, dict):
    gateway_defaults = {}

# Runtime currently applies the managed overlay while loading an existing
# config.yaml. Match that behavior exactly rather than making prepare's view
# broader than the gateway's.
effective_config = config
managed_dir_override = os.environ.get("HERMES_MANAGED_DIR", "").strip()
managed_scope_present = (
    os.path.isdir(managed_dir_override)
    if managed_dir_override
    else os.path.isdir("/etc/hermes")
)
if config_exists and managed_scope_present:
    from hermes_cli import managed_scope

    effective_config = managed_scope.apply_managed_overlay(dict(config))
gateway_data = dict(gateway_defaults)
nested_gateway = effective_config.get("gateway")
if (
    isinstance(nested_gateway, dict)
    and "multiplex_profiles" in nested_gateway
):
    gateway_data["multiplex_profiles"] = nested_gateway["multiplex_profiles"]
if "multiplex_profiles" in effective_config:
    gateway_data["multiplex_profiles"] = effective_config["multiplex_profiles"]

if GatewayConfig.from_dict(gateway_data).multiplex_profiles:
    raise SystemExit(0)

# The legacy top-level key has runtime precedence over gateway.*. Update that
# key when present; otherwise use the canonical nested form.
if "multiplex_profiles" in effective_config:
    print("multiplex_profiles")
else:
    print("gateway.multiplex_profiles")
PY
}

enable_agent_gateway_config() {
    mkdir -p "$HERMES_HOME"

    config_key="$(required_multiplex_config_key)"
    if [ -z "$config_key" ]; then
        return 0
    fi

    HERMES_HOME="$HERMES_HOME" "$HERMES_BIN" \
        config set "$config_key" true

    remaining_key="$(required_multiplex_config_key)"
    if [ -n "$remaining_key" ]; then
        echo "failed to enable Hermes $remaining_key" >&2
        return 1
    fi
}

stop_legacy_per_profile_gateways() {
    pids="$(ps -eo pid=,args= | awk -v root="$APP_BASE" 'index($0, root) && index($0, " gateway run") && index($0, " -p ") {print $1}' || true)"
    [ -n "$pids" ] || return 0

    echo "Stopping legacy per-profile gateway processes: $pids"
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
if [ "${1:-}" = "--dump-env" ]; then
    dump_persisted_env
    exit
fi
if [ ! -x "$HERMES_BIN" ]; then
    echo "hermes CLI is not ready: $HERMES_BIN" >&2
    exit 127
fi

secure_state_directories
acquire_prepare_lock
load_persisted_user_env
ZETTLAB_PRESETS_DIR="$(detect_zettlab_presets_dir || true)"
ZETTLAB_PRESETS_DIR="$(printf '%s' "$ZETTLAB_PRESETS_DIR" | tr -d '\r\n')"
write_agent_env
enable_agent_gateway_config
if [ "${HERMES_STOP_LEGACY_GATEWAYS:-0}" = "1" ]; then
    stop_legacy_per_profile_gateways
else
    echo "Legacy per-profile gateways left running; cleanup is deferred until the claw service is healthy and local-server is ready."
fi

echo "Zettlab Claw service prepared."
echo "  env: $ENV_FILE"
echo "  HERMES_HOME: $HERMES_HOME"
