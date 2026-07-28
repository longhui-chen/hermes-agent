#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
ENV_FILE="$APP_BASE/data/secrets/zettlab-claw.env"

load_env_after_prepare() {
    local key value complete=0
    while IFS= read -r -d '' key && IFS= read -r -d '' value; do
        if [ -z "$key" ]; then
            complete=1
            break
        fi
        case "$key" in
            ZET_AGENT_KEY|ZET_AGENT_ENABLED|ZET_AGENT_HOST|ZET_AGENT_PORT|ZETTLAB_PRESETS_DIR)
                export "$key=$value"
                ;;
            *)
                if [ "${!key+x}" != "x" ]; then
                    export "$key=$value"
                fi
                ;;
        esac
    done < <("$APP_ROOT/prepare-claw-service.sh" --dump-env)
    if [ "$complete" != "1" ]; then
        echo "failed to load reconciled environment: $ENV_FILE" >&2
        exit 1
    fi
}

# Prepare safely loads persisted user-managed values for its own process. Then
# load the reconciled values for the gateway without replacing real
# systemd/manual values; package-owned fields always use the generated values.
"$APP_ROOT/prepare-claw-service.sh"
load_env_after_prepare
# prepare-claw-service.sh migrates this legacy override out of the persisted
# environment file. Also remove an inherited service/manual value so runtime
# uses the package-reconciled config.yaml setting.
unset GATEWAY_MULTIPLEX_PROFILES

exec "$APP_ROOT/bin/hermes" gateway run --force --accept-hooks
