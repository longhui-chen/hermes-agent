#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
ENV_FILE="$APP_BASE/data/secrets/zettlab-claw.env"

# Reconcile the managed env on every systemd start. EnvironmentFile is parsed
# before ExecStart, so explicitly reload only safe KEY=VALUE lines afterwards.
"$APP_ROOT/prepare-claw-service.sh"
unset ZETTLAB_PRESETS_DIR
if [ -f "$ENV_FILE" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in
            [A-Za-z_]*=*)
                key="${line%%=*}"
                case "$key" in
                    *[!A-Za-z0-9_]*|'') continue ;;
                esac
                export "$line"
                ;;
        esac
    done < "$ENV_FILE"
fi

exec "$APP_ROOT/bin/hermes" gateway run --force --accept-hooks
