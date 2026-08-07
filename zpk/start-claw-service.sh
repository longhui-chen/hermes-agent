#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
ENV_FILE="$APP_BASE/data/secrets/zettlab-claw.env"

load_reconciled_env() {
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
    done < <("$APP_ROOT/prepare-claw-service.sh" --emit-env)
    if [ "$complete" != "1" ]; then
        echo "failed to load reconciled environment: $ENV_FILE" >&2
        exit 1
    fi
}

# Reconcile and load the shared runtime environment in one locked operation.
# Package-owned fields use the generated values; persisted user fields only
# fill values not already supplied by systemd or a manual operator environment.
load_reconciled_env

# The shared EnvironmentFile is also consumed by local-server and may contain
# legacy values for these Claw-owned paths. Reassert the current package slot
# after loading it so stale persisted values cannot redirect runtime state or
# bundled code.
export HERMES_HOME="$APP_BASE/data/hermes_home"
export HERMES_BUNDLED_SKILLS="$APP_ROOT/lib/hermes-agent/skills"
export HERMES_BUNDLED_PLUGINS="$APP_ROOT/lib/hermes-agent/plugins"
export HERMES_BUNDLED_LOCALES="$APP_ROOT/lib/hermes-agent/locales"
export HERMES_LAZY_INSTALL_TARGET="$APP_BASE/data/lazy-packages"
export HERMES_MANAGED_GATEWAY=1
export HERMES_MANAGED_CGROUP_UNIT=zettlab-claw.service
unset HERMES_MANAGED_CGROUP_ROOT

# systemd EnvironmentFile values override Environment= values regardless of
# their textual order in the unit. Reassert the package-required live override
# after removing a possible legacy assignment from the shared env file.
export GATEWAY_MULTIPLEX_PROFILES=true
unset ZETTLAB_CLAW_PRESETS_DIR

exec "$APP_ROOT/bin/hermes" gateway run --force --accept-hooks
