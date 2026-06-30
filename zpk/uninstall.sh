#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
APP_ID="com.zettlab.claw"
KEEP_DATA=false
HERMES_LINK="/usr/local/bin/hermes"
EXPECTED_DATA_TARGET="/zettos/main/data/$APP_ID"

for arg in "$@"; do
    if [ "$arg" = "--keep-data" ]; then
        KEEP_DATA=true
    fi
done

echo "Uninstalling zettlab-claw (keep data: $KEEP_DATA) ..."

if command -v systemctl >/dev/null 2>&1; then
    systemctl stop zettlab-claw.service 2>/dev/null || true
    systemctl disable zettlab-claw.service 2>/dev/null || true
    rm -f /lib/systemd/system/zettlab-claw.service /etc/systemd/system/zettlab-claw.service
    rm -rf /etc/systemd/system/zettlab-claw.service.d
    systemctl daemon-reload 2>/dev/null || true
fi

echo "  stopping hermes processes under $APP_BASE"
PIDS=""
if command -v pgrep >/dev/null 2>&1; then
    PIDS=$(pgrep -f "$APP_BASE" 2>/dev/null || true)
else
    PIDS=$(ps -eo pid=,args= | awk -v root="$APP_BASE" 'index($0, root) {print $1}' || true)
fi

for pid in $PIDS; do
    if [ "$pid" = "$$" ] || [ "$pid" = "${PPID:-}" ]; then
        continue
    fi
    kill "$pid" 2>/dev/null || true
done

sleep 1

for pid in $PIDS; do
    if [ "$pid" = "$$" ] || [ "$pid" = "${PPID:-}" ]; then
        continue
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
done

if [ -L "$HERMES_LINK" ]; then
    LINK_TARGET=$(readlink "$HERMES_LINK" 2>/dev/null || true)
    LINK_REAL=$(readlink -f "$HERMES_LINK" 2>/dev/null || true)
    if [ "$LINK_TARGET" = "$APP_BASE/current/bin/hermes" ] || [[ "$LINK_REAL" == "$APP_BASE/"* ]]; then
        rm -f "$HERMES_LINK"
        echo "  removed $HERMES_LINK"
    else
        echo "  leaving $HERMES_LINK; it points to $LINK_TARGET"
    fi
fi

if [ "$KEEP_DATA" = "false" ]; then
    echo "  removing runtime data"
    if [ -L "$APP_BASE/data" ]; then
        DATA_TARGET=$(readlink -f "$APP_BASE/data" 2>/dev/null || true)
        case "$DATA_TARGET" in
            "$EXPECTED_DATA_TARGET") ;;
            "")
                echo "  data symlink target is missing, removing symlink only"
                rm -f "$APP_BASE/data"
                DATA_TARGET=""
                ;;
            *)
                echo "refuse to remove unexpected data target: $DATA_TARGET" >&2
                exit 1
                ;;
        esac
        if [ -n "$DATA_TARGET" ]; then
            rm -f "$APP_BASE/data"
            rm -rf "$DATA_TARGET"
        fi
    else
        rm -rf "$APP_BASE/data"
    fi
fi

echo "  removing app files"
rm -f "$APP_BASE/current" "$APP_BASE/app.meta"

shopt -s nullglob dotglob
for item in "$APP_BASE"/*; do
    name=$(basename "$item")
    if [ "$KEEP_DATA" = "true" ] && [ "$name" = "data" ]; then
        continue
    fi
    rm -rf "$item"
done
shopt -u nullglob dotglob

rmdir "$APP_BASE" 2>/dev/null || true

echo "Uninstall complete."
