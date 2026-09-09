#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_BIN="$HERMES_SRC/venv/bin/hermes"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
HERMES_LINK="/usr/local/bin/hermes"
DATA_DIR="/volume1/system/zettos-main-data/com.zettlab.claw"

source "$APP_ROOT/zpk-systemd.sh"

# PyPI 镜像是辅助提速功能：其脚本缺失/损坏不应阻断 hermes 核心安装（HR2），
# 故容错 source；下面 setup_pypi_mirror 未定义时也会被 `|| true` 优雅跳过。
# shellcheck source=zpk/pypi-mirror.sh
source "$APP_ROOT/pypi-mirror.sh" 2>/dev/null || true

LEGACY_GATEWAY_SERVICE_REMOVED=false

cleanup_legacy_systemd_services() {
    if ! command -v systemctl >/dev/null 2>&1; then
        return 0
    fi

    # Legacy name from the first shared-gateway ZPK cut. Remove it before
    # enabling the device-facing zettlab-claw unit, otherwise the old unit can
    # keep binding the gateway port after upgrade.
    local legacy_service="hermes-agent-mux.service"
    local systemd_dir="${SYSTEMD_DIR:-/lib/systemd/system}"
    local found=false

    if systemctl cat "$legacy_service" >/dev/null 2>&1 \
        || [ -e "$systemd_dir/$legacy_service" ] \
        || [ -e "/etc/systemd/system/$legacy_service" ]; then
        found=true
    fi

    if [ "$found" = "false" ]; then
        return 0
    fi

    echo "Removing legacy gateway service $legacy_service ..."
    systemctl stop "$legacy_service" 2>/dev/null || true
    systemctl disable "$legacy_service" 2>/dev/null || true
    rm -f "$systemd_dir/$legacy_service" "/etc/systemd/system/$legacy_service"
    rm -rf "/etc/systemd/system/$legacy_service.d"
    rm -f "$DATA_DIR/secrets/hermes-agent-mux.env" "$APP_BASE/data/secrets/hermes-agent-mux.env"
    systemctl daemon-reload 2>/dev/null || true
    LEGACY_GATEWAY_SERVICE_REMOVED=true
}

start_replacement_service_after_legacy_cleanup() {
    if [ "$LEGACY_GATEWAY_SERVICE_REMOVED" != "true" ]; then
        return 0
    fi
    if ! command -v systemctl >/dev/null 2>&1; then
        return 0
    fi
    if ! systemctl cat zettlab-claw.service >/dev/null 2>&1; then
        echo "warning: zettlab-claw.service is not installed after legacy cleanup" >&2
        return 1
    fi
    systemctl start zettlab-claw.service
    echo "zettlab-claw.service started after legacy service cleanup."
}

echo "Installing zettlab-claw from $APP_ROOT ..."

if [ ! -f "$HERMES_SRC/pyproject.toml" ]; then
    echo "missing zettlab-claw source: $HERMES_SRC" >&2
    exit 1
fi

if [ ! -x "$HERMES_BIN" ]; then
    echo "prebuilt hermes binary is missing or not executable: $HERMES_BIN" >&2
    exit 1
fi

if [ ! -x "$HERMES_PYTHON" ]; then
    echo "prebuilt hermes python is missing or not executable: $HERMES_PYTHON" >&2
    exit 1
fi

"$APP_ROOT/prepare-claw-service.sh"
"$APP_ROOT/bin/hermes" --version

# The version probe imports Python modules and can create __pycache__ using
# the installer's inherited umask. Normalize after the probe so the installed
# venv starts from the same deterministic permission contract as the ZPK.
# Managed connector workers drop to isolated UIDs, so the root remains
# traversable without allowing those workers to list it.
chmod 0711 "$HERMES_SRC/venv"
find "$HERMES_SRC/venv" -mindepth 1 -type d -exec chmod 0755 {} +
find "$HERMES_SRC/venv" -type f -perm /0111 -exec chmod 0755 {} +
find "$HERMES_SRC/venv" -type f ! -perm /0111 -exec chmod 0644 {} +
if [ -f "$HERMES_SRC/venv/.lock" ]; then
    chown root:root "$HERMES_SRC/venv/.lock"
    chmod 0600 "$HERMES_SRC/venv/.lock"
fi

mkdir -p "$(dirname "$HERMES_LINK")"
ln -sfn "$APP_BASE/current/bin/hermes" "$HERMES_LINK"

# 探测并写 PyPI 镜像源（境内 lazy-install 提速）；失败不阻断安装
setup_pypi_mirror || true

install_systemd_services "$APP_ROOT"
cleanup_legacy_systemd_services
start_replacement_service_after_legacy_cleanup

echo "Install complete."
echo "  hermes: $APP_BASE/current/bin/hermes"
echo "  HERMES_HOME: $DATA_DIR/hermes_home"
