#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_BIN="$HERMES_SRC/venv/bin/hermes"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
HERMES_LINK="/usr/local/bin/hermes"

source "$APP_ROOT/zpk-systemd.sh"

# PyPI 镜像是辅助提速功能：其脚本缺失/损坏不应阻断 hermes 核心安装（HR2），
# 故容错 source；下面 setup_pypi_mirror 未定义时也会被 `|| true` 优雅跳过。
# shellcheck source=zpk/pypi-mirror.sh
source "$APP_ROOT/pypi-mirror.sh" 2>/dev/null || true

echo "Installing hermes-agent from $APP_ROOT ..."

if [ ! -f "$HERMES_SRC/pyproject.toml" ]; then
    echo "missing hermes-agent source: $HERMES_SRC" >&2
    exit 1
fi

mkdir -p "$APP_BASE/data/hermes_home" "$APP_BASE/data/profiles" "$APP_BASE/data/sessions"

if [ ! -x "$HERMES_BIN" ]; then
    echo "prebuilt hermes binary is missing or not executable: $HERMES_BIN" >&2
    exit 1
fi

if [ ! -x "$HERMES_PYTHON" ]; then
    echo "prebuilt hermes python is missing or not executable: $HERMES_PYTHON" >&2
    exit 1
fi

"$APP_ROOT/bin/hermes" --version

mkdir -p "$(dirname "$HERMES_LINK")"
ln -sfn "$APP_BASE/current/bin/hermes" "$HERMES_LINK"

# 探测并写 PyPI 镜像源（境内 lazy-install 提速）；失败不阻断安装
setup_pypi_mirror || true

"$APP_ROOT/prepare-mux-service.sh"
install_systemd_services "$APP_ROOT"

echo "Install complete."
echo "  hermes: $APP_BASE/current/bin/hermes"
echo "  HERMES_HOME: $APP_BASE/data/hermes_home"
