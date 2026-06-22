#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_BIN="$HERMES_SRC/venv/bin/hermes"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
HERMES_LINK="/usr/local/bin/hermes"

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

echo "Install complete."
echo "  hermes: $APP_BASE/current/bin/hermes"
echo "  HERMES_HOME: $APP_BASE/data/hermes_home"
