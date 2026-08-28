#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
exec "$APP_ROOT/install.sh" "$@"
