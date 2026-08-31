#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: BOARD_SSH_PASSWORD=... scripts/deploy-dev-direct.sh --host <ip-or-host> [--ref <git-ref>] [--port <port>]

Deploy the committed Hermes source directly to an OTA-ready development board.
The script preserves runtime state (venv and .env), replaces only tracked source,
keeps the ZPK wrapper, restarts local-server, and verifies plugins plus /health.
EOF
}

host=""
port="22"
ref="HEAD"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) host="${2:-}"; shift 2 ;;
    --port) port="${2:-}"; shift 2 ;;
    --ref) ref="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$host" ]] || { echo "--host is required" >&2; exit 2; }
[[ "$port" =~ ^[0-9]+$ ]] || { echo "--port must be numeric" >&2; exit 2; }
[[ -n "${BOARD_SSH_PASSWORD:-}" ]] || { echo "BOARD_SSH_PASSWORD is required" >&2; exit 2; }
command -v git >/dev/null
command -v sshpass >/dev/null

repo_root=$(git rev-parse --show-toplevel)
git -C "$repo_root" rev-parse --verify "${ref}^{commit}" >/dev/null
archive=$(mktemp "${TMPDIR:-/tmp}/hermes-src.XXXXXX.tgz")
trap 'rm -f "$archive"' EXIT
git -C "$repo_root" archive --format=tar.gz -o "$archive" "$ref"

ssh_opts=(-p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
export SSHPASS="$BOARD_SSH_PASSWORD"
sshpass -e scp -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  "$archive" "root@$host:/tmp/hermes-src.new.tgz"

sshpass -e ssh "${ssh_opts[@]}" "root@$host" bash -s <<'DEPLOY'
set -euo pipefail

hermes_link=$(readlink -f /usr/local/bin/hermes 2>/dev/null || true)
[[ -n "$hermes_link" && -e "$hermes_link" ]] || { echo "missing /usr/local/bin/hermes" >&2; exit 1; }
app_root=$(dirname "$(dirname "$hermes_link")")
hermes_src="$app_root/lib/hermes-agent"
wrapper="$app_root/bin/hermes"
case "$hermes_src" in
  */apps/com.zettlab.claw/*/lib/hermes-agent) ;;
  *) echo "unexpected Hermes source path: $hermes_src" >&2; exit 1 ;;
esac
[[ -x "$wrapper" ]] || { echo "missing ZPK wrapper: $wrapper" >&2; exit 1; }

stage=$(mktemp -d "${hermes_src}.new.XXXXXX")
keep=$(mktemp -d /tmp/hermes-runtime.XXXXXX)
cleanup() { rm -rf "$stage" "$keep" /tmp/hermes-src.new.tgz; }
trap cleanup EXIT
tar xzf /tmp/hermes-src.new.tgz -C "$stage"

systemctl stop zettlab-local-server
[[ -d "$hermes_src/venv" ]] && mv "$hermes_src/venv" "$keep/venv"
[[ -f "$hermes_src/.env" ]] && mv "$hermes_src/.env" "$keep/.env"
find "$hermes_src" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a "$stage"/. "$hermes_src"/
[[ -d "$keep/venv" ]] && mv "$keep/venv" "$hermes_src/venv"
[[ -f "$keep/.env" ]] && mv "$keep/.env" "$hermes_src/.env"

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
export UV_PYTHON_DOWNLOADS=never
cd "$hermes_src"
uv_bin=$(command -v uv || true)
[[ -n "$uv_bin" ]] || uv_bin="$HOME/.local/bin/uv"
if [[ -x "$uv_bin" && -x venv/bin/python ]]; then
  "$uv_bin" pip install --python venv/bin/python -e ".[all,langfuse]"
else
  ./init-hermes-env.sh
fi
venv/bin/python -c 'import langfuse'

ln -sfn "$wrapper" /usr/local/bin/hermes
echo "=== bundled plugins ==="
/usr/local/bin/hermes plugins list | head -25

systemctl start zettlab-local-server
for attempt in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 http://127.0.0.1:9090/health 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    echo "health: 200 (attempt $attempt)"
    echo "deployed source: $hermes_src"
    exit 0
  fi
  sleep 2
done

systemctl status zettlab-local-server --no-pager | tail -20 || true
journalctl -u zettlab-local-server --since '2 minutes ago' --no-pager | tail -40 || true
echo "local-server health check failed" >&2
exit 1
DEPLOY
