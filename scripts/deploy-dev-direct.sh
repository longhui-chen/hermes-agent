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
if command -v sshpass >/dev/null 2>&1; then
  transport="sshpass"
elif command -v expect >/dev/null 2>&1; then
  transport="expect"
else
  echo "sshpass or expect is required" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
git -C "$repo_root" rev-parse --verify "${ref}^{commit}" >/dev/null
archive=$(mktemp "${TMPDIR:-/tmp}/hermes-src.XXXXXX.tgz")
trap 'rm -f "$archive"' EXIT
git -C "$repo_root" archive --format=tar.gz -o "$archive" "$ref"
remote_helper="$repo_root/scripts/deploy-dev-direct-remote.sh"
[[ -x "$remote_helper" ]] || { echo "missing remote helper: $remote_helper" >&2; exit 1; }

if [[ "$transport" == "sshpass" ]]; then
  export SSHPASS="$BOARD_SSH_PASSWORD"
  sshpass -e scp -P "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "$archive" "$remote_helper" "root@$host:/tmp/"
  sshpass -e ssh -p "$port" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    "root@$host" bash /tmp/deploy-dev-direct-remote.sh
else
  export DEPLOY_HOST="$host" DEPLOY_PORT="$port" DEPLOY_ARCHIVE="$archive" DEPLOY_HELPER="$remote_helper"
  expect <<'EXPECT_SCP'
set timeout 120
spawn scp -P $env(DEPLOY_PORT) -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  $env(DEPLOY_ARCHIVE) $env(DEPLOY_HELPER) root@$env(DEPLOY_HOST):/tmp/
expect {
  -re "(?i)password:" { send -- "$env(BOARD_SSH_PASSWORD)\r"; exp_continue }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_SCP
  expect <<'EXPECT_SSH'
set timeout 900
spawn ssh -p $env(DEPLOY_PORT) -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
  root@$env(DEPLOY_HOST) bash /tmp/deploy-dev-direct-remote.sh
expect {
  -re "(?i)password:" { send -- "$env(BOARD_SSH_PASSWORD)\r"; exp_continue }
  eof
}
catch wait result
exit [lindex $result 3]
EXPECT_SSH
fi
