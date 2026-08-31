#!/usr/bin/env bash

set -euo pipefail

archive=${HERMES_DEPLOY_ARCHIVE:-/tmp/hermes-src.new.tgz}
hermes_bin=${HERMES_DEPLOY_HERMES_BIN:-/usr/local/bin/hermes}
service_name=${HERMES_DEPLOY_SERVICE:-zettlab-local-server}
health_url=${HERMES_DEPLOY_HEALTH_URL:-http://127.0.0.1:9090/health}
hermes_link=$(readlink -f "$hermes_bin" 2>/dev/null || true)
[[ -n "$hermes_link" && -e "$hermes_link" ]] || { echo "missing Hermes launcher: $hermes_bin" >&2; exit 1; }
app_root=$(dirname "$(dirname "$hermes_link")")
hermes_src="$app_root/lib/hermes-agent"
wrapper="$app_root/bin/hermes"
case "$hermes_src" in
  */apps/com.zettlab.claw/*/lib/hermes-agent) ;;
  *) echo "unexpected Hermes source path: $hermes_src" >&2; exit 1 ;;
esac
[[ -x "$wrapper" ]] || { echo "missing ZPK wrapper: $wrapper" >&2; exit 1; }
[[ -f "$archive" ]] || { echo "missing source archive: $archive" >&2; exit 1; }

stage=$(mktemp -d "${hermes_src}.new.XXXXXX")
# Keep the large venv beside the app source so mv remains a same-filesystem
# metadata operation. /tmp is tmpfs on the 2 GB boards and must never receive it.
keep=$(mktemp -d "${hermes_src}.runtime.XXXXXX")
rollback=$(mktemp -d "${hermes_src}.rollback.XXXXXX")
rmdir "$rollback"
activated=0
succeeded=0
cleanup() {
  if [[ "$succeeded" != "1" && "$activated" == "1" && -d "$rollback" ]]; then
    systemctl stop "$service_name" || true
    [[ -d "$hermes_src/venv" ]] && mv "$hermes_src/venv" "$keep/venv"
    [[ -f "$hermes_src/.env" ]] && mv "$hermes_src/.env" "$keep/.env"
    rm -rf "$hermes_src"
    mv "$rollback" "$hermes_src"
    [[ -d "$keep/venv" ]] && mv "$keep/venv" "$hermes_src/venv"
    [[ -f "$keep/.env" ]] && mv "$keep/.env" "$hermes_src/.env"
    echo "deployment failed; restored previous Hermes source" >&2
  fi
  rm -rf "$stage" "$keep" "$rollback" "$archive" "${plugin_output:-}" "${sync_output:-}" /tmp/hermes-src.XXXXXX.tgz /tmp/deploy-dev-direct-remote.sh
  systemctl is-active --quiet "$service_name" || systemctl start "$service_name" || true
}
trap cleanup EXIT
tar xzf "$archive" -C "$stage"
deps_changed=0
for dependency_file in pyproject.toml uv.lock; do
  if ! cmp -s "$hermes_src/$dependency_file" "$stage/$dependency_file"; then
    deps_changed=1
  fi
done

systemctl stop "$service_name"
[[ -d "$hermes_src/venv" ]] && mv "$hermes_src/venv" "$keep/venv"
[[ -f "$hermes_src/.env" ]] && mv "$hermes_src/.env" "$keep/.env"
mv "$hermes_src" "$rollback"
activated=1
mv "$stage" "$hermes_src"
[[ -d "$keep/venv" ]] && mv "$keep/venv" "$hermes_src/venv"
[[ -f "$keep/.env" ]] && mv "$keep/.env" "$hermes_src/.env"

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
export UV_PYTHON_DOWNLOADS=never
cd "$hermes_src"
if [[ "$deps_changed" == "1" ]]; then
  uv_bin=$(command -v uv || true)
  [[ -n "$uv_bin" ]] || uv_bin="$HOME/.local/bin/uv"
  [[ -x "$uv_bin" ]] || { echo "uv is required for locked dependency sync" >&2; exit 1; }
  [[ -x venv/bin/python ]] || { echo "existing Hermes venv is unavailable" >&2; exit 1; }
  sync_output=$(mktemp /tmp/hermes-sync.XXXXXX)
  if ! UV_PROJECT_ENVIRONMENT="$hermes_src/venv" "$uv_bin" sync \
    --frozen --no-dev --no-editable --no-install-project --no-build \
    --extra all --extra langfuse --extra anthropic --extra zpk-runtime \
    >"$sync_output" 2>&1; then
    tail -40 "$sync_output" >&2
    exit 1
  fi
  if ! UV_PROJECT_ENVIRONMENT="$hermes_src/venv" "$uv_bin" sync \
    --frozen --no-dev --no-editable --no-build-isolation \
    --reinstall-package hermes-agent \
    --extra all --extra langfuse --extra anthropic --extra zpk-runtime \
    >>"$sync_output" 2>&1; then
    tail -40 "$sync_output" >&2
    exit 1
  fi
  tail -20 "$sync_output"
else
  echo "dependency metadata unchanged; reusing existing venv"
fi
venv/bin/python -c 'import langfuse'

ln -sfn "$wrapper" "$hermes_bin"
echo "=== bundled plugins ==="
plugin_output=$(mktemp /tmp/hermes-plugins.XXXXXX)
"$hermes_bin" plugins list >"$plugin_output" 2>&1
head -25 "$plugin_output"
grep -q 'bundled' "$plugin_output" || { echo "no bundled plugins found" >&2; exit 1; }

systemctl start "$service_name"
for attempt in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$health_url" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    echo "health: 200 (attempt $attempt)"
    echo "deployed source: $hermes_src"
    succeeded=1
    exit 0
  fi
  sleep 2
done

systemctl status "$service_name" --no-pager | tail -20 || true
journalctl -u "$service_name" --since '2 minutes ago' --no-pager | tail -40 || true
echo "local-server health check failed" >&2
exit 1
