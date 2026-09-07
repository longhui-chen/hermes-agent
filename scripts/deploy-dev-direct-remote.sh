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
stage="$app_root/lib/hermes-agent.new"
rollback="$app_root/lib/hermes-agent.rollback"
wrapper="$app_root/bin/hermes"
recovery_helper=${HERMES_DEPLOY_RECOVERY_HELPER:-$app_root/bin/hermes-direct-deploy-recover}
systemd_dropin_dir=${HERMES_DEPLOY_SYSTEMD_DIR:-/etc/systemd/system/${service_name}.service.d}
lock_file=${HERMES_DEPLOY_LOCK_FILE:-$app_root/lib/.hermes-direct-deploy.lock}
case "$hermes_src" in
  */apps/com.zettlab.claw/*/lib/hermes-agent) ;;
  *) echo "unexpected Hermes source path: $hermes_src" >&2; exit 1 ;;
esac
[[ -x "$wrapper" ]] || { echo "missing ZPK wrapper: $wrapper" >&2; exit 1; }
[[ -f "$archive" ]] || { echo "missing source archive: $archive" >&2; exit 1; }

exec 9>"$lock_file"
flock -n 9 || { echo "another Hermes deployment is active" >&2; exit 1; }

activated=0
succeeded=0
cleanup() {
  if [[ "$succeeded" != "1" && "$activated" == "1" && -d "$rollback" ]]; then
    systemctl stop "$service_name" || true
    rm -rf "$hermes_src"
    mv "$rollback" "$hermes_src"
    echo "deployment failed; restored previous Hermes source and venv" >&2
  fi
  rm -rf "$stage"
  rm -f "$archive" "${plugin_output:-}" "${sync_output:-}" "${uv_installer:-}" /tmp/hermes-src.XXXXXX.tgz /tmp/deploy-dev-direct-remote.sh
  systemctl is-active --quiet "$service_name" || systemctl start "$service_name" || true
}
trap cleanup EXIT

install_recovery_helper() {
  local helper_tmp dropin_tmp
  mkdir -p "$(dirname "$recovery_helper")" "$systemd_dropin_dir"
  helper_tmp=$(mktemp "${recovery_helper}.new.XXXXXX")
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'hermes_src=%q\n' "$hermes_src"
    printf 'rollback=%q\n' "$rollback"
    echo 'if [[ ! -d "$hermes_src" && -d "$rollback" ]]; then'
    echo '  mv "$rollback" "$hermes_src"'
    echo 'fi'
    echo 'if [[ -d "$hermes_src" && -d "$rollback" ]]; then'
    echo '  rm -rf "$rollback"'
    echo 'fi'
    echo '[[ -d "$hermes_src" ]]'
  } >"$helper_tmp"
  chmod 0755 "$helper_tmp"
  mv -f "$helper_tmp" "$recovery_helper"

  dropin_tmp=$(mktemp "${systemd_dropin_dir}/95-hermes-direct-deploy-recover.conf.new.XXXXXX")
  {
    echo '[Service]'
    printf 'ExecStartPre=%s\n' "$recovery_helper"
  } >"$dropin_tmp"
  mv -f "$dropin_tmp" "$systemd_dropin_dir/95-hermes-direct-deploy-recover.conf"
  systemctl daemon-reload
}

find_uv() {
  local candidate
  for candidate in "${HERMES_DEPLOY_UV_BIN:-}" "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

bootstrap_uv() {
  local uv_version
  uv_version=${HERMES_DEPLOY_UV_VERSION:-0.12.5}
  uv_installer=$(mktemp /tmp/hermes-uv-installer.XXXXXX)
  curl --fail --location --silent --show-error \
    --proto '=https' --tlsv1.2 --max-time 60 --retry 3 \
    https://astral.sh/uv/install.sh -o "$uv_installer"
  env UV_VERSION="$uv_version" sh "$uv_installer" --no-modify-path
  rm -f "$uv_installer"
  uv_installer=""
}

install_recovery_helper
"$recovery_helper"
rm -rf "$stage"
mkdir "$stage"
tar xzf "$archive" -C "$stage"

deps_changed=0
for dependency_file in pyproject.toml uv.lock; do
  if ! cmp -s "$hermes_src/$dependency_file" "$stage/$dependency_file"; then
    deps_changed=1
  fi
done

[[ -d "$hermes_src/venv" ]] || { echo "existing Hermes venv is unavailable" >&2; exit 1; }
# Never mutate the live runtime. Reflink when the filesystem supports it and
# fall back to a full same-filesystem copy; either form leaves rollback intact.
if ! cp -a --reflink=auto "$hermes_src/venv" "$stage/venv" 2>/dev/null; then
  rm -rf "$stage/venv"
  cp -a "$hermes_src/venv" "$stage/venv"
fi
[[ -f "$hermes_src/.env" ]] && cp -a "$hermes_src/.env" "$stage/.env"

export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
export UV_PYTHON_DOWNLOADS=never
uv_bin=$(find_uv || true)
if [[ -z "$uv_bin" ]]; then
  bootstrap_uv
  uv_bin=$(find_uv || true)
fi
[[ -n "$uv_bin" && -x "$uv_bin" ]] || { echo "uv bootstrap failed" >&2; exit 1; }
sync_output=$(mktemp /tmp/hermes-sync.XXXXXX)
if [[ "$deps_changed" == "1" ]]; then
  if ! UV_PROJECT_ENVIRONMENT="$stage/venv" "$uv_bin" sync \
    --project "$stage" --frozen --inexact --no-dev --no-editable --no-install-project --no-build \
    --extra all --extra langfuse --extra anthropic --extra zpk-runtime \
    >"$sync_output" 2>&1; then
    tail -40 "$sync_output" >&2
    exit 1
  fi
else
  echo "dependency metadata unchanged; copied existing dependencies"
fi

# Always reinstall the project itself as editable. Hermes deliberately blocks
# ad-hoc device wheel builds, and skipping this phase would leave the previous
# source running whenever pyproject.toml and uv.lock are unchanged.
if ! UV_PROJECT_ENVIRONMENT="$stage/venv" "$uv_bin" sync \
  --project "$stage" --frozen --inexact --no-dev --no-build-isolation \
  --reinstall-package hermes-agent \
  --extra all --extra langfuse --extra anthropic --extra zpk-runtime \
  >>"$sync_output" 2>&1; then
  tail -40 "$sync_output" >&2
  exit 1
fi
tail -20 "$sync_output"

# uv writes absolute shebangs for the staging environment. The activated source
# returns to the stable ZPK path, so rewrite only those generated first lines.
while IFS= read -r entry; do
  sed -i "1 s|$stage/venv|$hermes_src/venv|" "$entry"
done < <(grep -Il "^#!$stage/venv/" "$stage/venv/bin/"* 2>/dev/null || true)

# Editable installs also persist the checkout path in .pth/finder metadata.
# Rewrite only text files containing the exact staging path so the activated
# environment follows the stable ZPK source directory after the rename.
while IFS= read -r metadata_file; do
  sed -i "s|$stage|$hermes_src|g" "$metadata_file"
done < <(grep -IlR -F "$stage" "$stage/venv/lib/"python*/site-packages 2>/dev/null || true)

"$stage/venv/bin/python" -c 'import langfuse'
plugin_output=$(mktemp /tmp/hermes-plugins.XXXXXX)
(
  cd "$stage"
  PYTHONPATH="$stage" \
  HERMES_BUNDLED_SKILLS="$stage/skills" \
  HERMES_BUNDLED_PLUGINS="$stage/plugins" \
  HERMES_BUNDLED_LOCALES="$stage/locales" \
    "$stage/venv/bin/python" -m hermes_cli.main plugins list
) >"$plugin_output" 2>&1
head -25 "$plugin_output"
grep -q 'bundled' "$plugin_output" || { echo "no bundled plugins found" >&2; exit 1; }

systemctl stop "$service_name"
rm -rf "$rollback"
mv "$hermes_src" "$rollback"
activated=1
mv "$stage" "$hermes_src"

ln -sfn "$wrapper" "$hermes_bin"
systemctl start "$service_name"
for attempt in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 2 "$health_url" 2>/dev/null || true)
  if [[ "$code" == "200" ]]; then
    echo "health: 200 (attempt $attempt)"
    echo "deployed source: $hermes_src"
    succeeded=1
    rm -rf "$rollback"
    exit 0
  fi
  sleep 2
done

systemctl status "$service_name" --no-pager | tail -20 || true
journalctl -u "$service_name" --since '2 minutes ago' --no-pager | tail -40 || true
echo "local-server health check failed" >&2
exit 1
