#!/usr/bin/env bash
# Local equivalent of .github/workflows/infist-dev-deploy.yml — deploy the
# current hermes-agent source (git HEAD) to a single OTA board over SSH,
# WITHOUT a self-hosted runner / GitHub Actions.
#
# Same semantics as the workflow:
#   - git archive HEAD (tracked files only; venv/.env are gitignored)
#   - scp to the board
#   - keep the board's venv + .env, back up old source, swap in new source
#   - install the new source into the venv as an editable package
#   - keep /usr/local/bin/hermes -> ZPK wrapper, restart local-server, health-check
#
# Differences vs the workflow (board .98 is provisioned slightly differently):
#   - Does NOT rely on /usr/local/bin/hermes (it's a broken symlink on .98);
#     resolves HERMES_SRC by globbing the OTA app dir, repairs the symlink at the end.
#   - Uses the board venv's own pip (no uv on .98).
#   - Default editable install is --no-deps (fast/safe: the board venv already
#     has [all,langfuse] from its ZPK). Pass FULL_DEPS=1 to do .[all,langfuse].
#
# Usage:
#   BOARD_HOST=192.168.31.98 BOARD_PASS=Zettlab2023 ./scripts/dev/deploy-hermes-to-board.sh
#   FULL_DEPS=1 BOARD_HOST=... ./scripts/dev/deploy-hermes-to-board.sh   # reinstall deps too

set -euo pipefail

BOARD_HOST="${BOARD_HOST:-192.168.31.98}"
BOARD_PORT="${BOARD_PORT:-22}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PASS="${BOARD_PASS:-Zettlab2023}"
FULL_DEPS="${FULL_DEPS:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

command -v sshpass >/dev/null || { echo "missing sshpass" >&2; exit 1; }
SHA="$(git rev-parse --short HEAD)"
TARBALL="/tmp/hermes-src-${SHA}.tgz"

echo ">>> git archive HEAD ($SHA) -> $TARBALL"
git archive --format=tar.gz -o "$TARBALL" HEAD
ls -lh "$TARBALL"

export SSHPASS="$BOARD_PASS"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)

echo ">>> scp -> ${BOARD_USER}@${BOARD_HOST}:/tmp/hermes-src.new.tgz"
sshpass -e scp -P "$BOARD_PORT" "${SSH_OPTS[@]}" "$TARBALL" "${BOARD_USER}@${BOARD_HOST}:/tmp/hermes-src.new.tgz"

echo ">>> deploy on board"
sshpass -e ssh -p "$BOARD_PORT" "${SSH_OPTS[@]}" "${BOARD_USER}@${BOARD_HOST}" \
  "FULL_DEPS=${FULL_DEPS} SHA=${SHA} bash -s" <<'DEPLOY'
set -euo pipefail

# 1. Resolve the running hermes source dir (don't trust the wrapper symlink on .98).
HERMES_SRC="$(ls -d /zettos/main/apps/com.zettlab.claw/*/lib/hermes-agent 2>/dev/null | head -1)"
# Prefer the version the 'current' symlink points at, if present.
CUR="$(readlink -f /zettos/main/apps/com.zettlab.claw/current 2>/dev/null || true)"
[ -n "$CUR" ] && [ -d "$CUR/lib/hermes-agent" ] && HERMES_SRC="$CUR/lib/hermes-agent"
[ -d "$HERMES_SRC" ] || { echo "ERR: hermes source dir not found"; exit 1; }
APP_ROOT="$(dirname "$(dirname "$HERMES_SRC")")"
WRAPPER="$APP_ROOT/bin/hermes"
PY="$HERMES_SRC/venv/bin/python"
echo "HERMES_SRC=$HERMES_SRC"
echo "WRAPPER=$WRAPPER (exists=$([ -x "$WRAPPER" ] && echo yes || echo no))"
[ -x "$PY" ] || { echo "ERR: venv python missing at $PY"; exit 1; }

ts=$(date +%Y%m%d-%H%M%S)

# 2. Preserve venv + .env; back up old source for rollback.
[ -d "$HERMES_SRC/venv" ] && mv "$HERMES_SRC/venv" "/tmp/hermes-venv-keep.$$"
[ -f "$HERMES_SRC/.env" ] && cp "$HERMES_SRC/.env" "/tmp/hermes-env-keep.$$" || true
BACKUP="/tmp/hermes-src.bak-$ts.tgz"
tar czf "$BACKUP" -C "$HERMES_SRC" . 2>/dev/null || true
echo "BACKUP_SRC=$BACKUP"

# 3. Clear source dir (venv moved out), extract new source.
find "$HERMES_SRC" -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
tar xzf /tmp/hermes-src.new.tgz -C "$HERMES_SRC"

# 4. Restore venv + .env.
[ -d "/tmp/hermes-venv-keep.$$" ] && mv "/tmp/hermes-venv-keep.$$" "$HERMES_SRC/venv"
[ -f "/tmp/hermes-env-keep.$$" ] && cp "/tmp/hermes-env-keep.$$" "$HERMES_SRC/.env" || true

# 5. Install new source into the venv as editable, using uv (the venv's own pip
#    can be broken on these boards — uv is self-contained, like the workflow).
export PATH="$HOME/.local/bin:$PATH"
export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple/"
export UV_PYTHON_DOWNLOADS=never
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo "=== installing uv ==="
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
cd "$HERMES_SRC"
if [ "${FULL_DEPS:-0}" = "1" ]; then
  echo "=== uv pip install -e .[all,langfuse] (full deps) ==="
  "$UV" pip install --python "$PY" -e ".[all,langfuse]" 2>&1 | tail -15
else
  echo "=== uv pip install -e . --no-deps (fast; reuse existing venv deps) ==="
  "$UV" pip install --python "$PY" -e . --no-deps 2>&1 | tail -15
fi

# 5b. Import smoke test — if a needed dep is missing (version jump), surface it
#     and retry with full deps.
if ! "$PY" -c "import run_agent, agent.agent_init, gateway.run, agent.auxiliary_client, agent.transports.chat_completions" 2>/tmp/hermes-import-err; then
  echo "::WARN import failed (likely a missing dep from the version jump):"
  tail -5 /tmp/hermes-import-err
  echo ">>> retrying with full deps .[all,langfuse]"
  "$UV" pip install --python "$PY" -e ".[all,langfuse]" 2>&1 | tail -15
  "$PY" -c "import run_agent, agent.agent_init, gateway.run" \
    || { echo "ERR: still failing to import after full deps"; exit 1; }
fi
echo "import smoke test: OK"

# 6. Repair /usr/local/bin/hermes -> wrapper (broken on .98).
if [ -x "$WRAPPER" ]; then
  ln -sfn "$WRAPPER" /usr/local/bin/hermes
  echo "symlink /usr/local/bin/hermes -> $WRAPPER"
fi

# 7. restart local-server (forks hermes lazily; no global daemon) + health.
systemctl restart zettlab-local-server
OK=0
for i in $(seq 1 30); do
  code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:9090/health 2>/dev/null || echo 000)
  [ "$code" = "200" ] && { echo "health 200 (after ${i}x2s)"; OK=1; break; }
  sleep 2
done
[ "$OK" = "1" ] || { echo "ERR: local-server health failed"; exit 1; }

echo "DEPLOYED_SRC=$HERMES_SRC  SHA=${SHA}  BACKUP=$BACKUP"
DEPLOY

echo ">>> done: hermes @ $SHA deployed to ${BOARD_HOST}"
