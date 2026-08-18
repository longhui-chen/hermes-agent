#!/bin/bash
# ============================================================================
# init-hermes-env.sh — minimal dev venv builder
# ============================================================================
#
# Purpose
#   Build a self-contained Python venv in this checkout's ./venv that
#   downstream tools (e.g. zettlab-local-server's multi-agent registry,
#   which spawns `hermes -p <id> gateway run` per agent) can point at via
#   an absolute path. After this script finishes, ./venv/bin/hermes is a
#   ready-to-spawn binary.
#
# Why a separate script (sibling to setup-hermes.sh)
#   setup-hermes.sh is the end-user installer and intentionally goes the
#   extra mile: it prompts for ripgrep, edits shell rc files, silently
#   ln -sf's into ~/.local/bin/hermes, syncs bundled skills into ~/.hermes,
#   and offers the setup wizard. All correct for someone installing hermes
#   as their primary CLI — but hostile to a teammate already running prod
#   hermes from ~/.hermes/hermes-agent who just needs a second monorepo-
#   local venv to point local-server at without clobbering anything.
#
#   This script is setup-hermes.sh trimmed down to *only* the bits
#   local-server's registry needs: uv + python + venv + editable install.
#
# What this script does
#   1. Locate or auto-install uv (matches setup-hermes.sh logic)
#   2. Provision Python 3.11 via uv (matches setup-hermes.sh logic)
#   3. Recreate ./venv (clean state — same as setup-hermes.sh)
#   4. uv sync --all-extras (lockfile-first, fall back to pip install)
#   5. Copy .env from .env.example if missing
#   6. Print absolute path of resulting hermes binary
#
# What this script does NOT do (vs setup-hermes.sh)
#   - touch ~/.local/bin/hermes (no symlink)
#   - touch ~/.hermes (no skills sync, no HERMES_HOME side effects)
#   - run the hermes setup wizard
#   - prompt for ripgrep install
#   - edit shell rc files
#   - install tinker-atropos (RL training submodule, not relevant for LS)
#
# Usage
#   ./init-hermes-env.sh
#
# Mirrors the cloudnas / devboard onboard pattern (uv venv + uv sync,
# then point hermes_binary at venv/bin/hermes — see the *-onboard skills).
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_VERSION="3.11"

is_termux() {
    [ -n "${TERMUX_VERSION:-}" ] || [[ "${PREFIX:-}" == *"com.termux/files/usr"* ]]
}

if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}✗${NC} $SCRIPT_DIR has no pyproject.toml; not a hermes-agent checkout" >&2
    exit 1
fi

echo ""
echo -e "${CYAN}⚕ Hermes dev venv init${NC}"
echo ""

# ============================================================================
# Install / locate uv  (lifted from setup-hermes.sh)
# ============================================================================

echo -e "${CYAN}→${NC} Checking for uv..."

UV_CMD=""
if is_termux; then
    echo -e "${CYAN}→${NC} Termux detected — using Python's stdlib venv + pip instead of uv"
else
    if command -v uv &> /dev/null; then
        UV_CMD="uv"
    elif [ -x "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    elif [ -x "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
    fi

    if [ -n "$UV_CMD" ]; then
        UV_VERSION=$($UV_CMD --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} uv found ($UV_VERSION)"
    else
        echo -e "${CYAN}→${NC} Installing uv..."
        if curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null; then
            if [ -x "$HOME/.local/bin/uv" ]; then
                UV_CMD="$HOME/.local/bin/uv"
            elif [ -x "$HOME/.cargo/bin/uv" ]; then
                UV_CMD="$HOME/.cargo/bin/uv"
            fi

            if [ -n "$UV_CMD" ]; then
                UV_VERSION=$($UV_CMD --version 2>/dev/null)
                echo -e "${GREEN}✓${NC} uv installed ($UV_VERSION)"
            else
                echo -e "${RED}✗${NC} uv installed but not found. Add ~/.local/bin to PATH and retry."
                exit 1
            fi
        else
            echo -e "${RED}✗${NC} Failed to install uv. Visit https://docs.astral.sh/uv/"
            exit 1
        fi
    fi
fi

# ============================================================================
# Python check (uv can provision it automatically)  (lifted from setup-hermes.sh)
# ============================================================================

echo -e "${CYAN}→${NC} Checking Python $PYTHON_VERSION..."

if is_termux; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_PATH="$(command -v python)"
        if "$PYTHON_PATH" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
            echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
        else
            echo -e "${RED}✗${NC} Termux Python must be 3.11+"
            echo "    Run: pkg install python"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Python not found in Termux"
        echo "    Run: pkg install python"
        exit 1
    fi
else
    if $UV_CMD python find "$PYTHON_VERSION" &> /dev/null; then
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION found"
    else
        echo -e "${CYAN}→${NC} Python $PYTHON_VERSION not found, installing via uv..."
        $UV_CMD python install "$PYTHON_VERSION"
        PYTHON_PATH=$($UV_CMD python find "$PYTHON_VERSION")
        PYTHON_FOUND_VERSION=$($PYTHON_PATH --version 2>/dev/null)
        echo -e "${GREEN}✓${NC} $PYTHON_FOUND_VERSION installed"
    fi
fi

# ============================================================================
# Virtual environment  (lifted from setup-hermes.sh)
# ============================================================================

echo -e "${CYAN}→${NC} Setting up virtual environment..."

if [ -d "venv" ]; then
    echo -e "${CYAN}→${NC} Removing old venv..."
    rm -rf venv
fi

if is_termux; then
    "$PYTHON_PATH" -m venv venv
    echo -e "${GREEN}✓${NC} venv created with stdlib venv"
else
    $UV_CMD venv venv --python "$PYTHON_VERSION"
    echo -e "${GREEN}✓${NC} venv created (Python $PYTHON_VERSION)"
fi

export VIRTUAL_ENV="$SCRIPT_DIR/venv"
SETUP_PYTHON="$SCRIPT_DIR/venv/bin/python"

# ============================================================================
# Dependencies  (lifted from setup-hermes.sh)
# ============================================================================

echo -e "${CYAN}→${NC} Installing dependencies..."

if is_termux; then
    export ANDROID_API_LEVEL="$(getprop ro.build.version.sdk 2>/dev/null || printf '%s' "${ANDROID_API_LEVEL:-}")"
    echo -e "${CYAN}→${NC} Termux detected — installing the tested Android bundle"
    "$SETUP_PYTHON" -m pip install --upgrade pip setuptools wheel
    if [ -f "constraints-termux.txt" ]; then
        "$SETUP_PYTHON" -m pip install -e ".[termux]" -c constraints-termux.txt || {
            echo -e "${YELLOW}⚠${NC} Termux bundle install failed, falling back to base install..."
            "$SETUP_PYTHON" -m pip install -e "." -c constraints-termux.txt
        }
    else
        "$SETUP_PYTHON" -m pip install -e ".[termux]" || "$SETUP_PYTHON" -m pip install -e "."
    fi
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    # Prefer uv sync with lockfile (hash-verified installs) when available,
    # fall back to pip install for compatibility or when lockfile is stale.
    # Some mirrors synthesize upload timestamps for old OpenAI SDK artifacts,
    # which can make the global exclude-newer window filter out our exact pin.
    # The dependency is still exact-pinned and locked; this only prevents mirror
    # metadata drift from making device installs unsatisfiable.
    UV_OPENAI_EXCLUDE_NEWER_ARGS=(--exclude-newer-package openai=false)
    if [ -f "uv.lock" ]; then
        echo -e "${CYAN}→${NC} Using uv.lock for hash-verified installation..."
        UV_PROJECT_ENVIRONMENT="$SCRIPT_DIR/venv" $UV_CMD sync --all-extras --locked "${UV_OPENAI_EXCLUDE_NEWER_ARGS[@]}" 2>/dev/null && \
            echo -e "${GREEN}✓${NC} Dependencies installed (lockfile verified)" || {
            echo -e "${YELLOW}⚠${NC} Lockfile install failed (may be outdated), falling back to pip install..."
            $UV_CMD pip install "${UV_OPENAI_EXCLUDE_NEWER_ARGS[@]}" -e ".[all]" || $UV_CMD pip install "${UV_OPENAI_EXCLUDE_NEWER_ARGS[@]}" -e "."
            echo -e "${GREEN}✓${NC} Dependencies installed"
        }
    else
        $UV_CMD pip install "${UV_OPENAI_EXCLUDE_NEWER_ARGS[@]}" -e ".[all]" || $UV_CMD pip install "${UV_OPENAI_EXCLUDE_NEWER_ARGS[@]}" -e "."
        echo -e "${GREEN}✓${NC} Dependencies installed"
    fi
fi

# ============================================================================
# Environment file  (lifted from setup-hermes.sh)
# ============================================================================

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✓${NC} Created .env from template"
    fi
else
    echo -e "${GREEN}✓${NC} .env exists"
fi

# ============================================================================
# Done — print the binary path the operator pastes into config.yaml
# ============================================================================

HERMES_BIN="$SCRIPT_DIR/venv/bin/hermes"
if [ ! -x "$HERMES_BIN" ]; then
    echo -e "${RED}✗${NC} Build finished but $HERMES_BIN is missing or not executable" >&2
    exit 2
fi

echo ""
echo -e "${GREEN}✓ dev hermes venv ready${NC}"
echo ""
echo "To use with zettlab-local-server, set in its config.yaml:"
echo ""
echo "  agent:"
echo "    hermes_binary: $HERMES_BIN"
echo ""
echo "(Existing prod hermes at ~/.local/bin/hermes is untouched.)"
