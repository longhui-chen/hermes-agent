#!/bin/bash
set -euo pipefail

APP_ROOT=$(dirname "$(readlink -f "$0")")
APP_BASE=$(dirname "$APP_ROOT")
HERMES_SRC="$APP_ROOT/lib/hermes-agent"
HERMES_PYTHON="$HERMES_SRC/venv/bin/python"
DATA_DIR="$APP_BASE/data"
HERMES_HOME="$DATA_DIR/hermes_home"
SECRET_DIR="$APP_BASE/data/secrets"
OTA_DATA_TARGET="$(dirname "$(dirname "$APP_BASE")")/data/$(basename "$APP_BASE")"
VOLUME_DATA_TARGET="/volume1/subvol/apps/$(basename "$APP_BASE")/data"
LOCK_FILE="$SECRET_DIR/prepare-claw-service.lock"
KEY_FILE="$SECRET_DIR/zet_agent.key"
ENV_FILE="$SECRET_DIR/zettlab-claw.env"
PROFILE_PERMISSIONS_MARKER="$SECRET_DIR/profile-permissions-v2.done"
SUBVOLUME_ZETTLAB_PRESETS_ROOT="/volume1/subvol/agents/zettlab-presets"
AGENTS_ZETTLAB_PRESETS_ROOT="/volume1/agents/zettlab-presets"
SUBVOLUME_ZETTLAB_PRESETS_DIR="$SUBVOLUME_ZETTLAB_PRESETS_ROOT/current"
AGENTS_ZETTLAB_PRESETS_DIR="$AGENTS_ZETTLAB_PRESETS_ROOT/current"

env_file_value() {
    local wanted="$1" name value
    while IFS= read -r -d '' name && IFS= read -r -d '' value; do
        [ -n "$name" ] || break
        [ "$name" = "$wanted" ] || continue
        printf '%s\n' "$value"
        return 0
    done < <(dump_persisted_env)
    return 1
}

presets_dir_is_trusted() {
    local candidate="$1" resolved mode group other
    [ -n "$candidate" ] || return 1
    case "$candidate" in
        *$'\n'*|*$'\r'*) return 1 ;;
        /*) ;;
        *) return 1 ;;
    esac
    resolved="$(readlink -f "$candidate" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -d "$resolved" ] || return 1
    presets_dir_is_protected "$candidate" "$resolved" || return 1

    # The device service runs as root. Validate both the concrete version tree
    # and the lexical parent that owns a possible `current` symlink before
    # persisting the selection path. Hermes resolves and pins that symlink once
    # per process, so an OTA flip is visible after restart without permanently
    # pinning future starts to an old version directory.
    if [ "$(id -u)" -eq 0 ]; then
        trusted_presets_path_chain "$resolved" "/" || return 1
        trusted_presets_path_chain "$(dirname "$candidate")" "/" || return 1
    else
        mode="$(stat -c '%a' "$resolved" 2>/dev/null || stat -f '%Lp' "$resolved" 2>/dev/null || true)"
        [ -n "$mode" ] || return 1
        group="${mode: -2:1}"
        other="${mode: -1}"
        (( (10#$group & 2) == 0 )) || return 1
        (( (10#$other & 2) == 0 )) || return 1
    fi
    printf '%s\n' "$candidate"
}

presets_dir_is_protected() {
    local candidate="$1" resolved="$2" root resolved_root
    for root in "$SUBVOLUME_ZETTLAB_PRESETS_ROOT" "$AGENTS_ZETTLAB_PRESETS_ROOT"; do
        case "$candidate" in
            "$root"|"$root"/*) ;;
            *) continue ;;
        esac
        resolved_root="$(readlink -f "$root" 2>/dev/null || true)"
        [ -n "$resolved_root" ] || continue
        case "$resolved" in
            "$resolved_root"|"$resolved_root"/*) return 0 ;;
        esac
    done
    return 1
}

detect_zettlab_presets_dir() {
    local explicit existing candidate trusted
    # ZETTLAB_CLAW_PRESETS_DIR is the systemd/drop-in override seam. The shared
    # EnvironmentFile also contains ZETTLAB_PRESETS_DIR, so reusing that name in
    # a unit drop-in cannot beat a persisted value under systemd's precedence.
    explicit="${ZETTLAB_CLAW_PRESETS_DIR:-${ZETTLAB_PRESETS_DIR:-}}"
    existing="$(env_file_value ZETTLAB_PRESETS_DIR || true)"
    for candidate in "$explicit" "$existing" "$SUBVOLUME_ZETTLAB_PRESETS_DIR" "$AGENTS_ZETTLAB_PRESETS_DIR"; do
        if trusted="$(presets_dir_is_trusted "$candidate")"; then
            printf '%s\n' "$trusted"
            return 0
        fi
    done
    return 1
}

trusted_data_symlink_target() {
    local resolved candidate expected=""
    resolved="$(readlink -f "$DATA_DIR" 2>/dev/null || true)"
    [ -n "$resolved" ] && [ -d "$resolved" ] || return 1
    for candidate in "$OTA_DATA_TARGET" "$VOLUME_DATA_TARGET"; do
        candidate="$(readlink -f "$candidate" 2>/dev/null || true)"
        if [ -n "$candidate" ] && [ "$resolved" = "$candidate" ]; then
            expected="$candidate"
            break
        fi
    done
    [ -n "$expected" ] || return 1

    printf '%s\n' "$resolved"
}

# Presets are executable content selected from a configurable path, so keep the
# ownership and mode checks for that input. Device data symlinks use the fixed
# OTA/volume target allowlist above and deliberately do not inherit this gate.
trusted_presets_path_chain() {
    local current="$1" trust_root="$2" process_uid uid mode group other
    process_uid="$(id -u)"
    while :; do
        uid="$(stat -c '%u' "$current" 2>/dev/null || stat -f '%u' "$current" 2>/dev/null || true)"
        mode="$(stat -c '%a' "$current" 2>/dev/null || stat -f '%Lp' "$current" 2>/dev/null || true)"
        [ -n "$uid" ] && [ -n "$mode" ] || return 1
        if [ "$process_uid" -eq 0 ]; then
            [ "$uid" = "0" ] || return 1
        else
            [ "$uid" = "0" ] || [ "$uid" = "$process_uid" ] || return 1
        fi
        group="${mode: -2:1}"
        other="${mode: -1}"
        (( (10#$group & 2) == 0 )) || return 1
        (( (10#$other & 2) == 0 )) || return 1
        [ "$current" = "/" ] && break
        if [ "$process_uid" -ne 0 ] && [ "$current" = "$trust_root" ]; then
            break
        fi
        current="$(dirname "$current")"
    done
}

pin_trusted_data_symlink() {
    local resolved_path
    [ -L "$DATA_DIR" ] || return 0
    resolved_path="$(trusted_data_symlink_target || true)"
    if [ -z "$resolved_path" ]; then
        echo "refusing untrusted data symlink: $DATA_DIR" >&2
        exit 1
    fi

    # Stop following the mutable app-level symlink after validation. All
    # prepare-time state writes below use this fixed canonical target.
    DATA_DIR="$resolved_path"
    HERMES_HOME="$DATA_DIR/hermes_home"
    SECRET_DIR="$DATA_DIR/secrets"
    LOCK_FILE="$SECRET_DIR/prepare-claw-service.lock"
    KEY_FILE="$SECRET_DIR/zet_agent.key"
    ENV_FILE="$SECRET_DIR/zettlab-claw.env"
    PROFILE_PERMISSIONS_MARKER="$SECRET_DIR/profile-permissions-v2.done"
}

secure_state_directories() {
    local path resolved_path mode expected_mode uid
    pin_trusted_data_symlink
    for path in "$DATA_DIR" "$HERMES_HOME" "$SECRET_DIR"; do
        if [ -L "$path" ] || { [ -e "$path" ] && [ ! -d "$path" ]; }; then
            echo "refusing non-directory state path: $path" >&2
            exit 1
        else
            mkdir -p "$path"
            if [ -L "$path" ] || [ ! -d "$path" ]; then
                echo "state path changed while preparing it: $path" >&2
                exit 1
            fi
            resolved_path="$path"
        fi
        if [ "$(id -u)" -eq 0 ]; then
            uid="$(stat -c '%u' "$resolved_path" 2>/dev/null || stat -f '%u' "$resolved_path" 2>/dev/null || true)"
            if [ "$uid" != "0" ]; then
                echo "refusing non-root-owned state directory: $path" >&2
                exit 1
            fi
        fi
        if [ "$path" = "$SECRET_DIR" ]; then
            expected_mode=0700
        else
            mode="$(stat -c '%a' "$resolved_path" 2>/dev/null || stat -f '%Lp' "$resolved_path" 2>/dev/null || true)"
            [ -n "$mode" ] || {
                echo "cannot verify state directory mode: $path" >&2
                exit 1
            }
            printf -v expected_mode '%04o' "$(( (8#$mode | 0700) & 0755 ))"
        fi
        chmod "$expected_mode" "$resolved_path"
        mode="$(stat -c '%a' "$resolved_path" 2>/dev/null || stat -f '%Lp' "$resolved_path" 2>/dev/null || true)"
        [ -n "$mode" ] || {
            echo "cannot verify state directory mode: $path" >&2
            exit 1
        }
    done
}

secure_profile_secret_files() {
    local profiles_root="$HERMES_HOME/profiles" path uid process_uid
    process_uid="$(id -u)"

    if [ -L "$profiles_root" ] || { [ -e "$profiles_root" ] && [ ! -d "$profiles_root" ]; }; then
        echo "refusing non-directory profile root: $profiles_root" >&2
        exit 1
    fi
    mkdir -p "$profiles_root"
    uid="$(stat -c '%u' "$profiles_root" 2>/dev/null || stat -f '%u' "$profiles_root" 2>/dev/null || true)"
    if [ "$uid" != "$process_uid" ]; then
        echo "refusing profile root not owned by service user: $profiles_root" >&2
        exit 1
    fi

    # This root-only traversal barrier is the durable isolation boundary. It is
    # O(1) on every start, so profile data growth cannot delay service startup.
    chmod 0700 "$profiles_root"

    path="$HERMES_HOME/.env"
    if [ -e "$path" ] || [ -L "$path" ]; then
        if [ -L "$path" ] || [ ! -f "$path" ]; then
            echo "refusing non-regular profile secret file: $path" >&2
            exit 1
        fi
        uid="$(stat -c '%u' "$path" 2>/dev/null || stat -f '%u' "$path" 2>/dev/null || true)"
        if [ "$uid" != "$process_uid" ]; then
            echo "refusing profile secret file not owned by service user: $path" >&2
            exit 1
        fi
        chmod 0600 "$path"
    fi

    if [ -e "$PROFILE_PERMISSIONS_MARKER" ] || [ -L "$PROFILE_PERMISSIONS_MARKER" ]; then
        if [ -L "$PROFILE_PERMISSIONS_MARKER" ] || [ ! -f "$PROFILE_PERMISSIONS_MARKER" ]; then
            echo "refusing non-regular profile permissions marker" >&2
            exit 1
        fi
        uid="$(stat -c '%u' "$PROFILE_PERMISSIONS_MARKER" 2>/dev/null || stat -f '%u' "$PROFILE_PERMISSIONS_MARKER" 2>/dev/null || true)"
        if [ "$uid" != "$process_uid" ]; then
            echo "refusing profile permissions marker not owned by service user" >&2
            exit 1
        fi
        chmod 0600 "$PROFILE_PERMISSIONS_MARKER"
        return
    fi

    # One bounded pass tightens legacy top-level profiles and their .env files.
    # Descendants remain protected by the root barrier above; new profiles are
    # created owner-only by hermes_cli.profiles. The marker keeps normal starts
    # independent of profile count and state-tree size.
    HERMES_PROFILES_ROOT="$profiles_root" \
    HERMES_PROFILE_PERMISSIONS_MARKER="$PROFILE_PERMISSIONS_MARKER" \
        "$HERMES_PYTHON" - <<'PY'
import json
import os
import stat
import tempfile

profiles_root = os.environ["HERMES_PROFILES_ROOT"]
marker = os.environ["HERMES_PROFILE_PERMISSIONS_MARKER"]
try:
    requested_limit = int(
        os.environ.get("HERMES_PROFILE_PERMISSION_MIGRATION_MAX_PROFILES", "4096")
    )
except ValueError:
    requested_limit = 4096
limit = max(1, min(requested_limit, 4096))
overflow = False

deletion_marker_limit = 64 * 1024
deletion_pending = b"deleted\n"
deletion_complete = b"deleted\ncleanup-complete\n"
deletion_origin_header = b"origin-json\n"


def reject_non_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def valid_deletion_origin(body):
    if not body.startswith(deletion_origin_header) or not body.endswith(b"\n"):
        return False
    try:
        value = json.loads(
            body[len(deletion_origin_header):-1],
            parse_constant=reject_non_json_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        return False
    return isinstance(value, dict)


def valid_local_server_deletion_blocker(body):
    if body in {deletion_pending, deletion_complete}:
        return True
    if body.startswith(deletion_complete):
        return valid_deletion_origin(body[len(deletion_complete):])
    if body.startswith(deletion_pending):
        return valid_deletion_origin(body[len(deletion_pending):])
    return False


def secure_local_server_deletion_marker(
    path, initial_info, *, dir_fd=None, display_path=None
):
    display_path = display_path or path
    if initial_info.st_uid != os.geteuid():
        raise SystemExit(
            f"refusing deletion marker not owned by service user: {display_path}"
        )
    if initial_info.st_size > deletion_marker_limit:
        return False

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise SystemExit(
            f"cannot safely open deletion marker: {display_path}: {exc}"
        )
    try:
        current_info = os.fstat(fd)
        if (
            not stat.S_ISREG(current_info.st_mode)
            or current_info.st_dev != initial_info.st_dev
            or current_info.st_ino != initial_info.st_ino
            or current_info.st_uid != os.geteuid()
        ):
            raise SystemExit(
                f"deletion marker changed while checking it: {display_path}"
            )

        chunks = []
        remaining = deletion_marker_limit + 1
        while remaining > 0:
            chunk = os.read(fd, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        body = b"".join(chunks)
        if len(body) > deletion_marker_limit or not valid_local_server_deletion_blocker(body):
            return False
        os.fchmod(fd, 0o600)
        return True
    finally:
        os.close(fd)


def secure_local_server_deletion_blocker(name, path, initial_info):
    # Current local-server never tombstones the permanent main profile and
    # rejects Hermes' virtual default profile. Older versions could leave a
    # default blocker, so accept that reserved name only when the durable
    # .deleted-agents sidecar independently proves the same deletion intent.
    if name == "main" or name.startswith("."):
        return False
    if name == "default":
        sidecar_root = os.path.join(profiles_root, ".deleted-agents")
        sidecar = os.path.join(sidecar_root, name)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            sidecar_root_fd = os.open(sidecar_root, directory_flags)
        except OSError:
            return False
        try:
            sidecar_root_info = os.fstat(sidecar_root_fd)
            if (
                not stat.S_ISDIR(sidecar_root_info.st_mode)
                or sidecar_root_info.st_uid != os.geteuid()
            ):
                return False
            try:
                sidecar_info = os.stat(
                    name,
                    dir_fd=sidecar_root_fd,
                    follow_symlinks=False,
                )
            except OSError:
                return False
            if not stat.S_ISREG(sidecar_info.st_mode):
                return False
            if not secure_local_server_deletion_marker(
                name,
                sidecar_info,
                dir_fd=sidecar_root_fd,
                display_path=sidecar,
            ):
                return False
        finally:
            os.close(sidecar_root_fd)
    return secure_local_server_deletion_marker(path, initial_info)


def secure_reserved_profile_path(name):
    path = os.path.join(profiles_root, name)
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        if secure_local_server_deletion_blocker(name, path, info):
            return
        raise SystemExit(f"refusing non-directory profile path: {path}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"refusing non-directory profile path: {path}")
    if info.st_uid != os.geteuid():
        raise SystemExit(
            f"refusing profile directory not owned by service user: {path}"
        )
    os.chmod(path, 0o700, follow_symlinks=False)


# These names carry deletion semantics that cannot be left to the bounded
# migration scan. Validate them directly before a successful run can publish
# the one-time completion marker, even when more than `limit` profiles exist.
for reserved_name in ("main", "default"):
    secure_reserved_profile_path(reserved_name)

with os.scandir(profiles_root) as entries:
    for index, entry in enumerate(entries):
        if index >= limit:
            overflow = True
            break
        info = entry.stat(follow_symlinks=False)
        # local-server deletion is fail-closed by an owner-only regular blocker
        # at profiles/<id>. Preserve only its exact bounded marker grammar; an
        # arbitrary file or symlink still aborts the migration.
        if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
            if secure_local_server_deletion_blocker(
                entry.name, entry.path, info
            ):
                continue
            raise SystemExit(f"refusing non-directory profile path: {entry.path}")
        if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise SystemExit(f"refusing non-directory profile path: {entry.path}")
        if info.st_uid != os.geteuid():
            raise SystemExit(
                f"refusing profile directory not owned by service user: {entry.path}"
            )
        os.chmod(entry.path, 0o700, follow_symlinks=False)

        env_path = os.path.join(entry.path, ".env")
        try:
            env_info = os.lstat(env_path)
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(env_info.st_mode) or stat.S_ISLNK(env_info.st_mode):
            raise SystemExit(f"refusing non-regular profile secret file: {env_path}")
        if env_info.st_uid != os.geteuid():
            raise SystemExit(
                f"refusing profile secret file not owned by service user: {env_path}"
            )
        os.chmod(env_path, 0o600, follow_symlinks=False)

marker_dir = os.path.dirname(marker)
fd, temporary = tempfile.mkstemp(
    prefix=".profile-permissions-v2.", suffix=".tmp", dir=marker_dir
)
try:
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w", encoding="ascii") as stream:
        stream.write("root-barrier-v2\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker)
finally:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass

if overflow:
    print(
        f"bounded legacy profile migration stopped after {limit} profiles; "
        "remaining entries are protected by the owner-only profile root",
        file=os.sys.stderr,
    )
PY
}

acquire_prepare_lock() {
    if [ -L "$LOCK_FILE" ] || { [ -e "$LOCK_FILE" ] && [ ! -f "$LOCK_FILE" ]; }; then
        echo "refusing non-regular prepare lock file: $LOCK_FILE" >&2
        exit 1
    fi
    exec 9> "$LOCK_FILE"
    chmod 0600 "$LOCK_FILE"
    HERMES_PREPARE_LOCK_FD=9 "$HERMES_PYTHON" - <<'PY'
import errno
import fcntl
import os
import sys
import time

fd = int(os.environ["HERMES_PREPARE_LOCK_FD"])
try:
    timeout = float(os.environ.get("HERMES_PREPARE_LOCK_TIMEOUT_SECONDS", "30"))
except ValueError:
    timeout = 30.0
timeout = max(0.1, min(timeout, 60.0))
deadline = time.monotonic() + timeout

while True:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        break
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.EAGAIN):
            raise
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"timed out waiting for Claw prepare lock after {timeout:g}s",
                file=sys.stderr,
            )
            raise SystemExit(1)
        time.sleep(min(0.1, remaining))
PY
}

dump_persisted_env() {
    if [ ! -e "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ]; then
        printf '\000\000'
        return 0
    fi
    if [ -L "$ENV_FILE" ] || [ ! -f "$ENV_FILE" ]; then
        echo "refusing non-regular environment file: $ENV_FILE" >&2
        return 1
    fi

    "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" "$ENV_FILE"
}

generate_key() {
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 32
        return
    fi
    "$HERMES_PYTHON" - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
}

write_agent_env() {
    if [ -L "$KEY_FILE" ] || { [ -e "$KEY_FILE" ] && [ ! -f "$KEY_FILE" ]; }; then
        echo "refusing non-regular ZET_AGENT_KEY file: $KEY_FILE" >&2
        exit 1
    fi
    if [ -L "$ENV_FILE" ] || { [ -e "$ENV_FILE" ] && [ ! -f "$ENV_FILE" ]; }; then
        echo "refusing non-regular environment file: $ENV_FILE" >&2
        exit 1
    fi

    if [ ! -s "$KEY_FILE" ]; then
        generate_key > "$KEY_FILE.tmp.$$"
        chmod 0600 "$KEY_FILE.tmp.$$"
        mv "$KEY_FILE.tmp.$$" "$KEY_FILE"
    fi
    chmod 0600 "$KEY_FILE"

    key="$(tr -d '\r\n' < "$KEY_FILE")"
    if [ -z "$key" ]; then
        echo "empty ZET_AGENT_KEY in $KEY_FILE" >&2
        exit 1
    fi

    {
        printf 'ZET_AGENT_KEY=%s\n' "$key"
        printf 'ZET_AGENT_ENABLED=true\n'
        printf 'ZET_AGENT_HOST=127.0.0.1\n'
        printf 'ZET_AGENT_PORT=7900\n'
        if [ -n "$ZETTLAB_PRESETS_DIR" ]; then
            printf 'ZETTLAB_PRESETS_DIR=%s\n' "$ZETTLAB_PRESETS_DIR"
        fi

        # Preserve user-managed EnvironmentFile entries verbatim while replacing
        # complete logical assignments owned by this package. Physical-line
        # filtering is unsafe because systemd permits multiline quoted values.
        # Generated assignments come first so an accepted EOF-unclosed user value
        # cannot swallow package-owned fields appended after it.
        if [ -f "$ENV_FILE" ]; then
            "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" \
                --filter-excluding "$ENV_FILE" \
                ZET_AGENT_KEY ZET_AGENT_ENABLED ZET_AGENT_HOST \
                ZET_AGENT_PORT ZETTLAB_PRESETS_DIR \
                GATEWAY_MULTIPLEX_PROFILES ZETTLAB_CLAW_PRESETS_DIR \
                HERMES_HOME HERMES_BUNDLED_SKILLS HERMES_BUNDLED_PLUGINS \
                HERMES_LAZY_INSTALL_TARGET \
                HERMES_MANAGED_GATEWAY HERMES_MANAGED_CGROUP_UNIT \
                HERMES_MANAGED_CGROUP_ROOT
        fi
    } > "$ENV_FILE.tmp.$$"
    if ! "$HERMES_PYTHON" "$APP_ROOT/parse-environment-file.py" \
        "$ENV_FILE.tmp.$$" >/dev/null; then
        rm -f "$ENV_FILE.tmp.$$"
        echo "refusing invalid generated environment file" >&2
        return 1
    fi
    chmod 0600 "$ENV_FILE.tmp.$$"
    if [ -f "$ENV_FILE" ] && [ ! -L "$ENV_FILE" ] \
        && cmp -s "$ENV_FILE.tmp.$$" "$ENV_FILE"; then
        chmod 0600 "$ENV_FILE"
        rm -f "$ENV_FILE.tmp.$$"
    else
        mv "$ENV_FILE.tmp.$$" "$ENV_FILE"
    fi
}

if [ ! -x "$HERMES_PYTHON" ]; then
    echo "hermes python is not ready: $HERMES_PYTHON" >&2
    exit 127
fi
emit_env=false
case "${1:-}" in
    "") ;;
    --emit-env) emit_env=true ;;
    *)
        echo "usage: $(basename "$0") [--emit-env]" >&2
        exit 2
        ;;
esac

secure_state_directories
acquire_prepare_lock
secure_profile_secret_files
ZETTLAB_PRESETS_DIR="$(detect_zettlab_presets_dir || true)"
write_agent_env

if [ "$emit_env" = "true" ]; then
    dump_persisted_env
else
    echo "Zettlab Claw runtime state prepared."
    echo "  env: $ENV_FILE"
fi
