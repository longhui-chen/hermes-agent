import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


_SOURCE = Path(__file__).parents[2] / "nix" / "safeProfileDirs.py"
_SPEC = importlib.util.spec_from_file_location("safe_profile_dirs", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
safe_profile_dirs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(safe_profile_dirs)


def test_safe_profile_directory_flow_creates_exact_managed_chain(tmp_path):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    current = home.stat()

    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )

    for path in (memories / ".imports", memories / ".imports" / "backups"):
        opened = path.stat(follow_symlinks=False)
        assert opened.st_uid == current.st_uid
        assert opened.st_gid == current.st_gid
        assert opened.st_mode & 0o0770 == 0o0770
        if sys.platform.startswith("linux"):
            assert opened.st_mode & stat.S_ISGID


@pytest.mark.parametrize("leaf", ["home", "memories", ".imports", "backups"])
def test_safe_profile_directory_flow_never_follows_transaction_symlink(
    tmp_path, leaf
):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    imports = memories / ".imports"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o755)
    if leaf == "home":
        home.symlink_to(outside, target_is_directory=True)
    elif leaf == "memories":
        home.mkdir()
        memories.symlink_to(outside, target_is_directory=True)
    elif leaf == ".imports":
        home.mkdir()
        memories.mkdir()
        imports.symlink_to(outside, target_is_directory=True)
    else:
        home.mkdir()
        memories.mkdir()
        imports.mkdir()
        (imports / "backups").symlink_to(outside, target_is_directory=True)
    before = outside.stat(follow_symlinks=False)
    current = tmp_path.stat()

    with pytest.raises(OSError):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    after = outside.stat(follow_symlinks=False)
    assert (after.st_uid, after.st_gid, after.st_mode) == (
        before.st_uid,
        before.st_gid,
        before.st_mode,
    )


@pytest.mark.parametrize("kind", ["file", "fifo"])
def test_safe_profile_directory_flow_rejects_non_directory_leaf(tmp_path, kind):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    invalid = memories / ".imports"
    if kind == "file":
        invalid.write_text("not a directory")
    else:
        os.mkfifo(invalid)
    current = home.stat()

    with pytest.raises(OSError):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    assert invalid.is_file() if kind == "file" else stat.S_ISFIFO(invalid.lstat().st_mode)


def test_safe_profile_directory_flow_detects_rename_before_open(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    imports = memories / ".imports"
    imports.mkdir(parents=True)
    imports.chmod(0o755)
    current = home.stat()
    original_open = safe_profile_dirs.os.open
    swapped = False

    def swap_imports_before_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == ".imports" and dir_fd is not None and not swapped:
            swapped = True
            os.rename(".imports", ".imports-old", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.mkdir(".imports", 0o711, dir_fd=dir_fd)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(safe_profile_dirs.os, "open", swap_imports_before_open)

    with pytest.raises(RuntimeError, match="changed during open"):
        safe_profile_dirs.ensure_transaction_directories(
            home, current.st_uid, current.st_gid
        )

    assert swapped is True
    assert stat.S_IMODE(imports.stat().st_mode) == 0o711


def test_safe_profile_directory_flow_normalizes_only_opened_inodes(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    current = home.stat()
    normalized = []
    original_fchown = safe_profile_dirs.os.fchown

    def record_fchown(fd, uid, gid):
        opened = os.fstat(fd)
        normalized.append((opened.st_dev, opened.st_ino))
        return original_fchown(fd, uid, gid)

    monkeypatch.setattr(safe_profile_dirs.os, "fchown", record_fchown)
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )

    for path in (home, memories, memories / ".imports", memories / ".imports/backups"):
        opened = path.stat(follow_symlinks=False)
        assert (opened.st_dev, opened.st_ino) in normalized


def test_exact_managed_child_rejects_foreign_device_before_mutation(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    memories = home / "memories"
    memories.mkdir(parents=True)
    current = home.stat()
    parent_fd = os.open(home, safe_profile_dirs._directory_flags())
    child_fd = os.open(memories, safe_profile_dirs._directory_flags())
    child_identity = os.fstat(child_fd)
    original_fstat = safe_profile_dirs.os.fstat

    def foreign_fstat(fd):
        opened = original_fstat(fd)
        if (opened.st_dev, opened.st_ino) == (
            child_identity.st_dev,
            child_identity.st_ino,
        ):
            values = list(opened)
            values[2] = opened.st_dev + 1
            return os.stat_result(values)
        return opened

    monkeypatch.setattr(
        safe_profile_dirs,
        "_open_or_create_child",
        lambda _parent_fd, _name: os.dup(child_fd),
    )
    monkeypatch.setattr(safe_profile_dirs.os, "fstat", foreign_fstat)
    monkeypatch.setattr(
        safe_profile_dirs.os,
        "fchown",
        lambda *_args: pytest.fail("foreign child was chowned before device gate"),
    )
    monkeypatch.setattr(
        safe_profile_dirs.os,
        "fchmod",
        lambda *_args: pytest.fail("foreign child was chmodded before device gate"),
    )
    try:
        with pytest.raises(OSError) as raised:
            safe_profile_dirs._normalize_child(
                parent_fd,
                "memories",
                current.st_uid,
                current.st_gid,
                0o2770,
                expected_device=current.st_dev,
            )
        assert raised.value.errno == safe_profile_dirs.errno.EXDEV
    finally:
        os.close(child_fd)
        os.close(parent_fd)


@pytest.mark.parametrize("operation", ["recursive_ownership", "shared_file_modes"])
def test_recursive_profile_scan_skips_normal_enoent_churn(
    tmp_path, monkeypatch, operation
):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    churn = home / "sessions" / "churn.txt"
    churn.write_text("gone")
    original_stat = safe_profile_dirs.os.stat

    def missing_during_scan(path, *args, **kwargs):
        if path == churn.name and kwargs.get("dir_fd") is not None:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(safe_profile_dirs.os, "stat", missing_during_scan)
    safe_profile_dirs.ensure_transaction_directories(
        home,
        current.st_uid,
        current.st_gid,
        **{operation: True},
    )


@pytest.mark.parametrize("operation", ["recursive_ownership", "shared_file_modes"])
def test_recursive_profile_scan_bounds_persistent_identity_churn(
    tmp_path, monkeypatch, operation
):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    (home / "sessions" / "racy").mkdir()
    original_check = safe_profile_dirs._require_visible_identity
    attempts = 0

    def always_changes(parent_fd, name, child_fd, expected=None):
        nonlocal attempts
        if name == "racy":
            attempts += 1
            raise RuntimeError("changed")
        return original_check(parent_fd, name, child_fd, expected)

    monkeypatch.setattr(
        safe_profile_dirs, "_require_visible_identity", always_changes
    )
    with pytest.raises(RuntimeError, match="kept changing"):
        safe_profile_dirs.ensure_transaction_directories(
            home,
            current.st_uid,
            current.st_gid,
            **{operation: True},
        )
    assert attempts == safe_profile_dirs._RECURSIVE_IDENTITY_RETRIES


def test_recursive_profile_scan_recovers_from_transient_identity_churn(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    (home / "sessions" / "racy").mkdir()
    original_check = safe_profile_dirs._require_visible_identity
    attempts = 0

    def changes_once(parent_fd, name, child_fd, expected=None):
        nonlocal attempts
        if name == "racy" and attempts == 0:
            attempts += 1
            raise RuntimeError("changed")
        return original_check(parent_fd, name, child_fd, expected)

    monkeypatch.setattr(safe_profile_dirs, "_require_visible_identity", changes_once)
    safe_profile_dirs.ensure_transaction_directories(
        home,
        current.st_uid,
        current.st_gid,
        recursive_ownership=True,
    )
    assert attempts == 1


@pytest.mark.parametrize("operation", ["recursive_ownership", "shared_file_modes"])
def test_recursive_profile_scan_skips_beyond_bounded_depth(
    tmp_path, monkeypatch, capsys, operation
):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    monkeypatch.setattr(safe_profile_dirs, "_MAX_RECURSIVE_DEPTH", 8)
    cursor = home / "sessions"
    for index in range(20):
        cursor = cursor / f"level-{index}"
        cursor.mkdir()
    deepest = cursor / "private.txt"
    deepest.write_text("private")
    deepest.chmod(0o600)

    safe_profile_dirs.ensure_transaction_directories(
        home,
        current.st_uid,
        current.st_gid,
        **{operation: True},
    )

    assert "skipping managed profile subtree deeper than 8" in capsys.readouterr().err
    assert stat.S_IMODE(deepest.stat().st_mode) == 0o600


@pytest.mark.parametrize("leaf", sorted(safe_profile_dirs._MANAGED_ROOT_LEAVES))
@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_safe_profile_directory_flow_rejects_unsafe_managed_leaf(
    tmp_path, leaf, kind
):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    unsafe = home / leaf
    if kind == "symlink":
        outside = tmp_path / "outside"
        outside.write_text("do not touch")
        unsafe.symlink_to(outside)
    else:
        os.mkfifo(unsafe)

    with pytest.raises(OSError, match="unsafe managed profile leaf"):
        safe_profile_dirs.ensure_transaction_directories(
            home,
            current.st_uid,
            current.st_gid,
            shared_file_modes=True,
        )

    if kind == "symlink":
        assert outside.read_text() == "do not touch"
    else:
        assert stat.S_ISFIFO(unsafe.lstat().st_mode)


def test_managed_leaf_write_is_anchored_and_replaces_symlink_not_target(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    identity = safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    outside = tmp_path / "outside"
    outside.write_text("do not touch")
    (home / ".managed").symlink_to(outside)
    home_fd = safe_profile_dirs._open_verified_home(home, identity)
    try:
        safe_profile_dirs._write_managed_leaf(
            home_fd,
            ".managed",
            b"managed",
            current.st_uid,
            current.st_gid,
            0o644,
        )
    finally:
        os.close(home_fd)

    assert outside.read_text() == "do not touch"
    assert not (home / ".managed").is_symlink()
    assert (home / ".managed").read_bytes() == b"managed"


def test_managed_leaf_action_rejects_replaced_profile_identity(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    identity = safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    home.rename(tmp_path / ".hermes-old")
    home.mkdir()

    with pytest.raises(RuntimeError, match="changed after secure setup"):
        safe_profile_dirs._open_verified_home(home, identity)


def test_profile_trust_anchor_binds_opened_home_identity(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    trust_dir = tmp_path / "run"
    trust_dir.mkdir()
    trust = trust_dir / "profile-trust"
    original_fchown = safe_profile_dirs.os.fchown

    def allow_synthetic_root_owner(fd, uid, gid):
        if (uid, gid) == (0, 0):
            return None
        return original_fchown(fd, uid, gid)

    monkeypatch.setattr(safe_profile_dirs.os, "fchown", allow_synthetic_root_owner)

    identity = safe_profile_dirs.ensure_transaction_directories(
        home,
        current.st_uid,
        current.st_gid,
        trust_anchor=trust,
    )

    fields = dict(line.split("=", 1) for line in trust.read_text().splitlines())
    assert fields["version"] == "2"
    assert f'{fields["dev"]}:{fields["ino"]}' == identity
    assert fields["home"] == str(home)


def test_safe_profile_cli_round_trips_anchored_leaf(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    setup = subprocess.run(
        [sys.executable, str(_SOURCE), str(home), str(current.st_uid), str(current.st_gid)],
        check=True,
        capture_output=True,
        text=True,
    )
    identity = setup.stdout.strip()
    content = tmp_path / "managed-content"
    content.write_text("managed")
    subprocess.run(
        [
            sys.executable,
            str(_SOURCE),
            "--expected-identity",
            identity,
            str(home),
            str(current.st_uid),
            str(current.st_gid),
            "--write-leaf",
            ".managed",
            "--content-file",
            str(content),
            "--leaf-mode",
            "0644",
        ],
        check=True,
    )
    read_back = subprocess.run(
        [
            sys.executable,
            str(_SOURCE),
            "--expected-identity",
            identity,
            str(home),
            str(current.st_uid),
            str(current.st_gid),
            "--read-leaf",
            ".managed",
        ],
        check=True,
        capture_output=True,
    )

    assert read_back.stdout == b"managed"


def test_managed_plugin_sync_replaces_only_managed_symlinks(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    identity = safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    plugins = home / "plugins"
    stale_target = tmp_path / "stale-target"
    stale_target.mkdir()
    (plugins / "nix-managed-stale").symlink_to(stale_target)
    regular = plugins / "nix-managed-regular"
    regular.write_text("preserve user data")
    desired_target = tmp_path / "desired-plugin"
    desired_target.mkdir()
    manifest = tmp_path / "plugins.json"
    manifest.write_text(
        json.dumps([{"name": "desired", "target": str(desired_target)}])
    )

    home_fd = safe_profile_dirs._open_verified_home(home, identity)
    try:
        safe_profile_dirs._sync_plugin_links(
            home_fd, manifest, current.st_uid, current.st_gid
        )
    finally:
        os.close(home_fd)

    assert not (plugins / "nix-managed-stale").exists()
    assert regular.read_text() == "preserve user data"
    desired = plugins / "nix-managed-desired"
    assert desired.is_symlink()
    assert os.readlink(desired) == str(desired_target)


def test_managed_plugin_sync_rejects_replaced_plugins_directory(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    current = home.stat()
    identity = safe_profile_dirs.ensure_transaction_directories(
        home, current.st_uid, current.st_gid
    )
    plugins = home / "plugins"
    plugins.rename(home / "plugins-original")
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "nix-managed-keep"
    marker.symlink_to(tmp_path / "keep-target")
    plugins.symlink_to(outside, target_is_directory=True)
    manifest = tmp_path / "plugins.json"
    manifest.write_text("[]")

    home_fd = safe_profile_dirs._open_verified_home(home, identity)
    try:
        with pytest.raises(OSError):
            safe_profile_dirs._sync_plugin_links(
                home_fd, manifest, current.st_uid, current.st_gid
            )
    finally:
        os.close(home_fd)

    assert marker.is_symlink()


def test_nixos_module_uses_nofollow_helper_for_transaction_directories():
    source = (Path(__file__).parents[2] / "nix" / "nixosModules.nix").read_text()

    assert "python3 ${safeProfileDirs}" in source
    assert "mkdir -p ${profileBackupsShell}" not in source
    assert "chown ${profileOwnerShell}" not in source
    assert 'memories/.imports 2770' not in source
    assert 'mkdir -p ${cfg.stateDir}/.hermes' not in source
    assert '"d ${cfg.stateDir}/.hermes' not in source
    assert "--recursive-ownership" in source
    assert "--shared-file-modes" in source
    assert 'find "$HERMES_HOME"' not in source
    assert "stat -c %u ${profileHomeShell}" not in source
    assert "stat -c %g ${profileHomeShell}" not in source
    assert source.count("--trust-anchor /run/hermes-agent/profile-trust") == 2
    assert "--expected-identity \"$PROFILE_IDENTITY\"" in source
    assert "--sync-plugins-manifest ${managedPluginsManifest}" in source
    assert "find ${cfg.stateDir}/.hermes/plugins" not in source
    assert "ln -sfn ${plugin}" not in source
    assert "chown -h ${cfg.user}:${cfg.group} ${cfg.stateDir}/.hermes/plugins" not in source
    for leaf in ("config.yaml", ".managed", ".container-mode", "auth.json", ".env"):
        assert f"--write-leaf {leaf}" in source or f"--remove-leaf {leaf}" in source
    assert "touch ${cfg.stateDir}/.hermes/.managed" not in source
    assert "cat > ${cfg.stateDir}/.hermes/.container-mode" not in source
    assert "install -o ${cfg.user} -g ${cfg.group} -m 0600 ${cfg.authFile}" not in source


def test_container_gid_uses_configured_group_for_host_users():
    source = (Path(__file__).parents[2] / "nix" / "nixosModules.nix").read_text()

    configured_gid = "getent group ${lib.escapeShellArg cfg.group}"
    assert source.count(configured_gid) >= 2
    assert "id -g ${cfg.user}" not in source
    assert "users.users = lib.genAttrs cfg.container.hostUsers" in source
    assert "schema = 6" in source
    assert "user = cfg.user" in source
    assert "group = cfg.group" in source
    assert 'EXPECTED_CONTAINER_IDENTITY="${containerIdentity}:$HERMES_UID:$HERMES_GID"' in source
    assert '"$(cat ${identityFile})" != "$EXPECTED_CONTAINER_IDENTITY"' in source
    assert 'echo "$EXPECTED_CONTAINER_IDENTITY" > ${identityFile}' in source
