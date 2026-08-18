import fcntl
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


_ENV_FILE_SIZE_LIMIT = 64 * 1024
_DEVICE_DATA_DIR_ASSIGNMENT = (
    'DATA_DIR="/volume1/system/zettos-main-data/com.zettlab.claw"'
)


def _use_fixture_data_dir(source: str, data_dir: Path) -> str:
    assert _DEVICE_DATA_DIR_ASSIGNMENT in source
    return source.replace(
        _DEVICE_DATA_DIR_ASSIGNMENT,
        f"DATA_DIR={shlex.quote(str(data_dir))}",
    )


def _readlink_f_available(tmp_path: Path) -> bool:
    return subprocess.run(
        ["readlink", "-f", str(tmp_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _prepare_script_fixture(
    tmp_path: Path,
    *,
    default_presets_dir: Path | None = None,
    protected_presets_root: Path | None = None,
) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    app_base = tmp_path / "zettos" / "main" / "apps" / "com.zettlab.claw"
    app_root = app_base / "current"
    script = app_root / "prepare-claw-service.sh"
    python_path = app_root / "lib" / "hermes-agent" / "venv" / "bin" / "python"
    hermes_path = python_path.with_name("hermes")
    hermes_wrapper_path = app_root / "bin" / "hermes"
    invocation_log = app_root / "hermes-invocations.jsonl"
    data_dir = (
        tmp_path
        / "volume1"
        / "system"
        / "zettos-main-data"
        / "com.zettlab.claw"
    )
    hermes_home = data_dir / "hermes_home"
    env_path = data_dir / "secrets" / "zettlab-claw.env"

    python_path.parent.mkdir(parents=True)
    os.symlink(sys.executable, python_path)
    # Packaged console scripts can retain their staging-venv shebang after the
    # slot is moved into place. The public zpk wrapper deliberately invokes the
    # script through the current slot's Python, so keep this shebang invalid to
    # ensure prepare-claw-service.sh uses that wrapper too.
    hermes_path.write_text(
        f"""#!/nonexistent/staging-venv/bin/python
import json
import os
import runpy
import sys
import time
from pathlib import Path

Path({str(invocation_log)!r}).open("a", encoding="utf-8").write(
    json.dumps(sys.argv[1:]) + "\\n"
)
if delay := os.environ.get("HERMES_TEST_HERMES_DELAY"):
    time.sleep(float(delay))
sys.path.insert(0, {str(repo_root)!r})
runpy.run_module("hermes_cli.main", run_name="__main__")
""",
        encoding="utf-8",
    )
    hermes_path.chmod(0o755)
    hermes_wrapper_path.parent.mkdir(parents=True)
    shutil.copy2(repo_root / "zpk" / "bin" / "hermes", hermes_wrapper_path)
    hermes_wrapper_path.chmod(0o755)
    prepare_source = (repo_root / "zpk" / "prepare-claw-service.sh").read_text(
        encoding="utf-8"
    )
    prepare_source = _use_fixture_data_dir(prepare_source, data_dir)
    legacy_data_dir = tmp_path / "legacy-claw-data"
    legacy_assignment = 'LEGACY_DATA_DIR="/zettos/main/apps/com.zettlab.claw/data"'
    assert legacy_assignment in prepare_source
    prepare_source = prepare_source.replace(
        legacy_assignment,
        f"LEGACY_DATA_DIR={shlex.quote(str(legacy_data_dir))}",
    )
    protected_root = protected_presets_root or tmp_path
    for assignment in (
        'SUBVOLUME_ZETTLAB_PRESETS_ROOT="/volume1/subvol/agents/zettlab-presets"',
        'AGENTS_ZETTLAB_PRESETS_ROOT="/volume1/agents/zettlab-presets"',
    ):
        assert assignment in prepare_source
        variable = assignment.split("=", 1)[0]
        prepare_source = prepare_source.replace(
            assignment,
            f"{variable}={shlex.quote(str(protected_root))}",
        )
    if default_presets_dir is not None:
        default_assignment = (
            'SUBVOLUME_ZETTLAB_PRESETS_DIR='
            '"$SUBVOLUME_ZETTLAB_PRESETS_ROOT/current"'
        )
        assert default_assignment in prepare_source
        prepare_source = prepare_source.replace(
            default_assignment,
            f"SUBVOLUME_ZETTLAB_PRESETS_DIR="
            f"{shlex.quote(str(default_presets_dir))}",
        )
    script.write_text(prepare_source, encoding="utf-8")
    script.chmod(0o755)
    shutil.copy2(
        repo_root / "zpk" / "parse-environment-file.py",
        app_root / "parse-environment-file.py",
    )
    return app_root, hermes_home, env_path


def _script_env(**overrides: str) -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.pop("ZETTLAB_PRESETS_DIR", None)
    env.pop("ZETTLAB_CLAW_PRESETS_DIR", None)
    env.pop("HERMES_MANAGED_DIR", None)
    env.pop("HERMES_MANAGED_GATEWAY", None)
    env.pop("HERMES_MANAGED_CGROUP_ROOT", None)
    env.pop("HERMES_MANAGED_CGROUP_UNIT", None)
    # The fixture's packaged Python must import this checkout, not an unrelated
    # editable Hermes installation that happens to exist in the test venv.
    env["PYTHONPATH"] = str(repo_root)
    env.update(overrides)
    return env


def test_environment_file_parser_matches_systemd_quoting_rules(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    env_path = tmp_path / "service.env"
    env_path.write_text(
        "# ignored\n"
        "UNQUOTED=  one\\ two  \n"
        'RAW=one"two"\n'
        "SINGLE='line one\nline two'\n"
        'DOUBLE="literal\\nvalue \\$HOME \\\\ end"\n'
        'FRAGMENTED="foo"\' bar\'\n'
        'QUOTED_THEN_RAW="foo"bar\n'
        "CONTINUED=one\\\ntwo\n"
        "SPACE_CONTINUED=one  \\\ntwo\n"
        "DUPLICATE=old\n"
        "DUPLICATE=new\n"
        "# continued comment \\\n"
        "COMMENTED=hidden\n"
        "UNCLOSED='accepted at eof",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "zpk" / "parse-environment-file.py"),
            str(env_path),
        ],
        check=True,
        capture_output=True,
    )
    fields = result.stdout.split(b"\0")
    assert fields[-3:] == [b"", b"", b""]
    parsed = {
        fields[index].decode(): fields[index + 1].decode()
        for index in range(0, len(fields) - 3, 2)
    }

    assert parsed == {
        "UNQUOTED": "one two",
        "RAW": 'one"two"',
        "SINGLE": "line one\nline two",
        "DOUBLE": r"literal\nvalue $HOME \ end",
        "FRAGMENTED": "foo bar",
        "QUOTED_THEN_RAW": "foobar",
        "CONTINUED": "onetwo",
        "SPACE_CONTINUED": "one  two",
        "DUPLICATE": "new",
        "UNCLOSED": "accepted at eof",
    }

    env_path.write_bytes(b'CR_ONE=one\rCR_TWO="two"\r')
    cr_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "zpk" / "parse-environment-file.py"),
            str(env_path),
        ],
        check=True,
        capture_output=True,
    )
    assert cr_result.stdout.split(b"\0") == [
        b"CR_ONE",
        b"one",
        b"CR_TWO",
        b"two",
        b"",
        b"",
        b"",
    ]

    env_path.write_bytes(b"# comment \\\r\nCRLF_NEXT=visible\r\n")
    crlf_result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "zpk" / "parse-environment-file.py"),
            str(env_path),
        ],
        check=True,
        capture_output=True,
    )
    assert crlf_result.stdout.split(b"\0") == [
        b"CRLF_NEXT",
        b"visible",
        b"",
        b"",
        b"",
    ]


def test_prepare_claw_service_leaves_config_untouched_and_removes_legacy_multiplex_env(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    original_config = "gateway:\n    multiplex_profiles: false\ninvalid: [\n"
    config_path.write_text(original_config, encoding="utf-8")
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "GATEWAY_MULTIPLEX_PROFILES=false\nCUSTOM_SAFE=keep-me\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(GATEWAY_MULTIPLEX_PROFILES="false"),
    )

    assert config_path.read_text(encoding="utf-8") == original_config
    env_text = env_path.read_text(encoding="utf-8")
    assert "GATEWAY_MULTIPLEX_PROFILES=" not in env_text
    assert "CUSTOM_SAFE=keep-me\n" in env_text
    assert not (app_root / "hermes-invocations.jsonl").exists()


def test_prepare_claw_service_does_not_require_hermes_cli(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    (app_root / "bin" / "hermes").unlink()

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert env_path.is_file()


def test_prepare_claw_service_fails_closed_on_oversized_environment_file(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    original = b"CUSTOM_SAFE=" + b"x" * _ENV_FILE_SIZE_LIMIT
    env_path.write_bytes(original)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "safety limit" in result.stderr
    assert env_path.read_bytes() == original


def test_prepare_claw_service_does_not_generate_oversized_environment_file(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    prefix = b"CUSTOM_SAFE="
    original = prefix + b"x" * (_ENV_FILE_SIZE_LIMIT - len(prefix))
    env_path.write_bytes(original)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "safety limit" in result.stderr
    assert "refusing invalid generated environment file" in result.stderr
    assert env_path.read_bytes() == original
    assert list(env_path.parent.glob("zettlab-claw.env.tmp.*")) == []


def test_prepare_claw_service_respects_presets_dir_override(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)

    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text


def test_prepare_claw_service_rejects_presets_override_outside_protected_roots(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    protected_root = tmp_path / "protected-presets"
    protected_root.mkdir()
    app_root, _hermes_home, env_path = _prepare_script_fixture(
        tmp_path,
        protected_presets_root=protected_root,
    )
    unprotected = tmp_path / "operator-presets"
    unprotected.mkdir()

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(unprotected)),
    )

    assert "ZETTLAB_PRESETS_DIR=" not in env_path.read_text(encoding="utf-8")


def test_prepare_claw_service_preserves_presets_selection_across_version_flips(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    version_one = tmp_path / "presets" / "v0.7.12"
    version_two = version_one.parent / "v0.7.13"
    version_one.mkdir(parents=True)
    version_two.mkdir()
    current = version_one.parent / "current"
    os.symlink(version_one.name, current)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(current)),
    )

    assert f"ZETTLAB_PRESETS_DIR={current}\n" in env_path.read_text(
        encoding="utf-8"
    )

    current.unlink()
    os.symlink(version_two.name, current)
    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert current.resolve() == version_two
    assert f"ZETTLAB_PRESETS_DIR={current}\n" in env_path.read_text(
        encoding="utf-8"
    )
    assert str(version_one) not in env_path.read_text(encoding="utf-8")


def test_prepare_claw_service_preserves_existing_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets" / "v 0.7.12"
    presets_dir.mkdir(parents=True)
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f'ZETTLAB_PRESETS_DIR="{presets_dir}"\nCUSTOM_SAFE=keep-me\n',
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text
    assert "CUSTOM_SAFE=keep-me\n" in env_text


def test_prepare_claw_service_existing_presets_dir_beats_available_default(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    default_dir = tmp_path / "presets" / "default"
    existing_dir = tmp_path / "presets" / "custom"
    default_dir.mkdir(parents=True)
    existing_dir.mkdir()
    app_root, _hermes_home, env_path = _prepare_script_fixture(
        tmp_path,
        default_presets_dir=default_dir,
    )
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f"ZETTLAB_PRESETS_DIR={existing_dir}\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={existing_dir}\n" in env_text
    assert f"ZETTLAB_PRESETS_DIR={default_dir}\n" not in env_text


def test_prepare_claw_service_distinct_presets_override_beats_existing_value(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    old_dir = tmp_path / "presets" / "v0.7.11"
    new_dir = tmp_path / "presets" / "v0.7.12"
    old_dir.mkdir(parents=True)
    new_dir.mkdir()
    env_path.parent.mkdir(parents=True)
    env_path.write_text(f"ZETTLAB_PRESETS_DIR={old_dir}\n", encoding="utf-8")

    for _ in range(2):
        subprocess.run(
            [str(app_root / "prepare-claw-service.sh")],
            check=True,
            cwd=str(app_root),
            env=_script_env(ZETTLAB_PRESETS_DIR=str(new_dir)),
        )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={new_dir}\n" in env_text
    assert f"ZETTLAB_PRESETS_DIR={old_dir}\n" not in env_text


def test_prepare_claw_service_does_not_replace_unchanged_env_file(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    command = [str(app_root / "prepare-claw-service.sh")]

    subprocess.run(
        command,
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )
    first_stat = env_path.stat()
    first_content = env_path.read_bytes()
    key_path = env_path.with_name("zet_agent.key")
    env_path.chmod(0o644)
    key_path.chmod(0o644)

    subprocess.run(
        command,
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    second_stat = env_path.stat()
    assert env_path.read_bytes() == first_content
    assert second_stat.st_ino == first_stat.st_ino
    assert second_stat.st_mtime_ns == first_stat.st_mtime_ns
    assert second_stat.st_mode & 0o777 == 0o600
    assert key_path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("default", b"deleted\n"),
        ("config.yaml", b"deleted\ncleanup-complete\n"),
        (
            "removed-agent",
            b'deleted\norigin-json\n{"import_job_id":"job-1"}\n',
        ),
        (
            "completed-agent",
            b'deleted\ncleanup-complete\norigin-json\n'
            b'{"clone_history_id":"clone-1"}\n',
        ),
    ],
)
def test_prepare_claw_service_preserves_local_server_deletion_blockers(
    tmp_path: Path,
    name: str,
    body: bytes,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    blocker = profiles_root / name
    blocker.write_bytes(body)
    blocker.chmod(0o644)
    if name == "default":
        tombstones = profiles_root / ".deleted-agents"
        tombstones.mkdir()
        sidecar = tombstones / name
        sidecar.write_bytes(b"deleted\ncleanup-complete\n")
        sidecar.chmod(0o600)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert blocker.read_bytes() == body
    assert blocker.stat().st_mode & 0o777 == 0o600
    assert (env_path.parent / "profile-permissions-v2.done").is_file()


@pytest.mark.parametrize(
    "body",
    [
        b"not-a-deletion-blocker\n",
        b"deleted\nunexpected-tail\n",
        b"deleted\norigin-json\nnot-json\n",
        b'deleted\norigin-json\n{"value":NaN}\n',
        pytest.param(
            b'deleted\norigin-json\n{"value":'
            + (b"[" * 30000)
            + b"0"
            + (b"]" * 30000)
            + b"}\n",
            id="deeply-nested-origin-json",
        ),
    ],
)
def test_prepare_claw_service_rejects_arbitrary_profile_root_files(
    tmp_path: Path,
    body: bytes,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    unexpected = profiles_root / "unexpected"
    unexpected.write_bytes(body)
    unexpected.chmod(0o600)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-directory profile path" in result.stderr
    assert unexpected.read_bytes() == body
    assert not (env_path.parent / "profile-permissions-v2.done").exists()


@pytest.mark.parametrize(
    ("name", "sidecar_body"),
    [
        ("default", None),
        ("default", b"not-a-deletion-marker\n"),
        ("main", b"deleted\ncleanup-complete\n"),
        (".hidden", b"deleted\ncleanup-complete\n"),
    ],
)
def test_prepare_claw_service_rejects_unproven_or_reserved_deletion_blockers(
    tmp_path: Path,
    name: str,
    sidecar_body: bytes | None,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    blocker = profiles_root / name
    blocker.write_bytes(b"deleted\n")
    blocker.chmod(0o600)
    if sidecar_body is not None:
        tombstones = profiles_root / ".deleted-agents"
        tombstones.mkdir()
        sidecar = tombstones / name
        sidecar.write_bytes(sidecar_body)
        sidecar.chmod(0o600)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-directory profile path" in result.stderr
    assert blocker.read_bytes() == b"deleted\n"
    assert not (env_path.parent / "profile-permissions-v2.done").exists()


def test_prepare_claw_service_rejects_symlinked_deletion_blocker(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    target = tmp_path / "outside-blocker"
    target.write_bytes(b"deleted\n")
    target.chmod(0o644)
    os.symlink(target, profiles_root / "removed-agent")

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-directory profile path" in result.stderr
    assert target.read_bytes() == b"deleted\n"
    assert target.stat().st_mode & 0o777 == 0o644
    assert not (env_path.parent / "profile-permissions-v2.done").exists()


def test_prepare_claw_service_rejects_symlinked_deletion_sidecar_root(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    profiles_root.mkdir(parents=True)
    blocker = profiles_root / "default"
    blocker.write_bytes(b"deleted\n")
    blocker.chmod(0o600)

    outside_root = tmp_path / "outside-deleted-agents"
    outside_root.mkdir()
    outside_sidecar = outside_root / "default"
    outside_sidecar.write_bytes(b"deleted\ncleanup-complete\n")
    outside_sidecar.chmod(0o644)
    os.symlink(outside_root, profiles_root / ".deleted-agents")

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-directory profile path" in result.stderr
    assert blocker.read_bytes() == b"deleted\n"
    assert outside_sidecar.read_bytes() == b"deleted\ncleanup-complete\n"
    assert outside_sidecar.stat().st_mode & 0o777 == 0o644
    assert not (env_path.parent / "profile-permissions-v2.done").exists()


@pytest.mark.parametrize("name", ["main", "default"])
def test_prepare_claw_service_checks_reserved_blockers_outside_bounded_scan(
    tmp_path: Path,
    name: str,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    (profiles_root / "agent-a").mkdir(parents=True)
    (profiles_root / name).write_bytes(b"deleted\n")

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(HERMES_PROFILE_PERMISSION_MIGRATION_MAX_PROFILES="1"),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"refusing non-directory profile path: {profiles_root / name}" in result.stderr
    assert not (env_path.parent / "profile-permissions-v2.done").exists()


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root-owned profile tree")
def test_prepare_claw_service_uses_root_barrier_and_one_time_migration(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    nested = hermes_home / "profiles" / "agent-a" / "state"
    nested.mkdir(parents=True)
    state_file = nested / "history.json"
    state_file.write_text("{}\n", encoding="utf-8")
    for path in (hermes_home / "profiles", nested.parent, nested):
        path.chmod(0o777)
    state_file.chmod(0o666)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    for path in (hermes_home / "profiles", nested.parent):
        assert path.stat().st_mode & 0o077 == 0
    assert nested.stat().st_mode & 0o077 != 0
    assert state_file.stat().st_mode & 0o077 != 0

    marker = env_path.parent / "profile-permissions-v2.done"
    first_marker = marker.stat()
    assert first_marker.st_mode & 0o777 == 0o600

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )
    second_marker = marker.stat()
    assert second_marker.st_ino == first_marker.st_ino
    assert second_marker.st_mtime_ns == first_marker.st_mtime_ns


@pytest.mark.skipif(os.geteuid() != 0, reason="requires root-owned profile tree")
def test_prepare_claw_service_caps_legacy_profile_scan_without_blocking_start(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    profiles_root = hermes_home / "profiles"
    for name in ("agent-a", "agent-b"):
        (profiles_root / name).mkdir(parents=True)

    env = _script_env()
    env["HERMES_PROFILE_PERMISSION_MIGRATION_MAX_PROFILES"] = "1"
    completed = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=env,
        text=True,
        capture_output=True,
    )

    assert "bounded legacy profile migration stopped after 1 profiles" in completed.stderr
    assert profiles_root.stat().st_mode & 0o777 == 0o700
    assert (env_path.parent / "profile-permissions-v2.done").is_file()


def test_prepare_claw_service_removes_complete_multiline_package_assignments(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        "export BAD=x\n"
        "BAD-NAME=y\n"
        "CUSTOM_SAFE='keep\nthis'\n"
        "ZET_AGENT_KEY='stale\nEVIL=1'\n"
        "GATEWAY_MULTIPLEX_PROFILES='false\nEVIL_MULTIPLEX=1'\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "export BAD=x\n" in env_text
    assert "BAD-NAME=y\n" in env_text
    assert "CUSTOM_SAFE='keep\nthis'\n" in env_text
    assert "stale" not in env_text
    assert "EVIL=1" not in env_text
    assert "EVIL_MULTIPLEX=1" not in env_text
    assert env_text.count("ZET_AGENT_KEY=") == 1
    assert "GATEWAY_MULTIPLEX_PROFILES=" not in env_text


@pytest.mark.parametrize(
    "custom_tail",
    ["CUSTOM='unterminated", 'CUSTOM="unterminated', "CUSTOM=trailing\\"],
)
def test_prepare_claw_service_keeps_package_env_before_eof_unclosed_user_value(
    tmp_path: Path,
    custom_tail: str,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    env_path.write_text(custom_tail, encoding="utf-8")

    dumped = subprocess.run(
        [str(app_root / "prepare-claw-service.sh"), "--emit-env"],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
    ).stdout

    assert b"ZET_AGENT_KEY\0" in dumped
    assert b"ZET_AGENT_ENABLED\0true\0" in dumped
    assert b"GATEWAY_MULTIPLEX_PROFILES\0" not in dumped
    assert env_path.read_text(encoding="utf-8").startswith("ZET_AGENT_KEY=")


def test_prepare_claw_service_filters_package_assignments_in_cr_only_env(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    env_path.write_bytes(
        b"ZET_AGENT_KEY=stale\r"
        b"GATEWAY_MULTIPLEX_PROFILES=false\r"
        b"CUSTOM_SAFE=keep\r"
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    env_bytes = env_path.read_bytes()
    assert b"stale" not in env_bytes
    assert b"CUSTOM_SAFE=keep\r" in env_bytes
    assert env_bytes.count(b"ZET_AGENT_KEY=") == 1
    assert b"GATEWAY_MULTIPLEX_PROFILES=" not in env_bytes


def test_prepare_claw_service_removes_legacy_langfuse_credentials(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    legacy_data = tmp_path / "legacy-claw-data"
    paths = [
        env_path,
        hermes_home / ".env",
        hermes_home / "profiles" / "main" / ".env",
        legacy_data / "secrets" / "zettlab-claw.env",
        legacy_data / "hermes_home" / ".env",
        legacy_data / "hermes_home" / "profiles" / "legacy" / ".env",
    ]
    body = (
        "SAFE_VALUE=keep\n"
        "HERMES_LANGFUSE_MODE=relay\n"
        "HERMES_LANGFUSE_BASE_URL=http://127.0.0.1:19092\n"
        "HERMES_LANGFUSE_PUBLIC_KEY=legacy-public\n"
        "HERMES_LANGFUSE_SECRET_KEY=legacy-secret\n"
        "LANGFUSE_PUBLIC_KEY=legacy-global-public\n"
        "LANGFUSE_SECRET_KEY=legacy-global-secret\n"
        "LANGFUSE_BASIC_AUTH='Basic YTpi'\n"
        "LANGFUSE_OTEL_TRACES_EXPORT_PATH=wrong/path\n"
    )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    for path in paths:
        env_text = path.read_text(encoding="utf-8")
        assert "SAFE_VALUE=keep" in env_text
        assert "HERMES_LANGFUSE_MODE=relay" in env_text
        assert "HERMES_LANGFUSE_BASE_URL=http://127.0.0.1:19092" in env_text
        assert "LANGFUSE_PUBLIC_KEY" not in env_text
        assert "LANGFUSE_SECRET_KEY" not in env_text
        assert "LANGFUSE_BASIC_AUTH" not in env_text
        assert "LANGFUSE_OTEL_TRACES_EXPORT_PATH" not in env_text
        assert path.stat().st_mode & 0o777 == 0o600


def test_prepare_claw_service_refuses_symlinked_langfuse_env(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.env"
    original = "LANGFUSE_SECRET_KEY=must-not-be-followed\n"
    outside.write_text(original, encoding="utf-8")
    (hermes_home / ".env").symlink_to(outside)

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [str(app_root / "prepare-claw-service.sh")],
            check=True,
            cwd=str(app_root),
            env=_script_env(),
        )

    assert outside.read_text(encoding="utf-8") == original


def test_prepare_claw_service_refuses_symlinked_profile_langfuse_env(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    profile = hermes_home / "profiles" / "main"
    profile.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside-profile.env"
    original = "LANGFUSE_SECRET_KEY=must-not-be-followed\n"
    outside.write_text(original, encoding="utf-8")
    (profile / ".env").symlink_to(outside)

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            [str(app_root / "prepare-claw-service.sh")],
            check=True,
            cwd=str(app_root),
            env=_script_env(),
        )

    assert outside.read_text(encoding="utf-8") == original


def test_prepare_claw_service_preserves_restrictive_data_directory_mode(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_dir = _hermes_home.parent
    data_dir.mkdir(parents=True)
    data_dir.chmod(0o700)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert data_dir.stat().st_mode & 0o777 == 0o700


def test_service_direct_data_layout_does_not_require_systemd_write_carveouts(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    data_path = hermes_home.parent

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    service = (
        Path(__file__).resolve().parents[2]
        / "zpk"
        / "init.d"
        / "zettlab-claw.service"
    ).read_text(encoding="utf-8")
    assert "ProtectSystem=" not in service
    assert "ReadOnlyPaths=" not in service
    assert "ReadWritePaths=" not in service
    probe = data_path / "service-write-boundary-probe"
    probe.write_text("writable\n", encoding="utf-8")
    assert probe.read_text(encoding="utf-8") == "writable\n"
    assert env_path.is_file()
    assert not (app_root.parent / "data").exists()


def test_r2_runtime_scripts_use_only_direct_system_data_path():
    repo_root = Path(__file__).resolve().parents[2]
    runtime_files = [
        repo_root / "zpk" / "prepare-claw-service.sh",
        repo_root / "zpk" / "start-claw-service.sh",
        repo_root / "zpk" / "bin" / "hermes",
        repo_root / "zpk" / "init.d" / "zettlab-claw.service",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in runtime_files)
    assert "/volume1/system/zettos-main-data/com.zettlab.claw" in text
    assert "ZETTLAB_CLAW_DATA_DIR" not in text
    assert "RequiresMountsFor=/volume1" in text
    assert "/volume1/subvol/apps" not in text
    assert "__APP_BASE__/data" not in text
    assert 'DATA_DIR="$APP_BASE/data"' not in text


@pytest.mark.parametrize("managed_name", ["zet_agent.key", "zettlab-claw.env"])
def test_prepare_claw_service_refuses_symlinked_managed_secret_files(
    tmp_path: Path,
    managed_name: str,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    secret_dir = env_path.parent
    secret_dir.mkdir(parents=True)
    key_path = secret_dir / "zet_agent.key"
    if managed_name != key_path.name:
        key_path.write_text("a" * 64 + "\n", encoding="utf-8")
        key_path.chmod(0o600)

    target = tmp_path / f"{managed_name}.target"
    target.write_text("do-not-touch\n", encoding="utf-8")
    target.chmod(0o644)
    os.symlink(target, secret_dir / managed_name)
    original_mode = target.stat().st_mode & 0o777

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-regular" in result.stderr
    assert target.read_text(encoding="utf-8") == "do-not-touch\n"
    assert target.stat().st_mode & 0o777 == original_mode


@pytest.mark.parametrize("state_path", ["data", "secrets"])
def test_prepare_claw_service_refuses_symlinked_state_directories(
    tmp_path: Path,
    state_path: str,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_dir = _hermes_home.parent
    target = tmp_path / f"{state_path}.target"
    target.mkdir()
    target.chmod(0o777)
    (target / "marker").write_text("do-not-touch\n", encoding="utf-8")

    if state_path == "data":
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, data_dir)
    else:
        data_dir.mkdir(parents=True)
        os.symlink(target, data_dir / "secrets")

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing non-directory state path" in result.stderr
    assert (target / "marker").read_text(encoding="utf-8") == "do-not-touch\n"
    assert target.stat().st_mode & 0o777 == 0o777


def test_prepare_claw_service_waits_for_cross_process_lock(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n",
        encoding="utf-8",
    )
    secret_dir = env_path.parent
    secret_dir.mkdir(parents=True)
    lock_path = secret_dir / "prepare-claw-service.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        process = subprocess.Popen(
            [str(app_root / "prepare-claw-service.sh")],
            cwd=str(app_root),
            env=_script_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(2)
        assert process.poll() is None
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        stdout, stderr = process.communicate(timeout=30)

    assert process.returncode == 0, (stdout, stderr)


def test_prepare_claw_service_times_out_waiting_for_cross_process_lock(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    secret_dir = env_path.parent
    secret_dir.mkdir(parents=True)
    lock_path = secret_dir / "prepare-claw-service.lock"

    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        started = time.monotonic()
        result = subprocess.run(
            [str(app_root / "prepare-claw-service.sh")],
            check=False,
            cwd=str(app_root),
            env=_script_env(HERMES_PREPARE_LOCK_TIMEOUT_SECONDS="0.2"),
            capture_output=True,
            text=True,
            timeout=5,
        )
        elapsed = time.monotonic() - started

    assert result.returncode != 0
    assert elapsed < 2
    assert "timed out waiting for Claw prepare lock" in result.stderr


def test_prepare_claw_service_concurrent_runs_keep_key_and_env_consistent(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    process_env = _script_env(HERMES_TEST_HERMES_DELAY="1")
    processes = [
        subprocess.Popen(
            [str(app_root / "prepare-claw-service.sh")],
            cwd=str(app_root),
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=30) for process in processes]

    for process, output in zip(processes, results):
        assert process.returncode == 0, output
    env_text = env_path.read_text(encoding="utf-8")
    assert "GATEWAY_MULTIPLEX_PROFILES=" not in env_text
    assert not (app_root / "hermes-invocations.jsonl").exists()
    key = env_path.with_name("zet_agent.key").read_text(encoding="utf-8").strip()
    assert f"ZET_AGENT_KEY={key}\n" in env_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("with_explicit_override", [False, True])
def test_start_claw_service_loads_reconciled_env_without_overriding_explicit(
    tmp_path: Path,
    with_explicit_override: bool,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("start-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    start_script = app_root / "start-claw-service.sh"
    start_source = (repo_root / "zpk" / "start-claw-service.sh").read_text(
        encoding="utf-8"
    )
    start_script.write_text(
        _use_fixture_data_dir(start_source, hermes_home.parent),
        encoding="utf-8",
    )
    start_script.chmod(0o755)

    hermes_home.mkdir(parents=True)
    (hermes_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    persisted_managed = tmp_path / "managed persisted"
    explicit_managed = tmp_path / "managed-explicit"
    for managed_dir in (persisted_managed, explicit_managed):
        managed_dir.mkdir()
        (managed_dir / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles: true\n",
            encoding="utf-8",
        )
    persisted_presets = tmp_path / "presets-persisted"
    explicit_presets = tmp_path / "presets-explicit"
    persisted_presets.mkdir()
    explicit_presets.mkdir()

    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f'HERMES_MANAGED_DIR="{persisted_managed}"\n'
        "HERMES_HOME=/stale/hermes-home\n"
        "HERMES_BUNDLED_SKILLS=/stale/skills\n"
        "HERMES_BUNDLED_PLUGINS=/stale/plugins\n"
        "HERMES_BUNDLED_LOCALES=/stale/locales\n"
        "HERMES_LAZY_INSTALL_TARGET=/stale/lazy-packages\n"
        f"ZETTLAB_PRESETS_DIR={persisted_presets}\n"
        "GATEWAY_MULTIPLEX_PROFILES=false\n"
        "HERMES_MANAGED_GATEWAY=0\n"
        "HERMES_MANAGED_CGROUP_ROOT=/stale/cgroup\n"
        "HERMES_MANAGED_CGROUP_UNIT=stale.service\n"
        "HERMES_LANGFUSE_MODE=relay\n"
        "HERMES_LANGFUSE_BASE_URL=http://127.0.0.1:19092\n"
        "HERMES_LANGFUSE_PUBLIC_KEY=legacy-public\n"
        "HERMES_LANGFUSE_SECRET_KEY=legacy-secret\n"
        "LANGFUSE_PUBLIC_KEY=legacy-global-public\n"
        "LANGFUSE_SECRET_KEY=legacy-global-secret\n"
        'CUSTOM_SAFE="keep "\'me\'\n'
        r"CUSTOM_UNQUOTED=one\ two" "\n"
        r'CUSTOM_DOUBLE="literal\nvalue"' "\n",
        encoding="utf-8",
    )

    gateway_log = app_root / "gateway-env.json"
    packaged_locales = app_root / "lib" / "hermes-agent" / "locales"
    shutil.copytree(repo_root / "locales", packaged_locales)
    hermes_entry = app_root / "bin" / "hermes"
    hermes_entry.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, {str(repo_root)!r})
from agent.i18n import t

Path({str(gateway_log)!r}).write_text(
    json.dumps({{
        "argv": sys.argv[1:],
        "managed": os.environ.get("HERMES_MANAGED_DIR"),
        "custom": os.environ.get("CUSTOM_SAFE"),
        "unquoted": os.environ.get("CUSTOM_UNQUOTED"),
        "double": os.environ.get("CUSTOM_DOUBLE"),
        "multiplex": os.environ.get("GATEWAY_MULTIPLEX_PROFILES"),
        "managed_gateway": os.environ.get("HERMES_MANAGED_GATEWAY"),
        "relay_core": os.environ.get("HERMES_NEMO_RELAY_CORE_ENABLED"),
        "managed_cgroup_root": os.environ.get("HERMES_MANAGED_CGROUP_ROOT"),
        "managed_cgroup_unit": os.environ.get("HERMES_MANAGED_CGROUP_UNIT"),
        "home": os.environ.get("HERMES_HOME"),
        "skills": os.environ.get("HERMES_BUNDLED_SKILLS"),
        "plugins": os.environ.get("HERMES_BUNDLED_PLUGINS"),
        "locales": os.environ.get("HERMES_BUNDLED_LOCALES"),
        "rendered": t("gateway.reset.header_default", lang="en"),
        "lazy_target": os.environ.get("HERMES_LAZY_INSTALL_TARGET"),
        "presets": os.environ.get("ZETTLAB_PRESETS_DIR"),
        "presets_override": os.environ.get("ZETTLAB_CLAW_PRESETS_DIR"),
        "langfuse_mode": os.environ.get("HERMES_LANGFUSE_MODE"),
        "langfuse_base_url": os.environ.get("HERMES_LANGFUSE_BASE_URL"),
        "langfuse_public": os.environ.get("HERMES_LANGFUSE_PUBLIC_KEY"),
        "langfuse_secret": os.environ.get("HERMES_LANGFUSE_SECRET_KEY"),
        "langfuse_global_public": os.environ.get("LANGFUSE_PUBLIC_KEY"),
        "langfuse_global_secret": os.environ.get("LANGFUSE_SECRET_KEY"),
    }}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    hermes_entry.chmod(0o755)

    # Simulate a legacy shared EnvironmentFile value overriding the unit's
    # Environment= value before ExecStart. The wrapper must reassert true.
    overrides = {
        "GATEWAY_MULTIPLEX_PROFILES": "false",
        "HERMES_MANAGED_GATEWAY": "0",
        "HERMES_MANAGED_CGROUP_ROOT": "/stale/cgroup",
        "HERMES_MANAGED_CGROUP_UNIT": "stale.service",
        "HERMES_HOME": "/stale/hermes-home",
        "HERMES_BUNDLED_SKILLS": "/stale/skills",
        "HERMES_BUNDLED_PLUGINS": "/stale/plugins",
        "HERMES_BUNDLED_LOCALES": "/stale/locales",
        "HERMES_LAZY_INSTALL_TARGET": "/stale/lazy-packages",
        "ZETTLAB_CLAW_PRESETS_DIR": str(explicit_presets),
        "HERMES_LANGFUSE_PUBLIC_KEY": "operator-public",
        "HERMES_LANGFUSE_SECRET_KEY": "operator-secret",
        "LANGFUSE_PUBLIC_KEY": "operator-global-public",
        "LANGFUSE_SECRET_KEY": "operator-global-secret",
    }
    if with_explicit_override:
        overrides["HERMES_MANAGED_DIR"] = str(explicit_managed)
        overrides["HERMES_NEMO_RELAY_CORE_ENABLED"] = "true"
    subprocess.run(
        [str(start_script)],
        check=True,
        cwd=str(app_root),
        env=_script_env(**overrides),
    )

    gateway_env = json.loads(gateway_log.read_text(encoding="utf-8"))
    assert gateway_env["argv"] == ["gateway", "run", "--force", "--accept-hooks"]
    assert gateway_env["managed"] == str(
        explicit_managed if with_explicit_override else persisted_managed
    )
    assert gateway_env["custom"] == "keep me"
    assert gateway_env["unquoted"] == "one two"
    assert gateway_env["double"] == r"literal\nvalue"
    assert gateway_env["multiplex"] == "true"
    assert gateway_env["managed_gateway"] == "1"
    assert gateway_env["relay_core"] == (
        "true" if with_explicit_override else None
    )
    assert gateway_env["managed_cgroup_root"] is None
    assert gateway_env["managed_cgroup_unit"] == "zettlab-claw.service"
    assert gateway_env["home"] == str(hermes_home)
    assert gateway_env["skills"] == str(app_root / "lib" / "hermes-agent" / "skills")
    assert gateway_env["plugins"] == str(
        app_root / "lib" / "hermes-agent" / "plugins"
    )
    assert gateway_env["locales"] == str(packaged_locales)
    assert gateway_env["rendered"] != "gateway.reset.header_default"
    assert gateway_env["lazy_target"] == str(hermes_home.parent / "lazy-packages")
    assert gateway_env["presets"] == str(explicit_presets)
    assert gateway_env["presets_override"] is None
    assert gateway_env["langfuse_mode"] == "relay"
    assert gateway_env["langfuse_base_url"] == "http://127.0.0.1:19092"
    assert gateway_env["langfuse_public"] is None
    assert gateway_env["langfuse_secret"] is None
    assert gateway_env["langfuse_global_public"] is None
    assert gateway_env["langfuse_global_secret"] is None
    env_text = env_path.read_text(encoding="utf-8")
    for removed in (
        "GATEWAY_MULTIPLEX_PROFILES=",
        "HERMES_MANAGED_GATEWAY=",
        "HERMES_MANAGED_CGROUP_ROOT=",
        "HERMES_MANAGED_CGROUP_UNIT=",
        "HERMES_HOME=",
        "HERMES_BUNDLED_SKILLS=",
        "HERMES_BUNDLED_PLUGINS=",
        "HERMES_BUNDLED_LOCALES=",
        "HERMES_LAZY_INSTALL_TARGET=",
        "ZETTLAB_CLAW_PRESETS_DIR=",
        "HERMES_LANGFUSE_PUBLIC_KEY=",
        "HERMES_LANGFUSE_SECRET_KEY=",
        "LANGFUSE_PUBLIC_KEY=",
        "LANGFUSE_SECRET_KEY=",
    ):
        assert removed not in env_text
    assert not (app_root / "hermes-invocations.jsonl").exists()


def test_prepare_claw_service_omits_untrusted_world_writable_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)
    presets_dir.chmod(0o777)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    assert "ZETTLAB_PRESETS_DIR=" not in env_path.read_text(encoding="utf-8")


def test_prepare_claw_service_omits_untrusted_group_writable_presets_dir(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)
    presets_dir.chmod(0o770)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    assert "ZETTLAB_PRESETS_DIR=" not in env_path.read_text(encoding="utf-8")


def test_prepare_claw_service_rejects_newlines_in_presets_dir(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR="/custom/presets/current\nEVIL=1\r\n"),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "ZETTLAB_PRESETS_DIR=" not in env_text
    assert "\nEVIL=1" not in env_text


def test_prepare_claw_service_rejects_existing_presets_path_with_newline(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    presets_dir = tmp_path / "presets\nEVIL=1"
    presets_dir.mkdir()
    presets_dir.chmod(0o755)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert "ZETTLAB_PRESETS_DIR=" not in env_text
    assert "\nEVIL=1" not in env_text


def test_zpk_agent_service_names_are_device_facing():
    repo_root = Path(__file__).resolve().parents[2]

    service = (repo_root / "zpk" / "init.d" / "zettlab-claw.service").read_text(encoding="utf-8")
    start_wrapper = (repo_root / "zpk" / "start-claw-service.sh").read_text(encoding="utf-8")
    hermes_wrapper = (repo_root / "zpk" / "bin" / "hermes").read_text(
        encoding="utf-8"
    )
    package_meta = (repo_root / "zpk" / "package.meta").read_text(encoding="utf-8")
    install = (repo_root / "zpk" / "install.sh").read_text(encoding="utf-8")
    start = (repo_root / "zpk" / "init.d" / "start.sh").read_text(encoding="utf-8")
    stop = (repo_root / "zpk" / "init.d" / "stop.sh").read_text(encoding="utf-8")
    uninstall = (repo_root / "zpk" / "uninstall.sh").read_text(encoding="utf-8")

    meta = json.loads(package_meta)

    assert (
        "EnvironmentFile=-/volume1/system/zettos-main-data/"
        "com.zettlab.claw/secrets/zettlab-claw.env" in service
    )
    assert "RequiresMountsFor=/volume1" in service
    assert "Environment=GATEWAY_MULTIPLEX_PROFILES=true" in service
    assert "Environment=HERMES_MANAGED_GATEWAY=1" in service
    assert "HERMES_NEMO_RELAY_CORE_ENABLED" not in service
    assert (
        "Environment=HERMES_BUNDLED_LOCALES="
        "__APP_BASE__/current/lib/hermes-agent/locales" in service
    )
    assert (
        "Environment=HERMES_MANAGED_CGROUP_UNIT=zettlab-claw.service"
        in service
    )
    assert "ExecStart=__APP_BASE__/current/start-claw-service.sh" in service
    assert "NoNewPrivileges=true" in service
    assert "ProtectProc=invisible" in service
    assert "ProtectProc=default" not in service
    assert "CapabilityBoundingSet=~CAP_SYS_ADMIN" in service
    assert "CapabilityBoundingSet=~CAP_SYS_PTRACE" not in service
    assert "Delegate=pids memory" in service
    assert "KillMode=control-group" in service
    assert "ProtectSystem=" not in service
    assert "ReadOnlyPaths=" not in service
    assert "ReadWritePaths=" not in service
    assert 'HERMES_LAUNCHER="$APP_ROOT/libexec/hermes-secure-launcher.py"' in (
        hermes_wrapper
    )
    assert (
        'exec "$HERMES_PYTHON" -I "$HERMES_LAUNCHER" "$HERMES_SCRIPT" "$@"'
        in hermes_wrapper
    )
    assert (
        'export HERMES_BUNDLED_LOCALES="${HERMES_BUNDLED_LOCALES:-$HERMES_SRC/locales}"'
        in hermes_wrapper
    )
    assert "HERMES_NEMO_RELAY_CORE_ENABLED" not in hermes_wrapper
    assert '"$APP_ROOT/prepare-claw-service.sh" --emit-env' in start_wrapper
    assert "load_reconciled_env" in start_wrapper
    assert "export GATEWAY_MULTIPLEX_PROFILES=true" in start_wrapper
    assert "export HERMES_MANAGED_GATEWAY=1" in start_wrapper
    assert "HERMES_NEMO_RELAY_CORE_ENABLED" not in start_wrapper
    assert (
        'export HERMES_BUNDLED_LOCALES="$APP_ROOT/lib/hermes-agent/locales"'
        in start_wrapper
    )
    assert 'export HERMES_LAZY_INSTALL_TARGET="$DATA_DIR/lazy-packages"' in start_wrapper
    assert (
        "export HERMES_MANAGED_CGROUP_UNIT=zettlab-claw.service"
        in start_wrapper
    )
    assert "unset HERMES_MANAGED_CGROUP_ROOT" in start_wrapper
    assert '"$APP_ROOT/prepare-claw-service.sh"\n' not in start_wrapper
    assert ". \"$ENV_FILE\"" not in start_wrapper
    assert meta["service_name"] == "zettlab-claw"
    assert "restart" not in meta
    assert "systemctl restart zettlab-claw.service" not in install
    assert _DEVICE_DATA_DIR_ASSIGNMENT in install
    assert "ZETTLAB_CLAW_DATA_DIR" not in "\n".join(
        (install, start_wrapper, hermes_wrapper)
    )
    assert install.index('"$APP_ROOT/prepare-claw-service.sh"') < install.index(
        '"$APP_ROOT/bin/hermes" --version'
    )
    assert "systemctl start zettlab-claw.service" in install
    assert "systemctl start zettlab-claw.service" in start
    assert "systemctl stop zettlab-claw.service" in stop
    assert "zettlab-claw.service" in uninstall
    assert 'DATA_DIR="/volume1/system/zettos-main-data/$APP_ID"' in uninstall


def _load_secure_launcher():
    repo_root = Path(__file__).resolve().parents[2]
    launcher_path = (
        repo_root / "zpk" / "libexec" / "hermes-secure-launcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hermes_secure_launcher_for_test",
        launcher_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zpk_secure_launcher_is_nondumpable_without_managed_gateway(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "zpk" / "libexec" / "hermes-secure-launcher.py"
    entry_point = tmp_path / "hermes-entry"
    entry_point.write_text(
        """
import ctypes
import json
import sys

state = {"argv": sys.argv[1:]}
if sys.platform.startswith("linux"):
    libc = ctypes.CDLL(None, use_errno=True)
    state["dumpable"] = libc.prctl(3, 0, 0, 0, 0)
    state["no_new_privs"] = libc.prctl(39, 0, 0, 0, 0)
print(json.dumps(state))
""".lstrip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.pop("HERMES_MANAGED_GATEWAY", None)
    env.pop("HERMES_MANAGED_CGROUP_ROOT", None)
    env.pop("HERMES_MANAGED_CGROUP_UNIT", None)
    inherited_no_new_privs = None
    if sys.platform.startswith("linux"):
        import ctypes

        inherited_no_new_privs = ctypes.CDLL(None, use_errno=True).prctl(
            39, 0, 0, 0, 0
        )
    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), str(entry_point), "gateway", "run"],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    state = json.loads(completed.stdout)
    assert state["argv"] == ["gateway", "run"]
    if sys.platform.startswith("linux"):
        assert state["dumpable"] == 0
        assert state["no_new_privs"] == inherited_no_new_privs


def test_zpk_secure_launcher_loads_packaged_site_packages_while_ignoring_pythonpath(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "zpk" / "libexec" / "hermes-secure-launcher.py"
    venv_dir = tmp_path / "venv"
    bin_dir = venv_dir / "bin"
    site_packages = (
        venv_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    attacker_dir = tmp_path / "attacker"
    bin_dir.mkdir(parents=True)
    site_packages.mkdir(parents=True)
    attacker_dir.mkdir()

    (site_packages / "isolated_probe.py").write_text(
        'ORIGIN = "packaged"\n',
        encoding="utf-8",
    )
    (attacker_dir / "isolated_probe.py").write_text(
        'ORIGIN = "external"\n',
        encoding="utf-8",
    )
    entry_point = bin_dir / "hermes"
    entry_point.write_text(
        """
import json
import sys

from isolated_probe import ORIGIN

print(json.dumps({"origin": ORIGIN, "sys_path": sys.path}))
""".lstrip(),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(attacker_dir)
    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), str(entry_point)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    state = json.loads(completed.stdout)
    assert state["origin"] == "packaged"
    assert str(site_packages.resolve()) in state["sys_path"]
    assert str(attacker_dir.resolve()) not in state["sys_path"]


def test_zpk_secure_launcher_fails_closed_without_packaged_site_packages(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "zpk" / "libexec" / "hermes-secure-launcher.py"
    entry_point = tmp_path / "venv" / "bin" / "hermes"
    entry_point.parent.mkdir(parents=True)
    entry_point.write_text(
        'raise AssertionError("entry point must not run")\n',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), str(entry_point)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 127
    assert completed.stdout == ""
    assert completed.stderr.strip() == (
        "packaged Hermes site-packages is unavailable"
    )


def test_zpk_secure_launcher_prioritizes_packaged_dependencies(
    monkeypatch,
    tmp_path: Path,
):
    launcher = _load_secure_launcher()
    entry_point = tmp_path / "venv" / "bin" / "hermes"
    site_packages = (
        tmp_path
        / "venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    system_site_packages = tmp_path / "system" / "site-packages"
    site_packages.mkdir(parents=True)
    system_site_packages.mkdir(parents=True)
    original_path = [
        "/stdlib",
        str(system_site_packages),
        str(site_packages),
    ]
    monkeypatch.setattr(launcher.sys, "path", original_path)

    launcher._add_packaged_site_packages(entry_point)

    assert launcher.sys.path == [
        "/stdlib",
        str(site_packages.resolve()),
        str(system_site_packages),
    ]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="prctl hardening is Linux-specific",
)
def test_zpk_secure_launcher_managed_hardening_sets_no_new_privileges(
    tmp_path: Path,
):
    launcher = Path(__file__).resolve().parents[2] / (
        "zpk/libexec/hermes-secure-launcher.py"
    )
    probe = tmp_path / "probe.py"
    probe.write_text(
        """
import ctypes
import json
import runpy

namespace = runpy.run_path(%r)
namespace["_harden_linux_process"](managed_gateway=True)
libc = ctypes.CDLL(None, use_errno=True)
print(json.dumps({
    "dumpable": libc.prctl(3, 0, 0, 0, 0),
    "no_new_privs": libc.prctl(39, 0, 0, 0, 0),
}))
"""
        % str(launcher),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [sys.executable, "-I", str(probe)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == {
        "dumpable": 0,
        "no_new_privs": 1,
    }


def _write_managed_service_limit_fixture(
    service: Path,
    *,
    include_swap: bool = True,
) -> None:
    limits = {
        "memory.high": "805306368\n",
        "memory.max": "1073741824\n",
        "pids.max": "512\n",
    }
    if include_swap:
        limits["memory.swap.max"] = "0\n"
    for name, value in limits.items():
        (service / name).write_text(value, encoding="ascii")


def test_zpk_secure_launcher_builds_supervisor_and_enables_controllers(
    monkeypatch,
    tmp_path: Path,
):
    launcher = _load_secure_launcher()
    cgroup_root = tmp_path / "cgroup"
    service_relative = "/system.slice/zettlab-claw.service"
    service = cgroup_root / service_relative.lstrip("/")
    service.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.procs").write_text("4242\n", encoding="ascii")
    (service / "cgroup.kill").write_text("", encoding="ascii")
    (service / "cgroup.subtree_control").write_text("", encoding="ascii")
    _write_managed_service_limit_fixture(service)
    proc_self = tmp_path / "proc-self-cgroup"
    proc_self.write_text(f"0::{service_relative}\n", encoding="ascii")
    monkeypatch.setattr(launcher, "_CGROUP2_ROOT", cgroup_root)
    monkeypatch.setattr(launcher, "_PROC_SELF_CGROUP", proc_self)
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    monkeypatch.setattr(launcher.os, "getpid", lambda: 4242)
    monkeypatch.setenv(
        launcher._MANAGED_CGROUP_UNIT_ENV,
        "zettlab-claw.service",
    )
    real_mkdir = os.mkdir

    def materialize_cgroup(path, mode):
        real_mkdir(path, mode)
        path = Path(path)
        (path / "cgroup.procs").write_text("", encoding="ascii")
        (path / "cgroup.events").write_text(
            "populated 0\n",
            encoding="ascii",
        )

    monkeypatch.setattr(launcher.os, "mkdir", materialize_cgroup)
    real_write = launcher._write_control_file

    def emulate_kernel_write(path, payload):
        path = Path(path)
        if path == service / "cgroup.subtree_control":
            path.write_text("memory pids\n", encoding="ascii")
            supervisor = service / launcher._MANAGED_SUPERVISOR_CGROUP
            (supervisor / "memory.max").write_text("max\n", encoding="ascii")
            (supervisor / "memory.swap.max").write_text(
                "max\n",
                encoding="ascii",
            )
            (supervisor / "pids.max").write_text("max\n", encoding="ascii")
            return
        real_write(path, payload)
        if path.name == "cgroup.procs":
            (service / "cgroup.procs").write_text("", encoding="ascii")
            proc_self.write_text(
                f"0::{service_relative}/"
                f"{launcher._MANAGED_SUPERVISOR_CGROUP}\n",
                encoding="ascii",
            )

    monkeypatch.setattr(launcher, "_write_control_file", emulate_kernel_write)

    launcher._prepare_managed_service_cgroup()

    supervisor = service / launcher._MANAGED_SUPERVISOR_CGROUP
    assert supervisor.is_dir()
    assert (service / "cgroup.subtree_control").read_text(
        encoding="ascii"
    ) == "memory pids\n"
    assert os.environ[launcher._MANAGED_CGROUP_ROOT_ENV] == service_relative

    # Child Hermes processes inherit the service identity and re-enter through
    # the same packaged launcher. They may verify, but must not rebuild, the
    # already-active supervisor topology.
    launcher._prepare_managed_service_cgroup()
    monkeypatch.setenv(
        launcher._MANAGED_CGROUP_ROOT_ENV,
        "/system.slice/other.service",
    )
    with pytest.raises(OSError, match="service cgroup identity"):
        launcher._prepare_managed_service_cgroup()


def test_zpk_secure_launcher_fails_before_move_without_controllers(
    monkeypatch,
    tmp_path: Path,
):
    launcher = _load_secure_launcher()
    cgroup_root = tmp_path / "cgroup"
    service_relative = "/system.slice/zettlab-claw.service"
    service = cgroup_root / service_relative.lstrip("/")
    service.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.controllers").write_text("pids\n", encoding="ascii")
    (service / "cgroup.procs").write_text("4242\n", encoding="ascii")
    (service / "cgroup.kill").write_text("", encoding="ascii")
    (service / "cgroup.subtree_control").write_text("", encoding="ascii")
    _write_managed_service_limit_fixture(service)
    proc_self = tmp_path / "proc-self-cgroup"
    proc_self.write_text(f"0::{service_relative}\n", encoding="ascii")
    monkeypatch.setattr(launcher, "_CGROUP2_ROOT", cgroup_root)
    monkeypatch.setattr(launcher, "_PROC_SELF_CGROUP", proc_self)
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    monkeypatch.setenv(
        launcher._MANAGED_CGROUP_UNIT_ENV,
        "zettlab-claw.service",
    )

    with pytest.raises(OSError, match="memory and pids delegation"):
        launcher._prepare_managed_service_cgroup()

    assert not (service / launcher._MANAGED_SUPERVISOR_CGROUP).exists()
    assert proc_self.read_text(encoding="ascii") == f"0::{service_relative}\n"


def test_zpk_secure_launcher_main_fails_closed_without_swap_accounting(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    launcher = _load_secure_launcher()
    cgroup_root = tmp_path / "cgroup"
    service_relative = "/system.slice/zettlab-claw.service"
    service = cgroup_root / service_relative.lstrip("/")
    service.mkdir(parents=True)
    (cgroup_root / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.controllers").write_text(
        "memory pids\n",
        encoding="ascii",
    )
    (service / "cgroup.procs").write_text("4242\n", encoding="ascii")
    (service / "cgroup.kill").write_text("", encoding="ascii")
    (service / "cgroup.subtree_control").write_text("", encoding="ascii")
    _write_managed_service_limit_fixture(service, include_swap=False)
    proc_self = tmp_path / "proc-self-cgroup"
    proc_self.write_text(f"0::{service_relative}\n", encoding="ascii")
    entry_point = tmp_path / "hermes-entry.py"
    marker = tmp_path / "entry-ran"
    entry_point.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_CGROUP2_ROOT", cgroup_root)
    monkeypatch.setattr(launcher, "_PROC_SELF_CGROUP", proc_self)
    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.sys, "argv", ["launcher", str(entry_point)])
    monkeypatch.setattr(launcher.os, "geteuid", lambda: 0)
    monkeypatch.setenv(launcher._MANAGED_GATEWAY_ENV, "1")
    monkeypatch.setenv(
        launcher._MANAGED_CGROUP_UNIT_ENV,
        "zettlab-claw.service",
    )

    assert launcher.main() == 125

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "requires memory.swap.max accounting" in captured.err
    assert not marker.exists()
    assert not (service / launcher._MANAGED_SUPERVISOR_CGROUP).exists()
    assert proc_self.read_text(encoding="ascii") == f"0::{service_relative}\n"


def test_zpk_secure_launcher_managed_startup_fails_closed_without_unit_root(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "zpk/libexec/hermes-secure-launcher.py"
    marker = tmp_path / "entry-ran"
    entry_point = tmp_path / "hermes-entry.py"
    entry_point.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["HERMES_MANAGED_GATEWAY"] = "1"
    env["HERMES_MANAGED_CGROUP_UNIT"] = "not-this-process.service"
    env.pop("HERMES_MANAGED_CGROUP_ROOT", None)

    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), str(entry_point)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 125
    assert completed.stdout == ""
    assert "managed gateway cgroup setup failed:" in completed.stderr
    assert not marker.exists()


def test_zpk_secure_launcher_fails_closed_without_packaged_entry(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    launcher = repo_root / "zpk" / "libexec" / "hermes-secure-launcher.py"

    completed = subprocess.run(
        [sys.executable, "-I", str(launcher), str(tmp_path / "missing-hermes")],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 127
    assert completed.stdout == ""
    assert completed.stderr.strip() == "packaged Hermes entry point is unavailable"


def test_zpk_install_removes_legacy_shared_gateway_unit():
    repo_root = Path(__file__).resolve().parents[2]

    install = (repo_root / "zpk" / "install.sh").read_text(encoding="utf-8")
    uninstall = (repo_root / "zpk" / "uninstall.sh").read_text(encoding="utf-8")

    assert "cleanup_legacy_systemd_services" in install
    assert "hermes-agent-mux.service" in install
    assert "systemctl stop \"$legacy_service\"" in install
    assert "systemctl disable \"$legacy_service\"" in install
    assert "LEGACY_GATEWAY_SERVICE_REMOVED=true" in install
    assert "start_replacement_service_after_legacy_cleanup" in install
    assert "hermes-agent-mux.service" in uninstall
