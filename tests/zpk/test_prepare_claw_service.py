import fcntl
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
    volume_data_target: Path | None = None,
) -> tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    app_base = tmp_path / "zettos" / "main" / "apps" / "com.zettlab.claw"
    app_root = app_base / "current"
    script = app_root / "prepare-claw-service.sh"
    python_path = app_root / "lib" / "hermes-agent" / "venv" / "bin" / "python"
    hermes_path = python_path.with_name("hermes")
    hermes_wrapper_path = app_root / "bin" / "hermes"
    invocation_log = app_root / "hermes-invocations.jsonl"
    hermes_home = app_base / "data" / "hermes_home"
    env_path = app_base / "data" / "secrets" / "zettlab-claw.env"

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
    if default_presets_dir is not None:
        default_assignment = (
            'DEFAULT_ZETTLAB_PRESETS_DIR="/volume1/subvol/agents/'
            'zettlab-presets/current"'
        )
        assert default_assignment in prepare_source
        prepare_source = prepare_source.replace(
            default_assignment,
            f"DEFAULT_ZETTLAB_PRESETS_DIR={shlex.quote(str(default_presets_dir))}",
        )
    if volume_data_target is not None:
        volume_assignment = (
            'VOLUME_DATA_TARGET="/volume1/subvol/apps/'
            '$(basename "$APP_BASE")/data"'
        )
        assert volume_assignment in prepare_source
        prepare_source = prepare_source.replace(
            volume_assignment,
            f"VOLUME_DATA_TARGET={shlex.quote(str(volume_data_target))}",
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
    env.pop("HERMES_MANAGED_DIR", None)
    # The fixture's packaged Python must import this checkout, not an unrelated
    # editable Hermes installation that happens to exist in the test venv.
    env["PYTHONPATH"] = str(repo_root)
    env.update(overrides)
    return env


def _hermes_invocations(app_root: Path) -> list[list[str]]:
    path = app_root / "hermes-invocations.jsonl"
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


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


def test_prepare_claw_service_config_set_flow_normalizes_disabled_inline_gateway(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)

    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "gateway: {host: 127.0.0.1, multiplex_profiles: false}\n"
        "skills:\n"
        "  external_dirs:\n"
        "    - /opt/zettlab/skills\n",
        encoding="utf-8",
    )
    config_path.chmod(0o640)

    presets_dir = tmp_path / "presets" / "v0.7.12"
    presets_dir.mkdir(parents=True)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(presets_dir)),
    )

    config = config_path.read_text(encoding="utf-8")

    assert config.count("gateway:") == 1
    assert "gateway:\n" in config
    assert "  host: 127.0.0.1\n" in config
    assert "  multiplex_profiles: true\n" in config
    assert "/opt/zettlab/skills" in config
    assert config_path.stat().st_mode & 0o777 == 0o640
    assert _hermes_invocations(app_root) == [
        ["config", "set", "gateway.multiplex_profiles", "true"]
    ]
    assert env_path.exists()
    env_text = env_path.read_text(encoding="utf-8")
    assert "ZET_AGENT_KEY=" in env_text
    assert f"ZETTLAB_PRESETS_DIR={presets_dir}\n" in env_text


def test_prepare_claw_service_skips_config_set_when_multiplex_is_enabled(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    original = (
        "# Keep this user formatting unchanged.\n"
        "gateway:\n"
        "  multiplex_profiles: true\n"
    )
    config_path.write_text(original, encoding="utf-8")
    config_path.chmod(0o600)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert config_path.read_text(encoding="utf-8") == original
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert _hermes_invocations(app_root) == []


def test_prepare_claw_service_normalizes_duplicate_enabled_multiplex_key(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "gateway:\n"
        "  multiplex_profiles: true\n"
        "  multiplex_profiles: true\n"
        "skills:\n"
        "  external_dirs:\n"
        "    - /opt/zettlab/skills\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )
    normalized = config_path.read_text(encoding="utf-8")

    assert normalized.count("multiplex_profiles: true\n") == 1
    assert "/opt/zettlab/skills" in normalized
    assert _hermes_invocations(app_root) == [
        ["config", "set", "gateway.multiplex_profiles", "true"]
    ]

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert config_path.read_text(encoding="utf-8") == normalized
    assert _hermes_invocations(app_root) == [
        ["config", "set", "gateway.multiplex_profiles", "true"]
    ]


def test_prepare_claw_service_fails_closed_on_invalid_config_yaml(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    original = "gateway:\n  multiplex_profiles: false\ninvalid: [\n"
    config_path.write_text(original, encoding="utf-8")

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert config_path.read_text(encoding="utf-8") == original
    assert _hermes_invocations(app_root) == []


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


def test_prepare_claw_service_updates_legacy_top_level_multiplex_override(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "multiplex_profiles: false\n"
        "gateway:\n"
        "  multiplex_profiles: true\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert _hermes_invocations(app_root) == [
        ["config", "set", "multiplex_profiles", "true"]
    ]


def test_prepare_claw_service_flow_honors_runtime_top_level_null_precedence(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "multiplex_profiles:\n"
        "gateway:\n"
        "  multiplex_profiles: true\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert _hermes_invocations(app_root) == [
        ["config", "set", "multiplex_profiles", "true"]
    ]
    assert "multiplex_profiles: true\n" in config_path.read_text(encoding="utf-8")


def test_prepare_claw_service_flow_honors_managed_multiplex_config(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    managed_dir = tmp_path / "managed"
    managed_dir.mkdir()
    (managed_dir / "config.yaml").write_text(
        "gateway:\n"
        "  multiplex_profiles: true\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(HERMES_MANAGED_DIR=str(managed_dir)),
    )

    assert config_path.read_text(encoding="utf-8") == "{}\n"
    assert _hermes_invocations(app_root) == []


@pytest.mark.parametrize("with_explicit_override", [False, True])
def test_prepare_claw_service_loads_persisted_managed_env_without_overriding_explicit(
    tmp_path: Path,
    with_explicit_override: bool,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    config_path = hermes_home / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    persisted_managed = tmp_path / "managed persisted"
    explicit_managed = tmp_path / "managed-explicit"
    persisted_managed.mkdir()
    explicit_managed.mkdir()
    (persisted_managed / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: true\n",
        encoding="utf-8",
    )
    (explicit_managed / "config.yaml").write_text(
        "gateway:\n  multiplex_profiles: false\n",
        encoding="utf-8",
    )
    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f'HERMES_MANAGED_DIR="{persisted_managed}"\n',
        encoding="utf-8",
    )

    overrides = (
        {"HERMES_MANAGED_DIR": str(explicit_managed)}
        if with_explicit_override
        else {}
    )
    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(**overrides),
        capture_output=True,
        text=True,
    )

    if with_explicit_override:
        assert result.returncode != 0
        assert "managed by your administrator" in result.stderr
        assert _hermes_invocations(app_root) == [
            ["config", "set", "gateway.multiplex_profiles", "true"]
        ]
        assert config_path.read_text(encoding="utf-8") == "{}\n"
    else:
        assert result.returncode == 0, result.stderr
        assert _hermes_invocations(app_root) == []
        assert config_path.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize(
    "gateway_json",
    [
        {"multiplex_profiles": True},
        {"gateway": {"multiplex_profiles": True}},
    ],
)
def test_prepare_claw_service_flow_honors_legacy_gateway_json_default(
    tmp_path: Path,
    gateway_json: dict,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    (hermes_home / "gateway.json").write_text(
        json.dumps(gateway_json) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert not (hermes_home / "config.yaml").exists()
    assert _hermes_invocations(app_root) == []


@pytest.mark.parametrize("managed_null", [False, True])
def test_prepare_claw_service_preserves_legacy_nested_true_through_null_override(
    tmp_path: Path,
    managed_null: bool,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    hermes_home.mkdir(parents=True)
    (hermes_home / "gateway.json").write_text(
        '{"gateway": {"multiplex_profiles": true}}\n',
        encoding="utf-8",
    )
    config_path = hermes_home / "config.yaml"
    config_path.write_text(
        "{}\n" if managed_null else "multiplex_profiles:\n",
        encoding="utf-8",
    )

    overrides: dict[str, str] = {}
    if managed_null:
        managed_dir = tmp_path / "managed"
        managed_dir.mkdir()
        (managed_dir / "config.yaml").write_text(
            "gateway:\n  multiplex_profiles:\n",
            encoding="utf-8",
        )
        overrides["HERMES_MANAGED_DIR"] = str(managed_dir)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(**overrides),
    )

    assert _hermes_invocations(app_root) == []
    assert config_path.read_text(encoding="utf-8") == (
        "{}\n" if managed_null else "multiplex_profiles:\n"
    )


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


def test_prepare_claw_service_pins_resolved_presets_version(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    version_dir = tmp_path / "presets" / "v0.7.12"
    version_dir.mkdir(parents=True)
    current = version_dir.parent / "current"
    os.symlink(version_dir.name, current)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(ZETTLAB_PRESETS_DIR=str(current)),
    )

    env_text = env_path.read_text(encoding="utf-8")
    assert f"ZETTLAB_PRESETS_DIR={version_dir}\n" in env_text
    assert f"ZETTLAB_PRESETS_DIR={current}\n" not in env_text


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
        "ZET_AGENT_KEY='stale\nEVIL=1'\n",
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
    assert env_text.count("ZET_AGENT_KEY=") == 1


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

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )
    dumped = subprocess.run(
        [str(app_root / "prepare-claw-service.sh"), "--dump-env"],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
    ).stdout

    assert b"ZET_AGENT_KEY\0" in dumped
    assert b"ZET_AGENT_ENABLED\0true\0" in dumped
    assert env_path.read_text(encoding="utf-8").startswith("ZET_AGENT_KEY=")


def test_prepare_claw_service_filters_package_assignments_in_cr_only_env(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, env_path = _prepare_script_fixture(tmp_path)
    env_path.parent.mkdir(parents=True)
    env_path.write_bytes(b"ZET_AGENT_KEY=stale\rCUSTOM_SAFE=keep\r")

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


def test_prepare_claw_service_preserves_restrictive_data_directory_mode(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, _hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_dir = _hermes_home.parent
    data_dir.mkdir()
    data_dir.chmod(0o700)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert data_dir.stat().st_mode & 0o777 == 0o700


def test_prepare_claw_service_allows_trusted_ota_data_symlink(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    data_link = hermes_home.parent
    expected_target = tmp_path / "zettos" / "main" / "data" / "com.zettlab.claw"
    expected_target.mkdir(parents=True)
    expected_target.chmod(0o750)
    data_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(expected_target, data_link)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert data_link.is_symlink()
    assert data_link.resolve() == expected_target.resolve()
    assert hermes_home.is_dir()
    assert env_path.is_file()
    assert expected_target.stat().st_mode & 0o777 == 0o750


def test_prepare_claw_service_allows_trusted_volume_data_symlink(tmp_path: Path):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    volume_target = tmp_path / "volume1" / "subvol" / "apps" / "com.zettlab.claw" / "data"
    app_root, hermes_home, env_path = _prepare_script_fixture(
        tmp_path,
        volume_data_target=volume_target,
    )
    data_link = hermes_home.parent
    volume_target.mkdir(parents=True)
    volume_target.chmod(0o750)
    data_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(volume_target, data_link)

    subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=True,
        cwd=str(app_root),
        env=_script_env(),
    )

    assert data_link.is_symlink()
    assert data_link.resolve() == volume_target.resolve()
    assert hermes_home.is_dir()
    assert env_path.is_file()
    assert volume_target.stat().st_mode & 0o777 == 0o750


def test_prepare_claw_service_refuses_writable_ota_data_symlink_target(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_link = hermes_home.parent
    expected_target = tmp_path / "zettos" / "main" / "data" / "com.zettlab.claw"
    expected_target.mkdir(parents=True)
    expected_target.chmod(0o770)
    marker = expected_target / "marker"
    marker.write_text("do-not-touch\n", encoding="utf-8")
    data_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(expected_target, data_link)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing untrusted data symlink" in result.stderr
    assert marker.read_text(encoding="utf-8") == "do-not-touch\n"
    assert expected_target.stat().st_mode & 0o777 == 0o770


def test_prepare_claw_service_refuses_writable_data_symlink_ancestor(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_link = hermes_home.parent
    data_parent = tmp_path / "zettos" / "main" / "data"
    expected_target = data_parent / "com.zettlab.claw"
    expected_target.mkdir(parents=True)
    expected_target.chmod(0o750)
    data_parent.chmod(0o770)
    data_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(expected_target, data_link)

    result = subprocess.run(
        [str(app_root / "prepare-claw-service.sh")],
        check=False,
        cwd=str(app_root),
        env=_script_env(),
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "refusing untrusted data symlink" in result.stderr
    assert not (expected_target / "secrets").exists()


def test_prepare_claw_service_pins_validated_data_symlink_target(
    tmp_path: Path,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("prepare-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, _env_path = _prepare_script_fixture(tmp_path)
    data_link = hermes_home.parent
    expected_target = tmp_path / "zettos" / "main" / "data" / "com.zettlab.claw"
    replacement_target = tmp_path / "replacement-data"
    expected_target.mkdir(parents=True)
    replacement_target.mkdir()
    expected_target.chmod(0o750)
    replacement_target.chmod(0o750)
    data_link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(expected_target, data_link)

    process = subprocess.Popen(
        [str(app_root / "prepare-claw-service.sh")],
        cwd=str(app_root),
        env=_script_env(HERMES_TEST_HERMES_DELAY="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    invocation_log = app_root / "hermes-invocations.jsonl"
    deadline = time.monotonic() + 10
    while not invocation_log.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert invocation_log.exists()

    data_link.unlink()
    os.symlink(replacement_target, data_link)
    stdout, stderr = process.communicate(timeout=30)

    assert process.returncode == 0, (stdout, stderr)
    assert (expected_target / "hermes_home" / "config.yaml").is_file()
    assert (expected_target / "secrets" / "zettlab-claw.env").is_file()
    assert list(replacement_target.iterdir()) == []


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
        os.symlink(target, data_dir)
    else:
        data_dir.mkdir()
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
    if state_path == "data":
        assert "refusing untrusted data symlink" in result.stderr
    else:
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
    assert _hermes_invocations(app_root) == [
        ["config", "set", "gateway.multiplex_profiles", "true"]
    ]
    key = env_path.with_name("zet_agent.key").read_text(encoding="utf-8").strip()
    assert f"ZET_AGENT_KEY={key}\n" in env_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("with_explicit_override", [False, True])
def test_start_claw_service_loads_managed_env_before_prepare_without_overriding_explicit(
    tmp_path: Path,
    with_explicit_override: bool,
):
    if not _readlink_f_available(tmp_path):
        pytest.skip("start-claw-service.sh uses GNU readlink -f")

    app_root, hermes_home, env_path = _prepare_script_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parents[2]
    start_script = app_root / "start-claw-service.sh"
    shutil.copy2(repo_root / "zpk" / "start-claw-service.sh", start_script)
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

    env_path.parent.mkdir(parents=True)
    env_path.write_text(
        f'HERMES_MANAGED_DIR="{persisted_managed}"\n'
        'CUSTOM_SAFE="keep "\'me\'\n'
        r"CUSTOM_UNQUOTED=one\ two" "\n"
        r'CUSTOM_DOUBLE="literal\nvalue"' "\n",
        encoding="utf-8",
    )

    gateway_log = app_root / "gateway-env.json"
    hermes_entry = app_root / "bin" / "hermes"
    hermes_entry.write_text(
        f"""#!{sys.executable}
import json
import os
import sys
from pathlib import Path

Path({str(gateway_log)!r}).write_text(
    json.dumps({{
        "argv": sys.argv[1:],
        "managed": os.environ.get("HERMES_MANAGED_DIR"),
        "custom": os.environ.get("CUSTOM_SAFE"),
        "unquoted": os.environ.get("CUSTOM_UNQUOTED"),
        "double": os.environ.get("CUSTOM_DOUBLE"),
    }}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    hermes_entry.chmod(0o755)

    overrides = (
        {"HERMES_MANAGED_DIR": str(explicit_managed)}
        if with_explicit_override
        else {}
    )
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
    assert _hermes_invocations(app_root) == []


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


def test_prepare_claw_service_strips_newlines_from_presets_dir(tmp_path: Path):
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


def test_zpk_agent_service_names_are_device_facing():
    repo_root = Path(__file__).resolve().parents[2]

    service = (repo_root / "zpk" / "init.d" / "zettlab-claw.service").read_text(encoding="utf-8")
    start_wrapper = (repo_root / "zpk" / "start-claw-service.sh").read_text(encoding="utf-8")
    package_meta = (repo_root / "zpk" / "package.meta").read_text(encoding="utf-8")
    install = (repo_root / "zpk" / "install.sh").read_text(encoding="utf-8")
    start = (repo_root / "zpk" / "init.d" / "start.sh").read_text(encoding="utf-8")
    stop = (repo_root / "zpk" / "init.d" / "stop.sh").read_text(encoding="utf-8")
    uninstall = (repo_root / "zpk" / "uninstall.sh").read_text(encoding="utf-8")

    meta = json.loads(package_meta)

    assert "EnvironmentFile=" not in service
    assert "ExecStart=__APP_BASE__/current/start-claw-service.sh" in service
    assert '"$APP_ROOT/prepare-claw-service.sh"' in start_wrapper
    prepare_call = start_wrapper.index('\n"$APP_ROOT/prepare-claw-service.sh"\n')
    assert start_wrapper.index("\nload_env_after_prepare\n") > prepare_call
    assert ". \"$ENV_FILE\"" not in start_wrapper
    assert meta["service_name"] == "zettlab-claw"
    assert "restart" not in meta
    assert "systemctl restart zettlab-claw.service" not in install
    assert "systemctl start zettlab-claw.service" in install
    assert "systemctl start zettlab-claw.service" in start
    assert "systemctl stop zettlab-claw.service" in stop
    assert "zettlab-claw.service" in uninstall


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
