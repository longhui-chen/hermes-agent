"""Multiplex gateway cron scheduler scoping."""

from types import SimpleNamespace

from gateway.config import GatewayConfig
from gateway.run import _multiplex_cron_profiles, _start_gateway_cron_schedulers


def test_multiplex_cron_profiles_skip_legacy_default_when_main_exists(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    family_home = default_home / "profiles" / "family-manager"

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", default_home),
            ("family-manager", family_home),
            ("main", main_home),
        ],
    )

    assert _multiplex_cron_profiles() == [
        ("family-manager", family_home),
        ("main", main_home),
    ]


def test_multiplex_cron_scheduler_runs_each_profile_under_profile_home(tmp_path, monkeypatch):
    from hermes_constants import get_hermes_home

    seen = []
    default_home = tmp_path / ".hermes"
    main_home = default_home / "profiles" / "main"
    family_home = default_home / "profiles" / "family-manager"
    main_adapters = {"api_server": object()}
    family_adapters = {"api_server": object()}

    class DummyScheduler:
        def start(self, stop_event, *, adapters=None, loop=None):
            seen.append((get_hermes_home(), adapters))

    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [
            ("default", default_home),
            ("family-manager", family_home),
            ("main", main_home),
        ],
    )
    monkeypatch.setattr("hermes_cli.profiles.get_active_profile_name", lambda: "default")
    monkeypatch.setattr("cron.scheduler_provider.resolve_cron_scheduler", lambda: DummyScheduler())

    runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True),
        adapters={},
        _profile_adapters={
            "family-manager": family_adapters,
            "main": main_adapters,
        },
    )
    import threading

    stop_event = threading.Event()
    threads = _start_gateway_cron_schedulers(runner, stop_event)
    for thread in threads:
        thread.join(timeout=2)

    assert sorted(seen, key=lambda item: str(item[0])) == sorted(
        [
            (family_home, family_adapters),
            (main_home, main_adapters),
        ],
        key=lambda item: str(item[0]),
    )


def test_cron_env_reads_active_profile_secret_scope(monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler

    monkeypatch.setenv("HERMES_MODEL", "outer-model")
    set_multiplex_active(True)
    token = set_secret_scope({"HERMES_MODEL": "profile-model"})
    try:
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model"
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


def test_cron_env_refreshes_profile_dotenv_each_read(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    env_file = profile_home / ".env"
    env_file.write_text("HERMES_MODEL=profile-model-a\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_MODEL", "outer-model")
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(profile_home))
    stale_scope_token = set_secret_scope({"HERMES_MODEL": "stale-profile-model"})
    try:
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model-a"
        env_file.write_text("HERMES_MODEL=profile-model-b\n", encoding="utf-8")
        assert scheduler._cron_env("HERMES_MODEL", "") == "profile-model-b"
    finally:
        reset_secret_scope(stale_scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_cron_env_keeps_process_global_cron_limits(tmp_path, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from cron import scheduler
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    profile_home = tmp_path / "profiles" / "coder"
    profile_home.mkdir(parents=True)
    (profile_home / ".env").write_text("HERMES_CRON_MAX_PARALLEL=99\n", encoding="utf-8")

    monkeypatch.setenv("HERMES_CRON_MAX_PARALLEL", "1")
    set_multiplex_active(True)
    home_token = set_hermes_home_override(str(profile_home))
    stale_scope_token = set_secret_scope({"HERMES_CRON_MAX_PARALLEL": "2"})
    try:
        assert scheduler._cron_env("HERMES_CRON_MAX_PARALLEL", "4") == "1"
    finally:
        reset_secret_scope(stale_scope_token)
        reset_hermes_home_override(home_token)
        set_multiplex_active(False)


def test_running_job_key_is_profile_qualified(tmp_path, monkeypatch):
    from cron import scheduler

    home_a = tmp_path / "profiles" / "a"
    home_b = tmp_path / "profiles" / "b"
    home_a.mkdir(parents=True)
    home_b.mkdir(parents=True)

    monkeypatch.setattr(scheduler, "_hermes_home", home_a)
    key_a = scheduler._running_job_key({"id": "same-job"})
    monkeypatch.setattr(scheduler, "_hermes_home", home_b)
    key_b = scheduler._running_job_key({"id": "same-job"})

    assert key_a != key_b
    assert key_a[1] == key_b[1] == "same-job"


def test_context_from_reads_active_profile_output_dir(tmp_path, monkeypatch):
    from cron.scheduler import _build_job_prompt
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "worker"
    source_job_id = "abcdef123456"
    default_output = default_home / "cron" / "output" / source_job_id
    profile_output = profile_home / "cron" / "output" / source_job_id
    default_output.mkdir(parents=True)
    profile_output.mkdir(parents=True)
    (default_output / "2026-01-01_00-00-00.md").write_text(
        "default output must not leak",
        encoding="utf-8",
    )
    (profile_output / "2026-01-01_00-00-00.md").write_text(
        "profile scoped output",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(default_home))
    token = set_hermes_home_override(str(profile_home))
    try:
        prompt = _build_job_prompt(
            {
                "id": "feed00000000",
                "name": "profile-job",
                "prompt": "Summarize it",
                "context_from": [source_job_id],
            }
        )
    finally:
        reset_hermes_home_override(token)

    assert "profile scoped output" in prompt
    assert "default output must not leak" not in prompt
