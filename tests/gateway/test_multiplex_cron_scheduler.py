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
