"""The governor refresh gate: defer BEFORE an agent turn is spent.

_governor_refresh_defer must fail open (no token / no URL / unreachable /
unparsable → run), cache the "not a maintainer" 404 answer, and honor a
defer decision. The run_one_job patch must turn a defer decision into an
invisible skip: no agent run, no output file, no failure record — just the
postponed schedule.
"""

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clean_gate_cache(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac

    monkeypatch.setattr(zac, "_refresh_permit_neg_cache", {})
    yield


def _env(monkeypatch, token="tok", origin="http://127.0.0.1:19090", agent="agent-1"):
    import gateway.platforms.zet_agent_cron as zac

    values = {
        "ZETTLAB_AGENT_ACTION_TOKEN": token,
        "ZET_CHAT_APPEND_URL": (origin + "/api/v1/internal/agent/chat/append") if origin else "",
        "ZET_AGENT_ID": agent,
    }
    monkeypatch.setattr(zac, "_scoped_env", lambda name, default="": values.get(name, default))


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode("utf-8")
        self.headers = {}
        self._pos = 0

    def read(self, amt=None):
        # Behave like a real HTTP response: advance a cursor and return b"" at
        # EOF, so the deadline-bounded read loop terminates instead of the mock
        # re-returning the whole body forever.
        if amt is None:
            chunk = self._body[self._pos:]
            self._pos = len(self._body)
            return chunk
        chunk = self._body[self._pos:self._pos + amt]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_gate_fails_open_without_credentials(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac

    _env(monkeypatch, token="", origin="")
    assert zac._governor_refresh_defer({"id": "j1"}) is None


def test_gate_fails_open_on_unreachable_server(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac

    _env(monkeypatch)
    with patch("urllib.request.urlopen", side_effect=OSError("refused")):
        assert zac._governor_refresh_defer({"id": "j1"}) is None


def test_gate_honors_defer_decision(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac

    _env(monkeypatch)
    body = '{"decision":"defer","retry_in_seconds":300,"reason":"memory_pressure"}'
    with patch("urllib.request.urlopen", return_value=_FakeResponse(200, body)):
        assert zac._governor_refresh_defer({"id": "j1"}) == (300.0, "memory_pressure")


def test_gate_allows_and_caches_unbound_agent(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac
    import time

    _env(monkeypatch)
    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req)

        class Resp(_FakeResponse):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Resp(404, "{}")

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        assert zac._governor_refresh_defer({"id": "j1"}) is None
        assert zac._governor_refresh_defer({"id": "j2"}) is None
    assert len(calls) == 1, "a 404 (no bound app) is cached, not re-asked per fire"
    # A negative cache entry exists and expires via TTL only.
    assert "agent-1" in zac._refresh_permit_neg_cache
    assert isinstance(zac._refresh_permit_neg_cache["agent-1"], float)
    _ = time.monotonic()


def test_gate_rejects_unparsable_body(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac

    _env(monkeypatch)
    with patch("urllib.request.urlopen", return_value=_FakeResponse(200, "not-json")):
        assert zac._governor_refresh_defer({"id": "j1"}) is None


def test_gate_run_one_job_skips_and_defers(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac
    import cron.jobs as jobs_mod

    deferred_ids = []

    def fake_defer(job_id, *, seconds=None, until=None, reason=None, clear_claim=False):
        deferred_ids.append((job_id, seconds, reason))
        return {"id": job_id, "next_run_at": "2099-01-01T00:00:00+00:00"}

    monkeypatch.setattr(jobs_mod, "defer_job", fake_defer)

    def original_body(job, **kwargs):
        raise AssertionError("a deferred job must never reach the firing body")

    monkeypatch.setattr(zac, "_governor_refresh_defer", lambda job: (60, "memory_pressure"))
    assert zac._gate_run_one_job(original_body, {"id": "job-x", "source": "app_refresh"}) is True
    assert deferred_ids == [("job-x", 60, "governor:memory_pressure")]

    # No deferral decision → the original fires untouched.
    monkeypatch.setattr(zac, "_governor_refresh_defer", lambda job: None)
    assert zac._gate_run_one_job(lambda job, **kwargs: False, {"id": "job-y", "source": "app_refresh"}) is False


def test_gate_run_one_job_fails_open_on_defer_failure(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac
    import cron.jobs as jobs_mod

    def broken_defer(job_id, *, seconds=None, until=None, reason=None, clear_claim=False):
        raise RuntimeError("jobs.json locked")

    monkeypatch.setattr(jobs_mod, "defer_job", broken_defer)

    fired = []

    def original_body(job, **kwargs):
        fired.append(job["id"])
        return True

    monkeypatch.setattr(zac, "_governor_refresh_defer", lambda job: (60, "memory_pressure"))
    # The push failed → fail open: the original firing body runs so a one-shot
    # is not silently lost to a stale next_run_at + reused dedup key.
    assert zac._gate_run_one_job(original_body, {"id": "job-z", "source": "app_refresh"}) is True
    assert fired == ["job-z"]


def test_gate_ignores_non_app_refresh_jobs(monkeypatch):
    import gateway.platforms.zet_agent_cron as zac
    import cron.jobs as jobs_mod

    deferred = []

    def fake_defer(job_id, *, seconds=None, until=None, reason=None, clear_claim=False):
        deferred.append(job_id)
        return {"id": job_id}

    monkeypatch.setattr(jobs_mod, "defer_job", fake_defer)
    # A profile-level defer must never gate ordinary reminder/report cron jobs.
    monkeypatch.setattr(zac, "_governor_refresh_defer", lambda job: (60, "memory_pressure"))

    def original_body(job, **kwargs):
        return "ran"

    assert zac._gate_run_one_job(original_body, {"id": "reminder", "source": None}) == "ran"
    assert zac._gate_run_one_job(original_body, {"id": "report"}) == "ran"
    assert deferred == []


def test_install_patches_run_one_job(tmp_path, monkeypatch):
    import importlib

    home = tmp_path / ".hermes"
    (home / "cron").mkdir(parents=True)
    (home / "scripts").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants
    import cron.jobs
    import cron.scheduler

    importlib.reload(hermes_constants)
    importlib.reload(cron.jobs)
    importlib.reload(cron.scheduler)
    import gateway.platforms.zet_agent_cron as zac

    importlib.reload(zac)
    zac.install()
    assert getattr(cron.scheduler.run_one_job, zac._PATCH_SENTINEL, False), (
        "install() must gate the shared firing body"
    )
