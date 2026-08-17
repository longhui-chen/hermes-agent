"""ADIC v1: cron's success/error verdict for app-import-tracked jobs.

See zettlab-local-docs/app-fullstack/2026-08-17-应用数据导入契约-ADIC-v1.md §4.5
and the paired interface-freeze doc §7. The root cause this closes: a
maintenance job's `last_status` used to be "the agent didn't throw and said
something" — completely blind to whether the app actually accepted the data
(the Hangzhou 17:57 incident). For a job carrying `app_slug` (written by
local-server at provision time, never by the model), cron must instead judge
by this turn's `app_operation(data.import)` outcomes, read from the
turn-scoped ledger in `gateway.session_context` right before `mark_job_run`.

Jobs without `app_slug` (the overwhelming majority of cron jobs — reminders,
watchdogs, digests) must be completely unaffected; that is `test_run_one_job.py`
today and stays that way (see the added `..._app_slug_absent_is_unaffected`
tests here as an explicit regression pin).
"""
import cron.scheduler as s
from gateway.session_context import import_attempts_snapshot


def _patch_pipeline(monkeypatch, *, success=True, final="final response", error=None,
                     import_attempts=None):
    """Same shape as tests/cron/test_run_one_job.py's helper, extended so
    fake_run_job can simulate the app_host tool recording into the
    turn-scoped ledger from *inside* the agent turn — exactly where the real
    app_host_tool._record_data_import_attempt call happens.
    """
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        calls.append(("run_job", job["id"]))
        from gateway.session_context import record_import_attempt
        for attempt in (import_attempts or []):
            record_import_attempt(**attempt)
        return (success, "out", final, error)

    def fake_save(jid, out):
        calls.append(("save", jid))
        return f"/tmp/{jid}.txt"

    def fake_deliver(job, content, adapters=None, loop=None):
        calls.append(("deliver", job["id"]))
        return None

    def fake_mark(jid, ok, err=None, delivery_error=None, **_kwargs):
        calls.append(("mark", jid, ok, err))

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", fake_save)
    monkeypatch.setattr(s, "_deliver_result", fake_deliver)
    monkeypatch.setattr(s, "mark_job_run", fake_mark)
    return calls


def _mark_call(calls):
    return [c for c in calls if c[0] == "mark"][0]


# --- app_slug jobs: judged purely by this turn's import ledger ---------------

def test_app_slug_job_with_confirmed_import_is_ok(monkeypatch):
    """A confirmed data.import overrides even a soft-failure agent verdict —
    business effect wins over 'the agent produced a plausible reply'."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="更新完成",
        import_attempts=[{"ok": True}],
    )

    s.run_one_job({"id": "j-app-ok", "name": "weather sync", "app_slug": "hangzhou-weather-live"})

    _, jid, ok, err = _mark_call(calls)
    assert (jid, ok, err) == ("j-app-ok", True, None)


def test_app_slug_job_overrides_a_clean_looking_reply_when_import_was_rejected(monkeypatch):
    """This is the exact bug: the agent replies coherently, success=True, but
    the only import attempt this round was rejected by the app. cron must
    still record error with the app's rejection reason."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="已更新（其实没有）",
        import_attempts=[{"ok": False, "error_code": "import_rejected",
                           "error_message": "湿度必须是 0-100 的整数"}],
    )

    s.run_one_job({"id": "j-app-rejected", "name": "weather sync", "app_slug": "hangzhou-weather-live"})

    _, jid, ok, err = _mark_call(calls)
    assert jid == "j-app-rejected"
    assert ok is False
    assert err == "湿度必须是 0-100 的整数"


def test_app_slug_job_overrides_to_error_when_not_confirmed(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch, success=True, final="done",
        import_attempts=[{"ok": False, "error_code": "import_not_confirmed",
                           "error_message": ""}],
    )

    s.run_one_job({"id": "j-app-unconfirmed", "name": "sync", "app_slug": "shenzhen-weather-live"})

    _, jid, ok, err = _mark_call(calls)
    assert ok is False
    # error_message was empty — falls back to the upstream error_code.
    assert err == "import_not_confirmed"


def test_app_slug_job_with_zero_import_attempts_is_hard_error(monkeypatch):
    """The agent replied cleanly (success=True, non-empty final_response) but
    never called data.import at all — e.g. it silently gave up after a 429.
    Interface-freeze §7: this is `error` with the exact frozen message."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="今天天气不错",
        import_attempts=[],
    )

    s.run_one_job({"id": "j-app-noattempt", "name": "sync", "app_slug": "hangzhou-weather-live"})

    _, jid, ok, err = _mark_call(calls)
    assert ok is False
    assert err == "no import attempted in this run"


def test_app_slug_job_uses_the_last_failed_attempt_reason(monkeypatch):
    """Multiple failed attempts this round (e.g. retry-after-rejection) — the
    verdict must report the LAST failure, not the first."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="done",
        import_attempts=[
            {"ok": False, "error_code": "import_rejected", "error_message": "first try: bad date"},
            {"ok": False, "error_code": "import_rejected", "error_message": "second try: bad humidity"},
        ],
    )

    s.run_one_job({"id": "j-app-last-failure", "name": "sync", "app_slug": "hangzhou-weather-live"})

    _, _, ok, err = _mark_call(calls)
    assert ok is False
    assert err == "second try: bad humidity"


def test_app_slug_job_any_confirmed_import_among_failures_is_still_ok(monkeypatch):
    """One rejected attempt followed by a confirmed retry this round is a
    successful round overall — 'has ok=True' wins regardless of order."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="done",
        import_attempts=[
            {"ok": False, "error_code": "import_rejected", "error_message": "bad humidity"},
            {"ok": True},
        ],
    )

    s.run_one_job({"id": "j-app-recovered", "name": "sync", "app_slug": "hangzhou-weather-live"})

    _, _, ok, err = _mark_call(calls)
    assert (ok, err) == (True, None)


# --- non-app_slug jobs: completely unaffected ---------------------------------

def test_job_without_app_slug_is_unaffected_by_import_ledger(monkeypatch):
    """A plain cron job (no app_slug) must keep the old agent-response-based
    verdict even when, hypothetically, a rejected import was recorded during
    its run — the override is strictly gated on app_slug."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="ok",
        import_attempts=[{"ok": False, "error_code": "import_rejected", "error_message": "n/a"}],
    )

    s.run_one_job({"id": "j-no-slug", "name": "reminder"})

    _, jid, ok, err = _mark_call(calls)
    assert (jid, ok, err) == ("j-no-slug", True, None)


def test_job_without_app_slug_and_no_imports_keeps_empty_response_soft_failure(monkeypatch):
    """Regression pin for issue #8585's existing behavior: unaffected by this
    change when app_slug is absent."""
    calls = _patch_pipeline(monkeypatch, success=True, final="   ", import_attempts=[])

    s.run_one_job({"id": "j-no-slug-empty", "name": "reminder"})

    _, jid, ok, err = _mark_call(calls)
    assert jid == "j-no-slug-empty" and ok is False


# --- ledger lifecycle: bounded to one job's turn, no cross-job leakage -------

def test_ledger_is_popped_after_the_job_and_does_not_leak_to_the_next_read(monkeypatch):
    _patch_pipeline(monkeypatch, success=True, final="done", import_attempts=[{"ok": True}])

    s.run_one_job({"id": "j-leak-check", "name": "sync", "app_slug": "hangzhou-weather-live"})

    # Outside any job's run, no scope is open — snapshot must be empty, not a
    # leftover from the job that just ran.
    assert import_attempts_snapshot() == []


def test_second_job_does_not_see_the_first_jobs_ledger(monkeypatch):
    """Two sequential jobs on the same thread must not share a ledger — each
    run_one_job call pushes and pops its own scope."""
    calls = []

    def fake_run_job(job, *, defer_agent_teardown=None):
        from gateway.session_context import record_import_attempt
        if job["id"] == "first":
            record_import_attempt(ok=False, error_code="import_rejected", error_message="boom")
        # "second" never calls record_import_attempt — its ledger must be empty.
        calls.append(job["id"])
        return (True, "out", "done", None)

    monkeypatch.setattr(s, "run_job", fake_run_job)
    monkeypatch.setattr(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt")
    monkeypatch.setattr(s, "_deliver_result", lambda *a, **k: None)
    marks = {}
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, **_kw: marks.__setitem__(jid, (ok, err)),
    )

    s.run_one_job({"id": "first", "name": "sync", "app_slug": "app-a"})
    s.run_one_job({"id": "second", "name": "sync", "app_slug": "app-b"})

    assert marks["first"] == (False, "boom")
    assert marks["second"] == (False, "no import attempted in this run")


# --- interactive/non-cron paths: no scope, nothing to override ---------------

def test_no_ledger_scope_leaks_outside_run_one_job(monkeypatch):
    """Before and after driving a job through run_one_job, no scope is open —
    matching interactive turns, which never push one at all."""
    assert import_attempts_snapshot() == []
    _patch_pipeline(monkeypatch, success=True, final="done", import_attempts=[{"ok": True}])
    s.run_one_job({"id": "j-scope-check", "name": "sync", "app_slug": "app-a"})
    assert import_attempts_snapshot() == []
