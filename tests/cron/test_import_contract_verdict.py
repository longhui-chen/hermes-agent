"""ADIC v1: cron's success/error verdict for app-import-tracked jobs.

See zettlab-local-docs/app-fullstack/2026-08-17-应用数据导入契约-ADIC-v1.md §4.5
and the paired interface-freeze doc §7. The root cause this closes: a
maintenance job's `last_status` used to be "the agent didn't throw and said
something" — completely blind to whether the app actually accepted the data
(the Hangzhou 17:57 incident). For a job carrying `app_slug` (written by
local-server at provision time, never by the model), cron must instead judge
by this turn's `app_operation` outcomes, read from the turn-scoped ledger in
`gateway.session_context` right before `mark_job_run`, filtered by the job's
`import_operation` — the APP's own declared write-operation name (e.g.
"records.refresh" for a blueprint app), not necessarily the literal
"data.import". A first pass of this fix hardcoded the "data.import" name at
the recording layer; that is a P0 mirror-image bug (see the
"operation-name filtering" section below): any app whose write operation
isn't literally named "data.import" would have nothing recorded and would
fail EVERY round, regardless of whether the import actually worked.

Jobs without `app_slug` (the overwhelming majority of cron jobs — reminders,
watchdogs, digests) must be completely unaffected; that is `test_run_one_job.py`
today and stays that way (see the added `..._app_slug_absent_is_unaffected`
tests here as an explicit regression pin).
"""
from unittest.mock import MagicMock, patch

import cron.scheduler as s
from gateway.session_context import import_attempts_snapshot


def _patch_pipeline(monkeypatch, *, success=True, final="final response", error=None,
                     import_attempts=None):
    """Same shape as tests/cron/test_run_one_job.py's helper, extended so
    fake_run_job can simulate the app_host tool recording into the
    turn-scoped ledger from *inside* the agent turn — exactly where the real
    app_host_tool._record_app_operation_attempt call happens. Each entry in
    ``import_attempts`` is passed verbatim as kwargs to
    gateway.session_context.record_import_attempt, so it must include
    ``operation``.
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
# These jobs never set import_operation, so the verdict falls back to the
# original fixed name "data.import" (see the fallback test in the
# operation-name-filtering section below for that behavior pinned explicitly).

def test_app_slug_job_with_confirmed_import_is_ok(monkeypatch):
    """A confirmed data.import overrides even a soft-failure agent verdict —
    business effect wins over 'the agent produced a plausible reply'."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="更新完成",
        import_attempts=[{"operation": "data.import", "ok": True}],
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
        import_attempts=[{"operation": "data.import", "ok": False, "error_code": "import_rejected",
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
        import_attempts=[{"operation": "data.import", "ok": False,
                           "error_code": "import_not_confirmed", "error_message": ""}],
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
            {"operation": "data.import", "ok": False, "error_code": "import_rejected",
             "error_message": "first try: bad date"},
            {"operation": "data.import", "ok": False, "error_code": "import_rejected",
             "error_message": "second try: bad humidity"},
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
            {"operation": "data.import", "ok": False, "error_code": "import_rejected",
             "error_message": "bad humidity"},
            {"operation": "data.import", "ok": True},
        ],
    )

    s.run_one_job({"id": "j-app-recovered", "name": "sync", "app_slug": "hangzhou-weather-live"})

    _, _, ok, err = _mark_call(calls)
    assert (ok, err) == (True, None)


# --- operation-name filtering: import_operation is the app's own declared name
# P0 (post-review): the first pass of this fix hardcoded "data.import" at the
# RECORDING layer (tools/apphost_tool.py), so an app whose declared write
# operation is named anything else — e.g. "records.refresh", the name a real
# blueprint app uses per zettlab-local-server's
# app_dedicated_create_flow_test.go — got nothing recorded, ever, and failed
# every single round regardless of whether the import actually worked. The
# fix: recording is unfiltered (any app_operation call lands in the ledger,
# tagged with its name); filtering by job["import_operation"] happens ONLY at
# verdict time, here in run_one_job.

def test_verdict_matches_the_jobs_own_import_operation_name(monkeypatch):
    """A confirmed call to the app's OWN declared operation name (not the
    literal "data.import") must be judged ok."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="quality board refreshed",
        import_attempts=[{"operation": "records.refresh", "ok": True}],
    )

    s.run_one_job({
        "id": "j-custom-op-ok", "name": "quality board sync",
        "app_slug": "quality-dashboard", "import_operation": "records.refresh",
    })

    _, jid, ok, err = _mark_call(calls)
    assert (jid, ok, err) == ("j-custom-op-ok", True, None)


def test_verdict_reports_failure_of_the_jobs_own_import_operation(monkeypatch):
    calls = _patch_pipeline(
        monkeypatch, success=True, final="done",
        import_attempts=[{"operation": "records.refresh", "ok": False,
                           "error_code": "import_rejected", "error_message": "duplicate record_id"}],
    )

    s.run_one_job({
        "id": "j-custom-op-rejected", "name": "quality board sync",
        "app_slug": "quality-dashboard", "import_operation": "records.refresh",
    })

    _, jid, ok, err = _mark_call(calls)
    assert (jid, ok) == ("j-custom-op-rejected", False)
    assert err == "duplicate record_id"


def test_read_only_operation_this_round_does_not_count_as_import_even_with_custom_name(monkeypatch):
    """The agent only called the read operation (data.import_schema) this
    round — never the job's declared write operation (records.refresh).
    Proves two things at once: (1) reads never count as an import, (2) a
    same-turn call to a DIFFERENT operation name doesn't accidentally satisfy
    a job whose import_operation is something else."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="read the schema, did not write",
        import_attempts=[{"operation": "data.import_schema", "ok": True}],
    )

    s.run_one_job({
        "id": "j-custom-op-read-only", "name": "quality board sync",
        "app_slug": "quality-dashboard", "import_operation": "records.refresh",
    })

    _, jid, ok, err = _mark_call(calls)
    assert jid == "j-custom-op-read-only"
    assert ok is False
    assert err == "no import attempted in this run"


def test_job_missing_import_operation_falls_back_to_data_import(monkeypatch):
    """A job with app_slug but no import_operation (older provisioning path,
    or any future path that omits it) must keep judging against the
    original fixed name "data.import" — not silently stop judging."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="done",
        import_attempts=[{"operation": "data.import", "ok": True}],
    )

    job = {"id": "j-fallback-name", "name": "sync", "app_slug": "hangzhou-weather-live"}
    assert "import_operation" not in job
    s.run_one_job(job)

    _, jid, ok, err = _mark_call(calls)
    assert (jid, ok, err) == ("j-fallback-name", True, None)


# --- non-app_slug jobs: completely unaffected ---------------------------------

def test_job_without_app_slug_is_unaffected_by_import_ledger(monkeypatch):
    """A plain cron job (no app_slug) must keep the old agent-response-based
    verdict even when, hypothetically, a rejected import was recorded during
    its run — the override is strictly gated on app_slug."""
    calls = _patch_pipeline(
        monkeypatch, success=True, final="ok",
        import_attempts=[{"operation": "data.import", "ok": False,
                           "error_code": "import_rejected", "error_message": "n/a"}],
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
    _patch_pipeline(monkeypatch, success=True, final="done",
                     import_attempts=[{"operation": "data.import", "ok": True}])

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
            record_import_attempt(operation="data.import", ok=False,
                                   error_code="import_rejected", error_message="boom")
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


# --- scope lifecycle on non-happy paths ---------------------------------------
# The finally in run_one_job is supposed to guarantee the ledger scope never
# leaks regardless of how the function exits. These exercise the three ways
# team-lead flagged: an early return before the scope is even pushed (claim
# rejected), an exception unwinding through the outer handler after the scope
# was pushed, and the interrupted-flag short-circuit that skips mark_job_run
# but still runs the function to its normal end.

def test_scope_does_not_leak_when_claim_dispatch_fails(monkeypatch):
    """claim_dispatch() rejecting the job returns True immediately, BEFORE
    mark_execution_running / push_import_attempts_scope ever run — the token
    stays None. The finally's `is not None` guard must not raise, and no
    scope must be left open for a later reader."""
    monkeypatch.setattr(s, "claim_dispatch", lambda _job_id: False)

    ok = s.run_one_job({"id": "j-claim-rejected", "name": "sync", "app_slug": "app-a"})

    assert ok is True
    assert import_attempts_snapshot() == []


def test_scope_does_not_leak_when_run_job_raises(monkeypatch):
    """An exception inside run_job unwinds through the outer `except
    BaseException` (which records the failure via mark_job_run); the ledger
    scope pushed earlier in the try must still be popped by the finally."""
    seen_mid_run = []

    def boom(job, *, defer_agent_teardown=None):
        from gateway.session_context import record_import_attempt
        record_import_attempt(operation="data.import", ok=False,
                               error_code="transport_error", error_message="mid-run")
        # Snapshot BEFORE raising: proves the scope was genuinely open (the
        # record above actually landed), not that it merely stayed empty by
        # coincidence of the ledger never having been pushed at all.
        seen_mid_run.append(import_attempts_snapshot())
        raise RuntimeError("kaboom")

    monkeypatch.setattr(s, "run_job", boom)
    marks = []
    monkeypatch.setattr(
        s, "mark_job_run",
        lambda jid, ok, err=None, delivery_error=None, **_kwargs: marks.append((jid, ok, err)),
    )

    ok = s.run_one_job({"id": "j-run-job-raises", "name": "sync", "app_slug": "app-a"})

    assert ok is False
    assert marks == [("j-run-job-raises", False, "kaboom")]
    assert len(seen_mid_run[0]) == 1 and seen_mid_run[0][0]["error_code"] == "transport_error"
    # Popped despite the raise, not merely empty by coincidence.
    assert import_attempts_snapshot() == []


def test_scope_does_not_leak_when_interrupted_flag_skips_mark_job_run(monkeypatch):
    """The shutdown path already wrote the authoritative status for this run
    (see test_shutdown_interrupt.py); run_one_job's own mark_job_run write is
    suppressed. The ledger scope must still be pushed AND popped normally
    around that suppressed write — it is not a function-level early return."""
    job = {"id": "j-interrupted", "name": "sync", "app_slug": "app-a"}
    s._interrupted_job_ids.add(job["id"])
    calls = _patch_pipeline(monkeypatch, success=True, final="final response",
                             import_attempts=[{"operation": "data.import", "ok": True}])

    try:
        ok = s.run_one_job(job)
    finally:
        s._interrupted_job_ids.discard(job["id"])

    assert ok is True
    assert not any(c[0] == "mark" for c in calls)  # suppressed by the interrupted flag
    assert import_attempts_snapshot() == []


# --- interactive/non-cron paths: no scope, nothing to override ---------------

def test_no_ledger_scope_leaks_outside_run_one_job(monkeypatch):
    """Before and after driving a job through run_one_job, no scope is open —
    matching interactive turns, which never push one at all."""
    assert import_attempts_snapshot() == []
    _patch_pipeline(monkeypatch, success=True, final="done",
                     import_attempts=[{"operation": "data.import", "ok": True}])
    s.run_one_job({"id": "j-scope-check", "name": "sync", "app_slug": "app-a"})
    assert import_attempts_snapshot() == []


# --- semantic coexistence with main's maintenance-task session-id branch -----
# Zero textual conflicts on rebase (327a609e03) only proves the two patches
# don't physically overlap in the source — it does not prove they cooperate
# on the same job. Main's branch (cron/scheduler.py:run_job, ~3308) picks a
# `cron_task_<id>_<ts>` session id — which _execution_headers then turns into
# an `X-Zettlab-App-Maintenance-Task-Id` header — whenever the job's prompt
# contains `maintenance-key=`. Ours picks the verdict based on `app_slug`. A
# real local-server-provisioned refresh maintenance task carries BOTH markers
# at once (the prompt is server-rendered and always has maintenance-key=; the
# app_slug is server-stamped on the same job). These tests drive run_one_job
# through the REAL run_job (only the AIAgent/SessionDB/provider boundary is
# mocked, unlike _patch_pipeline above which replaces run_job wholesale), so
# main's session-id branch and our ledger-based verdict both actually run on
# one job, one call — not two independently-mocked halves that merely don't
# collide on disk.

def _run_via_real_run_job(tmp_path, job, *, run_conversation_result="final reply",
                           record_attempts=()):
    """Drive job through run_one_job -> the real run_job, mocked only at the
    external boundary (LLM/session-db/provider). Returns
    (ok, mark_calls, session_id_passed_to_AIAgent)."""
    marks = []

    def fake_run_conversation(prompt, **kwargs):
        from gateway.session_context import record_import_attempt
        for attempt in record_attempts:
            record_import_attempt(**attempt)
        return {"final_response": run_conversation_result}

    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state.SessionDB", return_value=MagicMock()), \
         patch(
             "hermes_cli.runtime_provider.resolve_runtime_provider",
             return_value={
                 "api_key": "test-key", "base_url": "https://example.invalid/v1",
                 "provider": "openrouter", "api_mode": "chat_completions",
             },
         ), \
         patch("run_agent.AIAgent") as mock_agent_cls, \
         patch.object(s, "save_job_output", lambda jid, out: f"/tmp/{jid}.txt"), \
         patch.object(s, "_deliver_result", lambda *a, **k: None), \
         patch.object(
             s, "mark_job_run",
             lambda jid, ok, err=None, delivery_error=None, **_kw: marks.append((jid, ok, err)),
         ):
        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = fake_run_conversation
        mock_agent_cls.return_value = mock_agent

        ok = s.run_one_job(job)
        session_id = mock_agent_cls.call_args.kwargs["session_id"]

    return ok, marks, session_id


_MAINTENANCE_PROMPT = "fetch today's forecast maintenance-key=abc123 and import it"


def test_app_slug_and_maintenance_key_both_apply_on_the_same_job(tmp_path):
    """The real-world case: both markers present. main's session-kind branch
    AND our import-ledger verdict must both fire on this one run."""
    job = {
        "id": "coexist-both",
        "name": "hangzhou weather sync",
        "prompt": _MAINTENANCE_PROMPT,
        "app_slug": "hangzhou-weather-live",
        "import_operation": "data.import",
    }
    ok, marks, session_id = _run_via_real_run_job(
        tmp_path, job, record_attempts=[{"operation": "data.import", "ok": True}],
    )

    assert ok is True
    # main's branch: session id is the cron_task_<id>_<ts> form.
    assert session_id.startswith("cron_task_coexist-both_")
    # our branch: verdict came from the import ledger, not "agent replied".
    assert marks == [("coexist-both", True, None)]


def test_app_slug_present_without_maintenance_key_verdict_still_applies(tmp_path):
    """app_slug alone (no maintenance-key= in the prompt): our verdict must
    still fire — it does not depend on main's session-kind marker — while
    the session id stays in the plain (non-task) form."""
    job = {
        "id": "coexist-slug-only",
        "name": "sync",
        "prompt": "fetch today's forecast and import it",
        "app_slug": "hangzhou-weather-live",
        "import_operation": "data.import",
    }
    ok, marks, session_id = _run_via_real_run_job(tmp_path, job, record_attempts=[])

    assert not session_id.startswith("cron_task_")
    assert session_id.startswith("cron_coexist-slug-only_")
    # zero import attempts + app_slug present -> our hard-error branch,
    # despite the agent having replied non-emptily.
    assert marks == [("coexist-slug-only", False, "no import attempted in this run")]


def test_maintenance_key_present_without_app_slug_verdict_does_not_apply(tmp_path):
    """maintenance-key= alone (no app_slug on the job): main's session-kind
    branch still fires — it does not depend on our field — but our verdict
    override must NOT engage; the job is judged by the ordinary agent-reply
    rule, unaffected by there being zero import attempts."""
    job = {
        "id": "coexist-key-only",
        "name": "some other bound task",
        "prompt": _MAINTENANCE_PROMPT,
    }
    ok, marks, session_id = _run_via_real_run_job(tmp_path, job, record_attempts=[])

    assert session_id.startswith("cron_task_coexist-key-only_")
    # Ordinary success: agent replied non-emptily, no app_slug to override it.
    assert marks == [("coexist-key-only", True, None)]
