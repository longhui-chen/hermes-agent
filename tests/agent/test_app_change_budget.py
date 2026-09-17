"""The per-user-turn budget on rebuilding / republishing a generated app.

Background: the app-builder skill has always told the model to stop after about
three failed repair rounds and hand back to the user, and nothing counted. One
board turn ran three hours and 431 tool calls republishing the same app. The
gate lives in ``tools.apphost_tool`` — the single funnel every credentialed App
Host action passes through — and refuses locally, sending nothing, with a
written brief for the model to give the user.

The assertions people will be tempted to weaken later are the ones about what
does NOT spend budget (queued build-slot polls, locally rejected calls), what
does NOT reset it (anything short of a new user turn — compression especially),
and the one case where the gate deliberately stands down entirely: no trusted
turn identity, no budget.
"""

import json
import urllib.error

from unittest.mock import patch

import pytest

from agent import app_change_budget
from gateway import session_context
from tests.tools._profile_scope import mux_profile_scope
from tools.apphost_tool import app_host_tool

_BASE_URL = "http://127.0.0.1:18080/api/v1/internal/apphost"
_SCOPE = {
    "ZET_APPHOST_BASE_URL": _BASE_URL,
    "ZETTLAB_AGENT_ACTION_TOKEN": "profile-apphost-token",
}

# The refusal has to survive a careless edit: an "all clear, stopping" message
# that forgets this clause invites the model to tidy up by deleting the app.
_NO_SELF_SERVICE_UNDO = "不许自行回滚或删除"


class _Headers:
    def get(self, name, default=None):
        if name.lower() == "content-type":
            return "application/json"
        return default


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status
        self.headers = _Headers()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, *_):
        return json.dumps(self._payload).encode("utf-8")


def _responder(payload=None, status=200, calls=None):
    def _open(req, timeout=None):
        if calls is not None:
            calls.append(req)
        return _Resp({} if payload is None else payload, status=status)

    return _open


def _error_responder(status, body, calls=None):
    """An App Host answer that failed at the server, with a parsable error body."""

    class _Body:
        def read(self, *_):
            return json.dumps(body).encode("utf-8")

    def _open(req, timeout=None):
        if calls is not None:
            calls.append(req)
        raise urllib.error.HTTPError(_BASE_URL, status, "boom", None, _Body())

    return _open


class _Turn:
    """Bind / rebind the trusted per-request turn identity, as the gateway does."""

    def __init__(self):
        self._tokens = None

    def start(self, turn_id):
        self.end()
        self._tokens = session_context.set_turn_vars(turn_id=turn_id)

    def end(self):
        if self._tokens is not None:
            session_context.clear_turn_vars(self._tokens)
            self._tokens = None


@pytest.fixture
def turn(monkeypatch):
    monkeypatch.setattr(session_context, "_session_context_engaged", False, raising=False)
    app_change_budget.reset_all_for_tests()
    handle = _Turn()
    handle.start("turn-1")
    try:
        yield handle
    finally:
        handle.end()
        app_change_budget.reset_all_for_tests()


def _publish(monkeypatch, slug, opener):
    with mux_profile_scope(monkeypatch, dict(_SCOPE)):
        with patch("tools.apphost_tool._urlopen", opener):
            return json.loads(
                app_host_tool(
                    {"action": "publish", "mode": "reload", "in_place": True, "slug": slug}
                )
            )


def _acquire(monkeypatch, opener):
    with mux_profile_scope(monkeypatch, dict(_SCOPE)):
        with patch("tools.apphost_tool._urlopen", opener):
            return json.loads(app_host_tool({"action": "acquire_slot"}))


def _call(monkeypatch, args, opener):
    with mux_profile_scope(monkeypatch, dict(_SCOPE)):
        with patch("tools.apphost_tool._urlopen", opener):
            return json.loads(app_host_tool(args))


def test_one_delivery_and_three_repairs_pass_then_the_next_is_refused_locally(
    monkeypatch, turn
):
    calls = []
    for attempt in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        out = _publish(monkeypatch, "diary", _responder({"version": "v1"}, calls=calls))
        assert out["ok"] is True, f"attempt {attempt} should have been allowed"
    assert len(calls) == app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN

    refused = _publish(monkeypatch, "diary", _responder({"version": "v1"}, calls=calls))

    assert refused["ok"] is False
    assert refused["error"]["code"] == app_change_budget.PUBLISH_BUDGET_CODE
    # status 0 is the tool's "not one byte was sent" tier — a resend is
    # pointless, which is exactly what the model must conclude.
    assert refused["status"] == 0
    assert len(calls) == app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN, (
        "the refused publish must never reach App Host"
    )
    message = refused["error"]["message"]
    assert "diary" in message
    assert _NO_SELF_SERVICE_UNDO in message


def test_the_budget_is_per_app_not_per_turn(monkeypatch, turn):
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        _publish(monkeypatch, "diary", _responder({"version": "v1"}))
    assert _publish(monkeypatch, "diary", _responder({}))["ok"] is False

    # A second app in the same turn starts with its own full allowance: the
    # user asking for two apps is not the runaway this guards against.
    assert _publish(monkeypatch, "ledger", _responder({"version": "v1"}))["ok"] is True


def test_a_locally_rejected_publish_does_not_spend_the_budget(monkeypatch, turn):
    calls = []
    for _ in range(10):
        # No mode → _BadRequest → invalid_request, status 0, nothing sent.
        bad = _call(monkeypatch, {"action": "publish", "slug": "diary"}, _responder(calls=calls))
        assert bad["error"]["code"] == "invalid_request"
    assert calls == []

    # A malformed argument is not a repair round; the full allowance is intact.
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        assert _publish(monkeypatch, "diary", _responder({"version": "v1"}))["ok"] is True
    assert _publish(monkeypatch, "diary", _responder({}))["ok"] is False


def test_a_publish_that_failed_at_the_server_still_spends_the_budget(monkeypatch, turn):
    failure = {"code": "build_failed", "message": "vet 报错"}
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        out = _publish(monkeypatch, "diary", _error_responder(500, failure))
        assert out["ok"] is False and out["error"]["code"] == "build_failed"

    # "It failed, so it doesn't count" is the reasoning that produced the
    # three-hour turn. A failed round is still a round.
    refused = _publish(monkeypatch, "diary", _responder({}))
    assert refused["error"]["code"] == app_change_budget.PUBLISH_BUDGET_CODE


def test_queued_build_slot_polls_do_not_spend_the_build_budget(monkeypatch, turn):
    # The skill polls acquire_slot every 10s while queued; counting polls would
    # burn the whole allowance inside a minute of waiting for the device.
    for _ in range(30):
        queued = _acquire(monkeypatch, _responder({"queue_ahead": 2}))
        assert queued["ok"] is True and "token" not in queued["data"]

    for _ in range(app_change_budget.MAX_BUILD_ROUNDS_PER_TURN):
        granted = _acquire(monkeypatch, _responder({"token": "slot-1"}))
        assert granted["ok"] is True

    refused = _acquire(monkeypatch, _responder({"token": "slot-1"}))
    assert refused["ok"] is False
    assert refused["error"]["code"] == app_change_budget.BUILD_BUDGET_CODE
    assert refused["status"] == 0
    assert _NO_SELF_SERVICE_UNDO in refused["error"]["message"]


def test_the_refusal_names_the_work_site_prepare_handed_back(monkeypatch, turn):
    site = "/volume1/subvol/apps/diary"
    prepared = _call(
        monkeypatch, {"action": "prepare", "slug": "diary"}, _responder({"dir": site})
    )
    assert prepared["data"]["dir"] == site

    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        _publish(monkeypatch, "diary", _responder({"version": "v1"}))
    refused = _publish(monkeypatch, "diary", _responder({}))

    # Telling the model to "explain where the work is" without the path makes
    # it guess; the platform already knows, so it says it.
    assert site in refused["error"]["message"]


def test_a_new_user_turn_clears_the_budget(monkeypatch, turn):
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        _publish(monkeypatch, "diary", _responder({"version": "v1"}))
    assert _publish(monkeypatch, "diary", _responder({}))["ok"] is False

    turn.start("turn-2")
    assert _publish(monkeypatch, "diary", _responder({"version": "v1"}))["ok"] is True


def test_nothing_short_of_a_new_user_turn_clears_the_budget(monkeypatch, turn):
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        _publish(monkeypatch, "diary", _responder({"version": "v1"}))

    # Everything a model does after being compressed — re-reading its bearings,
    # re-preparing the app, polling the slot — happens inside the same turn and
    # keeps the same turn binding. None of it is a fresh allowance.
    _call(monkeypatch, {"action": "probe"}, _responder({"free_bytes": 1}))
    _call(monkeypatch, {"action": "logs", "slug": "diary"}, _responder({"lines": []}))
    _call(
        monkeypatch,
        {"action": "prepare", "slug": "diary"},
        _responder({"dir": "/volume1/subvol/apps/diary"}),
    )
    _acquire(monkeypatch, _responder({"queue_ahead": 1}))

    refused = _publish(monkeypatch, "diary", _responder({}))
    assert refused["error"]["code"] == app_change_budget.PUBLISH_BUDGET_CODE


def test_without_a_trusted_turn_identity_the_gate_stands_down(monkeypatch, turn):
    """No attested turn, no budget — and no refusal either.

    A per-turn promise the platform cannot attest is worse than no promise: the
    model would be refused on a boundary nobody drew, with no user message able
    to clear it. Entry points that bind no turn identity therefore pass through
    untouched rather than falling back to some process-local notion of a turn.
    """
    turn.end()
    assert session_context.current_turn_identity() is None

    calls = []
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN * 5):
        out = _publish(monkeypatch, "diary", _responder({"version": "v1"}, calls=calls))
        assert out["ok"] is True
    assert len(calls) == app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN * 5

    for _ in range(app_change_budget.MAX_BUILD_ROUNDS_PER_TURN * 5):
        assert _acquire(monkeypatch, _responder({"token": "slot-1"}))["ok"] is True


def test_bookkeeping_failure_never_breaks_the_tool_call(monkeypatch, turn):
    with patch.object(
        app_change_budget, "_ledger", side_effect=RuntimeError("ledger exploded")
    ):
        out = _publish(monkeypatch, "diary", _responder({"version": "v1"}))
    assert out["ok"] is True
