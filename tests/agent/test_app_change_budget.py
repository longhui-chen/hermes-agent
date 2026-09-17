"""The per-user-turn budget on rebuilding / republishing a generated app.

Background: the app-builder skill has always told the model to stop after about
three failed repair rounds and hand back to the user, and nothing counted. One
board turn ran three hours and 431 tool calls republishing the same app. These
tests pin the two levels that now count:

* the soft gate in ``tools.apphost_tool`` — refuses locally, sends nothing, and
  hands the model a written brief for the user;
* the per-turn loop cap in ``agent.tool_guardrails`` — ends the turn for a model
  that ignores the refusal and keeps submitting.

The assertions people will be tempted to weaken later are the ones about what
does NOT spend budget (queued build-slot polls, locally rejected calls) and what
does NOT reset it (anything short of a new user turn — compression especially).
"""

import json
import urllib.error

from unittest.mock import patch

import pytest

from agent import app_change_budget
from agent.tool_guardrails import (
    APP_PUBLISH_CAP_CODE,
    LoopCapConfig,
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
)
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


# ── level one: the soft gate in the tool ─────────────────────────────────


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
        with mux_profile_scope(monkeypatch, dict(_SCOPE)):
            with patch("tools.apphost_tool._urlopen", _responder(calls=calls)):
                bad = json.loads(app_host_tool({"action": "publish", "slug": "diary"}))
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
    with mux_profile_scope(monkeypatch, dict(_SCOPE)):
        with patch("tools.apphost_tool._urlopen", _responder({"dir": site})):
            prepared = json.loads(app_host_tool({"action": "prepare", "slug": "diary"}))
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

    # Compression runs inside the turn and never rebinds the turn identity, so
    # the ledger it would have to clear is not reachable from there. The
    # fallback generation used by unbound entry points must not clear a bound
    # turn's ledger either.
    app_change_budget.note_turn_boundary()
    app_change_budget.note_turn_boundary()

    refused = _publish(monkeypatch, "diary", _responder({}))
    assert refused["error"]["code"] == app_change_budget.PUBLISH_BUDGET_CODE


def test_bookkeeping_failure_never_breaks_the_tool_call(monkeypatch, turn):
    with patch.object(
        app_change_budget, "_ledger", side_effect=RuntimeError("ledger exploded")
    ):
        out = _publish(monkeypatch, "diary", _responder({"version": "v1"}))
    assert out["ok"] is True


# ── level two: the per-turn loop cap ─────────────────────────────────────


def _controller(**caps):
    return ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            # Explicitly off: this cap must fire anyway, like max_web_searches.
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(**caps),
        )
    )


def test_the_loop_cap_ends_the_turn_when_the_soft_refusal_is_ignored():
    controller = _controller(max_app_publishes=6)
    args = {"action": "publish", "mode": "reload", "in_place": True, "slug": "diary"}
    for attempt in range(6):
        assert controller.before_call("app_host", args).allows_execution, attempt

    decision = controller.before_call("app_host", args)
    assert decision.action == "block"
    assert decision.code == APP_PUBLISH_CAP_CODE
    assert controller.halt_decision is decision
    assert "diary" in decision.message
    assert _NO_SELF_SERVICE_UNDO in decision.message


def test_the_loop_cap_only_counts_submissions():
    controller = _controller(max_app_publishes=2)
    for _ in range(50):
        # Polling for a build slot, reading logs and probing storage are how a
        # careful run behaves; none of them submits a version.
        assert controller.before_call("app_host", {"action": "acquire_slot"}).allows_execution
        assert controller.before_call("app_host", {"action": "logs", "slug": "diary"}).allows_execution
        assert controller.before_call("app_host", {"action": "probe"}).allows_execution
    assert controller.before_call(
        "app_host", {"action": "reload", "slug": "diary", "staging_dir": "/tmp/s"}
    ).allows_execution


def test_the_loop_cap_resets_with_the_turn_and_can_be_disabled():
    controller = _controller(max_app_publishes=1)
    args = {"action": "publish", "mode": "reload", "in_place": True, "slug": "diary"}
    assert controller.before_call("app_host", args).allows_execution
    assert controller.before_call("app_host", args).action == "block"

    controller.reset_for_turn()
    assert controller.before_call("app_host", args).allows_execution

    unlimited = _controller(max_app_publishes=0)
    for _ in range(50):
        assert unlimited.before_call("app_host", args).allows_execution


def test_loop_cap_config_reads_max_app_publishes_from_config():
    parsed = ToolCallGuardrailConfig.from_mapping(
        {"loop_caps": {"max_app_publishes": 3}}
    )
    assert parsed.loop_caps.max_app_publishes == 3
    # 0 disables, junk falls back to the default — same contract as the other caps.
    assert ToolCallGuardrailConfig.from_mapping(
        {"loop_caps": {"max_app_publishes": 0}}
    ).loop_caps.max_app_publishes == 0
    assert ToolCallGuardrailConfig.from_mapping(
        {"loop_caps": {"max_app_publishes": "nonsense"}}
    ).loop_caps.max_app_publishes == LoopCapConfig().max_app_publishes


def test_the_halt_message_carries_the_rounds_and_the_site(monkeypatch, turn):
    site = "/volume1/subvol/apps/diary"
    with mux_profile_scope(monkeypatch, dict(_SCOPE)):
        with patch("tools.apphost_tool._urlopen", _responder({"dir": site})):
            app_host_tool({"action": "prepare", "slug": "diary"})
    for _ in range(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN):
        _publish(monkeypatch, "diary", _responder({"version": "v1"}))

    controller = _controller(max_app_publishes=1)
    args = {"action": "publish", "mode": "reload", "in_place": True, "slug": "diary"}
    controller.before_call("app_host", args)
    decision = controller.before_call("app_host", args)

    assert str(app_change_budget.MAX_PUBLISHES_PER_APP_PER_TURN) in decision.message
    assert site in decision.message
