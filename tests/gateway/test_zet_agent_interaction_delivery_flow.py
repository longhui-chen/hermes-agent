import gc
import json
import queue
import threading
import time
import weakref
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import zet_agent as zet_agent_module
from gateway.platforms import zet_agent_metrics
from gateway.platforms.zet_agent import ZetAgentAdapter, _ClarifyEntry
from gateway.session_context import clear_turn_vars, set_turn_vars
from tools import approval


class _LiveTask:
    def done(self):
        return False


class _WeakOwner:
    pass


@pytest.fixture(autouse=True)
def clear_approval_state():
    approval._gateway_queues.clear()
    approval._gateway_notify_cbs.clear()
    approval._gateway_prepared.clear()
    yield
    approval._gateway_queues.clear()
    approval._gateway_notify_cbs.clear()
    approval._gateway_prepared.clear()


def _adapter() -> ZetAgentAdapter:
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(
            on_interaction_pending=lambda _sid, **_kwargs: None,
            on_interaction_resolved=lambda _sid: None,
        ),
    )
    return adapter


def _app(adapter: ZetAgentAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get(
        "/v1/sessions/{session_id}/pending", adapter._handle_pending
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/approval/respond",
        adapter._handle_approval_respond,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/clarify/respond",
        adapter._handle_clarify_respond,
    )
    app.router.add_get(
        "/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}",
        adapter._handle_interaction_delivery,
    )
    app.router.add_post(
        "/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}/recovery-fence",
        adapter._handle_recovery_fence,
    )
    return app


def _auth():
    return {"Authorization": "Bearer test-key"}


async def _claim_and_ack(cli, delivery_id, receipt, session_id="session-1"):
    assert receipt["state"] == "active"
    assert receipt["fence_id"]
    claim = await cli.post(
        f"/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}/recovery-fence",
        headers=_auth(),
        json={
            "action": "claim",
            "expected_state_revision": receipt["state_revision"],
        },
    )
    claim_body = await claim.json()
    assert claim.status == 200
    assert claim_body["fence_id"] == receipt["fence_id"]
    ack = await cli.post(
        f"/v1/sessions/{session_id}/interaction-deliveries/{delivery_id}/recovery-fence",
        headers=_auth(),
        json={
            "action": "ack",
            "fence_id": receipt["fence_id"],
            "expected_state_revision": receipt["state_revision"],
        },
    )
    assert ack.status == 200
    return await ack.json()


def _notify_approval(adapter, interaction_id, turn_id, session_id="session-1"):
    data = {
        "interaction_id": interaction_id,
        "command": interaction_id,
        "description": "test",
    }
    queue_key = adapter._interaction_queue_key(session_id)
    entry = approval.enqueue_gateway_approval(queue_key, data)
    tokens = set_turn_vars(turn_id=turn_id)
    try:
        adapter._make_approval_cb(queue.Queue(), session_id, queue_key)(data)
    finally:
        clear_turn_vars(tokens)
    return entry


def _register_live_turn(adapter, turn_id):
    key = adapter._active_turn_key("session-1")
    adapter._active_session_tasks[key] = _LiveTask()
    adapter._active_session_turn_ids[key] = turn_id


def test_legacy_fifo_each_approval_admission_emits_opened_once():
    zet_agent_metrics.reset_for_tests()
    adapter = _adapter()
    _notify_approval(adapter, "interaction-durable", "turn-durable")
    _notify_approval(adapter, "interaction-follow-up", "turn-durable")
    snapshot = zet_agent_metrics.snapshot()
    assert snapshot["interaction_opened"] == 2


@pytest.mark.asyncio
async def test_approval_delivery_prepare_finalize_is_exact_and_idempotent():
    adapter = _adapter()
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    second = _notify_approval(adapter, "interaction-b", "turn-1")
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        stale = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-b",
                "interaction_id": "interaction-b",
                "phase": "prepare",
            },
        )
        assert stale.status == 409
        assert not first.event.is_set()
        assert not second.event.is_set()

        request = {
            "choice": "once",
            "delivery_id": "delivery-a",
            "interaction_id": "interaction-a",
            "phase": "prepare",
        }
        prepared = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json=request,
        )
        prepared_body = await prepared.json()
        assert prepared.status == 200
        assert prepared_body["resolved"] == 0
        assert prepared_body["state"] == "prepared"
        assert prepared_body["turn_id"] == "turn-1"
        assert len(prepared_body["payload_digest"]) == 64
        assert not first.event.is_set()

        duplicate = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json=request,
        )
        assert await duplicate.json() == prepared_body

        conflict = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**request, "choice": "deny"},
        )
        assert conflict.status == 409

        receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert (await receipt.json())["state"] == "prepared"

        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**request, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized.status == 200
        assert finalized_body["resolved"] == 1
        assert finalized_body["state"] == "pending"
        assert first.event.is_set()
        assert first.result == "once"
        assert not second.event.is_set()

        # The second interaction belongs to the same turn, so recovery must
        # report pending instead of incorrectly claiming the turn is active.
        receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert (await receipt.json())["state"] == "pending"

        duplicate_finalize = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**request, "phase": "finalize"},
        )
        assert duplicate_finalize.status == 200
        assert not second.event.is_set()


@pytest.mark.asyncio
async def test_new_protocol_empty_approval_is_404_and_records_no_receipt():
    adapter = _adapter()
    request = {
        "choice": "once",
        "delivery_id": "delivery-empty",
        "interaction_id": "interaction-empty",
        "phase": "prepare",
    }
    async with TestClient(TestServer(_app(adapter))) as cli:
        response = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json=request,
        )
        assert response.status == 404
        receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-empty",
            headers=_auth(),
        )
        assert receipt.status == 404


@pytest.mark.asyncio
async def test_approval_finalize_after_local_restart_does_not_require_raw_choice():
    adapter = _adapter()
    entry = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        prepared = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
                "phase": "prepare",
            },
        )
        assert prepared.status == 200
        prepared_body = await prepared.json()
        assert "choice" not in prepared_body
        assert "response" not in prepared_body
        assert adapter._prepared_raw_bytes == len("once".encode())

        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
                "phase": "finalize",
            },
        )
        assert finalized.status == 200
        assert entry.result == "once"
        finalized_body = await finalized.json()
        assert not entry.event.is_set()
        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert entry.event.is_set()
        assert adapter._prepared_raw_bytes == 0


@pytest.mark.asyncio
async def test_finalize_rejects_mismatched_raw_then_uses_prepared_clarify_response():
    adapter = _adapter()
    payload = {
        "type": "hermes.clarify",
        "interaction_id": "clarify-a",
        "interaction_generation": 1,
        "turn_id": "turn-1",
        "question": "first",
        "choices_offered": [],
    }
    entry = _ClarifyEntry("clarify-a", "turn-1", payload, 1)
    queue_key = adapter._interaction_queue_key("session-1")
    adapter._clarify_queues[queue_key] = [entry]
    adapter._pending_clarify[queue_key] = [payload]
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        prepared = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={
                "response": "原始回答",
                "delivery_id": "delivery-a",
                "interaction_id": "clarify-a",
                "phase": "prepare",
            },
        )
        assert prepared.status == 200
        assert adapter._prepared_raw_bytes == len("原始回答".encode("utf-8"))

        mismatch = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={
                "response": "篡改回答",
                "delivery_id": "delivery-a",
                "interaction_id": "clarify-a",
                "phase": "finalize",
            },
        )
        assert mismatch.status == 409
        assert not entry.event.is_set()

        finalized = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={
                "delivery_id": "delivery-a",
                "interaction_id": "clarify-a",
                "phase": "finalize",
            },
        )
        assert finalized.status == 200
        assert entry.response == "原始回答"
        finalized_body = await finalized.json()
        assert not entry.event.is_set()
        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert entry.event.is_set()
        assert adapter._prepared_raw_bytes == 0


@pytest.mark.asyncio
async def test_prepared_raw_budget_rejects_without_leaking_lease(monkeypatch):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_PREPARED_RAW_BUDGET", 3)
    adapter = _adapter()
    entry = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        rejected = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
                "phase": "prepare",
            },
        )
        assert rejected.status == 503
        assert adapter._prepared_raw_bytes == 0

        legacy = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 200
        assert entry.event.is_set()


@pytest.mark.asyncio
async def test_durable_session_capacity_keeps_unnegotiated_legacy_available(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    adapter = _adapter()
    durable_entry = _notify_approval(
        adapter, "interaction-durable", "turn-durable", "durable-session"
    )
    legacy_entry = _notify_approval(
        adapter, "interaction-legacy", "turn-legacy", "legacy-session"
    )

    async with TestClient(TestServer(_app(adapter))) as cli:
        durable_prepare = await cli.post(
            "/v1/sessions/durable-session/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-durable",
                "interaction_id": "interaction-durable",
                "phase": "prepare",
            },
        )
        assert durable_prepare.status == 200

        legacy_prepare = await cli.post(
            "/v1/sessions/legacy-session/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-legacy",
                "interaction_id": "interaction-legacy",
                "phase": "prepare",
            },
        )
        assert legacy_prepare.status == 503
        assert (await legacy_prepare.json())["error"]["code"] == (
            "interaction_delivery_capacity"
        )

        legacy_response = await cli.post(
            "/v1/sessions/legacy-session/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy_response.status == 200
        assert await legacy_response.json() == {"resolved": 1}
        assert legacy_entry.event.is_set()

        durable_downgrade = await cli.post(
            "/v1/sessions/durable-session/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert durable_downgrade.status == 409
        assert (await durable_downgrade.json())["error"]["code"] == (
            "durable_interaction_required"
        )
        assert not durable_entry.event.is_set()


@pytest.mark.asyncio
async def test_durable_session_capacity_releases_unnegotiated_clarify_lease(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    adapter = _adapter()
    _notify_approval(adapter, "interaction-durable", "turn-durable", "durable-session")
    payload = {
        "type": "hermes.clarify",
        "interaction_id": "clarify-legacy",
        "interaction_generation": 1,
        "turn_id": "turn-legacy",
        "question": "legacy question",
        "choices_offered": [],
    }
    clarify_entry = _ClarifyEntry("clarify-legacy", "turn-legacy", payload, 1)
    clarify_key = adapter._interaction_queue_key("legacy-session")
    adapter._clarify_queues[clarify_key] = [clarify_entry]
    adapter._pending_clarify[clarify_key] = [payload]

    async with TestClient(TestServer(_app(adapter))) as cli:
        durable_prepare = await cli.post(
            "/v1/sessions/durable-session/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-durable",
                "interaction_id": "interaction-durable",
                "phase": "prepare",
            },
        )
        assert durable_prepare.status == 200

        clarify_prepare = await cli.post(
            "/v1/sessions/legacy-session/clarify/respond",
            headers=_auth(),
            json={
                "response": "legacy answer",
                "delivery_id": "delivery-legacy",
                "interaction_id": "clarify-legacy",
                "phase": "prepare",
            },
        )
        assert clarify_prepare.status == 503
        assert clarify_entry.prepared_delivery_id is None
        assert clarify_entry.prepared_until == 0.0

        legacy_response = await cli.post(
            "/v1/sessions/legacy-session/clarify/respond",
            headers=_auth(),
            json={"response": "legacy answer"},
        )
        assert legacy_response.status == 200
        assert await legacy_response.json() == {"resolved": 1}
        assert clarify_entry.response == "legacy answer"
        assert clarify_entry.event.is_set()


def test_durable_session_ttl_reclaim_keeps_every_live_evidence(monkeypatch):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_TTL", 0)
    adapter = _adapter()
    old_key = adapter._interaction_queue_key("old-session")
    new_key = adapter._interaction_queue_key("new-session")

    with adapter._delivery_lock:
        assert adapter._mark_durable_session_locked(old_key)
        old_digest = adapter._durable_session_digest(old_key)
        adapter._durable_session_digests[old_digest]["last_active"] = 0.0

        # Prepared, finalized, and recovery-fenced receipts are all exact
        # delivery evidence. None may be evicted merely to admit a new scope.
        for index, receipt in enumerate(
            (
                {
                    "scope_key": old_key,
                    "finalized": False,
                    "prepare_expires_at": time.monotonic() + 60,
                },
                {
                    "scope_key": old_key,
                    "finalized": True,
                    "expires_at": time.monotonic() + 60,
                },
                {
                    "scope_key": old_key,
                    "finalized": True,
                    "expires_at": time.monotonic() + 60,
                    "_fence_event": threading.Event(),
                },
            )
        ):
            receipt_key = (old_key, f"delivery-{index}")
            adapter._interaction_deliveries[receipt_key] = receipt
            assert not adapter._mark_durable_session_locked(new_key)
            adapter._interaction_deliveries.pop(receipt_key)

        adapter._pending_approval[old_key] = [
            {"interaction_id": "approval-pending"}
        ]
        assert not adapter._mark_durable_session_locked(new_key)
        adapter._pending_approval.clear()

        assert adapter._pin_durable_session_source_locked(
            old_key, "approval", "source-only"
        )
        assert not adapter._mark_durable_session_locked(new_key)
        adapter._release_durable_session_source_locked(
            old_key, "approval", "source-only"
        )

        clarify_payload = {"interaction_id": "clarify-pending"}
        adapter._clarify_queues[old_key] = [
            _ClarifyEntry(
                "clarify-pending",
                "turn-old",
                clarify_payload,
                1,
            )
        ]
        assert not adapter._mark_durable_session_locked(new_key)
        adapter._clarify_queues.clear()

        adapter._active_session_tasks[old_key] = _LiveTask()
        assert not adapter._mark_durable_session_locked(new_key)
        adapter._active_session_tasks.clear()

        assert adapter._mark_durable_session_locked(new_key)
        assert old_digest not in adapter._durable_session_digests
        assert adapter._durable_session_digest(new_key) in (
            adapter._durable_session_digests
        )


def test_durable_session_tombstone_waits_for_ttl_without_live_evidence(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_TTL", 60)
    adapter = _adapter()
    old_key = adapter._interaction_queue_key("old-session")
    new_key = adapter._interaction_queue_key("new-session")

    with adapter._delivery_lock:
        assert adapter._mark_durable_session_locked(old_key)
        assert not adapter._mark_durable_session_locked(new_key)
        old_digest = adapter._durable_session_digest(old_key)
        adapter._durable_session_digests[old_digest]["last_active"] = 0.0
        assert adapter._mark_durable_session_locked(new_key)
        assert old_digest not in adapter._durable_session_digests


@pytest.mark.asyncio
async def test_terminal_durable_session_ttl_reclaim_allows_new_prepare_flow(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_TTL", 0)
    adapter = _adapter()
    old_entry = _notify_approval(
        adapter, "interaction-old", "turn-old", "old-session"
    )
    new_entry = _notify_approval(
        adapter, "interaction-new", "turn-new", "new-session"
    )
    old_body = {
        "choice": "once",
        "delivery_id": "delivery-old",
        "interaction_id": "interaction-old",
    }
    new_body = {
        "choice": "once",
        "delivery_id": "delivery-new",
        "interaction_id": "interaction-new",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (
            await cli.post(
                "/v1/sessions/old-session/approval/respond",
                headers=_auth(),
                json={**old_body, "phase": "prepare"},
            )
        ).status == 200

        blocked = await cli.post(
            "/v1/sessions/new-session/approval/respond",
            headers=_auth(),
            json={**new_body, "phase": "prepare"},
        )
        assert blocked.status == 503

        finalized = await cli.post(
            "/v1/sessions/old-session/approval/respond",
            headers=_auth(),
            json={**old_body, "phase": "finalize"},
        )
        assert finalized.status == 200
        assert old_entry.event.is_set()

        # Even a terminal finalized receipt is downgrade evidence until its
        # receipt TTL ends; legacy must remain rejected during that window.
        legacy = await cli.post(
            "/v1/sessions/old-session/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 409
        assert (await legacy.json())["error"]["code"] == (
            "durable_interaction_required"
        )
        still_blocked = await cli.post(
            "/v1/sessions/new-session/approval/respond",
            headers=_auth(),
            json={**new_body, "phase": "prepare"},
        )
        assert still_blocked.status == 503

        old_receipt_key = adapter._delivery_key("old-session", "delivery-old")
        adapter._interaction_deliveries[old_receipt_key]["expires_at"] = 0
        admitted = await cli.post(
            "/v1/sessions/new-session/approval/respond",
            headers=_auth(),
            json={**new_body, "phase": "prepare"},
        )
        assert admitted.status == 200
        assert not new_entry.event.is_set()
        assert not adapter._is_durable_session_locked(
            adapter._interaction_queue_key("old-session")
        )


@pytest.mark.asyncio
async def test_prepare_receipt_expiry_releases_raw_budget_and_underlying_lease():
    adapter = _adapter()
    entry = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        prepared = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
                "phase": "prepare",
            },
        )
        assert prepared.status == 200
        receipt_key = adapter._delivery_key("session-1", "delivery-a")
        adapter._interaction_deliveries[receipt_key]["prepare_expires_at"] = 0
        approval._gateway_prepared[(adapter._interaction_queue_key("session-1"), "interaction-a")] = (
            "delivery-a",
            0,
        )

        expired = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert expired.status == 404
        assert adapter._prepared_raw_bytes == 0

        legacy = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 409

        replacement = {
            "choice": "deny",
            "delivery_id": "delivery-b",
            "interaction_id": "interaction-a",
        }
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**replacement, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**replacement, "phase": "finalize"},
        )
        assert finalized.status == 200
        finalized_body = await finalized.json()
        assert not entry.event.is_set()
        await _claim_and_ack(cli, "delivery-b", finalized_body)
        assert entry.event.is_set()


@pytest.mark.asyncio
async def test_clarify_prepare_rejects_response_larger_than_16_kib():
    adapter = _adapter()
    payload = {
        "type": "hermes.clarify",
        "interaction_id": "clarify-a",
        "interaction_generation": 1,
        "turn_id": "turn-1",
        "question": "first",
        "choices_offered": [],
    }
    entry = _ClarifyEntry("clarify-a", "turn-1", payload, 1)
    queue_key = adapter._interaction_queue_key("session-1")
    adapter._clarify_queues[queue_key] = [entry]
    adapter._pending_clarify[queue_key] = [payload]

    async with TestClient(TestServer(_app(adapter))) as cli:
        rejected = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={
                "response": "x" * (16 * 1024 + 1),
                "delivery_id": "delivery-a",
                "interaction_id": "clarify-a",
                "phase": "prepare",
            },
        )

    assert rejected.status == 413
    assert not entry.event.is_set()
    assert adapter._prepared_raw_bytes == 0


@pytest.mark.asyncio
async def test_receipt_active_requires_the_exact_live_turn_then_becomes_terminal():
    adapter = _adapter()
    entry = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")
    body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        prepared = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "prepare"},
        )
        assert prepared.status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        assert finalized.status == 200
        finalized_body = await finalized.json()
        assert not entry.event.is_set()
        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert entry.event.is_set()

        receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert (await receipt.json())["state"] == "active"

        key = adapter._active_turn_key("session-1")
        adapter._active_session_turn_ids[key] = "turn-2"
        receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert (await receipt.json())["state"] == "terminal"


@pytest.mark.asyncio
async def test_legacy_approval_post_keeps_single_phase_empty_queue_contract():
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        response = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "once"},
        )
        assert response.status == 200
        assert await response.json() == {"resolved": 0}


def test_legacy_projection_keeps_source_interaction_id():
    """Callers that invoke the approval callback without a durable generation
    (legacy path) still get a mirror that carries the source-owned
    ``interaction_id``: /pending matches mirrors by that id and local-server
    keys the approval card by it (codex P1 on chat-ui-b0)."""
    adapter = _adapter()
    queue_key = adapter._interaction_queue_key("session-1")
    data = {
        "approval_id": "approval-legacy",
        "interaction_id": "interaction-legacy",
        "command": "rm -rf build",
        "description": "test",
    }
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        adapter._make_approval_cb(queue.Queue(), "session-1", queue_key)(data)
    finally:
        clear_turn_vars(tokens)
    mirrors = adapter._pending_approval[queue_key]
    assert len(mirrors) == 1
    assert mirrors[0]["approval_id"] == "approval-legacy"
    assert mirrors[0]["interaction_id"] == "interaction-legacy"
    assert mirrors[0]["turn_id"] == "turn-1"


@pytest.mark.asyncio
async def test_pending_approval_follows_source_fifo_not_callback_arrival_order():
    adapter = _adapter()
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    _notify_approval(adapter, "interaction-b", "turn-1")
    adapter._pending_approval[adapter._interaction_queue_key("session-1")].reverse()

    async with TestClient(TestServer(_app(adapter))) as cli:
        response = await cli.get(
            "/v1/sessions/session-1/pending",
            headers=_auth(),
        )
        body = await response.json()

    assert response.status == 200
    assert body["approval"]["interaction_id"] == "interaction-a"
    assert body["approval"]["interaction_delivery_version"] == 1
    assert {item["interaction_id"] for item in body["approvals"]} == {
        "interaction-a", "interaction-b"
    }

    # An old local-server ignores the additive version field and still uses
    # the legacy FIFO POST. Until this scoped session successfully prepares a
    # durable receipt, that compatibility path remains available.
    async with TestClient(TestServer(_app(adapter))) as cli:
        legacy = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 200
        assert await legacy.json() == {"resolved": 1}
    assert first.event.is_set()


@pytest.mark.asyncio
async def test_receipt_cap_never_evicts_active_or_pending_recovery_evidence(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_RECEIPT_CAP", 1)
    adapter = _adapter()
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")
    first_body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**first_body, "phase": "finalize"},
        )
        assert finalized.status == 200
        finalized_body = await finalized.json()
        assert not first.event.is_set()
        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert first.event.is_set()

        second = _notify_approval(adapter, "interaction-b", "turn-1")
        second_prepare = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "choice": "deny",
                "delivery_id": "delivery-b",
                "interaction_id": "interaction-b",
                "phase": "prepare",
            },
        )
        assert second_prepare.status == 503

        first_receipt = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert first_receipt.status == 200
        assert (await first_receipt.json())["state"] == "pending"

        # Capacity rollback releases the source lease. The public legacy path
        # stays disabled after this session used durable delivery, but an
        # internal resolver can prove the entry itself was not wedged.
        legacy = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 409
        assert (
            approval.resolve_gateway_approval(
                adapter._interaction_queue_key("session-1"), "deny"
            )
            == 1
        )
        assert second.event.is_set()


@pytest.mark.asyncio
async def test_clarify_delivery_cannot_answer_a_different_fifo_entry():
    adapter = _adapter()
    first_payload = {
        "type": "hermes.clarify",
        "interaction_id": "clarify-a",
        "interaction_generation": 1,
        "turn_id": "turn-1",
        "question": "first",
        "choices_offered": [],
    }
    second_payload = {
        **first_payload,
        "interaction_id": "clarify-b",
        "interaction_generation": 2,
        "question": "second",
    }
    first = _ClarifyEntry("clarify-a", "turn-1", first_payload, 1)
    second = _ClarifyEntry("clarify-b", "turn-1", second_payload, 2)
    queue_key = adapter._interaction_queue_key("session-1")
    adapter._clarify_queues[queue_key] = [first, second]
    adapter._pending_clarify[queue_key] = [first_payload, second_payload]
    _register_live_turn(adapter, "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        wrong = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={
                "response": "second answer",
                "delivery_id": "delivery-b",
                "interaction_id": "clarify-b",
                "phase": "prepare",
            },
        )
        assert wrong.status == 409
        assert not first.event.is_set()

        body = {
            "response": "first answer",
            "delivery_id": "delivery-a",
            "interaction_id": "clarify-a",
            "phase": "prepare",
        }
        prepared = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json=body,
        )
        assert prepared.status == 200
        assert not first.event.is_set()

        finalized = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        assert finalized.status == 200
        assert first.response == "first answer"
        assert first.event.is_set()
        assert not second.event.is_set()


@pytest.mark.asyncio
async def test_recovery_fence_is_revision_cas_and_old_receipt_never_reactivates():
    adapter = _adapter()
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")
    first_body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
                "phase": "finalize",
            },
        )
        finalized_body = await finalized.json()
        assert finalized_body["state"] == "active"
        revision = finalized_body["state_revision"]
        assert finalized_body["fence_id"]
        assert not first.event.is_set()

        stale_claim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={"action": "claim", "expected_state_revision": revision - 1},
        )
        assert stale_claim.status == 409

        claim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={"action": "claim", "expected_state_revision": revision},
        )
        claim_body = await claim.json()
        assert claim.status == 200
        assert claim_body["state"] == "active"
        assert claim_body["state_revision"] == revision
        fence_id = claim_body["fence_id"]

        queue_key = adapter._interaction_queue_key("session-1")
        second_data = {
            "interaction_id": "interaction-b",
            "command": "second",
            "description": "test",
        }
        second = approval.enqueue_gateway_approval(queue_key, second_data)
        callback_done = threading.Event()

        def notify_second():
            tokens = set_turn_vars(turn_id="turn-1")
            try:
                adapter._make_approval_cb(
                    queue.Queue(), "session-1", queue_key
                )(second_data)
            finally:
                clear_turn_vars(tokens)
                callback_done.set()

        callback_thread = threading.Thread(target=notify_second)
        callback_thread.start()
        time.sleep(0.05)
        assert not callback_done.is_set()

        fenced = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        fenced_body = await fenced.json()
        assert fenced_body["state"] == "active"
        assert fenced_body["state_revision"] == revision
        assert fenced_body["fence_id"] == fence_id

        ack = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "ack",
                "fence_id": fence_id,
                "expected_state_revision": revision,
            },
        )
        assert ack.status == 200
        ack_body = await ack.json()
        assert ack_body["released"] is True
        assert ack_body["state"] == "active"
        assert ack_body["state_revision"] == revision
        assert ack_body["fence_id"] == fence_id
        assert first.event.is_set()
        callback_thread.join(1)
        assert callback_done.is_set()

        pending = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        pending_body = await pending.json()
        assert pending_body["state"] == "pending"
        assert pending_body["state_revision"] > revision

        legacy = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={"choice": "deny"},
        )
        assert legacy.status == 409
        assert not second.event.is_set()

        # Simulate the second approval timing out. Its stale adapter mirror
        # must not make A active again, and A can never be claimed again.
        approval._gateway_queues.pop(queue_key, None)
        terminal = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        terminal_body = await terminal.json()
        assert terminal_body["state"] == "terminal"
        assert terminal_body["state_revision"] > pending_body["state_revision"]

        reclaim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "claim",
                "expected_state_revision": terminal_body["state_revision"],
            },
        )
        assert reclaim.status == 409


@pytest.mark.asyncio
async def test_active_finalize_defers_waiter_until_claim_and_ack():
    adapter = _adapter()
    goal_events = []
    setattr(
        adapter,
        "_goals",
        lambda: SimpleNamespace(
            on_interaction_pending=lambda _sid, **_kwargs: goal_events.append(
                "pending"
            ),
            on_interaction_resolved=lambda _sid: goal_events.append("resolved"),
        ),
    )
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    assert goal_events == ["pending"]
    _register_live_turn(adapter, "turn-1")
    first_body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**first_body, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized_body["state"] == "active"
        assert finalized_body["fence_id"]
        assert not first.event.is_set()
        assert goal_events == ["pending"]

        claim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "claim",
                "expected_state_revision": finalized_body["state_revision"],
            },
        )
        claim_body = await claim.json()
        assert claim.status == 200
        assert claim_body["fence_id"] == finalized_body["fence_id"]
        assert not first.event.is_set()

        verified = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        verified_body = await verified.json()
        assert verified_body["state"] == "active"
        assert verified_body["state_revision"] == finalized_body["state_revision"]
        assert not first.event.is_set()
        assert goal_events == ["pending"]

        ack = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "ack",
                "fence_id": finalized_body["fence_id"],
                "expected_state_revision": finalized_body["state_revision"],
            },
        )
        assert ack.status == 200
        assert first.event.is_set()
        assert goal_events == ["pending", "resolved"]


@pytest.mark.asyncio
async def test_source_fifo_generation_survives_out_of_order_approval_notify(
    monkeypatch,
):
    """Source enqueue order, not notifier arrival, owns prompt generation."""
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 3},
    )
    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    queue_key = adapter._interaction_queue_key("session-1")
    notify = adapter._make_approval_cb(queue.Queue(), "session-1", queue_key)
    first_notify_entered = threading.Event()
    release_first_notify = threading.Event()
    first_notified = threading.Event()
    second_notified = threading.Event()
    results = {}

    def notify_first(data):
        first_notify_entered.set()
        assert release_first_notify.wait(2)
        notify(data)
        first_notified.set()

    def notify_second(data):
        notify(data)
        second_notified.set()

    def await_approval(name, callback):
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            results[name] = approval._await_gateway_decision(
                queue_key,
                callback,
                {
                    "interaction_id": f"interaction-{name}",
                    "command": name,
                    "description": "test",
                },
            )
        finally:
            clear_turn_vars(tokens)

    first_thread = threading.Thread(
        target=await_approval, args=("a", notify_first)
    )
    second_thread = threading.Thread(
        target=await_approval, args=("b", notify_second)
    )
    first_thread.start()
    assert first_notify_entered.wait(1)
    second_thread.start()
    assert second_notified.wait(1)
    release_first_notify.set()
    assert first_notified.wait(1)

    source = approval.list_gateway_approvals(queue_key)
    assert [item["interaction_id"] for item in source] == [
        "interaction-a",
        "interaction-b",
    ]
    assert [item["interaction_generation"] for item in source] == sorted(
        item["interaction_generation"] for item in source
    )

    body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }
    async with TestClient(TestServer(_app(adapter))) as cli:
        prepared = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "prepare"},
        )
        assert prepared.status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized.status == 200
        assert finalized_body["state"] == "pending"
        assert "fence_id" not in finalized_body

    first_thread.join(1)
    assert not first_thread.is_alive()
    assert results["a"]["choice"] == "once"
    assert second_thread.is_alive()

    assert approval.resolve_gateway_approval(queue_key, "deny") == 1
    second_thread.join(1)
    assert not second_thread.is_alive()
    assert results["b"]["choice"] == "deny"


@pytest.mark.asyncio
async def test_compaction_new_tip_routes_to_same_turn_durable_interaction():
    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    stream = queue.Queue()
    adapter._make_status_cb(stream)(
        "context.compaction",
        {
            "state": "succeeded",
            "old_session_id": "session-1",
            "new_session_id": "session-2",
        },
    )
    entry = _notify_approval(adapter, "interaction-a", "turn-1")

    async with TestClient(TestServer(_app(adapter))) as cli:
        pending = await cli.get(
            "/v1/sessions/session-2/pending", headers=_auth()
        )
        pending_body = await pending.json()
        assert pending.status == 200
        assert pending_body["approval"]["interaction_id"] == "interaction-a"

        body = {
            "choice": "once",
            "delivery_id": "delivery-a",
            "interaction_id": "interaction-a",
        }
        prepared = await cli.post(
            "/v1/sessions/session-2/approval/respond",
            headers=_auth(),
            json={**body, "phase": "prepare"},
        )
        assert prepared.status == 200
        finalized = await cli.post(
            "/v1/sessions/session-2/approval/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized.status == 200
        assert finalized_body["state"] == "active"
        acked = await _claim_and_ack(
            cli, "delivery-a", finalized_body, session_id="session-2"
        )
        assert acked["released"] is True
        assert entry.event.is_set()


@pytest.mark.asyncio
async def test_new_task_registration_atomically_invalidates_old_turn_binding():
    adapter = _adapter()
    first_task = _LiveTask()
    first_agent = SimpleNamespace()
    adapter._register_active_session_turn(
        "session-1", [first_agent], first_task
    )
    first = _notify_approval(adapter, "interaction-a", "turn-a")
    body = {
        "choice": "once",
        "delivery_id": "delivery-a",
        "interaction_id": "interaction-a",
    }

    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**body, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized_body["state"] == "active"
        with adapter._delivery_lock:
            receipt = adapter._interaction_deliveries[
                (adapter._interaction_queue_key("session-1"), "delivery-a")
            ]
            receipt["_fence_expires_at"] = time.monotonic() - 1
            adapter._expire_recovery_fences_locked()
        assert first.event.is_set()

        second_agent = SimpleNamespace()
        adapter._register_active_session_turn(
            "session-1", [second_agent], _LiveTask()
        )
        assert not adapter._remember_active_turn_id(
            "session-1", "turn-a", owner_agent=first_agent
        )
        stale = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        stale_body = await stale.json()
        assert stale_body["state"] == "terminal"
        claim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "claim",
                "expected_state_revision": stale_body["state_revision"],
            },
        )
        assert claim.status == 409


def test_latest_global_approval_callback_rejects_old_turn_caller(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 0},
    )
    adapter = _adapter()
    queue_key = adapter._interaction_queue_key("session-1")
    old_agent = SimpleNamespace(_zettlab_active_turn_id="turn-old")
    adapter._register_active_session_turn(
        "session-1", [old_agent], _LiveTask()
    )
    approval.register_gateway_notify(
        queue_key,
        adapter._make_approval_cb(
            queue.Queue(), "session-1", queue_key, old_agent
        ),
    )

    new_stream = queue.Queue()
    new_agent = SimpleNamespace(_zettlab_active_turn_id="turn-new")
    adapter._register_active_session_turn(
        "session-1", [new_agent], _LiveTask()
    )
    approval.register_gateway_notify(
        queue_key,
        adapter._make_approval_cb(
            new_stream, "session-1", queue_key, new_agent
        ),
    )

    tokens = set_turn_vars(turn_id="turn-old")
    try:
        with approval._lock:
            latest_notify = approval._gateway_notify_cbs[queue_key]
        result = approval._await_gateway_decision(
            queue_key,
            latest_notify,
            {
                "interaction_id": "interaction-old",
                "command": "old command",
                "description": "test",
            },
        )
    finally:
        clear_turn_vars(tokens)

    assert result["notify_failed"] is True
    assert new_stream.empty()
    assert adapter._active_session_turn_ids[
        adapter._active_turn_key("session-1")
    ] == "turn-new"
    assert approval.list_gateway_approvals(queue_key) == []
    with adapter._pending_lock:
        assert adapter._pending_approval.get(queue_key, []) == []


def test_global_approval_callback_does_not_retain_collected_owner(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 0},
    )
    adapter = _adapter()
    queue_key = adapter._interaction_queue_key("session-1")
    owner = _WeakOwner()
    owner._zettlab_active_turn_id = "turn-1"
    owner_ref = weakref.ref(owner)
    active_ref = [owner]
    task = _LiveTask()
    adapter._register_active_session_turn("session-1", active_ref, task)
    approval.register_gateway_notify(
        queue_key,
        adapter._make_approval_cb(
            queue.Queue(),
            "session-1",
            queue_key,
            owner,
            bound_turn_id="turn-1",
        ),
    )

    adapter._clear_active_session_turn("session-1", active_ref, task)
    active_ref[0] = None
    del owner
    gc.collect()
    assert owner_ref() is None

    with approval._lock:
        notify = approval._gateway_notify_cbs[queue_key]
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        decision = approval._await_gateway_decision(
            queue_key,
            notify,
            {
                "interaction_id": "interaction-collected",
                "command": "collected owner",
                "description": "test",
            },
        )
    finally:
        clear_turn_vars(tokens)
        approval.unregister_gateway_notify(queue_key)

    assert decision["notify_failed"] is True
    assert approval.list_gateway_approvals(queue_key) == []


def test_approval_callback_accepts_only_exact_scoped_live_owner():
    adapter = _adapter()
    first_key = adapter._interaction_queue_key("session-1")
    owner = _WeakOwner()
    owner._zettlab_active_turn_id = "turn-1"
    first_ref = [owner]
    first_task = _LiveTask()
    adapter._register_active_session_turn("session-1", first_ref, first_task)
    stream = queue.Queue()
    notify = adapter._make_approval_cb(
        stream,
        "session-1",
        first_key,
        owner,
        bound_turn_id="turn-1",
    )
    approval_data = {
        "interaction_id": "interaction-live",
        "command": "live owner",
        "description": "test",
    }
    approval.enqueue_gateway_approval(first_key, approval_data)
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        notify(approval_data)
    finally:
        clear_turn_vars(tokens)
    assert stream.get_nowait()[1]["interaction_id"] == "interaction-live"

    adapter._clear_active_session_turn("session-1", first_ref, first_task)
    second_ref = [owner]
    second_task = _LiveTask()
    adapter._register_active_session_turn("session-2", second_ref, second_task)
    stale_data = {
        "interaction_id": "interaction-cross-session",
        "interaction_generation": approval_data["interaction_generation"] + 1,
        "command": "wrong session",
        "description": "test",
    }
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        with pytest.raises(
            RuntimeError, match="approval callback active owner mismatch"
        ):
            notify(stale_data)
    finally:
        clear_turn_vars(tokens)
        getattr(notify, "_gateway_interaction_dropped")(
            first_key, "interaction-live"
        )
        approval.cancel_gateway_approvals(first_key)
        adapter._clear_active_session_turn(
            "session-2", second_ref, second_task
        )


@pytest.mark.asyncio
async def test_published_generation_watermark_prevents_second_active_fence():
    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    clarify_stream = queue.Queue()
    clarify_result = {}

    def ask_second():
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            clarify_result["value"] = adapter._make_clarify_cb(
                clarify_stream, "session-1"
            )("second", None)
        finally:
            clear_turn_vars(tokens)

    clarify_thread = threading.Thread(target=ask_second)
    clarify_thread.start()
    second_payload = clarify_stream.get(timeout=1)[1]
    second_id = second_payload["interaction_id"]
    assert (
        first.data["interaction_generation"]
        < second_payload["interaction_generation"]
    )

    async with TestClient(TestServer(_app(adapter))) as cli:
        second_body = {
            "response": "answer",
            "delivery_id": "delivery-b",
            "interaction_id": second_id,
        }
        assert (
            await cli.post(
                "/v1/sessions/session-1/clarify/respond",
                headers=_auth(),
                json={**second_body, "phase": "prepare"},
            )
        ).status == 200
        second_finalized = await cli.post(
            "/v1/sessions/session-1/clarify/respond",
            headers=_auth(),
            json={**second_body, "phase": "finalize"},
        )
        second_receipt = await second_finalized.json()
        assert second_receipt["state"] == "active"

        first_body = {
            "choice": "once",
            "delivery_id": "delivery-a",
            "interaction_id": "interaction-a",
        }
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "prepare"},
            )
        ).status == 200
        first_finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**first_body, "phase": "finalize"},
        )
        first_receipt = await first_finalized.json()
        assert first_receipt["state"] != "active"
        assert "fence_id" not in first_receipt
        assert first.event.is_set()

        await _claim_and_ack(cli, "delivery-b", second_receipt)
        clarify_thread.join(1)
        assert clarify_result["value"] == "answer"

        terminal = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        terminal_body = await terminal.json()
        assert terminal_body["state"] == "terminal"
        claim = await cli.post(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a/recovery-fence",
            headers=_auth(),
            json={
                "action": "claim",
                "expected_state_revision": terminal_body["state_revision"],
            },
        )
        assert claim.status == 409


@pytest.mark.asyncio
async def test_generation_watermark_commits_before_sse_prompt_is_visible():
    class BlockingVisibleQueue:
        def __init__(self):
            self.item = None
            self.visible = threading.Event()
            self.release = threading.Event()

        def put(self, item):
            self.item = item
            self.visible.set()
            assert self.release.wait(2)

    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    blocked_stream = BlockingVisibleQueue()
    clarify_result = {}

    def ask_second():
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            clarify_result["value"] = adapter._make_clarify_cb(
                blocked_stream, "session-1"
            )("second", None)
        finally:
            clear_turn_vars(tokens)

    clarify_thread = threading.Thread(target=ask_second)
    clarify_thread.start()
    assert blocked_stream.visible.wait(1)
    assert blocked_stream.item is not None
    second_payload = blocked_stream.item[1]
    queue_key = adapter._interaction_queue_key("session-1")
    assert adapter._published_interaction_generations[
        (queue_key, "turn-1")
    ] == second_payload["interaction_generation"]
    blocked_stream.release.set()

    try:
        async with TestClient(TestServer(_app(adapter))) as cli:
            second_body = {
                "response": "answer",
                "delivery_id": "delivery-b",
                "interaction_id": second_payload["interaction_id"],
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/clarify/respond",
                    headers=_auth(),
                    json={**second_body, "phase": "prepare"},
                )
            ).status == 200
            second_finalized = await cli.post(
                "/v1/sessions/session-1/clarify/respond",
                headers=_auth(),
                json={**second_body, "phase": "finalize"},
            )
            second_receipt = await second_finalized.json()
            assert second_receipt["state"] == "active"
            await _claim_and_ack(cli, "delivery-b", second_receipt)

            first_body = {
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/approval/respond",
                    headers=_auth(),
                    json={**first_body, "phase": "prepare"},
                )
            ).status == 200
            first_finalized = await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "finalize"},
            )
            first_receipt = await first_finalized.json()
            assert first_receipt["state"] == "terminal"
            assert "fence_id" not in first_receipt
            assert first.event.is_set()
    finally:
        blocked_stream.release.set()
        clarify_thread.join(1)

    assert not clarify_thread.is_alive()
    assert clarify_result["value"] == "answer"


def test_concurrent_publication_rollbacks_recompute_watermark_without_clobber():
    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    first_generation = first.data["interaction_generation"]
    queue_key = adapter._interaction_queue_key("session-1")
    watermark_key = queue_key, "turn-1"
    first_failed_generation = first_generation + 1
    second_failed_generation = first_generation + 2
    assert adapter._begin_interaction_publication(
        queue_key, "turn-1", first_failed_generation
    )
    assert adapter._begin_interaction_publication(
        queue_key, "turn-1", second_failed_generation
    )
    assert first_generation < first_failed_generation < second_failed_generation
    assert (
        adapter._published_interaction_generations[watermark_key]
        == second_failed_generation
    )

    adapter._rollback_interaction_publication(
        queue_key, "turn-1", first_failed_generation
    )
    assert (
        adapter._published_interaction_generations[watermark_key]
        == second_failed_generation
    )

    adapter._rollback_interaction_publication(
        queue_key, "turn-1", second_failed_generation
    )
    assert (
        adapter._published_interaction_generations[watermark_key]
        == first_generation
    )


@pytest.mark.asyncio
async def test_failed_provisional_publication_restores_older_active_receipt():
    class BlockingFailQueue:
        def __init__(self):
            self.visible = threading.Event()
            self.release = threading.Event()

        def put(self, _item):
            self.visible.set()
            assert self.release.wait(2)
            raise RuntimeError("push failed")

    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    blocked_stream = BlockingFailQueue()
    clarify_result = {}

    def ask_second():
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            clarify_result["value"] = adapter._make_clarify_cb(
                blocked_stream, "session-1"
            )("second", None)
        finally:
            clear_turn_vars(tokens)

    async with TestClient(TestServer(_app(adapter))) as cli:
        body = {
            "choice": "once",
            "delivery_id": "delivery-a",
            "interaction_id": "interaction-a",
        }
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**body, "phase": "prepare"},
            )
        ).status == 200

        clarify_thread = threading.Thread(target=ask_second)
        clarify_thread.start()
        assert blocked_stream.visible.wait(1)
        release_timer = threading.Timer(0.1, blocked_stream.release.set)
        release_timer.start()
        try:
            finalized = await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**body, "phase": "finalize"},
            )
        finally:
            blocked_stream.release.set()
            release_timer.cancel()
            clarify_thread.join(1)

        assert not clarify_thread.is_alive()
        assert clarify_result["value"].startswith("[clarify:")
        assert "state=cancelled reason=delivery_failed" in clarify_result["value"]
        finalized_body = await finalized.json()
        assert finalized_body["state"] == "active"
        assert finalized_body["fence_id"]
        assert not first.event.is_set()
        with adapter._delivery_lock:
            receipt = adapter._interaction_deliveries[
                adapter._delivery_key("session-1", "delivery-a")
            ]
            assert not receipt.get("_superseded")

        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert first.event.is_set()


@pytest.mark.asyncio
async def test_published_turn_capacity_evicts_only_terminal_finalized_receipts(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_PUBLISHED_TURN_CAP", 1)
    adapter = _adapter()
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    _register_live_turn(adapter, "turn-1")
    queue_key = adapter._interaction_queue_key("session-1")
    first_watermark_key = queue_key, "turn-1"

    async with TestClient(TestServer(_app(adapter))) as cli:
        body = {
            "choice": "once",
            "delivery_id": "delivery-a",
            "interaction_id": "interaction-a",
        }
        assert (
            await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**body, "phase": "prepare"},
            )
        ).status == 200
        finalized = await cli.post(
            "/v1/sessions/session-1/approval/respond",
            headers=_auth(),
            json={**body, "phase": "finalize"},
        )
        finalized_body = await finalized.json()
        assert finalized_body["state"] == "active"
        await _claim_and_ack(cli, "delivery-a", finalized_body)
        assert first.event.is_set()

        # A live finalized receipt is recovery evidence and must not be evicted.
        other_key = adapter._interaction_queue_key("session-2")
        assert not adapter._begin_interaction_publication(other_key, "turn-2", 999)
        assert first_watermark_key in adapter._published_interaction_generations
        assert adapter._delivery_key("session-1", "delivery-a") in (
            adapter._interaction_deliveries
        )

        # Once the first turn is terminal, the next real prompt may reclaim
        # both its finalized receipt and its corresponding watermark slot.
        adapter._active_session_turn_ids[
            adapter._active_turn_key("session-1")
        ] = "turn-2"
        terminal = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert (await terminal.json())["state"] == "terminal"

        second = _notify_approval(adapter, "interaction-b", "turn-2")
        assert not second.event.is_set()
        assert first_watermark_key not in adapter._published_interaction_generations
        assert (queue_key, "turn-2") in adapter._published_interaction_generations
        evicted = await cli.get(
            "/v1/sessions/session-1/interaction-deliveries/delivery-a",
            headers=_auth(),
        )
        assert evicted.status == 404


@pytest.mark.asyncio
async def test_published_turn_capacity_keeps_live_source_watermark(monkeypatch):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_PUBLISHED_TURN_CAP", 1)
    adapter = _adapter()
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    clarify_stream = queue.Queue()
    clarify_result = {}

    def ask_second():
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            clarify_result["value"] = adapter._make_clarify_cb(
                clarify_stream, "session-1"
            )("second", None)
        finally:
            clear_turn_vars(tokens)

    clarify_thread = threading.Thread(target=ask_second)
    clarify_thread.start()
    second_payload = clarify_stream.get(timeout=1)[1]
    queue_key = adapter._interaction_queue_key("session-1")
    watermark_key = queue_key, "turn-1"
    second_generation = second_payload["interaction_generation"]
    other_key = adapter._interaction_queue_key("session-2")

    try:
        reserved = adapter._begin_interaction_publication(
            other_key, "turn-2", second_generation + 1
        )
        if reserved:
            adapter._rollback_interaction_publication(
                other_key, "turn-2", second_generation + 1
            )
        assert not reserved
        assert (
            adapter._published_interaction_generations[watermark_key]
            == second_generation
        )

        async with TestClient(TestServer(_app(adapter))) as cli:
            second_body = {
                "response": "answer",
                "delivery_id": "delivery-b",
                "interaction_id": second_payload["interaction_id"],
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/clarify/respond",
                    headers=_auth(),
                    json={**second_body, "phase": "prepare"},
                )
            ).status == 200
            second_finalized = await cli.post(
                "/v1/sessions/session-1/clarify/respond",
                headers=_auth(),
                json={**second_body, "phase": "finalize"},
            )
            second_receipt = await second_finalized.json()
            assert second_receipt["state"] == "active"

            first_body = {
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/approval/respond",
                    headers=_auth(),
                    json={**first_body, "phase": "prepare"},
                )
            ).status == 200
            first_finalized = await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "finalize"},
            )
            first_receipt = await first_finalized.json()
            assert first_receipt["state"] == "terminal"
            assert "fence_id" not in first_receipt
            assert first.event.is_set()

            await _claim_and_ack(cli, "delivery-b", second_receipt)
        clarify_thread.join(1)
        assert not clarify_thread.is_alive()
        assert clarify_result["value"] == "answer"
    finally:
        if clarify_thread.is_alive():
            with adapter._clarify_state_lock:
                entries = adapter._clarify_queues.pop(queue_key, [])
            for entry in entries:
                entry.response = ""
                entry.event.set()
            clarify_thread.join(1)
        approval.cancel_gateway_approvals(queue_key)


def test_published_turn_reservation_prunes_expired_receipt(monkeypatch):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_PUBLISHED_TURN_CAP", 1)
    adapter = _adapter()
    old_queue_key = adapter._interaction_queue_key("session-1")
    old_watermark_key = old_queue_key, "turn-1"
    adapter._published_interaction_generations[old_watermark_key] = 1
    adapter._committed_interaction_generations[old_watermark_key] = 1
    adapter._interaction_deliveries[(old_queue_key, "delivery-a")] = {
        "scope_key": old_queue_key,
        "turn_id": "turn-1",
        "delivery_id": "delivery-a",
        "finalized": True,
        "expires_at": 0,
    }

    new_queue_key = adapter._interaction_queue_key("session-2")
    assert adapter._begin_interaction_publication(new_queue_key, "turn-2", 2)
    assert old_watermark_key not in adapter._published_interaction_generations
    assert (old_queue_key, "delivery-a") not in adapter._interaction_deliveries
    assert adapter._published_interaction_generations[
        (new_queue_key, "turn-2")
    ] == 2


@pytest.mark.asyncio
async def test_new_prompt_transfers_deferred_goal_owner_and_ack_clears_sidecar(
    tmp_path, monkeypatch
):
    from pathlib import Path

    import hermes_state
    from gateway.platforms.zet_agent_goals import ZetGoalDriver
    from hermes_cli import goals

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", home / "state.db")
    goals._DB_CACHE.clear()
    adapter = ZetAgentAdapter(
        PlatformConfig(enabled=True, extra={"key": "test-key"})
    )
    driver = ZetGoalDriver(adapter)
    monkeypatch.setattr(driver, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        driver, "report_in_thread", lambda *args, **kwargs: None
    )
    setattr(adapter, "_goals", lambda: driver)
    driver._apply_action_sync(
        "session-1",
        "create",
        {
            "goal_id": "goal-1",
            "text": "finish safely",
            "max_rounds": 3,
            "app_session_id": "session-1",
        },
    )
    _register_live_turn(adapter, "turn-1")
    first = _notify_approval(adapter, "interaction-a", "turn-1")
    assert driver._interaction_flag_set("session-1")

    try:
        async with TestClient(TestServer(_app(adapter))) as cli:
            first_body = {
                "choice": "once",
                "delivery_id": "delivery-a",
                "interaction_id": "interaction-a",
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/approval/respond",
                    headers=_auth(),
                    json={**first_body, "phase": "prepare"},
                )
            ).status == 200
            first_finalized = await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**first_body, "phase": "finalize"},
            )
            assert (await first_finalized.json())["state"] == "active"
            with adapter._delivery_lock:
                first_receipt = adapter._interaction_deliveries[
                    (adapter._interaction_queue_key("session-1"), "delivery-a")
                ]
                first_receipt["_fence_expires_at"] = time.monotonic() - 1
                adapter._expire_recovery_fences_locked()
                assert first_receipt["_goal_resolve_deferred"] is True
            assert first.event.is_set()

            second = _notify_approval(adapter, "interaction-b", "turn-1")
            with adapter._delivery_lock:
                assert "_goal_resolve_deferred" not in first_receipt

            second_body = {
                "choice": "deny",
                "delivery_id": "delivery-b",
                "interaction_id": "interaction-b",
            }
            assert (
                await cli.post(
                    "/v1/sessions/session-1/approval/respond",
                    headers=_auth(),
                    json={**second_body, "phase": "prepare"},
                )
            ).status == 200
            second_finalized = await cli.post(
                "/v1/sessions/session-1/approval/respond",
                headers=_auth(),
                json={**second_body, "phase": "finalize"},
            )
            second_receipt = await second_finalized.json()
            await _claim_and_ack(cli, "delivery-b", second_receipt)
            assert second.event.is_set()
            assert not driver._interaction_flag_set("session-1")
    finally:
        goals._DB_CACHE.clear()


def test_approval_timeout_notifies_adapter_to_drop_unique_session_mirrors(
    monkeypatch,
):
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"timeout": 0},
    )
    adapter = _adapter()
    for index in range(12):
        session_id = f"session-{index}"
        queue_key = adapter._interaction_queue_key(session_id)
        stream = queue.Queue()
        tokens = set_turn_vars(turn_id=f"turn-{index}")
        try:
            result = approval._await_gateway_decision(
                queue_key,
                adapter._make_approval_cb(stream, session_id, queue_key),
                {"command": f"command-{index}", "description": "test"},
            )
        finally:
            clear_turn_vars(tokens)
        assert result["resolved"] is False

        # Timeout emits the terminal frame before the reconnect mirror is
        # deleted, so clients cannot retain an actionable approval card.
        frames = []
        while not stream.empty():
            frames.append(stream.get_nowait()[1])
        assert frames[-1]["state"] == "expired"
        assert frames[-1]["state_reason"] == "timeout"

    with adapter._pending_lock:
        assert adapter._pending_approval == {}


def test_interaction_interrupt_releases_source_and_mirror_accounting(monkeypatch):
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 3},
    )
    adapter = _adapter()
    queue_key = adapter._interaction_queue_key("session-1")
    result = {}

    def run():
        tokens = set_turn_vars(turn_id="turn-1")
        try:
            result["value"] = approval._await_gateway_decision(
                queue_key,
                adapter._make_approval_cb(
                    queue.Queue(), "session-1", queue_key
                ),
                {"command": "interrupt me", "description": "test"},
            )
        finally:
            clear_turn_vars(tokens)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with adapter._pending_lock:
            if adapter._pending_approval.get(queue_key):
                break
        time.sleep(0.01)

    adapter._interrupt_pending_interactions("session-1")
    thread.join(1)
    assert not thread.is_alive()
    assert result["value"]["choice"] == "deny"
    with adapter._pending_lock:
        assert adapter._pending_approval == {}
        assert adapter._pending_mirror_meta == {}
        assert adapter._pending_mirror_bytes == 0


def test_interrupt_without_turn_id_does_not_scan_other_session_approvals():
    """A missing active turn must still be scoped to the requested session."""
    adapter = _adapter()
    session_one = adapter._interaction_queue_key("session-1")
    session_two = adapter._interaction_queue_key("session-2")
    stream_one = queue.Queue()
    stream_two = queue.Queue()
    adapter._approval_stream_queues = {
        session_one: stream_one,
        session_two: stream_two,
    }
    adapter._pending_approval = {
        session_one: [{"interaction_id": "approval-1", "turn_id": ""}],
        session_two: [{"interaction_id": "approval-2", "turn_id": ""}],
    }

    adapter._interrupt_pending_interactions("session-1", session_one)

    terminal = stream_one.get_nowait()[1]
    assert terminal["interaction_id"] == "approval-1"
    assert terminal["state"] == "cancelled"
    assert stream_two.empty()
    assert session_two in adapter._pending_approval


def test_pending_mirror_ttl_prunes_stale_source_drop_fallback(monkeypatch):
    monkeypatch.setattr(
        zet_agent_module, "INTERACTION_PENDING_MIRROR_TTL", 0
    )
    adapter = _adapter()
    queue_key = adapter._interaction_queue_key("session-1")
    data = {"command": "stale", "description": "test"}
    approval.enqueue_gateway_approval(queue_key, data)
    tokens = set_turn_vars(turn_id="turn-1")
    try:
        adapter._make_approval_cb(
            queue.Queue(), "session-1", queue_key
        )(data)
    finally:
        clear_turn_vars(tokens)
    with adapter._pending_lock:
        assert adapter._pending_mirror_meta

    # Simulate a source owner disappearing without its normal drop callback.
    approval._gateway_queues.pop(queue_key, None)
    adapter._prune_pending_mirrors()

    with adapter._pending_lock:
        assert adapter._pending_approval == {}
        assert adapter._pending_mirror_meta == {}
        assert adapter._pending_mirror_bytes == 0


@pytest.mark.parametrize(
    ("mirror_cap", "byte_budget"),
    ((2, 100_000), (10, 700)),
)
def test_pending_mirror_global_entry_and_byte_budgets_reject_new_live_prompt(
    monkeypatch, mirror_cap, byte_budget
):
    monkeypatch.setattr(
        approval,
        "_get_approval_config",
        lambda: {"gateway_timeout": 3},
    )
    monkeypatch.setattr(
        zet_agent_module,
        "INTERACTION_PENDING_MIRROR_CAP",
        mirror_cap,
        raising=False,
    )
    monkeypatch.setattr(
        zet_agent_module,
        "INTERACTION_PENDING_MIRROR_BYTE_BUDGET",
        byte_budget,
        raising=False,
    )
    adapter = _adapter()
    threads = []
    results = {}

    def run(index):
        session_id = f"bounded-{index}"
        queue_key = adapter._interaction_queue_key(session_id)
        tokens = set_turn_vars(turn_id=f"turn-{index}")
        try:
            results[index] = approval._await_gateway_decision(
                queue_key,
                adapter._make_approval_cb(queue.Queue(), session_id, queue_key),
                {
                    "command": "x" * 400,
                    "description": "y" * 100,
                },
            )
        finally:
            clear_turn_vars(tokens)

    for index in range(3):
        thread = threading.Thread(target=run, args=(index,))
        thread.start()
        threads.append(thread)

    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        with adapter._pending_lock:
            count = sum(len(items) for items in adapter._pending_approval.values())
        if count >= 1 and any(not thread.is_alive() for thread in threads):
            break
        time.sleep(0.01)

    with adapter._pending_lock:
        count = sum(len(items) for items in adapter._pending_approval.values())
        mirror_bytes = getattr(adapter, "_pending_mirror_bytes", 10**9)
    assert count < 3
    assert count <= mirror_cap
    assert mirror_bytes <= byte_budget
    for index in range(3):
        queue_key = adapter._interaction_queue_key(f"bounded-{index}")
        if approval.has_blocking_approval(queue_key):
            with adapter._pending_lock:
                assert adapter._pending_approval.get(queue_key)

    for index in range(3):
        approval.resolve_gateway_approval(
            adapter._interaction_queue_key(f"bounded-{index}"), "deny"
        )
    for thread in threads:
        thread.join(1)


def test_durable_session_memory_rejects_overflow_without_global_downgrade_guard(
    monkeypatch,
):
    monkeypatch.setattr(zet_agent_module, "INTERACTION_DURABLE_SESSION_CAP", 1)
    adapter = _adapter()
    first_key = adapter._interaction_queue_key("first")
    second_key = adapter._interaction_queue_key("second")

    with adapter._delivery_lock:
        assert adapter._mark_durable_session_locked(first_key) is True
        assert adapter._is_durable_session_locked(first_key)
        assert not adapter._is_durable_session_locked(second_key)
        assert adapter._mark_durable_session_locked(second_key) is False
        assert adapter._is_durable_session_locked(first_key)
        assert not adapter._is_durable_session_locked(second_key)
        assert len(adapter._durable_session_digests) == 1
