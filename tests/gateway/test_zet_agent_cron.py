from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.zet_agent import ZetAgentAdapter


@pytest.mark.asyncio
async def test_send_requires_explicit_cron_delivery_metadata():
    adapter = ZetAgentAdapter(PlatformConfig(extra={"key": "test-key"}))

    normal = await adapter.send("session-1", "hello")
    assert isinstance(normal, SendResult)
    assert normal.success is False
    assert "no proactive send path" in normal.error

    cron = await adapter.send(
        "session-1",
        "hello",
        metadata={"zet_agent_cron_delivery": True},
    )
    assert cron.success is True
    assert cron.message_id == "zet_agent:session-1"


def test_install_normalizes_legacy_zettlab_origin():
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zet_agent_cron

    zet_agent_cron.install()

    job = {
        "origin": {
            "platform": "zettlab",
            "chat_id": "session-123",
            "chat_name": "App Chat",
        }
    }
    result = scheduler._resolve_origin(job)
    assert result == {
        "platform": "zet_agent",
        "chat_id": "session-123",
        "chat_name": "App Chat",
    }
    assert job["origin"]["platform"] == "zettlab"


def test_install_bypasses_scheduler_delivery_for_zet_agent(monkeypatch):
    import cron.scheduler as scheduler
    import gateway.platforms.zet_agent_cron as zet_agent_cron

    zet_agent_cron.install()

    adapter = MagicMock()
    loop = MagicMock()
    loop.is_running.return_value = True
    job = {
        "id": "job-1",
        "deliver": "origin",
        "origin": {"platform": "zet_agent", "chat_id": "session-123"},
    }

    result = scheduler._deliver_result(
        job,
        "Cron output",
        adapters={},
        loop=loop,
    )

    assert result is None
    adapter.send.assert_not_called()
