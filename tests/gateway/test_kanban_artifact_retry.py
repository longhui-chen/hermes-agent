"""kanban 交付物投递：瞬时故障要重试，⛔ 但不许把 tick 拖死、⛔ 不许重复投递。

现场：任务完成后交付物上传只试**一次**。平台一次限流（429）、一次网络抖动，
文件就永久送不到 —— 用户只拿到一句「未能送达」，而设备上的文件他多半再也
不会去找。⭐ 「completed」事件只发一次，⛔ 没有第二次机会。

⚠️ 但这条路径跑在 notifier tick 里、**串行**挡着其他订阅（``:560`` 直接
await），所以重试必须**有界**：次数、单次退避、以及整批共享的**总预算**。
⭐ 「无上限的重试」在小设备上是整机级问题的同一形状 —— 这里是它的小型版。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.kanban_watchers import GatewayKanbanWatchersMixin


def _res(success=True, error=None, retryable=False, retry_after=None):
    return SimpleNamespace(
        success=success, error=error, retryable=retryable, retry_after=retry_after)


def _adapter():
    return SimpleNamespace(
        extract_local_files=lambda text: ([], None),
        send_image_file=AsyncMock(),
        send_multiple_images=AsyncMock(),
        send_video=AsyncMock(),
        send_document=AsyncMock(),
    )


@pytest.fixture
def no_sleep(monkeypatch):
    """记录退避时长但不真睡 —— ⛔ 单元测试里不许真等 3 秒。

    ⭐ 返回记录列表，让「睡了几次、每次多久」变成**可断言的**，
    ⛔ 不是「跑完没报错就算过」。
    """
    slept: list[float] = []
    real = asyncio.sleep

    async def _fake(d, *a, **k):
        slept.append(d)
        return await real(0)

    monkeypatch.setattr("gateway.kanban_watchers.asyncio.sleep", _fake)
    return slept


async def _deliver(adapter, path):
    mixin = GatewayKanbanWatchersMixin()
    return await mixin._deliver_kanban_artifacts(
        adapter=adapter, chat_id="oc_1", metadata=None,
        event_payload={"artifacts": [path]}, task=SimpleNamespace(id="t1", result=None),
    )


def _pdf(tmp_path, name="doc.pdf"):
    f = tmp_path / name
    f.write_bytes(b"x")
    return str(f)


# ───────────────── 缺陷本体：瞬时故障要重试 ─────────────────


@pytest.mark.asyncio
async def test_transient_rejection_is_retried_and_succeeds(tmp_path, no_sleep):
    """🔴 一次限流不该让文件永久丢失。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.side_effect = [
        _res(success=False, error="rate limited", retryable=True),
        _res(success=True),
    ]

    failed = await _deliver(adapter, path)

    assert failed == [], f"重试后已送达，却仍报失败:{failed}"
    assert adapter.send_document.await_count == 2
    assert len(no_sleep) == 1, f"没有退避就直接重试:{no_sleep}"


@pytest.mark.asyncio
async def test_transient_exception_is_retried(tmp_path, no_sleep):
    """异常路径同形 —— ⛔ 只修返回值那一侧等于给异常侧发免检。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.side_effect = [
        ConnectionResetError("connection reset by peer"),
        _res(success=True),
    ]

    assert await _deliver(adapter, path) == []
    assert adapter.send_document.await_count == 2


@pytest.mark.asyncio
async def test_images_get_the_same_retry(tmp_path, no_sleep):
    """⭐ 兄弟调用点：图片 / 视频 / 文档三条上传路都要有重试，
    ⛔ 不是只给文档加。"""
    path = _pdf(tmp_path, "shot.png")
    adapter = _adapter()
    adapter.send_image_file.side_effect = [
        _res(success=False, error="temporarily unavailable", retryable=True),
        _res(success=True),
    ]
    assert await _deliver(adapter, path) == []
    assert adapter.send_image_file.await_count == 2


@pytest.mark.asyncio
async def test_video_gets_the_same_retry(tmp_path, no_sleep):
    path = _pdf(tmp_path, "clip.mp4")
    adapter = _adapter()
    adapter.send_video.side_effect = [
        _res(success=False, error="temporarily unavailable", retryable=True),
        _res(success=True),
    ]
    assert await _deliver(adapter, path) == []
    assert adapter.send_video.await_count == 2


@pytest.mark.asyncio
async def test_server_retry_after_wins_over_our_backoff(tmp_path, no_sleep):
    """服务端说等 5 秒就等 5 秒 —— ⛔ 不许用我们自己的 1 秒抢在它前面。

    ⭐ 照抄 ``_send_with_retry``:``retry_after`` 权威、且**只认一次**。
    """
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.side_effect = [
        _res(success=False, error="flood wait", retryable=True, retry_after=5.0),
        _res(success=True),
    ]

    await _deliver(adapter, path)
    assert no_sleep and 5.0 <= no_sleep[0] < 6.0, (
        f"没有采纳服务端要求的等待时长:{no_sleep}")


# ───────────── ⛔ 不许重试的两侧（改宽了比原缺陷更坏） ─────────────


@pytest.mark.asyncio
async def test_timeout_is_never_retried(tmp_path, no_sleep):
    """🔴 超时**不许**重传：请求可能已经送达，重传 = 用户收到两份文件。

    ⭐ 这条判据是从 ``_send_with_retry`` 逐字抄来的，⛔ 不是我的发明。
    """
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.return_value = _res(
        success=False, error="Read timed out", retryable=True)

    failed = await _deliver(adapter, path)

    assert [x.path for x in failed] == [path]
    assert adapter.send_document.await_count == 1, (
        f"超时被重传了 {adapter.send_document.await_count} 次 —— 用户会收到多份")
    assert no_sleep == []


@pytest.mark.asyncio
async def test_permanent_rejection_is_not_retried(tmp_path, no_sleep):
    """非瞬时失败（文件过大、鉴权失效）重试多少次都一样 ⇒ ⛔ 别浪费 tick。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.return_value = _res(
        success=False, error="file too large", retryable=False)

    assert [x.path for x in await _deliver(adapter, path)] == [path]
    assert adapter.send_document.await_count == 1
    assert no_sleep == []


@pytest.mark.asyncio
async def test_non_transient_exception_is_not_retried(tmp_path, no_sleep):
    """编程错误 / 明确的业务错误不许被当成网络抖动反复试。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.side_effect = ValueError("unsupported mime type")

    assert [x.path for x in await _deliver(adapter, path)] == [path]
    assert adapter.send_document.await_count == 1


# ───────────────── 有界性：⛔ 不许拖死 notifier tick ─────────────────


@pytest.mark.asyncio
async def test_retries_are_capped(tmp_path, no_sleep):
    """一直瞬时失败 ⇒ 试满上限就放弃，⛔ 不是无限重试。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.return_value = _res(
        success=False, error="connection reset", retryable=True)

    assert [x.path for x in await _deliver(adapter, path)] == [path]
    assert adapter.send_document.await_count == (
        GatewayKanbanWatchersMixin._ARTIFACT_MAX_RETRIES + 1)


@pytest.mark.asyncio
async def test_total_backoff_budget_is_shared_across_files(tmp_path, no_sleep):
    """🔴 总预算是**整批共享**的 —— ⛔ 不是每个文件各拿一份。

    ⭐ 只有 per-file 上限 = 20 个文件各退避 3 秒 ⇒ 一个 tick 被拖住一分钟，
    期间**所有其他订阅的通知都发不出去**。
    这条钉的是「睡的总时长有上限」，⛔ 不是「每个文件试了几次」。
    """
    paths = []
    for i in range(20):
        f = tmp_path / f"f{i}.pdf"
        f.write_bytes(b"x")
        paths.append(str(f))

    adapter = _adapter()
    adapter.send_document.return_value = _res(
        success=False, error="connection reset", retryable=True)

    mixin = GatewayKanbanWatchersMixin()
    failed = await mixin._deliver_kanban_artifacts(
        adapter=adapter, chat_id="oc_1", metadata=None,
        event_payload={"artifacts": paths},
        task=SimpleNamespace(id="t1", result=None),
    )

    assert sorted(x.path for x in failed) == sorted(paths)
    budget = GatewayKanbanWatchersMixin._ARTIFACT_RETRY_BUDGET_S
    assert sum(no_sleep) <= budget, (
        f"退避总时长 {sum(no_sleep):.1f}s 超过预算 {budget}s —— tick 被拖住")


# ───────────────── ⛔ 必须保持不变的行为 ─────────────────


@pytest.mark.asyncio
async def test_first_try_success_never_sleeps(tmp_path, no_sleep):
    """⛔ 正常路径一次成功 ⇒ 零退避、零重复投递。

    ⭐ 「加了重试」最容易弄坏的就是这条:多睡一次 / 多发一次，
    每条正常任务都会变慢或让用户收到两份。
    """
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.return_value = _res(success=True)

    assert await _deliver(adapter, path) == []
    assert adapter.send_document.await_count == 1
    assert no_sleep == []


@pytest.mark.asyncio
async def test_adapter_returning_none_still_counts_as_delivered(tmp_path, no_sleep):
    """⛔ 旧契约不许被弄坏:返回 ``None`` = 无异常即送达，且⛔不重试。"""
    path = _pdf(tmp_path)
    adapter = _adapter()
    adapter.send_document.return_value = None

    assert await _deliver(adapter, path) == []
    assert adapter.send_document.await_count == 1
