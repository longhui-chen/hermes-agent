"""微信入站媒体：成因要分得开，日志⛔不许泄漏签名 URL（RH 复审第五轮 P2-2 / P2-7）。

**P2-2** —— 我上一轮给 weixin 用 `List[str]` 而不是结构化失败，理由写的是
「weixin 这条路只有一种成因」。**那个前提是错的**：四个下载 helper 把下载、
解密**和落盘**包在同一个 `try` 里，磁盘满也会走到「请重新发送」——
而重发多少次都白搭。⭐ 又一次「补错的建议比不补更坏」。

**P2-7** —— 四个出口都 `logger.warning("... failed: %s", exc)` 直接打异常对象。
`aiohttp.ClientResponseError.__str__` **包含完整 URL**，而微信媒体 url 带鉴权
参数 ⇒ 403 时把 token 写进日志。⚠️ 我在 wecom 侧修过同一个形状（`netloc`
含 userinfo），却没扫到这里 —— **兄弟调用点没跟上**。
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.platforms import weixin
from gateway.platforms.base import MediaFailure
from tests.gateway.test_weixin import _make_adapter

_SIGNED = "https://cdn.example/media?token=TOP_SECRET_TOKEN&x=1"


def _adapter():
    a = _make_adapter()
    a._poll_session = Mock()
    a._token = None
    a._cdn_base_url = "https://cdn.example"
    a._text_batch_delay_seconds = 0
    a._text_batch_split_delay_seconds = 0
    a.handle_message = AsyncMock()
    a.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="m1"))
    return a


def _img_item():
    return {"type": weixin.ITEM_IMAGE,
            "image_item": {"media": {"full_url": _SIGNED}}}


async def _collect(a):
    paths, types, failures = [], [], []
    await a._collect_media(_img_item(), paths, types, failures)
    return paths, failures


# ───────── P2-2：成因必须分得开（四个 helper 全集） ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type,kind",
    [(weixin.ITEM_IMAGE, "image"), (weixin.ITEM_VIDEO, "video"),
     (weixin.ITEM_FILE, "file"), (weixin.ITEM_VOICE, "voice")],
)
async def test_download_failure_is_classified(monkeypatch, item_type, kind):
    """下载/解密失败 ⇒ ``download_failed``。四个 helper 全集，⛔ 不抽查。"""
    async def _boom(*_a, **_k):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _boom)
    a = _adapter()
    paths, types, failures = [], [], []
    item = {"type": item_type}
    slot = {weixin.ITEM_IMAGE: "image_item", weixin.ITEM_VIDEO: "video_item",
            weixin.ITEM_FILE: "file_item", weixin.ITEM_VOICE: "voice_item"}[item_type]
    item[slot] = {"media": {"full_url": _SIGNED}}

    await a._collect_media(item, paths, types, failures)
    assert failures == [MediaFailure(kind, "download_failed")]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type,kind,writer",
    [(weixin.ITEM_IMAGE, "image", "cache_image_from_bytes"),
     (weixin.ITEM_VIDEO, "video", "cache_document_from_bytes"),
     (weixin.ITEM_FILE, "file", "cache_document_from_bytes"),
     (weixin.ITEM_VOICE, "voice", "cache_audio_from_bytes")],
)
async def test_disk_failure_is_not_reported_as_download_failure(
    monkeypatch, item_type, kind, writer
):
    """🔴 P2-2 本体：落盘失败 ⇒ ``cache_write_failed``，⛔ 不是 download_failed。

    ⭐ 两者给用户的建议**必须不同**：磁盘满时说「请重新发送」是确定无效的
    指引，用户会反复重发然后以为是自己的问题。
    """
    async def _ok(*_a, **_k):
        return b"\x89PNG\r\n\x1a\n"

    def _full(*_a, **_k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(weixin, writer, _full)

    a = _adapter()
    paths, types, failures = [], [], []
    slot = {weixin.ITEM_IMAGE: "image_item", weixin.ITEM_VIDEO: "video_item",
            weixin.ITEM_FILE: "file_item", weixin.ITEM_VOICE: "voice_item"}[item_type]
    await a._collect_media(
        {"type": item_type, slot: {"media": {"full_url": _SIGNED}}},
        paths, types, failures)

    assert failures == [MediaFailure(kind, "cache_write_failed")], (
        f"落盘失败被误分类成下载失败 ⇒ 用户会拿到无效建议:{failures}")


@pytest.mark.asyncio
async def test_disk_failure_advice_differs_from_download_advice(monkeypatch):
    """给用户的两句话必须真的不一样，⛔ 不是「分类了但文案一样」。"""
    a = _adapter()
    await a._reply_media_intake_failed(
        "u1", [MediaFailure("image", "cache_write_failed")])
    disk = a.send.await_args.args[1]

    a.send.reset_mock()
    await a._reply_media_intake_failed(
        "u1", [MediaFailure("image", "download_failed")])
    net = a.send.await_args.args[1]

    assert disk != net, f"两种成因给了同一句话:{disk}"
    assert "存储" in disk, f"磁盘满没说清怎么办:{disk}"
    assert "重新发送" in net, f"下载失败没给下一步:{net}"
    assert "重新发送" not in disk, (
        f"磁盘满还在劝用户重发 —— 重发多少次都白搭:{disk}")


@pytest.mark.asyncio
async def test_missing_media_reference_is_classified(monkeypatch):
    """声明了 image、却没有可用引用 ⇒ ``payload_malformed`` 兜底，⛔ 不静默。"""
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media({"type": weixin.ITEM_IMAGE}, paths, types, failures)
    assert failures and failures[0].kind == "image"


# ───────── P2-7：日志⛔不许泄漏签名 URL ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type", [weixin.ITEM_IMAGE, weixin.ITEM_VIDEO,
                  weixin.ITEM_FILE, weixin.ITEM_VOICE])
async def test_failure_log_never_leaks_the_signed_url(
    monkeypatch, caplog, item_type
):
    """🔴 四个出口全集：日志里⛔ 不许出现 token / 完整 url / 原始异常字符串。

    ⭐ 判据不是「我没写 url」，而是「**不管异常长什么样**，token 都不能出现」
    —— 泄漏是异常对象的 ``__str__`` 带进去的，不是我显式打印的。
    """
    import aiohttp
    from yarl import URL

    ri = aiohttp.RequestInfo(URL(_SIGNED), "GET", {}, URL(_SIGNED))

    async def _403(*_a, **_k):
        raise aiohttp.ClientResponseError(ri, (), status=403, message="Forbidden")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _403)
    a = _adapter()
    slot = {weixin.ITEM_IMAGE: "image_item", weixin.ITEM_VIDEO: "video_item",
            weixin.ITEM_FILE: "file_item", weixin.ITEM_VOICE: "voice_item"}[item_type]

    with caplog.at_level(logging.DEBUG):
        await a._collect_media(
            {"type": item_type, slot: {"media": {"full_url": _SIGNED}}},
            [], [], [])

    assert "TOP_SECRET_TOKEN" not in caplog.text, f"签名 token 进了日志:{caplog.text}"
    assert _SIGNED not in caplog.text, f"完整 url 进了日志:{caplog.text}"
    # ⭐ 阳性对照：日志确实写了点东西 —— ⛔ 否则「没泄漏」只是因为什么都没记。
    assert "cdn.example" in caplog.text, f"连 host 都没记，无法定位:{caplog.text}"
    assert "reason=" in caplog.text, f"没记成因:{caplog.text}"


@pytest.mark.asyncio
async def test_user_reply_never_leaks_the_url(monkeypatch):
    """给用户的话同样⛔不许含 url / reason 码。"""
    a = _adapter()
    await a._reply_media_intake_failed(
        "u1", [MediaFailure("image", "download_failed")])
    body = a.send.await_args.args[1]
    assert "TOP_SECRET_TOKEN" not in body and "cdn.example" not in body
    assert "download_failed" not in body, f"reason 码泄漏给用户:{body}"


# ───────── ⛔ 必须保持不变 ─────────


@pytest.mark.asyncio
async def test_success_records_no_failure(monkeypatch, tmp_path):
    """⛔ 成功路径不许记任何失败。"""
    async def _ok(*_a, **_k):
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(
        weixin, "cache_image_from_bytes", lambda d, ext=".jpg": str(tmp_path / "a.jpg"))
    a = _adapter()
    paths, failures = await _collect(a)
    assert paths == [str(tmp_path / "a.jpg")] and failures == []


@pytest.mark.asyncio
async def test_non_media_item_records_no_failure():
    """⛔ 文本 item 不是媒体 ⇒ 一条失败都不许记。"""
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media(
        {"type": 1, "text_item": {"text": "你好"}}, paths, types, failures)
    assert failures == []


# ───────── try 作用域：解析 ≠ 下载 ─────────


@pytest.mark.asyncio
async def test_malformed_aeskey_is_not_a_download_failure(monkeypatch):
    """🔴 畸形 aeskey 是**载荷问题**，⛔ 不是下载失败。

    ``bytes.fromhex`` 原先在下载的同一个 ``try`` 里 ⇒ 抛 ValueError 被记成
    ``download_failed``，用户拿到「请重新发送」。
    ⭐ 通则：``try`` 包了几件事 = 错误分类的分辨率上限。⛔ 文案层救不了
    作用域太宽的 try。
    """
    called = {"n": 0}

    async def _never(*_a, **_k):
        called["n"] += 1
        return b"x"

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _never)
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media(
        {"type": weixin.ITEM_IMAGE,
         "image_item": {"aeskey": "ZZ-not-hex", "media": {"full_url": _SIGNED}}},
        paths, types, failures)

    assert failures == [MediaFailure("image", "payload_malformed")], (
        f"畸形 aeskey 被记成下载失败:{failures}")
    assert called["n"] == 0, "载荷都解析不了还去发了网络请求"


@pytest.mark.asyncio
async def test_valid_aeskey_still_reaches_the_downloader(monkeypatch, tmp_path):
    """⛔ 必须保持不变：合法 aeskey 照常转换并传下去。"""
    seen = {}

    async def _ok(*_a, **kw):
        seen.update(kw)
        return b"\x89PNG\r\n\x1a\n"

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(
        weixin, "cache_image_from_bytes", lambda d, ext=".jpg": str(tmp_path / "a.jpg"))
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media(
        {"type": weixin.ITEM_IMAGE,
         "image_item": {"aeskey": "00112233", "media": {"full_url": _SIGNED}}},
        paths, types, failures)

    assert failures == [] and paths == [str(tmp_path / "a.jpg")]
    assert seen.get("aes_key_b64") == "ABEiMw==", f"aeskey 没正确传下去:{seen.get('aes_key_b64')!r}"


# ───────── RH 第六轮：记账函数自身的健壮性 / except 作用域 ─────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "item_type,slot",
    [(weixin.ITEM_IMAGE, "image_item"), (weixin.ITEM_VIDEO, "video_item"),
     (weixin.ITEM_FILE, "file_item"), (weixin.ITEM_VOICE, "voice_item")])
async def test_non_dict_media_does_not_crash_the_accounting(item_type, slot):
    """🔴 上游把 ``*_item`` 塞成字符串 ⇒ 记账函数原先二次抛 AttributeError。

    后果：异常冒到 ``_process_message_safe`` 只记日志，**用户一个字都收不到**。
    ⭐ 记账函数是失败路径上的最后一道 —— 它自己崩了就什么都不剩。
    """
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media({"type": item_type, slot: "a-string"},
                           paths, types, failures)
    assert failures and failures[0].kind
    assert paths == []


@pytest.mark.asyncio
async def test_format_error_is_not_reported_as_disk_failure(monkeypatch):
    """🔴 ``ValueError("not an image")`` ⇒ ``not_a_valid_media``，
    ⛔ 不是 ``cache_write_failed``（那会让用户去清存储，白费）。

    ⭐ ``except Exception`` 把环境故障、格式错误、编程错误压成一格 ——
    又一次「try/except 的作用域就是错误分类的分辨率上限」。
    """
    async def _ok(*_a, **_k):
        return b"not-an-image"

    def _bad(*_a, **_k):
        raise ValueError("not an image")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(weixin, "cache_image_from_bytes", _bad)
    a = _adapter()
    paths, types, failures = [], [], []
    await a._collect_media(
        {"type": weixin.ITEM_IMAGE, "image_item": {"media": {"full_url": _SIGNED}}},
        paths, types, failures)
    assert failures == [MediaFailure("image", "not_a_valid_media")], f"{failures}"


@pytest.mark.asyncio
async def test_programming_error_still_propagates(monkeypatch):
    """⛔ 编程错误不许被伪装成「用户的文件有问题」—— 必须继续上抛。"""
    async def _ok(*_a, **_k):
        return b"x"

    def _bug(*_a, **_k):
        raise TypeError("wrong arity — our bug, not the disk's")

    monkeypatch.setattr(weixin, "_download_and_decrypt_media", _ok)
    monkeypatch.setattr(weixin, "cache_image_from_bytes", _bug)
    a = _adapter()
    with pytest.raises(TypeError):
        await a._collect_media(
            {"type": weixin.ITEM_IMAGE,
             "image_item": {"media": {"full_url": _SIGNED}}}, [], [], [])
