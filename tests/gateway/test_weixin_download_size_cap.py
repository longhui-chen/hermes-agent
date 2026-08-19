"""微信媒体下载必须有上限 —— 裸 ``read()`` 在 1C2G 设备上会 OOM。

RH 复审第六轮 P1[存量]：``_download_media_bytes`` 直接 ``await response.read()``，
**整个响应先进内存**，之后才轮到落盘处的大小校验 ⇒ 单个超大入站附件就能撑爆
进程，大小门根本来不及生效。⭐ 「先读完再校验」= 没有校验。

判据双门（照抄 ``base._read_httpx_body_with_limit``）：
① ``Content-Length`` 早拒 ② 累计字节复检（**头可以撒谎或缺失**）。
"""
from __future__ import annotations

import pytest

from gateway.platforms.base import (
    get_inbound_media_max_bytes,
    read_aiohttp_body_with_limit,
)


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = chunks

    async def iter_chunked(self, _n):
        for c in self._chunks:
            yield c


class _FakeResponse:
    def __init__(self, chunks, content_length=None):
        self.headers = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self.content = _FakeContent(chunks)


@pytest.mark.asyncio
async def test_oversized_content_length_is_rejected_before_reading():
    """① 头声明超限 ⇒ 立刻拒，⛔ 一个字节都不读。"""
    cap = get_inbound_media_max_bytes()
    read = {"n": 0}

    class _Counting(_FakeContent):
        async def iter_chunked(self, n):
            read["n"] += 1
            async for c in super().iter_chunked(n):
                yield c

    r = _FakeResponse([b"x"], content_length=cap + 1)
    r.content = _Counting([b"x"])
    with pytest.raises(Exception):
        await read_aiohttp_body_with_limit(r, media_type="test")
    assert read["n"] == 0, "头已经说超限了还去读了 body"


@pytest.mark.asyncio
async def test_lying_content_length_is_caught_while_streaming():
    """🔴 ② 头**撒谎**（说很小、实际很大）⇒ 累计复检必须抓住。

    ⭐ 只有第一道门 = 没有门:攻击者/坏代理只要不报或少报
    Content-Length 就能把无界 body 送进来。
    """
    cap = get_inbound_media_max_bytes()
    chunk = b"y" * 65536
    n = cap // 65536 + 2
    r = _FakeResponse([chunk] * n, content_length=10)
    with pytest.raises(Exception):
        await read_aiohttp_body_with_limit(r, media_type="test")


@pytest.mark.asyncio
async def test_absent_content_length_is_still_capped():
    """② 的另一半：头**缺失**时同样受累计上限约束。"""
    cap = get_inbound_media_max_bytes()
    chunk = b"z" * 65536
    n = cap // 65536 + 2
    r = _FakeResponse([chunk] * n, content_length=None)
    with pytest.raises(Exception):
        await read_aiohttp_body_with_limit(r, media_type="test")


@pytest.mark.asyncio
async def test_memory_is_not_blown_before_the_cap_trips():
    """⭐ 判据落在「**累计到超限就停**」，⛔ 不是「最后校验一次」。

    钉住:超限时已读入的字节数不应远超上限（这里放宽到 2 倍，
    因为最后一块可能整块读入）。
    """
    cap = get_inbound_media_max_bytes()
    chunk = b"w" * 65536
    consumed = {"n": 0}

    class _Counting(_FakeContent):
        async def iter_chunked(self, n):
            for c in self._chunks:
                consumed["n"] += len(c)
                yield c

    r = _FakeResponse([], content_length=None)
    r.content = _Counting([chunk] * (cap // 65536 * 4))
    with pytest.raises(Exception):
        await read_aiohttp_body_with_limit(r, media_type="test")
    assert consumed["n"] <= cap * 2, (
        f"超限后仍继续读入 {consumed['n']} 字节 —— 上限没有真正止住读取")


# ───────── ⛔ 必须保持不变 ─────────


@pytest.mark.asyncio
async def test_normal_sized_body_is_returned_intact():
    """⛔ 正常大小的响应逐字节原样返回。"""
    body = b"".join([b"abc", b"def", b"ghi"])
    r = _FakeResponse([b"abc", b"def", b"ghi"], content_length=len(body))
    assert await read_aiohttp_body_with_limit(r, media_type="test") == body


@pytest.mark.asyncio
async def test_invalid_content_length_header_does_not_break_download():
    """⛔ 头是垃圾时不许直接失败 —— 忽略它，靠累计门兜住。"""
    r = _FakeResponse([b"ok"], content_length=None)
    r.headers["content-length"] = "not-a-number"
    assert await read_aiohttp_body_with_limit(r, media_type="test") == b"ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", [11, None])
async def test_callers_can_apply_a_smaller_platform_cap(content_length):
    """LINE 图片 10 MiB 等平台上限必须能收紧全局默认值。"""
    r = _FakeResponse([b"x" * 11], content_length=content_length)
    with pytest.raises(ValueError, match="11 bytes > 10 bytes"):
        await read_aiohttp_body_with_limit(
            r, media_type="platform image", max_bytes=10,
        )


def test_weixin_has_no_bare_read_on_media_paths():
    """闭集：微信的两条媒体路径⛔ 不许再出现裸 ``response.read()``。

    ⚠️ 上传响应那处（只为排空、且响应来自服务端确认）**不在**本判据内 ——
    ⭐ 修复作用域要刚好等于缺陷作用域。
    """
    import inspect

    from gateway.platforms import weixin

    src = inspect.getsource(weixin)
    # 两条媒体路径都必须经过带上限的读取
    assert src.count("read_aiohttp_body_with_limit(") >= 2, (
        "微信媒体下载/拉取没有全部走带上限的读取")
