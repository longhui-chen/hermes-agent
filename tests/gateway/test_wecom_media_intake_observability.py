"""企微入站媒体：失败必须显性、可区分、不泄漏；类型判据必须闭集。

现场（族 A）：用户在企微发图，Agent 收不到，而**日志里什么都没有** ——
`_cache_media` 六条失败分支里 4 条只打 `logger.debug`（默认不输出）、1 条
连日志都没有。既是缺陷本身，也是它一直排查不出来的原因。

⛔ 作用域刻意限定在「**入站媒体获取失败**」这一类。全仓还有 70+ 处
`except → logger.debug`（slack 31 / telegram 26 / feishu 15），那批**不在本
lane 的面上**，⛔ 只报不改。
"""
from __future__ import annotations

import logging

import pytest

from plugins.platforms.wecom.adapter import (
    WeComAdapter,
    _WECOM_DOWNLOADABLE_MSGTYPES,
    _WECOM_KNOWN_MSGTYPES,
    _WECOM_LOCAL_KNOWN_MSGTYPES,
    _WECOM_MIXED_ITEM_MSGTYPES,
    _WECOM_OFFICIAL_MSGTYPES,
    _WECOM_TRANSCRIPT_MSGTYPES,
)


def _adapter():
    # name 是只读 property,⛔ 不能直接赋值;绕过 __init__ 后用类属性覆盖。
    class _A(WeComAdapter):
        name = "wecom"

    return _A.__new__(_A)


# ───────────────────────── 失败必须显性且可区分 ─────────────────────────

@pytest.mark.asyncio
async def test_no_media_reference_is_no_longer_silent(caplog):
    """🔴 原先这条【一句日志都没有】—— 六条里最不可观测的一条。"""
    a = _adapter()
    with caplog.at_level(logging.WARNING):
        assert await a._cache_media("image", {}) is None
    assert "no_media_reference" in caplog.text, (
        f"无 url 也无 base64 时仍然静默:{caplog.text!r}")


@pytest.mark.asyncio
async def test_base64_decode_failure_is_distinguishable(caplog):
    a = _adapter()
    with caplog.at_level(logging.WARNING):
        assert await a._cache_media("image", {"base64": "!!!not-base64!!!"}) is None
    assert "base64_decode_failed" in caplog.text


@pytest.mark.asyncio
async def test_failure_reasons_are_not_collapsed_into_one(caplog):
    """⛔ 禁止笼统归并：不同根因必须给出不同 reason。"""
    a = _adapter()
    seen = set()
    with caplog.at_level(logging.WARNING):
        await a._cache_media("image", {})
        await a._cache_media("image", {"base64": "!!!"})
    for line in caplog.text.splitlines():
        if "reason=" in line:
            seen.add(line.split("reason=", 1)[1].split()[0])
    assert len(seen) >= 2, (
        f"两种不同根因被压成同一个 reason,用户无从区分:{seen}")


@pytest.mark.asyncio
async def test_credentials_never_reach_the_log(caplog):
    """⛔ 不许打印 url 全文 / aeskey / base64 —— 企微媒体 url 带鉴权参数。

    原实现是 `logger.debug("... from %s", url)`，会把整条带凭据的 url 写进日志。
    """
    a = _adapter()
    secret_url = "https://wework.qpic.cn/media?token=SUPER_SECRET_TOKEN&x=1"
    with caplog.at_level(logging.WARNING):
        await a._cache_media(
            "image",
            {"base64": "!!!bad!!!", "aeskey": "AESKEY_SHOULD_NOT_LEAK",
             "url": secret_url},
        )
    assert "SUPER_SECRET_TOKEN" not in caplog.text, "url 里的凭据泄漏进日志"
    assert secret_url not in caplog.text, "url 全文泄漏进日志"
    assert "AESKEY_SHOULD_NOT_LEAK" not in caplog.text, "aeskey 泄漏进日志"
    # 仍要保留足以定位的摘要
    assert "url_host=wework.qpic.cn" in caplog.text, "连 host 都没留,失去可定位性"


@pytest.mark.asyncio
async def test_basic_auth_userinfo_never_reaches_the_log(caplog):
    """🔴 ``netloc`` 含 userinfo —— 用它"脱敏"等于换个地方继续泄漏。

    我上一版为了不打印 url 全文，改成记 ``urlparse(url).netloc``；
    但 ``https://user:secret@host/x`` 的 netloc 就是 ``user:secret@host``。
    ⭐ 修一个泄漏点时用的工具本身也要过一遍同一条判据。
    """
    a = _adapter()
    with caplog.at_level(logging.WARNING):
        await a._cache_media(
            "image",
            {"base64": "!!!bad!!!",
             "url": "https://botuser:P4ssw0rd_LEAK@wework.qpic.cn/media?x=1"},
        )
    assert "P4ssw0rd_LEAK" not in caplog.text, "url 里的 basic-auth 密码泄漏进日志"
    assert "botuser" not in caplog.text, "url 里的 basic-auth 用户名泄漏进日志"
    assert "url_host=wework.qpic.cn" in caplog.text, (
        f"剥 userinfo 时把 host 也剥掉了,失去可定位性:{caplog.text!r}")


@pytest.mark.asyncio
async def test_non_default_port_is_kept_for_locatability(caplog):
    """剥 userinfo ⛔ 不许连端口一起丢 —— 端口是定位信息，不是凭据。"""
    a = _adapter()
    with caplog.at_level(logging.WARNING):
        await a._cache_media(
            "image", {"base64": "!!!bad!!!", "url": "https://cache.internal:8443/m"})
    assert "url_host=cache.internal:8443" in caplog.text, (
        f"端口丢了:{caplog.text!r}")


# ───────────────────────── 类型判据必须闭集 ─────────────────────────

@pytest.mark.asyncio
async def test_mixed_only_carries_image_per_protocol(monkeypatch, caplog):
    """混排子项按官方限定只有 ``text`` / ``image``。

    🔴 我上一版在这里收 file/voice/video,还构造了**协议里不存在的** payload
    去喂它 —— 测试自己造了个假世界,然后在假世界里验证通过。
    ⭐ 假 payload 造出来的绿,比没有测试更坏:它会阻止后来人发现真相。
    """
    a = _adapter()
    grabbed: list[str] = []

    async def _fake_cache(kind, ref, failures=None):
        grabbed.append(kind)
        return (f"/cache/{kind}", f"{kind}/x")

    monkeypatch.setattr(a, "_cache_media", _fake_cache)
    # ⭐ 子项里**必须**放一个协议之外的类型，否则这条门是空转的:
    #    只放 text+image 时，把判据逆改成「收全部下载类型」行为完全一样,
    #    门检测不到差异 —— 实测逆改 C 一开始就是 12 passed(⛔ 出生即空转)。
    body = {
        "msgtype": "mixed",
        "mixed": {"msg_item": [
            {"msgtype": "text", "text": {"content": "看这张"}},
            {"msgtype": "image", "image": {"url": "u1", "aeskey": "k1"}},
            # 顶层合法、但**混排子项里不存在**的类型:
            {"msgtype": "file", "file": {"url": "u2", "aeskey": "k2"}},
        ]},
    }
    with caplog.at_level(logging.WARNING):
        paths, types, _fails = await a._extract_media(body)
    assert grabbed == ["image"], f"混排里取到了协议之外的类型:{grabbed}"
    assert len(paths) == 1 and len(types) == 1
    assert "未知 msg_item 类型" in caplog.text, (
        f"协议外的子项被静默丢弃,没有任何痕迹:{caplog.text!r}")


@pytest.mark.asyncio
@pytest.mark.parametrize("mt", ["file", "video"])
async def test_top_level_downloadable_msgtypes_are_handled(monkeypatch, mt):
    """``video`` 整类原先被静默丢弃（顶层只有 file 一条）—— 这条修复保留。"""
    a = _adapter()
    grabbed: list[str] = []

    async def _fake_cache(kind, ref, failures=None):
        grabbed.append(kind)
        return (f"/cache/{kind}", f"{kind}/x")

    monkeypatch.setattr(a, "_cache_media", _fake_cache)
    paths, _t, _fails = await a._extract_media(
        {"msgtype": mt, mt: {"url": "https://wework.qpic.cn/x", "aeskey": "k"}})
    assert grabbed == [mt], f"{mt} 类型未被取到:{grabbed}"
    assert len(paths) == 1


@pytest.mark.asyncio
async def test_voice_never_enters_the_download_path(monkeypatch, caplog):
    """🔴 回归门：``voice`` **没有可下载资源**，⛔ 不许走下载分支。

    官方 ``VoiceContent`` 只有 ``content: string``（语音转成的文本）。
    我上一版把 voice 并进可下载集，于是**每条合法语音**都走进 ``_cache_media``、
    拿不到 url/base64，最后误报 ``no_media_reference`` —— 一条正常消息被记成故障。
    """
    a = _adapter()
    grabbed: list[str] = []

    async def _fake_cache(kind, ref, failures=None):
        grabbed.append(kind)
        return None

    monkeypatch.setattr(a, "_cache_media", _fake_cache)
    with caplog.at_level(logging.WARNING):
        paths, _t, _fails = await a._extract_media(
            {"msgtype": "voice", "voice": {"content": "帮我查下天气"}})

    assert grabbed == [], f"voice 走进了下载路径:{grabbed}"
    assert paths == []
    assert "no_media_reference" not in caplog.text, (
        f"合法语音被误报成媒体获取失败:{caplog.text!r}")
    assert "未处理的 WeCom msgtype" not in caplog.text, (
        "voice 是已知类型,⛔ 不该被当成未知类型告警")


@pytest.mark.asyncio
async def test_unknown_msgtype_leaves_a_trace(caplog):
    """⭐ 闭集兜底：不管它是什么，没被任何分支接住就必须留痕。

    ⛔ 判据不是「我列到了哪几种形状」—— 那是开集，给没列到的发免检。
    """
    a = _adapter()
    with caplog.at_level(logging.WARNING):
        paths, _t, _fails = await a._extract_media({"msgtype": "some_future_type"})
    assert paths == []
    assert "未处理的 WeCom msgtype" in caplog.text, (
        f"未知 msgtype 被静默丢弃,没有任何痕迹:{caplog.text!r}")


def test_msgtype_sets_match_the_official_protocol_exactly():
    """🔴 协议锚点门：三个集合必须与官方定义**逐字相等**。

    ⛔ 上一版写的是**子集**关系（``<=``）—— 于是往媒体集合里塞任意虚构类型，
    十条测试**全绿**。⭐ 子集断言对「多了什么」完全免疫，而「多了一个不该
    下载的类型」恰恰就是我犯的那个错。

    oracle 来自**仓外的协议事实**，⛔ 不从实现导出（否则门与实现共享判据，
    实现错了门也跟着错）：
        https://github.com/WecomTeam/aibot-node-sdk
        src/types/message.ts —— ``MessageType`` 枚举 + 各 ``*Message`` 接口
      * 顶层 msgtype = text / image / mixed / voice / file / video
      * image·file·video → ``url``（5 分钟有效、已加密）+ ``aeskey``  ⇒ 可下载
      * voice            → 仅 ``content: string``（转写文本）          ⇒ ⛔ 不可下载
      * mixed.msg_item   → 仅 ``'text' | 'image'``
    """
    assert _WECOM_DOWNLOADABLE_MSGTYPES == {"image", "file", "video"}, (
        "可下载集合与官方 url+aeskey 的三种类型不符 —— "
        "多一个会让合法消息误报失败,少一个会静默丢附件")
    assert _WECOM_TRANSCRIPT_MSGTYPES == {"voice"}
    assert _WECOM_MIXED_ITEM_MSGTYPES == {"text", "image"}

    # 🔴 官方全集：**逐字相等**，⛔ 不是子集。
    # 上一版这里写的是 ``{...六种} <= _WECOM_KNOWN_MSGTYPES`` —— 子集断言，
    # 往 known 里塞任意虚构类型（RH 实测 ``future_fake``）测试仍绿，
    # 而那个类型会**绕过未知类型告警**。
    # ⚠️ 我却在报告里声称「逐字相等」⇒ 虚假声明，已经被转述给用户。
    assert _WECOM_OFFICIAL_MSGTYPES == {
        "text", "image", "mixed", "voice", "file", "video"}, (
        "官方 msgtype 全集与 SDK MessageType 枚举不符")

    # 本地扩展必须**逐项有代码依据**（见常量处注释），且与官方集合不重叠。
    assert _WECOM_LOCAL_KNOWN_MSGTYPES == {"appmsg", "event"}, (
        "本地扩展集合变了 —— 每一项都必须在 adapter 里有真实处理分支，"
        "⛔ 不许塞「印象里有」的名字（stream 就是这么进来又被删掉的）")
    assert not (_WECOM_OFFICIAL_MSGTYPES & _WECOM_LOCAL_KNOWN_MSGTYPES)

    # ⇒ known 全集 = 两者并集，**相等**。这才是闭集。
    assert _WECOM_KNOWN_MSGTYPES == (
        _WECOM_OFFICIAL_MSGTYPES | _WECOM_LOCAL_KNOWN_MSGTYPES)

    # 可下载 / 仅转写 ⛔ 不许重叠 —— 重叠就是上一版那个 bug 的形状
    assert not (_WECOM_DOWNLOADABLE_MSGTYPES & _WECOM_TRANSCRIPT_MSGTYPES)


def test_every_local_extension_has_a_real_handler():
    """🔴 本地扩展的每一项都必须在代码里**真的被处理**。

    ``stream`` 当初就是这么混进来的：登记在 known 集合里、**全仓零处理逻辑**、
    官方枚举里也没有。它的后果是「一个我们其实不认识的类型悄悄绕过未知告警」。
    ⭐ 与把 ``voice`` 并进可下载集同形：为不存在的东西写了登记。
    """
    import inspect
    from pathlib import Path

    from plugins.platforms.wecom import adapter as wa
    from plugins.platforms.wecom import callback_adapter as wca

    sources = "\n".join(
        Path(inspect.getfile(m)).read_text(encoding="utf-8") for m in (wa, wca))
    for name in _WECOM_LOCAL_KNOWN_MSGTYPES:
        # 出现在集合定义之外的地方 = 有真实分支在比较它
        occurrences = sources.count(f'"{name}"') + sources.count(f"'{name}'")
        assert occurrences >= 2, (
            f"本地扩展 {name!r} 只出现在集合登记里，没有任何处理分支 —— "
            f"它会悄悄绕过未知类型告警（stream 就是这么混进来的）")
