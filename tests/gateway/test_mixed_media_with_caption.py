"""图文混发:⛔ 一样不许吞掉另一样。

用户产品要求(原话):「**附件不管是视频图片还是文档和文字一起作为一条消息发的
时候,要处理好这种情况**」,且「包括其他渠道也要注意」。

═══════════════════════════════════════════════════════════════════════
本门关住 / 仍开集(⭐ 分格声明)
═══════════════════════════════════════════════════════════════════════
**关住(确定性,⛔ 不需要模型)**
  五种混发形状进到 ``/p/{profile}/v1/chat/completions`` 之后,
  **交给 agent 的那份 user_message 里两样都还在**:
    ① 文字 + 图片   ② 文字 + 文件行   ③ 文字 + 图片 + 文件行
    ④ 只有图片      ⑤ 只有文件行
  ⭐ 判据是**真的抓 `_run_agent` 收到的实参**,⛔ 不是「代码里看着像会保留」。
  ⇒ 「有图时文字被丢 / 有文件时图片被丢」这一格被钉死。

**⛔ 仍是开集(明说)**
  1. **模型是否真的用到了两样**,只能真机验 —— 已在云机做过,
     藏了随机标识(文字=紫水晶 / 图片=OTTER-7391 / 文件=BADGER-5528),
     三组混发**全中**;结果与逐字形状契约见
     ``~/Desktop/Test/CONTRACT-hermes-inbound-mixed-media-20260817.md``。
     ⛔ 单模型单次,不当「已证明」。
  2. **「同一条消息回两遍」**属 IM **投递层**,本端点观测不到(``choices`` 恒 1)
     ⇒ **第三态**,⛔ 不记成通过。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter

_KEY = "opensslrandhex32strongkeyformixed"
_AUTH = {"Authorization": f"Bearer {_KEY}"}
TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)
#: ⭐ 每样一个**只有真拿到才认得出**的标识 —— 与真机实验同一套思路。
TEXT_TOKEN = "紫水晶-AMETHYST-9042"
FILE_LINE = "[file: /root/.hermes/uploads/combo-doc.txt]"


def _app():
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
    adapter._check_auth = lambda _request: None  # noqa: E731 —— 见 ingress 门的说明
    app = web.Application()
    app["api_server_adapter"] = adapter
    adapter._register_profile_api_routes(app.router)
    return app, adapter


async def _send(content):
    """打一个真请求,返回 ``_run_agent`` 实际收到的 user_message。"""
    app, adapter = _app()
    captured = {}

    def _spy(*args, **kwargs):
        captured["user_message"] = kwargs.get("user_message", args[0] if args else None)
        captured["trusted"] = kwargs.get("trusted_user_message")
        raise RuntimeError("stop-after-capture")  # ⛔ 不真跑 agent

    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new=MagicMock(side_effect=_spy)):
            await cli.post(
                "/p/main/v1/chat/completions",
                json={"messages": [{"role": "user", "content": content}]},
                headers=_AUTH,
            )
    assert "user_message" in captured, (
        "⛔ `_run_agent` 根本没被调到 ⇒ 请求在更早的地方就被挡了,"
        "**量具坏了**,下面的断言一个都不可信"
    )
    return captured["user_message"]


def _flatten(user_message) -> str:
    if isinstance(user_message, str):
        return user_message
    out = []
    for p in user_message or []:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            out.append(str(p.get("text") or ""))
        elif p.get("type") == "image_url":
            ref = p.get("image_url")
            out.append(str(ref.get("url") if isinstance(ref, dict) else ref))
    return "\n".join(out)


def _has_image(user_message) -> bool:
    return isinstance(user_message, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in user_message
    )


class TestNeitherSideIsSwallowed:
    @pytest.mark.asyncio
    async def test_text_plus_image_keeps_both(self):
        um = await _send([
            {"type": "text", "text": f"看看这张图 {TEXT_TOKEN}"},
            {"type": "image_url", "image_url": {"url": TINY_PNG}},
        ])
        assert TEXT_TOKEN in _flatten(um), "🔴 有图时**文字被丢了** —— 用户明确要求文字不许丢"
        assert _has_image(um), "🔴 图片被丢了"

    @pytest.mark.asyncio
    async def test_text_plus_file_line_keeps_both(self):
        um = await _send(f"看看这个文件 {TEXT_TOKEN}\n{FILE_LINE}")
        flat = _flatten(um)
        assert TEXT_TOKEN in flat, "🔴 有文件时文字被丢了"
        assert FILE_LINE in flat, "🔴 `[file: …]` 行被丢了 ⇒ 模型拿不到文件"

    @pytest.mark.asyncio
    async def test_text_plus_image_plus_file_keeps_all_three(self):
        um = await _send([
            {"type": "text", "text": f"三样一起 {TEXT_TOKEN}\n{FILE_LINE}"},
            {"type": "image_url", "image_url": {"url": TINY_PNG}},
        ])
        flat = _flatten(um)
        assert TEXT_TOKEN in flat, "🔴 三样混发时文字被丢了"
        assert FILE_LINE in flat, "🔴 三样混发时文件行被丢了"
        assert _has_image(um), "🔴 三样混发时图片被丢了"


class TestAttachmentOnlyIsAccepted:
    """⛔ 缺文字不许把整条消息弄失败 —— IM 里「只发一张图」极常见。"""

    @pytest.mark.asyncio
    async def test_image_only(self):
        um = await _send([{"type": "image_url", "image_url": {"url": TINY_PNG}}])
        assert _has_image(um), "只发图片时图片没到 agent"

    @pytest.mark.asyncio
    async def test_file_line_only(self):
        um = await _send(FILE_LINE)
        assert FILE_LINE in _flatten(um), "只发文件时文件行没到 agent"


class TestShapesTheEndpointRefuses:
    """⛔ 防回归:LS 现在不送这三种 part;哪天送了必须**当场 400**,
    ⛔ 不许悄悄放行一个「能过但模型读不到」的形状。"""

    @pytest.mark.parametrize(
        "part",
        [
            {"type": "file", "file": {"file_id": "f_1"}},
            {"type": "input_file", "input_file": {"file_id": "f_1"}},
            {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        ],
    )
    @pytest.mark.asyncio
    async def test_unsupported_parts_still_400(self, part):
        app, adapter = _app()
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_run_agent", new=MagicMock()):
                resp = await cli.post(
                    "/p/main/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "x"}, part,
                    ]}]},
                    headers=_AUTH,
                )
        assert resp.status == 400, (
            f"{part['type']} 不再 400 ⇒ 要么真支持了(那要有证据证明模型读得到),"
            "要么悄悄吞掉了 —— 后者是把失败伪装成成功"
        )
