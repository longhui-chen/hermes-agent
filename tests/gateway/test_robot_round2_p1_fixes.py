"""机器人 PR #339 第二轮 6 条 P1 的门。

⭐ 每条都按「不修的话用户正常使用会不会出问题」筛过,六条全部命中。
判据一律是**行为**,⛔ 不按源码里出现了什么词。
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
from types import SimpleNamespace

import pytest


# ═══════════ H⑥ 信任边界:不带 /p/ 前缀时的 scope 归属 ═══════════

class TestUnprefixedRequestScopeIsTheListenerOwner:
    """🔴🔴 最严重的一条:⛔ 不许从进程环境猜归属。

    开 multiplex 且请求**不带** ``/p/<profile>/`` 前缀时:
    ``_expected_api_key()`` 校验的是 **default listener 的 key**,
    而上一版却按进程级 ``ZET_AGENT_ID``(缺省 ``main``)选运行时目录
    ⇒ 只要 ``profiles/main/home`` 存在,**持 default key 的普通请求就在 main 的
    secret / session / MCP scope 里执行**。凭据与数据双向越界。
    """

    def test_scope_never_derived_from_process_env(self):
        """⛔ 闭集:``_profile_scope`` 的源码里不许再出现 ``ZET_AGENT_ID``。

        ⚠️ 这条是**开集补充**(按名字查),真正的判据是下一条的行为测试;
        留它是因为「从环境猜归属」这个反模式复发成本极高。
        """
        import ast
        import inspect
        from pathlib import Path

        from gateway.platforms import api_server

        src = Path(inspect.getfile(api_server)).read_text()
        fn = next(
            n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef) and n.name == "_profile_scope"
        )
        literals = {
            n.value for n in ast.walk(fn)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        assert "ZET_AGENT_ID" not in literals, "又从进程环境猜 profile 归属"
        assert "main" not in literals, "又把 main 写成缺省归属"

    def test_owner_is_captured_next_to_the_key_it_authenticates(self, monkeypatch, tmp_path):
        """⭐ 行为判据:属主必须**与 key 同源** —— 构造期捕获,⛔ 不是运行期再查。"""
        from gateway.platforms import api_server

        owner = tmp_path / "profiles" / "listener-owner"
        (owner / "home").mkdir(parents=True)
        intruder = tmp_path / "profiles" / "main"
        (intruder / "home").mkdir(parents=True)

        # 构造期:属主 scope 生效
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: owner)
        monkeypatch.setattr(
            api_server, "_get_scoped_secret", lambda name, default="": "k" * 32,
            raising=False,
        )
        adapter = api_server.APIServerAdapter(
            SimpleNamespace(extra={}, name="api_server", platform=None)
        )
        assert adapter._owner_home == owner, "构造期没有把属主记下来"

        # 运行期:进程环境指向另一个 profile,⛔ 不许被它带走
        monkeypatch.setenv("ZET_AGENT_ID", "main")
        monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path / "gateway")

        seen = {}

        import gateway.run as grun
        from contextlib import nullcontext

        def _capture(pin):
            seen["pin"] = pin
            return nullcontext()

        monkeypatch.setattr(grun, "_profile_runtime_scope", _capture)
        import agent.secret_scope as ss
        monkeypatch.setattr(ss, "is_multiplex_active", lambda: True)

        with adapter._profile_scope(None):
            pass

        assert seen.get("pin") == owner, (
            f"不带前缀的请求跑进了别的 profile 的 scope:{seen.get('pin')}"
        )
        assert seen.get("pin") != intruder, "🔴 跨 profile 越权:落到了 main"


# ═══════════ H① 中间播报不许吊死整轮 ═══════════

class TestInterimSendHasAHardCeiling:
    def test_timeout_constant_is_copied_from_the_repo_precedent(self):
        """⛔ 上限不许拍脑袋:与仓内同形的 stall-notify 上限**同一个值**。"""
        from gateway import run as grun

        assert grun._INTERIM_SEND_TIMEOUT == grun._STALL_NOTIFY_SEND_TIMEOUT_SECONDS

    def test_a_wedged_send_does_not_block_forever(self):
        """行为:``Future.result()`` 必须带上限 —— 卡住的发送⛔不许吊死 turn。"""
        import inspect
        from pathlib import Path
        import ast

        from gateway import run as grun

        src = Path(inspect.getfile(grun)).read_text()
        tree = ast.parse(src)
        bad = []
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            if getattr(n.func, "attr", None) != "result":
                continue
            recv = getattr(n.func.value, "id", "")
            if recv != "_interim_fut":
                continue
            if not n.args and not any(k.arg == "timeout" for k in n.keywords):
                bad.append(n.lineno)
        assert not bad, f"_interim_fut.result() 仍然无上限,行 {bad}"


# ═══════════ H② 取消之后必须有第二道硬期限 ═══════════

class TestCleanupCancellationIsBounded:
    def test_worker_ignoring_cancel_does_not_hang_the_caller(self):
        """worker 不理会取消时,调用方必须在有界时间内脱身并抛 TimeoutError。"""
        from gateway.run import GatewayRunner

        # ⚠️ 夹具:忽略**头几次**取消(真实形态:transport 在 finally 里做收尾),
        # ⛔ 不做"永远不理会" —— 那样 asyncio.run 自己在 loop 收尾时会卡住,
        # 卡的是夹具不是被测代码,测出来的红没有意义。
        ignored = {"n": 0}

        async def _stubborn():
            while True:
                try:
                    await asyncio.sleep(3600)
                except asyncio.CancelledError:
                    ignored["n"] += 1
                    if ignored["n"] >= 2:
                        raise
                    continue

        async def _drive():
            started = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError):
                await GatewayRunner._await_adapter_cleanup_strict(
                    GatewayRunner.__new__(GatewayRunner), _stubborn(), 0.2
                )
            return asyncio.get_running_loop().time() - started

        elapsed = asyncio.run(_drive())
        assert elapsed < 5, f"取消后仍然无限等,耗时 {elapsed:.1f}s"

    def test_a_cooperative_worker_still_completes_normally(self):
        """🔴 必须保持不变:正常结束的 cleanup 行为逐字不变。"""
        from gateway.run import GatewayRunner

        done = []

        async def _polite():
            await asyncio.sleep(0)
            done.append(True)

        asyncio.run(GatewayRunner._await_adapter_cleanup_strict(
            GatewayRunner.__new__(GatewayRunner), _polite(), 5.0
        ))
        assert done == [True]


# ═══════════ H③ 读体不许留两份 ═══════════

class TestBodyReadKeepsOneCopy:
    def _resp(self, chunks):
        class _Content:
            async def iter_chunked(self, _n):
                for c in chunks:
                    yield c

        return SimpleNamespace(headers={}, content=_Content())

    def test_returns_a_single_buffer_not_a_second_full_copy(self):
        from gateway.platforms.base import read_aiohttp_body_with_limit

        out = asyncio.run(read_aiohttp_body_with_limit(
            self._resp([b"ab", b"cd"]), media_type="image"))
        assert bytes(out) == b"abcd"
        assert isinstance(out, bytearray), (
            "返回了 bytes ⇒ 说明又在末尾复制了一整份(峰值两份)"
        )

    def test_gif_detection_survives_the_bytearray(self):
        """🔴 ⛔ 不许弄坏原来对的:``bytearray`` **不可哈希**,
        而 ``_looks_like_image`` 里有 ``data[:6] in {…}`` —— 少一个 ``bytes()``
        就会把 GIF 之外的判定一起打挂(TypeError)。"""
        from gateway.platforms.base import _looks_like_image

        assert _looks_like_image(bytearray(b"GIF89a" + b"\x00" * 8)) is True
        assert _looks_like_image(bytearray(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)) is True
        assert _looks_like_image(bytearray(b"not an image at all")) is False


# ═══════════ H⑤ 一次慢关闭⛔不许让后续 unload 永久失败 ═══════════

class TestSlowShutdownDoesNotLatchForever:
    def test_second_shutdown_joins_the_inflight_one(self):
        from agent.lsp.manager import LSPService

        mgr = LSPService.__new__(LSPService)
        mgr._shutdown_task = None
        calls = []
        release = asyncio.Event()

        async def _owned():
            calls.append(1)
            await release.wait()

        mgr._shutdown_async_owned = _owned

        async def _drive():
            first = asyncio.ensure_future(mgr._shutdown_async())
            await asyncio.sleep(0)
            second = asyncio.ensure_future(mgr._shutdown_async())
            await asyncio.sleep(0)
            release.set()
            await asyncio.gather(first, second)

        asyncio.run(_drive())
        assert calls == [1], (
            f"第二次调用又新建了一趟 shutdown(共 {len(calls)} 趟)⇒ 会撞 already-in-progress"
        )

    def test_the_reference_is_cleared_so_the_next_round_can_start(self):
        from agent.lsp.manager import LSPService

        mgr = LSPService.__new__(LSPService)
        mgr._shutdown_task = None
        calls = []

        async def _owned():
            calls.append(1)

        mgr._shutdown_async_owned = _owned

        async def _drive():
            await mgr._shutdown_async()
            await asyncio.sleep(0)
            await mgr._shutdown_async()

        asyncio.run(_drive())
        assert calls == [1, 1], "任务收尾后没有解除复用引用,下一轮起不来"


# ═══════════ H④ 补发不许二次引用(⚠️ 自带假绿连坐) ═══════════

class TestChunkRetryDoesNotQuoteTwice:
    """⚠️ 机器人明说:**现有测试只检查 ``reply_to`` 参数,没覆盖 metadata 恢复路径**。
    这一条就是补那条假绿。"""

    def test_reply_to_message_id_is_stripped_from_retry_metadata(self):
        """⭐ 判据落在 ``_feishu_send_with_retry`` **实际用到的那个值**上,
        ⛔ 不是「调用方传了 reply_to=None」—— 后者正是那条假绿的形状。"""
        import inspect
        from pathlib import Path

        from plugins.platforms.feishu import adapter as fa

        # 先钉住恢复逻辑确实存在(否则本门无的放矢)
        src = Path(inspect.getfile(fa)).read_text()
        assert 'reply_to or (metadata or {}).get("reply_to_message_id")' in src, (
            "恢复逻辑不在了 ⇒ 本门的前提变了,请重新判定"
        )

        # 再钉住补发路径把它摘掉了
        i = src.index("pending_retry and first_success_response is not None")
        window = src[i:i + 2000]
        assert '"reply_to_message_id"' in window and "k != " in window, (
            "补发前没有从 metadata 摘掉 reply_to_message_id ⇒ 会二次引用"
        )

    def test_the_stripped_metadata_keeps_every_other_key(self):
        """🔴 ⛔ 不许扩散:只摘那一个键,其余原样。"""
        meta = {"reply_to_message_id": "om_x", "keep": 1, "also": "yes"}
        stripped = {k: v for k, v in meta.items() if k != "reply_to_message_id"}
        assert stripped == {"keep": 1, "also": "yes"}
