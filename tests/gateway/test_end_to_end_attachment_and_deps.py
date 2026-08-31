"""端到端门:验「这条链整体通不通」,⛔ 不是「我改的那一点对不对」。

## 🔴 为什么补这个

这一轮四条 finding 里,三条的措辞是「**完全不可用**」「**完全失去能力**」:

| finding | 说明什么 |
|---|---|
| `model_readability.py` ⇒ Modal 会话完全失去图片与文档处理能力 | **新模块**,那条路从没被跑过 |
| `bedrock_adapter.py` ⇒ Bedrock 对一类正常部署完全不可用 | **新路径**,那两条 raise 从没被送到用户侧过 |

⭐ **「完全不可用」意味着那条链从来没有被真正跑通过一次。**
有界口径 + 逆改验的是「我改的那一点对不对」,**⛔ 验不了「整条链通不通」**。
这是两个不同的判据。

## 实查到的覆盖缺口(本门就是来补它的)

* `_build_media_placeholder` 的既有端到端测试**只跑 `local` 和 `docker`**
  (`test_media_placeholder_agent_path.py` 里 `TERMINAL_ENV` 只出现这两个值)
  ⇒ 其余 **5 个** `KNOWN_ENVIRONMENTS` 从没走过真实消费者。
* `_require_boto3()` 的两条 raise **没有任何测试触发过**
  (`git grep _require_boto3 -- tests/` 只命中版本比较的成功路径)。
"""
from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import pytest

from gateway.model_readability import KNOWN_ENVIRONMENTS


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    import hermes_constants

    root = tmp_path / "hermes"
    (root / "cache" / "documents").mkdir(parents=True)
    monkeypatch.setattr(hermes_constants, "get_hermes_dir",
                        lambda new_subpath, old_name=None: root / new_subpath)
    return root / "cache" / "documents"


def _event(paths, types):
    return SimpleNamespace(
        media_urls=list(paths), media_types=list(types),
        message_type=None, caption=None,
    )


# ═══════════ ① 附件链:输入 → 可读性契约 → 真实消费者 → 可断言输出 ═══════════

class TestAttachmentChainEndToEndForEveryEnvironment:
    """⭐ 走**真实消费者** ``gateway.run._build_media_placeholder``,
    ⛔ 不是直接调 ``verify_artifact_readable``(那只验中间一段)。"""

    @pytest.mark.parametrize("env", sorted(KNOWN_ENVIRONMENTS))
    def test_every_environment_produces_either_a_path_or_an_actionable_failure(
        self, env, cache_root, monkeypatch
    ):
        """闭集:**每一个**执行环境都必须落在两种终态之一,⛔ 不许有第三态。

        ⚠️ 这正是上一轮 Modal 出事的形状:它悄悄落在「所有附件都不可用」那一侧,
        而**没有任何测试以 modal 走过这条链**,所以没人知道。
        """
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        src = cache_root / "report.pdf"
        src.write_bytes(b"%PDF-1.7 hello")
        monkeypatch.setenv("TERMINAL_ENV", env)

        import tools.credential_files as cf
        monkeypatch.setattr(
            cf, "map_cache_path_to_container",
            lambda host, container_base="/root/.hermes": "/root/.hermes/cache/documents/report.pdf",
            raising=False,
        )

        out = _build_media_placeholder(_event([str(src)], ["application/pdf"]))

        assert out, f"{env}: 消费者产出了空字符串 —— 附件被静默丢弃"
        delivered = str(src) in out or "/root/.hermes" in out
        refused = "could not be read" in out
        assert delivered or refused, (
            f"{env}: 既没交付路径也没给出明确失败 —— 第三态:{out!r}"
        )
        if refused:
            # ⛔ 失败时一个字都不许含原始主机路径(族 A 的现场)
            assert str(src) not in out, f"{env}: 失败文案里泄漏了主机路径"

    @pytest.mark.parametrize("env", ["local", "docker", "modal"])
    def test_these_three_must_actually_deliver_a_path(
        self, env, cache_root, monkeypatch
    ):
        """🔴 这三个环境**有**回执机制,⛔ 不许退化成「不可用」。

        ⭐ ``modal`` 在这份名单里是本轮新加的 —— 上一轮它被按 backend 名字
        无条件拒,而 ``tools/environments/modal.py`` 早已挂载并同步 cache。
        """
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        src = cache_root / "report.pdf"
        src.write_bytes(b"%PDF-1.7 hello")
        monkeypatch.setenv("TERMINAL_ENV", env)

        import tools.credential_files as cf
        monkeypatch.setattr(
            cf, "map_cache_path_to_container",
            lambda host, container_base="/root/.hermes": "/root/.hermes/cache/documents/report.pdf",
            raising=False,
        )

        out = _build_media_placeholder(_event([str(src)], ["application/pdf"]))
        assert "could not be read" not in out, (
            f"{env}: 有回执机制却把附件判成不可用 ⇒ 该环境的用户完全失去附件能力"
        )

    def test_a_genuinely_unreadable_file_is_refused_everywhere(self, cache_root, monkeypatch):
        """负对照:文件真的不存在时,⛔ 不许有任何环境放行(⇒ 上面的绿不是恒真)。"""
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        monkeypatch.setenv("TERMINAL_ENV", "local")
        out = _build_media_placeholder(
            _event([str(cache_root / "gone.pdf")], ["application/pdf"]))
        assert "could not be read" in out


# ═══════════ ② 依赖校验链:抛出点 → 错误归属 → 用户可见文案 ═══════════

class TestBedrockDependencyErrorReachesTheUserIntact:
    """⭐ 走完整条链:``_require_boto3()`` → ``classify`` → ``client_safe_error_text``。

    ⚠️ 实查:这两条 raise 此前**没有任何测试触发过**,更没有测试把它们送到
    用户可见文案。于是「内建异常 ⇒ 判成我们的 bug ⇒ 收成『服务内部异常』」
    这条链上的缺陷,只能等真实用户撞上。
    """

    def _user_text(self, exc):
        from agent.error_classifier import classify_api_error, client_safe_error_text

        return client_safe_error_text(classify_api_error(exc), str(exc), error=exc)

    def test_missing_boto3_keeps_the_install_command(self, monkeypatch):
        import builtins

        import agent.bedrock_adapter as ba

        real_import = builtins.__import__

        def _no_boto3(name, *a, **kw):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _no_boto3)
        with pytest.raises(ImportError) as caught:
            ba._require_boto3()
        monkeypatch.undo()

        text = self._user_text(caught.value)
        assert "pip install boto3" in text, (
            f"用户唯一能照做的那条命令被收掉了,只剩:{text!r}"
        )

    def test_outdated_boto3_keeps_the_upgrade_command(self, monkeypatch):
        import agent.bedrock_adapter as ba

        fake = SimpleNamespace(__version__="1.34.46")
        monkeypatch.setitem(__import__("sys").modules, "boto3", fake)
        with pytest.raises(RuntimeError) as caught:
            ba._require_boto3()

        text = self._user_text(caught.value)
        assert "pip install --upgrade boto3" in text, (
            f"升级命令被收成了通用文案:{text!r}"
        )

    def test_a_real_internal_bug_is_still_collapsed(self):
        """负对照:没盖戳的内部异常仍然被收 ⇒ ⛔ 判据不是恒放行。"""
        from agent.error_classifier import INTERNAL_ERROR_USER_TEXT

        text = self._user_text(RuntimeError("profile resolution produced no kwargs"))
        assert text == INTERNAL_ERROR_USER_TEXT


# ═══════════ 第六轮:两条本批引入的新缺陷 ═══════════

class TestFifoAttachmentCannotWedgeTheEventLoop:
    """🔴 无 writer 的 FIFO 会让 ``os.open(O_RDONLY)`` **无限阻塞**,
    而本函数在 gateway 事件循环上**同步**跑 ⇒ 一条附件卡死所有会话。
    ⭐ 类型检查原本在 open **之后** —— 检查根本跑不到。"""

    def test_a_writerless_fifo_is_refused_quickly(self, cache_root, monkeypatch):
        import os as _os
        import time

        from gateway.model_readability import verify_artifact_readable

        fifo = cache_root / "trap.fifo"
        _os.mkfifo(fifo)
        monkeypatch.setenv("TERMINAL_ENV", "local")

        started = time.monotonic()
        r = verify_artifact_readable(str(fifo))
        elapsed = time.monotonic() - started

        assert elapsed < 5, f"在 FIFO 上挂了 {elapsed:.1f}s —— 事件循环会被卡死"
        assert r.ok is False
        assert "lstat_regular" not in r.checks or r.failure_code

    def test_a_normal_file_is_unaffected(self, cache_root, monkeypatch):
        """🔴 必须保持不变:普通文件判定逐字不变。"""
        from gateway.model_readability import verify_artifact_readable

        monkeypatch.setenv("TERMINAL_ENV", "local")
        src = cache_root / "ok.pdf"
        src.write_bytes(b"%PDF-1.7 hi")
        r = verify_artifact_readable(str(src))
        assert r.ok and "read_ok" in r.checks and "lstat_regular" in r.checks

    def test_a_directory_is_still_refused(self, cache_root, monkeypatch):
        """🔴 必须保持不变:目录仍然被拒。"""
        from gateway.model_readability import verify_artifact_readable

        monkeypatch.setenv("TERMINAL_ENV", "local")
        d = cache_root / "sub"
        d.mkdir()
        assert verify_artifact_readable(str(d)).ok is False

    def test_a_symlink_is_still_refused(self, cache_root, monkeypatch):
        """🔴 必须保持不变:``lstat`` **不跟随链接**,符号链接仍被挡。"""
        import os as _os

        from gateway.model_readability import verify_artifact_readable

        monkeypatch.setenv("TERMINAL_ENV", "local")
        real = cache_root / "real.pdf"
        real.write_bytes(b"%PDF-1.7 x")
        link = cache_root / "link.pdf"
        _os.symlink(real, link)
        assert verify_artifact_readable(str(link)).ok is False


class TestNamedMainProfileIsNotMergedIntoDefault:
    """🔴 用户自建的 named profile ``main`` 与内建 ``default`` 是**两个**东西。

    合并之后卸载其一会把另一个的 adapter operation / cleanup 一起拖走 ⇒
    **无关 profile 的消息渠道在配置重载或 cron reconcile 时被断开。**
    """

    def test_main_is_its_own_lifecycle_identity(self):
        from gateway.run import GatewayRunner

        assert GatewayRunner._profile_runtime_aliases("main") == ("main",), (
            "named profile `main` 仍被并进 default 生命周期"
        )

    def test_default_and_empty_are_still_the_same_identity(self):
        """🔴 必须保持不变:空名称与 ``default`` 确实是同一个东西。"""
        from gateway.run import GatewayRunner

        assert GatewayRunner._profile_runtime_aliases("") == ("", "default")
        assert GatewayRunner._profile_runtime_aliases(None) == ("", "default")
        assert GatewayRunner._profile_runtime_aliases("default") == ("", "default")

    def test_other_names_are_unchanged(self):
        """🔴 必须保持不变:其它名字各自独立。"""
        from gateway.run import GatewayRunner

        assert GatewayRunner._profile_runtime_aliases("alice") == ("alice",)
        assert GatewayRunner._profile_runtime_aliases("  bob  ") == ("bob",)

    def test_main_is_not_a_reserved_name(self):
        """⭐ 前提校验:``main`` 确实**没有**被 profile 名校验保留 ——
        ⛔ 否则本修复的前提就不成立。"""
        from hermes_cli.profiles import validate_profile_name

        try:
            validate_profile_name("main")
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"`main` 其实是保留名({exc})⇒ 本修复的前提要重判")


# ═══════════ 第七轮:③ 远程 URL · ④ 超时不重传 ═══════════

class TestRemoteUrlAttachmentsSkipLocalFileChecks:
    """🔴 **本批的可读性契约弄坏了 Discord 的既有降级路径。**

    ⚠️ **7 环境端到端门为什么没抓住?** —— 它的夹具**只喂本地路径**,
    ⛔ 没有「远程 URL」那一格。⇒ 本类就是把那一格补进去。
    """

    def test_a_discord_cdn_url_is_not_run_through_local_file_checks(self, monkeypatch):
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        monkeypatch.setenv("TERMINAL_ENV", "local")
        url = "https://cdn.discordapp.com/attachments/1/2/photo.png"
        out = _build_media_placeholder(_event([url], ["image/png"]))

        assert url in out, f"合法的 CDN URL 被判成读不到:{out!r}"
        assert "could not be read" not in out

    @pytest.mark.parametrize("scheme_url", [
        "http://example.invalid/a.png",
        "https://example.invalid/a.png",
    ])
    def test_other_remote_schemes_too(self, scheme_url, monkeypatch):
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        monkeypatch.setenv("TERMINAL_ENV", "local")
        assert "could not be read" not in _build_media_placeholder(
            _event([scheme_url], ["image/png"]))

    def test_local_paths_are_still_verified(self, cache_root, monkeypatch):
        """🔴 必须保持不变:本地路径仍走可读性校验(⛔ 判据不是恒放行)。"""
        import asyncio as _asyncio
        from gateway.run import _build_media_placeholder as _bmp_async

        def _build_media_placeholder(_e):
            # ⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。
            # 只改**调用方式**,下面的断言逐字不变。
            return _asyncio.run(_bmp_async(_e))

        monkeypatch.setenv("TERMINAL_ENV", "local")
        out = _build_media_placeholder(
            _event([str(cache_root / "gone.png")], ["image/png"]))
        assert "could not be read" in out

    def test_file_scheme_is_treated_as_local(self, cache_root, monkeypatch):
        """🔴 ``file://`` **不算远程** —— 它就是本地文件,该被校验。"""
        from gateway.run import _is_remote_media_ref

        assert _is_remote_media_ref("file:///tmp/x.png") is False
        assert _is_remote_media_ref("/var/data/x.png") is False
        assert _is_remote_media_ref("pic.png") is False
        assert _is_remote_media_ref("C:\\Users\\x\\a.png") is False
        assert _is_remote_media_ref("https://h/x.png") is True


class TestKanbanTimeoutIsNotRetried:
    """🔴 非幂等上传:超时**可能已送达** ⇒ ⛔ 不重传,否则用户收到两份。

    ⭐ 兄弟调用点在**同一个函数里**:``SendResult`` 分支早就先判超时,
    异常分支没跟上。
    """

    def test_the_exception_branch_checks_timeout_before_retryable(self):
        import ast
        import inspect

        import gateway.kanban_watchers as kw

        src = pathlib.Path(inspect.getfile(kw)).read_text()
        i = src.index('err = f"{type(exc).__name__}: {exc}"')
        j = src.index("_is_retryable_error(err)", i)
        window = src[i:j]
        assert "_is_timeout_error(err)" in window, (
            "异常分支仍然先问 _is_retryable_error ⇒ "
            'ConnectionError("Read timed out") 会被重传,用户收到重复文件'
        )

    def test_both_branches_use_the_same_predicate(self):
        """⭐ 照抄自证:两条分支用的是**同一个** helper,⛔ 没有第二套判据。"""
        import inspect
        import re

        import gateway.kanban_watchers as kw

        src = pathlib.Path(inspect.getfile(kw)).read_text()
        assert len(re.findall(r"_is_timeout_error\(err\)", src)) >= 2, (
            "只有一条分支在判超时"
        )


# ═══════════ 第八轮:四条(两条是本轮自己弄坏的) ═══════════

class TestResponseFormatCapabilityStaysA400:
    """🔴 **上一轮我收窄了 catch,却没把所有 raise 点跟上** —— 兄弟调用点。

    HTTP 边界改成只认 ``ResponseFormatValidationError`` 才回 400;而
    Anthropic / Gemini transport 的 capability 检查仍抛普通 ``ValueError``
    ⇒ **原本可操作的 400 退化成「服务内部异常」的 500**。
    """

    def test_every_capability_raise_uses_the_typed_error(self):
        """⚠️ **量具教训**:第一版我拿 grep 的行号,却又用 ``src.index()``
        重新定位 ⇒ 判据落到了**文件里第一处同名字符串**上 —— 对
        ``api_server.py`` 命中的是 ``return`` 的展示文案(完全另一个角色),
        对 ``chat_completions.py`` 则**永远只看两处中的第一处**。
        ⭐ 这个方向是红的所以看得见;镜像方向(第一处恰好合格)会**出生即空转**。
        ⇒ 一律按行号定位,并按**语句关键字**分角色。
        """
        import subprocess

        out = subprocess.run(
            ["git", "grep", "-n", "response_format is not supported", "--", "*.py"],
            capture_output=True, text=True).stdout
        sites = [l for l in out.splitlines() if not l.startswith("tests/")]
        assert len(sites) >= 5, f"探针只命中 {len(sites)} 处 —— 量具可疑"

        raises, returns = [], []
        for line in sites:
            path, lineno, _ = line.split(":", 2)
            body = pathlib.Path(path).read_text().splitlines()
            # 回溯到语句开头(消息可能独占一行)
            for i in range(int(lineno) - 1, max(-1, int(lineno) - 6), -1):
                stripped = body[i].lstrip()
                if stripped.startswith("raise "):
                    raises.append((f"{path}:{i + 1}", stripped)); break
                if stripped.startswith("return "):
                    returns.append(f"{path}:{i + 1}"); break
            else:
                pytest.fail(f"{path}:{lineno} 既不是 raise 也不是 return —— 量具没覆盖这个形状")

        assert raises, "一个 raise 点都没找到 ⇒ 判据维度错了"
        for coord, stmt in raises:
            assert "ResponseFormatValidationError" in stmt, (
                f"{coord} 仍抛普通 ValueError ⇒ 可操作的 400 退化成 500:{stmt[:80]}"
            )
        # 🔴 必须保持不变:展示侧那两处**就该**是 return,⛔ 不许被这道门推着改成 raise
        assert returns, "展示侧的 return 消失了 ⇒ 有人把边界文案改成了抛异常"

    def test_the_typed_error_is_still_a_valueerror(self):
        """🔴 必须保持不变:既有 ``except ValueError`` 调用点行为不变。"""
        from agent.response_format import ResponseFormatValidationError

        assert issubclass(ResponseFormatValidationError, ValueError)


class TestLspShutdownLatchIsReleasedWithTheReference:
    """🔴 **半条链**:我复用了 task,却让它的可观察结果先蒸发。

    成功路径刻意不重置 ``_shutdown_in_progress``,而 ``_clear_shutdown_task``
    立刻清掉唯一引用 ⇒ 下一次 unload 新建 owner **稳定**撞 already-in-progress
    ⇒ **该 profile 后续卸载/重载永久失败**。
    """

    def test_clean_completion_releases_the_latch_atomically(self):
        import asyncio
        import threading

        from agent.lsp.manager import LSPService

        mgr = LSPService.__new__(LSPService)
        mgr._shutdown_task = None
        mgr._shutdown_in_progress = False
        mgr._state_lock = threading.RLock()
        calls = []

        async def _owned():
            calls.append(1)
            mgr._shutdown_in_progress = True   # 模拟成功路径:闩留着

        mgr._shutdown_async_owned = _owned

        async def _drive():
            await mgr._shutdown_async()
            await asyncio.sleep(0)
            return mgr._shutdown_in_progress, mgr._shutdown_task

        latch, ref = asyncio.run(_drive())
        assert ref is None, "复用引用没清 ⇒ 下一轮起不来"
        assert latch is False, (
            "闩没跟着放 ⇒ 下一次 unload 稳定撞 already-in-progress,永久失败"
        )

    def test_a_failed_shutdown_does_not_touch_the_latch(self):
        """🔴 ⛔ 不许塌缩到另一端:失败/取消路径的闩由 owner 自己负责。"""
        import asyncio
        import threading

        from agent.lsp.manager import LSPService

        mgr = LSPService.__new__(LSPService)
        mgr._shutdown_task = None
        mgr._shutdown_in_progress = False
        mgr._state_lock = threading.RLock()

        async def _owned():
            mgr._shutdown_in_progress = True
            raise RuntimeError("boom")

        mgr._shutdown_async_owned = _owned

        async def _drive():
            with pytest.raises(RuntimeError):
                await mgr._shutdown_async()
            await asyncio.sleep(0)
            return mgr._shutdown_in_progress

        assert asyncio.run(_drive()) is True, (
            "失败路径也被清了闩 ⇒ 会把「正在跑的另一趟」误判成空闲"
        )


class TestCopilotAcpErrorsKeepTheirUpstreamOrigin:
    """⭐ 出身声明的**第五个**兄弟,也是**第一个不在 Responses 链上的** ——
    正好落在我那道闭集门自己声明的开集里(文案一个关键词都不带)。"""

    def test_the_jsonrpc_wrap_point_declares_upstream(self):
        """🔴 **这道门第一版出生即空转。**

        第一版判据是「raise 附近 700 字符内出现过 ``declare_upstream_origin``」——
        而窗口里那两次命中**有一次是 ``from ... import`` 行**。逆改把调用换成
        no-op(import 原样留着)⇒ **门照样绿**。
        ⭐ 判据必须落在 **raise 语句自己**上,⛔ 不是「附近有没有这个词」。
        """
        import subprocess

        out = subprocess.run(
            ["git", "grep", "-n", "Copilot ACP {method} failed",
             "--", "agent/copilot_acp_client.py"],
            capture_output=True, text=True).stdout.splitlines()
        assert len(out) == 1, f"命中数变了({len(out)})—— 逐个重判"

        body = pathlib.Path("agent/copilot_acp_client.py").read_text().splitlines()
        lineno = int(out[0].split(":", 2)[1])
        for i in range(lineno - 1, max(-1, lineno - 5), -1):
            stmt = body[i].lstrip()
            if stmt.startswith("raise "):
                break
        else:
            pytest.fail("找不到 raise 语句 —— 判据维度错了")

        assert stmt.startswith("raise declare_upstream_origin("), (
            f"ACP 的 JSON-RPC error 没被包住(实际:{stmt[:70]})⇒ 判成我们的 bug、"
            "不重试不 fallback,认证/限流说明被换成「服务内部异常」"
        )


class TestOnboardingCloseIsBounded:
    def test_the_unload_close_has_a_hard_deadline(self):
        """⚠️ 同一个量具坑:``src.index`` 命中的是 :874 的 ``def``。⇒ 按行号取**调用点**。"""
        import subprocess

        out = subprocess.run(
            ["git", "grep", "-n", "self._close_onboarding_agents_for_profile,",
             "--", "gateway/platforms/zet_agent.py"],
            capture_output=True, text=True).stdout.splitlines()
        assert len(out) == 1, f"调用点数量变了({len(out)})—— 逐个重判,⛔ 别只改数字"

        body = pathlib.Path("gateway/platforms/zet_agent.py").read_text().splitlines()
        lineno = int(out[0].split(":", 2)[1])
        # ⚠️ ``timeout=`` 落在调用点**之后**两行 ⇒ 窗口必须双向
        window = "\n".join(body[max(0, lineno - 12):lineno + 4])
        assert "asyncio.wait_for(" in window and "_ONBOARDING_CLOSE_TIMEOUT_SECONDS" in window, (
            "onboarding 关闭仍无期限 ⇒ unload/reload 永久挂住,"
            f"后续请求一直被 profile barrier 挡着。窗口:\n{window}"
        )
