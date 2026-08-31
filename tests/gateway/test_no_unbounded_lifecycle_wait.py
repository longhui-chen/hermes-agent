"""无界等待:从**逐个修实例**换成**全仓强制登记**。

## 🔴 为什么必须换判据

同一个模式已经漏了**三轮**:

| 轮 | 实例 | |
|---|---|---|
| 2 | `_await_adapter_cleanup_strict` 取消后无限等(**首次**进入) | ✅ 修 |
| 4 | `_cleanup_unpublished_adapter` 重试入口再次无界(**后续**进入) | ✅ 修 |
| 5 | `_drain_profile_adapter_operations` → `shield(worker)`(**drain** 入口) | ✅ 修 |
| 5 | `tools/mcp_tool.py::_cancel_and_wait` 裸 `Event.wait()` | ✅ 修 |

⭐ **每修一个入口就冒下一个 —— 因为「按实例枚举」是开集。**
机器人自己的措辞点破了它:第三条「**绕过了**已有的硬期限」——
说明前两次修的是**具体入口**,而不是**这一类**。

⇒ 换成:**生命周期路径上任何阻塞 `await`,要么有限期限,要么在
`_UNBOUNDED_OK` / `_KNOWN_UNBOUNDED` 里逐条登记。** 新增未登记的 ⇒ **直接红**。

## ⚠️ 闭到哪、开在哪(⛔ 不许读成「扫干净了」)

* ✅ **闭**:**全仓**(⛔ 不是模块清单)—— 本门首次运行命中 **8 个文件**,
  按模块清单必漏 `mcp_tool` / `lsp` / `telegram` / `matrix` / `dingtalk` / `feishu`。
* ✅ **闭**:登记表僵尸条目会被抓红,⛔ 不会悄悄给新代码发免检。
* 🔴 **开(其一)**:作用域靠**函数名**里的生命周期动词
  (`cleanup|drain|shutdown|teardown|reload|disconnect|cancel|stop|unload|close|reap|retire`)。
  **名字不带这些动词的生命周期函数扫不到。**
* 🔴 **开(其二)**:只认 `shield()` / `Event.wait()` / `asyncio.wait()` /
  `gather()` / `.join()` / `.acquire()` **加上两个已知的仓内 barrier helper**;
  **其它自定义阻塞原语仍扫不到**。
  ⚠️ 这一维**又被撞过一次**:`zet_agent._handle_profile_unload` 用的是
  `_to_thread_with_completion_barrier(...)`,函数名在动词表里、但形态不在探针里
  ⇒ 漏了。⭐ **开集不是写下来就没事了 —— 它会真的漏。** 已把两个 helper 加进探针。
* ⚠️ **本门自己栽过一次,记下来**:第一版只扫 `ast.Await`,于是**同步**阻塞
  (`threading.Event().wait()` 在 `def` 里)整类漏掉 —— 而那正是触发本门的
  `tools/mcp_tool.py::_cancel_and_wait`。**逆改它时本门居然是绿的。**
  ⭐ 「门抓不住自己的触发案例」= 出生即空转。已补同步分支。
* ⭐ **两维都是开集,明说。** ⛔ 本门绿 ≠ 全仓没有无界等待。
"""
from __future__ import annotations

import ast
import pathlib
import re
import subprocess

import pytest

_VERBS = ("cleanup", "drain", "shutdown", "teardown", "reload", "disconnect",
          "cancel", "stop", "unload", "close", "reap", "retire")
_BLOCKING = re.compile(
    r"shield\(|\.wait\(\)|asyncio\.wait\(|\.join\(\)|\.acquire\(\)|gather\(|"
    # ⭐ 仓内自定义的阻塞原语 —— 它的 docstring 明写「取消后仍等 worker 结束」,
    #   语义上就是一个 barrier。⛔ 漏掉它 = 漏掉 zet_agent 的 onboarding 关闭。
    r"_to_thread_with_completion_barrier\(|_run_blocking_cleanup_with_completion_barrier\("
)

#: ⭐ **有界的**:登记「为什么它其实跑不飞」。键 = ``(路径, 函数名, 片段)``。
_UNBOUNDED_OK: dict[tuple[str, str, str], str] = {
    ("agent/lsp/client.py", "_cleanup_process", "await proc.wait()"):
        "紧跟在 kill()/terminate() 之后等**自己 spawn 的子进程**收尸;"
        "内核保证 SIGKILL 后进程会退出 ⇒ 有界。",
    ("agent/lsp/manager.py", "_reap_idle_once", "gather("):
        "等的是本函数刚创建的 `_shutdown_client_for_retry` 协程,"
        "它们自身各带超时 ⇒ 上界由内层给出。",
    ("agent/lsp/manager.py", "_shutdown_async_owned", "gather("):
        "同上:等的是自建的 `_shutdown_client_for_retry` / 已 cancel 的 reaper,"
        "内层有界。",
    ("agent/lsp/manager.py", "_cleanup_client_with_barrier", "await asyncio.shield(task)"):
        "**调用侧有界**:唯一调用点是 ``self._loop.run(..., timeout=1.0)``(本批新增,"
        "同一次 diff 里就带着上限)⇒ AST 在函数体内看不到那层包装,但实际有界。",
    ("gateway/run.py", "wait_for_shutdown", "self._shutdown_event.wait()"):
        "**这就是等停信号的地方**,无界是它的语义 —— 有界反而是缺陷。",
    ("agent/lsp/client.py", "_cleanup_process", "proc.wait()"):
        "同步版:同样紧跟 kill()/terminate(),内核保证 SIGKILL 后进程退出 ⇒ 有界。",
    ("tools/bounded_media_exec.py", "_reap", "proc.wait()"):
        "本轮自己写的收尸函数 —— 外层已用 asyncio.wait_for(_REAP_TIMEOUT) 包住,"
        "AST 看不到那层包装(它在调用点)⇒ 实际有界。",
    ("tools/mcp_tool.py", "_wait_for_reconnect_or_shutdown", "self._reconnect_event.wait()"):
        "**这就是等待重连/停止信号的地方**,无界是它的语义;"
        "退出由 _shutdown_event 驱动 ⇒ 有界由调用方的 shutdown 保证。",
    ("tools/mcp_tool.py", "_wait_for_reconnect_or_shutdown", "self._shutdown_event.wait()"):
        "同上。",
    ("gateway/run.py", "_await_adapter_cleanup_strict", "await asyncio.shield(worker)"):
        "只在 ``timeout <= 0`` 那一支 —— **调用方显式说了『不设期限』**。"
        "⚠️ 它由 ``HERMES_GATEWAY_ADAPTER_DISCONNECT_TIMEOUT`` 决定,配成 0 就是"
        "自愿放弃上限;⛔ 不在这里替调用方改主意。",
}

#: 🔴 **已知无界、本轮未修**:⛔ 这不是「判定为安全」,是「判定为已知风险」。
_KNOWN_UNBOUNDED: dict[tuple[str, str, str], str] = {
    ("plugins/platforms/dingtalk/adapter.py", "disconnect", "gather("):
        "等已 cancel 的后台 task;task 若吞掉取消则挂住。渠道 disconnect 面,"
        "本轮不扩范围。",
    ("plugins/platforms/feishu/adapter.py", "_cancel_pending_tasks", "gather("):
        "同上。",
    ("plugins/platforms/matrix/adapter.py", "disconnect", "gather("):
        "同上。",
    ("plugins/platforms/telegram/adapter.py", "_cancel_pending_delivery_tasks", "gather("):
        "同上。",
    ("plugins/platforms/telegram/adapter.py", "disconnect", "gather("):
        "同上。",
    ("tools/computer_use/doctor.py", "_close_mcp", "proc.wait()"):
        "诊断工具路径,⛔ 未逐条追;不在本轮 finding 面上,标已知风险。",
    ("tools/tts_tool.py", "close", "self._player.join()"):
        "等播放线程收尾;线程若卡住则挂住。标已知风险,⛔ 不是安全。",
    # ── zet_agent 的 profile unload / stale-import 清理族 ─────────────────
    # ⭐ 这一整族是**探针补上自定义 barrier 之后才现形的** —— 说明「开集」不是
    #   写下来就没事,它真的漏了一整族。⛔ 逐条登记,不合并成一条。
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "_close_owned_session_db"):
        "关本 profile 自己的 session DB。SQLite close 卡住的概率低但非零;"
        "⛔ 未逐条追,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "cleanup_managed_profile_environments"):
        "清理托管环境(容器/沙箱);⛔ 未逐条追,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "drv.invalidate_barrier_callbacks_for_home"):
        "纯内存回调表失效,实际瞬时;⛔ 但未加上限,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "process_registry.kill_all"):
        "杀本 profile 的子进程;kill 后等待理论有界,但未显式设上限。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "process_registry.purge_profile_state"):
        "清进程登记表(内存 + 落盘);⛔ 未逐条追,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "purge_profile_approval_state"):
        "清审批状态;⛔ 未逐条追,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_handle_profile_unload", "retire_managed_terminal_profile"):
        "退役托管终端;⛔ 未逐条追,标已知风险。",
    ("gateway/platforms/zet_agent.py", "_cleanup_stale_runtime_imports_once", "self._ensure_session_db"):
        "stale-import 清理族,同上。",
    ("gateway/platforms/zet_agent.py", "_cleanup_stale_runtime_imports_once", "self._open_profile_session_db"):
        "同上。",
    ("gateway/platforms/zet_agent.py", "_cleanup_stale_runtime_imports_once", "session_db.cleanup_stale_runtime_imports"):
        "同上。",
    ("gateway/platforms/zet_agent.py", "_cleanup_stale_runtime_imports_once", "session_db.close"):
        "同上。",
    ("tools/mcp_tool.py", "_shutdown_owned", "_kill_orphaned_mcp_children"):
        "杀本 server 的孤儿子进程;SIGKILL 后有界,但 barrier 本身未设上限。",
    ("gateway/run.py", "_cleanup_when_done", "await asyncio.shield(future)"):
        "等的是 agent turn 的 future;turn 自身有超时,但**这一层没有独立上界**。"
        "本轮 finding 未点到,标已知风险,⛔ 不是安全。",
    ("gateway/run.py", "_defer_agent_cleanup_until_future_done", "await asyncio.shield(future)"):
        "同上(同一形态的另一个入口)。",
    ("tools/mcp_tool.py", "_await_cleanup_until_complete", "await asyncio.shield(worker)"):
        "MCP cleanup barrier。⭐ 已改为**有界**:首次被取消起算 ``_MCP_CANCEL_REAP_SECONDS``,"
        "窗口内仍吞取消(barrier 语义),耗尽后让取消向外传播、残留交孤儿回收。"
        "这一行 ``shield`` 本身没有 timeout 参数,所以探针仍会命中 —— "
        "⛔ 不是漏登记,是判据维度只看单行。",
    ("tools/mcp_tool.py", "_run_blocking_cleanup_with_completion_barrier", "await asyncio.shield(worker)"):
        "同上。",
    ("tools/mcp_tool.py", "_shutdown", "gather("):
        "等一批 ``server.shutdown()``;每个 server 内部**有** 10s+5s 的分级超时,"
        "但若 server 数量很大,总时长仍无上界。标已知风险。",
    ("tools/mcp_tool.py", "_shutdown_owned", "gather(*self._pending_refresh_tasks"):
        "等已 cancel 的 refresh task;吞掉取消则挂住。标已知风险。",
    ("tools/mcp_tool.py", "shutdown_mcp_profile", "gather("):
        "同 ``_shutdown``。",
    ("tools/mcp_tool.py", "shutdown_mcp_servers", "gather("):
        "同 ``_shutdown``。",
}


def _files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], capture_output=True, text=True).stdout.split()
    return [f for f in out
            if not f.startswith(("tests/", "optional-skills/", "scripts/", "nix/"))
            and pathlib.Path(f).exists()]


def _unbounded_waits():
    hits = []
    for f in _files():
        try:
            src = pathlib.Path(f).read_text()
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(v in fn.name.lower() for v in _VERBS):
                continue
            for n in ast.walk(fn):
                # ⭐ 两类都要:``await <blocking>`` **和** 同步 ``<event>.wait()``。
                # ⛔ 只扫 Await 会整类漏掉 threading.Event 那一族。
                if isinstance(n, ast.Await):
                    node = n
                elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                        and n.func.attr in {"wait", "join", "acquire"}:
                    node = n
                else:
                    continue
                seg = " ".join((ast.get_source_segment(src, node) or "").split())
                if "wait_for(" in seg or "timeout=" in seg:
                    continue
                if _BLOCKING.search(seg):
                    hits.append((f, fn.name, seg))
    return hits


def _registered(hit) -> bool:
    f, fname, seg = hit
    for table in (_UNBOUNDED_OK, _KNOWN_UNBOUNDED):
        for (p, n, frag) in table:
            if p == f and n == fname and frag in seg:
                return True
    return False


class TestEveryLifecycleWaitIsAccountedFor:
    def test_no_unregistered_unbounded_wait(self):
        """⭐ 全仓闭集:新增一个没登记的阻塞点 ⇒ 红。"""
        offenders = sorted(
            f"{f}  [{fn}]  {seg[:100]}"
            for (f, fn, seg) in _unbounded_waits() if not _registered((f, fn, seg))
        )
        assert not offenders, (
            "生命周期路径上出现未登记的无界等待 —— 它会让上游所有 deadline 失效:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_registry_has_no_stale_entries(self):
        """⛔ 僵尸条目会悄悄给新代码发免检。"""
        live = _unbounded_waits()
        stale = [
            f"{p}::{n}  {frag!r}"
            for table in (_UNBOUNDED_OK, _KNOWN_UNBOUNDED)
            for (p, n, frag) in table
            if not any(p == f and n == fn and frag in seg for (f, fn, seg) in live)
        ]
        assert not stale, f"登记表里的条目已不存在:{stale}"

    def test_the_four_fixed_entry_points_are_no_longer_flagged(self):
        """阳性对照:本轮修掉的四个入口不该再出现在命中里。"""
        flagged = {(f, fn) for (f, fn, _s) in _unbounded_waits()}
        # ⚠️ ``_await_adapter_cleanup_strict`` 仍会命中 —— 那是它 ``timeout <= 0``
        # 的**刻意无界**分支(已登记)。本对照只查那些**应当完全消失**的。
        # ⚠️ 真正被 drain 路径调用、且被本轮修掉的是
        # ``_await_existing_adapter_tasks``(``_drain_profile_adapter_operations``
        # 以 cancel=False 把共享 task 传给它)—— ⭐ 按**内容**定位,⛔ 不按名字猜。
        assert not any(n == "_await_existing_adapter_tasks" for (_f, n) in flagged), (
            "_await_existing_adapter_tasks 仍被判为无界 —— 修复没生效或被回退"
        )
        # ⭐ 反向:硬期限必须真的在(⛔ 不是被登记条目掩盖成绿)
        import inspect

        from gateway import run as grun

        src = inspect.getsource(grun.GatewayRunner._await_existing_adapter_tasks)
        assert "wait_for(" in src and "drain_deadline" in src, "drain 的硬期限不见了"

    def test_the_probe_actually_finds_things(self):
        """⛔ 出生即空转的门比没有门更坏:探针必须真能命中。"""
        hits = _unbounded_waits()
        assert len(hits) >= 8, f"全仓只命中 {len(hits)} 处 —— 探针本身可疑"
        # ⭐ 必须同时命中**异步**与**同步**两类,⛔ 否则又是只扫了一半
        assert any("await" in seg for (_f, _n, seg) in hits), "异步分支没命中"

    def test_the_gate_catches_synchronous_blocking_too(self):
        """🔴 本门第一版**抓不住自己的触发案例** —— 记下来并钉死。

        ``tools/mcp_tool.py::_cancel_and_wait`` 是**同步**嵌套函数,
        ``lifecycle_completed.wait()`` 不是 ``await`` ⇒ 只扫 ``ast.Await`` 的
        第一版对它逆改**仍然是绿的**。⭐ 门抓不住自己的触发案例 = 出生即空转。
        """
        import ast as _ast

        sample = (
            "def _cancel_and_wait():\n"
            "    lifecycle_completed.wait()\n"
        )
        tree = _ast.parse(sample)
        fn = tree.body[0]
        found = [
            n for n in _ast.walk(fn)
            if isinstance(n, _ast.Call) and getattr(n.func, "attr", "") == "wait"
        ]
        assert found, "探针连最小样本里的同步 wait() 都认不出"
        seg = " ".join((_ast.get_source_segment(sample, found[0]) or "").split())
        assert _BLOCKING.search(seg) and "timeout=" not in seg

    def test_known_unbounded_is_not_a_safety_verdict(self):
        """⭐ ``_KNOWN_UNBOUNDED`` 必须非空且每条都有理由 —— 它是待办,⛔ 不是安全。"""
        assert _KNOWN_UNBOUNDED, "已知风险表被清空了 —— 那些点没修,⛔ 不许装作没有"
        assert all(v.strip() for v in _KNOWN_UNBOUNDED.values()), "有条目没写理由"

    def test_the_open_edges_are_declared_in_the_docstring(self):
        """⭐ 两维开集必须写在 docstring 里,⛔ 不许沉默。"""
        import sys

        doc = sys.modules[__name__].__doc__ or ""
        assert "🔴 **开(其一)**" in doc and "🔴 **开(其二)**" in doc, (
            "开集边界的声明被删了 —— 下一个人会把本门的绿读成「全仓没有无界等待」"
        )


class TestThisPrDidNotMakeAnyKnownUnboundedWaitMoreReachable:
    """⭐ 用户 08-16 的问题:那些「已知无界」里,有没有**本 PR 让它变可达**的?

    判据 = 本批 diff 有没有**新增通往它的路径**(净增调用点 / 把 fast-fail 换成
    join),⛔ 不是「文件动没动」。
    实查结论:**只有一条** —— ``LSPService._shutdown_async``。H⑤ 把「第二个并发
    调用者自己新建 worker ⇒ 一进 owner 就撞 already-in-progress ⇒ **快速失败**」
    换成了「加入在飞的那一趟」⇒ owner 卡住时**原本快速失败的调用现在会挂住**。
    ⇒ 它不再是存量问题,已在本轮加有界 join。
    """

    def test_the_join_introduced_by_the_task_reuse_is_bounded(self):
        import inspect

        from agent.lsp.manager import LSPService

        src = inspect.getsource(LSPService._shutdown_async)
        assert "wait_for(" in src and "_SHUTDOWN_JOIN_TIMEOUT_SECONDS" in src, (
            "H⑤ 引入的 join 又变回无界 ⇒ fast-fail 被换成了永久挂起"
        )

    def test_the_owner_is_still_not_cancelled_by_the_second_waiter(self):
        """🔴 ⛔ 不许塌缩到另一端:超时**不得**取消 owner、⛔ 不得清 _shutdown_task。"""
        import inspect

        from agent.lsp.manager import LSPService

        src = inspect.getsource(LSPService._shutdown_async)
        i = src.index("except asyncio.TimeoutError:")
        j = src.index("except asyncio.CancelledError", i)
        window = src[i:j]
        assert "worker.cancel()" not in window, "第二个 waiter 超时把 owner 取消了"
        assert "_shutdown_task = None" not in window, "超时清了复用引用 ⇒ 下次又新建撞闩"

    def test_shutdown_async_is_no_longer_in_the_known_unbounded_table(self):
        """⭐ 修好了就要从「已知风险」里移走,⛔ 不许留着当免检。"""
        assert not any(
            n == "_shutdown_async" and p.endswith("lsp/manager.py")
            for (p, n, _f) in _KNOWN_UNBOUNDED
        )
