"""事件循环上不许有同步等待 —— ⭐ **有上限也不行**。

🔴 **为什么另起一道门,而不是扩 ``test_no_unbounded_lifecycle_wait.py``。**

那道门漏掉了 ``gateway/model_readability.verify_artifact_readable`` 里的
``future.result(timeout=5)``,实查出**两处独立**的原因:

1. 它按 **函数名动词**(``cleanup/drain/shutdown/teardown/reload/…``)筛函数。
   ``verify_artifact_readable`` / ``_build_media_placeholder`` **一个动词都不命中**
   ⇒ 门**根本不看它们**。⭐ 「语义 ≠ 函数名」——同一个错今晚第四次。
2. 它的判据轴是「**无界**等待」。而那一行 **有** ``timeout=`` ——
   ⭐ **缺陷根本不是「无界」,是「在事件循环上等」**。5 秒的有界等待照样让
   所有会话停 5 秒,N 个附件就是 N×5 秒。**轴选错了,再怎么扩作用域也抓不住。**

⇒ 本门换轴:**任何**同步等待(有无上限都算),只要在事件循环上就必须登记。

## 作用域分格声明(⛔ 不许读成「全部收口」)

* **A 格(闭集,本门强制)**:同步等待原语**词法上直接出现在 ``async def`` 体内**。
  这一格是**可判定的** —— 在协程体里就必然在事件循环上。全仓 1047 个生产 ``.py``
  逐个 AST 扫描,⛔ 不按目录、⛔ 不按函数名收窄。未登记即红。
* **B 格(⚠️ 开集,明说)**:同步等待在 **sync def** 里,是否上循环**取决于调用方**。
  实测:按名字做传递闭包会得到 **13,651 个可达函数名 / 2,356 个候选**
  —— 名字级调用图太糙,**做不成闭集**。⛔ 本门不声称覆盖这一格。
* **C 格(闭集,针对已知会做阻塞 I/O 的同步入口)**:这类入口的**调用点全集**
  可枚举 ⇒ 逐个判定它们不在协程里裸调。今天这条漏网之鱼正落在这一格,
  ⭐ 把它变成结构性强制,而不是修一处了事。
"""

import ast
import pathlib
import subprocess

import pytest

_SINKS = ("result", "wait", "join", "acquire")

#: **已逐个实查**:这些 ``.result()`` 只在任务到达终态之后才调用 ⇒ **不阻塞**。
#: 判据是代码里那道守卫(``while not worker.done()`` / ``if done:`` /
#: ``if not task.done(): continue``),⛔ 不是「它们看起来是同一套写法」。
_LOOP_SYNC_WAIT_OK: dict[tuple[str, str, str], str] = {
    ("gateway/platforms/api_server.py", "_run_in_executor_with_completion_barrier",
     "worker.result()"): "守卫 while not worker.done() —— 实查过",
    ("gateway/platforms/zet_agent.py", "_to_thread_with_completion_barrier",
     "worker.result()"): "守卫 while not worker.done() —— 实查过",
    ("hermes_cli/web_server.py", "_to_thread_with_completion_barrier",
     "worker.result()"): "守卫 while not worker.done() —— 实查过",
    ("gateway/run.py", "_run_in_executor_with_context_completion_barrier",
     "worker.result()"): "守卫 while not worker.done() —— 实查过",
    ("gateway/run.py", "_reap_cancelled_cleanup",
     "worker.result()"): "守卫 if worker.done() —— 实查过",
    ("gateway/run.py", "_track_profile_adapter_operation",
     "task.result()"): "守卫 if not task.done(): continue —— 实查过",
    ("gateway/run.py", "_run_agent_inner",
     "_executor_task.result()"): "守卫 if done:(asyncio.wait 轮询)—— 实查过",
    ("tools/mcp_tool.py", "_run_blocking_cleanup_with_completion_barrier",
     "worker.result()"): "守卫 while not worker.done() —— 实查过",
    ("tools/mcp_tool.py", "_await_cleanup_until_complete",
     "worker.result()"): "守卫 if worker.done() —— 实查过",
}

#: ⚠️ **存量,本轮未逐条追** ⇒ 标已知风险,⛔ **不是判定为安全**。
#: 它们在本 PR 之前就存在,本 PR 也没让它们更可达。
#: ⭐ 这张表的价值不在「已经清干净了」,而在**新增的一律拦得住**。
_LOOP_SYNC_WAIT_KNOWN: dict[tuple[str, str, str], str] = {
    ("gateway/run.py", "start_gateway", "time.sleep(0.5)"): "启动期,尚未进入服务循环",
    ("gateway/run.py", "start_gateway", "time.sleep(0.25)"): "启动期,尚未进入服务循环",
    ("gateway/run.py", "start_gateway",
     "_planned_stop_watcher_thread.join(timeout=2)"): "关停期,有 2s 上限",
    ("hermes_cli/web_server.py", "get_action_status", "proc.wait(timeout=1)"): "存量,未逐条追",
    ("plugins/platforms/discord/adapter.py", "_wait_for_ready_or_bot_exit",
     "ready_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("plugins/platforms/photon/adapter.py", "_stop_sidecar", "proc.wait(timeout=3.0)"): "存量,未逐条追",
    ("plugins/platforms/photon/adapter.py", "_stop_sidecar", "proc.wait(timeout=2.0)"): "存量,未逐条追",
    ("plugins/platforms/telegram/adapter.py", "_start_polling_resilient",
     "progress.wait()"): "存量,未逐条追 —— **无上限**",
    ("plugins/platforms/telegram/adapter.py", "_start_polling_resilient",
     "strict_error_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_lifecycle_event",
     "self._shutdown_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_lifecycle_event",
     "self._reconnect_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_reconnect_or_shutdown",
     "self._shutdown_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_reconnect_or_shutdown",
     "self._reconnect_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_lazy_reconnect",
     "self._shutdown_event.wait()"): "存量,未逐条追 —— **无上限**",
    ("tools/mcp_tool.py", "_wait_for_lazy_reconnect",
     "self._reconnect_event.wait()"): "存量,未逐条追 —— **无上限**",
}


def _sink_of(node):
    if not isinstance(node, ast.Call):
        return None
    attr = getattr(node.func, "attr", None)
    if attr in _SINKS:
        if attr == "join" and node.args:
            return None                      # ``sep.join(xs)`` 不是阻塞
        return attr
    if attr == "sleep" and isinstance(getattr(node.func, "value", None), ast.Name) \
            and node.func.value.id == "time":
        return "time.sleep"
    return None


def _production_files():
    out = subprocess.run(["git", "ls-files", "*.py"],
                         capture_output=True, text=True).stdout.split()
    return [f for f in out
            if not f.startswith(("tests/", "optional-skills/", "scripts/", "nix/"))]


def _scan_async_bodies():
    """A 格:同步等待**词法上**落在 ``async def`` 体内。"""
    found = []
    for path in _production_files():
        try:
            src = pathlib.Path(path).read_text(encoding="utf-8")
            tree = ast.parse(src)
        except Exception:
            continue
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            nested = {id(x)
                      for sub in ast.walk(fn)
                      if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                      and sub is not fn
                      for x in ast.walk(sub)}
            # ⚠️ **量具修正**:只排除 ``await X.wait()`` 是不够的 ——
            # ``await asyncio.wait_for(proc.wait(), t)`` / ``await gather(..., proc.wait())``
            # 里的 ``proc.wait()`` 是**协程**,同样不阻塞循环。
            # ⇒ 排除**整棵 await 子树**,⛔ 不是只排除它的直接 value。
            awaited = {id(x) for a in ast.walk(fn) if isinstance(a, ast.Await)
                       for x in ast.walk(a)}
            for n in ast.walk(fn):
                if id(n) in nested or id(n) in awaited:
                    continue
                if _sink_of(n) is None:
                    continue
                seg = " ".join((ast.get_source_segment(src, n) or "").split())
                found.append((path, fn.name, seg, n.lineno))
    return found


class TestClosedSetA:
    def test_every_sync_wait_inside_a_coroutine_is_registered(self):
        found = _scan_async_bodies()
        assert found, "探针一个都没扫到 ⇒ 量具坏了(⛔ 空集不是合格)"
        registered = {**_LOOP_SYNC_WAIT_OK, **_LOOP_SYNC_WAIT_KNOWN}
        unregistered = [f for f in found if (f[0], f[1], f[2]) not in registered]
        assert not unregistered, (
            "以下同步等待直接落在协程体内 —— 事件循环会被整段占住,**有上限也不行**。\n"
            "修掉它,或在 _LOOP_SYNC_WAIT_OK 里登记理由:\n"
            + "\n".join(f"  {p}:{ln} {fn}  ->  {seg[:90]}" for p, fn, seg, ln in unregistered)
        )

    def test_the_probe_scope_is_the_whole_production_tree(self):
        """⛔ 作用域不许按目录/函数名收窄 —— 收窄一次 = 给范围外发免检。"""
        files = _production_files()
        assert len(files) > 900, f"作用域塌了,只剩 {len(files)} 个文件"
        assert any(f.startswith("plugins/") for f in files)
        assert any(f.startswith("agent/") for f in files)
        assert any(f.startswith("tools/") for f in files)

    def test_the_probe_counts_bounded_waits_too(self):
        """⭐ **轴的自证**:带 ``timeout=`` 的等待也必须被认出来 ——
        上一道门正是因为只认「无界」才漏掉了 5 秒的那一行。"""
        sample = (
            "import asyncio\n"
            "async def f(fut):\n"
            "    return fut.result(timeout=5)\n"
        )
        tree = ast.parse(sample)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
        hits = [n for n in ast.walk(fn) if _sink_of(n)]
        assert hits, "带 timeout 的 .result() 没被认出来 ⇒ 轴又选回「无界」了"

    def test_the_probe_ignores_awaited_waits(self):
        """🔴 **必须保持不变**:``await`` 掉的等待不是同步等待,⛔ 不许误报。"""
        sample = (
            "import asyncio\n"
            "async def f(ev):\n"
            "    await ev.wait()\n"
        )
        tree = ast.parse(sample)
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
        awaited = {id(a.value) for a in ast.walk(fn) if isinstance(a, ast.Await)}
        hits = [n for n in ast.walk(fn) if _sink_of(n) and id(n) not in awaited]
        assert not hits, "await 掉的等待被误判成同步等待"

    def test_str_join_is_not_a_blocking_sink(self):
        """🔴 **必须保持不变**:``sep.join(xs)`` ⛔ 不许被当成 Thread.join。"""
        tree = ast.parse("async def f(xs):\n    return ', '.join(xs)\n")
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef))
        assert not [n for n in ast.walk(fn) if _sink_of(n)]


class TestClosedSetC:
    """C 格:已知会做阻塞 I/O 的同步入口,其**调用点全集**不许落在协程里。"""

    #: 入口名 -> 它的异步替身。⭐ 新增这类入口时必须同时在这里登记。
    _BLOCKING_ENTRYPOINTS = {
        "verify_artifact_readable": "verify_artifact_readable_async",
    }

    def test_no_coroutine_calls_a_blocking_entrypoint_synchronously(self):
        offenders = []
        for path in _production_files():
            try:
                src = pathlib.Path(path).read_text(encoding="utf-8")
                tree = ast.parse(src)
            except Exception:
                continue

            def walk(node, owner):
                for ch in ast.iter_child_nodes(node):
                    if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        walk(ch, (ch.name, isinstance(ch, ast.AsyncFunctionDef)))
                        continue
                    if isinstance(ch, ast.Call):
                        name = getattr(ch.func, "id", None) or getattr(ch.func, "attr", None)
                        if name in self._BLOCKING_ENTRYPOINTS and owner and owner[1]:
                            offenders.append((path, ch.lineno, owner[0], name))
                    walk(ch, owner)

            walk(tree, None)

        assert not offenders, (
            "阻塞入口被协程**同步**调用 —— 挂载点一卡,所有会话一起停:\n"
            + "\n".join(
                f"  {p}:{ln} 在 async def {fn} 里裸调 {nm}()"
                f" —— 应改用 {self._BLOCKING_ENTRYPOINTS[nm]}()"
                for p, ln, fn, nm in offenders)
        )

    def test_every_registered_entrypoint_actually_has_its_async_twin(self):
        """⛔ 登记表不许指向不存在的替身(否则这道门的建议是空头支票)。"""
        import gateway.model_readability as mr

        for sync_name, async_name in self._BLOCKING_ENTRYPOINTS.items():
            assert hasattr(mr, sync_name), sync_name
            assert hasattr(mr, async_name), (
                f"{async_name} 不存在 ⇒ 门让人改成一个没有的函数")

    def test_the_probe_can_see_a_planted_violation(self):
        """⭐ 阳性对照:人造一个违规,判据必须命中(⛔ 不许恒绿)。"""
        sample = ("async def f(p):\n"
                  "    return verify_artifact_readable(p)\n")
        tree = ast.parse(sample)
        hits = []

        def walk(node, owner):
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    walk(ch, (ch.name, isinstance(ch, ast.AsyncFunctionDef)))
                    continue
                if isinstance(ch, ast.Call):
                    nm = getattr(ch.func, "id", None) or getattr(ch.func, "attr", None)
                    if nm in self._BLOCKING_ENTRYPOINTS and owner and owner[1]:
                        hits.append(nm)
                walk(ch, owner)

        walk(tree, None)
        assert hits == ["verify_artifact_readable"], "判据认不出人造违规 ⇒ 恒绿"


class TestOpenSetBIsDeclared:
    def test_the_module_docstring_declares_the_open_half(self):
        """⭐ ⛔ 扩不成闭集就**明说是开集** —— ⛔ 不许再报「全部收口」。"""
        doc = pathlib.Path(__file__).read_text(encoding="utf-8")
        assert "开集" in doc and "做不成闭集" in doc, (
            "开集声明被删了 ⇒ 下一个人会把这道门读成「事件循环已经干净了」"
        )


class TestTheRegistryItselfStaysHonest:
    def test_no_stale_registry_entries(self):
        """⛔ 登记表不许留僵尸:代码改掉了、条目还在 = 给未来的新代码发免检。"""
        live = {(p, fn, seg) for p, fn, seg, _ in _scan_async_bodies()}
        stale = [k for k in {**_LOOP_SYNC_WAIT_OK, **_LOOP_SYNC_WAIT_KNOWN} if k not in live]
        assert not stale, "登记表里的僵尸条目(对应代码已不存在):\n" + "\n".join(map(str, stale))

    def test_known_is_not_read_as_safe(self):
        """⭐ ``_LOOP_SYNC_WAIT_KNOWN`` 是**待办清单**,⛔ 不是结论。"""
        assert all("存量" in v or "启动期" in v or "关停期" in v
                   for v in _LOOP_SYNC_WAIT_KNOWN.values())
