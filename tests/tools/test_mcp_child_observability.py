"""MCP 子进程观测：**看不见 ≠ 没有**（RH 复审 P2-2）。

``_snapshot_child_pids()`` 原先只返回一个集合，于是空集合同时表示：
  (a) 观测成功、确实没有子进程 —— 安全
  (b) ``/proc`` 与 psutil **都不可用** —— 有没有子进程根本不知道

调用方拿不到区别，只能都放行。(b) 下真起了子进程也无人认领，
shutdown 回收不到 ⇒ **孤儿进程泄漏**。

⭐ 工具的沉默不是证据。⇒ 观测能力与观测结果必须分开返回，不可用时 fail closed。

⚠️ 本文件只驱动**判定逻辑**，⛔ 不拉起真实 stdio —— RH 指出旧的逆改测试
会 hang 15 秒而不是在目标断言上失败，那种门给不出可用信号。
"""
from __future__ import annotations

import builtins

import pytest

from tools import mcp_tool


def test_observation_reports_success_when_it_works():
    observable, pids = mcp_tool._observe_child_pids()
    assert observable is True, "本机 /proc 或 psutil 至少有一个可用，却报观测失败"
    assert isinstance(pids, set)


def test_observation_reports_failure_when_both_sources_are_down(monkeypatch):
    """两条观测途径都断掉 ⇒ 必须显式报「不可观测」，⛔ 不是返回空集合。"""
    real_open = builtins.open

    def _no_proc(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise OSError(13, "Permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr(builtins, "open", _no_proc)

    real_import = builtins.__import__

    def _no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_psutil)

    observable, pids = mcp_tool._observe_child_pids()
    assert observable is False, (
        "两条观测途径都断了，却报观测成功 —— 空集合会被当成「确实没有子进程」")
    assert pids == set()


def test_legacy_shell_still_returns_just_the_set():
    """⛔ 不许弄坏原来对的：仍有调用点只要集合。"""
    assert isinstance(mcp_tool._snapshot_child_pids(), set)


def test_mocked_symbol_matches_the_symbol_production_calls():
    """🔴 换掉一个被 mock 的符号名 = 悄悄关掉所有 mock。

    实际发生过（本轮）：我把生产调用从 ``_snapshot_child_pids`` 换成新函数，
    全仓 5 处 ``patch(...)`` **全部失效且不报错** —— 测试照跑，只是跑去做真实
    进程枚举，同样三个文件从 34 秒变成 10 分钟不收敛。

    ⭐ 所以门要钉的不是「必须用某个具体名字」（上一版就是这么写的，
    结果它把当时那个**错误的决定**钉成了契约、反过来阻止正确的修复），
    而是**「测试 patch 的符号」与「生产真正调用的符号」必须一致**。
    """
    import ast
    import inspect
    from pathlib import Path

    src_file = Path(inspect.getfile(mcp_tool))
    tree = ast.parse(src_file.read_text(encoding="utf-8"))

    target = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_run_stdio":
            target = node
    assert target is not None, "calibration: 找不到 _run_stdio"

    called = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
        for n in ast.walk(target) if isinstance(n, ast.Call)
    }

    # 测试里 patch 了哪些 tools.mcp_tool 的子进程观测符号
    tests_dir = Path(__file__).resolve().parent
    patched: set[str] = set()
    for f in tests_dir.glob("test_mcp*.py"):
        text = f.read_text(encoding="utf-8")
        for sym in ("_observe_child_pids", "_snapshot_child_pids"):
            if f'patch("tools.mcp_tool.{sym}"' in text:
                patched.add(sym)

    assert patched, "calibration: 没有任何测试 patch 子进程观测 —— 门失去意义"
    orphaned = patched - called
    assert not orphaned, (
        f"这些符号被测试 patch，但生产 _run_stdio 根本不调用它们：{orphaned}。"
        f"⇒ 那些 mock 已经静默失效，测试会跑去做真实进程枚举。"
        f"（生产实际调用：{sorted(n for n in called if 'child_pid' in (n or ''))}）")


def test_observation_is_atomic():
    """🔴 能力与结果必须来自**同一次**观测。

    上一版拆成「先问能不能观测、再单独枚举」两次独立调用（为的是保住旧 mock
    符号）。能力探测只验证 ``psutil`` 能 import，而真正枚举时
    ``Process.children()` 仍可能抛错被吞成空集 ⇒ 出现
    ``available=True, pids=set()`` 这种自相矛盾的组合，不可回收的子进程照样
    被放行。⭐ 两个判据来自同一次系统调用序列才谈得上一致。
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getfile(mcp_tool)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_child_pid_observation_available" not in names, (
        "又出现了独立的能力探测函数 —— 它与实际枚举可以矛盾")

    # 行为侧：psutil 能 import 但枚举抛错 ⇒ 必须报不可观测
    import builtins

    real_open = builtins.open

    def _no_proc(path, *a, **k):
        if str(path).startswith("/proc/"):
            raise OSError(13, "denied")
        return real_open(path, *a, **k)

    import psutil

    real_children = psutil.Process.children
    try:
        builtins.open = _no_proc
        psutil.Process.children = lambda self, *a, **k: (_ for _ in ()).throw(
            RuntimeError("enumeration failed"))
        observable, pids = mcp_tool._observe_child_pids()
    finally:
        builtins.open = real_open
        psutil.Process.children = real_children

    assert observable is False, (
        "psutil 能 import 但枚举失败，却报观测成功 —— "
        "空集合会被当成「确实没有子进程」而放行不可回收的子进程")
    assert pids == set()


class _FakeAsyncCM:
    """最小 async context manager，⛔ 不起任何子进程（照抄
    ``tests/tools/test_mcp_stdio_init_timeout.py`` 的 fake transport 模式）。"""

    def __init__(self, value):
        self._value = value

    async def __aenter__(self):
        return self._value

    async def __aexit__(self, *_exc):
        return False


class _TrivialSession:
    async def initialize(self):
        return None

    async def list_tools(self):
        return type("R", (), {"tools": []})()


def _drive_run_stdio(monkeypatch, *, observable: bool):
    """驱动**真实** ``_run_stdio``，只把观测结果替换掉。

    🔴 为什么必须这样：上一版我写了个 ``admit()`` **复制品**放在测试里，
    对着复制品断言 —— 生产那边把判断改成 ``if False and not (...)``，
    门照样 7 passed（RH 复审第三轮实测）。
    ⭐ **对着自己抄的一份逻辑做断言，等于没有测生产。**

    ⇒ 照抄仓内 ``test_mcp_stdio_init_timeout.py``：fake transport 驱动真实
    ``_run_stdio``，全程 hermetic（无子进程、无网络）。
    """
    import asyncio

    monkeypatch.setattr(
        mcp_tool, "stdio_client", lambda *a, **k: _FakeAsyncCM((object(), object())))
    monkeypatch.setattr(
        mcp_tool, "ClientSession", lambda *a, **k: _FakeAsyncCM(_TrivialSession()))
    monkeypatch.setattr(mcp_tool, "_resolve_stdio_command", lambda c, e: (c, e))
    monkeypatch.setattr(mcp_tool, "_write_stderr_log_header", lambda *a, **k: None)
    monkeypatch.setattr(mcp_tool, "_get_mcp_stderr_log", lambda: None)
    monkeypatch.setattr("tools.osv_check.check_package_for_malware", lambda *a, **k: None)
    monkeypatch.setattr(mcp_tool, "_observe_child_pids", lambda: (observable, set()))

    server = mcp_tool.MCPServerTask("obs-guard")
    config = {"command": "fake-mcp", "args": [], "connect_timeout": 2.0}
    return asyncio.run(asyncio.wait_for(server._run_stdio(config), timeout=10.0))


def test_run_stdio_refuses_to_start_when_observation_is_unavailable(monkeypatch):
    """🔴 驱动**真实** ``_run_stdio``：不可观测 ⇒ 必须拒绝启动。

    否则子进程起来了却没人认领，shutdown 回收不到 ⇒ 孤儿泄漏。
    """
    # 🔴 ⛔ 只接受「因不可观测而拒绝」这一种结果。
    # 我第一版写的是 ``pytest.raises((RuntimeError, asyncio.TimeoutError))``
    # —— 把「正确拒绝」和「后续握手超时」算成同一种通过。于是 RH 那个
    # ``if False and not (...)`` 的攻击下，判断被跳过、流程往下走到超时，
    # 门照样绿（实测 8 passed）。
    # ⭐ **一条断言只钉一个性质**；把两种不同结果并进一个 `raises` 元组，
    #   等于给其中一种发免检。
    with pytest.raises(RuntimeError) as exc:
        _drive_run_stdio(monkeypatch, observable=False)
    assert "observation is unavailable" in str(exc.value), (
        f"抛了 RuntimeError，但不是因为不可观测:{exc.value}")


def test_run_stdio_proceeds_when_observation_works(monkeypatch):
    """⛔ 不许弄坏原来对的：观测成功且确实没有新子进程 ⇒ 必须放行。

    ⭐ 只测「拒绝」那一侧，判据可以简单地永远拒绝来"通过" ——
    那会把所有 MCP server 都挡掉，比原缺陷严重得多。
    """
    import asyncio

    try:
        _drive_run_stdio(monkeypatch, observable=True)
    except RuntimeError as exc:
        assert "observation is unavailable" not in str(exc), (
            f"观测成功却仍被 fail-closed 拦住 —— 所有 MCP server 都会起不来:{exc}")
    except asyncio.TimeoutError:
        # ⚠️ 这一档是**刻意**放过的：后续握手阶段的超时与本门要钉的性质无关。
        # ⛔ 但上面那条「拒绝」用例不许这么写 —— 那里超时会掩盖判断被跳过。
        pass


def test_production_admission_path_consults_observability():
    """早期信号：生产判定处应当用上 ``_observe_child_pids`` 的第一个返回值。

    🔴 ⛔ **这条不是闭集，⛔ 不许当保障。** 上一版 docstring 写「闭集门」是
    **虚假声明**：把生产判断改成 ``if False and not (...)`` 后本门仍通过
    （AST 不管可达性，RH 复审第四轮实测）。

    ⭐ 承重的是同文件的
    ``test_run_stdio_refuses_to_start_when_observation_is_unavailable`` ——
    它用 fake transport 驱动**真实** ``_run_stdio``，同一攻击下会红。
    本门只作为早期提醒：有人删掉那个 raise 时先在这里现形。
    """
    import ast
    import inspect
    from pathlib import Path

    src = Path(inspect.getfile(mcp_tool)).read_text(encoding="utf-8")
    tree = ast.parse(src)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        if "_obs_before" not in test_src and "_observable" not in test_src:
            continue
        if any(isinstance(n, ast.Raise) for n in ast.walk(node)):
            found = True
    assert found, (
        "找不到「观测不可用就 raise」的判定 —— fail closed 没有接线")
