"""ZET-2111 的**生产接线**（RH 复审 P2-1 的假绿）。

现场：ZET-2111 那 10 条测试**直接调 adapter**，把 gateway 这一侧整个跳过 ——
RH 实测「切断真实 Gateway 接线后仍 10 passed」。⇒ 那是一条**已知假绿**：
gateway 若不再调 ``note_long_running_turn``，长任务跑完就没有完成提醒，
而所有门依然全绿。⭐ 假绿门比没有门更坏，它冒充保护。

本文件驱动的是 ``_deliver_heartbeat_and_note`` —— 心跳发送与记账的**共同
决策点**。它原先内联在 ``start_gateway`` → ``_run_agent_inner`` →
``_notify_long_running`` 的三层闭包里，外部驱动不了；提取时签名/行为逐字未改。

⚠️ 最要紧的一格是 **edit 路径也要记账**：第一次心跳走 send（首发），
之后每次走 edit（复用同一条消息）。只挂 send 那半 ⇒ **第二次之后全部漏记**，
表现是「短的长任务有提醒、真正跑很久的反而没有」—— 最反直觉的那种。
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.run import _deliver_heartbeat_and_note


def _source():
    return SimpleNamespace(chat_id="oc_1", platform="feishu", thread_id=None,
                           user_id="u1")


def _adapter(*, edit=None, send=None):
    noted: list = []
    a = SimpleNamespace(
        edit_message=AsyncMock(return_value=edit),
        send=AsyncMock(return_value=send),
        note_long_running_turn=lambda src: noted.append(src),
    )
    a._noted = noted
    return a


def _ok(mid="m1"):
    return SimpleNamespace(success=True, message_id=mid)


def _fail():
    return SimpleNamespace(success=False, message_id=None, error="boom")


@pytest.mark.asyncio
async def test_first_heartbeat_goes_through_send_and_is_noted():
    """首发（还没有 heartbeat_msg_id）⇒ 走 send，成功后必须记账。"""
    a = _adapter(send=_ok("m1"))
    src = _source()

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=src, text="⏳", heartbeat_msg_id=None, status_metadata=None)

    assert a.edit_message.await_count == 0, "没有已存在的心跳消息却去 edit"
    assert a.send.await_count == 1
    assert res.success is True
    assert fresh == "m1", "首发拿到的 message_id 必须回给调用方去登记 cleanup"
    assert a._noted == [src], "首次心跳没有记账 ⇒ 长任务跑完不会有完成提醒"


@pytest.mark.asyncio
async def test_edit_path_is_also_noted():
    """🔴 最关键的一格：复用已有心跳消息（edit）成功时**同样**要记账。

    只挂 send 那半 ⇒ 第二次之后的心跳全部漏记 ⇒ 跑得越久越收不到提醒。
    """
    a = _adapter(edit=_ok("m1"))
    src = _source()

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=src, text="⏳", heartbeat_msg_id="m1", status_metadata=None)

    assert a.edit_message.await_count == 1
    assert a.send.await_count == 0, "edit 已经成功了还去 send —— 会多发一条"
    assert res.success is True
    assert fresh is None, "edit 复用旧消息，⛔ 不该产生新的 cleanup id"
    assert a._noted == [src], (
        "edit 路径没有记账 —— 第二次之后的心跳全部漏记，"
        "真正跑很久的长任务反而收不到完成提醒")


@pytest.mark.asyncio
async def test_edit_failure_falls_back_to_send_and_still_notes():
    """edit 失败 → 回落 send，仍要记账（两条路径共同的成功判定）。"""
    a = _adapter(edit=_fail(), send=_ok("m2"))
    src = _source()

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=src, text="⏳", heartbeat_msg_id="m1", status_metadata=None)

    assert a.edit_message.await_count == 1 and a.send.await_count == 1
    assert fresh == "m2"
    assert a._noted == [src]


@pytest.mark.asyncio
async def test_edit_raising_falls_back_to_send():
    """edit **抛异常**（不是返回失败）同样要回落，⛔ 不许把心跳整个丢掉。

    ⭐ 孪生枚举：返回失败与抛异常是两条路径。
    """
    a = _adapter(send=_ok("m2"))
    a.edit_message = AsyncMock(side_effect=RuntimeError("network"))
    src = _source()

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=src, text="⏳", heartbeat_msg_id="m1", status_metadata=None)

    assert a.send.await_count == 1
    assert res.success is True
    assert a._noted == [src]


@pytest.mark.asyncio
async def test_nothing_is_noted_when_the_heartbeat_never_landed():
    """⛔ 不许弄坏原来对的：心跳压根没发出去就**不能**记账。

    记了就会在收口时补一个 ✅，而用户从来没见过任何「仍在处理」提示 ——
    凭空冒出一个对勾。
    """
    a = _adapter(edit=_fail(), send=_fail())
    src = _source()

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=src, text="⏳", heartbeat_msg_id="m1", status_metadata=None)

    assert res.success is False
    assert fresh is None
    assert a._noted == [], "心跳一次都没送达却记了账 ⇒ 会凭空补一个 ✅"


@pytest.mark.asyncio
async def test_note_hook_failure_never_breaks_the_heartbeat():
    """⛔ 记账只是记账：hook 抛异常不许把已经送达的心跳改判成失败。"""
    a = _adapter(send=_ok("m1"))

    def _boom(_src):
        raise RuntimeError("adapter bug")

    a.note_long_running_turn = _boom

    res, fresh = await _deliver_heartbeat_and_note(
        adapter=a, source=_source(), text="⏳", heartbeat_msg_id=None,
        status_metadata=None)

    assert res.success is True and fresh == "m1"


# ───────────── 接线门本身：⛔ 生产路径不许绕开这个决策 ─────────────

def _inner_code(root_code, name):
    """在编译产物里递归找嵌套函数的 code object。"""
    import types

    for const in root_code.co_consts:
        if isinstance(const, types.CodeType):
            if const.co_name == name:
                return const
            found = _inner_code(const, name)
            if found is not None:
                return found
    return None


def test_production_heartbeat_path_uses_the_shared_decision():
    """🔴 这条是「假绿」的正解：钉住 gateway 真的用了这个函数。

    ⚠️ **判据必须是字节码，⛔ 不能是 AST。** 上一版用 ``ast.walk`` 收集
    ``_notify_long_running`` 里的 Call —— **AST 不管可达性**，把生产调用塞进
    ``if False:`` 后新旧 22 条**全绿**（RH 复审第二轮实测）。
    ⭐ 我在 discovery 门刚修好这个问题，转头在这道新门里又犯了一次。

    CPython 在编译期就会消除 ``if False:`` 分支，所以字节码里根本不会出现
    那个名字 —— 这挡住了常量死代码这一类绕过。

    ⚠️ **如实标注开集边界**：字节码判据挡不住 ``if some_runtime_flag:``
    这种运行时条件分支。要做成闭集，需要把 ``_notify_long_running`` 整体提成
    可驱动的模块级函数（它现在是 ``start_gateway`` → ``_run_agent_inner``
    三层闭包内的 async 闭包，依赖十几个闭包变量）。**本轮没做**，
    ⛔ 不宣称这道门是闭集。
    """
    import dis
    import inspect
    from pathlib import Path

    src_path = Path(inspect.getfile(__import__("gateway.run", fromlist=["run"])))
    root = compile(src_path.read_text(encoding="utf-8"), str(src_path), "exec")

    target = _inner_code(root, "_notify_long_running")
    assert target is not None, "calibration: 编译产物里找不到 _notify_long_running"

    # 🔴 只收**名字加载**指令，⛔ 不收 ``LOAD_CONST``。
    # 上一版收了所有字符串型 argval —— 于是删掉真实调用、只留一行
    # ``_name = "_deliver_heartbeat_and_note"`` 字符串常量，门照样绿
    # （RH 复审第三轮实测：生产接线完全断开，两道门仍 2 passed）。
    # ⭐ 「名字出现在编译产物里」和「这个名字被加载并调用」是两回事。
    reachable = {
        ins.argval for ins in dis.get_instructions(target)
        if isinstance(ins.argval, str) and ins.opname.startswith("LOAD_")
        and ins.opname != "LOAD_CONST"
    }
    assert "_deliver_heartbeat_and_note" in reachable, (
        "gateway 的心跳路径没有【可达地】调用共享决策 —— "
        "记账接线可能已经断了（或被塞进了不可达分支），"
        "而直接调 adapter 的那些测试照样全绿")
    assert "note_long_running_turn" not in reachable, (
        "又在 _notify_long_running 里内联记账了 —— 两条记账路径必然漂移，"
        "这正是本缺陷的形状")


def test_bytecode_criterion_ignores_mere_string_constants():
    """🔴 字符串常量 ≠ 调用。

    RH 实测：删掉真实调用、只留 ``_name = "_deliver_heartbeat_and_note"``，
    上一版判据照样绿。⭐ 「名字出现在编译产物里」和「这个名字被加载并调用」
    是两回事 —— 收 ``LOAD_CONST`` 就是把前者当成了后者。
    """
    import dis

    code = compile(
        "def outer():\n"
        "    def inner():\n"
        "        _name = \"_deliver_heartbeat_and_note\"\n"
        "        return _name\n",
        "<probe>", "exec")
    inner = _inner_code(code, "inner")
    names = {
        i.argval for i in dis.get_instructions(inner)
        if isinstance(i.argval, str) and i.opname.startswith("LOAD_")
        and i.opname != "LOAD_CONST"
    }
    assert "_deliver_heartbeat_and_note" not in names, (
        "判据把字符串常量当成了调用 —— 只要在源码里写一次这个名字就能骗过它")


def test_bytecode_criterion_actually_sees_through_dead_code():
    """calibration：证明字节码判据确实能识破 ``if False:``。

    ⭐ 没有这条，上面那道门换成 AST 也照样绿，我不会知道判据退化了。
    """
    import dis
    import types

    code = compile(
        "def outer():\n"
        "    def inner():\n"
        "        if False:\n"
        "            _deliver_heartbeat_and_note()\n"
        "        pass\n",
        "<probe>", "exec")
    inner = _inner_code(code, "inner")
    names = {i.argval for i in dis.get_instructions(inner) if isinstance(i.argval, str)}
    assert "_deliver_heartbeat_and_note" not in names, (
        "编译器没有消除 if False 分支 —— 字节码判据不成立，"
        "上面那道门等于没有")
