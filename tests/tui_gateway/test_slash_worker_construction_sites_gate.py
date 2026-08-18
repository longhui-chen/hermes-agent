"""闭集门:`_SlashWorker` 的 profile 作用域,不许由构造点各自记得。

判据(⛔ 不是"某一处改对了没有",那是按位置的开集):

  ① `_SlashWorker` 的**每一个**构造点都必须传 `profile_home`;
  ② profile 作用域的应用必须发生在 `_SlashWorker.__init__` **自己**体内。

②是承重的那一半:把"判定"和"应用"放同一个地方,新增第 N 个构造点的人**不可能忘**。
①是它的补:`__init__` 做得再对,构造点不给 profile_home 也只能回落到 ambient/gateway 的
HERMES_HOME —— 那是个**全 agent 共用**的目录,而各 profile 绑的是不同的真人身份;
lark-cli 又会自己把目录建出来(设备上 `/root/.lark-cli/cache/` 就是这么来的),
于是"共享凭据库"会被静默地创造出来,没有任何报错。

⚠️ 参照系:出厂 0.0.57 部署字节里 `_SlashWorker(` 有 **4 个**构造点(1647/3141/5227/14810),
其中只有 1 个在 `set_hermes_home_override` 作用域内;origin/main 已把其余三处收敛掉,
今天只剩 `_restart_slash_worker` 这一个。⇒ 本门存在的意义正是**防止那 4 个的形态回潮**。

⚠️ 这是**源码形状**判据(开集):写 `cls = _SlashWorker; cls(...)` 能绕过去。
接受它的理由是运行时那几条契约测试(tests/tools/test_profile_scoped_home.py)已经按
**实际交给 Popen 的 env** 判定;本门只负责"新构造点不许悄悄少传一个参数"这一半。
"""

from __future__ import annotations

import ast
import pathlib

SERVER = pathlib.Path(__file__).resolve().parents[2] / "tui_gateway" / "server.py"


def _tree():
    return ast.parse(SERVER.read_text(encoding="utf-8"))


def _worker_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            if name == "_SlashWorker":
                yield node


def test_every_construction_site_passes_the_session_profile():
    tree = _tree()
    offenders = []
    checked = 0
    for call in _worker_calls(tree):
        checked += 1
        if not any(kw.arg == "profile_home" for kw in call.keywords):
            offenders.append(call.lineno)

    # ⛔ 非空转自检:一个构造点都没扫到 ⇒ 类被改名或路径写错,这门就恒绿了 ——
    # 恒绿和恒红一样,都是没有信号。
    assert checked, "GATE_MUST_NOT_BE_VACUOUS: 一个 _SlashWorker 构造点都没扫到"
    assert not offenders, (
        "EVERY_WORKER_CONSTRUCTION_MUST_PASS_PROFILE_HOME: "
        + repr(offenders)
        + " 行的构造点没传 profile_home —— 它会回落到全 agent 共用的 HOME"
    )


def test_profile_scope_is_applied_inside_the_constructor():
    tree = _tree()
    init = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_SlashWorker":
            for body in node.body:
                if isinstance(body, ast.FunctionDef) and body.name == "__init__":
                    init = body
    assert init is not None, "GATE_MUST_NOT_BE_VACUOUS: 找不到 _SlashWorker.__init__"

    applied = [
        n.lineno
        for n in ast.walk(init)
        if isinstance(n, ast.Call)
        and (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
        == "apply_profile_scoped_env"
    ]
    assert applied, (
        "PROFILE_SCOPE_MUST_BE_APPLIED_IN_THE_CONSTRUCTOR: __init__ 里没有 "
        "apply_profile_scoped_env —— 一旦挪回某个构造点,其余构造点就静默失去作用域"
    )
