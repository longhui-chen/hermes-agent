"""出身声明:从**枚举形状**换成**可强制的登记制**。

## 🔴 为什么必须换判据

同一个模式我已经宣布「全部收口」过一次,然后**第四个兄弟**又冒出来:

| 轮次 | 我以为收口了 | 实际漏的 |
|---|---|---|
| 上上轮 | `status=="failed"` | 空 `output`、无终止帧 |
| 上一轮 | 三个「全部收口」 | 🔴 `codex_runtime.py` 的 SSE 提前断流 |

⭐ **不是找得不认真 —— 枚举方式本身是开集**:靠 grep / 命名 / 已知形状去找
「哪些 raise 没带出身」,**下一个长得不一样的就漏**。

⇒ 换成 **全仓强制登记**:``agent/`` 下**任何**一条 raise,只要它的文案在讲
Responses 协议异常,就必须**要么带 `declare_upstream_origin()`,要么在
`_NOT_UPSTREAM` 里登记并写明理由**。新增一条没登记的 ⇒ **本门直接红**。

## ⚠️ 这道门闭到哪、开在哪(⛔ 不许读成「扫干净了」)

* **闭**:关键词命中的 raise —— **全仓 `agent/` 范围**,⛔ 不是某个模块清单。
  新模块里出现同类 raise 一样会被抓到(这正是上一轮漏掉第四个兄弟的原因)。
* **闭**:登记表本身 —— 僵尸条目会被 `test_the_registry_has_no_stale_entries` 抓红,
  ⛔ 不会悄悄给新代码发免检。
* 🔴 **开**:文案里**一个关键词都不带**的 raise 扫不到。
  ⭐ **这一维是开集,明说。** ⛔ 不许把本门的绿读成「这个模式已经全仓扫净」。
  (要真闭合,得把整层出口收口到一个包装函数 —— 那是重构,不在本 PR 范围。)
"""
from __future__ import annotations

import ast
import pathlib
import subprocess

import pytest

#: 判定「这条 raise 在讲 Responses 协议异常」的关键词。⛔ 闭集,加词要连理由一起加。
_KEYWORDS = (
    "Responses API",
    "Responses stream",
    "terminal response",
    "final response",
    "no output items",
)

#: ⭐ **强制登记表**:关键词命中、但**不该**带出身声明的 raise,逐条写明理由。
#: 键 = ``(相对路径, 异常文本片段)``。⛔ 不许用行号(会漂)。
_NOT_UPSTREAM: dict[tuple[str, str], str] = {
    (
        "agent/auxiliary_client.py",
        "Codex auxiliary Responses stream interrupted",
    ): (
        "**本地主动中断**(用户按了停止 / interrupt_check 触发),⛔ 不是上游故障。"
        "盖出身戳会让它变成可重试的上游失败 —— 用户明确要求停下,重试是错的。"
    ),
    (
        "agent/codex_responses_adapter.py",
        "Codex Responses stream flag is only allowed in fallback streaming requests",
    ): (
        "**我们自己的调用契约被违反**(内部参数组合非法),⛔ 与 provider 无关。"
        "它就该判成内部错误:同样的输入每次都会这样,重试不可能好。"
    ),
}


def _agent_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "agent/*.py", "agent/**/*.py"],
        capture_output=True, text=True,
    ).stdout.split()
    return [f for f in out if pathlib.Path(f).exists()]


def _keyword_raises(path: str):
    try:
        src = pathlib.Path(path).read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return []
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        seg = " ".join((ast.get_source_segment(src, node) or "").split())
        if any(k in seg for k in _KEYWORDS):
            hits.append((node.lineno, seg))
    return hits


def _declared(stmt: str) -> bool:
    return "declare_upstream_origin" in stmt


class TestEveryResponsesProtocolRaiseIsAccountedFor:
    def test_no_unregistered_raise_anywhere_under_agent(self):
        """⭐ 全仓闭集:新增一条没登记的 ⇒ 红。⛔ 不依赖「这次找全了」。"""
        offenders = []
        for f in _agent_files():
            for lineno, stmt in _keyword_raises(f):
                if _declared(stmt):
                    continue
                if any(k[0] == f and k[1] in stmt for k in _NOT_UPSTREAM):
                    continue
                offenders.append(f"{f}:{lineno}  {stmt[:120]}")
        assert not offenders, (
            "Responses 协议异常的 raise 没带出身声明,也没登记理由 —— "
            "它会被判成我们自己的 bug:既不重试也不 fallback:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_four_known_siblings_are_all_declared(self):
        """阳性对照:四个已知兄弟必须都在,⛔ 否则上一条的 0 可能是扫空了。"""
        declared = {
            f"{f}:{lineno}"
            for f in _agent_files()
            for lineno, stmt in _keyword_raises(f)
            if _declared(stmt)
        }
        files = {k.split(":")[0] for k in declared}
        assert "agent/codex_responses_adapter.py" in files
        assert "agent/codex_runtime.py" in files
        assert "agent/auxiliary_client.py" in files
        # ⚠️ 是 3 不是 4:``status=="failed"`` 那条抛的是
        # ``declare_upstream_origin(RuntimeError(error_msg))`` —— 消息是**变量**,
        # 源码文本里不含关键词,所以关键词探针看不见它(它确实带了声明)。
        # ⭐ 这正好是本门开集边界的一个**具体实例**,写在这里当活证据。
        assert len(declared) >= 3, f"只找到 {len(declared)} 条带声明的:{declared}"

    def test_the_registry_has_no_stale_entries(self):
        """⛔ 登记表不许留僵尸条目(它会悄悄给新代码发免检)。"""
        stale = [
            f"{path}  {frag!r}"
            for (path, frag) in _NOT_UPSTREAM
            if not any(frag in s for _l, s in _keyword_raises(path))
        ]
        assert not stale, f"登记表里的条目已不存在:{stale}"

    def test_the_gate_is_not_vacuous(self):
        """⛔ 出生即空转的门比没有门更坏 —— 拿真实形状喂它。"""
        assert not _declared('raise RuntimeError("Responses API blew up")')
        assert _declared('raise declare_upstream_origin(RuntimeError("x"))')
        # 关键词探针必须真能命中(⛔ 不是恒空)
        total = sum(len(_keyword_raises(f)) for f in _agent_files())
        assert total >= 4, f"关键词探针全仓只命中 {total} 条 —— 探针本身可疑"

    def test_the_open_edge_is_declared_in_the_docstring(self):
        """⭐ 本门有一维是开集,**必须写在 docstring 里**,⛔ 不许沉默。"""
        import sys

        doc = sys.modules[__name__].__doc__ or ""
        assert "🔴 **开**" in doc and "开集,明说" in doc, (
            "开集边界的声明被删了 —— 下一个人会把本门的绿读成「全仓扫净」"
        )
