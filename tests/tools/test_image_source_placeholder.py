"""占位标记 ⛔ 不许被当成媒体路径去打开。

## 现场(2026-08-16 云机实测,用户正在测时撞到两次)

```
ERROR tools.vision_tools: Error analyzing image: media file not found: '[图片]'
  File ".../tools/vision_tools.py", line 1173, in vision_analyze_tool
    resolved = await resolve_image_source(image_url, ResolveContext(...))
ValueError: media file not found: '[图片]'
```

``image_url`` 是**工具参数** —— 是**模型自己**读到正文里的 ``[图片]``,把它当
图片引用传了进来。``[图片]`` 是适配器为「无说明文字的媒体」插入的占位标记。

⇒ 模型收到「文件找不到」,于是告诉用户「图片分析失败」;而真相是**这一轮
没有可用的图片**。两条提示指向完全不同的动作:一个让人以为是系统故障、
可以重试,另一个才会让人重发图片。

## ⛔ 判据不能是「像不像路径」

``resolve_image_source`` 里那条注释记着这件事已经被证否过:

    # Everything else is a filesystem path — including bare relative names
    # like "pic.png" (accepted on main; a path-shape gate here regressed them).

⭐ 判据换成**闭集、且我们说了算**的那一面:**这个标记是不是我们自己产出的**。
复用 ``agent.conversation_loop._is_trusted_image_media_placeholder``
(它的 docstring 明说「adapter-generated placeholders for captionless media」),
⛔ 不在 image_source 里新开第三份清单 —— 仓里已有的两份就已经互相不一致了
(``conversation_loop`` 3 条 / ``yuanbao`` 8 条)。
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

# ⚠️ ⛔ 不在模块顶层绑定这些符号。
#
# tests/tools/ 目录里有别的用例会 reload ``tools.image_source``。顶层 import
# 会把**当时那一份**类对象钉在本模块的全局里,reload 之后
# ``pytest.raises(SourceNotFound)`` 拿到的是**旧身份**,于是即便被测代码抛出的
# 正是 SourceNotFound,也匹配不上、直接穿透成失败。
#
# 实测:本文件单跑 18 条全绿,放进整个 tests/tools 目录跑就红 14 条 ——
# ⭐ 单跑绿 ⇒ ⛔ 不代表这道门是好的。每次进函数再取,身份永远是当前那一份。
def _mod():
    import tools.image_source as m
    return m


def _resolve(src):
    m = _mod()
    return asyncio.run(m.resolve_image_source(src, m.ResolveContext(task_id="t")))


# ───────── ① 缺陷本体 ─────────

@pytest.mark.parametrize("placeholder", [
    "[图片]",
    "[image]",
    "[image attachment]",
    "[Image]",                      # 判定是 casefold 的
    "  [图片]  ",                    # 前后空白
    "[user attached image: photo.jpg]",
])
def test_adapter_placeholders_are_not_media_references(placeholder):
    with pytest.raises(_mod().NotAMediaReference) as caught:
        _resolve(placeholder)
    msg = str(caught.value)
    # 🔴 提示必须可行动:告诉模型「别重试」+「让用户重发」
    assert "not an image reference" in msg, msg
    assert "Do not retry" in msg, msg
    assert "resend" in msg, msg


def test_the_old_message_pointed_the_model_at_the_wrong_action():
    """🔒 回归锁:⛔ 不许再退回「文件找不到」那种说法。

    ⚠️ 这里用**正面断言**(异常类型 + 文案要点),⛔ 不写
    ``"media file not found" not in msg`` —— 反面断言只要把字面量拼错就恒真。
    """
    with pytest.raises(_mod().NotAMediaReference):
        _resolve("[图片]")


# ───────── ② 🔒 必须保持不变:真的媒体引用仍要能用 ─────────

@pytest.mark.parametrize("src", [
    "pic.png",                       # ⭐ 裸相对名 —— 曾被形状门误伤过的那一类
    "photo.jpeg",
    "/tmp/does-not-exist.png",
    "./sub/dir/img.webp",
    "[not-a-known-placeholder]",     # 方括号但不是我们产出的标记
    "[图片].png",                     # 以标记开头但确实是个文件名
])
def test_real_paths_are_still_treated_as_paths(src):
    """⛔ 别为了挡占位文本把正常路径也挡了。

    这些都应该继续走**文件路径**那条分支 —— 文件不存在时报
    ``SourceNotFound``(而不是 ``NotAMediaReference``),证明判据没有外扩。
    """
    with pytest.raises(_mod().SourceNotFound):
        _resolve(src)


def test_unsupported_scheme_still_reports_scheme_not_placeholder():
    from tools.image_source import UnsupportedScheme

    with pytest.raises(UnsupportedScheme):
        _resolve("s3://bucket/key.png")


def test_empty_input_still_reports_required():
    with pytest.raises(_mod().SourceNotFound) as caught:
        _resolve("   ")
    assert "required" in str(caught.value)


# ───────── ③ 判据来源:⛔ 没有第三份清单 ─────────


def test_the_predicate_is_borrowed_not_reinvented():
    """⭐ 「照抄」是可验证的断言:这里必须**引用**适配器那份判定。

    ⛔ 若有人在 image_source 里新写一份占位符集合,仓里就会有三份互相漂移的
    清单 —— 而现有两份已经不一致了。
    """
    src = Path(inspect.getfile(_mod().resolve_image_source)).read_text()
    tree = ast.parse(src)
    imported = {
        n.names[0].name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module == "agent.conversation_loop"
    }
    assert "_is_trusted_image_media_placeholder" in imported, (
        "没有复用适配器那份判定 —— 是不是又新开了一份清单?"
    )
    # ⛔ 本模块里不许出现占位符字面量(那就是第三份清单的样子)
    literals = [
        n.value for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and n.value.strip().casefold() in {"[图片]", "[image]", "[image attachment]"}
    ]
    assert not literals, f"image_source 里出现了占位符字面量:{literals}"


def test_predicate_is_shared_with_the_adapter_side():
    """阳性对照:借来的那个判定确实认得这些标记(⛔ 不是空转)。"""
    from tools.image_source import _adapter_placeholder_predicate

    pred = _adapter_placeholder_predicate()
    assert pred is not None, "判定取不到 —— 本轮所有断言都会因为别的原因通过"
    assert pred("[图片]") and pred("[image]")
    assert not pred("pic.png") and not pred("/tmp/a.png")


# ───────── ④ 兄弟调用点:模型给的字符串只走这一个收口 ─────────


def test_every_model_supplied_source_goes_through_the_one_resolver():
    """⚠️ **开集门,明说**:只覆盖「生产代码里直接 import 解析器」这一种形状。

    作用是防漂移 —— 修在收口处的前提,是三个调用点确实都走收口。
    """
    root = Path(inspect.getfile(_mod().resolve_image_source)).parents[1]
    callers = []
    for rel in ["tools/vision_tools.py", "tools/flux3_video_tool.py"]:
        text = (root / rel).read_text()
        callers.append((rel, text.count("resolve_image_source(")))
    for rel, n in callers:
        assert n > 0, f"{rel} 不再经过收口解析器了?"
    assert sum(n for _, n in callers) >= 3, callers
