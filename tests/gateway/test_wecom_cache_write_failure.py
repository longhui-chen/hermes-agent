"""企微入站媒体落盘失败：必须降级成**消息级**失败，⛔ 不许击穿 transport。

现场：``cache_document_from_bytes`` / ``cache_image_from_bytes`` 的 ``OSError``
（磁盘满、只读挂载、权限不足）原先一路冒到 ``_on_message`` 再到监听循环 ⇒
**整条连接断开**；而这条消息的 ID 已经进了 dedup ⇒ 重连后也不会重放，
用户那条消息**永久消失**。一次磁盘故障 = 一次静默丢消息 + 一次掉线。

⭐ 门的作用域 = **四个写盘点的全集**（base64/url × image/document），
⛔ 不是「我测了 image 那一条」——四个点是同构的，只测一个等于给另外三个发免检。
"""
from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

import pytest

from plugins.platforms.wecom import adapter as wecom_adapter
from plugins.platforms.wecom.adapter import WeComAdapter

#: 会真正写盘的函数名 —— 闭集，新增写盘方式必须同步加进来
_CACHE_WRITERS = ("cache_image_from_bytes", "cache_document_from_bytes")

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _adapter():
    class _A(WeComAdapter):
        name = "wecom"

    return _A.__new__(_A)


def _boom(*_a, **_kw):
    raise OSError(28, "No space left on device")


# ───────────────── 四个写盘点全集：一个都不许击穿 ─────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,kind",
    [("base64", "image"), ("base64", "file"), ("url", "image"), ("url", "file")],
    ids=["base64+image", "base64+document", "url+image", "url+document"],
)
async def test_disk_failure_never_escapes_to_the_listener(
    monkeypatch, caplog, source, kind
):
    a = _adapter()
    for w in _CACHE_WRITERS:
        monkeypatch.setattr(wecom_adapter, w, _boom)

    if source == "base64":
        import base64 as _b64

        media = {"base64": _b64.b64encode(_PNG).decode()}
    else:
        async def _fake_dl(url, max_bytes=None):
            return _PNG, {"content-type": "image/png" if kind == "image" else "application/pdf"}

        monkeypatch.setattr(a, "_download_remote_bytes", _fake_dl)
        media = {"url": "https://wework.qpic.cn/media?x=1"}

    with caplog.at_level(logging.WARNING):
        # ⛔ 这里**不许**抛：抛出去就是断连 + 丢消息
        result = await a._cache_media(kind, media)

    assert result is None, "落盘失败却返回了成功结果"
    assert "cache_write_failed" in caplog.text, (
        f"落盘失败没有留下可区分的痕迹:{caplog.text!r}")


@pytest.mark.asyncio
async def test_whole_message_survives_a_disk_failure(monkeypatch, caplog):
    """整条消息级别：写盘炸了，``_extract_media`` 仍要正常返回空。

    ⭐ 只测 ``_cache_media`` 不够 —— 缺陷的**后果**发生在它的调用者那一层。
    """
    a = _adapter()
    for w in _CACHE_WRITERS:
        monkeypatch.setattr(wecom_adapter, w, _boom)

    async def _fake_dl(url, max_bytes=None):
        return _PNG, {"content-type": "image/png"}

    monkeypatch.setattr(a, "_download_remote_bytes", _fake_dl)

    with caplog.at_level(logging.WARNING):
        paths, types, _fails = await a._extract_media(
            {"msgtype": "image", "image": {"url": "https://wework.qpic.cn/x"}}
        )

    assert paths == [] and types == []
    assert "cache_write_failed" in caplog.text


@pytest.mark.asyncio
async def test_programming_errors_still_propagate(monkeypatch):
    """⛔ 别把守卫写成 ``except Exception``。

    只有环境故障（``OSError``）才降级成消息级失败；编程错误必须继续上抛，
    否则下一个真 bug 会被伪装成「用户发的图有问题」。
    """
    a = _adapter()

    def _bug(*_a, **_kw):
        raise TypeError("wrong arity — this is our bug, not the disk's")

    for w in _CACHE_WRITERS:
        monkeypatch.setattr(wecom_adapter, w, _bug)

    import base64 as _b64

    with pytest.raises(TypeError):
        await a._cache_media("file", {"base64": _b64.b64encode(_PNG).decode()})


# ───────────────── 闭集门：不许出现第五个裸写盘点 ─────────────────

def test_every_cache_write_goes_through_the_guard():
    """辅助门：拦住**模块内**绕过写法（直接调用 / 别名赋值）。

    🔴 ⛔ **这条不是闭集，⛔ 不许拿它当保障。** RH 三轮下来依次绕过了：
      v1 直接调用 → v2 模块内 alias → v3 ``from ... import X as _writer``。
    每被绕一次我就加一层静态分析，而每次都还有新形状 ——
    ⭐ **给 Python 名字做静态白名单是一场必输的开集追逐战。**

    ⇒ **承重的是上面那四条行为门**（``test_disk_failure_never_escapes_to_
    the_listener``，parametrize 覆盖 base64/url × image/document 全集）：
    它们直接 monkeypatch 写盘函数抛 ``OSError`` 并驱动真实
    ``_cache_media``，**不管你用什么写法调用它，OSError 逃逸就会红**。
    RH 自己实测 import-alias 攻击下行为门给出 ``4 failed``。

    本条只作为**早期信号**：新增写盘点时提醒作者套守卫。
    """
    src = Path(inspect.getfile(wecom_adapter)).read_text(encoding="utf-8")
    tree = ast.parse(src)

    guard = "_write_cache_or_report"

    # ⭐ 判据极简且闭集：合法用法是把写盘函数**当对象**交给守卫
    # （``self._write_cache_or_report(kind, media, cache_x_from_bytes, ...)``），
    # 那时它是 ``Call.args`` 里的 ``Name``，**不会**产生 ``Call.func == 写盘函数``
    # 的节点。⇒ 凡是出现「直接调用」形态，一律违规。
    #
    # 🔴 上一版判据是「这个函数名在文件里某处被正确传给过守卫吗」——
    # 那是**按名字**判而不是**按调用点**判：只要有一处合法用法，所有裸调用
    # 都拿到免检。逆改 E（把一个写盘点改回裸调用）当场证明它抓不住。
    # ⭐ 判据的作用域必须刚好等于缺陷的作用域：缺陷在**单个调用点**上。
    # ⭐ 判据收紧成「这些名字只允许出现在两个位置」：
    #   ① ``from ... import cache_x_from_bytes``（导入）
    #   ② 作为 ``_write_cache_or_report(...)`` 的**实参**
    # 其它任何位置一律违规 —— 包括**别名赋值**（``_w = cache_x_from_bytes``
    # 然后 ``_w(...)``）。
    # 🔴 上一版只查「直接调用」形态，于是 alias 一步就绕过去了
    #   （RH 复审第二轮实测）。⭐ 按「长什么形状」列清单永远是开集；
    #   换成「只有这两个位置合法」才是闭集。
    # ⚠️ 实测补充：alias 攻击下运行时那四条门**抓住了**（OSError 逃逸），
    #   所以缺陷本身不会漏网；但这条门作为**结构保障**确实被绕过了。
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                allowed.add(id(alias))
        if isinstance(node, ast.Call):
            fname = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fname == guard:
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        allowed.add(id(arg))

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _CACHE_WRITERS:
            if id(node) not in allowed:
                offenders.append(f"{node.id} @line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in _CACHE_WRITERS:
            offenders.append(f"{node.attr} @line {node.lineno}")

    assert not offenders, (
        f"这些写盘点是直接调用，没走 {guard}，磁盘故障会击穿到监听循环：\n  "
        + "\n  ".join(offenders)
    )


def test_the_guard_itself_is_still_wired():
    """calibration：上一条门只查「没有裸调用」，⛔ 它对「守卫被整个删掉」免疫。

    ⭐ 一条断言只钉一个性质 —— 这条负责钉「守卫确实还在、且确实被用着」。
    """
    src = Path(inspect.getfile(wecom_adapter)).read_text(encoding="utf-8")
    tree = ast.parse(src)
    guard = "_write_cache_or_report"

    assert any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == guard
        for n in ast.walk(tree)
    ), f"{guard} 被删了"

    used = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == guard
    ]
    assert len(used) == 4, (
        f"守卫的调用点应为 4 个（base64/url × image/document），实为 {len(used)} 个"
        " —— 少了说明有写盘点绕开了它，多了说明出现了未登记的写盘路径")
