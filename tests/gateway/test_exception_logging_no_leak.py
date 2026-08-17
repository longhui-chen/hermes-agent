"""异常对象⛔不许直接进日志 —— 它们的 ``__str__`` 会带出 URL / 凭据 / 路径。

**病因（我自己命名的）**：「我在 wecom 修过同一形状，兄弟调用点没跟上。」
⇒ 那就把它当**模式**做孪生枚举，⛔ 不是修掉发现的那几处。

已有两个独立实证：
  · wecom 媒体链接（带鉴权参数）—— 通过 ``netloc`` 含 userinfo 泄漏
  · weixin 四个下载出口 —— 403 时 ``aiohttp.ClientResponseError.__str__``
    把 ``?token=…`` 整条写进 warning

哪些异常会漏，是**实测**出来的（``test_the_leaky_exception_types_still_leak``
把这份清单钉死，以后依赖升级导致某个类型不再泄漏时它会提醒）：

=================================  ============================================
``aiohttp.ClientResponseError``    ``url='https://u:pw@h/x?token=…'``
``aiohttp.InvalidURL``             整条 URL
``httpx`` 的 ``raise_for_status``  ``for url 'https://…?token=…'``
``OSError`` / ``FileNotFoundError``  ``: '/Users/x/.secret/token.json'``
=================================  ============================================
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from gateway.platforms.base import safe_exc

#: 扫描范围 —— 本 lane 负责的面。⛔ 这是**开集边界**：仓里其它文件不在本门
#: 覆盖内（它们归别的 lane）。⭐ 明说，⛔ 不冒充全仓闭集。
_SCOPE = [
    "gateway/platforms/weixin.py",
    "gateway/kanban_watchers.py",
    "plugins/platforms/wecom/adapter.py",
    "plugins/platforms/wecom/callback_adapter.py",
]

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SECRETS = ("TOP_SECRET", "u:pw", "sig=ABC", "/Users/", ".secret-config")
_URL = "https://u:pw@cdn.example:8443/media?token=TOP_SECRET&sig=ABC"


def _leaky_exceptions():
    """实际会泄漏的异常实例 —— ⛔ 现造，不用录制好的字符串。"""
    import aiohttp
    import httpx
    from yarl import URL

    ri = aiohttp.RequestInfo(URL(_URL), "GET", {}, URL(_URL))
    out = [
        aiohttp.ClientResponseError(ri, (), status=403, message="Forbidden"),
        aiohttp.InvalidURL(_URL),
    ]
    r = httpx.Request("GET", _URL)
    try:
        httpx.Response(403, request=r).raise_for_status()
    except Exception as e:  # noqa: BLE001
        out.append(e)
    try:
        open("/Users/nobody/.secret-config/token.json")
    except OSError as e:
        out.append(e)
    return out


# ───────── ① 判据自身：这些类型**确实**会泄漏（阳性对照） ─────────


def test_the_leaky_exception_types_still_leak():
    """⭐ 阳性对照：⛔ 不许出现「清洗器有效是因为异常本来就不带机密」。

    这条同时是**依赖升级的哨兵**：某天 aiohttp 不再把 URL 放进 ``__str__``，
    它会红，提醒我们重新确认清单，⛔ 而不是让下面那条门悄悄变成空转。
    """
    leaked = [type(e).__name__ for e in _leaky_exceptions()
              if any(s in str(e) for s in _SECRETS)]
    assert len(leaked) >= 4, (
        f"只有 {leaked} 会泄漏 —— 清单可能过时，下面的清洗门可能已在空转")


# ───────── ② 清洗器：机密不出，诊断力还在 ─────────


@pytest.mark.parametrize("idx", range(4))
def test_safe_exc_strips_every_secret(idx):
    out = safe_exc(_leaky_exceptions()[idx])
    for s in _SECRETS:
        assert s not in out, f"清洗后仍含 {s!r}:{out}"


def test_safe_exc_keeps_enough_to_diagnose():
    """⛔ 不许「安全但没用」—— 403 / Errno / 超时必须还能分得开。

    ⭐ 一刀切成 ``type(exc).__name__`` 是最安全的，也是最没用的:
    排查时「ClientResponseError」既可能是 403 也可能是 500。
    """
    outs = [safe_exc(e) for e in _leaky_exceptions()]
    assert any("403" in o for o in outs), f"状态码全丢了:{outs}"
    assert any("Errno 2" in o for o in outs), f"errno 全丢了:{outs}"
    assert all(type(e).__name__ in safe_exc(e) for e in _leaky_exceptions())
    # host 保留（定位得到是哪个 CDN），但 userinfo / query 去掉
    assert any("cdn.example" in o for o in outs), f"连 host 都没了:{outs}"


def test_safe_exc_leaves_harmless_messages_readable():
    """⛔ 不许弄坏原来对的:不含机密的消息应当基本原样。"""
    out = safe_exc(TimeoutError("read timed out after 30s"))
    assert "read timed out after 30s" in out and "TimeoutError" in out


# ───────── ③ 闭集门：作用域内⛔不许再有裸异常进日志 ─────────


def _bare_exception_log_sites(path: pathlib.Path):
    """返回「把 ``except ... as X`` 绑定的 X 直接传给 logger」的位置。

    ⭐ 判据来自 **AST 的 except 绑定**，⛔ 不是「名字看起来像不像异常」——
    后者是开集（``e`` / ``exc`` / ``err`` / ``batch_err`` … 列不完）。
    绑定集合由语法结构决定，在单个文件内是**闭集**。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound = {n.name for n in ast.walk(tree)
             if isinstance(n, ast.ExceptHandler) and n.name}
    hits = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        fn = n.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id == "logger"
                and fn.attr in ("debug", "info", "warning", "error",
                                "exception", "critical")):
            continue
        for a in n.args:
            if isinstance(a, ast.Name) and a.id in bound:
                hits.append(f"{path.name}:{a.lineno} 直接传了 {a.id}")
    return hits


@pytest.mark.parametrize("rel", _SCOPE)
def test_no_bare_exception_reaches_the_logger(rel):
    """作用域内每个文件:异常必须经 ``safe_exc()``，⛔ 不许裸传。"""
    hits = _bare_exception_log_sites(_REPO / rel)
    assert not hits, (
        "这些日志点把异常对象直接交给 logger —— 它的 __str__ 可能带 URL /"
        " 凭据 / 绝对路径：\n  " + "\n  ".join(hits)
        + "\n⇒ 改成 safe_exc(<异常>)")


def test_the_scan_actually_finds_things():
    """⭐ 阳性对照：⛔ 一处 logger 都没扫到说明扫描器坏了，那不是「通过」。

    （只统计 logger 调用总数，⛔ 不看有没有违规。）
    """
    total = 0
    for rel in _SCOPE:
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        total += sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name) and n.func.value.id == "logger")
    assert total >= 50, f"只扫到 {total} 个 logger 调用 —— 扫描器可能坏了"


# ───────── ④ 🔴 明确标注:这道门**挡不住**的那一面 ─────────


def test_exc_info_true_is_a_known_open_edge():
    """🔴 ``exc_info=True`` 会打印 **traceback**，里面同样有异常消息。

    ⭐ 本门**⛔ 挡不住**它 —— 这里显式登记为**开集**，⛔ 不冒充闭集。
    「假闭集比没有门更坏，它让人以为这一面守住了。」

    为什么不一并禁掉:traceback 是排查关键路径的主要手段，删掉它换来的
    安全收益远小于可观测性损失（且日志本身通常不外发）。
    ⇒ 折中:本条只**统计并钉住数量**，新增一处就会红，逼作者显式确认
    「这个 except 捕到的异常不会带机密」。
    """
    sites = []
    for rel in _SCOPE:
        tree = ast.parse((_REPO / rel).read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id == "logger"):
                continue
            for kw in n.keywords:
                if kw.arg == "exc_info" and getattr(kw.value, "value", None) is True:
                    sites.append(f"{rel}:{n.lineno}")
    assert len(sites) <= 6, (
        "新增了 exc_info=True 的日志点 —— traceback 里会有完整异常消息。\n"
        "确认这个 except 捕到的异常不含 URL/凭据/路径后，再把上限加一：\n  "
        + "\n  ".join(sites))
