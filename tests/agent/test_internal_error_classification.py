"""内部异常⛔不许被归成「上游问题,请稍后重试」。

**现场(2026-08-16 用户实撞)**:界面显示「⚠️ 上游模型服务错误 / 💡 这是临时问题,
可以稍后重试」,真实根因是 ``AttributeError: 'CustomProfile' object has no
attribute 'supports_prompt_cache_key'`` —— ⛔ 跟上游一点关系没有,而且内部已经
重试 3/3 全挂,让用户再点「重试」是**纯粹无效的动作**。

**⛔ 不是某个分支写错,是闭集里缺一格。** ``FailoverReason`` 的 25 个类别
全部关于上游/传输/请求,没有任何一个表示「我们自己的代码坏了」;
``normalized_provider_error_code()`` 的返回值也全部以 ``provider_`` 开头。
⇒ 内部异常在结构上没有容身之处,只能落进 ``unknown``
(注释写死 "Unclassifiable — retry with backoff")。

**判据从「你叫什么名字」翻成「有没有上游证据」**:

  仓里已有先例 ``conversation_loop.is_local_validation_error`` 枚举了
  ``ValueError``/``TypeError``,**⛔ 漏了 ``AttributeError``** —— 因为
  「异常类型在不在我列的清单里」**本身就是开集**,补一个下次照样漏
  ``NameError``。⇒ 照抄第四问「它的判据在我这边还成立吗」= **不成立**,
  ⛔ 只照抄它「本地 bug 不重试」的意图,不照抄判据形式。

  闭集判据只问四件事,原料全部已存在:
  ① 有 HTTP 状态码 ② 有响应体 ③ 类型名在 ``_TRANSPORT_ERROR_TYPES``
  ④ ``OSError`` 且 errno 属于网络 errno 闭集 —— 都不是 ⇒ **内部**。

⭐ 下面 5 条里**只有第 1 组是缺陷本体,其余 4 组是锁** ——
⛔ 只测第一组的话,「把所有错误都判成内部」同样全绿。
"""
from __future__ import annotations

import errno
import json
import socket
import ssl

import pytest

from agent.error_classifier import (
    FailoverReason,
    classify_api_error,
    normalized_provider_error_code,
)


def _code(exc, **kw):
    return normalized_provider_error_code(classify_api_error(exc, **kw))


# ───────────── ① 缺陷本体:我们自己的 bug ⇒ internal,⛔ 不重试 ─────────────


@pytest.mark.parametrize("exc", [
    # 🔴 今晚用户实撞的那一个,逐字复刻
    AttributeError("'CustomProfile' object has no attribute 'supports_prompt_cache_key'"),
    NameError("name 'foo' is not defined"),
    KeyError("missing_key"),
    IndexError("list index out of range"),
    ZeroDivisionError("division by zero"),
])
def test_our_own_bugs_are_internal_and_not_retryable(exc):
    """⭐ 判据是「没有上游证据」,⛔ 不是「类型名在不在清单里」。

    所以这里**故意放了 5 种不同的异常** —— 只补 ``AttributeError``
    的实现会在 ``NameError`` 这条红,这正是本门要挡的东西。
    """
    c = classify_api_error(exc, provider="custom", model="pro")
    assert c.reason is FailoverReason.internal_error, (
        f"{type(exc).__name__} 被判成 {c.reason.value} —— 它没有任何上游证据")
    assert c.retryable is False, (
        f"{type(exc).__name__} 被判成可重试 —— 内部 bug 重试多少次都一样失败")


def test_internal_error_code_is_not_a_provider_code():
    """🔴 即使判对了,只要 code 还叫 ``provider_*``,用户看到的仍是「上游」。"""
    code = _code(AttributeError("boom"), provider="custom")
    assert not code.startswith("provider_"), (
        f"内部错误的 code 是 {code!r} —— 用户会以为是上游的问题")
    assert code.startswith("internal_"), f"code 应属 internal_ 族,实际 {code!r}"


# ───────────── ② 🔒 锁:真实 socket 错误仍是传输,仍重试 ─────────────


@pytest.mark.parametrize("no", [
    errno.ECONNRESET, errno.ETIMEDOUT, errno.ECONNREFUSED,
    errno.EHOSTUNREACH, errno.ENETUNREACH, errno.EPIPE,
])
def test_real_socket_errors_stay_transport_and_retryable(no):
    """⛔ 别把网络抖动一起判成内部。

    ⚠️ 真实 socket 错误就是 ``OSError`` 的子类 —— 把 ``OSError`` 整个
    踢出传输会让网络抖动不再重试,那比原缺陷更坏。
    ⇒ 作用域必须**刚好等于**缺陷:按 errno 收窄,⛔ 不按父类一刀切。
    """
    c = classify_api_error(OSError(no, "socket failure"), provider="custom")
    assert c.reason is FailoverReason.timeout, f"errno={no} 被判成 {c.reason.value}"
    assert c.retryable is True, f"errno={no} 不再重试 —— 网络抖动会直接失败"


@pytest.mark.parametrize("exc", [
    socket.gaierror(socket.EAI_NONAME, "Name or service not known"),
    socket.herror(1, "Unknown host"),
])
def test_dns_failures_are_transport_not_our_bug(exc):
    """🔴 DNS 解析失败是**网络问题**,⛔ 不是我们的 bug。

    ⚠️ 它踩中 errno 判据的一个盲区:``socket.gaierror`` 是 ``OSError`` 子类,
    但它的 errno 属于 **``EAI_*`` 空间**(``EAI_NONAME`` = 8),
    与 ``ECONNRESET``/``ETIMEDOUT`` 那套 ``E*`` 完全不是一个编号体系 ——
    实测 ``gaierror(EAI_NONAME).errno == 8``,⛔ 不在网络 errno 闭集里;
    类型名 ``gaierror`` 也不在 ``_TRANSPORT_ERROR_TYPES`` 里。

    ⇒ 不特判的话,**网络切换 / DNS 抖动会被报成「服务内部异常,请联系技术支持」**,
    而它其实重试一次就好 —— 正是「提示把用户指向错误方向」。
    """
    c = classify_api_error(exc, provider="custom")
    assert c.reason is FailoverReason.timeout, (
        f"{type(exc).__name__} 判成了 {c.reason.value} —— DNS 抖动会不再重试")
    assert c.retryable is True


class _ProviderWrapped(Exception):
    """SDK 把底层传输异常重新包装后抛出(Gemini/Bedrock/Vertex 都这么干)。"""


@pytest.mark.parametrize("inner_name,inner", [
    ("ConnectError", None),      # 运行时填,见下
    ("ReadTimeout", None),
    ("ECONNRESET", None),
])
def test_transport_error_wrapped_by_provider_sdk_is_still_transport(inner_name, inner):
    """🔴 provider SDK 把传输异常包一层,⛔ 不该因此变成"我们的 bug"。

    ⭐ **仓内先例就在同一个文件**:``_extract_status_code`` / ``_extract_error_body``
    都**沿 cause chain 最多遍历 5 层**(``__cause__`` 或 ``__context__``)——
    因为 SDK 重新包装是常态。我的 ``_has_upstream_evidence`` 只看最外层 ⇒
    **兄弟调用点没跟上**。

    实测后果:``GeminiAPIError <- httpx.ConnectError`` 被判 ``internal_error``
    且 ``retryable=False`` ⇒ **DNS / 连接建立 / 读流失败都不再重试**,
    还告诉用户「内部异常,请联系技术支持」。
    """
    import httpx
    made = {
        "ConnectError": httpx.ConnectError("dns fail"),
        "ReadTimeout": httpx.ReadTimeout("slow"),
        "ECONNRESET": OSError(errno.ECONNRESET, "connection reset"),
    }[inner_name]
    try:
        raise made
    except Exception as e:
        try:
            raise _ProviderWrapped("provider stream failed") from e
        except Exception as wrapped:
            c = classify_api_error(wrapped, provider="gemini")

    assert c.reason is not FailoverReason.internal_error, (
        f"被 SDK 包了一层的 {inner_name} 判成了内部错误 —— 网络问题会不再重试")
    assert c.retryable is True


def test_wrapper_without_transport_cause_is_still_internal():
    """⭐ 锁:⛔ 别把"有 __cause__"当成"有上游证据"。

    包了一层但里面是我们自己的 bug ⇒ **仍然是 internal**。
    ⛔ 否则「沿 cause chain 找」会退化成「只要有 cause 就放行」。
    """
    try:
        raise AttributeError("'CustomProfile' object has no attribute 'x'")
    except Exception as e:
        try:
            raise _ProviderWrapped("wrapped") from e
        except Exception as wrapped:
            c = classify_api_error(wrapped, provider="custom")
    assert c.reason is FailoverReason.internal_error
    assert c.retryable is False


def test_socket_timeout_type_still_transport():
    """``socket.timeout`` / ``TimeoutError`` 走类型名闭集这一支。"""
    c = classify_api_error(socket.timeout("timed out"), provider="custom")
    assert c.reason is FailoverReason.timeout and c.retryable is True


# ───────────── ③ 文件类 OSError ⇒ ⛔ 不再冒充 timeout ─────────────


@pytest.mark.parametrize("exc", [
    FileNotFoundError(errno.ENOENT, "No such file or directory", "/tmp/x"),
    PermissionError(errno.EACCES, "Permission denied", "/tmp/y"),
    IsADirectoryError(errno.EISDIR, "Is a directory", "/tmp/z"),
])
def test_file_errors_are_not_network_timeouts(exc):
    """🔴 这条比 ``unknown`` 兜底更坏:它给出一个**明确但错误**的分类。

    原实现 ``isinstance(error, OSError)`` 把整个 ``OSError`` 家族吞进
    transport ⇒ 读不到文件被报成「网络超时」⇒ **用户会去查网络**。
    ⭐「明确但错误」比「归类不明」更坏。
    """
    c = classify_api_error(exc, provider="custom")
    assert c.reason is not FailoverReason.timeout, (
        f"{type(exc).__name__} 仍被判成 timeout —— 用户会被指去查网络")
    assert c.retryable is False


# ───────────── ④ 🔒 锁:有上游证据的,分类一字不变 ─────────────


class _Upstream(Exception):
    """带状态码与响应体的上游错误(SDK 形状)。"""

    def __init__(self, status, body):
        super().__init__(f"upstream said {status}")
        self.status_code = status
        self.body = body


@pytest.mark.parametrize("status,expect", [
    (401, FailoverReason.auth),
    (402, FailoverReason.billing),
    (429, FailoverReason.rate_limit),
    (500, FailoverReason.server_error),
    (503, FailoverReason.overloaded),
])
def test_upstream_errors_classify_exactly_as_before(status, expect):
    """⭐ 锁:⛔ 别弄坏原来对的。有状态码 ⇒ 走原有分类,一字不动。"""
    c = classify_api_error(
        _Upstream(status, {"error": {"message": "upstream"}}), provider="custom")
    assert c.reason is expect, f"{status} 被改判成 {c.reason.value}"
    assert c.status_code == status


def test_body_only_upstream_error_is_not_internal():
    """没有状态码但**有响应体** ⇒ 仍是上游,⛔ 不许判成内部。"""
    e = _Upstream(None, {"error": {"message": "rate limit exceeded"}})
    e.status_code = None
    c = classify_api_error(e, provider="custom")
    assert c.reason is not FailoverReason.internal_error


# ───────────── ⑤ 🔒 锁:无状态码的传输错误仍是传输 ─────────────


@pytest.mark.parametrize("name", [
    "ServerDisconnectedError", "APIConnectionError", "APITimeoutError",
    "ReadTimeout", "RemoteProtocolError",
])
def test_sdk_wrapped_transport_errors_have_no_status_but_stay_transport(name):
    """⛔ 不把「无状态码」一律判成内部 —— 流式中断、SDK 包装的传输错误都没状态码。

    ⭐ 这就是判据第 3 支(``_TRANSPORT_ERROR_TYPES`` 类型名闭集)必须保留的原因。
    """
    exc = type(name, (Exception,), {})("connection dropped mid-stream")
    c = classify_api_error(exc, provider="custom")
    assert c.reason is not FailoverReason.internal_error, (
        f"{name} 被判成内部 —— 流式中断会不再重试")
    assert c.retryable is True


# ───────────── ⑥ 🔒 锁:别人踩坑换来的排除项,⛔ 一条都不许删 ─────────────


def test_json_decode_error_still_retries_issue_14782():
    """#14782:``JSONDecodeError`` 是 ``ValueError`` 子类,但它表示上游响应体
    被截断/损坏(转发层出错),**应当重试**,⛔ 不是本地编程 bug。"""
    c = classify_api_error(
        json.JSONDecodeError("Expecting value", "", 0), provider="custom")
    assert c.retryable is True, "JSONDecodeError 不再重试 —— #14782 回归"
    assert c.reason is not FailoverReason.internal_error


def test_ssl_error_still_transport_mro_trap():
    """``ssl.SSLError`` 通过 MRO **同时**继承 ``OSError`` 和 ``ValueError``。

    ⚠️ 任何「``ValueError`` ⇒ 本地 bug」或「``OSError`` ⇒ 内部」的粗判据
    都会把 TLS 传输失败误伤。这条锁住它仍走传输。
    """
    c = classify_api_error(ssl.SSLError("handshake failure"), provider="custom")
    assert c.reason is not FailoverReason.internal_error
    assert c.retryable is True


def test_unicode_encode_error_still_excluded():
    """``UnicodeEncodeError`` 是 ``ValueError`` 子类,由 surrogate 清洗路径
    单独处理 ⇒ ⛔ 不许被新判据抢走。"""
    c = classify_api_error(
        UnicodeEncodeError("utf-8", "x", 0, 1, "surrogate"), provider="custom")
    assert c.reason is not FailoverReason.internal_error


# ───── ⑦ 🔴 承重:判得准没用,得让端上真的收到「别重试」 ─────
#
# 「半条链 = 没做完」:加了新分类,**谁消费它**?端上按 payload 的
# ``recoverable`` 决定给什么操作 —— ``run_agent._provider_error_payload``
# 的注释写着 "clients key their affordance off this flag",并记了
# 2026-08-03 的设备实证:该标记为真时 App 渲染「重试」通道。
# ⇒ ``internal_error`` 必须让它为假,否则用户照样看到一个保证无效的按钮。


def _payload(classified):
    """按 ``_provider_error_payload`` 的真实依赖构造最小 self。"""
    from types import SimpleNamespace

    import run_agent

    stub = SimpleNamespace(
        provider="custom",
        model="pro",
        _summarize_api_error=lambda e: f"{type(e).__name__}",
    )
    return run_agent.AIAgent._provider_error_payload(
        stub, classified, RuntimeError("x"))


def test_internal_error_reaches_the_client_as_non_recoverable():
    """🔴 这条断的话,前面 31 条判得再准用户也看不到任何变化。"""
    c = classify_api_error(
        AttributeError("'CustomProfile' object has no attribute 'x'"),
        provider="custom", model="pro")
    p = _payload(c)
    assert p["code"] == "internal_error", p["code"]
    assert p["retryable"] is False, "retryable 没传下去"
    assert p["recoverable"] is False, (
        "recoverable=True —— 端上会给一个保证无效的「重试」按钮")


def test_internal_error_payload_carries_no_provider_error_code():
    """🔴 ``provider_error_code`` **会上屏**(端上把它当 detail 的白名单锚点之一)。

    ⇒ 里面若出现 ``AttributeError`` 或 ``supports_prompt_cache_key``,
    就是同一个「裸露底层报错」换了个字段。

    目前它安全,而且**不是巧合**:``provider_error_code`` 只从响应体提取
    (``_extract_error_code`` 首行 ``if not body: return ""``),而
    ``internal_error`` 的判定条件里「body 非空 ⇒ 有上游证据」——
    两者**结构互斥**,走到 internal 时 body 必空 ⇒ 该字段 falsy ⇒ 不进 payload。

    ⚠️ 这条门守的就是那个互斥关系:哪天有人放宽 ``_has_upstream_evidence``
    让带 body 的错误也能判成内部,它会红。
    """
    c = classify_api_error(
        AttributeError("'CustomProfile' object has no attribute 'supports_prompt_cache_key'"),
        provider="custom", model="pro")
    assert c.provider_error_code == "", f"内部错误带上了 {c.provider_error_code!r}"
    p = _payload(c)
    assert "provider_error_code" not in p, (
        f"该字段会上屏,却带了 {p.get('provider_error_code')!r}")
    # 前缀判据:端上只对 provider_* 保留 detail
    assert not p["code"].startswith("provider_")
    # 🔴 兜底:**整个** payload 里不许出现内部符号名。
    #
    # ⚠️ 这里原本写的是 `flat.replace(p.get("provider_message",""), "")` ——
    # 把 provider_message 从待检字符串里剔掉再查。那等于**给它发了免检**,
    # 于是对真实泄漏假绿(RH 第七轮 finding ④ 抓到)。
    # ⭐ 该函数的 docstring 自称 "safe, structured provider-error payload",
    # 里面出现实现符号就是违背它自己的契约 —— ⛔ 不许开这个口子。
    flat = repr(p)
    for leak in ("AttributeError", "supports_prompt_cache_key", "CustomProfile"):
        assert leak not in flat, f"{leak} 出现在给客户端的 payload 里:{p}"


def test_internal_error_never_asks_for_a_fallback_provider():
    """换个备用模型修不好我们自己的 bug —— ``should_fallback`` 必须为假。

    ⚠️ **本门只覆盖【产出侧】。** 消费侧(``conversation_loop`` 里
    ``_policy_no_fallback`` 那段)内联在一个几千行的重试循环中,
    ⛔ 无法在不重构的前提下单独驱动 —— **这一半没有运行时门,已显式登记。**
    ⭐ 同文件里那段注释自己写着「``classified.should_fallback`` …
    **下面这段从来不读它**」,所以产出侧为假**并不足以**保证不 fallback;
    消费侧是照抄 content-policy 的写法加进同一个集合的。
    """
    c = classify_api_error(AttributeError("boom"), provider="custom")
    assert c.should_fallback is False
    assert c.should_rotate_credential is False, "换凭据也修不好我们自己的 bug"
    assert c.should_compress is False


def test_a_retryable_upstream_error_stays_recoverable():
    """⭐ 锁:⛔ 别把原来对的一起弄成不可恢复。"""
    c = classify_api_error(_Upstream(503, {"error": {"message": "busy"}}),
                           provider="custom")
    p = _payload(c)
    assert p["retryable"] is True and p["recoverable"] is True


# ───── ⑨ 🔴 出站文本的单一出口:⛔ 不许有绕过安全化的字段 ─────
#
# 第一次修泄漏时我只安全化了 `provider_message`,而同一次内部异常还写进
# `final_response` / `error` / 状态行 —— 而 /v1/chat/completions(**活跃路由**,
# 24h 86 次请求)优先展示的正是 `final_response`。⇒ 用户照样看到实现符号。
# ⭐ 根因不是漏了某个字段,是**没有单一出口**:规则写了两遍就会漏第三处。


def test_client_safe_error_text_collapses_only_internal():
    from agent.error_classifier import INTERNAL_ERROR_USER_TEXT, client_safe_error_text

    internal = classify_api_error(
        AttributeError("'CustomProfile' object has no attribute 'x'"), provider="custom")
    assert client_safe_error_text(internal, "raw internals here") == INTERNAL_ERROR_USER_TEXT

    # ⭐ 锁:上游失败的文本是 provider 自己的解释,用户需要它逐字保留
    upstream = classify_api_error(_Upstream(429, {"error": {"message": "slow down"}}),
                                  provider="custom")
    assert client_safe_error_text(upstream, "Rate limit: retry in 30s") == "Rate limit: retry in 30s"


def test_no_outbound_field_bypasses_the_safe_text_helper():
    """🔴 闭集门:`_nonretryable_summary` 的**每一处赋值**都必须经过 helper。

    ⭐ 判据贴的是**数据流**(该变量在单文件内的赋值集合,由语法结构决定 ⇒ 闭集),
    ⛔ 不是「文件里存在某处调用了 helper」那种存在性判据。
    它一处赋值、七处消费(`_emit_status` / `final_response` / `error` / …)——
    只要赋值端安全,所有消费端就都安全;新增消费端也不会漏。
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "agent" / "conversation_loop.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))

    assigns, guarded = [], []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "_nonretryable_summary"
                   for t in node.targets):
            continue
        assigns.append(node.lineno)
        if any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
               and c.func.id.endswith("client_safe_error_text")
               for c in ast.walk(node.value)):
            guarded.append(node.lineno)

    # 阳性对照:一处都没扫到 ⇒ 变量被改名了,门空转,⛔ 那不是"通过"
    assert assigns, "没扫到 _nonretryable_summary 的赋值 —— 门可能已空转"
    unguarded = sorted(set(assigns) - set(guarded))
    assert not unguarded, (
        f"conversation_loop.py 这些行给 _nonretryable_summary 赋了未经安全化的值:"
        f"{unguarded} —— 内部异常原文会经 final_response/error 上屏")


# ───────────── ⑧ 阳性对照:判据本身没坏 ─────────────


def test_the_classifier_still_produces_varied_reasons():
    """⭐ ⛔ 一切都判成同一类 = 判据坏了,那不是「通过」。"""
    reasons = {
        classify_api_error(_Upstream(429, {}), provider="c").reason,
        classify_api_error(_Upstream(500, {}), provider="c").reason,
        classify_api_error(OSError(errno.ECONNRESET, "reset"), provider="c").reason,
        classify_api_error(AttributeError("boom"), provider="c").reason,
    }
    assert len(reasons) >= 3, f"分类器只产出 {reasons} —— 判据可能已塌缩"
