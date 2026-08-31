"""聊天 / IM 面的错误 payload:按**来源**判,⛔ 不按文案形状判。

## 现场(2026-08-16 在跑着的云机槽里实取,⛔ 不是从源码推断)

HTTP 边界那一轮把判据换成了「有没有上游证据」,但**聊天这条兄弟路径没跟上** ——
``_provider_error_payload`` / ``client_safe_error_text`` 仍在用
``classified.reason``。于是本地异常只要文案里带上 provider 关键词:

    RuntimeError("agent step timed out: /volume1/private/profile/config.json")
      → reason=timeout          code=provider_network_error    原文出屏
    RuntimeError("model not found at /volume1/private/state.db")
      → reason=model_not_found  code=provider_model_not_found  原文出屏
    ValueError("invalid api key path /volume1/private/creds.yaml")
      → reason=auth             code=provider_auth             原文出屏

⇒ 用户看到「网络错误 / 模型不存在 / 鉴权失败,请重试」+ 一条 ``/volume1/…``
内部路径。**既指错方向,又泄漏。** 而 IM 聊天正是用户实际在用的那一面。

⭐ 判据本身早就在仓里(``has_upstream_evidence``),这条路只是**没去问它**。
"""
from __future__ import annotations

import ast
import inspect
import socket
from pathlib import Path

import pytest

from agent.error_classifier import (
    classify_api_error,
    client_safe_error_text,
    normalized_provider_error_code,
    INTERNAL_ERROR_USER_TEXT,
)


class _Upstream(Exception):
    """带 HTTP 状态码的 provider 异常 —— 上游证据的最小形态。"""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


def _outcome(exc):
    classified = classify_api_error(exc)
    return (
        normalized_provider_error_code(classified, error=exc),
        client_safe_error_text(classified, str(exc), error=exc),
    )


# ───────── ① 应该改变的行为 ─────────

@pytest.mark.parametrize("exc", [
    RuntimeError("agent step timed out: /volume1/private/profile/config.json"),
    RuntimeError("model not found at /volume1/private/state.db"),
    ValueError("invalid api key path /volume1/private/creds.yaml"),
    RuntimeError("rate limit exceeded while reading /volume1/private/cache.db"),
    AttributeError("'CustomProfile' object has no attribute 'supports_prompt_cache_key'"),
    KeyError("profile_id"),
    NameError("name '_cfg' is not defined"),
])
def test_our_own_failure_is_collapsed_and_coded_as_internal(exc):
    code, text = _outcome(exc)
    assert "/volume1" not in text, f"内部路径出屏:{text}"
    assert text == INTERNAL_ERROR_USER_TEXT, text
    # 🔴 码和文案必须同一个说法 —— 文案收了、码还说 provider_* 会让界面劝重试
    assert code == "internal_error", f"码指错方向:{code}"


# ───────── ② 🔒 必须保持不变的行为 ─────────

@pytest.mark.parametrize("exc, expect_code", [
    (_Upstream("Rate limit reached for gpt-4o. Please retry in 30s.", 429), "provider_rate_limit"),
    (_Upstream("Incorrect API key provided.", 401), "provider_auth"),
    (_Upstream("The model `foo-1` does not exist.", 404), "provider_model_not_found"),
    (_Upstream("upstream connect error at https://api.example.com/v1/chat", 502), "provider_bad_gateway"),
    (socket.gaierror(8, "nodename nor servname provided"), "provider_network_error"),
    (ConnectionResetError(104, "Connection reset by peer"), "provider_network_error"),
    (TimeoutError("Request timed out"), "provider_network_error"),
])
def test_real_upstream_failures_keep_their_text_and_code(exc, expect_code):
    """provider 的解释告诉用户等多久 / 换哪个模型 / 去哪充值 —— 抹掉比泄漏更糟。"""
    code, text = _outcome(exc)
    assert text == str(exc), f"上游原话被改了:{text!r} != {str(exc)!r}"
    assert code == expect_code, code


def test_expired_credentials_keep_the_one_line_the_user_can_act_on():
    """🔴 ``AuthError`` 是本地抛的、既无状态码也无响应体 —— 纯证据判据会把它
    收成「服务内部异常」,而它恰恰是**最需要原文**的一条。

    收掉它 = 把用户唯一能照做的那句话换成一个参考编号。
    """
    from hermes_cli.auth import AuthError

    exc = AuthError("OpenAI credentials expired; run `hermes auth openai`")
    code, text = _outcome(exc)
    assert "hermes auth openai" in text, text
    assert code != "internal_error", code


# ───────── ③ 门:调用点全集 ─────────
#
# ⚠️ **开集,明说**:它只覆盖「生产代码里这两个函数的调用」这一种形状,
# 证明不了别处没有第二条出站路径。作用是**防漂移** —— 上一轮正是
# HTTP 边界改了、聊天这条没跟上。

_MUST_PASS_ERROR = {"client_safe_error_text", "normalized_provider_error_code"}
_PRODUCTION_FILES = ["run_agent.py", "agent/conversation_loop.py"]


def _calls_missing_error_kwarg(src: str) -> list[str]:
    tree = ast.parse(src)
    lines = src.splitlines()
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
        # 调用点可能用了 `as _client_safe_error_text` 这类别名
        base = name.lstrip("_")
        if base not in _MUST_PASS_ERROR:
            continue
        if not any(k.arg == "error" for k in node.keywords):
            bad.append(f"L{node.lineno}: {lines[node.lineno - 1].strip()}")
    return sorted(set(bad))


@pytest.mark.parametrize("rel", _PRODUCTION_FILES)
def test_every_outbound_call_passes_the_original_exception(rel):
    root = Path(inspect.getfile(classify_api_error)).parents[1]
    bad = _calls_missing_error_kwarg((root / rel).read_text())
    assert not bad, (
        f"{rel} 里这些调用没传 error= ⇒ 判据退回 classified.reason,"
        f"本地异常会穿着 provider 外衣出屏:\n" + "\n".join(bad)
    )


def test_the_callsite_gate_is_not_vacuous():
    """⛔ 出生即空转的门比没有门更坏 —— 拿漏掉 error= 的真实写法喂它。"""
    missing = "x = client_safe_error_text(classified, message)\n"
    assert _calls_missing_error_kwarg(missing), "门抓不住漏传 error= 的写法"
    fixed = "x = client_safe_error_text(classified, message, error=error)\n"
    assert not _calls_missing_error_kwarg(fixed), "门在修好之后仍报红"
    aliased = "x = _client_safe_error_text(classified, message)\n"
    assert _calls_missing_error_kwarg(aliased), "门被 `as _alias` 绕过去了"


# ───────── ④ 阳性对照:判据有分辨力 ─────────


def test_the_two_directions_really_differ():
    """⛔ 若两边输出一样,上面所有断言都会因为别的原因通过。

    ⭐ 两格用**同一批字符串**,差别只可能来自 provenance,⛔ 不可能来自文案。
    """
    text = "model not found at /volume1/private/state.db"
    ours = _outcome(RuntimeError(text))
    theirs = _outcome(_Upstream(text, 404))
    assert ours != theirs
    assert ours[0] == "internal_error" and theirs[0] == "provider_model_not_found"
    assert theirs[1] == text and ours[1] != text


# ───────── ⑤ RH 第十二轮 P1-1:证据链取证有洞,**两个方向** ─────────
#
# ⭐ 根因:`has_upstream_evidence` 沿 cause 链取证,回答的是「能不能重试」;
# 而展示时要问的是「**这段文本是谁写的**」。两者不是一个契约。


def test_local_error_raised_while_handling_a_transport_error_is_still_ours():
    """🔴 泄漏方向:本地异常**在处理传输错误时抛出**,不许因此继承到「上游证据」。"""
    try:
        try:
            raise ConnectionError("connection reset by peer")
        except ConnectionError as inner:
            raise RuntimeError(
                "'CustomProfile' object has no attribute 'x' "
                "at /volume1/private/profile/config.yaml"
            ) from inner
    except RuntimeError as exc:
        code, text = _outcome(exc)
    assert "/volume1" not in text, f"内部路径出屏:{text}"
    assert code == "internal_error", code


def test_a_provider_sdk_wrapping_a_transport_error_still_passes_through():
    """🔒 必须保持不变:SDK 包一层传输错误那格是上一轮修对的,⛔ 别弄坏。

    ⭐ 与上一条的差别**只有一个自变量**:最外层异常的**类定义在谁的模块里**。
    """
    import httpx

    try:
        try:
            raise ConnectionError("connection reset")
        except ConnectionError as inner:
            raise httpx.ConnectError("connection failed") from inner
    except Exception as exc:  # noqa: BLE001
        code, text = _outcome(exc)
    assert text == "connection failed", text
    assert code == "provider_network_error", code


@pytest.mark.parametrize("factory, needle", [
    (lambda: __import__("agent.errors", fromlist=["x"]).MoAPresetNotFoundError(
        'preset "fast" not found; run: hermes moa list'), "hermes moa list"),
    (lambda: __import__("agent.errors", fromlist=["x"]).SSLConfigurationError(
        "CA bundle missing; set HERMES_CA_BUNDLE"), "HERMES_CA_BUNDLE"),
    (lambda: __import__("hermes_cli.auth", fromlist=["x"]).AuthError(
        "credentials expired; run `hermes auth openai`"), "hermes auth openai"),
])
def test_user_actionable_local_errors_keep_their_instruction(factory, needle):
    """🔴 吞掉方向:**有意面向用户**的本地失败,原文必须原样到。

    ⭐ 这比泄漏更该防 —— 它把用户唯一能照做的那句话拿掉了。
    判据是异常类**自己声明** ``hermes_user_actionable``,⛔ 不是在分类器里维护名单。
    """
    exc = factory()
    code, text = _outcome(exc)
    assert needle in text, text
    assert code != "internal_error", (
        f"码说 internal_error 而文案是可行动的 ⇒ 客户端按契约不展示 message,"
        f"这句话照样丢了(code={code})"
    )


def test_the_marker_is_a_contract_not_a_hardcoded_list():
    """⭐ 任何模块加一个类属性就能加入,⛔ 不需要改 error_classifier。"""
    from agent.error_classifier import USER_ACTIONABLE_ATTR, is_our_own_failure

    class _MyActionable(RuntimeError):
        pass

    setattr(_MyActionable, USER_ACTIONABLE_ATTR, True)
    exc = _MyActionable("do X to fix this")
    assert not is_our_own_failure(classify_api_error(exc), exc)
    # 负对照:不声明的同族异常仍判为我们的
    assert is_our_own_failure(classify_api_error(RuntimeError("boom")), RuntimeError("boom"))


# ───────── ⑥ RH 第十二轮 P1-2:provider-auth 收窄的调用点全集 ─────────


def test_no_resolver_wraps_arbitrary_exceptions_as_provider_auth():
    """🔴 上一轮只收窄了三个入口里的一个,而**生产走的是没改的那条**。

    ⚠️ **开集门,明说**:按 AST 找 `format_runtime_provider_error(...)` 的
    `raise` 点,检查其所在 `except` 子句是否还捕获裸 `Exception`。
    只覆盖这一种写法,防漂移,⛔ 不证明没有别的包装路径。
    """
    import ast
    from pathlib import Path

    root = Path(inspect.getfile(classify_api_error)).parents[1]
    bad = []
    for rel in ["gateway/run.py", "gateway/platforms/api_server.py"]:
        src = (root / rel).read_text()
        tree = ast.parse(src)
        for h in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
            raises_wrap = any(
                isinstance(n, ast.Call)
                and (getattr(n.func, "id", "") == "format_runtime_provider_error")
                for n in ast.walk(h)
            )
            if not raises_wrap:
                continue
            t = h.type
            names = ([getattr(e, "id", "") for e in t.elts]
                     if isinstance(t, ast.Tuple) else [getattr(t, "id", "")])
            if "Exception" in names or "BaseException" in names:
                bad.append(f"{rel}:{h.lineno} except {names}")
    assert not bad, (
        "这些 resolver 仍把**任何**异常包成 provider 认证失败 ⇒ 内部错误会以\n"
        "HTTP 200 当作 assistant 的回答送出:\n" + "\n".join(bad)
    )
