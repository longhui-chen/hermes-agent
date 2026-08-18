"""API error classification for smart failover and recovery.

Provides a structured taxonomy of API errors and a priority-ordered
classification pipeline that determines the correct recovery action
(retry, rotate credential, fallback to another provider, compress
context, or abort).

Replaces scattered inline string-matching with a centralized classifier
that the main retry loop in run_agent.py consults for every API failure.
"""

from __future__ import annotations

import enum
import errno
import functools
import json
import logging
import os
import socket
import ssl
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Error taxonomy ──────────────────────────────────────────────────────

class FailoverReason(enum.Enum):
    """Why an API call failed — determines recovery strategy."""

    # Authentication / authorization
    auth = "auth"                        # Transient auth (401/403) — refresh/rotate
    auth_permanent = "auth_permanent"    # Auth failed after refresh — abort

    # Billing / quota
    billing = "billing"                  # 402 or confirmed credit exhaustion — rotate immediately
    rate_limit = "rate_limit"            # 429 or quota-based throttling — backoff then rotate
    # Upstream model rate-limited (aggregator 429) — fallback to a different
    # model, NOT credential rotation. The user's key is healthy.
    upstream_rate_limit = "upstream_rate_limit"

    # Server-side
    overloaded = "overloaded"            # 503/529 — provider overloaded, backoff
    server_error = "server_error"        # 500/502 — internal server error, retry

    # Transport
    timeout = "timeout"                  # Connection/read timeout — rebuild client + retry
    # TLS certificate verification failure — deterministic for the host
    # (TLS-inspecting proxy, missing/expired CA bundle, self-signed cert).
    # Retrying reproduces the identical handshake failure, so fail fast
    # with actionable guidance instead of burning retries.
    ssl_cert_verification = "ssl_cert_verification"

    # Context / payload
    context_overflow = "context_overflow"  # Context too large — compress, not failover
    payload_too_large = "payload_too_large"  # 413 — compress payload
    image_too_large = "image_too_large"   # Native image part exceeds provider's per-image limit — shrink and retry

    # Model / provider policy
    model_not_found = "model_not_found"  # 404 or invalid model — fallback to different model
    provider_policy_blocked = "provider_policy_blocked"  # Aggregator (e.g. OpenRouter) blocked the only endpoint due to account data/privacy policy
    content_policy_blocked = "content_policy_blocked"  # Provider safety filter rejected this prompt — deterministic per-request, don't retry unchanged

    # Request format
    format_error = "format_error"        # 400 bad request — abort or strip + retry
    invalid_encrypted_content = "invalid_encrypted_content"  # Responses replay blob rejected — strip replay state and retry
    multimodal_tool_content_unsupported = "multimodal_tool_content_unsupported"  # Provider rejected list-type content in tool messages (e.g. Xiaomi MiMo) — downgrade to text and retry

    # Provider-specific
    thinking_signature = "thinking_signature"  # Anthropic thinking block sig invalid
    long_context_tier = "long_context_tier"    # Anthropic "extra usage" tier gate
    oauth_long_context_beta_forbidden = "oauth_long_context_beta_forbidden"  # Anthropic OAuth subscription rejects 1M context beta — disable beta and retry
    llama_cpp_grammar_pattern = "llama_cpp_grammar_pattern"  # llama.cpp json-schema-to-grammar rejects regex escapes in `pattern` / `format` — strip from tools and retry

    # Our own code broke — NOT the provider's fault
    # Every other member of this enum describes something that happened
    # upstream (auth, quota, overload, transport, request shape).  Without
    # this one an exception raised by *our* code has nowhere to go: it falls
    # through to `unknown`, which is defined as "retry with backoff", and
    # surfaces to the user as a `provider_*` code — i.e. "the model service
    # had a temporary problem, please retry".  Retrying is a guaranteed
    # no-op for a programming bug, so the user is handed an action that
    # cannot possibly work.  Deterministic per-request: never retry.
    internal_error = "internal_error"

    # Catch-all
    unknown = "unknown"                  # Unclassifiable — retry with backoff


# ── Classification result ───────────────────────────────────────────────

@dataclass
class ClassifiedError:
    """Structured classification of an API error with recovery hints."""

    reason: FailoverReason
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    provider_error_code: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    # Recovery action hints — the retry loop checks these instead of
    # re-classifying the error itself.
    retryable: bool = True
    should_compress: bool = False
    should_rotate_credential: bool = False
    should_fallback: bool = False

    @property
    def is_auth(self) -> bool:
        return self.reason in {FailoverReason.auth, FailoverReason.auth_permanent}


def content_policy_fallback_disabled() -> bool:
    """True when a GENERAL provider content-policy refusal must end the turn
    instead of failing over to a second model.

    Default is off, preserving general-purpose behaviour: a refusal from one
    provider (OpenAI usage policy, Codex cyber, Anthropic safety) may be
    legitimately answered by a different model, so failover is allowed.

    Scope: this switch does NOT govern a Zettlab moderation-GATEWAY block
    (code=moderation_input_blocked / type=content_policy_violation). A gateway
    verdict is a compliance decision that never fails over — unconditionally,
    handled at the classification site via ``_is_moderation_gateway_block`` — so
    a CN/compliance deployment needs no env flag for it. Set
    ``HERMES_CONTENT_POLICY_NO_FALLBACK=1`` only to additionally stop failover
    on general provider refusals too.
    """
    return os.getenv("HERMES_CONTENT_POLICY_NO_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


#: What the user is told when the failure was our own code.  The original
#: exception text (class names, attribute names, file paths) stays in the
#: logs; none of it means anything to the person waiting for an answer, and
#: shipping it leaks our internals to whatever renders the message.
INTERNAL_ERROR_USER_TEXT = "服务内部异常"


def has_upstream_evidence(error: BaseException) -> bool:
    """这个失败有没有**来自上游的证据**?

    ⚠️ ⛔ 不要用 ``classify_api_error(...).reason is internal_error`` 代替本函数。
    分类流水线是**按恢复策略**排序的:文本模式匹配在第 4 步,证据检查在第 8 步。
    于是本地的 ``RuntimeError("agent step timed out: /volume1/private/…")``
    先撞上 timeout 模式就被判成上游失败,整条路径原样出屏 —— 实测复现。
    ⭐ 「分类器认为可以重试」和「这个错误来自上游」是**两个问题**,
    第一个的答案证明不了第二个。

    证据是闭集:HTTP 状态码、响应体、已知传输类型名、网络 errno(沿 cause
    链 5 层)。四样都没有 ⇒ 这个异常只可能来自我们自己的代码。

    ⚠️ **已知开集侧**:provider SDK 若抛出一个既无状态码、又无响应体、
    类型名也不在传输集里的纯文本异常,这里会判成「我们的」而把它的解释
    收成安全文案。调用方必须把原文记进日志(见 ``_internal_error_text``),
    否则那条解释就真的丢了。
    """
    if not isinstance(error, Exception):
        return False
    return _has_upstream_evidence(
        error, _extract_status_code(error), _extract_error_body(error)
    )


#: 异常类可以**自己声明**「我是有意面向用户的本地失败,原文必须原样送达」。
#: ⭐ 这是一个**契约**,任何模块加一个类属性就能加入,⛔ 不需要在本文件维护类型名单
#: (名单是开集,而且改一次要动两个仓)。
USER_ACTIONABLE_ATTR = "hermes_user_actionable"

#: 一个**我们构造的**异常可以声明「我手上这段文本是上游给的原话」。
#: ⭐ 与 ``USER_ACTIONABLE_ATTR`` 严格对称,而且是同一个错误的**另一端**:
#: 前者说「这段我们写的话必须送达」,后者说「这段上游的话必须送达」。
#:
#: 为什么需要它:重新包装会**抹掉出身**。``_normalize_codex_response`` 收到
#: ``status="failed"`` 的 Responses 响应时,把 provider 在 ``error`` 里给的原因
#: 包成一个**裸 ``RuntimeError``** —— 没有 ``status_code``、没有响应体、没有
#: ``__cause__``。四样证据一样都没有,于是分类第 9 步判「这是我们自己的 bug」:
#: **不重试、不 fallback**,provider 那句真正的解释还被压成「服务内部异常」。
#: 一次上游侧的瞬时失败,就这样变成用户这条消息的永久失败。
#:
#: ⛔ 不用「函数名 / 模块名白名单」来救 —— 那是开集,下一个包装点又会漏。
#: ⭐ 判据落在**包装的人自己声明**上:谁抹掉了出身,谁负责重新写明。
#: 同形先例见 ZET-2473「只投递 producer 显式声明的交付物」。
UPSTREAM_ORIGIN_ATTR = "hermes_upstream_origin"

#: 我们自己的顶层包 —— 这些模块里定义的异常,**文本是我们写的**。
#: ⛔ 这不是「按名字判」:判的是**类定义在谁的代码里**,即**谁写了那段文案**,
#: 这正是「这段文本能不能给用户看」要问的事。
_OUR_TOP_LEVEL_PACKAGES = (
    "agent", "gateway", "tools", "hermes_cli", "plugins", "cron",
    "tui_gateway", "acp_adapter", "run_agent", "hermes_constants",
)


def _is_user_actionable_local(error: BaseException) -> bool:
    """这个异常有没有声明「我面向用户、原文必须到」。

    ⚠️ 查的是**实例**,⛔ 不是 ``type(error)``。实例查找**天然回退到类**,
    所以这是严格超集:类上声明的(``SSLConfigurationError`` / ``AuthError`` /
    ``MoAPresetNotFoundError``)行为逐字不变,同时允许在**抛出点**给一个内建
    异常盖戳 —— 那正是 ``bedrock_adapter._require_boto3()`` 需要的:
    它抛的是内建 ``ImportError`` / ``RuntimeError``,消息里**只有安装/升级命令
    是用户唯一能照做的事**,而第 ③ 问会因为「类定义在内建里」把它收成
    「服务内部异常」。⛔ 不放宽全体内部异常,只让抛出点能显式登记。
    ⭐ 与 ``_declares_upstream_origin`` 的查找方式一致。
    """
    return bool(getattr(error, USER_ACTIONABLE_ATTR, False))


def declare_upstream_origin(error: BaseException) -> BaseException:
    """给一个**我们构造的**异常盖上「文本来自上游」的戳,并原样返回。

    只在**重新包装 provider 原话**的地方调用 —— 包装会抹掉 status / body /
    cause 三样证据,这里把出身补回去。⛔ 不要给我们自己写的文案盖戳:
    那等于把我们的话冒充成上游的,是本判据的反向滥用。
    """
    setattr(error, UPSTREAM_ORIGIN_ATTR, True)
    return error


def _declares_upstream_origin(error: BaseException) -> bool:
    """包装者有没有显式声明「这段文本是上游的」。"""
    return bool(getattr(error, UPSTREAM_ORIGIN_ATTR, False))


def _carries_own_upstream_evidence(error: BaseException) -> bool:
    """这个异常**自己**带上游证据 —— ⛔ 不看 cause 链。

    ⭐ 与 ``has_upstream_evidence`` 的区别就是本轮缺陷的根:
    链式取证回答的是「**这次失败能不能重试**」,而我们在展示时问的是
    「**手上这段文本是谁写的**」。两者不是一个契约。
    """
    if not isinstance(error, Exception):
        return False
    if _declares_upstream_origin(error):
        return True
    if getattr(error, "status_code", None) is not None:
        return True
    if isinstance(_extract_error_body(error), dict) and _extract_error_body(error):
        return True
    if type(error).__name__ in _TRANSPORT_ERROR_TYPES:
        return True
    return _is_network_oserror(error)


def _outer_class_is_ours(error: BaseException) -> bool:
    """最外层异常的**类**是不是我们(或 Python 内建)定义的。

    内建 ``RuntimeError`` / ``ValueError`` / ``AttributeError`` 由我们的代码
    抛出 ⇒ 消息是我们写的;provider SDK 自己的异常类 ⇒ 消息是它写的。
    """
    module = getattr(type(error), "__module__", "") or ""
    if module in ("builtins", "__builtin__", ""):
        return True
    return module.split(".", 1)[0] in _OUR_TOP_LEVEL_PACKAGES


def is_our_own_failure(classified: ClassifiedError, error: Optional[BaseException]) -> bool:
    """这次失败是不是**我们自己的代码**坏了。

    ⭐ 单一判据,给所有面向用户的出站文本共用 —— ⛔ 不许在调用点各写一套。

    ``classified.reason`` **不足以回答这个问题**:分类流水线是按**恢复策略**
    排序的,文本模式匹配在第 4 步、证据检查在第 8 步。于是本地的
    ``RuntimeError("agent step timed out: /volume1/private/…")`` 先撞上
    timeout 模式,被判成 ``timeout`` / ``provider_network_error`` ——
    界面说「网络错误,请重试」,消息里还挂着一条内部路径。
    (同形的还有 "model not found" → ``model_not_found``、
    "invalid api key" → ``auth``,实测三例全泄漏。)

    ## 🔴 上一版只问 ``has_upstream_evidence`` —— **两个方向都错**

    那个判据**沿 cause 链**取证,回答的是「这次失败能不能重试」。可展示时要问的是
    「**手上这段文本是谁写的**」,两者不是一个契约:

    · **泄漏方向**:本地异常只要是在处理传输错误时抛出的,就沿链继承到「上游证据」
      ⇒ ``raise RuntimeError("… at /volume1/private/config.yaml") from ConnectionError``
      整条内部路径原样出屏(实测复现)。
    · **吞掉方向**:``MoAPresetNotFoundError("run: hermes moa list")`` 这种**有意
      面向用户**的本地失败没有上游证据,被压成「服务内部异常」——
      **把用户唯一能照做的那句话拿掉了**,比泄漏更该防。

    ⇒ 按下面的顺序问,每一问都是闭集:
      1. 异常类**自己声明**面向用户(``USER_ACTIONABLE_ATTR``)⇒ 原文必须到
      2. 异常**自己**带上游证据(⛔ 不看链)⇒ 上游的文本
      3. 异常类定义在**我们的包或内建**里 ⇒ **文本是我们写的** ⇒ 收
      4. 其余(provider SDK 自己的异常类)⇒ 退回链式判据,
         保住「SDK 包一层传输错误」那格 —— ⛔ 别把上一轮修对的东西弄坏
    """
    if error is None:
        return classified.reason == FailoverReason.internal_error
    # 🔴 ``classified`` **自己**可能就带着上游证据 —— 一个 HTTP 状态码只可能来自
    # 一次真实的响应。上一版只盘问异常对象,于是
    # ``ClassifiedError(status_code=400, reason=content_policy_blocked)`` 配一个
    # 裸 ``Exception("boom")`` 时,第 ③ 问("类定义在内建里 ⇒ 文本是我们写的")
    # 判成我们的 bug ⇒ **内容合规拦截被压成「服务内部异常」**,App 那边的合规
    # 提示直接消失,换成一句用户照做也没用的「请稍后重试」。
    # ⭐ 证据不止长在异常上,也长在分类结果上;哪一侧有都算。
    # ⛔ 这不会放回原来的泄漏:HTTP/SSE 边界走 ``error_text_is_ours``(手上没有
    #    ``classified``),而本地异常经 ``classify_api_error`` 得到的
    #    ``status_code`` 本来就是 ``None``。
    if classified.status_code is not None:
        return False
    return error_text_is_ours(error)


def error_text_is_ours(error: BaseException) -> bool:
    """手上这段异常文本**是不是我们写的** —— 四问阶梯的唯一实现。

    ⭐ 抽出来是因为**有两个入口需要同一个答案**:
    ``is_our_own_failure``(带 ``ClassifiedError``,给聊天出站用)和
    HTTP/SSE 边界的 ``_boundary_error_text``(手上只有一个异常)。
    ⛔ 后者原先自己问 ``has_upstream_evidence``,于是
    ``raise RuntimeError("… /volume1/private/config.yaml") from ConnectionError``
    沿链拿到「上游证据」,把内部路径原样发给了所有接该边界的客户端。
    **同一个问题两处各写一套判据,必然漂移** —— 这已经是本线第二次了。

    四问逐条闭集,顺序不可换:
      ① 异常类自己声明面向用户 ⇒ 原文必须到(⛔ 不是我们的"内部错误")
      ② 异常**自己**带上游证据(⛔ 不看 cause 链)⇒ 上游的文本
      ③ 异常类定义在我们的包或内建里 ⇒ 文本是我们写的 ⇒ 收
      ④ 其余(provider SDK 自己的异常类)⇒ 退回链式判据,
         保住「SDK 包一层传输错误」那格
    """
    if _is_user_actionable_local(error):
        return False
    if _carries_own_upstream_evidence(error):
        return False
    if _outer_class_is_ours(error):
        return True
    return not has_upstream_evidence(error)


def client_safe_error_text(
    classified: ClassifiedError,
    raw_text: str,
    *,
    error: Optional[BaseException] = None,
) -> str:
    """Collapse an internal failure's text before it crosses to a client.

    Single source of truth: every outbound surface (``provider_message``,
    ``final_response``, ``error``, status lines) must route its text through
    here, otherwise the ones that don't become the leak — which is exactly
    how the first fix missed `final_response`.

    ⛔ 只有「我们自己的错」才被改写。上游失败的文本是 provider 自己的解释,
    用户需要它逐字到达 —— 抹掉比泄漏更糟。
    """
    if is_our_own_failure(classified, error):
        return INTERNAL_ERROR_USER_TEXT
    return raw_text


def normalized_provider_error_code(
    classified: ClassifiedError,
    *,
    error: Optional[BaseException] = None,
) -> str:
    """Return the stable Zettlab chat error code for a provider failure.

    ⭐ 传 ``error`` 时用与 ``client_safe_error_text`` **同一个** 判据
    (``is_our_own_failure``)—— 否则码和文案会各说各话:文案已经收成
    「服务内部异常」,码却还是 ``provider_network_error``,界面照着码劝用户
    「检查网络后重试」。⛔ 一次失败只能有一个说法。
    """
    if is_our_own_failure(classified, error):
        return "internal_error"

    status = classified.status_code
    reason = classified.reason

    if status in {408, 504}:
        return "provider_timeout"
    if status == 409:
        return "provider_conflict"
    if status == 413:
        return "payload_too_large"
    if status in {423, 424, 425}:
        return "provider_unavailable"
    if status is not None and 400 <= status < 500 and status not in {
        400,
        401,
        402,
        403,
        404,
        422,
        429,
    }:
        return "provider_client_error"
    if status == 502:
        return "provider_bad_gateway"
    if status == 507:
        return "provider_billing"

    # Our own failure — must NOT wear a `provider_*` code, or the surface
    # renders "the model service had a problem, retry later" for a bug that
    # retrying cannot fix.  Checked before the status-code fallbacks below:
    # an internal error carries no status, so it would otherwise land on
    # `provider_error`.
    if reason == FailoverReason.internal_error:
        # ⚠️ 走到这里说明 is_our_own_failure() 已判定「不是我们的错」
        # (有意面向用户的本地失败,如 SSLConfigurationError / MoAPresetNotFound),
        # 而分类器出于**恢复策略**仍标了 internal_error(它确实不该重试)。
        # ⛔ 此时不许发 internal_error 码:按交付契约,客户端见到该码而没有
        # reference_id 就**不展示 message** ⇒ 那句可行动的原文照样被丢掉。
        # ⭐ 码要和文案说同一件事。
        if error is not None and not is_our_own_failure(classified, error):
            return "agent_error"
        return "internal_error"

    if reason == FailoverReason.billing:
        return "provider_billing"
    if reason == FailoverReason.rate_limit:
        return "provider_rate_limit"
    if reason in {FailoverReason.auth, FailoverReason.auth_permanent}:
        return "provider_forbidden" if status == 403 else "provider_auth"
    if reason == FailoverReason.provider_policy_blocked:
        return "provider_policy_blocked"
    if reason == FailoverReason.content_policy_blocked:
        return "content_blocked"
    if reason == FailoverReason.model_not_found:
        return "provider_model_not_found"
    if reason == FailoverReason.timeout:
        if status is None:
            return "provider_network_error"
        return "provider_timeout"
    if reason == FailoverReason.overloaded:
        return "provider_overloaded"
    if reason == FailoverReason.server_error:
        return "provider_server_error"
    if reason == FailoverReason.context_overflow:
        return "context_overflow"
    if reason in {FailoverReason.payload_too_large, FailoverReason.image_too_large}:
        return "payload_too_large"
    if reason in {
        FailoverReason.format_error,
        FailoverReason.thinking_signature,
        FailoverReason.llama_cpp_grammar_pattern,
    }:
        return "provider_bad_request"
    if reason in {
        FailoverReason.long_context_tier,
        FailoverReason.oauth_long_context_beta_forbidden,
    }:
        return "provider_forbidden"

    if status is None:
        return "provider_error"
    if status in {400, 422}:
        return "provider_bad_request"
    if status == 401:
        return "provider_auth"
    if status == 402:
        return "provider_billing"
    if status == 403:
        return "provider_forbidden"
    if status == 404:
        return "provider_endpoint_not_found"
    if status == 429:
        return "provider_rate_limit"
    if status == 500:
        return "provider_server_error"
    if status in {503, 529}:
        return "provider_overloaded"
    if 400 <= status < 500:
        return "provider_client_error"
    if 500 <= status < 600:
        return "provider_server_error"
    return "provider_error"



# ── Provider-specific patterns ──────────────────────────────────────────

# Patterns that indicate billing exhaustion (not transient rate limit)
_BILLING_PATTERNS = [
    "insufficient credits",
    "insufficient_credits",
    "insufficient_quota",
    "insufficient balance",
    "credit balance",
    "credits exhausted",
    "credits have been exhausted",
    "no usable credits",
    "top up your credits",
    "payment required",
    "billing hard limit",
    "exceeded your current quota",
    "account is deactivated",
    "plan does not include",
    "out of extra usage",  # Anthropic OAuth Pro/Max overage bucket depleted (HTTP 400)
    "out of funds",
    "run out of funds",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    "not available on the free tier",
]

# xAI's explicit Grok credit-exhaustion code. Keep the HTTP 403 special case
# provider-scoped: other providers' generic billing codes historically remain
# auth failures when they arrive as 403.
_XAI_SPENDING_LIMIT_ERROR_CODE = "personal-team-blocked:spending-limit"

# Structured provider codes that mean the account cannot serve paid traffic
# until credits/subscription capacity is restored. xAI returns its explicit
# Grok spending-limit signal as HTTP 403 rather than 402.
_BILLING_ERROR_CODES = frozenset({
    "insufficient_quota",
    "billing_not_active",
    "payment_required",
    "insufficient_credits",
    "no_usable_credits",
    "balance_depleted",
    "model_not_supported_on_free_tier",
    _XAI_SPENDING_LIMIT_ERROR_CODE,
})

# Patterns that indicate rate limiting (transient, will resolve)
_RATE_LIMIT_PATTERNS = [
    "rate limit",
    "rate_limit",
    "too many requests",
    "throttled",
    "requests per minute",
    "tokens per minute",
    "requests per day",
    "try again in",
    "please retry after",
    "resource_exhausted",
    "rate increased too quickly",  # Alibaba/DashScope throttling
    # AWS Bedrock throttling
    "throttlingexception",
    "too many concurrent requests",
    "servicequotaexceededexception",
    # Generic throttle prefix — Bedrock (and some proxies) surface throttling
    # as "Throttling error: Too many tokens, please wait before trying
    # again."  Without this entry the message falls through to the
    # context-overflow list (which contains "too many tokens") and the retry
    # loop compresses a healthy session instead of backing off.  Matched
    # BEFORE _CONTEXT_OVERFLOW_PATTERNS in the message-only path, so the
    # throttle wins.  (port of anomalyco/opencode#37848's exclusion guard)
    "throttling",
]

# Patterns that indicate provider-side overload, NOT a per-credential rate
# limit or billing problem.  The credential is valid — the server is just
# busy — so the correct recovery is "back off and retry the same key", never
# "rotate the credential" (rotating exhausts the pool while the endpoint is
# still busy; a single-key user has nothing to rotate to).  Some providers
# (notably Z.AI / Zhipu) reuse HTTP 429 for server-wide overload, so the 429
# status path matches the body against this list before falling through to
# the rate_limit default.  Phrases are kept narrow and overload-flavoured so a
# normal rate-limit message ("you have been rate-limited") doesn't hit this
# bucket. (#14038, #15297)
_OVERLOADED_PATTERNS = [
    "overloaded",
    "temporarily overloaded",
    "service is temporarily overloaded",
    "service may be temporarily overloaded",
    "server is overloaded",
    "server overloaded",
    "service overloaded",
    "service is overloaded",
    "upstream overloaded",
    "currently overloaded",
    "at capacity",
    "over capacity",
]

# Usage-limit patterns that need disambiguation (could be billing OR rate_limit)
_USAGE_LIMIT_PATTERNS = [
    "usage limit",
    "quota",
    "limit exceeded",
    "key limit exceeded",
]

# Patterns confirming usage limit is transient (not billing)
_USAGE_LIMIT_TRANSIENT_SIGNALS = [
    "try again",
    "retry",
    "resets at",
    "reset in",
    "wait",
    "requests remaining",
    "periodic",
    "window",
]

# Payload-too-large patterns detected from message text (no status_code attr).
# Proxies and some backends embed the HTTP status in the error message.
_PAYLOAD_TOO_LARGE_PATTERNS = [
    "request entity too large",
    "payload too large",
    "error code: 413",
    # Anthropic's structured 413 error type.  Normally arrives with an HTTP
    # 413 status (handled by the status path), but aggregators/proxies can
    # re-wrap it into a plain message with no status attribute — route it to
    # the same compression recovery.  (port of anomalyco/opencode#37848)
    "request_too_large",
    "request exceeds the maximum size",
]

# Image-size patterns.  Matched against 400 bodies (not 413) because most
# providers return a 400 with a specific image-too-big message before the
# whole request hits the 413 size limit.  Anthropic's wording is the most
# important here (hard 5 MB per image, returned as
# "messages.N.content.K.image.source.base64: image exceeds 5 MB maximum").
_IMAGE_TOO_LARGE_PATTERNS = [
    "image exceeds",        # Anthropic: "image exceeds 5 MB maximum"
    "image too large",      # generic
    "image_too_large",      # error_code variant
    "image size exceeds",   # variant
    "image dimensions exceed",  # Anthropic: "image dimensions exceed max allowed size: 8000 pixels"
    "dimensions exceed max allowed size",  # Anthropic dimension-cap (wording variant)
    "max allowed size: 8000",  # Anthropic dimension-cap (explicit pixel ceiling)
    # "request_too_large" on a request known to contain an image → image is
    # the likely culprit; we still try the shrink path before giving up.
]

# Providers that follow the OpenAI spec strictly require tool message
# ``content`` to be a string.  Some (Anthropic native, Codex Responses,
# Gemini native, first-party OpenAI) extend this to accept a content-parts
# list (text + image_url) so screenshots from computer_use survive.  Others
# (Xiaomi MiMo, some Alibaba endpoints, a long tail of OpenAI-compatible
# providers) reject the list with a 400 — the patterns below are the most
# common error shapes we see.  Recovery: strip image parts from tool
# messages in-place, record the (provider, model) for the rest of the
# session so we don't waste another call learning the same lesson, retry.
#
# See: https://github.com/NousResearch/hermes-agent/issues/27344
_MULTIMODAL_TOOL_CONTENT_PATTERNS = [
    # Xiaomi MiMo: {"error":{"code":"400","message":"Param Incorrect","param":"text is not set"}}
    "text is not set",
    # Generic "tool message must be string" shapes
    "tool message content must be a string",
    "tool content must be a string",
    "tool message must be a string",
    # OpenAI-compat servers that reject list-type tool content with a
    # schema-validation message
    "expected string, got list",
    "expected string, got array",
    # Alibaba/DashScope variant
    "tool_call.content must be string",
]

# Context overflow patterns
_CONTEXT_OVERFLOW_PATTERNS = [
    "context length",
    "context size",
    "maximum context",
    "token limit",
    "too many tokens",
    "reduce the length",
    "exceeds the limit",
    "context window",
    "prompt is too long",
    "prompt exceeds max length",
    # NOTE: bare "max_tokens" is load-bearing — the output-cap-retry path keys
    # off it (e.g. "max_tokens: 65536 > context_window: 200000 ..."). Do NOT
    # remove it. Provider empty-response advisories also contain "very low
    # max_tokens", but those are intercepted by _EMPTY_PROVIDER_RESPONSE_PATTERNS
    # BEFORE this list is consulted, so they never mis-route into compression.
    "max_tokens",
    "maximum number of tokens",
    # vLLM / local inference server patterns
    "exceeds the max_model_len",
    "max_model_len",
    "prompt length",             # "engine prompt length X exceeds"
    "input is too long",
    "maximum model length",
    # Ollama patterns
    "context length exceeded",
    "truncating input",
    # llama.cpp / llama-server patterns
    "slot context",              # "slot context: N tokens, prompt N tokens"
    "n_ctx_slot",
    # Chinese error messages (some providers return these)
    "超过最大长度",
    "上下文长度",
    # Z.AI / Zhipu GLM pattern (English form; error code 1210)
    "tokens in request more than max tokens allowed",
    # AWS Bedrock Converse API error patterns
    "input is too long",
    "max input token",
    "input token",
    "exceeds the maximum number of input tokens",
    # Together/Fireworks-style: "Input length 131393 exceeds the maximum
    # allowed input length of 131040 tokens."  No other pattern in this list
    # matches that wording.  (port of anomalyco/opencode#37848)
    "maximum allowed input length",
]

# Model not found patterns
_MODEL_NOT_FOUND_PATTERNS = [
    "is not a valid model",
    "invalid model",
    "model not found",
    "model_not_found",
    "does not exist",
    "no such model",
    "unknown model",
    "unsupported model",
    # OpenRouter returns 404 with this message when none of the candidate
    # endpoints for the selected model support tool/function calling.
    # Classifying this as model_not_found triggers fallback to a different
    # model or provider that does support tools.  Without this entry the
    # pattern falls through to ``unknown`` with ``retryable=True``, the
    # retry loop burns all attempts on the same deterministic rejection,
    # and the error surfaces as a confusing "model not found" message
    # instead of automatically failing over.  See PR #58446.
    "no endpoints found that support tool use",
]

# Malformed-message-array 400s.  Deterministic request-shape rejections that
# describe the *transcript* being invalid, not a parameter.  The canonical
# case: a stream dies mid-response and Hermes persists a content-less
# assistant stub; on the next turn the Anthropic message schema (and the
# litellm/Bedrock proxies in front of it) reject the whole request with
#   "all messages must have non-empty content except for the optional final
#    assistant message"  /  errorCode INVALID_REQUEST_BODY
# These are NOT context overflow — the input may be tiny — but a large
# session used to mis-route them into the compression loop via the generic
# "400 + large session" heuristic below, ending in "Cannot compress further"
# every retry (the input is unchanged, so compression cannot help).  Match
# the message-shape signals explicitly and fail fast as a format_error so the
# loop stops looping.  The empty-stub creation is the root cause (fixed in
# chat_completion_helpers); this pattern stops the misclassification symptom
# for transcripts that already contain a poisoned stub.
_INVALID_MESSAGE_BODY_PATTERNS = [
    "must have non-empty content",
    "messages must have non-empty",
    "invalid_request_body",
    "text content blocks must be non-empty",
    "content field is required",
    "messages: at least one message is required",
]

# Request-validation patterns — the request is malformed and will fail
# identically on every retry. Some OpenAI-compatible gateways (notably
# codex.nekos.me) return these as 5xx instead of the standard 4xx, which
# makes the generic "5xx → retryable server_error" rule misfire: the retry
# loop hammers the same deterministic rejection 3+ times, then the
# transport-recovery path resets the counter and does it again, producing
# a request flood. When a 5xx body carries one of these unambiguous
# request-validation signals, classify as a non-retryable format_error so
# the loop fails fast and falls back instead of looping.
_REQUEST_VALIDATION_PATTERNS = [
    "unknown parameter",
    "unsupported parameter",
    "unrecognized request argument",
    "invalid_request_error",
    "unknown_parameter",
    "unsupported_parameter",
]

# OpenRouter aggregator policy-block patterns.
#
# When a user's OpenRouter account privacy setting (or a per-request
# `provider.data_collection: deny` preference) excludes the only endpoint
# serving a model, OpenRouter returns 404 with a *specific* message that is
# distinct from "model not found":
#
#   "No endpoints available matching your guardrail restrictions and
#    data policy. Configure: https://openrouter.ai/settings/privacy"
#
# We classify this as `provider_policy_blocked` rather than
# `model_not_found` because:
#   - The model *exists* — model_not_found is misleading in logs
#   - Provider fallback won't help: the account-level setting applies to
#     every call on the same OpenRouter account
#   - The error body already contains the fix URL, so the user gets
#     actionable guidance without us rewriting the message
_PROVIDER_POLICY_BLOCKED_PATTERNS = [
    "no endpoints available matching your guardrail",
    "no endpoints available matching your data policy",
    "no endpoints found matching your data policy",
]

# Provider content-policy / safety-filter blocks. Distinct from
# ``provider_policy_blocked`` above (which is an OpenRouter *account*-level
# data/privacy guardrail) — these are *per-prompt* safety decisions made by
# the upstream model provider. They are deterministic for the unchanged
# request, so retrying the same prompt three times just reproduces the same
# block and burns paid attempts on a refusal. The recovery is to switch to a
# configured fallback model/provider immediately, or surface the block to
# the user with actionable guidance if no fallback exists.
#
# Patterns are intentionally narrow — each phrase is a verbatim string from
# a specific provider's safety pipeline, not a generic word like "policy" or
# "violation" that could collide with billing/auth/format errors:
#   • OpenAI Codex cybersecurity refusal (gpt-5.5, the case from #18028)
#   • OpenAI moderation refusal ("violates our usage policies", with
#     "usage policies" disambiguating from billing's "exceeded ... policy")
#   • Anthropic safety refusal ("prompt was flagged by ... safety system")
#   • OpenAI Responses content filter
_CONTENT_POLICY_BLOCKED_PATTERNS = [
    # OpenAI Codex (#18028) — message may arrive without an HTTP status
    "flagged for possible cybersecurity risk",
    "trusted access for cyber",
    # OpenAI moderation — chat completions / responses
    "violates our usage policies",
    "violates openai's usage policies",
    "your request was flagged by",
    # Anthropic safety system
    "prompt was flagged by our safety",
    "responses cannot be generated due to safety",
    # Generic content-filter wording seen on Azure / OpenAI Responses.
    # ``content_filter`` (underscore) is the OpenAI-standard error/finish
    # token surfaced verbatim by their SDKs when a request is blocked.
    # ``responsibleaipolicyviolation`` is Azure OpenAI's error code.
    # Deliberately NOT matching the space variant ("content filter") — it
    # appears in benign config descriptions and tooltip text that providers
    # echo back; the underscore form is provider-specific enough.
    "content_filter",
    "responsibleaipolicyviolation",
    # Zettlab CN content-moderation gateway. ``moderation_input_blocked`` is
    # the error code the gateway returns (HTTP 400) when the mainland-China
    # compliance scan rejects the prompt or an attached image; the paired
    # ``content_policy_violation`` is its error ``type``. Both tokens are
    # verbatim from our own gateway, so they cannot collide with a provider's
    # billing/auth/format strings. Without them the 400 falls through to the
    # status-based default and surfaces as ``provider_bad_request``, which
    # tells the user nothing about why the message was refused.
    "moderation_input_blocked",
    "content_policy_violation",
    # MiniMax output-layer safety filter. The error string is surfaced
    # verbatim by MiniMax SDK / OpenAI-compatible endpoints, usually in the
    # form "output new_sensitive (1027)" when the model's *output* (often a
    # large tool-call argument block) trips the upstream safety filter and
    # the SSE stream is truncated mid-flight. ``new_sensitive`` is the
    # filter name and is narrow enough that billing / format / auth error
    # strings will not collide. See #32421.
    "new_sensitive",
]

# Identifies a Zettlab moderation-GATEWAY verdict (mainland-China green-cip), as
# opposed to a general provider content-policy refusal. A gateway block is a
# compliance decision that must never fail over: every cloud model sits behind
# the same gateway (identical verdict) and a user-configured custom model does
# not, so failover would answer the very content the gateway just rejected.
# General provider refusals (OpenAI usage policy, Codex cyber, Anthropic safety,
# MiniMax new_sensitive) must stay failover-eligible, so they are excluded.
#
# A gateway verdict is recognized ONLY by the gateway's own error shapes — never
# by the ai-proxy route alone, and never by refusal text a custom provider could
# emit:
#   1. ``moderation_input_blocked`` — the gateway's own unambiguous error code
#      (any route; a robust secondary for call sites that don't thread the flag).
#   2. ``via_moderation_gateway`` AND the STRUCTURED ``error.type`` equalling
#      ``content_policy_violation`` — on the verified ai-proxy route the gateway's
#      generic ``code="400"`` shape carries this as its error ``type``. Matched on
#      the parsed ``error.type`` field via an exact compare, NOT as a substring of
#      the flattened haystack: the token appearing in some provider's ``error.code``
#      or message text must not count. The route flag is also REQUIRED so a custom
#      endpoint that merely reuses the token is not matched, and so an UPSTREAM
#      model's own safety refusal passed through the proxy (``content_filter`` /
#      "flagged by our safety system" / ``new_sensitive``, all in
#      _CONTENT_POLICY_BLOCKED_PATTERNS) stays failover-eligible.
# A localized ``内容不合规`` message is deliberately NOT a signal on its own.
# (PR #299 review; AGENTS.md HR2/HR3 — do not act on unverifiable attribution.)
_MODERATION_GATEWAY_CODE = "moderation_input_blocked"
_MODERATION_GATEWAY_GENERIC_TYPE = "content_policy_violation"


def _is_moderation_gateway_block(
    policy_haystack: str, error_type: str, via_moderation_gateway: bool
) -> bool:
    """True only for a Zettlab moderation-gateway verdict — its unambiguous
    ``moderation_input_blocked`` code, or (on the verified ai-proxy route) its
    generic-code shape identified by the STRUCTURED ``error.type`` equalling
    ``content_policy_violation``. Never the route alone (an upstream model refusal
    also traverses the proxy), never a localized message, and never the generic
    type appearing merely as a code/message substring. ``policy_haystack`` is the
    lowered message+code+type string used for content-policy pattern matching;
    ``error_type`` is the parsed ``error.type`` (see _error_type_of)."""
    if _MODERATION_GATEWAY_CODE in policy_haystack:
        return True
    return via_moderation_gateway and error_type == _MODERATION_GATEWAY_GENERIC_TYPE

# Auth patterns (non-status-code signals)
_AUTH_PATTERNS = [
    "invalid api key",
    "invalid_api_key",
    "gateway_auth_failed",
    "authentication",
    "unauthorized",
    "forbidden",
    "invalid token",
    "token expired",
    "token revoked",
    "access denied",
]

# Anthropic thinking block signature patterns
_THINKING_SIG_PATTERNS = [
    "signature",  # Combined with "thinking" check
]

# Message-string patterns that indicate a provider-side timeout even when
# the exception type is generic (e.g. RuntimeError from a local shim that
# wraps a subprocess timeout).  Checked before the type-based transport
# heuristics so custom-provider "timed out" errors don't fall through to
# Provider empty-response advisories (OpenRouter / nano-gpt / similar).
# Checked before context-overflow matching because the advisory text often
# mentions "max_tokens" as a possible cause, which historically sat in
# _CONTEXT_OVERFLOW_PATTERNS and sent healthy sessions into a compression
# death spiral ending in "Cannot compress further".
_EMPTY_PROVIDER_RESPONSE_PATTERNS = [
    "returned an empty response",
    "empty response despite retries",
    "provider returned an empty response",
    "model returning empty responses",
    "empty response stream",
]

# the unknown bucket and get misreported as empty responses.
_TIMEOUT_MESSAGE_PATTERNS = [
    "timed out",
    "turn timed out",
    "request timed out",
    "deadline exceeded",
    "operation timed out",
    "upstream timed out",
]

# Transport error type names
_TRANSPORT_ERROR_TYPES = frozenset({
    "ReadTimeout", "ConnectTimeout", "PoolTimeout",
    "ConnectError", "RemoteProtocolError",
    "ConnectionError", "ConnectionResetError",
    "ConnectionAbortedError", "BrokenPipeError",
    "TimeoutError", "ReadError",
    "ServerDisconnectedError",
    # SSL/TLS transport errors — transient mid-stream handshake/record
    # failures that should retry rather than surface as a stalled session.
    # ssl.SSLError subclasses OSError (caught by isinstance) but we list
    # the type names here so provider-wrapped SSL errors (e.g. when the
    # SDK re-raises without preserving the exception chain) still classify
    # as transport rather than falling through to the unknown bucket.
    "SSLError", "SSLZeroReturnError", "SSLWantReadError",
    "SSLWantWriteError", "SSLEOFError", "SSLSyscallError",
    # OpenAI SDK errors (not subclasses of Python builtins)
    "APIConnectionError",
    "APITimeoutError",
})

#: errno values that mean "the network failed", as opposed to "a file
#: operation failed".  Both raise ``OSError``, so ``isinstance(error,
#: OSError)`` cannot tell them apart — and treating the whole family as
#: transport reported ``FileNotFoundError`` as a network timeout, which is
#: worse than not classifying it at all: the user goes and checks their
#: network.  The scope of the fix has to be exactly the scope of the defect,
#: so narrow by errno rather than dropping ``OSError`` entirely (dropping it
#: would stop retrying genuine socket failures).
_NETWORK_ERRNOS = frozenset({
    errno.ECONNRESET, errno.ECONNREFUSED, errno.ECONNABORTED,
    errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH,
    errno.ENETDOWN, errno.ENETRESET, errno.EPIPE, errno.ENOTCONN,
    errno.EHOSTDOWN, errno.EADDRNOTAVAIL,
})

#: ``ValueError``/``OSError`` subclasses that look like our own bug but are
#: not.  Every entry here was paid for by someone else's incident — do not
#: drop one because a newer predicate "should" cover it:
#:   · ``json.JSONDecodeError`` — truncated/corrupt upstream response body
#:     (routing layer, cut stream). Retryable. (#14782)
#:   · ``UnicodeEncodeError``   — handled by the surrogate-sanitisation path.
#:   · ``ssl.SSLError``         — inherits OSError *and* ValueError through
#:     the MRO, so any coarse "ValueError ⇒ local bug" test misfires on a
#:     TLS transport failure.
_NOT_OUR_BUG_TYPES: tuple = (
    json.JSONDecodeError,
    UnicodeEncodeError,
    ssl.SSLError,
)


@functools.lru_cache(maxsize=1)
def _deliberate_local_failure_types() -> tuple:
    """本地抛出、但**有意面向用户**的失败类型 —— ⛔ 不是我们的 bug。

    ``hermes_cli.auth.AuthError`` 是 ``RuntimeError`` 的子类,既没有状态码也
    没有响应体,按纯证据判据会被收成「服务内部异常」——**而它恰恰是最需要
    原文的一条**:「凭据已过期,运行 `hermes auth openai`」。收掉它就等于把
    用户唯一能照做的那句话换成一个参考编号。

    ⚠️ 惰性 + 缓存导入:``hermes_cli.auth`` 会拉起 ``agent.credential_persistence``
    等一串模块,在本模块顶层导入会拖慢启动、也可能成环。这条只在错误路径上跑。
    """
    out: list = []
    try:
        from hermes_cli.auth import AuthError
        out.append(AuthError)
    except Exception:  # pragma: no cover - 纯 SDK 用法下 CLI 层可能缺席
        pass
    return tuple(out)


def _is_network_oserror(error: Exception) -> bool:
    """True for socket-level ``OSError``s, False for file-level ones."""
    if not isinstance(error, OSError):
        return False
    # Name-resolution failures are network problems, but they cannot be
    # recognised by errno: `socket.gaierror` carries an `EAI_*` code
    # (EAI_NONAME == 8), a numbering space unrelated to the `E*` errnos
    # below — errno 8 is ENOEXEC there.  Without this branch a DNS blip
    # during a network switch is reported as "internal error, contact
    # support" when one retry would have fixed it.
    if isinstance(error, (socket.gaierror, socket.herror)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return getattr(error, "errno", None) in _NETWORK_ERRNOS


def _has_upstream_evidence(error: Exception, status_code, body) -> bool:
    """Did this failure actually come from the provider?

    ⭐ The question is *not* "do I recognise this exception's name" — that
    predicate is open-ended, which is exactly why the existing
    ``is_local_validation_error`` in conversation_loop enumerated
    ``ValueError``/``TypeError`` and missed ``AttributeError``.  Adding one
    more name would miss ``NameError`` next time.

    Evidence is closed: an HTTP status, a response body, a known transport
    type name, or a network errno.  Nothing else can have originated
    upstream, whatever the exception happens to be called.
    """
    # 包装者显式声明的出身排在最前:重新包装会把 status / body / cause 三样
    # 证据一起抹掉,而抹掉证据的正是我们自己的代码。⛔ 让它落到下面的链式
    # 取证,结果必然是「无证据 ⇒ 我们的 bug ⇒ 不重试不 fallback」。
    if _declares_upstream_origin(error):
        return True
    if status_code is not None:
        return True
    if isinstance(body, dict) and body:
        return True
    # Provider SDKs routinely re-wrap the underlying transport failure
    # (Gemini raises GeminiAPIError from httpx.ConnectError, and it is not
    # alone).  Looking only at the outermost exception reported those as our
    # own bug — "internal error, contact support", no retry — for what is
    # actually a DNS or connection failure.
    #
    # Walk the cause chain exactly like `_extract_status_code` /
    # `_extract_error_body` already do in this same module: max depth 5,
    # `__cause__` then `__context__`, stop on None or self-reference.  Only
    # the *evidence check* differs; the traversal is theirs verbatim.
    #
    # NB: the mere presence of a `__cause__` proves nothing — each link is
    # tested on its own merits, so a wrapper around our own AttributeError
    # still classifies as internal.
    current: Any = error
    for _ in range(5):
        if type(current).__name__ in _TRANSPORT_ERROR_TYPES:
            return True
        if _is_network_oserror(current):
            return True
        if isinstance(current, _NOT_OUR_BUG_TYPES):
            return True
        _deliberate = _deliberate_local_failure_types()
        if _deliberate and isinstance(current, _deliberate):
            return True
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return False


# Server disconnect patterns (no status code, but transport-level).
# These are the "ambiguous" patterns — a plain connection close could be
# transient transport hiccup OR server-side context overflow rejection
# (common when the API gateway disconnects instead of returning an HTTP
# error for oversized requests).  A large session + one of these patterns
# triggers the context-overflow-with-compression recovery path.
_SERVER_DISCONNECT_PATTERNS = [
    "server disconnected",
    "peer closed connection",
    "connection reset by peer",
    "connection was closed",
    "network connection lost",
    "unexpected eof",
    "incomplete chunked read",
]

# SSL certificate verification failures — deterministic, NOT transient.
#
# A failed certificate chain (TLS-inspecting corporate proxy, missing
# custom CA in the trust store, expired certificate, self-signed cert)
# fails identically on every retry. Burning the retry budget before
# surfacing the error hides the actionable fix from the user for minutes.
# Inspired by Claude Code v2.1.199 (July 2026), which made SSL certificate
# errors fail immediately with a fix hint instead of retrying.
#
# Must be checked BEFORE _SSL_TRANSIENT_PATTERNS — "certificate verify
# failed" messages usually also contain "[SSL:" which would otherwise
# match the transient list and retry forever.
_SSL_CERT_VERIFY_PATTERNS = [
    "certificate verify failed",       # Python ssl module canonical text
    "certificate_verify_failed",       # OpenSSL error token
    "unable to get local issuer certificate",
    "self-signed certificate",
    "self signed certificate",
    "certificate has expired",
    "hostname mismatch, certificate is not valid",
    "unable to verify the first certificate",  # Node/undici phrasing (MCP bridges)
]

# SSL/TLS transient failure patterns — intentionally distinct from
# _SERVER_DISCONNECT_PATTERNS above.
#
# An SSL alert mid-stream is almost always a transport-layer hiccup
# (flaky network, mid-session TLS renegotiation failure, load balancer
# dropping the connection) — NOT a server-side context overflow signal.
# So we want the retry path but NOT the compression path; lumping these
# into _SERVER_DISCONNECT_PATTERNS would trigger unnecessary (and
# expensive) context compression on any large-session SSL hiccup.
#
# The OpenSSL library constructs error codes by prepending a format string
# to the uppercased alert reason; OpenSSL 3.x changed the separator
# (e.g. `SSLV3_ALERT_BAD_RECORD_MAC` → `SSL/TLS_ALERT_BAD_RECORD_MAC`),
# which silently stopped matching anything explicit.  Matching on the
# stable substrings (`bad record mac`, `ssl alert`, `tls alert`, etc.)
# survives future OpenSSL format churn without code changes.
_SSL_TRANSIENT_PATTERNS = [
    # Space-separated (human-readable form, Python ssl module, most SDKs)
    "bad record mac",
    "ssl alert",
    "tls alert",
    "ssl handshake failure",
    "tlsv1 alert",
    "sslv3 alert",
    # Underscore-separated (OpenSSL error code tokens, e.g.
    # `ERR_SSL_SSL/TLS_ALERT_BAD_RECORD_MAC`, `SSLV3_ALERT_BAD_RECORD_MAC`)
    "bad_record_mac",
    "ssl_alert",
    "tls_alert",
    "tls_alert_internal_error",
    # Python ssl module prefix, e.g. "[SSL: BAD_RECORD_MAC]"
    "[ssl:",
]


# ── Classification pipeline ─────────────────────────────────────────────

def classify_api_error(
    error: Exception,
    *,
    provider: str = "",
    model: str = "",
    approx_tokens: int = 0,
    context_length: int = 200000,
    num_messages: int = 0,
    via_moderation_gateway: bool = False,
) -> ClassifiedError:
    """Classify an API error into a structured recovery recommendation.

    ``via_moderation_gateway`` is a verifiable origin flag the caller sets when
    the request was routed through the local ai-proxy → Zettlab moderation
    gateway; it makes a content-policy block a non-failover compliance verdict
    regardless of the error's code/type shape. See _is_moderation_gateway_block.

    Priority-ordered pipeline:
      1. Special-case provider-specific patterns (thinking sigs, tier gates)
      2. HTTP status code + message-aware refinement
      3. Error code classification (from body)
      4. Message pattern matching (billing vs rate_limit vs context vs auth)
      5. SSL/TLS transient alert patterns → retry as timeout
      6. Server disconnect + large session → context overflow
      7. Transport error heuristics
      8. Fallback: unknown (retryable with backoff)

    Args:
        error: The exception from the API call.
        provider: Current provider name (e.g. "openrouter", "anthropic").
        model: Current model slug.
        approx_tokens: Approximate token count of the current context.
        context_length: Maximum context length for the current model.

    Returns:
        ClassifiedError with reason and recovery action hints.
    """
    status_code = _extract_status_code(error)
    error_type = type(error).__name__
    # Copilot/GitHub Models RateLimitError may not set .status_code; force 429
    # so downstream rate-limit handling (classifier reason, pool rotation,
    # fallback gating) fires correctly instead of misclassifying as generic.
    if status_code is None and error_type == "RateLimitError":
        status_code = 429
    body = _extract_error_body(error)
    error_code = _extract_error_code(body)

    # Build a comprehensive error message string for pattern matching.
    # str(error) alone may not include the body message (e.g. OpenAI SDK's
    # APIStatusError.__str__ returns the first arg, not the body).  Append
    # the body message so patterns like "try again" in 402 disambiguation
    # are detected even when only present in the structured body.
    #
    # Also extract metadata.raw — OpenRouter wraps upstream provider errors
    # inside {"error": {"message": "Provider returned error", "metadata":
    # {"raw": "<actual error JSON>"}}} and the real error message (e.g.
    # "context length exceeded") is only in the inner JSON.
    _raw_msg = str(error).lower()
    _body_msg = ""
    _metadata_msg = ""
    if isinstance(body, dict):
        _err_obj = body.get("error", {})
        if isinstance(_err_obj, dict):
            _body_msg = str(_err_obj.get("message") or "").lower()
            # Parse metadata.raw for wrapped provider errors
            _metadata = _err_obj.get("metadata", {})
            if isinstance(_metadata, dict):
                _raw_json = _metadata.get("raw") or ""
                if isinstance(_raw_json, str) and _raw_json.strip():
                    try:
                        import json
                        _inner = json.loads(_raw_json)
                        if isinstance(_inner, dict):
                            _inner_err = _inner.get("error", {})
                            if isinstance(_inner_err, dict):
                                _metadata_msg = str(_inner_err.get("message") or "").lower()
                    except (json.JSONDecodeError, TypeError):
                        pass
        if not _body_msg:
            _body_msg = str(body.get("message") or "").lower()
    # Combine all message sources for pattern matching
    parts = [_raw_msg]
    if _body_msg and _body_msg not in _raw_msg:
        parts.append(_body_msg)
    if _metadata_msg and _metadata_msg not in _raw_msg and _metadata_msg not in _body_msg:
        parts.append(_metadata_msg)
    error_msg = " ".join(parts)
    if not error_code:
        error_code = _extract_error_code_from_text(error_msg)
    provider_lower = (provider or "").strip().lower()
    model_lower = (model or "").strip().lower()

    def _result(reason: FailoverReason, **overrides) -> ClassifiedError:
        defaults = {
            "reason": reason,
            "status_code": status_code,
            "provider": provider,
            "model": model,
            "message": _extract_message(error, body),
            "provider_error_code": error_code,
        }
        defaults.update(overrides)
        return ClassifiedError(**defaults)

    # ── 1. Provider-specific patterns (highest priority) ────────────

    # Provider content-policy / safety-filter block. The provider has made a
    # deterministic refusal decision about THIS prompt — retrying unchanged
    # just reproduces the same refusal and burns paid attempts. Must run
    # before status-based classification so a 400 safety block isn't
    # downgraded to a generic ``format_error`` and a status-less block
    # (OpenAI Codex SDK can raise without one) isn't left in the retryable
    # ``unknown`` bucket. See issue #18028.
    # ``error_code`` is searched alongside the message because a gateway may
    # carry the machine token only in ``error.code`` and put a localized,
    # pattern-free sentence in ``error.message`` (the Zettlab CN moderation
    # gateway returns code=moderation_input_blocked with message="内容不合规").
    # ``error.type`` is added separately because ``_extract_error_code`` cannot
    # be relied on to surface it: it reads ``code or type``, and a generic but
    # truthy ``code`` (the CN gateway sends ``"400"`` on some paths) short-
    # circuits the ``or`` and is then dropped by the ``!= "400"`` guard, so the
    # type is never consulted again. Without it the frame falls through to
    # ``format_error``/``provider_bad_request``: the client never sees
    # ``content_blocked``, and the request may be retried on a fallback model
    # even though the compliance gateway already refused it.
    _policy_haystack = f"{error_msg} {(error_code or '').lower()} {_error_type_of(body)}"
    if any(p in _policy_haystack for p in _CONTENT_POLICY_BLOCKED_PATTERNS):
        # A moderation-GATEWAY verdict never fails over (compliance), and this
        # holds unconditionally — it is not gated on the env switch, so an OTA
        # that never sets the flag still fails toward compliance instead of
        # routing the rejected prompt to a fallback model. A general provider
        # refusal stays failover-eligible unless a deployment opts out via
        # content_policy_fallback_disabled(). See _is_moderation_gateway_block.
        _gateway_moderation = _is_moderation_gateway_block(
            _policy_haystack, _error_type_of(body), via_moderation_gateway
        )
        return _result(
            FailoverReason.content_policy_blocked,
            retryable=False,
            should_fallback=(
                False
                if _gateway_moderation
                else not content_policy_fallback_disabled()
            ),
        )

    # Anthropic thinking block recovery (400).  Two distinct failure modes,
    # same recovery (strip all reasoning_details and retry without thinking
    # blocks — see the thinking_signature handler in conversation_loop.py):
    #   1. Signature mismatch: a thinking block is signed against the full
    #      turn content; any upstream mutation (context compression, session
    #      truncation, message merging) invalidates the signature.
    #      Pattern: "signature" + "thinking".
    #   2. Frozen-block mutation: Anthropic rejects any change to the
    #      thinking/redacted_thinking blocks in the *latest* assistant
    #      message — "`thinking` or `redacted_thinking` blocks in the latest
    #      assistant message cannot be modified. These blocks must remain as
    #      they were in the original response."  This carries no "signature"
    #      token, so the original pattern missed it and the turn hard-aborted
    #      as a non-retryable client error instead of self-healing.
    #      Pattern: "thinking" + ("cannot be modified" | "must remain as they were").
    # Don't gate on provider — OpenRouter proxies Anthropic errors, so the
    # provider may be "openrouter" even though the error is Anthropic-specific.
    # The combined patterns are unique enough.
    if (
        status_code == 400
        and "thinking" in error_msg
        and (
            "signature" in error_msg
            or "cannot be modified" in error_msg
            or "must remain as they were" in error_msg
        )
    ):
        return _result(
            FailoverReason.thinking_signature,
            retryable=True,
            should_compress=False,
        )

    # Anthropic long-context tier gate (429 "extra usage" + "long context")
    if (
        status_code == 429
        and "extra usage" in error_msg
        and "long context" in error_msg
    ):
        return _result(
            FailoverReason.long_context_tier,
            retryable=True,
            should_compress=True,
        )

    # Anthropic OAuth subscription rejects the 1M-context beta header.
    # Observed error body: "The long context beta is not yet available for
    # this subscription." Returned as HTTP 400 from native Anthropic when
    # the subscription doesn't include 1M context, even though the request
    # carries ``anthropic-beta: context-1m-2025-08-07``. The recovery path
    # in run_agent.py rebuilds the Anthropic client with the beta stripped
    # and retries once. Pattern is narrow enough that it won't collide with
    # the 429 tier-gate pattern above (different status, different phrase).
    if (
        status_code == 400
        and "long context beta" in error_msg
        and "not yet available" in error_msg
    ):
        return _result(
            FailoverReason.oauth_long_context_beta_forbidden,
            retryable=True,
            should_compress=False,
        )

    # llama.cpp's ``json-schema-to-grammar`` converter (used by its OAI
    # server to build GBNF tool-call parsers) rejects regex escape classes
    # like ``\d``/``\w``/``\s`` and most ``format`` values. MCP servers
    # routinely emit ``"pattern": "\\d{4}-\\d{2}-\\d{2}"`` for date/phone/
    # email params. llama.cpp surfaces this as HTTP 400 with one of a few
    # recognizable phrases; on match we strip ``pattern``/``format`` from
    # ``self.tools`` in the retry loop and retry once. Cloud providers are
    # unaffected — they accept these keywords and we never hit this branch.
    if (
        status_code == 400
        and (
            "error parsing grammar" in error_msg
            or "json-schema-to-grammar" in error_msg
            or (
                "unable to generate parser" in error_msg
                and "template" in error_msg
            )
        )
    ):
        return _result(
            FailoverReason.llama_cpp_grammar_pattern,
            retryable=True,
            should_compress=False,
        )

    # xAI Grok subscription entitlement errors.
    #
    # xAI returns "You have either run out of available resources or do not
    # have an active Grok subscription" through two distinct code paths:
    #
    #   • HTTP 403 — status_code is set; _classify_by_status (step 2) routes
    #     it to FailoverReason.auth correctly, and _is_entitlement_failure
    #     then prevents the credential-refresh loop.
    #
    #   • SSE ``type=error`` frame — surfaced as _StreamErrorEvent with
    #     status_code=None.  _classify_by_status is skipped entirely, and
    #     "grok subscription" / "out of available resources" appear in none
    #     of the message-pattern lists below.  Without this guard the error
    #     falls through to FailoverReason.unknown (retryable=True), burning
    #     max_retries before the agent stops — and _is_entitlement_failure
    #     is never called because it only runs under FailoverReason.auth.
    #
    # Both X Premium+ and SuperGrok subscribers hit this path when their
    # subscription tier does not cover the requested model or feature.
    if (
        "do not have an active grok subscription" in error_msg
        or ("out of available resources" in error_msg and "grok" in error_msg)
    ):
        return _result(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    # ── 2. HTTP status code classification ──────────────────────────

    if status_code is not None:
        classified = _classify_by_status(
            status_code, error_msg, error_code, body,
            provider=provider_lower, model=model_lower,
            approx_tokens=approx_tokens, context_length=context_length,
            num_messages=num_messages,
            result_fn=_result,
        )
        if classified is not None:
            return classified

    # Local MoA streaming compatibility errors are adapter-shape bugs, not a
    # provider outage. Falling back to another model would silently switch the
    # user's selected MoA route to a single-model answer (#55933 follow-up).
    if provider_lower == "moa" and (
        "'types.SimpleNamespace' object is not iterable" in str(error)
        or "'types.SimpleNamespace' object has no attribute 'index'" in str(error)
    ):
        return _result(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=False,
        )

    # Local MoA config drift is deterministic: a persisted session can retain
    # a preset name that was later renamed/deleted. Retrying the same lookup
    # cannot recover and makes a clear config error look like an API outage.
    from agent.errors import MoAPresetNotFoundError

    if isinstance(error, MoAPresetNotFoundError):
        return _result(FailoverReason.model_not_found, retryable=False)

    # ── 3. Error code classification ────────────────────────────────

    if error_code:
        classified = _classify_by_error_code(error_code, error_msg, _result)
        if classified is not None:
            return classified

    # ── 4. Message pattern matching (no status code) ────────────────

    classified = _classify_by_message(
        error_msg, error_type,
        approx_tokens=approx_tokens,
        context_length=context_length,
        result_fn=_result,
    )
    if classified is not None:
        return classified

    # ── 5. SSL certificate verification failures → fail fast ────────
    # A broken certificate chain (TLS-inspecting proxy, missing custom CA,
    # expired/self-signed cert) is deterministic for the host — every retry
    # reproduces the identical handshake failure. Fail immediately with
    # actionable guidance instead of burning the retry budget first.
    # Checked BEFORE the transient-SSL patterns: cert-verify messages also
    # contain "[ssl:" which would otherwise match the transient list.
    # Inspired by Claude Code v2.1.199 (July 2026).
    if any(p in error_msg for p in _SSL_CERT_VERIFY_PATTERNS):
        return _result(
            FailoverReason.ssl_cert_verification,
            retryable=False,
            should_fallback=False,
        )

    # ── 5b. SSL/TLS transient errors → retry as timeout (not compression) ──
    # SSL alerts mid-stream are transport hiccups, not server-side context
    # overflow signals.  Classify before the disconnect check so a large
    # session doesn't incorrectly trigger context compression when the real
    # cause is a flaky TLS handshake.  Also matches when the error is
    # wrapped in a generic exception whose message string carries the SSL
    # alert text but the type isn't ssl.SSLError (happens with some SDKs
    # that re-raise without chaining).
    if any(p in error_msg for p in _SSL_TRANSIENT_PATTERNS):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 6. Server disconnect + large session → context overflow ─────
    # Must come BEFORE generic transport error catch — a disconnect on
    # a large session is more likely context overflow than a transient
    # transport hiccup.  Without this ordering, RemoteProtocolError
    # always maps to timeout regardless of session size.

    is_disconnect = any(p in error_msg for p in _SERVER_DISCONNECT_PATTERNS)
    if is_disconnect and not status_code:
        # Reasoning-model override: a transport disconnect on a reasoning
        # model is much more likely the upstream proxy idle-killing a
        # long thinking stream than a true context overflow — even on
        # large sessions.  The default disconnect+large-session routing
        # below would otherwise send the user into the compression
        # branch (should_compress=True) and silently delete
        # conversation history on a phantom context-length error.
        # Reasoning models have multi-minute thinking phases that
        # routinely exceed the cloud gateway's idle window (NVIDIA
        # NIM ~120s — first-party repro at NVIDIA/NemoClaw#4846;
        # OpenAI worker / Anthropic stream-idle similar).  The
        # per-reasoning-model stale-timeout floor in
        # agent/reasoning_timeouts.py raises the stale-detector
        # threshold to tolerate long thinking, so a true
        # transport-layer failure here is recoverable via the retry
        # path — not via context compression.  Reclassify as timeout.
        # (Part 1 of Fixes #52310.)
        from agent.reasoning_timeouts import get_reasoning_stale_timeout_floor
        if get_reasoning_stale_timeout_floor(model) is not None:
            return _result(FailoverReason.timeout, retryable=True)
        # Absolute token/message-count thresholds are only a proxy for smaller
        # context windows.  Large-context sessions can have hundreds of
        # messages while still being far below their actual token budget.
        is_large = approx_tokens > context_length * 0.6 or (
            context_length <= 256000 and (approx_tokens > 120000 or num_messages > 200)
        )
        if is_large:
            return _result(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return _result(FailoverReason.timeout, retryable=True)

    # ── 7b. Stale-call circuit breaker → failover immediately ──────
    # _check_stale_giveup() in agent/chat_completion_helpers.py raises a
    # RuntimeError when the provider has been unresponsive for N
    # consecutive stale attempts (default 5).  The error is NOT a transport
    # timeout — the circuit breaker fires *before* any network call to avoid
    # an indefinite stall.  Without this classification the RuntimeError
    # falls through to FailoverReason.unknown (retryable=True), which burns
    # all max_retries against the same dead provider (each retry hitting the
    # circuit breaker instantly with zero network overhead) before fallback
    # is attempted.  Classify as non-retryable + should_fallback so the
    # retry loop activates the next fallback provider on the first hit.
    if (
        error_type == "RuntimeError"
        and "consecutive stale attempts" in error_msg
        and "aborting this call" in error_msg
    ):
        return _result(
            FailoverReason.timeout,
            retryable=False,
            should_fallback=True,
        )

    # ── 8. Transport / timeout heuristics ───────────────────────────

    # A bare ``isinstance(error, OSError)`` used to stand here.  It also
    # swallowed FileNotFoundError / PermissionError / IsADirectoryError,
    # which then surfaced as network timeouts — a *confident but wrong*
    # verdict, worse than no verdict at all because it sends the user off
    # to check their connection.  Narrowed to socket errnos;
    # ConnectionError / TimeoutError stay whitelisted by type inside the
    # helper, so genuine socket failures keep retrying exactly as before.
    if error_type in _TRANSPORT_ERROR_TYPES or _is_network_oserror(error):
        return _result(FailoverReason.timeout, retryable=True)

    # ── 9. No upstream evidence at all → our own bug ────────────────
    #
    # Deterministic per request: the same code path raises the same
    # exception every time.  Retrying burns the budget and then tells the
    # user "temporary problem, try again later" — an action that cannot
    # possibly work.  Fail fast so the surface can say what actually
    # happened and hand over a reference instead.
    if not _has_upstream_evidence(error, status_code, body):
        return _result(FailoverReason.internal_error, retryable=False)

    # ── 10. Fallback: unknown ───────────────────────────────────────

    return _result(FailoverReason.unknown, retryable=True)


# ── Status code classification ──────────────────────────────────────────

def _classify_by_status(
    status_code: int,
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on HTTP status code with message-aware refinement."""

    if status_code == 401:
        # Not retryable on its own — credential pool rotation and
        # provider-specific refresh (Codex, Anthropic, Nous) run before
        # the retryability check in run_agent.py.  If those succeed, the
        # loop `continue`s.  If they fail, retryable=False ensures we
        # hit the client-error abort path (which tries fallback first).
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 403:
        # OpenRouter 403 "key limit exceeded" is actually billing. Other
        # providers also use 403 for account-plan or credit exhaustion.
        if (
            (
                provider == "xai-oauth"
                and error_code.lower() == _XAI_SPENDING_LIMIT_ERROR_CODE
            )
            or "key limit exceeded" in error_msg
            or "spending limit" in error_msg
            or any(p in error_msg for p in _BILLING_PATTERNS)
        ):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_fallback=True,
        )

    if status_code == 402:
        return _classify_402(error_msg, result_fn)

    if status_code == 404:
        # Nous API currently surfaces HA/NAS credit depletion as a paid model
        # becoming unavailable on the Free Tier, returned as 404 rather than
        # 402. Treat that as entitlement/billing exhaustion, not a missing
        # model, so the retry loop can show credit/top-up guidance.
        if any(p in error_msg for p in _BILLING_PATTERNS):
            return result_fn(
                FailoverReason.billing,
                retryable=False,
                should_rotate_credential=True,
                should_fallback=True,
            )
        # OpenRouter policy-block 404 — distinct from "model not found".
        # The model exists; the user's account privacy setting excludes the
        # only endpoint serving it. Falling back to another provider won't
        # help (same account setting applies).  The error body already
        # contains the fix URL, so just surface it.
        if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
            return result_fn(
                FailoverReason.provider_policy_blocked,
                retryable=False,
                should_fallback=False,
            )
        if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
            return result_fn(
                FailoverReason.model_not_found,
                retryable=False,
                should_fallback=True,
            )
        # Generic 404 with no "model not found" signal — could be a wrong
        # endpoint path (common with local llama.cpp / Ollama / vLLM when
        # the URL is slightly misconfigured), a proxy routing glitch, or
        # a transient backend issue.  Classifying these as model_not_found
        # silently falls back to a different provider and tells the model
        # the model is missing, which is wrong and wastes a turn.  Treat
        # as unknown so the retry loop surfaces the real error instead.
        return result_fn(
            FailoverReason.unknown,
            retryable=True,
        )

    if status_code == 413:
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    if status_code == 429:
        # Already checked long_context_tier above. Some providers (notably
        # Z.AI / Zhipu) reuse HTTP 429 for server-wide overload — same status
        # code as a true per-credential rate limit, but the credential is
        # valid and the correct recovery is "back off and retry the same key",
        # NOT "rotate the credential" (which exhausts the pool while the
        # endpoint is still busy, and does nothing for a single-key user).
        # Disambiguate on the error body so an overload 429 takes the
        # transient-overload path instead of burning the pool. (#14038)
        if any(p in error_msg for p in _OVERLOADED_PATTERNS):
            return result_fn(
                FailoverReason.overloaded,
                retryable=True,
            )
        # Distinguish an OpenRouter-aggregator upstream 429 (an upstream model
        # like DeepSeek rate-limited OpenRouter's aggregate traffic) from an
        # account-level 429 (the user's key is actually throttled). OpenRouter
        # wraps upstream errors with the outer message "Provider returned
        # error" — the user's key is healthy, so marking it exhausted / rotating
        # is wrong and burns the key for ~24min. Fall back to a different model.
        if _is_openrouter_upstream_error(body, provider):
            upstream_provider = _extract_upstream_provider_name(body)
            ctx = {"upstream_provider": upstream_provider} if upstream_provider else {}
            return result_fn(
                FailoverReason.upstream_rate_limit,
                retryable=True,
                should_rotate_credential=False,
                should_fallback=True,
                error_context=ctx,
            )
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if status_code == 400:
        return _classify_400(
            error_msg, error_code, body,
            provider=provider, model=model,
            approx_tokens=approx_tokens,
            context_length=context_length,
            num_messages=num_messages,
            result_fn=result_fn,
        )

    if status_code in {500, 502}:
        # Some OpenAI-compatible gateways return request-validation errors
        # with a 5xx status (codex.nekos.me returns 502 for unknown/
        # unsupported parameters). These are deterministic — every retry
        # gets the identical rejection — so the generic "5xx → retryable
        # server_error" rule turns one bad request into a retry flood.
        # Detect the unambiguous request-validation signals (in either the
        # message text or the structured error code) and fail fast.
        if (
            any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS)
            or error_code.lower() in {"invalid_request_error", "unknown_parameter",
                                      "unsupported_parameter"}
        ):
            return result_fn(
                FailoverReason.format_error,
                retryable=False,
                should_fallback=True,
            )
        # Some local inference servers (notably llama.cpp / llama-server)
        # report context overflow with an HTTP 500 instead of the standard
        # 400/413. The request-validation guard above already ran, so any
        # remaining explicit context-overflow signal routes into the
        # compression-and-retry path (mirroring _classify_400) instead of
        # blind server_error retries that exhaust and drop the turn.
        # Empty-response advisories that mention "max_tokens" must not enter
        # that compression path.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.server_error, retryable=True)

    if status_code in {503, 529}:
        # Same overflow-as-5xx variant (server busy / model-load OOM, or a
        # Cloudflare/Tailscale hop relabeling the status). Route explicit
        # overflow bodies into compression; otherwise treat as transient
        # overload and retry.
        if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
            return result_fn(
                FailoverReason.server_error,
                retryable=True,
                should_compress=False,
            )
        if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
            return result_fn(
                FailoverReason.context_overflow,
                retryable=True,
                should_compress=True,
            )
        return result_fn(FailoverReason.overloaded, retryable=True)

    # 408 Request Timeout — a transient timing failure the server itself flags
    # as safe to retry (RFC 9110 §15.5.9), not a malformed request. Commonly
    # emitted by reverse proxies sitting in front of self-hosted backends
    # (llama.cpp / Ollama / vLLM) when a long generation outruns the proxy's
    # request-read window. Route to the dedicated ``timeout`` reason (rebuild
    # client + retry) instead of falling through to the generic 4xx bucket
    # below, which would abort the turn on a retry-safe error the same way it
    # aborts a 400 Bad Request.
    if status_code == 408:
        return result_fn(FailoverReason.timeout, retryable=True)

    # Other 4xx — non-retryable
    if 400 <= status_code < 500:
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Other 5xx — retryable
    if 500 <= status_code < 600:
        return result_fn(FailoverReason.server_error, retryable=True)

    return None


def _classify_402(error_msg: str, result_fn) -> ClassifiedError:
    """Disambiguate 402: billing exhaustion vs transient usage limit.

    The key insight from OpenClaw: some 402s are transient rate limits
    disguised as payment errors.  "Usage limit, try again in 5 minutes"
    is NOT a billing problem — it's a periodic quota that resets.
    """
    # Check for transient usage-limit signals first
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)

    if has_usage_limit and has_transient_signal:
        # Transient quota — treat as rate limit, not billing
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Confirmed billing exhaustion
    return result_fn(
        FailoverReason.billing,
        retryable=False,
        should_rotate_credential=True,
        should_fallback=True,
    )


def _classify_400(
    error_msg: str,
    error_code: str,
    body: dict,
    *,
    provider: str,
    model: str,
    approx_tokens: int,
    context_length: int,
    num_messages: int = 0,
    result_fn,
) -> ClassifiedError:
    """Classify 400 Bad Request — context overflow, format error, or generic."""

    # Multimodal tool content rejected from 400.  Must be checked BEFORE
    # image_too_large because the recovery is different (strip image parts
    # from tool messages, mark the model as no-list-tool-content for the
    # rest of the session) and BEFORE context_overflow because some of the
    # patterns ("text is not set") are ambiguous in isolation but become
    # specific when combined with a 400 on a request known to contain
    # multimodal tool content.
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large from 400 (Anthropic's 5 MB per-image check fires this way).
    # Must be checked BEFORE context_overflow because messages can trip both
    # patterns ("exceeds" + "image") and image-shrink is a cheaper recovery.
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Invalid encrypted reasoning replay blob (OpenAI Responses API).  Must be
    # checked BEFORE context_overflow because some surfaces emit messages that
    # contain context-like phrasing ("encrypted content … could not be
    # verified") which could otherwise trip the context_overflow heuristics.
    # ``error_msg`` is lowercased upstream — match accordingly.
    error_code_lower = (error_code or "").lower()
    if (
        error_code_lower == "invalid_encrypted_content"
        or "invalid_encrypted_content" in error_msg
        or (
            "encrypted content for item" in error_msg
            and "could not be verified" in error_msg
        )
        or "could not decrypt the provided encrypted_content" in error_msg
    ):
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    # Request-validation errors (unsupported / unknown parameter) MUST be
    # checked BEFORE context_overflow.  A GPT-5 model rejecting max_tokens
    # returns:
    #   "Unsupported parameter: 'max_tokens' is not supported with this model.
    #    Use 'max_completion_tokens' instead."
    # That string contains the literal substring "max_tokens", which historically
    # sat in _CONTEXT_OVERFLOW_PATTERNS — so without this guard the 400 is
    # misclassified as context_overflow, routed into the compression loop,
    # re-sent with the same bad parameter, and ends in "Cannot compress
    # further".  These errors are deterministic (every retry gets the identical
    # rejection), so classify as a non-retryable format_error and fall back.
    #
    # NOTE: we deliberately do NOT key off the generic ``invalid_request_error``
    # code here — OpenAI stamps that same code on genuine context-overflow 400s,
    # so matching it would mis-route real overflows away from compression. The
    # unambiguous signals are the explicit "unsupported/unknown parameter"
    # message text and the specific parameter-level error codes.
    if (
        any(p in error_msg for p in _REQUEST_VALIDATION_PATTERNS
            if p != "invalid_request_error")
        or error_code_lower in {"unknown_parameter", "unsupported_parameter"}
    ):
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Malformed message array (empty-content assistant stub, etc.). Must be
    # checked BEFORE context_overflow: the input can be tiny, so the generic
    # "400 + large session" heuristic would otherwise mis-route it into the
    # compression loop and thrash until "Cannot compress further" on every
    # retry (the request is unchanged, so compression cannot fix it). This is
    # a deterministic request-shape rejection — fail fast as a non-retryable
    # format_error and fall back. Checked against the message text AND the
    # structured error code, since proxies (litellm/Bedrock) surface the
    # signal in errorCode=INVALID_REQUEST_BODY.
    if (
        any(p in error_msg for p in _INVALID_MESSAGE_BODY_PATTERNS)
        or error_code_lower == "invalid_request_body"
    ):
        logger.warning(
            "Malformed message array 400 (invalid request body) classified as "
            "format_error, NOT context overflow — failing fast + falling back "
            "instead of entering the compression loop. This usually means an "
            "empty-content assistant stub is in the transcript; num_messages=%s "
            "approx_tokens=%s. error=%.200s",
            num_messages, approx_tokens, error_msg,
        )
        return result_fn(
            FailoverReason.format_error,
            retryable=False,
            should_fallback=True,
        )

    # Empty-provider-response advisories must not enter compression. They
    # often mention "max_tokens" as a possible cause and used to match the
    # bare overflow pattern, then thrash compress until "Cannot compress
    # further" on an otherwise healthy session (custom endpoints / nano-gpt).
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow from 400
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Some providers return model-not-found as 400 instead of 404 (e.g. OpenRouter).
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Some providers return rate limit / billing errors as 400 instead of 429/402.
    # Check these patterns before falling through to format_error.
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Generic 400 + large session → probable context overflow
    # Anthropic sometimes returns a bare "Error" message when context is too large
    err_body_msg = ""
    if isinstance(body, dict):
        err_obj = body.get("error", {})
        if isinstance(err_obj, dict):
            err_body_msg = str(err_obj.get("message") or "").strip().lower()
        # Responses API (and some providers) use flat body: {"message": "..."}
        if not err_body_msg:
            err_body_msg = str(body.get("message") or "").strip().lower()
        # litellm / Bedrock proxies use a custom shape: {"errorMessage": "...",
        # "errorCode": "...", "errorArgs": {"reason": "..."}}.  Without these
        # keys err_body_msg stays "" and a long, descriptive rejection is
        # wrongly treated as a "generic" (bare) error below, which — on a
        # large session — mis-routes into the compression loop.  Recognize
        # them so the is_generic heuristic sees the real message length.
        if not err_body_msg:
            err_body_msg = str(body.get("errorMessage") or "").strip().lower()
        if not err_body_msg:
            _args = body.get("errorArgs")
            if isinstance(_args, dict):
                err_body_msg = str(_args.get("reason") or "").strip().lower()
    is_generic = len(err_body_msg) < 30 or err_body_msg in {"error", ""}
    # Absolute token/message-count thresholds are only a proxy for smaller
    # context windows.  Large-context sessions can have many messages while
    # still being far below their actual token budget.
    is_large = approx_tokens > context_length * 0.4 or (
        context_length <= 256000 and (approx_tokens > 80000 or num_messages > 80)
    )

    if is_generic and is_large:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Non-retryable format error
    return result_fn(
        FailoverReason.format_error,
        retryable=False,
        should_fallback=True,
    )


# ── Error code classification ───────────────────────────────────────────

def _classify_by_error_code(
    error_code: str, error_msg: str, result_fn,
) -> Optional[ClassifiedError]:
    """Classify by structured error codes from the response body."""
    code_lower = error_code.lower()

    if code_lower in {"resource_exhausted", "throttled", "rate_limit_exceeded"}:
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
        )

    if code_lower in _BILLING_ERROR_CODES:
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    if code_lower in {"model_not_found", "model_not_available", "invalid_model"}:
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    if code_lower in {"context_length_exceeded", "max_tokens_exceeded"}:
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    if code_lower == "invalid_encrypted_content":
        return result_fn(
            FailoverReason.invalid_encrypted_content,
            retryable=True,
            should_fallback=False,
        )

    return None


# ── Message pattern classification ──────────────────────────────────────

def _classify_by_message(
    error_msg: str,
    error_type: str,
    *,
    approx_tokens: int,
    context_length: int,
    result_fn,
) -> Optional[ClassifiedError]:
    """Classify based on error message patterns when no status code is available."""

    # Payload-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _PAYLOAD_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.payload_too_large,
            retryable=True,
            should_compress=True,
        )

    # Multimodal tool content patterns (from message text when no status_code)
    if any(p in error_msg for p in _MULTIMODAL_TOOL_CONTENT_PATTERNS):
        return result_fn(
            FailoverReason.multimodal_tool_content_unsupported,
            retryable=True,
        )

    # Image-too-large patterns (from message text when no status_code)
    if any(p in error_msg for p in _IMAGE_TOO_LARGE_PATTERNS):
        return result_fn(
            FailoverReason.image_too_large,
            retryable=True,
        )

    # Usage-limit patterns need the same disambiguation as 402: some providers
    # surface "usage limit" errors without an HTTP status code.  A transient
    # signal ("try again", "resets at", …) means it's a periodic quota, not
    # billing exhaustion.
    has_usage_limit = any(p in error_msg for p in _USAGE_LIMIT_PATTERNS)
    if has_usage_limit:
        has_transient_signal = any(p in error_msg for p in _USAGE_LIMIT_TRANSIENT_SIGNALS)
        if has_transient_signal:
            return result_fn(
                FailoverReason.rate_limit,
                retryable=True,
                should_rotate_credential=True,
                should_fallback=True,
            )
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Overloaded / server-busy patterns — must come BEFORE the rate_limit and
    # billing checks so that a message-only "overloaded" (no 503/529 status,
    # e.g. some Anthropic-compatible proxies) classifies as a transient
    # overload (backoff + retry) instead of falling through to `unknown` or
    # incorrectly triggering credential rotation.
    if any(p in error_msg for p in _OVERLOADED_PATTERNS):
        return result_fn(
            FailoverReason.overloaded,
            retryable=True,
        )

    # Billing patterns
    if any(p in error_msg for p in _BILLING_PATTERNS):
        return result_fn(
            FailoverReason.billing,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Rate limit patterns
    if any(p in error_msg for p in _RATE_LIMIT_PATTERNS):
        return result_fn(
            FailoverReason.rate_limit,
            retryable=True,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Empty-provider-response advisories (often mention "max_tokens") must
    # retry without compression — see the matching 400-path guard above.
    if any(p in error_msg for p in _EMPTY_PROVIDER_RESPONSE_PATTERNS):
        return result_fn(
            FailoverReason.server_error,
            retryable=True,
            should_compress=False,
        )

    # Context overflow patterns
    if any(p in error_msg for p in _CONTEXT_OVERFLOW_PATTERNS):
        return result_fn(
            FailoverReason.context_overflow,
            retryable=True,
            should_compress=True,
        )

    # Auth patterns
    # Auth errors should NOT be retried directly — the credential is invalid and
    # retrying with the same key will always fail.  Set retryable=False so the
    # caller triggers credential rotation (should_rotate_credential=True) or
    # provider fallback rather than an immediate retry loop.
    if any(p in error_msg for p in _AUTH_PATTERNS):
        return result_fn(
            FailoverReason.auth,
            retryable=False,
            should_rotate_credential=True,
            should_fallback=True,
        )

    # Provider policy-block (aggregator-side guardrail) — check before
    # model_not_found so we don't mis-label as a missing model.
    if any(p in error_msg for p in _PROVIDER_POLICY_BLOCKED_PATTERNS):
        return result_fn(
            FailoverReason.provider_policy_blocked,
            retryable=False,
            should_fallback=False,
        )

    # Model not found patterns
    if any(p in error_msg for p in _MODEL_NOT_FOUND_PATTERNS):
        return result_fn(
            FailoverReason.model_not_found,
            retryable=False,
            should_fallback=True,
        )

    # Timeout message patterns — generic exception types (e.g. RuntimeError)
    # raised by local shims or custom providers that internally wrap a
    # subprocess/HTTP timeout.  Classified as transport timeout so the retry
    # loop rebuilds the client instead of treating the turn as an empty
    # model response.
    if any(p in error_msg for p in _TIMEOUT_MESSAGE_PATTERNS):
        return result_fn(FailoverReason.timeout, retryable=True)

    return None


# ── Helpers ─────────────────────────────────────────────────────────────

def _extract_status_code(error: Exception) -> Optional[int]:
    """Walk the error and its cause chain to find an HTTP status code."""
    current = error
    for _ in range(5):  # Max depth to prevent infinite loops
        code = getattr(current, "status_code", None)
        if isinstance(code, int):
            return code
        # Some SDKs use .status instead of .status_code
        code = getattr(current, "status", None)
        if isinstance(code, int) and 100 <= code < 600:
            return code
        # Walk cause chain
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return None


def _extract_error_body(error: Exception) -> dict:
    """Extract the structured error body from an SDK exception or its cause chain."""
    current = error
    for _ in range(5):  # Match _extract_status_code() traversal depth.
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            return body
        # Some errors have .response.json()
        response = getattr(current, "response", None)
        if response is not None:
            try:
                json_body = response.json()
                if isinstance(json_body, dict):
                    return json_body
            except Exception:
                pass
        cause = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
        if cause is None or cause is current:
            break
        current = cause
    return {}


_ERROR_CODE_TEXT_PATTERNS = (
    "insufficient_credits",
    "insufficient_quota",
    "billing_not_active",
    "payment_required",
    "rate_limit_exceeded",
    "resource_exhausted",
    "model_not_found",
    "model_not_available",
    "invalid_model",
    "context_length_exceeded",
    "max_tokens_exceeded",
    "invalid_api_key",
    "invalid_request_error",
)


def _error_type_of(body) -> str:
    """Return ``error.type`` lowercased, or "" when absent.

    Kept separate from :func:`_extract_error_code` on purpose — that function
    answers "what is this error's code", and its ``code or type`` fallback is
    right for that question. Content-policy matching needs the type even when a
    code exists, which is a different question.
    """
    if not isinstance(body, dict):
        return ""
    err = body.get("error")
    if not isinstance(err, dict):
        return ""
    t = err.get("type")
    return t.strip().lower() if isinstance(t, str) else ""


def _extract_error_code(body: dict) -> str:
    """Extract an error code string from the response body."""
    if not body:
        return ""

    def _code_from_payload(payload) -> str:
        """Extract a code/type from a nested error payload dict (defensive)."""
        if not isinstance(payload, dict):
            return ""
        payload_error = payload.get("error", {})
        if isinstance(payload_error, dict):
            nested = payload_error.get("code") or payload_error.get("type") or ""
            if isinstance(nested, str) and nested.strip() and nested.strip() != "400":
                return nested.strip()
        code = payload.get("code") or payload.get("error_code") or ""
        if isinstance(code, (str, int)):
            text = str(code).strip()
            if text and text != "400":
                return text
        return ""

    error_obj = body.get("error", {})
    if isinstance(error_obj, dict):
        code = error_obj.get("code") or error_obj.get("type") or ""
        if isinstance(code, str) and code.strip() and code.strip() != "400":
            return code.strip()

        # Some providers wrap the real JSON error body as a string inside
        # error.message — peek into it for a nested code (e.g. Responses API
        # surfaces ``invalid_encrypted_content`` this way).
        message = error_obj.get("message")
        if isinstance(message, str) and message.strip().startswith("{"):
            import json
            try:
                inner = json.loads(message)
            except (json.JSONDecodeError, TypeError):
                inner = None
            nested_code = _code_from_payload(inner)
            if nested_code:
                return nested_code

    # Top-level code
    code = body.get("code") or body.get("error_code") or body.get("errorCode") or ""
    if isinstance(code, (str, int)):
        text = str(code).strip()
        if text and text != "400":
            return text
    if isinstance(error_obj, str) and error_obj.strip():
        return error_obj.strip()
    return ""


def _extract_error_code_from_text(text: str) -> str:
    lower = text.lower()
    for pattern in _ERROR_CODE_TEXT_PATTERNS:
        if pattern in lower:
            return pattern
    return ""


def _extract_message(error: Exception, body: dict) -> str:
    """Extract the most informative error message."""
    # Try structured body first
    if body:
        error_obj = body.get("error", {})
        if isinstance(error_obj, dict):
            msg = error_obj.get("message", "")
            if isinstance(msg, str) and msg.strip():
                return msg.strip()[:500]
        msg = body.get("message", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
        # litellm / Bedrock proxy shape: {"errorMessage": "...",
        # "errorArgs": {"reason": "..."}}.
        msg = body.get("errorMessage", "")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()[:500]
        args = body.get("errorArgs")
        if isinstance(args, dict):
            reason = args.get("reason", "")
            if isinstance(reason, str) and reason.strip():
                return reason.strip()[:500]
    # Fallback to str(error)
    return str(error)[:500]


def _is_openrouter_upstream_error(body: Any, provider: str) -> bool:
    """Detect OpenRouter's aggregator-wrapped upstream provider errors.

    OpenRouter returns errors from upstream model providers (DeepSeek,
    Anthropic, etc.) wrapped with the outer message "Provider returned error"
    and the real error nested in ``metadata.raw``. This signal means the
    user's OpenRouter key is healthy — the upstream provider is the one that
    failed — so credential rotation is the wrong recovery.
    """
    if not isinstance(body, dict):
        return False
    provider_lower = (provider or "").strip().lower()
    err = body.get("error")
    if not isinstance(err, dict):
        return False
    outer_msg = str(err.get("message") or "").strip().lower()
    if outer_msg != "provider returned error":
        return False
    # Require either the explicit OpenRouter provider OR the metadata shape
    # that only OpenRouter produces (metadata.raw / metadata.provider_name).
    if provider_lower == "openrouter":
        return True
    metadata = err.get("metadata")
    if isinstance(metadata, dict) and (
        "raw" in metadata or "provider_name" in metadata
    ):
        return True
    return False


def _extract_upstream_provider_name(body: Any) -> Optional[str]:
    """Pull the upstream provider name out of OpenRouter's error metadata."""
    if not isinstance(body, dict):
        return None
    err = body.get("error")
    if not isinstance(err, dict):
        return None
    metadata = err.get("metadata")
    if not isinstance(metadata, dict):
        return None
    name = metadata.get("provider_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None
