"""重新包装 provider 原话时,出身必须由包装者显式声明。

**缺陷(Codex PR #339 P1-4)**:`_normalize_codex_response` 收到
`status="failed"` 的 Responses 响应,把 provider 在 `error` 里给的原因包成一个
**裸 `RuntimeError`** —— 没有 `status_code`、没有响应体、没有 `__cause__`。
分类器第 9 步于是判「四样证据都没有 ⇒ 我们自己的 bug」:

  · `internal_error` + `retryable=False` ⇒ **既不重试,也不 fallback**
  · provider 那句真正的解释被压成「服务内部异常」

一次上游侧的瞬时失败,就这样变成用户这条消息的**永久失败**,而且用户看不到
任何能照做的信息。

⭐ 这是 P1-1 的**反方向**:P1-1 是把我们的文本当成上游的(泄漏),这条是把
上游的文本当成我们的(吞掉)。**同一个判据的两端,一起修。**

⛔ 修法不用「函数名 / 模块名白名单」——那是开集,下一个包装点还会漏。
判据落在**包装的人自己声明**上:谁抹掉了出身,谁负责重新写明。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.error_classifier import (
    UPSTREAM_ORIGIN_ATTR,
    FailoverReason,
    classify_api_error,
    declare_upstream_origin,
    error_text_is_ours,
)


class TestDeclarationIsHonouredByBothContracts:
    """一个声明,两个契约都要认:恢复(重试/fallback)与展示(文本归属)。"""

    def test_bare_runtimeerror_without_declaration_is_ours(self):
        """阳性对照:没有声明的裸 RuntimeError 仍判「我们的」——⛔ 判据不是恒真。

        ⚠️ 文案刻意**不含**任何会被第 4 步文本模式认走的词(timeout / not found /
        invalid api key …)。第一版这里写的是 "agent step timed out",于是先撞上
        timeout 模式、根本走不到第 9 步 —— 阳性对照自己踩了这份文件在讲的那个坑。
        """
        exc = RuntimeError("profile resolution produced no runtime kwargs")
        assert error_text_is_ours(exc) is True
        assert classify_api_error(exc).reason == FailoverReason.internal_error

    #: 一句**不含任何第 4 步文本模式**的 provider 失败原因。
    #:
    #: ⚠️ 这行常量是逆改验证逼出来的:第一版用的是
    #: ``"Responses API failed: overloaded"``,而 "overloaded" 自己就是一条文本
    #: 模式 ⇒ 有没有声明都返回 ``overloaded``,断言 ``!= internal_error`` **恒真**。
    #: 拆掉整个判据它都不红 —— 一条出生即空转的断言,冒充了恢复侧的保护。
    #: ⭐ 想验第 9 步,输入就必须真的能走到第 9 步。
    PATTERN_FREE = "the model produced an unusable result"

    def test_declared_wrapper_is_retryable_not_internal(self):
        """恢复侧:声明了出身 ⇒ ⛔ 不许判成 internal_error 而掐掉重试/fallback。"""
        plain = classify_api_error(RuntimeError(self.PATTERN_FREE))
        marked = classify_api_error(declare_upstream_origin(RuntimeError(self.PATTERN_FREE)))

        # 负对照:同一句话没有声明时,必须仍然落到第 9 步判「我们的 bug」。
        # 少了这半边,上面那条断言就无法区分「判据生效」和「判据根本没被问到」。
        assert plain.reason == FailoverReason.internal_error, (
            f"负对照失效:这句话根本走不到第 9 步(实得 {plain.reason})—— 换一句"
        )
        assert marked.reason != FailoverReason.internal_error, (
            "上游失败被判成我们的 bug ⇒ 用户这条消息直接失败,连 fallback 都不试"
        )
        assert marked.retryable is True, "掐掉了重试 ⇒ 一次瞬时上游故障变永久失败"

    def test_declared_wrapper_text_reaches_the_user(self):
        """展示侧:provider 的原话必须逐字送达,⛔ 不许压成「服务内部异常」。"""
        exc = declare_upstream_origin(RuntimeError(self.PATTERN_FREE))
        assert error_text_is_ours(exc) is False
        # 负对照:同一句话没有声明 ⇒ 判「我们的」。⛔ 判据不是恒假。
        assert error_text_is_ours(RuntimeError(self.PATTERN_FREE)) is True

    def test_declaration_returns_the_same_exception_object(self):
        """盖戳不许换对象 —— 调用点写的是 `raise declare_upstream_origin(...)`。"""
        original = RuntimeError("x")
        assert declare_upstream_origin(original) is original
        assert getattr(original, UPSTREAM_ORIGIN_ATTR) is True

    def test_declaration_does_not_leak_onto_sibling_exceptions(self):
        """盖在实例上,⛔ 不许污染同类的其它异常。"""
        declared = declare_upstream_origin(RuntimeError("declared"))
        plain = RuntimeError("plain")

        assert error_text_is_ours(declared) is False
        assert error_text_is_ours(plain) is True


class TestCodexResponsesAdapterDeclaresAtTheWrapSite:
    """真实调用点:`status="failed"` 那条路径必须盖戳。"""

    def _failed_response(self, message: str):
        return SimpleNamespace(
            status="failed",
            output=[SimpleNamespace(type="message", role="assistant",
                                    status="completed", content=[])],
            error=SimpleNamespace(message=message, code="server_error"),
            incomplete_details=None,
            output_text=None,
        )

    def test_failed_response_raises_a_declared_error(self):
        from agent.codex_responses_adapter import _normalize_codex_response

        # ⚠️ 同样刻意避开第 4 步的文本模式(见 PATTERN_FREE 那条注释):
        # 用 "overloaded" 之类的词,分类断言会恒真、验不到这个包装点。
        reason = "the model produced an unusable result"
        with pytest.raises(RuntimeError) as caught:
            _normalize_codex_response(self._failed_response(reason))

        exc = caught.value
        assert reason in str(exc), "provider 给的原因没被带出来"
        assert error_text_is_ours(exc) is False, (
            "provider 的原因被判成我们的 bug ⇒ 不重试、不 fallback、原话还被抹掉"
        )
        classified = classify_api_error(exc)
        assert classified.reason != FailoverReason.internal_error
        assert classified.retryable is True, "掐掉了重试与 fallback"

    def test_no_output_items_is_also_declared_upstream(self):
        """🔴 **同一模式的第三个兄弟** —— 上一轮我判「不修,含取舍」,已被推翻。

        我的理由是「改它要动重试预算与 fallback 语义」。⭐ 错在**前提**:
        用出身声明根本不需要动那两样 —— 声明说的是「**这次失败来自上游**」,
        分类器照既有规则去决定重试与 fallback,⛔ 没有新语义被引入。

        「响应里没有 output」是**纯粹的上游属性**:到达该 raise 时,
        ``output_text`` 回退与 ``content_filter`` 两条都已排除;请求构造错误会
        得到 4xx(自带 ``status_code``,走不到这里)。
        ⇒ ⛔ **不存在「非上游」的成因,不需要分两类。**

        ⚠️ 与 ``status=="failed"`` 分支的区别:那条带着 provider 的**原话**,
        这条的文本是我们写的 —— 但出身声明说的是「失败来自上游」,
        ⛔ 不是「这段话是上游写的」,两者不冲突。
        """
        from agent.codex_responses_adapter import _normalize_codex_response

        empty = SimpleNamespace(
            status="completed", output=[], error=None,
            incomplete_details=None, output_text=None,
        )
        with pytest.raises(RuntimeError) as caught:
            _normalize_codex_response(empty)

        exc = caught.value
        assert "no output items" in str(exc)
        assert getattr(exc, UPSTREAM_ORIGIN_ATTR, False) is True, (
            "上游协议异常没有声明出身 ⇒ 会被判成我们的 bug,不重试不 fallback"
        )
        assert classify_api_error(exc).reason != FailoverReason.internal_error
        assert classify_api_error(exc).retryable is True

    def test_the_auxiliary_stream_sibling_is_declared_too(self):
        """兄弟三:``auxiliary_client`` 的「流没有终止帧」同形,⛔ 不许漏。"""
        import inspect
        from pathlib import Path

        import agent.auxiliary_client as ac

        src = Path(inspect.getfile(ac)).read_text()
        i = src.index("did not return a final response")
        window = src[max(0, i - 400):i + 200]
        assert "declare_upstream_origin" in window, (
            "第三个兄弟没有声明出身 —— 一个模式只修两个实例 = 兄弟调用点没跟上"
        )


class TestPreservedBehaviour:
    """🔴 上一轮修对的东西,这一轮⛔ 不许弄坏。"""

    def test_status_code_bearing_error_is_still_upstream(self):
        exc = RuntimeError("rate limited")
        exc.status_code = 429
        assert error_text_is_ours(exc) is False

    def test_network_oserror_is_still_upstream(self):
        import socket

        assert error_text_is_ours(socket.gaierror(8, "nodename nor servname provided")) is False

    def test_user_actionable_local_error_is_still_delivered(self):
        from agent.errors import MoAPresetNotFoundError

        assert error_text_is_ours(MoAPresetNotFoundError("run: hermes moa list")) is False
