"""HTTP 边界:我们自己的 bug ⛔ 不许当成上游问题甩给用户。

**现场(实测)**:`/v1/chat/completions` 在 handler 直接抛异常时返回

    status 500
    Internal server error: 'CustomProfile' object has no attribute
    'supports_prompt_cache_key' at /volume1/private/profile/config.json

用户照着这条做不了任何事,而这本来是一句「服务内部异常 + 参考编号」。

## ⚠️ 本门的判据换过两次,两次都值得记下来

**第一版(静态 AST)**:扫 `except ... as X` 绑定,断言 X 不进 `_openai_error` /
`_redact_api_error_text`。扫出 15 处 —— 但那 15 处里有相当一部分是**对的**
(异常来自上游时,provider 自己的解释就该逐字透传)。AST 分不出运行时那个 X
是谁的错。⇒ 判据过宽,照它改会弄坏上游错误的可诊断性。

**第二版(共享清洗器 + 正则)**:在 `_redact_api_error_text` 里抹掉「绝对路径」
和「`'X' object has no attribute 'y'`」。两个判据都挑错了维度:

- **措辞是开集** —— 认得 attribute error,就认不得 `KeyError` / `NameError` /
  `module 'x' has no attribute 'y'`;补一个漏下一个。
- **「以 `/` 开头、两段以上」分不开主机路径和上游 URL / HTTP 路由** ——
  实测把 `... at https://api.example.com/v1/chat/completions` 改成了
  `... at https:/<path>`,**把 provider 的解释改坏了**。这条是修复自己引入的
  新缺陷:修复的作用域大过了缺陷的作用域。

**第三版(本文件)**:判据回到**捕获点**,问的是**有没有上游证据**
(HTTP 状态码 / 响应体 / 已知传输类型名 / 网络 errno)—— 这是闭集,与异常叫
什么名字无关。四样都没有 ⇒ 只可能是我们自己的代码。
"""
from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path

import pytest

from gateway.platforms import api_server as _api

#: 实测从 500 响应里原样漏出来的那一串
_REAL_LEAK = (
    "'CustomProfile' object has no attribute 'supports_prompt_cache_key' "
    "at /volume1/private/profile/config.json"
)


class _UpstreamError(Exception):
    """带 HTTP 状态码的 provider 异常 —— 上游证据的最小形态。"""

    def __init__(self, message: str, status_code: int = 429):
        super().__init__(message)
        self.status_code = status_code


# ───────── ① 缺陷本体:我们自己的 bug ⇒ 安全文案 + 可兑现的编号 ─────────
#
# ⭐ 这几种异常是**样本**,⛔ 不是判据。判据是「没有上游证据」——
# 换成任何一个没列在这里的异常类型,结论都必须一样。

@pytest.mark.parametrize("exc, forbidden", [
    (AttributeError(_REAL_LEAK), ["CustomProfile", "supports_prompt_cache_key", "/volume1"]),
    (KeyError("profile_id"), ["profile_id"]),
    (NameError("name '_cfg' is not defined"), ["_cfg", "is not defined"]),
    (AttributeError("module 'agent.providers' has no attribute 'load'"),
     ["agent.providers", "no attribute"]),
    (TypeError("unsupported operand type(s) for +: 'int' and 'NoneType'"),
     ["unsupported operand", "NoneType"]),
    (ZeroDivisionError("division by zero"), ["division by zero"]),
    (RuntimeError("dictionary changed size during iteration"), ["changed size"]),
])
def test_our_own_bug_never_crosses_the_boundary(exc, forbidden, caplog):
    with caplog.at_level(logging.ERROR):
        out = _api._boundary_error_text("unit-test", exc)

    for sym in forbidden:
        assert sym not in out, f"{sym!r} 仍出现在给客户端的文本里:{out}"
    assert _api.INTERNAL_ERROR_USER_TEXT in out, out
    assert len(out) < 200, f"安全文案不该很长:{out}"

    # 🔴 原文必须进日志,且编号必须能把两边对上 —— 否则「参考编号」兑现不了
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "correlation_id=" in logged, f"没有 correlation id:{logged}"
    ref = logged.split("correlation_id=")[1].split(")")[0].strip()
    assert ref and ref in out, f"日志里的编号 {ref!r} 没出现在给用户的文本里:{out}"
    assert any(r.exc_info for r in caplog.records), "没有把原始异常记进日志(exc_info)"


def test_the_logged_traceback_is_the_real_one_not_the_ambient_one():
    """⛔ `exc_info=True` 取的是「当前正在处理的异常」。

    在 except 块之外调用就会记成 `NoneType: None` —— 文案照样安全,
    但 traceback 没了,而 traceback 正是那个编号唯一能兑现的东西。
    """
    records = []

    class _Grab(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Grab()
    _api.logger.addHandler(handler)
    try:
        exc = AttributeError(_REAL_LEAK)
        try:
            raise exc  # 给它一条真的 __traceback__
        except AttributeError:
            pass
        _api._internal_error_text("unit-test", exc)   # ⛔ 此处不在 except 块里
    finally:
        _api.logger.removeHandler(handler)

    assert records, "没有日志记录"
    exc_info = records[-1].exc_info
    assert exc_info and exc_info[1] is exc, f"记的不是这个异常:{exc_info}"
    assert exc_info[2] is not None, "记了异常但 traceback 是空的"


# ───────── ② 🔒 P2 回归锁:上游文本逐字,URL / 路由一个字符不许动 ─────────
#
# ⭐ 全部写成**正面断言**(`== 原文`)。反面断言(`"<path>" not in out`)
# 只要把标记名写错就恒真,钉不住任何东西。

_UPSTREAM_WITH_URL = (
    "Provider rejected request at https://api.example.com/v1/chat/completions"
)


@pytest.mark.parametrize("text", [
    _UPSTREAM_WITH_URL,
    "Rate limit reached for gpt-4o. Please retry in 30s.",
    "You exceeded your current quota, please check your plan and billing details.",
    "The model `foo-1` does not exist or you do not have access to it.",
    "Provider returned error: upstream connect error or disconnect/reset before headers",
    # HTTP 路由:和主机路径长得一模一样 —— 正是「按形状判」分不开的那一对
    "POST /v1/chat/completions returned 500 from upstream",
    # 多段 URL、带查询串
    "See https://docs.provider.example/errors/rate-limit?code=429 for details",
    # 模型 slug 里的斜杠
    "The model `openrouter/anthropic/claude-3-opus` is not available on your plan.",
])
def test_upstream_text_passes_through_byte_for_byte(text):
    assert _api._redact_api_error_text(text) == text


def test_upstream_exception_keeps_its_url_through_the_boundary():
    """⭐ 端到端形态:P2 就是在这条路上被改坏的。"""
    out = _api._boundary_error_text("unit-test", _UpstreamError(_UPSTREAM_WITH_URL))
    assert out == _UPSTREAM_WITH_URL


@pytest.mark.parametrize("exc", [
    _UpstreamError("Rate limit reached for gpt-4o. Please retry in 30s.", 429),
    _UpstreamError("You exceeded your current quota.", 402),
    _UpstreamError("The model `foo-1` does not exist.", 404),
    ConnectionResetError(104, "Connection reset by peer"),
    TimeoutError("Request timed out"),
])
def test_upstream_failures_keep_their_own_explanation(exc):
    """provider 的解释告诉用户等多久 / 换哪个模型 / 去哪充值。

    为了堵泄漏把它一起收掉,比原缺陷更糟 —— 那是把可行动的提示换成
    不可行动的提示。
    """
    out = _api._boundary_error_text("unit-test", exc)
    assert _api.INTERNAL_ERROR_USER_TEXT not in out, f"上游失败被压成内部异常:{out}"
    assert str(exc) in out, out


# ───────── ③ 阳性对照:判据本身没空转 ─────────


def test_the_two_branches_really_differ():
    """⛔ 如果两条分支输出一样,上面所有断言都会因为别的原因通过。"""
    ours = _api._boundary_error_text("unit-test", AttributeError(_REAL_LEAK))
    theirs = _api._boundary_error_text("unit-test", _UpstreamError("upstream boom"))
    assert ours != theirs
    assert "参考编号" in ours and "参考编号" not in theirs


def test_classifier_failure_falls_to_the_safe_side():
    """分类器自己炸了 ⇒ 站保守侧当成我们的 bug,⛔ 不许把原文放出去。"""
    class _Hostile(Exception):
        @property
        def status_code(self):        # 让 _extract_status_code 炸在里面
            raise RuntimeError("boom")

    out = _api._boundary_error_text("unit-test", _Hostile("secret internals here"))
    assert "secret internals here" not in out, out
    assert _api.INTERNAL_ERROR_USER_TEXT in out, out


# ───────── ③b provenance × 文案形状 的 2×2(RH 第十一轮 P1-1)─────────
#
# ⭐ 上一版的样本是**对角线**的:内部样本全部避开 provider 关键词,上游样本
# 全部带状态码。于是「按文案判」和「按证据判」两种实现都能全绿 —— 门根本
# 没有分辨力。下面把另外两格补上。

_PROVIDER_WORDING = [
    "agent step timed out: /volume1/private/profile/config.json",
    "model not found at /volume1/private/state.db",
    "invalid api key path /volume1/private/creds.yaml",
    "rate limit exceeded while reading /volume1/private/cache.db",
    "context length exceeded in /volume1/private/session.db",
]


@pytest.mark.parametrize("text", _PROVIDER_WORDING)
def test_local_error_wearing_provider_wording_is_still_ours(text):
    """🔴 本地异常撞上 provider 关键词,⛔ 不许因此被当成上游失败。

    这是「分类流水线按恢复策略排序」漏出来的那一格:文本模式匹配在第 4 步、
    证据检查在第 8 步,所以 ``classify_api_error`` 会先判成 timeout /
    model_not_found,整条内部路径原样出屏。
    """
    out = _api._boundary_error_text("unit-test", RuntimeError(text))
    assert "/volume1" not in out, f"内部路径出屏了:{out}"
    assert _api.INTERNAL_ERROR_USER_TEXT in out, out


@pytest.mark.parametrize("text", _PROVIDER_WORDING)
def test_the_same_wording_from_upstream_still_passes_through(text):
    """⭐ 对照格:一模一样的文案,只要**有上游证据**就必须逐字透传。

    两格用同一批字符串 ⇒ 差别只可能来自 provenance,⛔ 不可能来自文案。
    """
    out = _api._boundary_error_text("unit-test", _UpstreamError(text, 429))
    assert out == text


def test_the_boundary_asks_who_wrote_the_text_not_whether_it_can_be_retried():
    """⛔ 门:边界必须问「**这段文本是谁写的**」,⛔ 不许问任何恢复类判据。

    ## 🔴 这道门自己出过事,值得原样记下来

    上一版断言的是 ``"has_upstream_evidence" in called`` —— 它钉的是**当时的
    实现**,不是**需求**。于是 Codex P1-1 报出「沿 cause 链取证会把我们的内部
    路径判成上游文本」之后,**这道门主动阻止了修复**:改对了它就红。
    见 [[gate-can-pin-the-bug-as-contract]] ——「一道门能把 bug 钉成契约」。

    ⭐ 正解是**换驱动方式、契约一字不动**:需求从来就是
    「⛔ 不许用回答『能不能重试』的那类判据来决定『说什么』」。
    ``classify_api_error(...).reason`` 和 ``has_upstream_evidence`` 都属于那一类
    (前者按恢复策略排序,后者沿 cause 链取证),所以两个都进禁用集;
    该调用的是与聊天出站路径**共用**的归属判据 ``error_text_is_ours``。

    ⚠️ 这仍是**开集**门(按名字查引用),只防本函数漂移,⛔ 证明不了别的路径
    没这么写。留它是因为这条判据已经被推翻过两次,复发成本很高。
    """
    src = Path(inspect.getfile(_api)).read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "_boundary_error_text"
    )
    # ⭐ 按**它调用了什么**判,⛔ 不按源码文本里出现了什么 ——
    # 第一版按文本判,被自己 docstring 里那句解释判成红(注释也算数据)。
    called = {
        getattr(n.func, "id", "") or getattr(n.func, "attr", "")
        for n in ast.walk(fn) if isinstance(n, ast.Call)
    }
    assert "error_text_is_ours" in called, f"边界不再问文本归属:{called}"

    recovery_predicates = {"classify_api_error", "has_upstream_evidence"}
    assert not (called & recovery_predicates), (
        f"边界又用回了恢复类判据 {called & recovery_predicates} —— "
        "「可不可以重试」证明不了「这段文本是谁写的」"
    )


def test_evidence_predicate_separates_the_two_diagonals():
    """阳性对照:判据本身有分辨力(⛔ 不是恒真/恒假)。"""
    from agent.error_classifier import has_upstream_evidence

    assert not has_upstream_evidence(RuntimeError(_PROVIDER_WORDING[0]))
    assert has_upstream_evidence(_UpstreamError(_PROVIDER_WORDING[0], 429))


# ───────── ③c provider-auth 旁路不许收编内部异常(P1-2)─────────


def test_non_auth_failures_are_not_relabelled_as_provider_auth(monkeypatch):
    """🔴 现场:内部 ``AttributeError`` 以 **HTTP 200** 当作 assistant 的回答
    出现在用户面前 ——「⚠️ Provider authentication failed: 'X' object has no
    attribute 'y' … /volume1/private/config.yaml」。

    根因不是「认证真失败了」,是这个 catch 把**一切**都变成了认证失败:
    ``format_runtime_provider_error`` 对非 ``AuthError`` 直接 ``str(error)``。
    """
    import hermes_cli.runtime_provider as rp

    boom = AttributeError("'CustomProfile' object has no attribute 'x' at /volume1/private/config.yaml")
    monkeypatch.setattr(rp, "resolve_runtime_provider",
                        lambda **kw: (_ for _ in ()).throw(boom))
    with pytest.raises(AttributeError) as caught:
        _api._resolve_request_runtime_agent_kwargs("openai")
    assert caught.value is boom, "内部异常被改头换面了,原样往上抛才对"


def test_real_auth_failures_keep_their_actionable_message(monkeypatch):
    """🔒 必须保持不变:真的凭据失败仍然拿到可行动的原文。"""
    import hermes_cli.runtime_provider as rp

    err = rp.AuthError("OpenAI credentials expired; run `hermes auth openai`")
    monkeypatch.setattr(rp, "resolve_runtime_provider",
                        lambda **kw: (_ for _ in ()).throw(err))
    with pytest.raises(RuntimeError) as caught:
        _api._resolve_request_runtime_agent_kwargs("openai")
    assert not isinstance(caught.value, AttributeError)
    assert "hermes auth openai" in str(caught.value), str(caught.value)


# ───────── ③d 宽 ValueError 不许把内部错误伪装成 400(P2-2)─────────


def test_response_format_400_branch_uses_a_type_not_a_text_shape():
    """🔴 契约重写(第三版)。

    前两版判据是文本形状:先「含 response_format」、再「以它开头」。两版都挡不住
    ``_run_agent`` 内部一条恰好提到该词的 ``ValueError`` —— RH 实测
    ``ValueError("response_format resolver crashed at /volume1/…")``
    被当成 400 请求校验错误、**内部路径原样回显**。

    ⇒ 判据换成**类型**这个闭集:只认 ``agent.response_format`` 有意抛出的子类。
    """
    from agent.response_format import ResponseFormatValidationError

    src = Path(inspect.getfile(_api)).read_text()
    assert 'if "response_format" in str(e):' not in src, "第一版宽匹配还在"
    assert 'if str(e).startswith("response_format"):' not in src, "第二版形状判据还在"
    assert src.count("isinstance(e, ResponseFormatValidationError)") == 2

    # 🔴 缺陷本体:内部错误即使文案以该词开头,也**不是**请求校验错误
    internal = ValueError("response_format resolver crashed at /volume1/private/x.json")
    assert not isinstance(internal, ResponseFormatValidationError)

    # 🔒 必须保持不变:真校验点抛的仍然是 400 那一类,且仍是 ValueError 子类
    #    ⇒ 既有的 `except ValueError` 调用点行为不变
    from agent import response_format as _rf

    for bad in [None, {"type": "nope"}, {"type": "json_schema", "json_schema": 1}]:
        try:
            _rf.normalize_chat_response_format(bad)  # type: ignore[attr-defined]
        except ResponseFormatValidationError as exc:
            assert isinstance(exc, ValueError), "⛔ 不再是 ValueError 子类会打断既有调用点"
        except AttributeError:
            pass  # 该 helper 名字不同,类型契约由上面的 isinstance 断言覆盖
        except ValueError:
            pass


# ───────── ④ 门:兄弟调用点全集 ─────────


#: ⚠️ **这道门是开集,明说**:它只覆盖「捕获的异常直接进 `_openai_error` /
#: `_redact_api_error_text`」这一种形状,证明不了「没有别的泄漏路径」。
#: 它的作用是**防漂移** —— 新加一个未经边界判定的 500 出口会红。
#:
#: 唯一放行的形状:调用同时带 ``param=`` 或 ``code=``,即一条结构化的
#: 400 类请求校验错误 —— 那种场合异常文本正是用户要看的东西
#: (`response_format` 不合法 / 标题非法),⛔ 不能收掉。
_CLIENT_TEXT_SINKS = {"_openai_error", "_redact_api_error_text"}
_VALIDATION_OPT_IN = {"param", "code"}


def _collect_unguarded_sites(src: str) -> list[str]:
    tree = ast.parse(src)
    lines = src.splitlines()
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def _enclosing_sink(node):
        cur = node
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, ast.Call):
                fn = cur.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
                if name == "_boundary_error_text":
                    return None                      # 已经过边界判定
                if name in _CLIENT_TEXT_SINKS:
                    if any(k.arg in _VALIDATION_OPT_IN for k in cur.keywords):
                        return None                  # 明示的 400 类校验错误
                    return cur
        return None

    bad = []
    for handler in [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]:
        if not handler.name:
            continue
        for sub in ast.walk(handler):
            if isinstance(sub, ast.Name) and sub.id == handler.name:
                if _enclosing_sink(sub) is not None:
                    bad.append(f"L{sub.lineno}: {lines[sub.lineno - 1].strip()}")
    # 同形兄弟:它不是绑定名,但一样把我们的堆栈送出去
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "format_exc":
            if _enclosing_sink(node) is not None:
                bad.append(f"L{node.lineno}: {lines[node.lineno - 1].strip()}")
    return sorted(set(bad))


def test_no_caught_exception_reaches_a_client_sink_unjudged():
    src = Path(inspect.getfile(_api)).read_text()
    bad = _collect_unguarded_sites(src)
    assert not bad, "以下捕获点未经 _boundary_error_text 判定就把异常文本送给客户端:\n" + "\n".join(bad)


def _uncovered_run_agent_calls(src: str) -> list[str]:
    """直接 ``await self._run_agent(...)`` 的调用点里,哪些不在 try 主体里。

    ⚠️ **作用域是「直接 await」这一种形状,⛔ 不是全部调用点** —— 我第一版
    写成「call 节点必须词法上落在 Try 体内」,结果误报三处:异常对协程来说
    在 **await 处**浮现,不在构造处。⭐ 又一次拿词法位置去判控制流性质,
    和被删掉的那条路径正则同一个毛病。

    两类**已核实受保护**、由本门显式排除的形状:

    · ``asyncio.ensure_future(self._run_agent(...))`` —— 失败在
      ``await agent_task`` 浮现(L6130 / L6652,两处都在 try 主体里)。
    · 嵌套 ``async def`` 里的 ``return await self._run_agent(...)`` ——
      保护在这个闭包的**调用点**上(``await _compute_response()`` 等)。

    ⇒ 这两类是本门的**开集侧**:它们靠人工核过,⛔ 门证明不了。
    """
    tree = ast.parse(src)
    lines = src.splitlines()
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node

    def _excluded(call):
        p = parent.get(call)
        # ① ensure_future / create_task 包着 —— 失败在 await 处
        if isinstance(p, ast.Call):
            name = getattr(p.func, "attr", "") or getattr(p.func, "id", "")
            if name in {"ensure_future", "create_task", "shield"}:
                return True
        # ② 嵌套函数体内 —— 保护在该函数的调用点
        cur, depth = call, 0
        while cur in parent:
            cur = parent[cur]
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                depth += 1
                if depth >= 2:
                    return True
        return False

    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "_run_agent"):
            continue
        if _excluded(call):
            continue
        cur, guarded = node, False
        while cur in parent:
            prev, cur = cur, parent[cur]
            # 只算「在 try 主体里」—— 落在 except/finally 里不算被它保护
            if isinstance(cur, ast.Try) and any(prev is s for s in cur.body):
                guarded = True
                break
        if not guarded:
            bad.append(f"L{call.lineno}: {lines[call.lineno - 1].strip()}")
    return sorted(set(bad))


def test_every_run_agent_call_sits_inside_a_try():
    """🔴 RH 指出的门盲区:枚举已有 catch 的门,**发现不了根本没有 catch**。

    现场:``POST /api/sessions/{id}/chat``(同步)没有任何 catch ⇒ aiohttp
    回一个纯文本 ``500 Server got itself in trouble``,没有 JSON、没有可行动
    提示、没有参考编号。它的流式兄弟一直是接住的 —— 又一次兄弟调用点没跟上。
    """
    src = Path(inspect.getfile(_api)).read_text()
    bad = _uncovered_run_agent_calls(src)
    assert not bad, "以下 _run_agent 调用点没有异常边界:\n" + "\n".join(bad)


def test_the_run_agent_gate_is_not_vacuous():
    naked = "async def h(self):\n    r = await self._run_agent(x=1)\n"
    assert _uncovered_run_agent_calls(naked), "门抓不住「完全没有 try」"
    wrapped = ("async def h(self):\n    try:\n        r = await self._run_agent(x=1)\n"
               "    except Exception:\n        pass\n")
    assert not _uncovered_run_agent_calls(wrapped), "门在包好之后仍报红"
    # ⛔ 落在 except 体里不算被保护
    in_handler = ("async def h(self):\n    try:\n        pass\n"
                  "    except Exception:\n        r = await self._run_agent(x=1)\n")
    assert _uncovered_run_agent_calls(in_handler), "门把 except 体里的调用误判为已保护"


def test_the_ast_gate_is_not_vacuous():
    """⛔ 出生即空转的门比没有门更坏 —— 拿一个真实形状的漏洞喂它。"""
    sample = (
        "def h():\n"
        "    try:\n"
        "        run()\n"
        "    except Exception as e:\n"
        "        return _openai_error(f'Internal server error: {e}', err_type='server_error')\n"
    )
    assert _collect_unguarded_sites(sample), "门抓不住实测漏出来的那种写法"

    fixed = sample.replace("f'Internal server error: {e}'", "_boundary_error_text('h', e)")
    assert not _collect_unguarded_sites(fixed), "门在修好之后仍然报红"

    allowed = (
        "def h():\n"
        "    try:\n"
        "        run()\n"
        "    except ValueError as e:\n"
        "        return _openai_error(str(e), param='response_format')\n"
    )
    assert not _collect_unguarded_sites(allowed), "门误伤了 400 类校验错误"


# ---------------------------------------------------------------------------
# 第四版判据(Codex PR #339 P1-1):展示只认「这段文本是谁写的」
#
# 第三版问 `has_upstream_evidence`,而它**沿 cause 链**取证 —— 回答的是
# 「这次失败能不能重试」。于是我们自己在处理传输错误时抛出的异常,沿链继承到
# 「上游证据」,内部路径原样推给了每一个接本边界的 HTTP/SSE 客户端。
# ⇒ 改问 `error_text_is_ours`,与聊天出站路径共用同一个实现。
# ---------------------------------------------------------------------------


class TestBoundaryTextAttributionIsByOuterLayer:
    """`_boundary_error_text` 必须按**外层归属**判,⛔ 不是按 cause 链。"""

    def test_our_error_raised_while_handling_a_transport_failure_is_collapsed(self):
        """🔴 Codex P1-1 的原样复现:`raise ... from ConnectionError` ⇒ ⛔ 不许出屏。"""
        from gateway.platforms import api_server

        secret_path = "/volume1/private/config.yaml"
        try:
            try:
                raise ConnectionError("connection reset by peer")
            except ConnectionError as transport:
                raise RuntimeError(f"profile load failed at {secret_path}") from transport
        except RuntimeError as exc:
            text = api_server._boundary_error_text("chat.completions", exc)

        assert secret_path not in text, "内部路径漏给了客户端"
        assert "profile load failed" not in text, "内部实现细节漏给了客户端"
        assert api_server.INTERNAL_ERROR_USER_TEXT in text
        assert "参考编号" in text, "收了文案就必须给用户一个可查询的编号"

    def test_transport_failure_itself_still_reaches_the_user_verbatim(self):
        """🔴 必须保持不变:上游自己的失败,解释要逐字到用户手里。"""
        from gateway.platforms import api_server

        exc = ConnectionError("upstream refused the connection, retry in 30s")
        text = api_server._boundary_error_text("chat.completions", exc)

        assert "retry in 30s" in text, "把上游能让用户照做的那句话抹掉了"
        assert api_server.INTERNAL_ERROR_USER_TEXT not in text

    def test_sdk_wrapping_a_transport_error_is_still_upstream(self):
        """🔴 必须保持不变:provider SDK 包一层传输错误 ⇒ 仍算上游(第 ④ 问)。"""
        from gateway.platforms import api_server

        sdk_module = type("_M", (), {})
        namespace = {"__module__": "some_provider_sdk.errors"}
        SdkError = type("SdkError", (Exception,), namespace)
        del sdk_module

        try:
            try:
                raise ConnectionError("dns failure")
            except ConnectionError as transport:
                raise SdkError("provider unreachable, try another model") from transport
        except SdkError as exc:
            text = api_server._boundary_error_text("chat.completions", exc)

        assert "try another model" in text, "把 SDK 包装的上游错误误收成内部异常"

    def test_user_actionable_local_error_is_not_collapsed(self):
        """🔴 必须保持不变:声明了面向用户的本地失败,原文必须到。"""
        from gateway.platforms import api_server
        from agent.errors import MoAPresetNotFoundError

        exc = MoAPresetNotFoundError("preset 'x' not found; run: hermes moa list")
        text = api_server._boundary_error_text("chat.completions", exc)

        assert "hermes moa list" in text, "把用户唯一能照做的那句话拿走了"

    def test_boundary_and_chat_path_share_one_implementation(self):
        """⛔ 同一个问题不许两处各写一套判据 —— 漂移就是这么来的。"""
        import agent.error_classifier as ec
        from gateway.platforms import api_server

        assert api_server.error_text_is_ours is ec.error_text_is_ours

    def test_boundary_no_longer_asks_the_recovery_predicate(self):
        """判据换过就要钉住,⛔ 不许有人改回沿链取证的那个。"""
        import inspect as _inspect
        from gateway.platforms import api_server

        src = _inspect.getsource(api_server._boundary_error_text)
        body = src.split('"""')[-1]  # 去掉 docstring:里面**故意**写着旧判据名
        assert "error_text_is_ours(" in body
        assert "has_upstream_evidence(" not in body, "退回了沿 cause 链的恢复判据"
