"""入站媒体失败**记账**门 —— ⛔ 不许静默吞掉,⛔ 不许把凭据写进日志。

═══════════════════════════════════════════════════════════════════════
本门关住哪几格 / 哪几格仍是开集(⭐ 分格声明,⛔ 不许含糊)
═══════════════════════════════════════════════════════════════════════

**关住(闭集)**
  对下面 ``_INBOUND_FETCH`` 里逐个列出的**入站媒体获取调用**,凡是包住它的
  ``try/except``:
    (a) 每个 handler 必须调 ``log_media_intake_failure`` —— 抓「静默吞掉」;
        ⭐ 这一条让「删掉调用点」的逆改必然变红。
    (b) 该 handler 里**任何** ``logger.*`` / ``print`` 调用,⛔ 不许
        ① 裸传异常绑定变量 ② 传 ``exc_info=True``
        ③ 再次传入「被喂给该获取调用的那个变量」(通常就是 url)
        —— 除非包在 ``safe_exc`` / ``safe_traceback`` / ``safe_url_for_log``
        里,或作为 ``log_media_intake_failure`` 的 ``exc=`` / ``url=`` 关键字。
        ⭐ 这一条让「还原成原始写法」的逆改必然变红。

  ⭐ (b) 的 url 判据**不靠变量名猜**,靠控制流:「谁被喂给了下载函数」。
     ⇒ 换个变量名绕不过去。

**仍是开集(⛔ 明说,不假装闭)**
  1. ``_INBOUND_FETCH`` 本身是一张**清单**。我没列进去的入站路径,本门看不见。
     兜底见 ``test_no_half_converted_handler``:它不依赖这张清单 —— 凡是**已经**
     调了 ``log_media_intake_failure`` 的 handler,同块里⛔ 不许再有裸异常/
     ``exc_info=True``(抓「改了一半」)。
  2. ``logging`` 的 ``extra=`` / f-string 拼接:f-string 里的 ``FormattedValue``
     本门会展开检查,但 ``"%s" % exc`` 这种预格式化不覆盖。
  3. **本门只看源码,零真机验证** —— 没有任何一条是在真实渠道上跑出来的。

═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

#: provider ⇒ (adapter 相对路径, 入站媒体获取调用的**闭集**, 函数级锚点)
#:
#: 「入站媒体获取调用」= 真的去把用户发来的附件字节拿回来的那个调用。
#: ⛔ 不含出站(send_*)、不含缩略图/历史回填。
#: 「函数级锚点」用于获取动作是通过局部变量间接发起的场合(Discord 的
#: ``reader()`` 就是 ``att.read`` 取出来的局部变量,按调用名匹配不到)。
_INBOUND_FETCH: dict[str, tuple[str, frozenset[str], frozenset[str]]] = {
    # ⚠️ 飞书的取件走 `self._run_blocking(self._client.im.v1.message_resource.get, …)`
    #    —— 调用名是通用的 `_run_blocking`,按调用名匹配会把全仓每个阻塞 SDK 调用
    #    都算进来。⇒ 这里只能用**函数级锚点**(实查 4316 / 4348 两个定义)。
    "feishu": (
        "plugins/platforms/feishu/adapter.py",
        frozenset(),
        frozenset({"_download_feishu_image", "_download_feishu_message_resource"}),
    ),
    "slack": (
        "plugins/platforms/slack/adapter.py",
        frozenset({"_download_slack_file", "_download_slack_file_bytes"}),
        frozenset(),
    ),
    "teams": (
        "plugins/platforms/teams/adapter.py",
        frozenset({"_fetch_attachment_bytes", "cache_image_from_url"}),
        frozenset(),
    ),
    "matrix": (
        "plugins/platforms/matrix/adapter.py",
        frozenset({"download_media"}),
        frozenset(),
    ),
    "discord": (
        "plugins/platforms/discord/adapter.py",
        frozenset(
            {"_cache_discord_image", "_cache_discord_audio", "_cache_discord_document"}
        ),
        frozenset({"_read_attachment_bytes"}),
    ),
}

_ACCOUNTING = "log_media_intake_failure"
#: 允许把「敏感值」包进去的清洗器 —— 闭集,⛔ 加新的必须同时改这里。
_SANITIZERS = frozenset({"safe_exc", "safe_traceback", "safe_url_for_log"})
_LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "exception", "critical"}
)


def _parse(rel: str) -> tuple[str, ast.Module]:
    src = (_REPO / rel).read_text()
    return src, ast.parse(src)


def _call_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return ""


def _is_log_call(node: ast.Call) -> bool:
    """是不是一条「往日志/标准输出写东西」的调用。

    ⛔ 不按 logger 变量名判(那是名字判,换个名就绕过去)——
    按**方法名属于 logging 的闭集**,外加内建 ``print``。
    """
    f = node.func
    if isinstance(f, ast.Name) and f.id == "print":
        return True
    return isinstance(f, ast.Attribute) and f.attr in _LOG_METHODS


def _walk_calls(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            yield n


def _walk_calls_stop_at_try(node: ast.AST):
    """遍历调用,但**不下钻进嵌套的 try** —— 那一层有自己的 handler。"""
    if isinstance(node, ast.Call):
        yield node
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Try):
            continue
        yield from _walk_calls_stop_at_try(child)


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _sanitized_subtrees(node: ast.AST) -> list[ast.AST]:
    """已被清洗器包住、因而豁免的子树。"""
    return [c for c in _walk_calls(node) if _call_name(c) in _SANITIZERS]


def _exposed_names(arg: ast.AST) -> set[str]:
    """实参里**没有**被清洗器包住的裸 Name 集合。"""
    exempt: set[int] = set()
    for s in _sanitized_subtrees(arg):
        for n in ast.walk(s):
            exempt.add(id(n))
    return {n.id for n in ast.walk(arg) if isinstance(n, ast.Name) and id(n) not in exempt}


def _fetched_arg_names(try_node: ast.Try, fetch: frozenset[str]) -> set[str]:
    """try body 里喂给「入站获取调用」的那些变量名 —— url 判据的来源。

    ⭐ 靠控制流,⛔ 不靠变量叫不叫 ``url``。
    """
    out: set[str] = set()
    for stmt in try_node.body:
        for call in _walk_calls_stop_at_try(stmt):
            if _call_name(call) in fetch:
                for a in list(call.args) + [k.value for k in call.keywords]:
                    out |= _names_in(a)
    return out


def _guarded_tries(
    tree: ast.Module, fetch: frozenset[str], func_anchors: frozenset[str]
) -> list[ast.Try]:
    """包住入站获取调用的 try 节点全集。"""
    found: list[ast.Try] = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Try):
            continue
        # ⭐ 只认**最内层**包住取件调用的那个 try:遇到嵌套 try 就停,
        #    否则外层的兜底 handler 会被误判成「该为这次取件记账」。
        #    (实测:slack 的 thread-root 恢复流程是 try 套 try,不停会误报。)
        hit = any(
            _call_name(c) in fetch for stmt in n.body for c in _walk_calls_stop_at_try(stmt)
        )
        found.append(n) if hit else None
    for fn in ast.walk(tree):
        if (
            isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name in func_anchors
        ):
            for n in ast.walk(fn):
                if isinstance(n, ast.Try) and n not in found:
                    found.append(n)
    return found


def _offending_args(handler: ast.ExceptHandler, forbidden: set[str]) -> list[str]:
    """handler 里所有 logging/print 调用上的违规实参描述。"""
    bad: list[str] = []
    for call in _walk_calls(handler):
        if not _is_log_call(call):
            continue
        for kw in call.keywords:
            if kw.arg == "exc_info":
                bad.append(f"line {call.lineno}: 传了 exc_info=（traceback 会重新格式化原始异常）")
        for a in list(call.args) + [k.value for k in call.keywords]:
            leaked = _exposed_names(a) & forbidden
            if leaked:
                bad.append(f"line {call.lineno}: 裸传了 {sorted(leaked)}")
    return bad


# ───────────────────────── 门 (a):⛔ 不许静默吞掉 ─────────────────────────


@pytest.mark.parametrize("provider", sorted(_INBOUND_FETCH))
def test_every_inbound_fetch_handler_accounts(provider: str) -> None:
    rel, fetch, anchors = _INBOUND_FETCH[provider]
    _src, tree = _parse(rel)
    tries = _guarded_tries(tree, fetch, anchors)
    assert tries, (
        f"{provider}: 一个包住入站获取调用的 try 都没找到 —— "
        f"⛔ 这不是「通过」,是**量具坏了**。闭集 {sorted(fetch) + sorted(anchors)} "
        f"很可能已经与 {rel} 漂移。"
    )

    missing: list[str] = []
    checked = 0
    for t in tries:
        for h in t.handlers:
            checked += 1
            has = any(_call_name(c) == _ACCOUNTING for c in _walk_calls(h))
            # 只往外抛(re-raise / return 之前不记账)也算已交代:异常继续冒泡,
            # ⛔ 不是被吞掉。判据是「有没有痕迹」,不是「有没有调某个函数」。
            reraises = any(
                isinstance(n, ast.Raise) and n.exc is None for n in ast.walk(h)
            )
            if not (has or reraises):
                missing.append(f"{rel}:{h.lineno}")
    assert checked, f"{provider}: 找到 try 却一个 handler 都没有 —— 量具坏了"
    assert not missing, (
        f"{provider}: 下列 handler 把入站媒体失败**静默吞掉**了 —— "
        f"用户发的附件没到,日志里一个字都没有:\n  " + "\n  ".join(missing)
    )


# ────────────────── 门 (b):⛔ 不许把凭据写进日志 ──────────────────


@pytest.mark.parametrize("provider", sorted(_INBOUND_FETCH))
def test_inbound_failure_logs_leak_nothing(provider: str) -> None:
    rel, fetch, anchors = _INBOUND_FETCH[provider]
    _src, tree = _parse(rel)
    tries = _guarded_tries(tree, fetch, anchors)
    assert tries, f"{provider}: 闭集与 {rel} 漂移了 —— 量具坏了"

    problems: list[str] = []
    for t in tries:
        fetched = _fetched_arg_names(t, fetch)
        for h in t.handlers:
            forbidden = set(fetched)
            if h.name:
                forbidden.add(h.name)
            for msg in _offending_args(h, forbidden):
                problems.append(f"{rel} {msg}")
    assert not problems, (
        f"{provider}: 入站媒体失败日志把**凭据面**写出去了。\n"
        "  · 各家媒体 URL 都是凭据:Teams=预授权 SharePoint `?tempauth=`、\n"
        "    Discord CDN=`?ex=&is=&hm=` 签名、Slack 公开分享=`?pub_secret=`。\n"
        "  · `httpx.HTTPStatusError.__str__` / `aiohttp.ClientResponseError.__str__`\n"
        "    **本身就含整条 URL**,`exc_info=True` 的 traceback 同样会带出来。\n"
        "  ⇒ 走 `log_media_intake_failure`(只留 hostname + 异常类型名),\n"
        "     或把值包进 safe_exc / safe_url_for_log。\n"
        + "\n  ".join(problems)
    )


# ────────── 兜底(⛔ 不依赖上面那张清单):⛔ 不许只改一半 ──────────


def test_no_half_converted_handler() -> None:
    """凡是**已经**调了记账 helper 的 handler,同块里⛔ 不许还留着裸异常。

    ⭐ 这一条不看 ``_INBOUND_FETCH``,所以清单漂移时它仍然有效 ——
    它关的是「补了新的、却没删旧的」这个**半条链**形态。
    """
    problems: list[str] = []
    scanned = 0
    for rel, _f, _a in _INBOUND_FETCH.values():
        _src, tree = _parse(rel)
        for h in ast.walk(tree):
            if not isinstance(h, ast.ExceptHandler):
                continue
            if not any(_call_name(c) == _ACCOUNTING for c in _walk_calls(h)):
                continue
            scanned += 1
            forbidden = {h.name} if h.name else set()
            for msg in _offending_args(h, forbidden):
                problems.append(f"{rel} {msg}")
    assert scanned >= 12, (
        f"只扫到 {scanned} 个记账 handler —— 预期至少 12 个"
        "(feishu 5 + slack 6 + teams 3 + matrix 1 + discord 4)。"
        "⛔ 数字对不上先判量具坏,不是判通过。"
    )
    assert not problems, "记账补上了、旧的裸异常没删(半条链):\n  " + "\n  ".join(problems)


# ────────── 「飞书/钉钉现在是好的」—— 既有行为逐条显式说没变 ──────────


def test_feishu_accounting_sites_unchanged() -> None:
    """飞书那 5 处记账在本轮**一个字没动** —— 显式钉住,⛔ 不靠「我记得没动」。"""
    src, _tree = _parse("plugins/platforms/feishu/adapter.py")
    assert src.count(f"{_ACCOUNTING}(") == 5, (
        "飞书的入站媒体记账出口不再是 5 个 —— 本轮不该碰它。"
    )


def test_dingtalk_outbound_untouched() -> None:
    """钉钉出站:用户已拍板**不做**(⛔ 不引入 appKey/appSecret)。

    这里只钉住「本轮没有偷偷去接它」——⛔ 不断言它能用(它不能)。
    """
    src, _tree = _parse("plugins/platforms/dingtalk/adapter.py")
    # ⚠️ ⛔ 不许把 `robot_code` 写进禁用词:它是**既有**的 stream 模式配置字段
    #    (`extra.get("robot_code")`),与 OpenAPI 凭据面无关 —— 我第一版这么写,
    #    门出生即红,红在一个完全无辜的既有字段上。判据要贴**那个动作**,
    #    ⛔ 不贴一个恰好同名的字符串。
    for forbidden in ("org_group_send", "OrgGroupSendRequest", "batch_send_oto"):
        assert forbidden not in src, (
            f"钉钉 adapter 里出现了 {forbidden} —— 用户当轮明确拍板**不引入**"
            "钉钉 OpenAPI 凭据面。"
        )
