"""受信 runner 必须与进程环境隔离。

背景(真实缺陷,不是假想):这些 runner 会把 action token 通过 FD 交给脚本,脚本随后拿
``ZETTLAB_LOCAL_SERVER_URL`` 访问 **loopback** 上的 local-server。而 Python 的
requests/urllib **不会自动豁免 loopback** —— 只要进程环境里有 ``HTTP_PROXY`` /
``HTTPS_PROXY`` / ``ALL_PROXY`` 且没配 ``NO_PROXY``,那条带
``X-Zettlab-Agent-Action-Token`` 的请求就会被发到**外部代理**。

本线的设备上确实跑着 mihomo、确实配着代理变量,所以这不是理论风险。

同一批环境变量还带来第二个问题:``ZET_CHAT_APPEND_URL`` 这类**跨 profile 的陈旧回调
地址**会被脚本读到,把连接卡片/状态回写到错误的会话或 profile。

⛔ 判据不是「某个函数里有没有把代理变量删掉」——那是黑名单,是开集,下一个键还得再补
一次。两道门分工:
  - 行为门:脚本**自己报告**它看见了什么环境(唯一能证明运行时真的隔离了的办法);
  - 全集门:AST 扫**所有**受信 runner 调用点,防止「只修了被指出的那一处」。
"""

from __future__ import annotations

import ast
import json
import textwrap
from pathlib import Path

import pytest

import tools.terminal_tool as terminal_tool

from tests.tools.test_terminal_overseas_connect_runner import (  # noqa: F401
    canonical,
    cardonly,
    configure,
    reset_runtime,
    scope,
)


TERMINAL_TOOL_SOURCE = Path(terminal_tool.__file__)
TRUSTED_RUNNER = "run_trusted_python_script"

# 实测基线(2026-08-11):4 处调用点(connector / agent_creator / overseas_connect /
# camera)。低于它说明调用点被摘走或 runner 改了名字,而门会因为「一个都没扫到」恒绿。
MIN_CALL_SITES = 4

STALE_CALLBACK_SENTINEL = "http://stale.example/append"

PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


def _trusted_runner_call_sites() -> list[ast.Call]:
    tree = ast.parse(TERMINAL_TOOL_SOURCE.read_text(encoding="utf-8"))
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == TRUSTED_RUNNER:
            calls.append(node)
    return calls


def test_proxy_and_stale_callback_env_never_reach_the_script(monkeypatch, tmp_path):
    """⭐ 行为门:脚本自己报告它看见的环境,判据是**运行时真的隔离了**。

    ⛔ 不是「源码里有没有出现某个字符串」,也不是「传下去的 dict 长什么样」——那些都
    绕得过去(比如把变量塞进 injected_env)。让脚本自己说它看见了什么,是唯一闭集的问法。
    """
    configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os

        # 代理键与诱饵:**一律不许出现**(它们没有合法的注入用途)。
        leaked = sorted(
            key for key in os.environ
            if key.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
            or key == "HERMES_CANARY_FROM_PARENT"
        )
        print(json.dumps({
            "leaked": leaked,
            # ⭐ ZET_CHAT_APPEND_URL 在受信 runner 路径下**没有**任何合法注入用途:
            # afebbf011 钉的契约是「回调 URL 不进子进程,只派生出 base」,而 c9cc64540
            # 那次注入已由 da96b3d36 撤回,local.py 的 build_overseas_connect_runtime_env
            # 里现在立着一块「⛔ 别在这里注入」的路标。所以判据就是最强的那个:**键根本
            # 不该出现**,⛔ 不是「值不等于父进程那个」——后者在有人重新注入时会恒真。
            "chat_append_url": os.environ.get("ZET_CHAT_APPEND_URL", ""),
            # 镜像判据:injected 的键必须**确实到达**。只查「泄漏为空」抓不住
            # 「injected 也一起丢了」——而那正是 base_env={} 掐掉兜底的那个缺陷形态。
            "local_server_url": os.environ.get("ZETTLAB_LOCAL_SERVER_URL", ""),
        }))
        """,
    )
    # 造出真实的前置条件:进程环境里确实配着代理(设备上跑 mihomo 时就是这样)。
    for key in PROXY_KEYS:
        monkeypatch.setenv(key, "http://127.0.0.1:7890")
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", STALE_CALLBACK_SENTINEL)
    # ⭐ 反空转诱饵:一个普通的、不在任何黑名单里的自定义变量。黑名单式修法会放它过去,
    # 而「不继承进程环境」会连它一起挡掉 —— 少了这个诱饵,门就只能验证黑名单补全没补全。
    monkeypatch.setenv("HERMES_CANARY_FROM_PARENT", "1")

    with scope("a" * 64, ZETTLAB_LOCAL_SERVER_URL="http://127.0.0.1:19090"):
        result = terminal_tool._run_overseas_connect_command_if_allowed(
            cardonly(), cwd=str(tmp_path), timeout=30
        )

    assert result is not None, "runner 没有接管这条命令,门是空转的"
    payload = json.loads(
        [line for line in str(result).splitlines() if line.strip().startswith("{")][-1]
    )
    seen = json.loads(payload["output"].strip().splitlines()[-1])
    # ⭐ 判据就是 afebbf011 的原契约本身:这个键**根本不该出现在子进程里**。
    #
    # ⚠️ 这里曾经放宽成「值 != 父进程那个哨兵」,理由是「它有合法的注入用途」。那个理由
    # 随 da96b3d36 一起没了,而放宽留了下来 —— 放宽后的判据在有人重新注入时会**恒真**
    # (注入值 != 哨兵),门看着在跑、其实什么都不管。收回最强判据。
    #
    # 契约来源:afebbf011(Turing, 2026-08-06)—— **回调 URL 不进子进程,只派生出 base**;
    # 同 commit 的 test_direct_runner_uses_loopback_base_from_profile_scope 断言 append == ""。
    # 受信路径拿地址靠的是 ZETTLAB_LOCAL_SERVER_URL(下面那条镜像判据钉它)。
    #
    # ⚠️ 这条断言**曾经短暂失去鉴别力**:c9cc64540 为了让 connect.py 的第三条兜底
    # local_server_from_chat_url() "复活",往 runtime env 里注入了本 turn 的
    # ZET_CHAT_APPEND_URL ⇒ 子进程看到的永远是注入值、`!= 哨兵` 恒真 ⇒ 重言式。
    # 那次注入**推翻了 afebbf011 的契约**,已由 da96b3d36 撤回;撤回后本条恢复鉴别力:
    # 子进程不该拿到该键,一旦谁把继承恢复回来,哨兵值就会在这里现形。
    #
    # ⛔ 别再往 build_overseas_connect_runtime_env 注入这个键(已翻过两次烧饼)。
    # 顺带纠正当时的归因:scope 里的值本来就不进 os.environ,所以在受信 runner 路径下
    # 那条兜底从 afebbf011 起就是死代码 —— **不是被 base_env={} 掐掉的**。
    #
    # ⭐ 「不继承」的承重探针是**代理三键**(不许出现且不被注入);这条哨兵与它们同向。
    assert seen["chat_append_url"] == "", (
        f"子进程看到了 ZET_CHAT_APPEND_URL({seen['chat_append_url']!r})—— 继承自父进程的话"
        f"(哨兵 {STALE_CALLBACK_SENTINEL})它可能属于别的 profile,卡片/状态会回写到错误的会话;"
        "被谁重新注入的话则推翻了 afebbf011 的契约。受信路径的地址来源只有 "
        "ZETTLAB_LOCAL_SERVER_URL"
    )
    # 镜像判据:injected 必须真的到达(⛔ 少了这条,把 injected 也丢光同样能让上面全绿)。
    assert seen["local_server_url"], (
        "ZETTLAB_LOCAL_SERVER_URL 没有到达子进程 —— injected_env 这条链路断了。"
        "connect.py 会因为拿不到设备服务地址直接 fail fast,用户看到「未能打开连接卡片」"
    )
    leaked = seen["leaked"]
    assert leaked == [], (
        f"受信脚本看到了来自父进程的环境变量 {leaked} —— "
        "脚本随后拿 action token 访问 loopback,而 Python 不会自动豁免 loopback:"
        "带 X-Zettlab-Agent-Action-Token 的请求会经 HTTP_PROXY 外发。"
        "ZET_CHAT_APPEND_URL 则会让卡片/状态回写到错误的会话或 profile。"
        "修法是 base_env={}(照抄 _run_camera_runtime_command_if_allowed),"
        "⛔ 不是往黑名单里再补几个键"
    )


def test_every_trusted_runner_call_passes_an_empty_base_env() -> None:
    """全集门:**每一处** run_trusted_python_script 的 base_env 都必须是空字面量。

    ⭐ 判据是闭集的:它不问「这一处带不带 token」「这一处访不访问 loopback」——那种逐案
    判断正是漏网的原因(review 只点了 overseas_connect 一处,而 agent_creator 与
    connector 是同一形状、同样把整个 os.environ 传下去)。

    ⛔ 别把判据放宽成「base_env 不含代理键」:那样又变回黑名单,ZET_CHAT_APPEND_URL 这类
    非代理的跨 profile 变量照样漏过去。
    """
    calls = _trusted_runner_call_sites()
    assert len(calls) >= MIN_CALL_SITES, (
        f"只扫到 {len(calls)} 处 {TRUSTED_RUNNER} 调用点,少于基线 {MIN_CALL_SITES} —— "
        "要么 runner 改了名字、要么调用点被摘走了,门已经对它们恒绿。"
        "⛔ 别直接调低这个数来换绿:先确认那些调用点去哪了"
    )

    offenders: list[str] = []
    for call in calls:
        base_env = next((kw.value for kw in call.keywords if kw.arg == "base_env"), None)
        if base_env is None:
            offenders.append(f"{TERMINAL_TOOL_SOURCE.name}:{call.lineno} 没有显式传 base_env")
            continue
        if not (isinstance(base_env, ast.Dict) and not base_env.keys):
            offenders.append(
                f"{TERMINAL_TOOL_SOURCE.name}:{call.lineno} 的 base_env 不是空字面量"
            )

    assert not offenders, (
        "受信 runner 继承了进程环境:\n  "
        + "\n  ".join(offenders)
        + "\n照抄 _run_camera_runtime_command_if_allowed 的 base_env={},"
        "⛔ 别用 _sanitize_subprocess_env(os.environ)(那是黑名单,只剥 Hermes 自己的密钥)"
    )


# ⭐ action token 的**格式**是签发方(local-server)的事,这里⛔ 不重新实现一套。
#
# 原先写的是 re.fullmatch(r"[0-9a-f]{64}", token) —— 比签发方还严。local-server 的权威
# 判据是 internal/agent/actiontoken/store.go 的 looksLikeIssuedToken:
#     if len(token) != 64 { return false }
#     _, err := hex.DecodeString(token); return err == nil
# 而 hex.DecodeString **接受大写**。于是 profile .env 里已存的大写 hex token,签发方认、
# 我们拒,用户点连接卡只看到「secure flow 不可用」,而毛病不在他那边。
#
# ⛔ 判据不是「正则写成什么样」,是**签发方会认的 token 我们必须也认**。
@pytest.mark.parametrize(
    "token",
    [
        "A" * 64,                    # 大写 hex —— local-server 的 hex.DecodeString 认
        "aB" * 32,                   # 大小写混合 hex —— 同上
        "0123456789abcdef" * 4,      # 小写 hex(原来唯一能过的形状)
        "opaque-token-with-dashes",  # 非 hex 的 opaque token:格式是签发方的事,不是我们的
    ],
)
def test_action_token_formats_the_issuer_accepts_are_not_rejected_here(monkeypatch, token):
    from tools.environments import local as local_env

    monkeypatch.setattr(
        local_env, "get_secret",
        lambda key, default="": {
            "ZETTLAB_AGENT_ACTION_TOKEN": token,
            "ZET_CHAT_APPEND_URL": "http://127.0.0.1:19090/api/v1/internal/chat/append",
        }.get(key, default),
        raising=False,
    )
    with monkeypatch.context() as m:
        m.setattr("agent.secret_scope.get_secret",
                  lambda key, default="": {
                      "ZETTLAB_AGENT_ACTION_TOKEN": token,
                      "ZET_CHAT_APPEND_URL": "http://127.0.0.1:19090/api/v1/internal/chat/append",
                  }.get(key, default), raising=False)
        _env, got = local_env.build_overseas_connect_runtime_env()
    assert got == token, "签发方会认的 token 在这里被改写或拒绝了"


# 反面陪审:放宽格式**不等于**什么都收。不安全地进环境/FD 的值仍必须被拒。
# ⛔ 少了这条,把校验整个删掉也能让上面那组全绿。
@pytest.mark.parametrize("token", ["", "   ", "has\x00nul", "x" * (4 * 1024 + 1)])
def test_unsafe_action_tokens_are_still_rejected(monkeypatch, token):
    from tools.environments import local as local_env

    with monkeypatch.context() as m:
        m.setattr("agent.secret_scope.get_secret",
                  lambda key, default="": {
                      "ZETTLAB_AGENT_ACTION_TOKEN": token,
                      "ZET_CHAT_APPEND_URL": "http://127.0.0.1:19090/api/v1/internal/chat/append",
                  }.get(key, default), raising=False)
        with pytest.raises(RuntimeError):
            local_env.build_overseas_connect_runtime_env()


# ⭐ 收紧判据:进子进程的环境有**两个入口** —— base_env 和 injected_env。
#
# 上一轮只把 base_env 收成 {},而 injected_env 由 build_*_runtime_env 提供,其中
# build_connector_runtime_env 是从 `_sanitize_subprocess_env(os.environ, ...)` 起手的
# —— **同一个泄漏换了个参数进来**,代理变量照样到子进程。门当时只盯 base_env,所以
# 全绿。
#
# ⛔ 判据不能是「某个参数是不是空的」,必须贴**最终进入子进程的环境全集**:凡是给受信
# runner 供 env 的构造函数,都不许从进程环境起手。
RUNTIME_ENV_BUILDERS_SOURCE = Path(
    __import__("tools.environments.local", fromlist=["local"]).__file__
)

# 已知不归本线负责、且确实从 os.environ 起手的构造函数。
# ⚠️ 这是**显式豁免**,不是判据放宽:它们的归属是别的团队(video-edit 线),越界改属于
# 动别人的面。门仍然扫全集,只是把这几个的失败翻译成「报清单」而不是「本线的红」。
# ⛔ 别往这里加本线自己的函数来换绿。
NOT_OUR_SURFACE = {"build_video_edit_runtime_env"}


def _runtime_env_builders() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(RUNTIME_ENV_BUILDERS_SOURCE.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name.startswith("build_")
        and node.name.endswith("_runtime_env")
    }


def _starts_from_process_env(fn: ast.FunctionDef) -> bool:
    """这个构造函数是不是拿 os.environ **当底座**。

    ⛔ 判据必须是 AST,不是字符串匹配:上一版用 `"...(os.environ" in source` 判,结果
    命中了注释里引用旧写法的那一行 —— 函数明明已经改成白名单构造,门照样红。
    源码片段里有那串字符 ≠ 那句代码还在执行。

    逐键的 `os.environ.get(key)` 是**白名单读取**,合法;整个 os.environ 作为实参
    传给底座构造(_sanitize_subprocess_env(os.environ, ...) / dict(os.environ))才是问题。
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        for arg in node.args:
            if (
                isinstance(arg, ast.Attribute)
                and arg.attr == "environ"
                and isinstance(arg.value, ast.Name)
                and arg.value.id == "os"
            ):
                return True
    return False


def test_no_runtime_env_builder_starts_from_the_process_environment() -> None:
    builders = _runtime_env_builders()
    assert len(builders) >= 5, (
        f"只扫到 {len(builders)} 个 build_*_runtime_env,少于实测基线 5 —— "
        "命名变了或函数被摘走了,门已经对它们恒绿"
    )

    offenders = []
    for name, fn in sorted(builders.items()):
        if _starts_from_process_env(fn):
            offenders.append(name)

    ours = [n for n in offenders if n not in NOT_OUR_SURFACE]
    assert not ours, (
        f"这些受信 runner 的 env 构造函数从进程环境起手:{ours} —— "
        "把 run_trusted_python_script 的 base_env 收成 {} 拦不住它们,"
        "这份 dict 是走 injected_env 进子进程的,HTTP_PROXY / ZET_CHAT_APPEND_URL 照样过去。"
        "照抄 build_camera_runtime_env / build_overseas_connect_runtime_env 的白名单构造"
    )


# ⭐ 凡是把凭据交给子进程的受信 runner,都必须**先建立进程边界再取凭据**。
#
# _ensure_sensitive_runtime_boundary() 收紧本进程的 ptrace 可见性。少了它,模型启动的
# 本地子进程若能 ptrace 父进程,就能在 action token 进入内存的那一刻拿到连接授权。
#
# ⛔ 判据不是「这个函数里有没有那句调用」这种逐个点名,是**闭集的蕴含关系**:
#     注入 injected_secrets ⇒ 必须调 _ensure_sensitive_runtime_boundary
# 不注入凭据的 runner(lark_cli / video_edit)不在约束内 —— 它们没有可泄漏的东西。
BOUNDARY_FN = "_ensure_sensitive_runtime_boundary"

# 已知违反、但不归本线负责的 runner。
# ⚠️ _run_agent_creator_command_if_allowed 由 Gareth 引入(feat: add trusted zettctl
# execution path),本线零改动。它注入 ZETTLAB_AGENT_ACTION_TOKEN 却不调边界,与我们刚
# 修的 overseas_connect 是**同一形状** —— 已报清单,⛔ 不越界改别人的面。
# ⛔ 别往这里加本线自己的 runner 来换绿。
BOUNDARY_NOT_OUR_SURFACE = {"_run_agent_creator_command_if_allowed"}


def test_every_credential_injecting_runner_establishes_the_boundary_first() -> None:
    tree = ast.parse(TERMINAL_TOOL_SOURCE.read_text(encoding="utf-8"))
    runners = [
        fn
        for fn in ast.walk(tree)
        if isinstance(fn, ast.FunctionDef)
        and fn.name.startswith("_run_")
        and fn.name.endswith("_command_if_allowed")
    ]
    assert len(runners) >= 6, (
        f"只扫到 {len(runners)} 个受信 runner,少于实测基线 6 —— 命名变了或被摘走了,门已恒绿"
    )

    offenders = []
    late_boundary = []
    checked = 0
    for fn in runners:
        dumped = ast.dump(fn)
        if "injected_secrets" not in dumped:
            continue  # 不交凭据给子进程,不在约束内
        checked += 1
        boundary_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", "") == BOUNDARY_FN
        ]
        if not boundary_lines:
            offenders.append(fn.name)
            continue
        # ⭐ 存在性不够:把 boundary 挪到取凭据**之后**,上面那个 any() 照样绿,而 token
        # 已经在内存里了 —— 边界要防的正是那一刻。所以还要比顺序。
        #
        # ⚠️ 老实说清楚:行号是**词法**顺序,不是控制流。它挡的是「有人把这行挪到下面」
        # 这类直白漂移;真正的语义由行为门
        # test_a_failed_boundary_blocks_the_runner_before_any_credential_is_built 保证 ——
        # 那条让边界建立失败,断言凭据构造函数一次都没被调用。⛔ 别把这条 AST 判据
        # 当成语义证明。
        credential_lines = [
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", "").startswith("build_")
            and getattr(n.func, "id", "").endswith("_runtime_env")
        ]
        if credential_lines and min(boundary_lines) > min(credential_lines):
            late_boundary.append(fn.name)

    assert checked >= 3, (
        f"只有 {checked} 个 runner 被判定为「注入凭据」,少于实测基线 3 —— "
        "injected_secrets 可能改了名字,门对其余的已恒绿"
    )
    ours_late = [n for n in late_boundary if n not in BOUNDARY_NOT_OUR_SURFACE]
    assert not ours_late, (
        f"这些 runner 调了 {BOUNDARY_FN} 但排在取凭据**之后**:{ours_late} —— "
        "边界要防的就是「token 进入内存的那一刻子进程能 ptrace 父进程」,晚一步等于没有"
    )
    ours = [n for n in offenders if n not in BOUNDARY_NOT_OUR_SURFACE]
    assert not ours, (
        f"这些 runner 把凭据交给子进程却没先建立进程边界:{ours} —— "
        f"照抄 _run_camera_runtime_command_if_allowed / "
        f"_run_connector_runtime_command_if_allowed:取凭据**之前**调 {BOUNDARY_FN}(),"
        "建不起来就 fail closed 阻断该 runner,⛔ 不降级放行"
    )


# ⭐ connector runtime 的执行级定案:让子进程把它**实际拿到**的环境键打印出来。
#
# ⛔ 这条不许用 grep / AST / 「我读了那个白名单循环」来判 —— 同一个问题上读源码给过两个
# 相反的错答案:先是 `grep 'env["..."]'` 只抓显式赋值、漏了 CONNECTOR_RUNTIME_ENV_KEYS
# 的循环填充,误判成「base_env={} 打断了 connector 线」;后来反事实又只查了改动前。
# 判据换维度:跑一次,看子进程手里到底有什么。
def test_connector_runtime_keys_actually_reach_the_subprocess(monkeypatch, tmp_path):
    from agent import secret_scope
    from tools.environments.local import build_connector_runtime_env
    from tools.trusted_direct_runner import run_trusted_python_script

    script = tmp_path / "probe.py"
    script.write_text(
        "import json, os\n"
        "print(json.dumps(sorted(os.environ)))\n",
        encoding="utf-8",
    )
    # 造出 connector 那三个键的来源(profile scope),外加父进程的代理变量与诱饵。
    for key in PROXY_KEYS:
        monkeypatch.setenv(key, "http://127.0.0.1:7890")
    monkeypatch.setenv("HERMES_CANARY_FROM_PARENT", "1")
    token = secret_scope.set_secret_scope({
        "ZETTLAB_CONNECTORS_URL": "http://127.0.0.1:19090/api/v1/connectors",
        "ZETTLAB_CONNECTORS_AUTH_TOKEN": "c" * 32,
        "ZET_AGENT_ID": "agent-probe",
    })
    try:
        connector_env = build_connector_runtime_env()
        completed = run_trusted_python_script(
            script=script,
            argv=[],
            cwd=tmp_path,
            base_env={},
            injected_env=connector_env,
            timeout=30,
            secret_values=[],
            stdlib_only=True,
        )
    finally:
        secret_scope.reset_secret_scope(token)

    seen = set(json.loads(completed.output.strip().splitlines()[-1]))

    # ① connector 脚本真正消费的三个键必须到达(skills/intl/monday/scripts/
    #    connector_runtime.py 读的正是它们)。⭐ 逐个分别钉,⛔ 不用一个代表 ——
    #    本轮已经栽过一次「镜像断言只钉一个键、漏掉的那个逆改照样全绿」。
    for key in ("ZETTLAB_CONNECTORS_URL", "ZETTLAB_CONNECTORS_AUTH_TOKEN", "ZET_AGENT_ID"):
        assert key in seen, (
            f"{key} 没有到达 connector runtime 的子进程 —— "
            "该 skill 的执行路径被 base_env={} 打断了"
        )
    # ② 同时证明 base_env={} 仍在生效:父进程的代理变量与诱饵一个都不许过去。
    leaked = sorted(
        k for k in seen
        if k.upper() in {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"} or k == "HERMES_CANARY_FROM_PARENT"
    )
    assert leaked == [], f"connector 子进程看到了父进程的 {leaked}"


# ⭐ 补上 AST 全集门的语义盲区:它查的是**语法形状**(base_env 是不是空字面量),
# 而我们要的是**语义**(子进程拿不到父进程环境)。形状对、语义错是可能的 ——
# 比如 base_env={} 写着,却把 os.environ 塞进 injected_env,AST 门照样绿。
#
# ⚠️ 这条判据来自前端那边被 reviewer 揭穿的一道假绿门:探针和目标在一个**与门无关的
# 维度**上有差异,红的是那个无关维度。⇒ 每次逆改都要问「我的探针除了『该被抓住』
# 之外还和目标差在哪」。AST 门的探针(改字面量)与目标(不继承环境)差的正是这一层。
#
# 这道门钉在**四条受信 runner 的共同底层** run_trusted_python_script 上:只要
# base_env={},父进程环境就一个字节都不许到子进程。与 AST 全集门配合 ——
# 那道保证每个调用点确实传 {},这道保证传 {} 时语义真的成立。
# ⇒ agent_creator 与 camera 两条路径此前只有 AST 门覆盖,现在有了行为级证据。
def test_empty_base_env_really_isolates_the_subprocess(monkeypatch, tmp_path):
    from tools.trusted_direct_runner import run_trusted_python_script

    script = tmp_path / "probe.py"
    script.write_text("import json, os\nprint(json.dumps(sorted(os.environ)))\n", encoding="utf-8")

    # 父进程里塞满各类值:代理、跨 profile 回调、凭据样本、以及一个普通自定义变量。
    # ⭐ 最后那个是反空转诱饵:它不在任何黑名单里,黑名单式实现会放它过去。
    for key in PROXY_KEYS:
        monkeypatch.setenv(key, "http://127.0.0.1:7890")
    monkeypatch.setenv("ZET_CHAT_APPEND_URL", STALE_CALLBACK_SENTINEL)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "parent-token-should-not-leak")
    monkeypatch.setenv("HERMES_CANARY_FROM_PARENT", "1")

    completed = run_trusted_python_script(
        script=script,
        argv=[],
        cwd=tmp_path,
        base_env={},
        injected_env={"ZETTLAB_LOCAL_SERVER_URL": "http://127.0.0.1:19090"},
        timeout=30,
        secret_values=[],
        stdlib_only=True,
    )
    seen = set(json.loads(completed.output.strip().splitlines()[-1]))

    parent_only = {
        *PROXY_KEYS,
        "ZET_CHAT_APPEND_URL",
        "ZETTLAB_AGENT_ACTION_TOKEN",
        "HERMES_CANARY_FROM_PARENT",
    }
    leaked = sorted(parent_only & seen)
    assert leaked == [], (
        f"base_env={{}} 却让父进程的 {leaked} 到了子进程 —— "
        "AST 全集门只看得到「字面量是不是 {}」这层形状,语义要靠这条钉住"
    )
    # 反空转:injected 的键必须真的到达,否则「什么都没到」也能让上面全绿。
    assert "ZETTLAB_LOCAL_SERVER_URL" in seen, "injected_env 没到达,本用例是空转的"


# ⭐ boundary 门的**行为级**对应物。
#
# 上面那道 AST 门查的是「函数体里有没有调 boundary」加一条**词法**顺序(行号)。两者都
# 证明不了语义:AST 看不见控制流,行号更看不见 —— 把 boundary 包进一个永假的分支里,
# 两条判据全绿,而 token 照样进内存。
#
# 这条从另一头钉,判据是**数据流**:让边界建立失败,凭据构造函数必须**一次都没被调用**。
# 边界要防的就是「模型启动的本地子进程能 ptrace 父进程、在 token 进内存那一刻拿到连接
# 授权」,晚一步等于没有;而「有没有构造过凭据」是这件事唯一的闭集判据。
def test_a_failed_boundary_blocks_the_runner_before_any_credential_is_built(monkeypatch, tmp_path):
    from tools.environments import local as local_env

    built: list[str] = []
    real_builder = local_env.build_overseas_connect_runtime_env

    def spy():
        built.append("built")
        return real_builder()

    configure(monkeypatch, tmp_path, "print('{}')\n")
    monkeypatch.setattr(local_env, "build_overseas_connect_runtime_env", spy)
    monkeypatch.setattr(terminal_tool, "_ensure_sensitive_runtime_boundary", lambda: False)

    with scope("a" * 64, ZETTLAB_LOCAL_SERVER_URL="http://127.0.0.1:19090"):
        result = terminal_tool._run_overseas_connect_command_if_allowed(
            cardonly(), cwd=str(tmp_path), timeout=30
        )

    assert result is not None, "runner 没有接管这条命令,门是空转的"
    assert built == [], (
        "边界建立失败了,却仍然调用了 build_overseas_connect_runtime_env —— "
        "action token 已经进了本进程内存,而这正是边界要防的那一刻。"
        "boundary 必须排在取凭据**之前**并 fail closed"
    )
    payload = json.loads(
        [line for line in str(result).splitlines() if line.strip().startswith("{")][-1]
    )
    assert "boundary" in json.dumps(payload, ensure_ascii=False), (
        f"边界失败没有如实告知调用方,返回的是 {payload} —— "
        "⛔ 不许静默降级成别的失败原因,用户与日志都会被指向错误的方向"
    )


# 反面同格:边界正常时,凭据**必须**被构造 —— 否则上面那条断言会因为「这条路径本来就
# 不调 builder」而恒真(⛔ 对照组同参同值 = 重言式)。
def test_a_healthy_boundary_still_lets_the_runner_build_its_credentials(monkeypatch, tmp_path):
    from tools.environments import local as local_env

    built: list[str] = []
    real_builder = local_env.build_overseas_connect_runtime_env

    def spy():
        built.append("built")
        return real_builder()

    configure(monkeypatch, tmp_path, "print('{}')\n")
    monkeypatch.setattr(local_env, "build_overseas_connect_runtime_env", spy)
    monkeypatch.setattr(terminal_tool, "_ensure_sensitive_runtime_boundary", lambda: True)

    with scope("a" * 64, ZETTLAB_LOCAL_SERVER_URL="http://127.0.0.1:19090"):
        terminal_tool._run_overseas_connect_command_if_allowed(
            cardonly(), cwd=str(tmp_path), timeout=30
        )

    assert built == ["built"], (
        f"边界正常时凭据构造被调用 {len(built)} 次,want 1 —— "
        "上面那条「失败时零调用」的断言会因此变成重言式,永远绿"
    )
