"""The profile-scoped subprocess env contract.

The bug these pin: nothing pointed the worker's wecom-cli at the session
profile's credential store, so it reported WeCom "not initialised" on a device
where the credentials were present and live API calls worked.

Two shapes reach that outcome and both are covered below. In per-profile-process
mode local-server injects ``WECOM_CLI_CONFIG_DIR`` computed for the GATEWAY's
agent, so a worker on another profile inherits a pointer to the wrong store. In
multiplex_gateway mode -- what the device runs -- there is no per-agent child at
all, so the key never arrives and wecom-cli falls back to one global directory
that every agent shares. Re-pointing a wrong value and supplying a missing one
are different tests; only the second matches the device.

⛔ The assertions below spawn a real child and read the environment from
*inside* it. Checking the dict we built would only prove we wrote a line;
``/proc/<pid>/environ`` would not help either, since it freezes at exec and
cannot see what Python put into ``os.environ`` afterwards. The only honest
observation is the child reporting its own environment.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

from agent import secret_scope
from hermes_constants import (
    apply_context_profile_scoped_env,
    apply_profile_scoped_env,
    reset_hermes_home_override,
    set_hermes_home_override,
)
from tools.environments.local import (
    LocalEnvironment,
    _sanitize_subprocess_env,
    hermes_subprocess_env,
)

# Reads the keys under test out of the child's own environment.
_REPORTER = (
    "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
    "('HERMES_HOME', 'WECOM_CLI_CONFIG_DIR', 'WECOM_SKILLS_DIR', 'LARK_SKILLS_DIR')}))"
)


def _child_env_of(env: dict[str, str]) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, "-c", _REPORTER],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"reporter failed: {result.stderr}"
    return json.loads(result.stdout)


def _gateway_env(profile_root: Path) -> dict[str, str]:
    """What local-server hands the gateway: keys computed for the GATEWAY's agent."""
    env = hermes_subprocess_env(inherit_credentials=True)
    env["HERMES_HOME"] = str(profile_root / "gateway-agent")
    env["WECOM_CLI_CONFIG_DIR"] = str(profile_root / "gateway-agent" / "wecom-cli-config")
    # Both skills dirs are one shared global path on the board, not per-agent.
    env["WECOM_SKILLS_DIR"] = "/root/.agents/skills"
    env["LARK_SKILLS_DIR"] = "/root/.agents/skills"
    return env


def test_wecom_config_dir_follows_the_profile_into_the_spawned_child(tmp_path):
    session_profile = tmp_path / "profiles" / "main"
    env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(env, session_profile)

    seen = _child_env_of(env)
    assert seen["HERMES_HOME"] == str(session_profile)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config"), (
        "the child still points at the gateway agent's credential store; wecom-cli "
        "reads this key and nothing else, so it finds no credentials and reports "
        "'not initialised'"
    )


def test_the_key_is_supplied_when_the_gateway_never_had_it(tmp_path):
    """The shape the device is actually in, and it is not the one above.

    multiplex_gateway mode has no per-agent child process, so registry.go's
    gatewayEnv injection never runs; per-profile values go into
    <hermesHome>/profiles/<agent>/.env instead, and the interactive path's
    load_hermes_dotenv reads the LAUNCH home's .env, not the profile's. The key
    therefore never arrives at all -- and an absent WECOM_CLI_CONFIG_DIR makes
    wecom-cli fall back to one global directory, so every agent shares a single
    identity and none see their own credentials.

    ⛔ Re-pointing a wrong value and supplying a missing one are different
    tests; only the second matches the device.
    """
    session_profile = tmp_path / "profiles" / "main"
    env = hermes_subprocess_env(inherit_credentials=True)
    env.pop("WECOM_CLI_CONFIG_DIR", None)
    assert "WECOM_CLI_CONFIG_DIR" not in env, "the premise is that nobody injected it"

    apply_profile_scoped_env(env, session_profile)

    seen = _child_env_of(env)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config")


def test_the_value_survives_the_terminal_hop_the_agent_actually_uses(tmp_path):
    """wecom-cli is not the worker: the agent runs it through the terminal path.

    That path uses a different sanitizer (``_sanitize_subprocess_env``), so the
    worker having the key proves nothing on its own.
    """
    session_profile = tmp_path / "profiles" / "main"
    worker_env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(worker_env, session_profile)

    terminal_env = _sanitize_subprocess_env(worker_env)
    assert "PATH" in terminal_env, "calibration: an empty base makes every key look stripped"

    seen = _child_env_of(terminal_env)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config")


def test_shared_skills_dirs_are_not_rewritten_per_profile(tmp_path):
    """⛔ These two are one global directory on the board, not per-agent.

    Deriving them from the profile would aim wecom-cli and lark-cli at a
    directory that has no skills in it -- a regression that looks like the fix.
    """
    env = _gateway_env(tmp_path / "profiles")
    apply_profile_scoped_env(env, tmp_path / "profiles" / "main")

    seen = _child_env_of(env)
    assert seen["WECOM_SKILLS_DIR"] == "/root/.agents/skills"
    assert seen["LARK_SKILLS_DIR"] == "/root/.agents/skills"


def test_the_multiplex_main_path_gets_the_key_from_the_context_pin(tmp_path):
    """mux 主路径只从 ContextVar 取得当前 profile，不能依赖其它 PTY 路径。"""
    session_profile = tmp_path / "profiles" / "main"
    token = set_hermes_home_override(session_profile)
    try:
        env = _sanitize_subprocess_env(os.environ.copy())
    finally:
        reset_hermes_home_override(token)

    assert "PATH" in env, "calibration: an empty base makes every key look stripped"
    seen = _child_env_of(env)
    assert seen["HERMES_HOME"] == str(session_profile)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config"), (
        "the multiplex agent re-homed to this profile still points wecom-cli at "
        "the launch profile's credential store, or at the one global directory "
        "every agent shares -- another agent's WeCom identity"
    )


def test_context_profile_scoped_env_carries_every_key_to_a_child(tmp_path):
    """共享桥接契约必须把当前 profile 的全部路径键带进真实子进程。"""
    session_profile = tmp_path / "profiles" / "main"
    token = set_hermes_home_override(session_profile)
    try:
        env = os.environ.copy()
        env.pop("HERMES_HOME", None)
        env.pop("WECOM_CLI_CONFIG_DIR", None)
        apply_context_profile_scoped_env(env)
    finally:
        reset_hermes_home_override(token)

    seen = _child_env_of(env)
    assert seen["HERMES_HOME"] == str(session_profile)
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(session_profile / "wecom-cli-config"), (
        "ContextVar bridge only carried HERMES_HOME; the child cannot find its "
        "own wecom-cli credential directory"
    )


def test_single_profile_child_derives_wecom_dir_without_context_pin(tmp_path):
    """单 profile 由最终 HERMES_HOME 派生，不能被 MCP whitelist 擦掉。"""
    profile_home = tmp_path / "single-profile"
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    token = set_hermes_home_override(None)
    try:
        env = {"HERMES_HOME": str(profile_home)}
        apply_context_profile_scoped_env(env)
    finally:
        reset_hermes_home_override(token)
        secret_scope.set_multiplex_active(previous_multiplex)

    seen = _child_env_of({**os.environ, **env})
    assert seen["WECOM_CLI_CONFIG_DIR"] == str(profile_home / "wecom-cli-config")


def test_multiplex_child_without_context_pin_drops_inherited_wecom_dir(tmp_path):
    """multiplex 没有当前身份时继续 fail-closed，不能继承启动身份。"""
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = set_hermes_home_override(None)
    try:
        env = {
            "HERMES_HOME": str(tmp_path / "launch"),
            "WECOM_CLI_CONFIG_DIR": str(tmp_path / "foreign"),
        }
        apply_context_profile_scoped_env(env)
    finally:
        reset_hermes_home_override(token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert "WECOM_CLI_CONFIG_DIR" not in env


def test_a_child_nobody_re_pointed_keeps_its_inherited_pointer(tmp_path):
    """没有 pin 时不改指，不能凭环境中的 HERMES_HOME 虚构 profile。"""
    base = os.environ.copy()
    base.pop("WECOM_CLI_CONFIG_DIR", None)
    base["HERMES_HOME"] = str(tmp_path / "launch-home")

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = set_hermes_home_override(None)
    try:
        env = _sanitize_subprocess_env(base)
    finally:
        reset_hermes_home_override(token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert "WECOM_CLI_CONFIG_DIR" not in env, (
        "nobody pinned a profile for this child, so there is no profile to scope "
        "it to; supplying one overrides wecom-cli's own fallback"
    )


def test_wecom_config_dir_is_reinjected_after_a_profile_switch_in_one_shell_snapshot(
    tmp_path,
):
    """复用 LocalEnvironment 时，后一个 profile 不能 source 到前一个目录。"""
    first = tmp_path / "profiles" / "first"
    second = tmp_path / "profiles" / "second"
    first_config = first / "wecom-cli-config"
    second_config = second / "wecom-cli-config"
    first_config.mkdir(parents=True)
    second_config.mkdir(parents=True)

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    first_token = set_hermes_home_override(str(first))
    environment = None
    try:
        environment = LocalEnvironment(
            cwd=str(tmp_path),
            timeout=10,
            env={
                "HERMES_HOME": str(first),
                "WECOM_CLI_CONFIG_DIR": str(first_config),
            },
        )
        first_result = environment.execute("printf %s \"$WECOM_CLI_CONFIG_DIR\"")
        assert first_result["output"] == str(first_config)
    finally:
        reset_hermes_home_override(first_token)

    second_token = set_hermes_home_override(str(second))
    try:
        second_result = environment.execute("printf %s \"$WECOM_CLI_CONFIG_DIR\"")
        assert second_result["output"] == str(second_config), (
            "复用的 shell snapshot 仍把前一个 profile 的 wecom-cli 凭据目录"
            "暴露给当前 profile"
        )
    finally:
        reset_hermes_home_override(second_token)
        secret_scope.set_multiplex_active(previous_multiplex)
        if environment is not None:
            environment.cleanup()


def test_multiplex_terminal_without_a_profile_pin_does_not_inherit_wecom_config_dir(
    tmp_path,
):
    """没有当前身份时，不能把服务 profile 的配置目录冒充为当前身份。"""
    ambient = tmp_path / "profiles" / "gateway-agent" / "wecom-cli-config"
    ambient.mkdir(parents=True)
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    token = set_hermes_home_override(None)
    try:
        environment = LocalEnvironment.__new__(LocalEnvironment)
        environment.env = {
            "HERMES_HOME": str(ambient.parent),
            "WECOM_CLI_CONFIG_DIR": str(ambient),
        }
        exports = environment._snapshot_ephemeral_env_exports()
    finally:
        reset_hermes_home_override(token)
        secret_scope.set_multiplex_active(previous_multiplex)

    assert not any(export.startswith("export WECOM_CLI_CONFIG_DIR=") for export in exports), (
        "没有当前 profile pin 时，快照恢复仍导出了服务身份的 wecom-cli 凭据目录"
    )


def test_every_context_pin_reaches_env_through_the_contract():
    """早期信号：读 ContextVar 的子进程入口应当走共享契约。

    🔴 ⛔ **这条不是闭集，⛔ 不许当保障。** 上一版 docstring 第一句就写着
    「闭集门」—— 那是**虚假声明**。RH 复审第四轮实测：新增两个模块，
    A 读 ContextVar 并 return profile home、B 调 A 后手写 ``HERMES_HOME``
    再 spawn —— 生产契约被完全绕开，本门仍 **1 passed**。
    根因是它按**单文件源码字面量**筛 reader / bridge，⛔ 追不了跨 helper 的
    数据流。

    ⭐ 真正承重的形态是「把 invariant 收敛到唯一生产 env-builder，
    使违规状态**不可表示**」+ 真实 child-env 集成测试；
    ⛔ 不是继续静态追名字（那是必输的开集追逐战）。**本轮没做**。

    本门保留的价值：新增**单文件内**的 inline 写入时会红，作为早期提醒。
    """
    root = Path(__file__).resolve().parents[2]
    skip = ("/tests/", "/.venv/", "/node_modules/", "/.git/")
    readers, offenders = [], {}
    for path in root.rglob("*.py"):
        posix = path.as_posix()
        if any(part in posix for part in skip) or path.name == "hermes_constants.py":
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        # ⚠️ 扫描范围必须是「读 pin 的」∪「调契约的」。旧版只看前者，于是
        # copilot_acp_client / lsp/client / codex_app_server 这三个 bridge
        # 永远进不了 readers（它们只调 helper、不直接读 pin），offenders 检查
        # 就永远不会碰它们 —— 又一个按"长什么样"划范围造成的免检区。
        is_reader = "get_hermes_home_override" in source
        is_bridge = "apply_context_profile_scoped_env(" in source
        if not (is_reader or is_bridge):
            continue
        if is_reader:
            readers.append(path.relative_to(root).as_posix())
        # ⛔ 判据不许用字面量检索（旧版只认 `HERMES_HOME"] =` 这一种写法）：
        # dict 字面量 `{"HERMES_HOME": x}`、`env.update({...})`、换个引号或换行
        # 都能绕过去，门静默恒绿。改用 AST 覆盖全部写入形态。
        inline = _hermes_home_writes_outside_the_contract(ast.parse(source))
        if inline:
            offenders[path.relative_to(root).as_posix()] = inline

    assert readers, "calibration: nothing read the ContextVar, so the sweep found nothing"
    assert not offenders, (
        f"these read the profile pin and write HERMES_HOME themselves: {offenders}. "
        "Every profile-scoped key has to move together, which is what "
        "apply_context_profile_scoped_env is for"
    )

    expected_bridges = {
        "agent/copilot_acp_client.py",
        "agent/lsp/client.py",
        "agent/transports/codex_app_server.py",
        "tools/environments/local.py",
        "tools/mcp_tool.py",
    }
    actual_bridges = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if not any(part in path.as_posix() for part in skip)
        and path.name != "hermes_constants.py"
        and "apply_context_profile_scoped_env(" in path.read_text(
            encoding="utf-8", errors="replace"
        )
    }
    assert actual_bridges == expected_bridges, (
        "ContextVar 子进程桥接点变化后必须同步审查："
        f"实际={sorted(actual_bridges)}，期望={sorted(expected_bridges)}"
    )


# ── 门 F 的显式分类：把子进程指向"另一个 profile"的 spawn 点 ────────────────
#
# ⛔ 旧判据只列了两个文件名（`tui_gateway/server.py` + `hermes_cli/web_server.py`），
# 这是按"文件名单"筛 —— 名单外的一律免检。实测代价：全仓真正把 HERMES_HOME
# 写进子进程 env 的函数有 10 个，旧门只罩 1 个（另一个 web_server.py 压根不写
# HERMES_HOME，等于在检查一个没有该缺陷的文件）。漏掉的里面有两个真缺陷：
# a2a `_forward_to_profile` 和 kanban `_default_spawn` —— 正是本门 docstring
# 自己描述的那个病（"HERMES_HOME 被 inline 设置，兄弟 key 因此被忘记"）。
#
# 新判据贴数据流：函数体内**既** spawn(env=)**又**写 HERMES_HOME，且写入值不是
# "当前 / 默认根"（`get_hermes_home()` / `get_default_hermes_root()`）⇒ 说明它
# 在把子进程指向**另一个** profile ⇒ 必须走共享契约。
#
# 闭集边界：只覆盖**仓内 Python 可见**的 spawn / env builder。第三方 SDK 内部
# 自己起的进程属于开集，由 MCP/LSP 的真实 child-env 集成测试兜底。
_SPAWN_SINKS = frozenset({
    "Popen", "run", "call", "check_call", "check_output",
    "create_subprocess_exec", "create_subprocess_shell",
    "execve", "execvpe", "spawnve", "posix_spawn", "posix_spawnp",
})
# 指向"当前 / 默认根"而非另一个 profile ⇒ 不在本缺陷作用域内（兄弟 key 由
# os.environ 原样继承本就正确）。
_CURRENT_HOME_RESOLVERS = frozenset({"get_hermes_home", "get_default_hermes_root"})
# 显式豁免（⛔ 每条必须带理由；⛔ 不许拿这里当消音桶）。
_SPAWN_REPOINT_EXEMPT = {
    # 子进程只跑 tools.skills_sync 做技能文件同步，不消费任何渠道凭据；
    # 它是在给**新建**的 profile 播种，不是把 agent 转到别的档案下运行。
    "hermes_cli/profiles.py::seed_profile_skills",
}


def _hermes_home_write_lines(tree, *, include_dict_literals: bool = True) -> list[int]:
    """AST 枚举 HERMES_HOME 的写入形态（下标赋值 / dict 字面量 / update）。

    ⚠️ `include_dict_literals` 存在的原因：`{"HERMES_HOME": x}` 有两种完全不同
    的用途 —— ①构造子进程 env（真缺陷面）②作为参数传给纯计算 helper 去算
    scope/tag/uid（例：tools/environments/local.py 的
    `_managed_terminal_profile_scope({"HERMES_HOME": profile_home})`、
    tools/process_registry.py 的 `_managed_terminal_identity(...)`）。光看字面量
    分不出来，会把 ② 误报成缺陷。
    ⇒ 分工：门 F 带 spawn 上下文，判得准，包含 dict 字面量；门 E 没有 spawn
    上下文，只管"往一个既有 env 映射里塞键"（Subscript / update）这种一定是
    在改环境的形态。两门合起来仍是闭集：读 pin 的文件若用 dict 字面量构造 env
    并 spawn，会被门 F 抓到。
    """
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "HERMES_HOME"
                ):
                    lines.append(node.lineno)
        elif (
            include_dict_literals
            and isinstance(node, ast.Dict)
            and any(
                isinstance(k, ast.Constant) and k.value == "HERMES_HOME"
                for k in node.keys
            )
        ):
            lines.append(node.lineno)
        elif isinstance(node, ast.Call) and (
            getattr(node.func, "attr", None) == "update"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Dict) and any(
                    isinstance(k, ast.Constant) and k.value == "HERMES_HOME"
                    for k in arg.keys
                ):
                    lines.append(node.lineno)
    return sorted(set(lines))


_CONTRACT_CALLS = ("apply_profile_scoped_env(", "apply_context_profile_scoped_env(")


def _hermes_home_writes_outside_the_contract(tree) -> list[int]:
    """写 HERMES_HOME 且**所在函数没走共享契约**的行号。

    同一函数里已经调过 `apply_*_profile_scoped_env` 的写入不算违规：契约已经把
    兄弟 key 一起补齐了，之后再显式带上 HERMES_HOME 是合法的（例如把 pin 钉成
    更具体的构造参数，见 tui_gateway/server.py 的注释）。
    """
    guarded: list[tuple[int, int]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(call in ast.unparse(fn) for call in _CONTRACT_CALLS):
            continue
        # 🔴 调过契约**不等于**这个函数里所有 HERMES_HOME 写入都合法。
        # 上一版给了**函数级**豁免 ⇒ 下面这种写法门是绿的（实测坐实）：
        #     apply_context_profile_scoped_env(env)
        #     env["HERMES_HOME"] = str(get_profile_dir(other))   # 契约后覆盖
        # 兄弟 key（WECOM_CLI_CONFIG_DIR…）仍指着契约设的那个 profile，
        # 而 HERMES_HOME 被指到另一个 —— 正是这道门要防的错配，却拿到免检。
        # ⭐ 与「按函数名/按文件判而不是按调用点判」同形。
        #
        # ⇒ 豁免只给「调过契约 **且** 写入的值仍指向当前/默认根」的函数。
        #   注释里那个合法用例（把 pin 钉成更具体的构造参数）值来自
        #   ``get_hermes_home()`` 一类，不会被 ``_repoints_at_another_profile``
        #   判成 re-point，因此不受影响。
        if _repoints_at_another_profile(fn):
            continue
        guarded.append((fn.lineno, getattr(fn, "end_lineno", None) or fn.lineno))
    return [
        line
        # ⚠️ 这里保持 ``include_dict_literals=False``：本门的作用域是「读了
        # profile pin 的模块」，把 dict literal 一并算进来会误报三处**根本不是
        # 子进程 env** 的构造（``_managed_terminal_profile_scope({...})`` /
        # ``_managed_terminal_profile_tag({...})`` 是查询用的字典；
        # ``tui_gateway/server.py`` 那处前一行就调了契约、值是同一个
        # profile_home）。误报会让门恒红 ⇒ 门失效。
        # ⭐ dict literal 这一路由下面
        #   ``test_dict_literal_env_builders_must_use_the_contract`` 单独覆盖，
        #   判据收窄到「该 dict 真的流向子进程 env」。
        for line in _hermes_home_write_lines(tree, include_dict_literals=False)
        if not any(start <= line <= end for start, end in guarded)
    ]


def _repoints_at_another_profile(fn_node) -> bool:
    """写入 HERMES_HOME 的值是否来自"当前 / 默认根"以外的来源。

    ⚠️ 值常常先落进局部变量再用（`home = str(get_hermes_home())` 然后
    `env={..., "HERMES_HOME": home}`）。只看写入点那一个表达式会把这种
    合法用法误判成 re-point，所以要在函数体内回溯变量来源。
    """
    local_sources: dict[str, list] = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    local_sources.setdefault(target.id, []).append(node.value)

    def _resolvers_of(value, depth: int = 0) -> set:
        names = {
            getattr(c.func, "id", None) or getattr(c.func, "attr", None)
            for c in ast.walk(value)
            if isinstance(c, ast.Call)
        }
        if depth < 4:  # 深度上限：防 a = b / b = a 这类互相引用打转
            for n in ast.walk(value):
                if isinstance(n, ast.Name) and n.id in local_sources:
                    for src in local_sources[n.id]:
                        names |= _resolvers_of(src, depth + 1)
        return names

    for node in ast.walk(fn_node):
        value = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "HERMES_HOME"
                ):
                    value = node.value
        elif isinstance(node, ast.Dict):
            for key, val in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "HERMES_HOME":
                    value = val
        if value is None:
            continue
        if not (_resolvers_of(value) & _CURRENT_HOME_RESOLVERS):
            return True
    return False


def test_no_spawn_site_re_points_hermes_home_on_its_own():
    """孪生门：把子进程指向另一个 profile 的每一处都必须走共享契约。

    HERMES_HOME 曾被到处 inline 设置，兄弟 key（WECOM_CLI_CONFIG_DIR）因此被
    忘记 —— 子进程指着上一个 profile 的凭据目录，找不到东西却报"未初始化"。
    """
    root = Path(__file__).resolve().parents[2]
    # ⛔ 按路径排除（不是按行内容），否则 `grep -v test` 会连 `testChannel`
    # 这类目标标识符一起滤掉。scripts/ 是开发脚本，不在产品运行路径上。
    skip = ("/tests/", "/.venv/", "/node_modules/", "/.git/", "/scripts/")
    offenders: dict[str, str] = {}
    scanned = 0

    for path in sorted(root.rglob("*.py")):
        if any(part in path.as_posix() for part in skip):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(root).as_posix()
        for fn_node in ast.walk(tree):
            if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            spawns = [
                n
                for n in ast.walk(fn_node)
                if isinstance(n, ast.Call)
                and (getattr(n.func, "attr", None) or getattr(n.func, "id", None))
                in _SPAWN_SINKS
                and any(kw.arg == "env" for kw in n.keywords)
            ]
            if not spawns or not _hermes_home_write_lines(fn_node):
                continue
            if not _repoints_at_another_profile(fn_node):
                continue  # 指向当前/默认根，不在本缺陷作用域内
            scanned += 1
            key = f"{rel}::{fn_node.name}"
            if key in _SPAWN_REPOINT_EXEMPT:
                continue
            src = ast.unparse(fn_node)
            if "apply_profile_scoped_env(" in src or "apply_context_profile_scoped_env(" in src:
                continue
            offenders[key] = f"写@L{_hermes_home_write_lines(fn_node)}"

    # 阳性对照：一个都没扫到说明扫描本身坏了，⛔ 不许当"闭集成立"。
    assert scanned, "校准失败：全仓一个 re-point spawn 点都没扫到"
    assert not offenders, (
        "这些函数把子进程指向另一个 profile 却自己 inline 写 HERMES_HOME，"
        f"兄弟 key 不会跟着走：{offenders}。必须改用 apply_profile_scoped_env。"
    )



# ════════ 判据自己的门：把真实漏洞摆在它面前 ════════
#
# ⭐ 判定一道门有不有用，唯一可信的方法不是读它的代码，是**拿真实漏洞跑一遍**。
# 下面两段是 RH 复审 P2-4 描述的攻击形态。上一版判据在**攻击 A 上是绿的**
# （实测坐实），因为它给了「调过契约的函数」**函数级**豁免 —— 契约调用之后
# 再把 HERMES_HOME 覆盖成另一个 profile，照样免检。
#
# ⛔ 这两条一旦变绿，说明判据又被改松了。

_ATTACK_OVERRIDE_AFTER_CONTRACT = '''
def build_child_env(profile):
    env = dict(os.environ)
    apply_context_profile_scoped_env(env)
    env["HERMES_HOME"] = str(get_profile_dir(profile))
    return env
'''

_ATTACK_BUILDER_SPLIT_FROM_SPAWN = '''
def _build_env(profile):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(get_profile_dir(profile))
    return env

def _spawn(profile):
    subprocess.Popen(["x"], env=_build_env(profile))
'''

_LEGIT_PIN_TO_CURRENT_ROOT = '''
def build_child_env():
    env = dict(os.environ)
    apply_context_profile_scoped_env(env)
    env["HERMES_HOME"] = str(get_hermes_home())
    return env
'''


def test_criterion_catches_override_after_contract_call():
    """🔴 契约调用**之后**覆盖成另一个 profile —— 上一版判据在这里是绿的。

    兄弟 key（WECOM_CLI_CONFIG_DIR…）仍指着契约设的 profile，
    HERMES_HOME 却被指到另一个 —— 正是这道门要防的错配。
    """
    hits = _hermes_home_writes_outside_the_contract(
        ast.parse(_ATTACK_OVERRIDE_AFTER_CONTRACT))
    assert hits, (
        "契约后覆盖成别的 profile 没被判为违规 —— "
        "函数级豁免又回来了（调过契约 ≠ 函数内所有写入都合法）")


def test_criterion_catches_builder_split_from_spawn():
    """🔴 builder 与 spawn 拆成两个函数 —— 孪生门（要求同函数内有 spawn）漏掉它。

    ⭐ 这条由本判据兜住：它扫**全仓所有**写 HERMES_HOME 的函数，
    ⛔ 不要求 spawn 在同一个函数里。按函数边界划范围正是漏洞的来源。
    """
    hits = _hermes_home_writes_outside_the_contract(
        ast.parse(_ATTACK_BUILDER_SPLIT_FROM_SPAWN))
    assert hits, "builder 单独 re-point 且不走契约，却没被判违规"


def test_criterion_does_not_flag_pinning_to_the_current_root():
    """⛔ 收紧判据不许误伤合法用例。

    ⭐ 这条是「不许弄坏原来对的」的那一半：只报 attack 不验 legit，
    判据可以简单地永远返回违规来"通过"上面两条。
    """
    hits = _hermes_home_writes_outside_the_contract(
        ast.parse(_LEGIT_PIN_TO_CURRENT_ROOT))
    assert not hits, (
        f"把 pin 钉成当前根是合法的（契约已跑过），却被判违规:{hits}")


_ATTACK_DICT_LITERAL_BUILDER = '''
def _build_env(profile):
    return {"HERMES_HOME": str(get_profile_dir(profile)), "PATH": "/usr/bin"}

def _spawn(profile):
    subprocess.Popen(["x"], env=_build_env(profile))
'''

_LEGIT_DICT_LITERAL_NOT_AN_ENV = '''
def retire(profile_home):
    tag = _managed_terminal_profile_tag({"HERMES_HOME": profile_home})
    return tag
'''


def _dict_literal_env_builders(tree) -> list[str]:
    """在 dict literal 里写 ``HERMES_HOME``、**且该 dict 真的流向子进程 env**
    的函数名。

    🔴 为什么要单独一条：``env["HERMES_HOME"] = v`` 和 ``{"HERMES_HOME": v}``
    是同一件事的两种写法，上一版只认前者 ⇒ 后者整个免检
    （RH 复审第二轮实测：在独立 builder 里用 dict literal 构造、
    由另一个函数 spawn，两条门全绿）。
    ⭐ 按「写法」列清单永远是开集。

    ⚠️ 但作用域必须**刚好等于**缺陷：只有流向子进程 env 的才算。
    仓里有三处 dict literal 写 ``HERMES_HOME`` 却根本不是 env
    （构造查询用的 scope / 算 tag），把它们算进来就是误报，
    而误报会让门恒红 ⇒ 门失效。
    ⇒ 判据 = 该 dict 被 ``return`` 出去，或直接作为 ``env=`` 实参。

    🔴 ⛔ **本判据是开集，⛔ 不许当保障。** RH 实测可绕过的写法至少有：
      * ``{**{"HERMES_HOME": v}, ...}``（dict 解包）
      * ``dict(HERMES_HOME=v)``（关键字构造）
      * 跨 helper 传递后再进 ``env=``
    ⭐ 与 wecom 那条同形：给写法列清单永远追不完。
    真正的闭集需要**驱动真实子进程 env 并断言四个 profile key 同源**
    —— 那要能起子进程或完整打桩 spawn 层，本轮没做。
    本条只作为早期信号：新增 env builder 时提醒作者走契约。
    """
    out: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        src = ast.unparse(fn)
        if "apply_profile_scoped_env(" in src or "apply_context_profile_scoped_env(" in src:
            continue

        def _is_home_dict(node) -> bool:
            return isinstance(node, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == "HERMES_HOME"
                for k in node.keys)

        # 直接 return 一个含 HERMES_HOME 的 dict literal
        flows = any(
            isinstance(n, ast.Return) and _is_home_dict(n.value)
            for n in ast.walk(fn))
        # 或者赋给某个名字，而这个名字被 return / 当作 env= 传出去
        named: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Assign) and _is_home_dict(n.value):
                named |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        for n in ast.walk(fn):
            if isinstance(n, ast.Return) and isinstance(n.value, ast.Name) \
                    and n.value.id in named:
                flows = True
            if isinstance(n, ast.Call):
                for kw in n.keywords:
                    if kw.arg == "env" and (
                        _is_home_dict(kw.value)
                        or (isinstance(kw.value, ast.Name) and kw.value.id in named)
                    ):
                        flows = True
        if flows and _repoints_at_another_profile(fn):
            out.append(fn.name)
    return out


def test_criterion_catches_dict_literal_in_a_separate_builder():
    """🔴 dict literal + builder/spawn 分离 —— 上一版判据在这里是绿的。"""
    assert _dict_literal_env_builders(ast.parse(_ATTACK_DICT_LITERAL_BUILDER)), (
        "dict literal 形式的 re-point env builder 没被判为违规")


def test_criterion_does_not_flag_dict_literals_that_are_not_env():
    """⛔ 不许误报：仓里有 dict literal 写 HERMES_HOME 却根本不是子进程 env。

    ⭐ 只报 attack 不验 legit，判据可以简单地永远返回违规来"通过"上一条；
    而误报会让门恒红 ⇒ 门失效，比没有门更坏。
    """
    assert not _dict_literal_env_builders(ast.parse(_LEGIT_DICT_LITERAL_NOT_AN_ENV))


def test_dict_literal_env_builders_must_use_the_contract():
    """全仓扫描：dict literal 构造的子进程 env 也必须走契约。"""
    root = Path(__file__).resolve().parents[2]
    skip = ("/tests/", "/.venv/", "/node_modules/", "/.git/", "/scripts/")
    offenders: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in path.as_posix() for part in skip):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        rel = path.relative_to(root).as_posix()
        # ⚠️ 沿用孪生门**同一份**豁免名单 —— ⛔ 不另立第二份。
        # 两份名单必然漂移，而漂移出来的那一半就是免检区。
        hits = [h for h in _dict_literal_env_builders(tree)
                if f"{rel}::{h}" not in _SPAWN_REPOINT_EXEMPT]
        if hits:
            offenders[rel] = hits
    assert not offenders, (
        f"这些函数用 dict literal 构造指向另一个 profile 的子进程 env，"
        f"却没走共享契约，兄弟 key 不会跟着走：{offenders}")
