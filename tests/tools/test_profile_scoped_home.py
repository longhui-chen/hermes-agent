"""交互式会话(slash_worker)子进程环境的 profile 作用域契约。

背景(云真机 cloud-gt002 实测):AI 助手在会话里自己调 lark-cli 时报 "not bound",
于是每次都提议重新绑定 —— 而设备上授权早就成功过。根因是子进程环境里**根本没有
HOME**:lark-cli 只认 ``$HOME``(绑定在 ``$HOME/.lark-cli/hermes/config.json``、
密钥在 ``$HOME/.local/share/lark-cli/*.enc``;二进制里除 HERMES_HOME/OPENCLAW_HOME
外没有任何配置路径键),HOME 没设时回落 ``/root``。

⭐ 而 ``/root`` 是**全 agent 共用**的:设备上三个 profile 各自绑着**不同的真人**
(appId 与绑定用户指纹互不相同)⇒ HOME 塌成共享不是"配置串了",是一个 agent 拿
另一个人的身份去操作。所以下面这些断言是安全边界,⛔ 不是整洁度。

⚠️ **这个文件的第一版是假绿,教训写在这里以免重犯**:那一版用自己拼的辅助函数去调
``build_subprocess_env``,把 ``inherit_profile_home=True`` 写死在辅助函数里 ⇒
**把生产代码的 True 改回 False,测试照样全绿**。它测的是替身,不是接线。
⇒ 现在一律**构造真正的 ``_SlashWorker`` 并断言 ``Popen(env=...)``**,与
tests/tui_gateway/test_slash_worker_profile_home.py 同一打法。
⇒ ⭐ 通则:**一条新测试写完先做一次逆改;逆改不红 ⇒ 这条测试不算存在。**
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _spawn_worker_env(profile_home, monkeypatch):
    """起一个真的 _SlashWorker,把它交给 Popen 的那份环境原样取回来。

    ⛔ 不许改成"自己拼一个 build_subprocess_env 调用" —— 那样 tui_gateway/server.py
    里的实参(尤其 inherit_profile_home)就不参与判定了,逆改会打不红。
    """
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.delenv("HOME", raising=False)
    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.stdout = MagicMock()
        mock_popen.return_value.stderr = MagicMock()
        from tui_gateway.server import _SlashWorker

        _SlashWorker(session_key="k", model="m", profile_home=str(profile_home))
        assert mock_popen.called, "Popen 没被调用,测试没打到接线"
        return mock_popen.call_args[1]["env"]


def _profile(tmp_path, name):
    """造一个形如 <hermes_home>/profiles/<agent> 且带 home/ 的 profile。"""
    profile = tmp_path / "profiles" / name
    (profile / "home").mkdir(parents=True)
    return profile


# ⭐ 承重断言:HOME 必须跟着**本会话**的 profile 走。
# 逆改口径:把 tui_gateway/server.py 的 inherit_profile_home 改回 False ⇒ 本条必须红。
def test_home_follows_the_session_profile(tmp_path, monkeypatch):
    session = _profile(tmp_path, "session-agent")
    env = _spawn_worker_env(session, monkeypatch)
    assert env.get("HOME") == str(session / "home"), (
        "HOME_MUST_FOLLOW_THE_SESSION_PROFILE: got "
        + repr(env.get("HOME"))
        + " —— HOME 指错 profile 等于让这个 agent 读另一个真人的凭据库"
    )


# extra 在工厂里最后应用,是"调用方 always wins"的既有语义;这条钉住它没被打掉。
def test_hermes_home_is_the_session_profile(tmp_path, monkeypatch):
    session = _profile(tmp_path, "session-agent")
    env = _spawn_worker_env(session, monkeypatch)
    assert env.get("HERMES_HOME") == str(session), (
        "HERMES_HOME_MUST_BE_THE_SESSION_PROFILE: got " + repr(env.get("HERMES_HOME"))
    )


# wecom-cli 只认 WECOM_CLI_CONFIG_DIR(不认 HOME 也不认 HERMES_HOME)⇒ 必须一起改指。
# ⛔ 少了这条,企微会在"凭据明明在盘上"的设备上报 not initialised。
def test_wecom_config_dir_follows_the_session_profile(tmp_path, monkeypatch):
    session = _profile(tmp_path, "session-agent")
    env = _spawn_worker_env(session, monkeypatch)
    assert env.get("WECOM_CLI_CONFIG_DIR") == str(session / "wecom-cli-config"), (
        "WECOM_CONFIG_DIR_MUST_FOLLOW_THE_SESSION_PROFILE: got "
        + repr(env.get("WECOM_CLI_CONFIG_DIR"))
    )


_SCOPED_KEYS = ("HOME", "HERMES_HOME", "WECOM_CLI_CONFIG_DIR")


def _scoped(env):
    """三个 profile 作用域键一起取。

    ⛔ 不拆成三条 assert:第一条红会把后两条**遮掉**(本文件第一版正是被这类遮蔽坑过),
    整字典比一次,任何一次红都把三个键的实际值一起摆出来。
    """
    return {key: env.get(key) for key in _SCOPED_KEYS}


def _expected(profile):
    return {
        "HOME": str(profile / "home"),
        "HERMES_HOME": str(profile),
        "WECOM_CLI_CONFIG_DIR": str(profile / "wecom-cli-config"),
    }


# 🔴 承重:**ambient pin 不许压过构造参数**。
#
# 为什么这条必须存在(实测出来的,不是推演):hermes_constants._profile_home_path 取
# HERMES_HOME 的顺序是 `get_hermes_home_override() or env["HERMES_HOME"] or os.getenv(...)`
# —— **context override 排在 env 字典前面**。而 HOME 正是由它派生的。
# ⇒ 只要构造 worker 时外层还挂着**别的会话**的 pin,就会劈叉成:
#     HERMES_HOME / WECOM_CLI_CONFIG_DIR = 本会话(apply_profile_scoped_env + extra 写的)
#     HOME                                = **别的会话**(ambient pin 派生的)
#   而 lark-cli 只认 $HOME ⇒ 这个 worker 会拿**另一个真人**的凭据库去操作。
#
# ⚠️ 今天四条构造路径都还没踩到:唯一挂着 pin 的那条(run@9442 → _restart_slash_worker)
# 的 pin 与 session["profile_home"] **同源**,所以两者恰好一致。⇒ 这是**潜伏陷阱不是活缺陷**。
# 但"正确性靠两个来源恰好相等"不是闭集判据:任何新构造点只要落在别的会话的 pin 作用域里,
# 就静默劈叉、且无日志。⇒ 由本条把"构造参数是唯一真相源"钉死。
def test_ambient_pin_must_not_override_the_constructor_profile(tmp_path, monkeypatch):
    other = _profile(tmp_path, "other-agent")  # 必须真建出 home/,否则派生不出错误的 HOME
    session = _profile(tmp_path, "session-agent")
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = set_hermes_home_override(str(other))
    try:
        env = _spawn_worker_env(session, monkeypatch)
    finally:
        reset_hermes_home_override(token)

    assert _scoped(env) == _expected(session), (
        "AMBIENT_PIN_MUST_NOT_OVERRIDE_THE_CONSTRUCTOR_PROFILE: 实际 "
        + repr(_scoped(env))
        + " —— 构造参数才是本会话的唯一真相源;劈叉意味着 lark-cli 读另一个真人的凭据库"
    )


# ⭐ 走**真正的重启入口**,不是直接 new。
# 理由:worker 崩了/切模型/压缩后换 session key/一轮结束 restore,都从 _restart_slash_worker 进 ——
# 它是本仓里 _SlashWorker 的**唯一构造点**(AST 全树扫描,见下面那道门)。
# 「重启后掉进共享 HOME」正是最难发现的形态:没有报错、没有日志,只是身份换了个人。
def test_restart_entry_keeps_the_session_profile(tmp_path, monkeypatch):
    session_profile = _profile(tmp_path, "session-agent")
    monkeypatch.setenv("TERMINAL_HOME_MODE", "profile")
    monkeypatch.delenv("HOME", raising=False)

    with patch("subprocess.Popen") as mock_popen:
        mock_popen.return_value.stdout = MagicMock()
        mock_popen.return_value.stderr = MagicMock()
        from tui_gateway import server

        session = {
            "session_key": "k",
            # 非 None 才会真的重启(None 表示这个会话从没起过 worker)
            "slash_worker": MagicMock(),
            "agent": MagicMock(model="m"),
            "profile_home": str(session_profile),
        }
        server._restart_slash_worker("sid-not-registered", session)
        assert mock_popen.called, "Popen 没被调用 —— 重启路径没走到,这条测试是空转"
        env = mock_popen.call_args[1]["env"]

    assert _scoped(env) == _expected(session_profile), (
        "RESTART_MUST_NOT_FALL_BACK_TO_A_SHARED_HOME: 实际 " + repr(_scoped(env))
    )


# 交替构造两个 profile ⇒ 各自指各自。钉住"上一次构造的残留"不会漏给下一个会话。
def test_two_profiles_do_not_share_a_home(tmp_path, monkeypatch):
    first = _profile(tmp_path, "agent-one")
    second = _profile(tmp_path, "agent-two")

    got = {
        "one": _scoped(_spawn_worker_env(first, monkeypatch)),
        "two": _scoped(_spawn_worker_env(second, monkeypatch)),
        "one-again": _scoped(_spawn_worker_env(first, monkeypatch)),
    }
    assert got == {
        "one": _expected(first),
        "two": _expected(second),
        "one-again": _expected(first),
    }, "PROFILES_MUST_NOT_BLEED_ACROSS_CONSTRUCTIONS: 实际 " + repr(got)


# ⛔ HOME 是**策略产物**,不是路径事实:它归 apply_subprocess_home_env 的
# TERMINAL_HOME_MODE 三档策略管。这条钉住我们没把它塞进路径事实那一侧 ——
# 塞进去就等于造出第二套 HOME 政策,把用户钉死的 TERMINAL_HOME_MODE=real 一并改写。
def test_profile_scoped_env_does_not_own_home(tmp_path):
    session = _profile(tmp_path, "session-agent")
    from hermes_constants import apply_profile_scoped_env

    base: dict[str, str] = {}
    apply_profile_scoped_env(base, session)
    assert "HOME" not in base, (
        "PROFILE_SCOPED_ENV_MUST_NOT_OWN_HOME: HOME 出现在路径事实里 —— "
        "它归 apply_subprocess_home_env 的 TERMINAL_HOME_MODE 策略管"
    )
