import json
import os
import textwrap
from contextlib import contextmanager

import pytest

from agent import secret_scope
from gateway.session_context import set_zettlab_turn_id, zettlab_turn_id
from tools import terminal_tool


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch):
    previous_multiplex = secret_scope.is_multiplex_active()
    previous_turn_id = zettlab_turn_id()
    scope_token = secret_scope.set_secret_scope(None)
    secret_scope.set_multiplex_active(True)
    set_zettlab_turn_id("")
    monkeypatch.setattr(terminal_tool, "_CONNECTOR_RUNTIME_ROOT_ANCHOR", None)
    yield
    set_zettlab_turn_id(previous_turn_id)
    secret_scope.set_multiplex_active(previous_multiplex)
    secret_scope.reset_secret_scope(scope_token)


@contextmanager
def scope(token: str, **runtime_values: str):
    scope_token = secret_scope.set_secret_scope({
        "ZETTLAB_AGENT_ACTION_TOKEN": token,
        "ZET_CHAT_APPEND_URL": "http://127.0.0.1:19090/api/v1/internal/chat/append",
        **runtime_values,
    })
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(scope_token)


def configure(monkeypatch, tmp_path, body: str):
    script = (
        tmp_path / "presets" / "skills" / "overseas-connect" / "scripts" / "connect.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    (script.parent.parent / "manifest.yaml").write_text(
        "id: overseas-connect\n"
        "runtime_capabilities:\n"
        "  - zettlab.agent_action_token_fd.v1\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TERMINAL_ENV", "local")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(tmp_path / "presets"))
    monkeypatch.setattr(
        terminal_tool,
        "_connector_runtime_path_is_trusted",
        lambda *_args, **_kwargs: True,
    )
    return script


def canonical(command: str = "capability") -> str:
    return (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/overseas-connect/'
        f'scripts/connect.py" {command} --platform telegram'
    )


def cardonly(command: str = "status-card") -> str:
    """无参形态：改版后 AI 面前只剩这一条命令，它不带 --platform。"""

    return (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/overseas-connect/'
        f'scripts/connect.py" {command}'
    )


def test_direct_runner_injects_token_only_through_fd(monkeypatch, tmp_path):
    configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys

        descriptor = int(os.environ.pop("ZETTLAB_AGENT_ACTION_TOKEN_FD"))
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            token = stream.read().decode("ascii")
        print("token=" + token)
        print(json.dumps({
            "argv": sys.argv[1:],
            "token_ok": token == "a" * 64,
            "token_env_absent": "ZETTLAB_AGENT_ACTION_TOKEN" not in os.environ,
            "turn_ok": os.environ.get("ZETTLAB_TURN_ID") == "turn-current",
        }))
        """,
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "wrong-global-token")
    set_zettlab_turn_id("turn-current")

    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(cardonly(), task_id="overseas-connect")
        )

    assert result["overseas_connect_direct"] is True
    assert result["exit_code"] == 0
    output_lines = result["output"].splitlines()
    # The final generic response sanitizer normalizes the runner's streaming
    # marker to its public `***` form before the tool result is exposed.
    assert output_lines[0] == "token=***"
    assert json.loads(output_lines[1]) == {
        # 载体从 `capability --platform telegram` 换成无参 status-card:带 --platform 的
        # 5-token 子命令已整组从受信通道摘除(见 _OVERSEAS_CONNECT_PLATFORM_COMMANDS)。
        # ⚠️ 本用例钉的契约(token 只经 FD 递、⛔ 不进环境变量)没有变,只是不能再拿一条
        # 已被禁的命令当载体 —— 那样它会连带把禁令本身钉反。
        "argv": ["status-card"],
        "token_ok": True,
        "token_env_absent": True,
        "turn_ok": True,
    }
    assert "token=***" in result["output"]
    assert "a" * 64 not in result["output"]
    assert "wrong-global-token" not in result["output"]


def test_direct_runner_uses_loopback_base_from_profile_scope(monkeypatch, tmp_path):
    configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        print(json.dumps({
            "base": os.environ.get("ZETTLAB_LOCAL_SERVER_URL", ""),
            "append": os.environ.get("ZET_CHAT_APPEND_URL", ""),
        }))
        """,
    )

    with scope(
        "a" * 64,
        ZET_CHAT_APPEND_URL="http://127.0.0.1:19090/api/v1/internal/chat/append",
    ):
        result = json.loads(
            terminal_tool.terminal_tool(cardonly(), task_id="overseas-connect")
        )

    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "base": "http://127.0.0.1:19090",
        "append": "",
    }


def test_direct_runner_blocks_invalid_profile_url_instead_of_falling_back(monkeypatch, tmp_path):
    configure(
        monkeypatch,
        tmp_path,
        "raise AssertionError('runner must not start')\n",
    )
    monkeypatch.setenv("ZETTLAB_LOCAL_SERVER_URL", "http://127.0.0.1:9090")

    with scope(
        "a" * 64,
        ZET_CHAT_APPEND_URL="https://outside.example/api/v1/internal/chat/append",
    ):
        result = json.loads(
            terminal_tool.terminal_tool(cardonly(), task_id="overseas-connect")
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_scope_unavailable"


def test_noncanonical_command_never_acquires_a_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "print('must not run')\n")

    def forbidden():
        raise AssertionError("invalid command acquired a token")

    monkeypatch.setattr(
        "tools.environments.local.build_overseas_connect_runtime_env",
        forbidden,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                canonical("capability; echo escaped"),
                task_id="overseas-connect-invalid",
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_command_blocked"


def test_unsupported_platform_never_acquires_a_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "print('must not run')\n")

    def forbidden():
        raise AssertionError("unsupported platform acquired a token")

    monkeypatch.setattr(
        "tools.environments.local.build_overseas_connect_runtime_env",
        forbidden,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                canonical().replace("telegram", "unlisted-platform"),
                task_id="overseas-connect-unsupported-platform",
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_command_blocked"


def test_status_card_runs_without_a_platform_argument(monkeypatch, tmp_path):
    """无参入口命令必须能进受信通道。

    原实现硬要求 len(tokens)==5 且 tokens[3]=="--platform"，于是 status-card
    这条三 token 的命令永远解析失败、拿不到 action token。改版把 AI 面前收缩
    到只剩这一条命令之后，这条路断掉就等于整个功能在真机上不可用。
    """

    configure(
        monkeypatch,
        tmp_path,
        """
        import json
        import os
        import sys

        descriptor = int(os.environ.pop("ZETTLAB_AGENT_ACTION_TOKEN_FD"))
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            token = stream.read().decode("ascii")
        print(json.dumps({
            "argv": sys.argv[1:],
            "token_ok": token == "a" * 64,
            "token_env_absent": "ZETTLAB_AGENT_ACTION_TOKEN" not in os.environ,
        }))
        """,
    )

    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(cardonly(), task_id="overseas-connect-status")
        )

    # ⛔ 不要拿 overseas_connect_direct 当「放行了」的判据：
    # _overseas_connect_blocked_result() 在**拒绝**时同样写 direct=True，
    # 只有 overseas_connect_blocked 的缺席才真正区分放行与拒绝。
    assert result.get("overseas_connect_blocked") is None
    assert result["exit_code"] == 0
    assert json.loads(result["output"]) == {
        "argv": ["status-card"],
        "token_ok": True,
        "token_env_absent": True,
    }


def test_status_card_rejects_a_platform_argument(monkeypatch, tmp_path):
    """无参入口命令不接受 --platform。

    改版的收益正是「AI 不再需要认平台」。静默放行会把平台重新摆回 AI 面前，
    而脚本那边 status-card 根本不读这个参数 —— 于是「我指定了平台」变成一个
    不会报错的错觉。
    """

    configure(monkeypatch, tmp_path, "print('must not run')\n")

    def forbidden():
        raise AssertionError("status-card with a platform acquired a token")

    monkeypatch.setattr(
        "tools.environments.local.build_overseas_connect_runtime_env",
        forbidden,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                canonical("status-card"),
                task_id="overseas-connect-status-with-platform",
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_command_blocked"


def test_status_card_shell_injection_never_acquires_a_token(monkeypatch, tmp_path):
    """放宽 token 数量之后，注入形态仍然拿不到 action token。

    ⚠️ 这条守的是**结果**，不是某一道具体的墙。注入命令被 shlex 切成 6 个
    token，它同时撞上标点守卫和 3-token 的数量校验；把标点守卫整段删掉，这条
    测试依然绿（数量校验会拒）。

    进一步说：在当前**精确 arity** 的实现下，纯标点 token 根本无处可放——3
    token 的三个位置被 python / 脚本路径 / 子命令占死，5 token 的被 --platform
    与平台名占死。标点守卫因此是冗余的防御深度，它的价值只在将来有人放宽 arity
    时才兑现。别把这条测试当成「标点守卫还在」的证据。
    """

    configure(monkeypatch, tmp_path, "print('must not run')\n")

    def forbidden():
        raise AssertionError("injected status-card acquired a token")

    monkeypatch.setattr(
        "tools.environments.local.build_overseas_connect_runtime_env",
        forbidden,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                cardonly("status-card; echo escaped"),
                task_id="overseas-connect-status-injection",
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_command_blocked"


@pytest.mark.parametrize("action", ["connect-card", "disconnect"])
def test_card_and_disconnect_actions_are_refused_by_the_trusted_runner(
    monkeypatch, tmp_path, action
):
    """⚠️ 本用例原来断言的是**反面**:「connect-card / disconnect 会被同一个受信 runner
    接管并执行」。那条断言把一个缺陷钉成了契约。

    真实设计(2026-08-07 拍板)是「不再弹单平台卡 → 弹一张海外渠道列表卡 → 用户在卡内
    自己选平台」,即 AI 只剩弹卡。改版加了 status-card 与 SKILL.md 的文字契约,却没拆掉
    这两条旧入口;而模型不受自然语言契约约束,直接发 terminal 命令就能**替用户选定平台**、
    绕开总列表卡。原断言绿着,恰恰证明那条路还通。

    现在反过来钉:这两条命令必须**拿不到受信通道**。
    """
    configure(
        monkeypatch,
        tmp_path,
        "raise AssertionError('a retired platform command reached the script')\n",
    )

    with scope("a" * 64):
        raw = terminal_tool.terminal_tool(
            canonical(action),
            task_id=f"overseas-connect-{action}",
        )

    payload = json.loads(raw)
    # ⭐ 判据是「脚本有没有跑起来 / 有没有拿到 token」,⛔ 不是 `overseas_connect_direct`:
    # 那个标记只说明「这条命令归 overseas-connect 这条面处理」,**阻断响应也带它**。
    # 拿它当判据会把「已被正确阻断」误判成「仍然可达」。
    assert payload.get("overseas_connect_blocked") is True, (
        f"RETIRED_PLATFORM_COMMANDS_MUST_NOT_REACH_THE_RUNNER: `{action} --platform telegram` "
        f"没有被阻断 —— 模型可以据此替用户选定平台,绕开海外渠道列表卡。实际返回:{raw[:400]}"
    )
    assert payload.get("output") == "", (
        f"RETIRED_PLATFORM_COMMANDS_MUST_NOT_REACH_THE_RUNNER: 脚本被执行了(output={payload.get('output')!r})"
        " —— 夹具里那条 raise 就是为了让「脚本跑起来」这件事无法被静默吞掉"
    )
    assert "a" * 64 not in raw, "被拒的命令不该看见 action token"


def test_manifest_without_fd_capability_blocks_before_secret_acquisition(
    monkeypatch, tmp_path
):
    script = configure(monkeypatch, tmp_path, "print('must not run')\n")
    (script.parent.parent / "manifest.yaml").write_text(
        "id: overseas-connect\nruntime_capabilities: []\n",
        encoding="utf-8",
    )

    def forbidden():
        raise AssertionError("invalid manifest acquired a token")

    monkeypatch.setattr(
        "tools.environments.local.build_overseas_connect_runtime_env",
        forbidden,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                cardonly(), task_id="overseas-connect-old-bundle"
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_runtime_capability_unavailable"


def test_unscoped_turn_never_uses_a_process_environment_token(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "print('must not run')\n")
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "a" * 64)

    result = json.loads(
        terminal_tool.terminal_tool(cardonly(), task_id="overseas-connect-unscoped")
    )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_scope_unavailable"


def test_runner_failure_is_not_misreported_as_missing_authorization(
    monkeypatch, tmp_path
):
    configure(monkeypatch, tmp_path, "print('must not run')\n")

    def broken_runner(**_kwargs):
        raise OSError("runner unavailable")

    monkeypatch.setattr(
        "tools.trusted_direct_runner.run_trusted_python_script",
        broken_runner,
    )
    with scope("a" * 64):
        result = json.loads(
            terminal_tool.terminal_tool(
                cardonly(), task_id="overseas-connect-runner-failure"
            )
        )

    assert result["overseas_connect_blocked"] is True
    assert result["errorCode"] == "overseas_connect_execution_unavailable"
    assert "a" * 64 not in json.dumps(result)


def test_overseas_connect_platform_allowlist_is_the_agreed_set():
    """受信通道的平台集合必须与另外两个仓同集。

    三处字面量分散在三个仓里,谁都 import 不到谁:
      - hermes  tools/terminal_tool.py   _OVERSEAS_CONNECT_PLATFORMS(本处)
      - presets skills/cn/overseas-connect/scripts/connect.py SUPPORTED_PLATFORMS
      - board   internal/agent/handler/via_skill_channels.go  imConnectPlatforms
    跨仓比对没有基建,所以每一侧各钉各的,任一侧改动都会先红在自己这边。

    2026-08-07 摘 WhatsApp 时,presets 与板端都改了、**这一侧漏了** —— 当时没有
    任何东西会红。孪生枚举扫了两个仓却漏掉第三个,这条门就是补上那一格。
    """

    assert terminal_tool._OVERSEAS_CONNECT_PLATFORMS == frozenset({
        "telegram", "slack", "discord",
    })


# ⭐ 模型视角的收尾自证:判据是「**该泄漏的东西有没有泄漏**」,⛔ 不是「命令还在不在」。
#
# 「命令不存在」是形状(开集):换个写法、加个空格、走另一条子命令都可能绕过。真正要守住的
# 产品边界只有两条,直接钉它们:
#   ① 海外 IM 的**绑定状态**不得进入聊天记录 —— 所以夹具脚本把那几个字段原样打印出来,
#      只要命令被放行,它们就会出现在 tool 结果里,进而进入模型/tool transcript;
#   ② 平台必须由**用户在总卡内选** —— 所以模型指定平台的那条路必须拿不到受信通道。
#
# 三条命令 × 三个平台全枚举:⛔ 别只测被点名的那一条,兄弟路径是同一个形状。
@pytest.mark.parametrize("action", ["capability", "connect-card", "disconnect"])
@pytest.mark.parametrize("platform", ["telegram", "slack", "discord"])
def test_a_model_cannot_reach_binding_state_through_any_platform_command(
    monkeypatch, tmp_path, action, platform
):
    leak_markers = ("ZETT_LEAK_configured", "ZETT_LEAK_paired_owner", "ZETT_LEAK_created_via_chat")
    configure(
        monkeypatch,
        tmp_path,
        # 夹具刻意**泄漏**:命令一旦被放行,这些字段就会进 tool 结果 —— 那正是产品禁止的。
        "print('" + " ".join(leak_markers) + "')\n",
    )

    command = (
        'python3 "$ZETTLAB_PRESETS_DIR/skills/overseas-connect/'
        f'scripts/connect.py" {action} --platform {platform}'
    )
    with scope("a" * 64):
        raw = terminal_tool.terminal_tool(command, task_id=f"model-view-{action}-{platform}")

    for marker in leak_markers:
        assert marker not in raw, (
            f"BINDING_STATE_MUST_NOT_REACH_THE_TRANSCRIPT: 模型发 `{action} --platform {platform}` "
            f"拿到了绑定状态({marker})—— 海外 IM 的绑定状态进入聊天记录是产品明令禁止的。"
            f"实际返回:{raw[:400]}"
        )
    assert "a" * 64 not in raw, "被拒的命令不该看见 action token"


# 反面同格:唯一留给 AI 的无参入口必须**仍然可用**。
# ⛔ 少了这一格,上面那组会因为「整个 skill 都被打死了」而全绿 —— 那是把功能删没了,
# 不是把边界守住了。
def test_the_card_only_entry_still_works_for_the_model(monkeypatch, tmp_path):
    configure(monkeypatch, tmp_path, "print('CARD_ONLY_REACHED')\n")

    with scope("a" * 64):
        raw = terminal_tool.terminal_tool(cardonly(), task_id="model-view-status-card")

    assert "CARD_ONLY_REACHED" in raw, (
        "唯一留给 AI 的无参入口 status-card 也够不着了 —— 上面那组「拿不到平台能力」"
        f"会因此变成重言式。实际返回:{raw[:400]}"
    )
