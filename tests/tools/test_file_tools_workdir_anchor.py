"""受管网关下文件工具相对路径锚到平台 output 目录的回归测试。

multiplex 网关形态里终端 live cwd / 注册 session cwd / $TERMINAL_CWD 都无人
供给，改动前相对路径兜底到 root 守护进程的 HOME：``write_file("notes.md")``
落在 scope 外被守卫 403，``read_file("Documents/x.md")`` 报出误导性的
``/root/...`` 路径。这里钉住新的兜底层：

  - 受管网关 + output 可用 + 无 session cwd → 锚到 ``ZET_AGENT_OUTPUT_DIR``；
  - 已有 session cwd（live 记录或注册 override）时优先级不变，不被 output 抢走；
  - 非受管网关行为与改动前逐字一致；
  - output 不可用（未设置 / 非绝对 / 目录不存在）时沿用原兜底，绝不抛错。
"""

import pytest

import tools.file_tools as ft
import tools.terminal_tool as terminal_tool


@pytest.fixture
def _managed_gateway(tmp_path, monkeypatch):
    """受管网关形态：三个常规锚点全空，进程 cwd 是守护进程 HOME 的类比。"""
    output = tmp_path / "output"
    daemon_home = tmp_path / "daemon-home"
    output.mkdir()
    daemon_home.mkdir()
    monkeypatch.chdir(daemon_home)
    monkeypatch.setattr(terminal_tool, "_session_cwd", {})
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    monkeypatch.setenv("HERMES_MANAGED_GATEWAY", "1")
    return output, daemon_home


def test_managed_gateway_anchors_relative_path_to_output_dir(_managed_gateway, monkeypatch):
    """无任何 session cwd 时，相对路径必须锚到平台 output 目录而非进程 cwd。"""
    output, daemon_home = _managed_gateway
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))

    resolved = ft._resolve_path_for_task("notes.md", task_id="mux-session")

    assert resolved.is_absolute()
    assert resolved == output / "notes.md"
    assert not str(resolved).startswith(str(daemon_home))


def test_profile_scope_output_dir_wins_over_environ(_managed_gateway, monkeypatch):
    """multiplex 下 os.environ 可能是别的 profile 的值，scope 供给必须优先。"""
    from tests.tools._profile_scope import mux_profile_scope

    output, daemon_home = _managed_gateway

    with mux_profile_scope(
        monkeypatch,
        {"ZET_AGENT_OUTPUT_DIR": str(output)},
        poison_environ=True,
    ):
        resolved = ft._resolve_path_for_task("notes.md", task_id="mux-session")

    assert resolved == output / "notes.md"


def test_live_session_cwd_wins_over_output_dir(_managed_gateway, monkeypatch, tmp_path):
    """终端已 cd 过的 live cwd 仍是最高优先级，不被 output 抢走。"""
    output, daemon_home = _managed_gateway
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    terminal_tool.record_session_cwd("mux-session", str(workspace))

    resolved = ft._resolve_path_for_task("notes.md", task_id="mux-session")

    assert resolved == workspace / "notes.md"
    assert not str(resolved).startswith(str(output))


def test_registered_task_cwd_override_wins_over_output_dir(_managed_gateway, monkeypatch, tmp_path):
    """TUI/Desktop 注册的 session cwd override 优先级同样不被 output 抢走。"""
    output, daemon_home = _managed_gateway
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    terminal_tool.register_task_env_overrides("mux-desktop", {"cwd": str(workspace)})

    resolved = ft._resolve_path_for_task("notes.md", task_id="mux-desktop")

    assert resolved == workspace / "notes.md"
    assert not str(resolved).startswith(str(output))


def test_non_managed_gateway_ignores_output_dir(_managed_gateway, monkeypatch):
    """CLI / 桌面 / 容器等非受管形态：即使 env 存在也一字不改，仍走进程 cwd。"""
    output, daemon_home = _managed_gateway
    monkeypatch.delenv("HERMES_MANAGED_GATEWAY", raising=False)
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))

    resolved = ft._resolve_path_for_task("notes.md", task_id="mux-session")

    assert resolved.is_absolute()
    assert resolved == (daemon_home / "notes.md").resolve()
    assert not str(resolved).startswith(str(output))


@pytest.mark.parametrize("case", ["unset", "relative", "missing-dir"])
def test_unusable_output_dir_falls_back_to_process_cwd(_managed_gateway, monkeypatch, tmp_path, case):
    """output 不可用时视为无此层：沿用原进程 cwd 兜底，且绝不抛异常。"""
    output, daemon_home = _managed_gateway
    if case == "unset":
        monkeypatch.delenv("ZET_AGENT_OUTPUT_DIR", raising=False)
    elif case == "relative":
        monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", "relative/output")
    else:
        monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(tmp_path / "never-created"))

    resolved = ft._resolve_path_for_task("notes.md", task_id="mux-session")

    assert resolved.is_absolute()
    assert resolved == (daemon_home / "notes.md").resolve()
