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


class TestV4AHeaderRewrite:
    """V4A header 路径必须换成本层已解析、已加锁、已过 scope 检查的绝对路径。

    shell 层按自己的 cwd 解析相对 header，和本层的解析基准不是同一个——检查的
    文件和真正被 patch 的文件因此可能不是一个（Codex review P1）。
    """

    def test_update_header_uses_resolved_path(self):
        patch = "*** Begin Patch\n*** Update File: notes.md\n@@\n-a\n+b\n*** End Patch"
        out = ft._rewrite_v4a_header_paths(patch, {"notes.md": "/out/notes.md"})
        assert "*** Update File: /out/notes.md" in out
        assert "-a\n+b" in out

    def test_move_header_rewrites_both_endpoints(self):
        patch = "*** Begin Patch\n*** Move File: a.txt -> b.txt\n*** End Patch"
        out = ft._rewrite_v4a_header_paths(
            patch, {"a.txt": "/out/a.txt", "b.txt": "/out/b.txt"}
        )
        assert "*** Move File: /out/a.txt -> /out/b.txt" in out

    def test_unresolved_path_is_left_alone(self):
        patch = "*** Begin Patch\n*** Add File: keep.md\n*** End Patch"
        assert ft._rewrite_v4a_header_paths(patch, {"other.md": "/out/other.md"}) == patch
        assert ft._rewrite_v4a_header_paths(patch, {}) == patch

    def test_body_lines_are_never_touched(self):
        # 正文里出现同名字符串不能被当成 header 改掉。
        patch = (
            "*** Begin Patch\n*** Update File: x.py\n@@\n-print('x.py')\n"
            "+print('x.py updated')\n*** End Patch"
        )
        out = ft._rewrite_v4a_header_paths(patch, {"x.py": "/out/x.py"})
        assert "*** Update File: /out/x.py" in out
        assert "-print('x.py')" in out
        assert "+print('x.py updated')" in out

    def test_no_space_after_asterisks_still_rewritten(self):
        # patch_parser 容忍 ``***Update File:``，路径检查也按这个宽松度做，
        # 重写漏掉它就会让一条能跑通的 patch 绕过重写。
        patch = "*** Begin Patch\n***Update File: notes.md\n*** End Patch"
        out = ft._rewrite_v4a_header_paths(patch, {"notes.md": "/out/notes.md"})
        assert "/out/notes.md" in out


class TestOpsPathBackendScoping:
    """把解析结果交给 ops 层，只在 ops 层作用于本机文件系统时成立。

    ssh 后端在对端执行读/搜/patch：本机绝对路径在那边指向另一个文件或根本
    不存在，所以本轮还没有终端命令记录 cwd 时，相对路径必须原样交给远端 shell
    去按远端 cwd 解析。
    """

    @pytest.fixture(autouse=True)
    def _no_session_cwd(self, monkeypatch):
        monkeypatch.setattr(terminal_tool, "_session_cwd", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})

    def test_local_backend_uses_resolved_path(self, monkeypatch):
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "local")
        assert ft._ops_path("notes.md", "/base/notes.md", "t") == "/base/notes.md"

    def test_ssh_backend_without_anchor_keeps_relative_path(self, monkeypatch):
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "ssh")
        monkeypatch.setattr(ft, "_authoritative_workspace_root", lambda _t="default": None)
        assert ft._ops_path("notes.md", "/host/cwd/notes.md", "t") == "notes.md"

    def test_ssh_backend_with_recorded_cwd_still_keeps_relative_path(self, monkeypatch):
        # 有 live 记录也不例外：_authoritative_workspace_root 会一路兜底到裸的
        # $TERMINAL_CWD，那是宿主机路径，对远端 / 容器都不成立。
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "ssh")
        monkeypatch.setattr(ft, "_authoritative_workspace_root", lambda _t="default": "/remote/work")
        assert ft._ops_path("notes.md", "/remote/work/notes.md", "t") == "notes.md"

    def test_container_backend_never_gets_host_terminal_cwd(self, monkeypatch):
        # TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE 下宿主项目映射成容器内
        # /workspace，把宿主绝对路径交给容器 shell 会找错文件。
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "docker")
        monkeypatch.setattr(
            ft, "_authoritative_workspace_root", lambda _t="default": "/home/u/project"
        )
        assert ft._ops_path("foo", "/home/u/project/foo", "t") == "foo"
        assert ft._ops_uses_resolved_paths("t") is False

    def test_absent_resolution_always_falls_back_to_raw(self, monkeypatch):
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "local")
        assert ft._ops_path("notes.md", None, "t") == "notes.md"

    def test_v4a_headers_are_not_rewritten_for_remote_backend(self, monkeypatch):
        # patch 走的是同一判定：远端无锚点时 header 必须保持相对。
        monkeypatch.setattr(ft, "_terminal_env_type_for_task", lambda _t="default": "ssh")
        monkeypatch.setattr(ft, "_authoritative_workspace_root", lambda _t="default": None)
        assert ft._ops_uses_resolved_paths("t") is False


def test_relative_path_cannot_escape_into_a_sibling_agent_output(
    _managed_gateway, monkeypatch
):
    """受管终端把每个 agent 的 output 归自己 UID、0700，shell 天然进不去别人的
    产出；但文件工具跑在 root 网关进程里没有这层保护，而相对路径此刻正锚在自己
    的 output 上（HR3）。"""
    output, _daemon_home = _managed_gateway
    agents_root = output.parent.parent
    sibling = agents_root / "agent-b" / "output"
    sibling.mkdir(parents=True)
    (sibling / "secret.md").write_text("theirs", encoding="utf-8")
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))

    denied = ft._managed_sibling_profile_error(
        "../../agent-b/output/secret.md", task_id="mux-session"
    )
    assert denied is not None
    assert "another agent's data directory" in denied

    # 自己 output 里的相对路径照常放行。
    assert ft._managed_sibling_profile_error("notes.md", task_id="mux-session") is None
    # 绝对路径不走这个锚点，交给正常的 scope gate。
    assert (
        ft._managed_sibling_profile_error(
            str(output / "notes.md"), task_id="mux-session"
        )
        is None
    )
