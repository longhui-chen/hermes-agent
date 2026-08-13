"""受管终端 HOME 必须能看见当前 profile 的飞书凭据。"""

from __future__ import annotations

import os
from pathlib import Path

from tools.environments.local import (
    _link_profile_lark_cli_credentials,
    _managed_terminal_profile_tag,
)


def _profile(tmp_path, agent):
    profile_home = tmp_path / "profiles" / agent
    home = profile_home / "home"
    (home / ".lark-cli" / "hermes").mkdir(parents=True)
    (home / ".local" / "share" / "lark-cli").mkdir(parents=True)
    (home / ".lark-cli" / "hermes" / "config.json").write_text("{}")
    return profile_home, home


def test_sandbox_home_name_is_scoped_to_the_session_profile(tmp_path):
    profile_home, _home = _profile(tmp_path, "agent-one")

    tag = _managed_terminal_profile_tag({"HERMES_HOME": str(profile_home)})

    assert tag == "-agent-one"


def test_session_profile_credentials_are_linked_into_the_sandbox(
    tmp_path, monkeypatch
):
    profile_home, source_home = _profile(tmp_path, "agent-one")
    sandbox = tmp_path / "terminal-homes" / "1000-agent-one"
    sandbox.mkdir(parents=True)
    monkeypatch.setattr(
        "tools.environments.local._validate_managed_root_directory_chain",
        lambda path: path.resolve(),
    )

    _link_profile_lark_cli_credentials(
        sandbox,
        {"HERMES_HOME": str(profile_home)},
        "-agent-one",
    )

    # 键用 posix 形式:os.path.join 在 Windows 上给的是反斜杠,跟字面量比永远不等,
    # 这条测试在非 POSIX 开发机上一直是红的(与被测逻辑无关)。
    linked = {
        Path(relative).as_posix(): (
            (sandbox / relative).is_symlink()
            and (sandbox / relative).resolve() == (source_home / relative).resolve()
        )
        for relative in (
            ".lark-cli",
            os.path.join(".local", "share", "lark-cli"),
        )
    }
    assert linked == {
        ".lark-cli": True,
        ".local/share/lark-cli": True,
    }


def test_unscoped_sandbox_does_not_receive_credentials(tmp_path):
    profile_home, _source_home = _profile(tmp_path, "agent-one")
    sandbox = tmp_path / "terminal-homes" / "1000"
    sandbox.mkdir(parents=True)

    _link_profile_lark_cli_credentials(
        sandbox,
        {"HERMES_HOME": str(profile_home)},
        "",
    )

    assert not (sandbox / ".lark-cli").exists()


def test_occupied_credential_dir_is_moved_aside_instead_of_bricking(
    tmp_path, monkeypatch
):
    """受管 HOME 里已有真目录时,挪开重建软链,不许抛错。

    2026-08-13 板 .212:软链建成之前跑过的命令(lark-cli 自己首当其冲)会按 $HOME
    直接创建 ~/.lark-cli;之后每一轮都撞 rmdir 失败,原来直接抛 OSError ——
    结果这台设备上所有 agent 的任何 lark-cli 脚本全挂,用户只看到
    「配置暂未推进」。同一坑 08-06 撞过一次、手工绕过没根治。
    """

    profile_home, source_home = _profile(tmp_path, "agent-one")
    sandbox = tmp_path / "terminal-homes" / "1000-agent-one"
    sandbox.mkdir(parents=True)
    occupied = sandbox / ".lark-cli"
    (occupied / "cache").mkdir(parents=True)          # 非空:rmdir 必失败
    (occupied / "update-state.json").write_text("{}")
    monkeypatch.setattr(
        "tools.environments.local._validate_managed_root_directory_chain",
        lambda path: path.resolve(),
    )

    _link_profile_lark_cli_credentials(
        sandbox,
        {"HERMES_HOME": str(profile_home)},
        "-agent-one",
    )

    assert occupied.is_symlink()
    assert occupied.resolve() == (source_home / ".lark-cli").resolve()
    # 挪开而不是删掉:凭据类目录不许静默销毁
    retired = [p for p in sandbox.iterdir() if p.name.startswith(".lark-cli.replaced-")]
    assert len(retired) == 1, sorted(p.name for p in sandbox.iterdir())
    assert (retired[0] / "update-state.json").read_text() == "{}"


def test_occupied_plain_file_is_also_moved_aside(tmp_path, monkeypatch):
    """挡路的可能是个文件而不是目录——rmdir 对它同样无效。"""

    profile_home, source_home = _profile(tmp_path, "agent-one")
    sandbox = tmp_path / "terminal-homes" / "1000-agent-one"
    sandbox.mkdir(parents=True)
    (sandbox / ".lark-cli").write_text("stray")
    monkeypatch.setattr(
        "tools.environments.local._validate_managed_root_directory_chain",
        lambda path: path.resolve(),
    )

    _link_profile_lark_cli_credentials(
        sandbox,
        {"HERMES_HOME": str(profile_home)},
        "-agent-one",
    )

    assert (sandbox / ".lark-cli").is_symlink()
    assert (sandbox / ".lark-cli").resolve() == (source_home / ".lark-cli").resolve()


def test_untrusted_source_home_is_still_rejected(tmp_path, monkeypatch):
    """挪开挡路实体是为了别把设备卡死,不是放松信任校验:源不可信照样拒绝。"""

    import pytest

    profile_home, _source_home = _profile(tmp_path, "agent-one")
    sandbox = tmp_path / "terminal-homes" / "1000-agent-one"
    sandbox.mkdir(parents=True)
    monkeypatch.setattr(
        "tools.environments.local._validate_managed_root_directory_chain",
        lambda path: (tmp_path / "elsewhere").resolve(),
    )

    with pytest.raises(OSError):
        _link_profile_lark_cli_credentials(
            sandbox,
            {"HERMES_HOME": str(profile_home)},
            "-agent-one",
        )
