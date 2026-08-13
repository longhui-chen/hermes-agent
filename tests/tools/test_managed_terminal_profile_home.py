"""受管终端 HOME 必须能看见当前 profile 的飞书凭据。"""

from __future__ import annotations

import os

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

    linked = {
        str(relative): (
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
