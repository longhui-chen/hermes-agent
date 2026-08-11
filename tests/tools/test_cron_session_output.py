"""ZET-2604: cron 产出必须落进 session 级 agent output 目录。

三层断言：
1. ``prepare_cron_session_output_dir`` —— 从 origin.chat_id 推导
   ``<ZET_AGENT_OUTPUT_DIR>/<session>``（复用 local-server sessionscope 的
   「最后一个冒号后缀」语义），origin 是 untrusted 持久化输入，非法后缀必须
   确定性回落 agent 根目录（仍在沙箱内），绝不 join 脏值。
2. ``push/pop_cron_output_scope`` —— run 期间 ``agent_output`` 别名解析到
   session 目录，run 结束后还原，不污染进程环境。
3. ``_build_cron_execution_contract`` —— 注入精确绝对路径契约。
"""
from __future__ import annotations

import os

import pytest

from tools.runtime_workdir import (
    agent_output_dir,
    cron_session_suffix,
    prepare_cron_session_output_dir,
    pop_cron_output_scope,
    push_cron_output_scope,
)


# ── session 后缀推导（最后一个冒号后缀，对齐 sessionscope.go） ──────────


@pytest.mark.parametrize(
    ("chat_id", "expected"),
    [
        ("zettlab:5p0qrjx02zn0:319005bc-47b6:ImQ9EzmhqcVh", "ImQ9EzmhqcVh"),
        # board154 实测：chat_id 被 HERMES_HOME 前缀污染，后缀语义依然成立
        (
            "/zettos/x/hermes_home/profiles/eae0707d|zettlab:u:eae0707d:Z62lqKRcFaTR",
            "Z62lqKRcFaTR",
        ),
        ("no-colon-at-all", None),
        ("trailing-colon:", None),
        ("zettlab:u:a:../../etc", None),  # traversal
        ("zettlab:u:a:has/slash", None),
        ("zettlab:u:a:" + "x" * 65, None),  # over length cap
        ("", None),
        (None, None),
    ],
)
def test_cron_session_suffix(chat_id, expected):
    assert cron_session_suffix(chat_id) == expected


# ── prepare：推导 + mkdir + 回落 ────────────────────────────────────────


def test_prepare_creates_session_dir_under_agent_output(monkeypatch, tmp_path):
    base = tmp_path / "output"
    base.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(base))

    got = prepare_cron_session_output_dir("zettlab:u:agent1:SessAbc")
    assert got == str(base / "SessAbc")
    assert os.path.isdir(got)


def test_prepare_falls_back_to_agent_root_on_bad_suffix(monkeypatch, tmp_path):
    base = tmp_path / "output"
    base.mkdir()
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(base))

    assert prepare_cron_session_output_dir("zettlab:u:a:../../etc") == str(base)
    assert prepare_cron_session_output_dir(None) == str(base)
    # 回落时绝不额外建目录
    assert sorted(os.listdir(base)) == []


def test_prepare_returns_none_without_platform_dir(monkeypatch):
    monkeypatch.delenv("ZET_AGENT_OUTPUT_DIR", raising=False)
    assert prepare_cron_session_output_dir("zettlab:u:a:Sess") is None


# ── scope 覆盖：run 内生效、run 外还原 ──────────────────────────────────


def test_push_pop_cron_output_scope(monkeypatch, tmp_path):
    base = tmp_path / "output"
    sess = base / "SessScope"
    sess.mkdir(parents=True)
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(base))

    token = push_cron_output_scope(str(sess))
    try:
        assert agent_output_dir() == str(sess)
    finally:
        pop_cron_output_scope(token)
    assert agent_output_dir() == str(base)
    # 进程环境从未被改写
    assert os.environ["ZET_AGENT_OUTPUT_DIR"] == str(base)


# ── 执行契约注入 ────────────────────────────────────────────────────────


def test_cron_execution_contract_includes_session_output_rule():
    from cron.scheduler import _build_cron_execution_contract

    job = {
        "id": "j1",
        "prompt": "巡检",
        "_zet_session_output_dir": "/volume1/subvol/agents/data/a1/output/SessX",
    }
    contract = _build_cron_execution_contract(job)
    assert "/volume1/subvol/agents/data/a1/output/SessX" in contract
    assert "cron/output" in contract  # 明确禁止写 run-record 目录


def test_cron_execution_contract_unchanged_without_platform_dir():
    from cron.scheduler import _build_cron_execution_contract

    contract = _build_cron_execution_contract({"id": "j1", "prompt": "x"})
    # 无平台目录（上游部署）：契约保持原状，不出现路径规则
    assert "FILE OUTPUT" not in contract
