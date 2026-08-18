"""从完成散文里【猜】出来的产物路径，必须带来源标记（ZET-2473 配套）。

背景：``_merge_completion_prose_artifacts`` 会把 summary/result 里出现的 scratch
文件路径并进 ``metadata["artifacts"]``。这个用途本身正当 —— scratch 清理前
``_persist_scratch_completion_artifacts`` 要靠它把用户被承诺的文件复制出来，
⛔ 不能取消。问题是它和 ``kanban_complete(artifacts=[...])`` 显式声明的混在
同一个键里，到投递侧已分不清来源，于是猜出来的也被当交付物推给用户 ⇒ 刷屏。

⇒ 契约：并进 artifacts 的同时，必须在 ``_prose_discovered_artifacts`` 里留下
来源标记；投递侧据此减掉。

本文件两侧都钉：
  · 上半：生产标记（``_merge_completion_prose_artifacts`` 打标 + persist 改写
    路径后标记跟着搬）
  · 下半：消费侧（``_deliverable_artifacts`` 决定哪些真的推送给用户）
⭐ 只钉上半是不够的 —— 「被标记」和「没被投递」之间隔着整个消费侧，把减法
删掉，上半三条依然全绿而刷屏原样复发。用户能看见的是后者。

⚠️ 仍未覆盖（如实标注）：``complete_task`` 把 ``_deliverable_artifacts`` 的
结果写进 event payload、以及 notifier 消费该 payload 这两跳，本文件没有驱动。
它们分别由「调用点唯一」（`_deliverable_artifacts` 全仓 1 处调用）和
``tests/gateway/test_kanban_artifact_delivery.py`` 覆盖，⛔ 但两者之间没有
端到端用例把它们串起来。
"""
from __future__ import annotations

import sqlite3

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture()
def conn_with_scratch_task(tmp_path):
    """最小 tasks 表 —— _merge_completion_prose_artifacts 只读这三列。"""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, workspace_kind TEXT, workspace_path TEXT)"
    )
    workspace = tmp_path / "scratch-ws"
    workspace.mkdir()
    conn.execute(
        "INSERT INTO tasks (id, workspace_kind, workspace_path) VALUES (?, 'scratch', ?)",
        ("t1", str(workspace)),
    )
    return conn, workspace


def _force_managed(monkeypatch):
    # _is_managed_scratch_path 认的是真实 scratch 根；本用例的 tmp_path 不在
    # 那底下，放行它，让判据聚焦在"来源标记有没有留下"这一件事上。
    monkeypatch.setattr(kb, "_is_managed_scratch_path", lambda _p: True)


def test_prose_discovered_paths_are_marked_as_such(conn_with_scratch_task, monkeypatch):
    """猜出来的路径进了 artifacts，同时必须留下来源标记。"""
    _force_managed(monkeypatch)
    conn, workspace = conn_with_scratch_task
    guessed = workspace / "report.md"
    guessed.write_text("x", encoding="utf-8")
    declared = workspace / "declared.bin"
    declared.write_text("y", encoding="utf-8")

    out = kb._merge_completion_prose_artifacts(
        conn,
        "t1",
        {"artifacts": [str(declared)]},
        summary=f"产物见 {guessed}",
        result=None,
    )

    assert str(guessed) in (out or {}).get("artifacts", []), (
        "calibration: 猜的路径没被并进 artifacts,后面的标记断言会恒真")
    marks = (out or {}).get("_prose_discovered_artifacts")
    assert marks == [str(guessed)], (
        f"猜出来的路径必须且只能被标成 prose-discovered,实际={marks!r}")
    assert str(declared) not in (marks or []), (
        "⛔ 显式声明的交付物被误标成 prose-discovered —— 它会被投递侧减掉")


def test_declared_only_completion_leaves_no_marker(conn_with_scratch_task, monkeypatch):
    """没有可猜的路径时 ⛔ 不许留下标记 —— 陈旧标记会误伤显式交付物。"""
    _force_managed(monkeypatch)
    conn, workspace = conn_with_scratch_task
    declared = workspace / "only.bin"
    declared.write_text("y", encoding="utf-8")

    out = kb._merge_completion_prose_artifacts(
        conn, "t1", {"artifacts": [str(declared)]}, summary="干完了", result=None
    )

    assert (out or {}).get("artifacts") == [str(declared)]
    assert "_prose_discovered_artifacts" not in (out or {})


def test_marker_survives_persist_path_rewrite(conn_with_scratch_task, monkeypatch, tmp_path):
    """复制会把路径整体换成 attachment_dir 下的新路径,标记必须【跟着搬】。

    ⭐ 这是最容易漏的一环:_persist_scratch_completion_artifacts 会把
    metadata["artifacts"] 整体替换成复制后的路径。若标记还停在旧路径上,
    投递侧按它去减就一个都减不掉 —— 刷屏原样复现,而且两个函数各自看起来都对。
    """
    _force_managed(monkeypatch)
    conn, workspace = conn_with_scratch_task
    guessed = workspace / "guessed.txt"
    guessed.write_text("x", encoding="utf-8")

    metadata = kb._merge_completion_prose_artifacts(
        conn, "t1", {"artifacts": []}, summary=f"see {guessed}", result=None
    )
    assert metadata["_prose_discovered_artifacts"] == [str(guessed)]

    kb._persist_scratch_completion_artifacts(conn, "t1", metadata)

    marks = metadata.get("_prose_discovered_artifacts") or []
    arts = metadata.get("artifacts") or []
    assert arts, "calibration: persist 之后 artifacts 为空,隔离断言会恒真"
    assert marks, "复制之后来源标记丢了 —— 投递侧将无法减掉猜出来的路径"
    assert set(marks) <= set(arts), (
        f"来源标记指向的路径不在 artifacts 里(没跟着搬):marks={marks} artifacts={arts}")


# ───────────────────────── 投递侧（消费者）─────────────────────────
# ⭐ 上面三条只钉「标记被生产出来」。用户能看见的是【有没有被推送】，
# 两者之间隔着整个消费侧 —— 把减法删掉，上面三条依然全绿而刷屏复发。
# ⇒ 下面这组直接钉消费侧：哪些产物会进投递集合。

def test_prose_discovered_paths_are_not_delivered():
    """🔴 用户报的那件事：猜出来的路径 ⛔ 不许出现在投递集合里。"""
    declared = "/ws/declared.bin"
    guessed = "/ws/guessed.md"
    out = kb._deliverable_artifacts(
        {
            "artifacts": [declared, guessed],
            "_prose_discovered_artifacts": [guessed],
        }
    )
    assert declared in out, "calibration: 显式声明的没进投递集合,下面的断言会恒真"
    assert guessed not in out, (
        f"猜出来的路径出现在投递集合里 —— 这正是用户看到的刷屏:{out}")


def test_declared_artifacts_delivery_is_unchanged():
    """⛔ 正常功能:没有标记时,显式声明的一个不许少、顺序不许变。"""
    paths = ["/ws/a.bin", "/ws/b.png", "/ws/c.pdf"]
    assert kb._deliverable_artifacts({"artifacts": list(paths)}) == paths


def test_all_prose_means_nothing_to_deliver():
    """全是猜出来的 ⇒ 投递集合为空(⛔ 不是"退回全发")。"""
    guessed = ["/ws/x.md", "/ws/y.md"]
    assert kb._deliverable_artifacts(
        {"artifacts": list(guessed), "_prose_discovered_artifacts": list(guessed)}
    ) == []


def test_whitespace_cannot_smuggle_a_prose_path_past_the_filter():
    """🔴 RH 复审 P3：比较两侧必须同形，否则空白就能绕过减法。

    上一版比较用**未 trim** 的原值、输出却是 trim 后的路径 ⇒
    metadata 里写 ``" /tmp/guess.txt "``、标记里是 ``"/tmp/guess.txt"``
    时减法落空，**猜出来的路径照样被推给用户** —— ZET-2473 原样复发。
    ⭐ 判据两侧不同形 = 判据无效（与「参照系不一致的门恒绿」同形）。
    """
    from hermes_cli.kanban_db import _deliverable_artifacts

    out = _deliverable_artifacts({
        "artifacts": ["  /tmp/guess.txt  ", "/tmp/real.pdf"],
        "_prose_discovered_artifacts": ["/tmp/guess.txt"],
    })
    assert out == ["/tmp/real.pdf"], (
        f"带空白的散文路径绕过了「不投递」过滤:{out}")


def test_whitespace_on_the_marker_side_also_matches():
    """孪生：空白出现在**标记侧**同样要能对上。

    ⛔ 只 trim 一边等于换个方向重演同一个 bug。
    """
    from hermes_cli.kanban_db import _deliverable_artifacts

    out = _deliverable_artifacts({
        "artifacts": ["/tmp/guess.txt", "/tmp/real.pdf"],
        "_prose_discovered_artifacts": ["  /tmp/guess.txt  "],
    })
    assert out == ["/tmp/real.pdf"], f"标记侧带空白就对不上了:{out}"
