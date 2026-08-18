"""ZET-2473：kanban 任务完成后向对话框倾泻大量内部 JSON，把用户刷屏。

现场实证（原始 bug 单里用户手打的九个文件名）：
`.card_data.json` · `chat_index.json` · `chat_state.json` · `minutes_index.json`
`minutes_state.json` · `task_index.json` · `task_state.json` · `sync_state.json`
`index.json` —— **九个全是内部状态文件，零个用户交付物**。

根因：`_deliver_kanban_artifacts` 从三路聚合待投递文件，其中
② `event_payload['summary']` 和 ③ `task.result` 是**从自由文本里扫出所有本地
路径**，代码无从区分「用户交付物」和「agent 顺手读过的内部状态文件」。

⛔ 为什么不能用扩展名 / 文件名黑名单收紧：那是**开集**。
  - 不能黑 `.json` —— 真交付物也可能是 JSON
  - 黑 `*_state.json` / `*_index.json` 则换个命名（`meta.json`、`cache.json`）就漏
闭集只有一个：**producer 显式声明**（`kanban_complete(artifacts=[...])`，
`prompt_builder` 已把它定为 top-level 契约）。

⚠️ 这两路此前**零测试覆盖**（本文件之前，整个 kanban watchers 只有一条
「方法是否存在」的检查，连行为都没测）——这就是它能刷屏这么久的原因。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.kanban_watchers import ArtifactFailure, GatewayKanbanWatchersMixin


def _adapter(extracted: list[str] | None = None):
    """adapter 桩：`extract_local_files` 模拟从自由文本里扫出路径。"""
    return SimpleNamespace(
        extract_local_files=lambda text: (list(extracted or []), None),
        # 🔴 桩必须**长得像现场**:少一个方法,被测代码会抛 AttributeError,
        # 而 AttributeError 也会让路径进 failed ⇒ 断言照样成立、门却空转。
        # 这正是本文件两条门在改成逐张投递后**为错误的理由而绿**的原因。
        send_image_file=AsyncMock(),
        send_multiple_images=AsyncMock(),
        send_video=AsyncMock(),
        send_document=AsyncMock(),
    )


def _delivered(adapter) -> list[str]:
    out = [c.kwargs["file_path"] for c in adapter.send_document.await_args_list]
    out += [c.kwargs["video_path"] for c in adapter.send_video.await_args_list]
    out += [c.kwargs["image_path"] for c in adapter.send_image_file.await_args_list]
    for call in adapter.send_multiple_images.await_args_list:
        out += [u for u, _ in call.kwargs["images"]]
    return out


async def _deliver(adapter, *, payload, task):
    mixin = GatewayKanbanWatchersMixin()
    await mixin._deliver_kanban_artifacts(
        adapter=adapter, chat_id="oc_1", metadata=None,
        event_payload=payload, task=task,
    )


@pytest.mark.asyncio
async def test_explicitly_declared_artifacts_are_delivered(tmp_path):
    """① 显式 artifacts —— ⛔ 这一路是正常功能，一个字节都不许变。"""
    deliverable = tmp_path / "report.json"
    deliverable.write_text("{}", encoding="utf-8")
    adapter = _adapter()

    await _deliver(
        adapter,
        payload={"artifacts": [str(deliverable)]},
        task=SimpleNamespace(id="t1", result=None),
    )

    assert _delivered(adapter) == [str(deliverable)]


@pytest.mark.asyncio
async def test_paths_only_mentioned_in_summary_are_not_delivered(tmp_path):
    """② summary 里扫出来的路径 ⇒ ⛔ 不投递（ZET-2473 的刷屏源之一）。"""
    internal = tmp_path / "chat_state.json"
    internal.write_text("{}", encoding="utf-8")
    adapter = _adapter([str(internal)])

    await _deliver(
        adapter,
        payload={"summary": f"读了 {internal} 之后完成了任务"},
        task=SimpleNamespace(id="t2", result=None),
    )

    assert _delivered(adapter) == []


@pytest.mark.asyncio
async def test_paths_only_mentioned_in_task_result_are_not_delivered(tmp_path):
    """③ legacy task.result 里扫出来的路径 ⇒ ⛔ 不投递。"""
    internal = tmp_path / "sync_state.json"
    internal.write_text("{}", encoding="utf-8")
    adapter = _adapter([str(internal)])

    await _deliver(
        adapter,
        payload=None,
        task=SimpleNamespace(id="t3", result=f"done, see {internal}"),
    )

    assert _delivered(adapter) == []


@pytest.mark.asyncio
async def test_the_real_zet_2473_payload_delivers_nothing(tmp_path):
    """现场那九个文件名：⛔ 一个都不许发出去。

    ⭐ 这条同时证明了**为什么黑名单不行** —— 它们没有共同的命名规律可黑，
    唯一的共同点是「producer 没有声明它们是交付物」。
    """
    names = [
        ".card_data.json", "chat_index.json", "chat_state.json",
        "minutes_index.json", "minutes_state.json", "task_index.json",
        "task_state.json", "sync_state.json", "index.json",
    ]
    paths = []
    for name in names:
        f = tmp_path / name
        f.write_text("{}", encoding="utf-8")
        paths.append(str(f))
    adapter = _adapter(paths)

    await _deliver(
        adapter,
        payload={"summary": "同步完成"},
        task=SimpleNamespace(id="t4", result="all synced"),
    )

    assert _delivered(adapter) == [], "ZET-2473 现场的九个内部状态文件仍被投递"


@pytest.mark.asyncio
async def test_declared_and_undeclared_mixed_delivers_only_the_declared(tmp_path):
    """混合场景：只投显式声明的那个，扫出来的不投。"""
    declared = tmp_path / "deliverable.pdf"
    declared.write_text("x", encoding="utf-8")
    internal = tmp_path / "task_state.json"
    internal.write_text("{}", encoding="utf-8")
    adapter = _adapter([str(internal)])

    await _deliver(
        adapter,
        payload={"artifacts": [str(declared)], "summary": f"see {internal}"},
        task=SimpleNamespace(id="t5", result=None),
    )

    assert _delivered(adapter) == [str(declared)]


@pytest.mark.asyncio
async def test_undelivered_paths_are_observable_without_leaking_filenames(
    tmp_path, caplog
):
    """⭐ 排查可观测：未投递的路径必须留痕，但 info 级 ⛔ 不许打文件名。

    这条投递路径原先只在**失败**时记日志，成功零留痕 —— ZET-2473 查不出来自
    哪一路，根源就是它；这次是靠用户在 bug 单里手打了九个文件名才定的案，
    下次没这运气。
    """
    internal = tmp_path / "chat_state.json"
    internal.write_text("{}", encoding="utf-8")
    adapter = _adapter([str(internal)])

    with caplog.at_level(logging.INFO):
        await _deliver(
            adapter,
            payload={"summary": f"see {internal}"},
            task=SimpleNamespace(id="t6", result=None),
        )

    info = "\n".join(
        r.getMessage() for r in caplog.records if r.levelno == logging.INFO
    )
    assert "unclaimed" in info, f"未投递的路径没有留痕：{info!r}"
    assert "t6" in info, "日志没带 task id，无法关联"
    assert "chat_state.json" not in info, (
        f"文件名泄漏进 info 级日志：{info!r} —— 它可能带用户内容，"
        "而且对「判来自哪一路」毫无帮助"
    )


# ════════════ RH 复审 P1-2：上传失败被永久判成通知成功 ════════════
#
# 现场：三个上传点（``send_multiple_images`` / ``send_video`` /
# ``send_document``）**把返回值整个丢掉**，只 catch 异常 ⇒
# ``SendResult(success=False)``（平台拒收 / 超限 / 鉴权过期）被当成投递成功；
# 外层随后重置失败计数、推进订阅游标 ⇒ 用户收到「任务完成」，**文件永久缺失
# 且不会重试**。而调用处的 except 只写 ``logger.debug`` —— 线上一个字都看不到。
#
# ⭐ 同一函数 :540 的文本通知**早就做对了**（``success is False`` ⇒ raise）。
#    这是典型的兄弟调用点没跟上。
# ⛔ 但**不能照抄它 raise** —— 完成文本已经发出去，整体重试会让用户再收一遍。

def _fail_result(err="platform rejected"):
    return SimpleNamespace(success=False, error=err)


def _ok_result():
    return SimpleNamespace(success=True, error=None)


async def _deliver_returning(adapter, *, payload, task):
    mixin = GatewayKanbanWatchersMixin()
    return await mixin._deliver_kanban_artifacts(
        adapter=adapter, chat_id="oc_1", metadata=None,
        event_payload=payload, task=task,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fname,sender",
    [("shot.png", "send_image_file"), ("clip.mp4", "send_video"),
     ("doc.pdf", "send_document")],
)
async def test_rejected_upload_is_reported_not_swallowed(tmp_path, fname, sender):
    """三个上传点全集：``success=False`` 必须被检出，⛔ 不许当成送达。"""
    f = tmp_path / fname
    f.write_bytes(b"x")
    adapter = _adapter()
    getattr(adapter, sender).return_value = _fail_result()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )

    assert [x.path for x in failed] == [str(f)], (
        f"{sender} 返回 success=False 却被当成投递成功:{failed}")


@pytest.mark.asyncio
async def test_exception_during_upload_is_reported(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    adapter = _adapter()
    adapter.send_document.side_effect = RuntimeError("connection reset")

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert [x.path for x in failed] == [str(f)]


@pytest.mark.asyncio
async def test_adapters_returning_none_keep_the_legacy_contract(tmp_path):
    """⛔ 不许弄坏原来对的：返回 ``None`` 的适配器沿用「无异常即送达」。

    判据与 :540 文本通知那条逐字一致（``getattr(res,"success",True)``）。
    ⭐ 若这条红了，说明新判据把一大批正常适配器判成了失败 —— 那比原 bug 更坏。
    """
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    adapter = _adapter()
    adapter.send_document.return_value = None

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [], f"返回 None 的适配器被误判成失败:{failed}"


@pytest.mark.asyncio
async def test_all_success_reports_nothing(tmp_path):
    """全成功 ⇒ 空清单 ⇒ 上层⛔不发提示。正常路径不许被打扰。"""
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"x")
    adapter = _adapter()
    adapter.send_document.return_value = _ok_result()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == []


@pytest.mark.asyncio
async def test_images_are_judged_one_by_one(tmp_path):
    """🔴 图片改成**逐张**投递 ⇒ 一张失败一张成功时，只报失败那张。

    上一版用 ``send_multiple_images`` 批量发，一次调用只有一个返回值,
    于是「一张超限」只能整批算失败。⭐ 逐张之后精度提高了 ——
    而这不是顺带的好处,是**换成逐张的原因**:批量那条路的返回值
    在基类和 7 个 adapter 上都是 ``None``,失败**根本检不出来**。
    """
    imgs = []
    for n in ("a.png", "b.jpg"):
        f = tmp_path / n
        f.write_bytes(b"x")
        imgs.append(str(f))

    adapter = _adapter()
    adapter.send_image_file.side_effect = [
        _fail_result("too large"), _ok_result()]

    failed = await _deliver_returning(
        adapter, payload={"artifacts": imgs},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert [x.path for x in failed] == [imgs[0]], f"逐张判定不准:{failed}"


@pytest.mark.asyncio
async def test_batch_image_api_is_never_used_here(tmp_path):
    """🔴 钉死这个决定:交付物投递⛔不许走 ``send_multiple_images``。

    它在基类与 7 个 adapter 上声明 ``-> None``、feishu 返回 ``bool`` ⇒
    ``getattr(res, "success", True) is not False`` 对**全部 8 个恒为 True**。
    ⭐ 换回批量 = 图片失败重新变成不可观测,而**注释会依然写着**
    「判据是 SendResult.success」—— 假闭集比没有门更坏。
    ⛔ 这条不是风格检查,它拦的是一条真实的静默丢失路径。
    """
    f = tmp_path / "shot.png"
    f.write_bytes(b"x")
    adapter = _adapter()
    adapter.send_image_file.return_value = _ok_result()

    await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert adapter.send_multiple_images.await_count == 0, (
        "又走回批量图片 API —— 那条路上平台拒收检不出来")
    assert adapter.send_image_file.await_count == 1


# ───────────── 用户侧：拿不到文件必须被告知 ─────────────

@pytest.mark.asyncio
async def test_user_is_told_once_when_artifacts_are_missing():
    """🔴 用户刚收到「任务完成」却没文件 ⇒ 他会以为文件没生成、重跑整个任务。

    ⭐ 判据「不写它用户会不会做错事」= 会 ⇒ 这条属于「错误必须可行动」，
    ⛔ 不在「别堆文案」该砍的范围里。但**只发一条汇总**，⛔ 不是每个文件一条。
    """
    mixin = GatewayKanbanWatchersMixin()
    adapter = SimpleNamespace(send=AsyncMock(return_value=_ok_result()))

    await mixin._notify_artifact_delivery_failure(
        adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
        failed=[ArtifactFailure("/x/a.png", "upload_failed"), ArtifactFailure("/x/b.pdf", "upload_failed")],
    )

    assert adapter.send.await_count == 1, (
        f"每个文件发了一条 —— 刷屏:{adapter.send.await_count}")
    body = adapter.send.await_args.kwargs["content"]
    assert "a.png" in body and "b.pdf" in body, f"没说清少了哪些文件:{body}"
    assert "2" in body, f"没给出数量:{body}"


@pytest.mark.asyncio
async def test_notification_failure_never_escalates():
    """⛔ 提示本身失败也不许上抛 —— 上抛会让整个事件重试，
    用户就会**再收到一遍「任务完成」**（比缺附件更糟）。
    """
    mixin = GatewayKanbanWatchersMixin()
    adapter = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("down")))

    # ⛔ 这里不许抛
    await mixin._notify_artifact_delivery_failure(
        adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
        failed=[ArtifactFailure("/x/a.png", "upload_failed")],
    )


@pytest.mark.asyncio
async def test_notification_rejection_is_logged_not_swallowed(caplog):
    """🔴 平台**拒收**（不抛异常、返回 success=False）时必须留痕。

    ⭐ 这个函数正是为了修「三个上传点丢掉返回值」而写的，
    而它自己一开始也只 catch 异常、不看返回值 —— 同一个错，同一个文件，
    相隔十几行。**修一类缺陷时，新写的代码要先过一遍同一条判据。**
    """
    import logging

    mixin = GatewayKanbanWatchersMixin()
    adapter = SimpleNamespace(send=AsyncMock(return_value=_fail_result("rate limited")))

    with caplog.at_level(logging.ERROR):
        await mixin._notify_artifact_delivery_failure(
            adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
            failed=[ArtifactFailure("/x/a.png", "upload_failed")],
        )

    assert "被平台拒收" in caplog.text, (
        f"提示被拒收却静默返回 —— 用户既没拿到附件也没被告知:{caplog.text!r}")


# ───────── RH 第五轮 P1：声明了却送不到的交付物⛔不许静默 ─────────


@pytest.mark.asyncio
async def test_declared_but_missing_artifact_is_reported(tmp_path):
    """🔴 producer 显式声明、文件却不存在 ⇒ 必须进失败清单。

    原先 `_add()` 在 `isfile=False` 时直接 return ⇒ `failed` 为空 ⇒
    用户只收到「任务完成」，既没有文件也没有提示，会以为文件没生成
    然后把整个任务重跑一遍。⭐ 又一次「空列表二义」。
    """
    gone = str(tmp_path / "report.pdf")          # ⛔ 故意不创建
    adapter = _adapter()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [gone]},
        task=SimpleNamespace(id="t1", result=None),
    )

    assert failed == [ArtifactFailure(gone, "missing")], f"{failed}"
    assert adapter.send_document.await_count == 0, "不存在的文件不该触发上传"


@pytest.mark.asyncio
async def test_declared_but_filtered_artifact_is_reported(tmp_path, monkeypatch):
    """🔴 被投递安全过滤拒绝的**已声明**交付物同样不许静默。

    ⭐ 孪生：`_add` 的 isfile 检查和 `filter_local_delivery_paths` 是
    **两个**丢弃点，只修一个等于给另一个发免检。
    """
    f = tmp_path / "secret.pdf"
    f.write_bytes(b"x")
    adapter = _adapter()

    from gateway.platforms import base as _base
    monkeypatch.setattr(
        _base.BasePlatformAdapter, "filter_local_delivery_paths",
        staticmethod(lambda paths: []))

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(f)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [ArtifactFailure(str(f), "policy_blocked")], f"{failed}"


@pytest.mark.asyncio
async def test_missing_and_deliverable_are_both_handled(tmp_path):
    """混合档：一个能送、一个不存在 ⇒ 各归各的，⛔ 不许一坏全丢。"""
    ok = tmp_path / "ok.pdf"
    ok.write_bytes(b"x")
    gone = str(tmp_path / "gone.pdf")
    adapter = _adapter()
    adapter.send_document.return_value = _ok_result()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(ok), gone]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [ArtifactFailure(gone, "missing")]
    assert adapter.send_document.await_count == 1


@pytest.mark.asyncio
async def test_unclaimed_missing_path_is_still_silent(tmp_path):
    """⛔ 作用域上界：自由文本里扫出的**未声明**路径不存在 ⇒ 仍然静默。

    ⭐ 那些本来就允许是「顺手提到的引用」（ZET-2473）。改宽了就会对每条
    提到路径的 summary 都回一句「附件未送达」。
    """
    adapter = _adapter(extracted=[str(tmp_path / "nope.txt")])

    failed = await _deliver_returning(
        adapter, payload={"summary": "见 /tmp/nope.txt"},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [], f"未声明路径被误报成失败:{failed}"


# ═════════ RH 第六轮：canonical identity / 成因分开 / 去重 ═════════


@pytest.mark.asyncio
async def test_symlink_alias_is_not_reported_as_failed(tmp_path):
    """🔴 安全过滤返回 **canonical** 路径，原先拿它和原始路径做字符串差集
    ⇒ 合法符号链接的 target 被成功上传，alias 又进失败清单，
    用户收到自相矛盾的「附件未送达」。

    ⭐ 判据必须是 canonical identity，⛔ 不是字符串相等。
    """
    real = tmp_path / "real.pdf"
    real.write_bytes(b"x")
    alias = tmp_path / "alias.pdf"
    alias.symlink_to(real)

    adapter = _adapter()
    adapter.send_document.return_value = _ok_result()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [str(alias)]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [], f"合法符号链接被同时判成送达和失败:{failed}"
    assert adapter.send_document.await_count == 1


@pytest.mark.asyncio
async def test_same_missing_artifact_declared_twice_counts_once(tmp_path):
    """🔴 去重按**声明 identity** —— 原先 seen 只在文件存在后写入，
    同一个缺失路径声明两次会被计成两个失败，给用户的数量直接说错。"""
    gone = str(tmp_path / "gone.pdf")
    adapter = _adapter()

    failed = await _deliver_returning(
        adapter, payload={"artifacts": [gone, gone]},
        task=SimpleNamespace(id="t1", result=None),
    )
    assert failed == [ArtifactFailure(gone, "missing")], f"重复计数:{failed}"


@pytest.mark.asyncio
async def test_missing_is_not_told_the_file_is_still_on_the_device():
    """🔴 三种成因给三种话。``missing`` 说「仍在设备上」是**假的**。"""
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=_ok_result())
    mixin = GatewayKanbanWatchersMixin()
    await mixin._notify_artifact_delivery_failure(
        adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
        failed=[ArtifactFailure("/x/gone.pdf", "missing")])
    body = adapter.send.await_args.kwargs["content"]
    assert "找不到" in body, f"没说清文件没了:{body}"
    assert "仍在设备上" not in body, f"对缺失文件说了假话:{body}"


@pytest.mark.asyncio
async def test_policy_blocked_never_echoes_the_basename():
    """🔴 安全策略拒绝时⛔ 一个 basename 都不许回显。

    ⭐ 回显等于向聊天对方**确认这个路径存在** —— 那正是策略要挡的东西。
    """
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=_ok_result())
    mixin = GatewayKanbanWatchersMixin()
    await mixin._notify_artifact_delivery_failure(
        adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
        failed=[ArtifactFailure("/etc/shadow-ish/secret-payroll.pdf", "policy_blocked")])
    body = adapter.send.await_args.kwargs["content"]
    assert "secret-payroll" not in body, f"回显了被策略拒绝的文件名:{body}"
    assert "安全策略" in body, f"没说清为什么:{body}"


@pytest.mark.asyncio
async def test_upload_failed_still_lists_names_and_says_where_they_are():
    """⛔ 必须保持不变：真的上传失败时，文件确实还在设备上，要列名字。"""
    adapter = _adapter()
    adapter.send = AsyncMock(return_value=_ok_result())
    mixin = GatewayKanbanWatchersMixin()
    await mixin._notify_artifact_delivery_failure(
        adapter=adapter, chat_id="oc_1", metadata=None, task_id="t1",
        failed=[ArtifactFailure("/x/report.pdf", "upload_failed")])
    body = adapter.send.await_args.kwargs["content"]
    assert "report.pdf" in body and "仍在设备上" in body, body
