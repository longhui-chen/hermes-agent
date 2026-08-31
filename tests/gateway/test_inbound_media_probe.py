"""入站媒体观测点门 —— 让一次真机实测**能产出判定**。

═══════════════════════════════════════════════════════════════════════
它为什么存在
═══════════════════════════════════════════════════════════════════════
上一次排查卡在「21:26 窗口里 6 条命中**全部来自 traceback 本身**」——
我搜的那个字符串同时出现在源码行里,异常一打印就"命中",量具和被测对象
混在一起,**证不了任何事**。

⇒ 本门钉住三件事,让那种事不可能再发生:
  ① ``[IM-MEDIA-PROBE]`` 这个前缀在**生产代码里只有一个产出点**;
  ② 它**只在正常路径**的 ``logger.info`` 上打,⛔ 永远不在 except 块里
     ⇒ 它出现 = 真有一条入站媒体消息走到了漏斗,⛔ 不可能是 traceback 顺带;
  ③ 它确实**被 ``handle_message`` 调用**(删调用点逆改会红)。

═══════════════════════════════════════════════════════════════════════
关住 / 仍开集
═══════════════════════════════════════════════════════════════════════
关住:产出点唯一性 · 三种输入的判定语义 · ⛔ 不泄漏 media_urls 值 ·
      探针自身不抛 · 在漏斗上真的被调。
开集:⛔ **本门证明不了「板子上真的打出来了」** —— 那要真机流量。
      **零真机验证**。
"""

from __future__ import annotations

import ast
import logging
import pathlib

import pytest

from gateway.platforms.base import (
    MessageEvent,
    MessageType,
    _IM_MEDIA_PROBE,
    _IM_MEDIA_PROBE_ERR,
    log_media_intake_failure,
    log_inbound_media_probe,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BASE = "gateway/platforms/base.py"


def _event(**kw):
    from gateway.config import Platform
    from gateway.session import SessionSource

    src = SessionSource(
        platform=kw.pop("platform", Platform.FEISHU),
        chat_id=kw.pop("chat_id", "oc_x"),
        chat_type=kw.pop("chat_type", "group"),
        user_id="u1",
        user_name="tester",
    )
    return MessageEvent(
        text=kw.pop("text", ""),
        source=src,
        message_type=kw.pop("message_type", MessageType.TEXT),
        media_urls=kw.pop("media_urls", None),
        media_types=kw.pop("media_types", None),
        message_id=kw.pop("message_id", "om_1"),
        **kw,
    )


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def _probe_lines(caplog) -> list[str]:
    return [m for m in _lines(caplog) if m.startswith(_IM_MEDIA_PROBE)]


# ───────────── ① 产出点唯一性:⛔ 不许和 traceback 混淆 ─────────────


def test_probe_prefix_has_exactly_one_production_emit_site() -> None:
    """``[IM-MEDIA-PROBE]`` 在生产代码里**只有一个**产出点。

    ⭐ 这是「上次 6 条命中全是 traceback」那个坑的直接解药:
       前缀只在一处被写出来 ⇒ 日志里出现它,来源没有歧义。
    """
    hits: list[str] = []
    for p in sorted(_REPO.glob("gateway/**/*.py")) + sorted(
        _REPO.glob("plugins/**/*.py")
    ):
        try:
            src = p.read_text()
        except Exception:
            continue
        if "IM-MEDIA-PROBE" not in src:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            if "[IM-MEDIA-PROBE]" in ln:
                hits.append(f"{p.relative_to(_REPO)}:{i}")
    assert len(hits) == 1, (
        "字面量 `[IM-MEDIA-PROBE]` 的出现处不是恰好 1 个 —— "
        "多一个就意味着日志里的命中来源有歧义(上次就是这么废掉一整轮排查的):\n  "
        + "\n  ".join(hits)
    )


def test_probe_is_never_emitted_from_an_except_block() -> None:
    """探针**只在正常路径**打 —— ⛔ 不许出现在任何 except 块里。

    ⭐ 这条才是「和 traceback 分得开」的**结构性**保证:
       它不在异常路径上 ⇒ 一条异常再怎么打印也带不出它。
    """
    tree = ast.parse((_REPO / _BASE).read_text())
    bad: list[str] = []
    for h in ast.walk(tree):
        if not isinstance(h, ast.ExceptHandler):
            continue
        for n in ast.walk(h):
            if isinstance(n, ast.Name) and n.id == "_IM_MEDIA_PROBE":
                bad.append(f"{_BASE}:{n.lineno}")
    assert not bad, "探针前缀出现在 except 块里 ⇒ 它会被异常路径带出来:\n  " + "\n  ".join(bad)


def test_handle_message_actually_calls_the_probe() -> None:
    """探针挂在**所有 adapter 的唯一漏斗**上 —— ⛔ 不是挂在某一家。

    ⭐ 「删调用点」逆改:把 handle_message 里那行删掉,本条必红。
    """
    tree = ast.parse((_REPO / _BASE).read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.AsyncFunctionDef) and fn.name == "handle_message":
            called = {
                n.func.id
                for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert "log_inbound_media_probe" in called, (
                "handle_message 不再调用探针 ⇒ 真机实测又会变成「没反应」"
            )
            return
    pytest.fail("找不到 handle_message —— 量具坏了")


# ───────────── ② 三种输入的判定语义(这才是用户实测要的东西) ─────────────


class TestProbeDiscriminatesOwnership:
    def test_attachment_arrived_and_recognised(self, caplog):
        """附件到了 + 关口认出来 ⇒ 链路通(飞书**阳性对照**长这样)。"""
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.PHOTO,
                    media_urls=["/cache/a.jpg"],
                    media_types=["image/jpeg"],
                )
            )
        (line,) = _probe_lines(caplog)
        assert "n_urls=1" in line
        assert "types=[image/jpeg]" in line
        assert "refs=[local]" in line
        assert "image=T" in line, f"关口没认出图片: {line}"

    def test_attachment_arrived_but_type_misjudged(self, caplog):
        """附件到了、但 MIME 判不出 ⇒ **归 Hermes**(钉钉那三条路就是这一格)。"""
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.DOCUMENT,
                    media_urls=["/cache/a.bin"],
                    media_types=["application/octet-stream"],
                )
            )
        (line,) = _probe_lines(caplog)
        assert "n_urls=1" in line
        assert "image=F" in line, (
            "这一格必须判 False —— 它正是「附件到了但类型判错」的signature"
        )

    def test_media_message_with_zero_attachments_still_speaks(self, caplog):
        """🔴 最要紧的一格:消息是媒体型、附件**一个都没有**。

        ⇒ 差异在下载/投递,**归上游**(板端 / adapter 取件)。
        ⭐ 这一格如果不打日志,用户实测就还是「没反应」——什么都判不出来。
        """
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(message_type=MessageType.PHOTO, media_urls=[], media_types=[])
            )
        (line,) = _probe_lines(caplog)
        assert "n_urls=0" in line and "msg_type=PHOTO" in line

    def test_plain_text_stays_silent(self, caplog):
        """纯文本⛔ 不出声 —— 否则真正要看的那行被淹没。"""
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(_event(text="hello"))
        assert not _probe_lines(caplog)

    def test_remote_vs_local_is_reported_without_the_value(self, caplog):
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.PHOTO,
                    media_urls=["https://cdn.example.com/x.png?sig=SECRET"],
                    media_types=["image/png"],
                )
            )
        (line,) = _probe_lines(caplog)
        assert "refs=[remote]" in line


# ───────────── ③ ⛔ 不许泄漏 · ⛔ 不许把消息处理弄挂 ─────────────


class TestProbeLeaksNothingAndNeverThrows:
    def test_sender_controlled_mime_is_normalized_to_a_closed_label(self, caplog):
        hostile = "image/jpeg;\nINJECT=" + "x" * 4096
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.PHOTO,
                    media_urls=["/cache/a.jpg"],
                    media_types=[hostile],
                )
            )
        (line,) = _probe_lines(caplog)
        assert "types=[image/jpeg]" in line
        assert "INJECT" not in line and "x" * 100 not in line

    def test_probe_samples_are_bounded_and_overflow_is_explicit(self, caplog):
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.PHOTO,
                    media_urls=[f"/cache/{i}.jpg" for i in range(10)],
                    media_types=["image/jpeg"] * 10,
                )
            )
        (line,) = _probe_lines(caplog)
        assert "sample_more=2" in line
        assert line.count("image/jpeg") == 8

    def test_failure_probe_sanitizes_sender_controlled_extra_fields(self, caplog):
        with caplog.at_level(logging.WARNING, logger="gateway.platforms.base"):
            log_media_intake_failure(
                logging.getLogger("gateway.platforms.base"),
                "teams",
                "image",
                "download_failed",
                filename="photo\nFORGED=" + "x" * 4096,
                content_type="image/jpeg;\r\nSECRET",
            )
        line = _lines(caplog)[-1]
        assert "content_type=image/jpeg" in line
        assert "SECRET" not in line and "x" * 100 not in line
        assert "\nFORGED" not in line and len(line) < 500

    @pytest.mark.parametrize(
        "secret",
        [
            "https://files.slack.com/files-pri/T1-F1/x.png?pub_secret=SHHH",
            "/Users/someone/.hermes/cache/private-photo.jpg",
            "data:image/png;base64,QUJDREVGRw==",
        ],
    )
    def test_media_url_values_never_reach_the_log(self, caplog, secret):
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    message_type=MessageType.PHOTO,
                    media_urls=[secret],
                    media_types=["image/png"],
                )
            )
        blob = "\n".join(_lines(caplog))
        for needle in ("pub_secret", "SHHH", "someone", "private-photo", "QUJDREVGRw"):
            assert needle not in blob, (
                f"探针把 media_urls 的值写出去了({needle}) —— "
                "这轮刚修完三处凭据泄漏,⛔ 不许自己又开一个:\n" + blob
            )

    def test_raw_identifiers_are_hashed(self, caplog):
        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(
                _event(
                    chat_id="oc_VERY_IDENTIFYING_CHAT",
                    message_id="om_VERY_IDENTIFYING_MSG",
                    message_type=MessageType.PHOTO,
                    media_urls=["/cache/a.jpg"],
                    media_types=["image/jpeg"],
                )
            )
        blob = "\n".join(_lines(caplog))
        assert "VERY_IDENTIFYING" not in blob, "原始 chat_id/message_id 落进日志了"
        assert "id=" in blob

    def test_correlation_id_is_stable_for_the_same_message(self, caplog):
        def once():
            caplog.clear()
            with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
                log_inbound_media_probe(
                    _event(
                        message_type=MessageType.PHOTO,
                        media_urls=["/cache/a.jpg"],
                        media_types=["image/jpeg"],
                    )
                )
            return _probe_lines(caplog)[0].split("id=")[1].split()[0]

        assert once() == once(), "同一条消息两次拿到不同关联 ID ⇒ 用户对不上号"

    def test_broken_event_never_raises_but_never_goes_silent(self, caplog):
        """探针自己坏了:⛔ 不许抛(会把消息处理弄挂),⛔ 也不许静默。"""

        class Exploding:
            @property
            def media_urls(self):
                raise RuntimeError("boom: /Users/secret/path")

        with caplog.at_level(logging.INFO, logger="gateway.platforms.base"):
            log_inbound_media_probe(Exploding())  # ⛔ 不抛
        errs = [m for m in _lines(caplog) if m.startswith(_IM_MEDIA_PROBE_ERR)]
        assert errs, "探针坏了却一个字都没留 —— 静默失败"
        assert "/Users/secret/path" not in errs[0], (
            "探针的错误分支自己泄漏了绝对路径(safe_exc 没生效)"
        )


def test_probe_reuses_the_production_gates_not_a_copy() -> None:
    """⭐ 关口判据必须从 ``gateway.run`` 取 —— ⛔ 不许在 base.py 里重写一遍。

    重写必然漂移 ⇒ 观测点会说一套、生产做另一套,那比没有观测点更坏。
    """
    src = (_REPO / _BASE).read_text()
    i = src.index("def log_inbound_media_probe")
    j = src.index("\ndef ", i + 10)
    body = src[i:j]
    assert "from gateway.run import" in body, "探针没有复用生产关口"
    for gate in ("_event_media_is_image", "_event_media_is_audio",
                 "_event_media_is_stt_input", "_event_media_is_video"):
        assert f"def {gate}" not in body, f"{gate} 被在 base.py 里重写了一份 ⇒ 会漂移"
