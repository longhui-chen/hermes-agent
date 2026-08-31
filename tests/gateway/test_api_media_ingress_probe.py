"""入站媒体观测点(**真正的入口**:`/p/{profile}/v1/chat/completions`)。

═══════════════════════════════════════════════════════════════════════
🔴 上一版为什么失败 —— 门全绿,而生产流量一次没执行
═══════════════════════════════════════════════════════════════════════
上一轮我把探针挂在 ``BasePlatformAdapter.handle_message``,理由是
``RelayAdapter`` 继承了它。当时的 15 条门:①调用点存在 ②前缀唯一
③不在 except 块里 ④不泄漏 ⑤复用生产关口 —— **全绿,4 次逆改全红**。
真机实测:``IM-MEDIA-PROBE = 0``,``handle_message = 0``。

⭐ 病因不在门的严格程度,在**门的 oracle**:
   那些门的判据全部来自**我自己写的那段代码**(它在不在、长什么样),
   ⛔ 没有一条来自「**谁真的会调用它**」。
   于是它们能证明「代码写对了」,**证明不了「这条代码在生产路径上」**。
   ⇒ ``gate-oracle-must-not-share-the-implementation`` 的原话:
     oracle ⛔ 不许与实现共享判据。「结构上在调用链里」和「实际被调用」
     是两件事 —— **继承了⛔ 推不出会被执行**。

═══════════════════════════════════════════════════════════════════════
⭐ 这一版怎么不踩同一个坑 —— 三条**独立于我的代码**的判据
═══════════════════════════════════════════════════════════════════════
1. **oracle 来自生产路由表**:app 由 ``_register_profile_api_routes``
   **自己**注册(⛔ 不是我在测试里手工 ``add_post``),再用真 HTTP 客户端
   POST 进去。它答的是「一个真请求打进来会不会经过探针」。
2. **端到端而非源码扫描**:断言的是**日志里真的出现了那一行**,
   ⛔ 不是「源码里有这个调用」。
3. ⭐ **`seq` 单调递增**:探针**每个请求都出声**(无媒体、坏 JSON 也出声)。
   ⇒ 「没有那行」只剩**一种**成因:没被执行。
   上一版对纯文本沉默,于是沉默有两种成因,混在一起就什么都判不了 ——
   这正是 ``instrument-silence-is-not-evidence``:**工具的沉默不是证据**。

⚠️ 仍是开集(⛔ 明说):本门跑在测试进程里,证明的是「这条路由上的请求会
   打到探针」。**板端真实请求**是否走这条路由,只能靠云机上的 access log +
   探针计数对照 —— 那一半**零真机验证**,已在报告里保留。
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    MAX_CONTENT_LIST_SIZE,
    _API_MEDIA_PROBE,
    _probe_describe_image_ref,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_API = "gateway/platforms/api_server.py"

TINY_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


_KEY = "opensslrandhex32strongkeyforprobe"


def _app_from_production_routing() -> tuple[web.Application, APIServerAdapter]:
    """⭐ 路由由**生产的注册方法**装,⛔ 不是我在测试里手工 add_post。

    这样 oracle 就来自路由表本身 —— 「这条 URL 上真的挂着谁」。
    """
    # ⚠️ 必须带上 API key:本端点在**进入 handler 之前**就鉴权,
    #    不带 key 是 401 ⇒ 探针根本轮不到执行。
    #    (这本身也是一条边界:探针覆盖的是「过了鉴权、进到 chat handler」的请求;
    #     401 在 access log 里是另一种可区分的信号 —— 已在文件头声明。)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": _KEY}))
    # ⚠️ 命名 profile 走的是**profile 作用域**的 API_SERVER_KEY,测试进程里没有
    #    那份 secret ⇒ 一律 401,探针根本轮不到执行。这里把鉴权打桩放行。
    #    ⭐ 明说这是打桩:本门证明的是「**过了鉴权的**请求会经过探针」。
    #      板端真实请求确实过了鉴权(用户实测里是 `200`),所以这个前提成立;
    #      401 在 access log 里是另一种可区分的信号,⛔ 不会与探针沉默混淆。
    adapter._check_auth = lambda _request: None  # noqa: E731
    app = web.Application()
    app["api_server_adapter"] = adapter
    adapter._register_profile_api_routes(app.router)
    return app, adapter


class _Captured:
    """自带 handler 的日志捕获器 —— ⛔ 不用 pytest 的 caplog。

    🔴 为什么:实测本仓 pytest 会话里 ``gateway.platforms.api_server`` 的
    **有效级别是 WARNING**(root=30 且该 logger 是 NOTSET),``logger.info``
    在源头就被丢掉;``caplog.at_level`` 在本仓的 logging 配置下没能把它打开
    —— 一条都收不到。**那是量具坏,不是探针没跑**,⛔ 不许当成结论。
    ⇒ 直接给目标 logger 挂 handler 并强制 INFO,用完还原。

    ⭐ 每次进入都先打一条**金丝雀**并断言收得到 —— 量具自校准,
       ⛔ 不许出现「捕获器坏了 ⇒ 空列表 ⇒ 断言全过」这种假绿。
    """

    _CANARY = "__probe_capture_canary__"

    def __init__(self):
        import gateway.platforms.api_server as _m
        self._logger = _m.logger
        self.records: list[logging.LogRecord] = []

    def __enter__(self):
        outer = self

        class _H(logging.Handler):
            def emit(self, record):
                outer.records.append(record)

        self._h = _H(level=logging.DEBUG)
        self._old_level = self._logger.level
        self._logger.addHandler(self._h)
        self._logger.setLevel(logging.DEBUG)
        self._logger.info(self._CANARY)
        assert any(self._CANARY in r.getMessage() for r in self.records), (
            "捕获器自校准失败:金丝雀都没收到 ⇒ **量具坏了**,"
            "⛔ 后面的空结果一个字都不可信"
        )
        self.records.clear()
        return self

    def __exit__(self, *exc):
        self._logger.removeHandler(self._h)
        self._logger.setLevel(self._old_level)
        return False

    def lines(self, prefix: str = _API_MEDIA_PROBE) -> list[str]:
        return [r.getMessage() for r in self.records
                if r.getMessage().startswith(prefix)]


def _probe_lines(cap) -> list[str]:
    return cap.lines()


def _seq_of(line: str) -> int:
    return int(line.split("seq=")[1].split()[0])


_AUTH = {"Authorization": f"Bearer {_KEY}"}


async def _post(cli, payload):
    return await cli.post("/p/main/v1/chat/completions", json=payload, headers=_AUTH)


# ═════════ ① 决定性:真请求打进来,探针真的被执行 ═════════


class TestARealRequestActuallyHitsTheProbe:
    @pytest.mark.asyncio
    async def test_route_registered_by_production_code_reaches_the_probe(self):
        """⭐ 这条就是上一轮缺的那一条。

        路由由 ``_register_profile_api_routes`` 注册,请求由真 HTTP 客户端发出,
        断言**日志里真的出现了探针行** —— ⛔ 不是「源码里有这个调用」。
        """
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "看这张图"},
                            {"type": "image_url", "image_url": {"url": TINY_PNG}},
                        ]}],
                    })
        lines = _probe_lines(caplog)
        assert lines, (
            "一个真请求打到生产注册的路由上,探针**一行都没打** —— "
            "这正是上一轮那个坑:代码在、门绿、但不在生产路径上。"
        )
        profile_id = hashlib.sha256(b"main").hexdigest()[:8]
        assert "image=1" in lines[-1] and f"profile={profile_id}" in lines[-1]

    @pytest.mark.asyncio
    async def test_seq_advances_per_request(self):
        """⭐ 判据从「有没有那行」换成「seq 有没有前进」。

        seq 停着不动 = 没被执行;seq 前进而 image=0 = 执行了、请求里真没图。
        **只有这样,沉默才只剩一种成因。**
        """
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    for _ in range(3):
                        await _post(cli, {
                            "messages": [{"role": "user", "content": "纯文本"}]
                        })
        seqs = [_seq_of(l) for l in _probe_lines(caplog)]
        assert len(seqs) == 3, f"3 个请求只打了 {len(seqs)} 行"
        assert seqs == sorted(seqs) and len(set(seqs)) == 3, f"seq 没有单调递增: {seqs}"


class TestItSpeaksOnEveryRequestSoSilenceHasOneCause:
    @pytest.mark.asyncio
    async def test_no_media_still_emits(self):
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": "hi"}]})
        (line,) = _probe_lines(caplog)
        assert "image=0" in line and "verdict=no_media" in line

    @pytest.mark.asyncio
    async def test_bad_json_still_emits(self):
        """⭐ 连 JSON 都解不开也要出声 —— 否则沉默又有第二种成因。"""
        app, _adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                await cli.post(
                    "/p/main/v1/chat/completions",
                    data=b"{not json",
                    headers={"Content-Type": "application/json", **_AUTH},
                )
        (line,) = _probe_lines(caplog)
        assert "verdict=bad_json" in line


class TestCurrentTurnOnly:
    @pytest.mark.asyncio
    async def test_history_media_cannot_change_the_current_turn_verdict(self):
        """历史里的图/文件不能冒充当前轮入站媒体。"""
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": "[file: /old/report.pdf]"},
                            {"type": "image_url", "image_url": {"url": TINY_PNG}},
                        ]},
                        {"role": "assistant", "content": "历史回答"},
                        {"role": "user", "content": "当前轮纯文本"},
                    ]})
        (line,) = _probe_lines(caplog)
        assert "shapes=[str]" in line
        assert "image=0" in line and "filenote=0" in line
        assert "verdict=no_media" in line, line

    @pytest.mark.asyncio
    async def test_current_turn_counts_match_the_normalizer_limit(self):
        """探针不能把下游会丢弃的第 1001 个 part 记成已入站。"""
        app, adapter = _app_from_production_routing()
        parts = [
            {"type": "image_url", "image_url": {"url": TINY_PNG}}
            for _ in range(MAX_CONTENT_LIST_SIZE + 1)
        ]
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": parts}]})
        (line,) = _probe_lines(caplog)
        assert f"image={MAX_CONTENT_LIST_SIZE}" in line
        assert "parts_more=1" in line and f"img_more={MAX_CONTENT_LIST_SIZE - 8}" in line

    @pytest.mark.asyncio
    async def test_current_turn_type_normalization_matches_the_normalizer(self):
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": " " * 65 + "image_url", "image_url": {"url": TINY_PNG}},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "image=1" in line and "other=0" in line
        assert "unsupported=[]" in line and "verdict=image_only" in line


# ═════════ ② LS 那两个 bug 的【镜像侧证据】 ═════════


class TestMirrorEvidenceForTheLocalServerBugs:
    """⭐ 两侧**独立判据**:LS 说「我没送出去」,这边说「我收到的是什么」。

    ⛔ 不共享参照系 —— 同一参照系 = 同一盲区。
    """

    @pytest.mark.asyncio
    async def test_a_file_arrives_as_a_text_note_not_a_part(self):
        """🔴 探针原本的盲区 —— 2026-08-17 读 LS 源码实证后补上。

        ``internal/backend/hermes/chat.go`` 逐字写着:
          「{type:image_url,…}; **non-image media (files) are appended to the
            user text as a line "[file: <url>]"** since hermes' Chat Completions
            endpoint does not (today) expose a file content part」
        ⇒ 文件**根本不是 part**;而且没有图片时 content 是**纯字符串**。
        只按 part 类型枚举的话,一条文件消息在探针眼里 = 一条纯文本消息,
        **正好落在用户最要紧的那一格上**。

        ⭐ 这是 `enumerating-shapes-exempts-the-rest` 的又一次:
           判据按「形状清单」枚举 ⇒ 没列到的形状自动免检。
        """
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content":
                        "看看这个\n[file: /root/.hermes/uploads/report.pdf]"}]})
        (line,) = _probe_lines(caplog)
        assert "shapes=[str]" in line, f"没认出「纯字符串 content」这一支: {line}"
        assert "filenote=1" in line, f"没数出 [file: …] 文本行: {line}"
        assert "verdict=filenote_only" in line
        assert "/root/.hermes" not in line and "report.pdf" not in line, (
            "⛔ 探针把文件路径写出去了 —— 那会泄漏 HERMES_HOME 布局"
        )

    @pytest.mark.asyncio
    async def test_image_plus_file_note_is_distinguishable(self):
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "[file: /a/b.pdf]\n[file: /a/c.docx]"},
                        {"type": "image_url", "image_url": {"url": TINY_PNG}},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "shapes=[list]" in line and "image=1" in line and "filenote=2" in line
        assert "verdict=image_and_filenote" in line

    @pytest.mark.asyncio
    async def test_image_missing_shows_up_as_image_zero(self):
        """LS 报 ``model input data url missing`` ⇒ 这边应当看到 image=0。"""
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "[图片]"},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "image=0" in line and "text=1" in line

    @pytest.mark.asyncio
    async def test_file_part_is_recorded_as_unsupported(self):
        """🔴 LS 报 ``relay request rejected: invalid_request``。

        本端点对 ``file`` / ``input_file`` part 是**直接 400**
        (``_normalize_multimodal_content`` 抛 ``unsupported_content_type``)。
        ⇒ 探针把它记成 ``unsupported=[file]``,**当场定位**:
          ⛔ 不是链路把文件弄丢了,是这个端点根本不收。
        """
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    resp = await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": "text", "text": "看这个文件"},
                        {"type": "file", "file": {"file_id": "f_1"}},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "filepart=1" in line and "unsupported=[file]" in line
        assert resp.status == 400, (
            "本测试同时钉住端点的既有契约:file part 就是 400。"
            "⛔ 若哪天它不再 400,这条 finding 的前提就变了,必须重判。"
        )

    @pytest.mark.asyncio
    async def test_audio_part_is_recorded_as_unsupported(self):
        """音频同理:``input_audio`` 不在任何一个已支持集合里 ⇒ 400。"""
        app, adapter = _app_from_production_routing()
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    resp = await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "other=1" in line and "input_audio" in line
        assert resp.status == 400


# ═════════ ③ 形状描述:出形状,⛔ 不出内容 ═════════


class TestImageRefShapeCarriesNoContent:
    @pytest.mark.parametrize(
        "ref,expect",
        [
            ("data:image/png;base64,QUJD", "data:image/png;b64;4b"),
            ("data:image/jpeg;base64,", "data:image/jpeg;b64;0b"),
            ("data:application/pdf;base64,QQ==", "data:other;b64;4b"),
            ("data:image/png;base64", "data:malformed"),
            ("https://cdn.example.com/x.png?sig=SECRET", "https"),
            ("http://h/x.png", "http"),
            ("file:///etc/passwd", "other"),
            ("", "empty"),
            (None, "empty"),
        ],
    )
    def test_shape_only(self, ref, expect):
        assert _probe_describe_image_ref(ref) == expect

    @pytest.mark.parametrize("ref", [
        "data:image/png\nFORGED=1;base64,QQ==",
        "data:" + "x" * 300 + ";base64,QQ==",
    ])
    def test_shape_never_logs_a_client_controlled_header(self, ref):
        shape = _probe_describe_image_ref(ref)
        assert "FORGED" not in shape and "\n" not in shape
        assert len(shape) < 64, shape

    def test_oversized_data_header_is_malformed(self):
        assert _probe_describe_image_ref("data:" + "x" * 300 + ",QQ==") == "data:malformed"

    @pytest.mark.asyncio
    async def test_end_to_end_never_logs_the_payload(self):
        app, adapter = _app_from_production_routing()
        secret_png = "data:image/png;base64,U0VDUkVUUEFZTE9BRA=="
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": secret_png}},
                        {"type": "image_url", "image_url": {
                            "url": "https://files.slack.com/x.png?pub_secret=SHHH"}},
                    ]}]})
        blob = "\n".join(_probe_lines(caplog))
        for needle in ("U0VDUkVU", "SECRETPAYLOAD", "pub_secret", "SHHH", "files.slack.com"):
            assert needle not in blob, f"探针把内容/凭据写出去了({needle}):\n{blob}"
        assert "data:image/png;b64;" in blob and "https" in blob

    @pytest.mark.asyncio
    async def test_unknown_part_type_is_logged_as_a_fixed_placeholder(self):
        app, adapter = _app_from_production_routing()
        attacker_type = "custom\nFORGED=1-" + "x" * 4096
        with _Captured() as caplog:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_run_agent", new=MagicMock()):
                    await _post(cli, {"messages": [{"role": "user", "content": [
                        {"type": attacker_type, "payload": "ignored"},
                    ]}]})
        (line,) = _probe_lines(caplog)
        assert "unsupported=[other]" in line
        assert "FORGED" not in line and "\n" not in line

    def test_profile_is_hashed_before_logging(self):
        from gateway.platforms.api_server import _log_api_media_ingress

        profile = "main\nFORGED=1"
        with _Captured() as caplog:
            _log_api_media_ingress(None, {"messages": []}, profile=profile)
        (line,) = _probe_lines(caplog)
        assert profile not in line and "\n" not in line
        assert f"profile={hashlib.sha256(profile.encode()).hexdigest()[:8]}" in line


# ═════════ ④ 结构性:前缀唯一 + 不在 except 块里(这两条仍然要) ═════════


def test_prefix_has_exactly_one_production_emit_site() -> None:
    hits = []
    for p in sorted(_REPO.glob("gateway/**/*.py")) + sorted(_REPO.glob("plugins/**/*.py")):
        try:
            src = p.read_text()
        except Exception:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            if "[API-MEDIA-INGRESS]" in ln:
                hits.append(f"{p.relative_to(_REPO)}:{i}")
    assert len(hits) == 1, "产出点不唯一 ⇒ 日志命中来源有歧义:\n  " + "\n  ".join(hits)


def test_probe_prefix_never_appears_in_an_except_block() -> None:
    tree = ast.parse((_REPO / _API).read_text())
    bad = [
        f"{_API}:{n.lineno}"
        for h in ast.walk(tree)
        if isinstance(h, ast.ExceptHandler)
        for n in ast.walk(h)
        if isinstance(n, ast.Name) and n.id == "_API_MEDIA_PROBE"
    ]
    assert not bad, "探针前缀出现在 except 块里 ⇒ 会被 traceback 带出来:\n  " + "\n  ".join(bad)


def test_probe_never_raises_into_the_request_path() -> None:
    """探针自己坏了:⛔ 不许抛(会把请求弄挂),⛔ 也不许静默。"""
    from gateway.platforms.api_server import _log_api_media_ingress

    class Exploding(dict):
        # ⚠️ 必须非空:`(body or {})` 对**空** dict 子类会退化成真 dict,
        #    于是它永远不炸 —— 我第一版就是这么写的,量具自己失效了。
        def __init__(self):
            super().__init__(messages=[])

        def get(self, *_a, **_k):
            raise RuntimeError("boom /Users/secret/path")

    with _Captured() as caplog:
        _log_api_media_ingress(None, Exploding())  # ⛔ 不抛
    errs = [m for m in (r.getMessage() for r in caplog.records)
            if m.startswith("[API-MEDIA-INGRESS-ERR]")]
    assert errs, "探针坏了却一个字没留 —— 静默失败"
    assert "/Users/secret/path" not in errs[0], "错误分支自己泄漏了绝对路径"
