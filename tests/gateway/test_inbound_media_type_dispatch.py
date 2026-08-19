"""入站附件的类型分派:**逐附件 MIME 优先**,⛔ 不许退回裸类别。

🔴 催生本门的原始案例(⭐ 门必须抓得住它):
钉钉一条 richText 里同时有图片和语音时,整条按**第一个**附件定类型
⇒ 语音被送进视觉模型 / 图片进 STT。

⭐ 修法的机制在**消费者**侧(`gateway/run.py`):
`_event_media_is_image/_audio/_video(event, index)` 逐附件读 MIME,
拿不到才落回消息级 `message_type`。
⇒ 那么 provider 侧的义务就是:**往 `media_types` 里塞真 MIME,⛔ 不塞裸类别**。

## 作用域分格(⛔ 不许读成「入站分派已经全对了」)

* **闭集(本门强制)**:任何 adapter 都不许把 `"image"` / `"audio"` / `"video"` /
  `"file"` 这类**裸类别**赋给 `media_types`。全仓 AST+文本扫描,新增即红。
  实测:20 个有 `media_urls` 的 adapter **全部使用真 MIME**,裸类别命中 **0**
  ⇒ 钉钉那条缺陷**没有兄弟**。本门把这个状态钉住。
* **⚠️ 开集(明说)**:`media_types` **比 `media_urls` 短**时,超出部分会静默
  落回消息级类型。那是**运行时长度关系**,静态判据罩不住 ⇒ ⛔ 本门不声称覆盖。
"""

import glob
import pathlib
import re

import pytest

_BARE = ("image", "audio", "video", "file", "photo", "voice", "document")


def _adapters():
    files = sorted(glob.glob("plugins/platforms/*/adapter.py"))
    files += [f for f in sorted(glob.glob("gateway/platforms/*.py"))
              if pathlib.Path(f).stem not in (
                  "base", "helpers", "__init__", "api_server", "media_cache",
                  "webhook", "webhook_filters", "_http_client_limits",
                  "signal_format", "signal_rate_limit", "whatsapp_common",
                  "yuanbao_proto", "yuanbao_sticker", "yuanbao_media",
                  "msgraph_webhook", "zet_agent", "zet_agent_cron",
                  "zet_agent_goals")]
    return [f for f in files
            if "media_urls" in pathlib.Path(f).read_text(encoding="utf-8", errors="replace")]


class TestNoAdapterFallsBackToBareCategories:
    def test_media_types_never_receives_a_bare_category(self):
        files = _adapters()
        assert len(files) >= 15, f"作用域塌了,只扫到 {len(files)} 个 adapter"
        bad = []
        pat = re.compile(
            r'media_types\s*(?:=\s*\[?|\.append\(\s*)"(' + "|".join(_BARE) + r')"')
        for f in files:
            src = pathlib.Path(f).read_text(encoding="utf-8", errors="replace")
            for m in pat.finditer(src):
                line = src[:m.start()].count("\n") + 1
                bad.append(f"  {f}:{line} 塞了裸类别 {m.group(1)!r}")
        assert not bad, (
            "以下 adapter 把**裸类别**塞进 media_types —— 消费者只认真 MIME,\n"
            "拿不到就落回**消息级**类型 ⇒ 混合附件消息会整条按第一个附件分派\n"
            "(语音进视觉模型 / 图片进 STT):\n" + "\n".join(bad))

    def test_the_probe_can_see_a_planted_bare_category(self):
        """⭐ 阳性对照:人造一条裸类别赋值,判据必须命中(⛔ 不许恒绿)。"""
        pat = re.compile(
            r'media_types\s*(?:=\s*\[?|\.append\(\s*)"(' + "|".join(_BARE) + r')"')
        assert pat.search('media_types = ["image"]')
        assert pat.search('media_types.append("voice")')
        # 🔴 必须保持不变:真 MIME ⛔ 不许被误判
        assert not pat.search('media_types = ["image/png"]')
        assert not pat.search('media_types.append("audio/amr")')

    def test_every_adapter_uses_real_mime_literals(self):
        """⭐ 反面还不够 —— 正面也要有:每个 adapter 都得**真的**出现过 MIME 字面量。"""
        mime = re.compile(r'"(?:image|audio|video|application|text)/[a-z0-9.+*-]+"')
        missing = [f for f in _adapters()
                   if not mime.search(pathlib.Path(f).read_text(encoding="utf-8",
                                                                errors="replace"))]
        assert not missing, (
            "以下 adapter 一个 MIME 字面量都没有 ⇒ 它的 media_types 从哪来存疑:\n"
            + "\n".join(f"  {f}" for f in missing))


class TestTheConsumerDispatchesPerAttachment:
    """消费者机制 —— ⭐ 这是飞书/钉钉「做对了」的那份,其余 provider 复用的正是它。"""

    @staticmethod
    def _event(types, message_type=None):
        from gateway.platforms.base import MessageType  # noqa: F401
        from types import SimpleNamespace
        return SimpleNamespace(media_urls=["/a", "/b"], media_types=list(types),
                               message_type=message_type)

    def test_a_mixed_message_routes_each_attachment_on_its_own_mime(self):
        """✅ 原始案例:图片 + 语音同一条 ⇒ 各归各,⛔ 不互相污染。"""
        from gateway.run import _event_media_is_audio, _event_media_is_image

        ev = self._event(["image/png", "audio/amr"])
        assert _event_media_is_image(ev, 0) and not _event_media_is_image(ev, 1)
        assert _event_media_is_audio(ev, 1) and not _event_media_is_audio(ev, 0)

    def test_a_missing_per_attachment_mime_falls_back_to_the_message_type(self):
        """🔴 **必须保持不变**:拿不到逐附件 MIME 时仍落回消息级 —— 那是既有兜底。"""
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_image

        ev = self._event([], message_type=MessageType.PHOTO)
        assert _event_media_is_image(ev, 0) is True

    def test_an_out_of_range_index_is_safe(self):
        """🔴 **必须保持不变**:``media_types`` 比 ``media_urls`` 短时⛔不许炸。"""
        from gateway.run import _event_media_type_at

        assert _event_media_type_at(self._event(["image/png"]), 1) == ""

    def test_a_document_next_to_an_image_is_not_treated_as_an_image(self):
        """🔴 既有 docstring 明写的现场:文档挨着图片一起发,⛔ 不许被 base64 进视觉。"""
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_image

        ev = self._event(["image/png", "application/pdf"],
                         message_type=MessageType.PHOTO)
        assert _event_media_is_image(ev, 0) is True
        assert _event_media_is_image(ev, 1) is False, (
            "文档被当成图片 ⇒ base64 进 vision content part ⇒ provider 400")

    def test_stt_never_swallows_a_plain_audio_attachment(self):
        """🔴 **必须保持不变**:``AUDIO`` / ``DOCUMENT`` ⛔ 不进自动 STT。"""
        from gateway.platforms.base import MessageType
        from gateway.run import _event_media_is_stt_input

        assert _event_media_is_stt_input(
            self._event(["audio/amr"], MessageType.AUDIO), 0) is False
        assert _event_media_is_stt_input(
            self._event(["audio/amr"], MessageType.VOICE), 0) is True


class TestFeishuAndDingtalkReferenceBehaviourIsUnchanged:
    """🔴 飞书/钉钉是**已知可用的参照** —— 本门⛔ 不许推着它们改。"""

    def test_dingtalk_still_maps_each_attachment_separately(self):
        src = pathlib.Path("plugins/platforms/dingtalk/adapter.py").read_text()
        # ⚠️ 钉钉用的是**通配子类型** ``"image/*"`` / ``"audio/*"``(实取得出,⛔ 不是猜)
        #    —— 消费者用 ``startswith("image/")`` 判,通配形式同样成立。
        assert re.search(r'"image/[a-z*]+"', src), "钉钉的逐附件 image MIME 没了"
        assert re.search(r'"audio/[a-z0-9*]+"', src), "钉钉的逐附件 audio MIME 没了"

    def test_feishu_still_emits_real_mime(self):
        src = pathlib.Path("plugins/platforms/feishu/adapter.py").read_text()
        assert re.search(r'"(?:image|audio|video|application)/[a-z0-9.+*-]+"', src)

    def test_the_open_half_is_declared(self):
        """⭐ ⛔ 扩不成闭集就明说 —— ⛔ 不许把本门读成「入站分派全对了」。"""
        doc = pathlib.Path(__file__).read_text(encoding="utf-8")
        assert "开集" in doc and "⛔ 本门不声称覆盖" in doc


# ═══════════ 第一批入站失败可观测性:飞书 ═══════════

class TestFeishuMediaFailuresAreObservableWithoutLeaking:
    """🔴 飞书媒体路径原本把 **原始 url + 原始异常 + traceback** 一起写进日志。

    飞书媒体 url 带鉴权参数,``ClientResponseError.__str__`` 会把整条 url 带出来,
    ``exc_info=True`` 还会让 logging 重新格式化原始异常
    ⇒ 一次下载失败就把渠道凭据**持久化**进 ``agent.log``。

    ⭐ 「照抄」三问(先例 = ``gateway/platforms/weixin.py`` 那份):
      ① 先例:调共享 ``log_media_intake_failure``,只记异常**类型名** + host 摘要,
         host 取 ``.hostname`` 而非 ``.netloc``(netloc 含 userinfo);
         **记账**与**用户侧提示**分开(后者⛔ 不带 reason 码)。
      ② 我这份:同一个共享 helper、同样传 kind / reason / url / exc。
      ③ 差异:飞书的 ``send image`` 那处 ``url=""`` —— 那是**本地路径**不是 url,
         传进去只会被当成 host 解析,⛔ 传空更诚实。其余逐条相同。
    """

    _SITES = 5

    def test_no_raw_url_or_traceback_on_the_media_paths(self):
        src = pathlib.Path("plugins/platforms/feishu/adapter.py").read_text()
        bad = []
        for m in re.finditer(r'logger\.(?:warning|error)\((.{0,220}?)\)\n', src, re.S):
            body = " ".join(m.group(1).split())
            if not re.search(r'download|cache.*resource|Failed to send image', body, re.I):
                continue
            if re.search(r'\b(image_url|animation_url|image_path)\b', body) or "exc_info=True" in body:
                bad.append(f"  :{src[:m.start()].count(chr(10))+1}  {body[:100]}")
        assert not bad, (
            "飞书媒体日志仍在写 raw url / raw exception / traceback ⇒ 渠道凭据落盘:\n"
            + "\n".join(bad))

    def test_every_site_moved_to_the_shared_helper(self):
        src = pathlib.Path("plugins/platforms/feishu/adapter.py").read_text()
        n = src.count("log_media_intake_failure(")
        assert n >= self._SITES, (
            f"只有 {n} 处接上共享记账 helper,应为 {self._SITES} —— 兄弟调用点没跟全")

    def test_the_helper_records_host_not_credentials(self):
        """⭐ 判据落在 helper 的**实际输出**上,⛔ 不是「它被调用了」。"""
        import logging

        from gateway.platforms.base import log_media_intake_failure

        seen = []

        class _L:
            def warning(self, fmt, *args):
                seen.append(fmt % args)

        log_media_intake_failure(
            _L(), "feishu", "image", "download_failed",
            url="https://user:hunter2@open.feishu.cn/x?token=TOPSECRET",
            exc=ValueError("403 url='https://open.feishu.cn/x?token=TOPSECRET'"))
        out = seen[0]
        for leak in ("TOPSECRET", "hunter2", "user:hunter2"):
            assert leak not in out, f"泄漏 {leak!r}:{out}"
        assert "open.feishu.cn" in out, "host 也被抹了 ⇒ 定位不到是哪个渠道"
        assert "ValueError" in out, "异常类型名丢了 ⇒ 分不清超时和未授权"
        assert "download_failed" in out, "reason 码丢了 ⇒ 失败不可区分"

    def test_the_bare_category_closed_set_still_holds(self):
        """🔴 上一轮那个「裸类别 0 命中」的闭集结论,**加改动后必须重新成立**。"""
        pat = re.compile(
            r'media_types\s*(?:=\s*\[?|\.append\(\s*)"(' + "|".join(_BARE) + r')"')
        src = pathlib.Path("plugins/platforms/feishu/adapter.py").read_text()
        assert not pat.search(src)

    def test_feishu_send_paths_are_untouched(self):
        """🔴 **必须保持不变**:飞书是参照 —— 发送侧的既有分支一个都没动。"""
        src = pathlib.Path("plugins/platforms/feishu/adapter.py").read_text()
        for marker in ("refusing to fall back to the chat timeline",
                       "_build_create_message_request(\"thread_id\", body)",
                       "reply_in_thread=bool(thread_id)"):
            assert marker in src, f"飞书发送侧的既有行为被动了:{marker}"
