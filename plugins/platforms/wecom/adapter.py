"""
WeCom (Enterprise WeChat) platform adapter.

Uses the WeCom AI Bot WebSocket gateway for inbound and outbound messages.
The adapter focuses on the core gateway path:

- authenticate via ``aibot_subscribe``
- receive inbound ``aibot_msg_callback`` events
- send outbound markdown messages via ``aibot_send_msg``
- upload outbound media via ``aibot_upload_media_*`` and send native attachments
- best-effort download of inbound image/file attachments for agent context

Configuration in config.yaml:
    platforms:
      wecom:
        enabled: true
        extra:
          bot_id: "your-bot-id"          # or WECOM_BOT_ID env var
          secret: "your-secret"          # or WECOM_SECRET env var
          websocket_url: "wss://openws.work.weixin.qq.com"
          dm_policy: "pairing"           # open | allowlist | disabled | pairing
          allow_from: ["user_id_1"]
          group_policy: "pairing"        # open | allowlist | disabled | pairing
          group_allow_from: ["group_id_1"]
          groups:
            group_id_1:
              allow_from: ["user_id_1"]
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from urllib.parse import unquote, urlparse

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None  # type: ignore[assignment]

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    safe_exc,
    safe_traceback,
    safe_url_for_log,
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    cache_document_from_bytes,
    cache_image_from_bytes,
)
from utils import env_float

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret


def _get_scoped_secret(name, default=None):
    """Scope-aware credential read with the default-profile startup fallback.

    Secondary profiles construct their adapters under a profile secret
    scope -- the scope is authoritative and a scoped miss returns ``default``
    (no cross-profile borrow from ``os.environ``, which may hold another
    profile's value). The DEFAULT profile's adapter constructs and sends
    *unscoped* under multiplexing, where a bare ``get_secret`` would raise
    ``UnscopedSecretError`` and crash this path; there ``os.environ`` is that
    profile's own value, so fall back to it. Same pattern as the Slack
    ``SLACK_APP_TOKEN`` read (#59739) and
    ``gateway/platforms/whatsapp_common.py::_get_wsecret``.
    """
    try:
        val = _scoped_get_secret(name, default)
    except _UnscopedSecretError:
        val = os.getenv(name)
    return val if val is not None else default


logger = logging.getLogger(__name__)

#: 企业微信回调 msgtype 的**已知全集**，按官方 AI Bot SDK 的类型定义切分。
#
# 来源（⛔ 不是从我们自己的实现反推 —— 那样 oracle 会与实现共享判据）：
#   https://github.com/WecomTeam/aibot-node-sdk
#   src/types/message.ts  —— `MessageType` 枚举 + 各 *Message 接口
#
# ⭐ 判据是**闭集**：`_extract_media` 不靠「我列到了哪几种形状」决定取不取媒体，
# 凡是不在已知全集里的 msgtype 一律留痕告警，⛔ 不许"没写到的就静默丢"。
#
# 🔴 三个集合必须分开，⛔ 不许合并 —— 合并过一次，代价见下：
#   * 可下载：`image` / `file` / `video` 三者结构同构（`url` + `aeskey`，
#     url 五分钟有效且已加密）。
#   * 仅转写：`voice` **没有任何可下载资源**，`VoiceContent` 只有
#     `content: string`（语音转成的文本）。文字由 `_extract_text` 取。
#     ⚠️ 我曾把 `voice` 并进可下载集，理由写成「本模块只负责语音文件本体」
#     —— **根本不存在这个本体**，结果是每条合法语音都误报
#     `no_media_reference`。⭐ 错误的理由留在注释里比错误的代码活得久。
#   * 纯文本 / 容器：其余。
_WECOM_DOWNLOADABLE_MSGTYPES: frozenset[str] = frozenset({"image", "file", "video"})
_WECOM_TRANSCRIPT_MSGTYPES: frozenset[str] = frozenset({"voice"})
#: 图文混排子项的 msgtype —— 官方限定 `'text' | 'image'`，⛔ 不是顶层那一套。
_WECOM_MIXED_ITEM_MSGTYPES: frozenset[str] = frozenset({"text", "image"})


class MediaFailure(NamedTuple):
    """一次入站媒体没取到 —— ``kind`` 是哪类附件，``reason`` 是**为什么**。

    🔴 原先这里只有 ``kind``(``failed_kinds: List[str]``)，于是七种成因被压成
    一个词，用户拿到的建议永远是同一句「可以先压缩后再试」——
    对**链接过期**、**解密失败**、**磁盘写满**全都是错误指引:他压缩完再发
    还是失败,而真正该做的(立刻重发 / 清理空间)一个字都没说。
    ⭐ 全局硬规则「面向用户的错误必须分类、可行动」在这里被违反了,
      ⛔ 而且「补错的建议比不补更坏」——它让用户去做一件确定无效的事。
    """

    kind: str
    reason: str


#: reason ⇒ 给**用户**的一句可行动的话。闭集,新增 reason 必须同步登记。
#: ⛔ 每条只有一句 —— 判据「不写它用户会不会做错事」,不写就会:
#:   过期类不立刻重发就永远拿不到,存储类不清理空间重发多少次都白搭。
#: ⛔ 一个字都不含 reason 码 / 路径 / 原始错误(那些在日志里)。
_MEDIA_FAILURE_ADVICE: Dict[str, str] = {
    # 企微媒体 url 仅 5 分钟有效 ⇒ 过期与网络故障在这里表现相同
    "download_failed": "没能下载到（附件链接 5 分钟后失效）。请重新发送。",
    "decrypt_failed": "内容已损坏或链接已失效。请重新发送。",
    "cache_write_failed": "设备存储写入失败。请清理存储空间后重试。",
    "base64_decode_failed": "内容已损坏。请重新发送。",
    "not_an_image": "这个图片格式无法识别。请改用 JPG 或 PNG 重发。",
    "no_media_reference": "这条消息里没有附件内容。请重新发送。",
    # ⭐ 唯一一处「压缩」是**正确**建议的地方 —— 它对应的是真的太大。
    # ⛔ 而它此前被误用在所有成因上,那才是 P2-4 的病根。
    "too_large": "文件超出企业微信的大小上限。请压缩后再发，或改发较小的文件。",
    "payload_malformed": "这条消息里没有附件内容。请重新发送。",
}
#: ⭐ 未知 reason 也**不许伪装成成功**、⛔ 也不许把 reason 码甩给用户。
_MEDIA_FAILURE_ADVICE_FALLBACK = "暂时读不到。请稍后重新发送。"

#: 给**用户**看的附件名字 —— ⛔ 不许把内部标识 ``image`` / ``file`` 直接拼进
#: 面向用户的文案(我上一轮就是这么写的,用户会看到「未能读取你发送的附件
#: (image)」)。照抄 ``weixin.py`` 的 ``_MEDIA_KIND_LABEL``。
class WeComMediaTooLarge(ValueError):
    """远端媒体超过大小上限 —— ⛔ 与「下载失败」是**两种**成因。

    🔴 原先它是裸 ``ValueError``,和网络故障一起被记成 ``download_failed``,
    于是用户收到「请重新发送」—— 同一个文件重发**必然再次失败**。
    ⭐ 又一次「补错的建议比不补更坏」。继承 ``ValueError`` 保持向后兼容
    （既有 ``except ValueError`` 的调用点行为不变）。
    """


def _mixed_items(container: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """把 ``mixed`` 容器拆成子项列表。返回 ``(items, malformed)``。

    🔴 官方 ``MixedMessage`` 要求 ``mixed: MixedContent`` 且 ``msg_item``
    是数组 ⇒ 这两层**任一层**形状不对就是载荷畸形,⛔ 不是「一条空消息」。
    原先两处都写成 ``if isinstance(...) else {}`` / ``else []`` 把畸形
    **归一成空**,于是纯附件的畸形 mixed 消息静默消失。
    ⚠️ 我上一轮只补了**子项**畸形,漏了**外层容器** —— 兄弟调用点没跟上。

    ⛔ 合法的空数组(``msg_item: []``)⛔ 不算畸形:形状是对的。
    """
    if not isinstance(container, dict):
        return [], True
    raw = container.get("msg_item")
    if not isinstance(raw, list):
        return [], True
    # 🔴 元素层同形:``msg_item`` 是 list 但里面混了非 dict 时,原先直接
    # **过滤掉**并返回 ``malformed=False`` ⇒ 纯畸形的 mixed 消息仍然静默消失。
    # ⭐ 我上一轮修了「容器层」就以为修完了 —— 容器 / 元素是**两层**,
    #   这正是同一个缺陷的第三次「兄弟点没跟上」。
    # ⛔ 有效项照常保留(混排里合法的那半不该被连坐丢掉)。
    items = [i for i in raw if isinstance(i, dict)]
    return items, len(items) != len(raw)


_WECOM_KIND_LABEL: Dict[str, str] = {
    "image": "图片",
    "file": "文件",
    "video": "视频",
    "voice": "语音",
}
#: 官方 AI Bot 回调的 msgtype **全集**（``MessageType`` 枚举，六种）。
#  oracle 是仓外协议事实，⛔ 不从实现导出。
_WECOM_OFFICIAL_MSGTYPES: frozenset[str] = frozenset({
    "text", "image", "mixed", "voice", "file", "video",
})

#: 官方枚举之外、但本仓**确实在处理**的类型。⭐ 每一项都必须给出代码依据 ——
#  ⛔ 不许再往这里塞「印象里有」的名字。
#
#    appmsg —— WeCom AI Bot 的 PDF/Word/Excel 附件走这个 msgtype，
#              见 `_extract_text` 与 `_extract_media` 两处真实分支。
#    event  —— 回调侧的事件消息，见
#              `callback_adapter.py:370/374/381` 三处真实分支。
#
#  🔴 曾经这里还有一个 ``stream``：**全仓零处理逻辑**，官方枚举里也没有 ——
#  是我凭印象加进去的（`git log -S` 指向 028c9c187b，就是我那个 commit）。
#  ⭐ 与把 ``voice`` 并进可下载集同形：**为一个不存在的东西写了登记**。
#  ⇒ 已删。它若真出现，会走「未处理的 WeCom msgtype」告警 —— 那正是我们想要的。
_WECOM_LOCAL_KNOWN_MSGTYPES: frozenset[str] = frozenset({"appmsg", "event"})

_WECOM_KNOWN_MSGTYPES: frozenset[str] = (
    _WECOM_OFFICIAL_MSGTYPES | _WECOM_LOCAL_KNOWN_MSGTYPES
)

#: 兼容别名：本模块内曾用 `_WECOM_MEDIA_MSGTYPES` 表示"要下载的类型"。
#  ⛔ 不要用它做新判断 —— 名字里的 "MEDIA" 会诱导把 voice 也算进来。
_WECOM_MEDIA_MSGTYPES = _WECOM_DOWNLOADABLE_MSGTYPES

DEFAULT_WS_URL = "wss://openws.work.weixin.qq.com"

APP_CMD_SUBSCRIBE = "aibot_subscribe"
APP_CMD_CALLBACK = "aibot_msg_callback"
APP_CMD_LEGACY_CALLBACK = "aibot_callback"
APP_CMD_EVENT_CALLBACK = "aibot_event_callback"
APP_CMD_SEND = "aibot_send_msg"
APP_CMD_RESPONSE = "aibot_respond_msg"
APP_CMD_PING = "ping"
APP_CMD_UPLOAD_MEDIA_INIT = "aibot_upload_media_init"
APP_CMD_UPLOAD_MEDIA_CHUNK = "aibot_upload_media_chunk"
APP_CMD_UPLOAD_MEDIA_FINISH = "aibot_upload_media_finish"

CALLBACK_COMMANDS = {APP_CMD_CALLBACK, APP_CMD_LEGACY_CALLBACK}
NON_RESPONSE_COMMANDS = CALLBACK_COMMANDS | {APP_CMD_EVENT_CALLBACK}

MAX_MESSAGE_LENGTH = 4000
CONNECT_TIMEOUT_SECONDS = 20.0
REQUEST_TIMEOUT_SECONDS = 15.0
HEARTBEAT_INTERVAL_SECONDS = 30.0
RECONNECT_BACKOFF = [2, 5, 10, 30, 60]

DEDUP_MAX_SIZE = 1000

IMAGE_MAX_BYTES = 10 * 1024 * 1024
VIDEO_MAX_BYTES = 10 * 1024 * 1024
VOICE_MAX_BYTES = 2 * 1024 * 1024
FILE_MAX_BYTES = 20 * 1024 * 1024
ABSOLUTE_MAX_BYTES = FILE_MAX_BYTES
UPLOAD_CHUNK_SIZE = 512 * 1024
MAX_UPLOAD_CHUNKS = 100
VOICE_SUPPORTED_MIMES = {"audio/amr"}


def check_wecom_requirements() -> bool:
    """Check if WeCom runtime dependencies are available."""
    return AIOHTTP_AVAILABLE and HTTPX_AVAILABLE


def _coerce_list(value: Any) -> List[str]:
    """Coerce config values into a trimmed string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_entry(raw: str) -> str:
    """Normalize allowlist entries such as ``wecom:user:foo``."""
    value = str(raw).strip()
    value = re.sub(r"^wecom:", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^(user|group):", "", value, flags=re.IGNORECASE)
    return value.strip()


def _entry_matches(entries: List[str], target: str) -> bool:
    """Case-insensitive allowlist match with ``*`` support."""
    normalized_target = str(target).strip().lower()
    for entry in entries:
        normalized = _normalize_entry(entry).lower()
        if normalized == "*" or normalized == normalized_target:
            return True
    return False


class WeComAdapter(BasePlatformAdapter):
    """WeCom AI Bot adapter backed by a persistent WebSocket connection."""

    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    SUPPORTS_MESSAGE_EDITING = False
    # Threshold for detecting WeCom client-side message splits.
    # When a chunk is near the 4000-char limit, a continuation is almost certain.
    _SPLIT_THRESHOLD = 3900

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WECOM)

        extra = config.extra or {}
        self._bot_id = str(extra.get("bot_id") or os.getenv("WECOM_BOT_ID", "")).strip()
        self._secret = str(extra.get("secret") or _get_scoped_secret("WECOM_SECRET", "")).strip()
        self._ws_url = str(
            extra.get("websocket_url")
            or extra.get("websocketUrl")
            or os.getenv("WECOM_WEBSOCKET_URL", DEFAULT_WS_URL)
        ).strip() or DEFAULT_WS_URL

        self._dm_policy = str(extra.get("dm_policy") or os.getenv("WECOM_DM_POLICY", "pairing")).strip().lower()
        # dm_policy already honors WECOM_DM_POLICY, so the allowlist must honor
        # WECOM_ALLOWED_USERS too. Without the env fallback an env-only setup
        # (dm_policy=allowlist via env, no config extra) runs with an empty
        # allowlist and drops every authorized DM at intake.
        self._allow_from = _coerce_list(
            extra.get("allow_from")
            or extra.get("allowFrom")
            or os.getenv("WECOM_ALLOWED_USERS", "")
        )

        self._group_policy = str(extra.get("group_policy") or os.getenv("WECOM_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = _coerce_list(extra.get("group_allow_from") or extra.get("groupAllowFrom"))
        self._groups = extra.get("groups") if isinstance(extra.get("groups"), dict) else {}

        self._session: Optional["aiohttp.ClientSession"] = None
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None
        self._http_client: Optional["httpx.AsyncClient"] = None
        self._listen_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._pending_responses: Dict[str, asyncio.Future] = {}
        self._dedup = MessageDeduplicator(max_size=DEDUP_MAX_SIZE)
        self._reply_req_ids: Dict[str, str] = {}

        # Text batching: merge rapid successive messages (Telegram-style).
        # WeCom clients split long messages around 4000 chars.
        self._text_batch_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_DELAY_SECONDS", 0.6)
        self._text_batch_split_delay_seconds = env_float("HERMES_WECOM_TEXT_BATCH_SPLIT_DELAY_SECONDS", 2.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}
        self._device_id = uuid.uuid4().hex
        self._last_chat_req_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect to the WeCom AI Bot gateway."""
        if not AIOHTTP_AVAILABLE:
            message = "WeCom startup failed: aiohttp not installed"
            self._set_fatal_error("wecom_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s. Run: pip install aiohttp", self.name, message)
            return False
        if not HTTPX_AVAILABLE:
            message = "WeCom startup failed: httpx not installed"
            self._set_fatal_error("wecom_missing_dependency", message, retryable=True)
            logger.warning("[%s] %s. Run: pip install httpx", self.name, message)
            return False
        if not self._bot_id or not self._secret:
            message = "WeCom startup failed: WECOM_BOT_ID and WECOM_SECRET are required"
            self._set_fatal_error("wecom_missing_credentials", message, retryable=True)
            logger.warning("[%s] %s", self.name, message)
            return False

        try:
            # Tighter keepalive so idle CLOSE_WAIT drains promptly (#18451).
            from gateway.platforms._http_client_limits import platform_httpx_limits
            from gateway.platforms.base import _ssrf_redirect_guard
            from tools.url_safety import create_ssrf_safe_async_client

            self._http_client = create_ssrf_safe_async_client(
                timeout=30.0,
                follow_redirects=True,
                event_hooks={"response": [_ssrf_redirect_guard]},
                limits=platform_httpx_limits(),
            )
            await self._open_connection()
            self._mark_connected()
            self._listen_task = asyncio.create_task(self._listen_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            logger.info("[%s] Connected to %s", self.name, self._ws_url)
            return True
        except Exception as exc:
            message = f"WeCom startup failed: {safe_exc(exc)}"
            self._set_fatal_error("wecom_connect_error", message, retryable=True)
            logger.error("[%s] Failed to connect:\n%s", self.name, safe_traceback(exc))
            await self._cleanup_ws()
            if self._http_client:
                await self._http_client.aclose()
                self._http_client = None
            return False

    async def disconnect(self) -> None:
        """Disconnect from WeCom."""
        self._running = False
        self._mark_disconnected()

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        self._fail_pending_responses(RuntimeError("WeCom adapter disconnected"))
        await self._cleanup_ws()

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        self._dedup.clear()
        logger.info("[%s] Disconnected", self.name)

    async def _cleanup_ws(self) -> None:
        """Close the live websocket/session, if any."""
        if self._ws and not self._ws.closed:
            await self._ws.close()
        self._ws = None

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    async def _open_connection(self) -> None:
        """Open and authenticate a websocket connection."""
        await self._cleanup_ws()
        self._session = aiohttp.ClientSession(trust_env=True)
        self._ws = await self._session.ws_connect(
            self._ws_url,
            heartbeat=HEARTBEAT_INTERVAL_SECONDS * 2,
            timeout=CONNECT_TIMEOUT_SECONDS,
        )

        req_id = self._new_req_id("subscribe")
        await self._send_json(
            {
                "cmd": APP_CMD_SUBSCRIBE,
                "headers": {"req_id": req_id},
                "body": {
                    "bot_id": self._bot_id,
                    "secret": self._secret,
                    "device_id": self._device_id,
                },
            }
        )

        auth_payload = await self._wait_for_handshake(req_id)
        errcode = auth_payload.get("errcode", 0)
        if errcode not in {0, None}:
            errmsg = auth_payload.get("errmsg", "authentication failed")
            raise RuntimeError(f"{errmsg} (errcode={errcode})")

    async def _wait_for_handshake(self, req_id: str) -> Dict[str, Any]:
        """Wait for the subscribe acknowledgement."""
        if not self._ws:
            raise RuntimeError("WebSocket not initialized")

        deadline = asyncio.get_running_loop().time() + CONNECT_TIMEOUT_SECONDS
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for WeCom subscribe acknowledgement")

            msg = await asyncio.wait_for(self._ws.receive(), timeout=remaining)
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if not payload:
                    continue
                if payload.get("cmd") == APP_CMD_PING:
                    continue
                if self._payload_req_id(payload) == req_id:
                    return payload
                logger.debug("[%s] Ignoring pre-auth payload: %s", self.name, payload.get("cmd"))
            elif msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR}:
                raise RuntimeError("WeCom websocket closed during authentication")

    async def _listen_loop(self) -> None:
        """Read websocket events forever, reconnecting on errors."""
        backoff_idx = 0
        while self._running:
            try:
                await self._read_events()
                backoff_idx = 0
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if not self._running:
                    return
                logger.warning("[%s] WebSocket error: %s", self.name, safe_exc(exc))
                self._fail_pending_responses(RuntimeError("WeCom connection interrupted"))

                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                backoff_idx += 1
                await asyncio.sleep(delay)

                try:
                    await self._open_connection()
                    backoff_idx = 0
                    self._mark_connected()
                    logger.info("[%s] Reconnected", self.name)
                except Exception as reconnect_exc:
                    logger.warning("[%s] Reconnect failed: %s", self.name, safe_exc(reconnect_exc))

    async def _read_events(self) -> None:
        """Read websocket frames until the connection closes."""
        if not self._ws:
            raise RuntimeError("WebSocket not connected")

        while self._running and self._ws and not self._ws.closed:
            msg = await self._ws.receive()
            if msg.type == aiohttp.WSMsgType.TEXT:
                payload = self._parse_json(msg.data)
                if payload:
                    await self._dispatch_payload(payload)
            elif msg.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSING}:
                raise RuntimeError("WeCom websocket closed")

    async def _heartbeat_loop(self) -> None:
        """Send lightweight application-level pings."""
        try:
            while self._running:
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                if not self._ws or self._ws.closed:
                    continue
                try:
                    await self._send_json(
                        {
                            "cmd": APP_CMD_PING,
                            "headers": {"req_id": self._new_req_id("ping")},
                            "body": {},
                        }
                    )
                except Exception as exc:
                    logger.debug("[%s] Heartbeat send failed: %s", self.name, safe_exc(exc))
        except asyncio.CancelledError:
            pass

    async def _dispatch_payload(self, payload: Dict[str, Any]) -> None:
        """Route inbound websocket payloads."""
        req_id = self._payload_req_id(payload)
        cmd = str(payload.get("cmd") or "")

        if req_id and req_id in self._pending_responses and cmd not in NON_RESPONSE_COMMANDS:
            future = self._pending_responses.get(req_id)
            if future and not future.done():
                future.set_result(payload)
            return

        if cmd in CALLBACK_COMMANDS:
            await self._on_message(payload)
            return
        if cmd in {APP_CMD_PING, APP_CMD_EVENT_CALLBACK}:
            return

        logger.debug("[%s] Ignoring websocket payload: %s", self.name, cmd or payload)

    def _fail_pending_responses(self, exc: Exception) -> None:
        """Fail all outstanding request futures."""
        for req_id, future in list(self._pending_responses.items()):
            if not future.done():
                future.set_exception(exc)
            self._pending_responses.pop(req_id, None)

    async def _send_json(self, payload: Dict[str, Any]) -> None:
        """Send a raw JSON frame over the active websocket."""
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")
        await self._ws.send_json(payload)

    async def _send_request(self, cmd: str, body: Dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
        """Send a JSON request and await the correlated response."""
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

        req_id = self._new_req_id(cmd)
        future = asyncio.get_running_loop().create_future()
        self._pending_responses[req_id] = future
        try:
            await self._send_json({"cmd": cmd, "headers": {"req_id": req_id}, "body": body})
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        finally:
            self._pending_responses.pop(req_id, None)

    async def _send_reply_request(
        self,
        reply_req_id: str,
        body: Dict[str, Any],
        cmd: str = APP_CMD_RESPONSE,
        timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> Dict[str, Any]:
        """Send a reply frame correlated to an inbound callback req_id."""
        if not self._ws or self._ws.closed:
            raise RuntimeError("WeCom websocket is not connected")

        normalized_req_id = str(reply_req_id or "").strip()
        if not normalized_req_id:
            raise ValueError("reply_req_id is required")

        future = asyncio.get_running_loop().create_future()
        self._pending_responses[normalized_req_id] = future
        try:
            await self._send_json(
                {"cmd": cmd, "headers": {"req_id": normalized_req_id}, "body": body}
            )
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        finally:
            self._pending_responses.pop(normalized_req_id, None)

    @staticmethod
    def _new_req_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    @staticmethod
    def _payload_req_id(payload: Dict[str, Any]) -> str:
        headers = payload.get("headers")
        if isinstance(headers, dict):
            return str(headers.get("req_id") or "")
        return ""

    @staticmethod
    def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except Exception:
            logger.debug("Failed to parse WeCom payload: %r", raw)
            return None
        return payload if isinstance(payload, dict) else None

    # ------------------------------------------------------------------
    # Inbound message parsing
    # ------------------------------------------------------------------

    async def _on_message(self, payload: Dict[str, Any]) -> None:
        """Process an inbound WeCom message callback event."""
        body = payload.get("body")
        if not isinstance(body, dict):
            return

        msg_id = str(body.get("msgid") or self._payload_req_id(payload) or uuid.uuid4().hex)
        if self._dedup.is_duplicate(msg_id):
            logger.debug("[%s] Duplicate message %s ignored", self.name, msg_id)
            return
        self._remember_reply_req_id(msg_id, self._payload_req_id(payload))

        sender = body.get("from") if isinstance(body.get("from"), dict) else {}
        sender_id = str(sender.get("userid") or "").strip()
        chat_id = str(body.get("chatid") or sender_id).strip()
        if not chat_id:
            logger.debug("[%s] Missing chat id, skipping message", self.name)
            return

        is_group = str(body.get("chattype") or "").lower() == "group"
        if is_group:
            if not self._is_group_allowed(chat_id, sender_id):
                logger.debug("[%s] Group %s / sender %s blocked by policy", self.name, chat_id, sender_id)
                return
        elif not self._is_dm_intake_allowed(sender_id):
            logger.debug("[%s] DM sender %s blocked by policy", self.name, sender_id)
            return

        # Cache the inbound req_id after policy checks so proactive sends to
        # this chat can fall back to APP_CMD_RESPONSE (required for groups —
        # WeCom AI Bots cannot initiate APP_CMD_SEND in group chats).
        self._remember_chat_req_id(chat_id, self._payload_req_id(payload))

        text, reply_text = self._extract_text(body)
        # Strip leading @mention in group chats so slash commands like
        # "@BotName /approve" are correctly recognized as "/approve".
        # Mirrors what the Telegram adapter does (re.sub @botname).
        if is_group and text:
            text = re.sub(r"^@\S+\s*", "", text).strip()
        media_urls, media_types, media_failures = await self._extract_media(body)
        message_type = self._derive_message_type(body, text, media_types)
        has_reply_context = bool(reply_text and (text or media_urls))

        if not text and reply_text and not media_urls:
            text = reply_text

        if not text and not media_urls:
            if media_failures:
                # 🔴 有附件、却一个都没取到 —— ⛔ 不许当成空消息静默丢弃。
                # 原先这里直接 return，handle_message 从不被调用,
                # 用户看到的就是「发了图没反应」(族 A 现场)。
                await self._reply_media_intake_failed(chat_id, media_failures)
                return
            # ⚠️ 真的空消息(没有任何媒体引用)⇒ 跳过,行为逐字不变。
            logger.debug("[%s] Empty WeCom message skipped", self.name)
            return

        # 有正文、但部分/全部附件没取到 ⇒ ⛔ 不许静默:Agent 必须知道用户发过
        # 附件,否则它会答非所问("你说的图片我没看到"都说不出来)。
        # ⚠️ ⛔ 不单独给用户发消息 —— 正文已经在处理中,再发一条就是刷屏。
        #
        # 🔴 **层级**:这条提示走 ``channel_prompt``,⛔ 不许拼进 ``text``。
        # 我上一版拼进了 text —— 那等于**把系统生成的内容伪装成用户原话**:
        #   · 它会进命令解析（带附件的 ``/command`` 会把提示当成参数）
        #   · 它会进文本批处理的合并结果
        #   · 它会被**持久化进对话历史**，以后每一轮都带着它
        # ⭐ 仓内早有正确层级:``MessageEvent.channel_prompt``
        #   （``gateway/platforms/base.py:2241`` 注释写明「Applied at API call
        #   time and **never persisted** to transcript history」）。
        #   ⇒ 照抄它，⛔ 不自造第二套。
        _media_prompt = (
            self._media_failure_note(media_failures) if media_failures else None
        )

        source = self.build_source(
            chat_id=chat_id,
            chat_type="group" if is_group else "dm",
            user_id=sender_id or None,
            user_name=sender_id or None,
        )

        event = MessageEvent(
            text=text,
            channel_prompt=_media_prompt,
            message_type=message_type,
            source=source,
            raw_message=payload,
            message_id=msg_id,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=f"quote:{msg_id}" if has_reply_context else None,
            reply_to_text=reply_text if has_reply_context else None,
            timestamp=datetime.now(tz=timezone.utc),
        )

        # Only batch plain text messages — commands, media, etc. dispatch
        # immediately since they won't be split by the WeCom client.
        if message_type == MessageType.TEXT and self._text_batch_delay_seconds > 0:
            self._enqueue_text_event(event)
        else:
            await self.handle_message(event)

    # ------------------------------------------------------------------
    # Text message aggregation (handles WeCom client-side splits)
    # ------------------------------------------------------------------

    def _text_batch_key(self, event: MessageEvent) -> str:
        """Session-scoped key for text message batching."""
        from gateway.session import build_session_key
        return build_session_key(
            event.source,
            group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
            thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            profile=event.source.profile,
        )

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Buffer a text event and reset the flush timer.

        When WeCom splits a long user message at 4000 chars, the chunks
        arrive within a few hundred milliseconds.  This merges them into
        a single event before dispatching.
        """
        key = self._text_batch_key(event)
        existing = self._pending_text_batches.get(key)
        chunk_len = len(event.text or "")
        if existing is None:
            event._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            self._pending_text_batches[key] = event
        else:
            if event.text:
                existing.text = f"{existing.text}\n{event.text}" if existing.text else event.text
            existing._last_chunk_len = chunk_len  # type: ignore[attr-defined]
            # Merge any media that might be attached
            if event.media_urls:
                existing.media_urls.extend(event.media_urls)
                existing.media_types.extend(event.media_types)
            # 🔴 ``channel_prompt`` 必须一起并 —— ⛔ 不许只并 text/media。
            # 这是「半条链」:我在上面**新造**了 per-event 的 channel_prompt
            # (附件取不到的提示),但合并分支只搬 text 和 media ⇒ 用户先发一条
            # 纯文本、再发「正文+失败附件」时,后一条的提示被静默丢弃,
            # Agent 又不知道有附件 —— 缺陷原样复活在批处理路径上。
            #
            # ⚠️ 这条**只对本 adapter 成立**:其余 7 个 batcher 的 channel_prompt
            # 全部来自 ``resolve_channel_prompt(chat_id)`` = 每会话常量,同 key
            # 下每个 event 值相同,丢了也无影响 ⇒ ⛔ 不去改它们(越界扩散)。
            #
            # ⭐ 合并语义直接复用 ``_merge_caption``(base.py:5311):它做的正是
            # 「去重 + \n\n 拼接」,与这里需要的**逐字相同** ⇒ ⛔ 不自造第二套。
            if event.channel_prompt:
                existing.channel_prompt = self._merge_caption(
                    existing.channel_prompt, event.channel_prompt)

        # Cancel any pending flush and restart the timer
        prior_task = self._pending_text_batch_tasks.get(key)
        if prior_task and not prior_task.done():
            prior_task.cancel()
        self._pending_text_batch_tasks[key] = asyncio.create_task(
            self._flush_text_batch(key)
        )

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for the quiet period then dispatch the aggregated text.

        Uses a longer delay when the latest chunk is near WeCom's 4000-char
        split point, since a continuation chunk is almost certain.
        """
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            # Guard against the cancel-delivery race: when the sleep timer
            # fires just before cancel() is called, CPython sets
            # Task._must_cancel but cannot cancel the already-done sleep
            # future, so CancelledError is delivered at the *next* await
            # (handle_message) rather than here.  By that point this task
            # has already popped the merged event, so the superseding task
            # sees an empty batch and silently drops the message.
            # This check is synchronous — no await between the sleep and
            # the pop — so no other coroutine can modify the task registry
            # in between.
            if self._pending_text_batch_tasks.get(key) is not current_task:
                return
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            logger.info(
                "[WeCom] Flushing text batch %s (%d chars)",
                key, len(event.text or ""),
            )
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    @staticmethod
    def _extract_text(body: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        """Extract plain text and quoted text from a callback payload."""
        text_parts: List[str] = []
        reply_text: Optional[str] = None
        msgtype = str(body.get("msgtype") or "").lower()

        def _mixed_text(container: Any) -> List[str]:
            out: List[str] = []
            for item in _mixed_items(container)[0]:
                if str(item.get("msgtype") or "").lower() != "text":
                    continue
                _raw_text = item.get("text")
                text_block = _raw_text if isinstance(_raw_text, dict) else {}
                content = str(text_block.get("content") or "").strip()
                if content:
                    out.append(content)
            return out

        if msgtype == "mixed":
            text_parts.extend(_mixed_text(body.get("mixed")))
        else:
            text_block = body.get("text") if isinstance(body.get("text"), dict) else {}
            content = str(text_block.get("content") or "").strip()
            if content:
                text_parts.append(content)

            if msgtype == "voice":
                voice_block = body.get("voice") if isinstance(body.get("voice"), dict) else {}
                voice_text = str(voice_block.get("content") or "").strip()
                if voice_text:
                    text_parts.append(voice_text)

            # Extract appmsg title (filename) for WeCom AI Bot attachments
            if msgtype == "appmsg":
                appmsg = body.get("appmsg") if isinstance(body.get("appmsg"), dict) else {}
                title = str(appmsg.get("title") or "").strip()
                if title:
                    text_parts.append(title)

        quote = body.get("quote") if isinstance(body.get("quote"), dict) else {}
        quote_type = str(quote.get("msgtype") or "").lower()
        if quote_type == "text":
            quote_text = quote.get("text") if isinstance(quote.get("text"), dict) else {}
            reply_text = str(quote_text.get("content") or "").strip() or None
        elif quote_type == "voice":
            quote_voice = quote.get("voice") if isinstance(quote.get("voice"), dict) else {}
            reply_text = str(quote_voice.get("content") or "").strip() or None
        elif quote_type == "mixed":
            # 🔴 官方 ``QuoteContent`` 允许 ``mixed``,而原先这里只认 text/voice
            # ⇒ 用户引用一条图文混排消息再问「看这张图」时,Agent **既拿不到
            # 引用的文字、也拿不到引用的图片**,只能答非所问。
            # ⭐ 与下面 ``_extract_media`` 的 quoted 分支共用同一个 walker,
            #   ⛔ 不各写一套(两套就会漂移)。
            reply_text = "\n".join(_mixed_text(quote.get("mixed"))) or None

        return "\n".join(part for part in text_parts if part).strip(), reply_text

    async def _extract_media(
        self, body: Dict[str, Any]
    ) -> Tuple[List[str], List[str], List[MediaFailure]]:
        """取入站媒体到本地缓存。返回 ``(paths, types, failures)``。

        🔴 第三个返回值是本轮新增的（RH 复审 P1-3）。原先只返回前两个，于是
        ``_on_message`` 无法区分这两种**完全不同**的情况：

          * 用户发的就是一条空消息 ⇒ 跳过是对的
          * 用户发了图/文件，但**一个都没取到** ⇒ 整条消息被静默丢弃，
            ``handle_message`` 从不被调用（实测 ``handle_message_calls=0``），
            用户看到的是「发了图没反应」——这正是族 A 的现场。

        ⭐ 空列表有二义时，就必须把「有没有尝试过」单独带出来 ——
        与 MCP 那边 ``(observable, pids)`` 是同一个形状的问题。
        """
        media_paths: List[str] = []
        media_types: List[str] = []
        failures: List[MediaFailure] = []
        refs: List[Tuple[str, Dict[str, Any]]] = []
        msgtype = str(body.get("msgtype") or "").lower()

        def _walk_mixed(container: Any) -> None:
            """扫一个 mixed 容器的子项;外层或子项畸形都要留痕。"""
            items, malformed = _mixed_items(container)
            if malformed:
                # 🔴 ``mixed`` 不是 dict / ``msg_item`` 不是数组 / 数组里混了
                # 非 dict 元素。原先被归一成空集合 ⇒ 畸形 mixed 静默消失。
                failures.append(MediaFailure("mixed", "payload_malformed"))
            # ⚠️ ⛔ 这里**不 return**:容器层畸形时 ``items`` 本来就是空的,
            # 而元素层畸形时混排里**合法的那半仍要处理** —— 我上一版在这里
            # 直接 return,等于让一个坏元素把整条消息里的好图片一起丢掉。
            # ⭐ 「修一个缺陷时别弄坏原来对的东西」,作用域要刚好。
            for item in items:
                item_type = str(item.get("msgtype") or "").lower()
                # 官方把混排子项限定为 `'text' | 'image'`（message.ts），
                # ⇒ 这里能下载的只有 image。⛔ 别照抄顶层那套类型集：
                # 我上一版把 file/voice/video 也收进来，全是**协议里不存在的
                # 分支**，还顺手让测试构造了假 payload 去喂它。
                if item_type == "image" and isinstance(item.get("image"), dict):
                    refs.append(("image", item["image"]))
                elif item_type == "image":
                    # 🔴 P2-3:类型说是 image、``image`` 对象却不是 dict
                    # (缺失 / null / 被上游改成字符串)。原先这里**什么都不做**
                    # ⇒ refs 为空 ⇒ 整条消息被当成真空消息静默丢弃。
                    failures.append(MediaFailure("image", "payload_malformed"))
                elif item_type not in _WECOM_MIXED_ITEM_MSGTYPES:
                    logger.warning(
                        "[%s] mixed 消息里出现未知 msg_item 类型: %s（本条已跳过）",
                        self.name, item_type or "<empty>",
                    )

        if msgtype == "mixed":
            _walk_mixed(body.get("mixed"))
        else:
            if isinstance(body.get("image"), dict):
                refs.append(("image", body["image"]))
            elif msgtype == "image":
                failures.append(MediaFailure("image", "payload_malformed"))
            # file / video —— 官方回调各有独立顶层对象，与 image 同构
            # （`url` 五分钟有效 + `aeskey`）。⛔ 原先只有 file，video 整类被静默丢弃。
            #
            # 🔴 ⛔ voice 不在这里：`VoiceContent` 只有 `content: string`
            #    （语音转成的文本），**没有 url / aeskey / media_id**。
            #    把它并进来会让每条合法语音都走进下载分支、然后误报
            #    `no_media_reference`。文字由 `_extract_text` 取。
            # ⭐ 由集合驱动，⛔ 不写字面量元组：写死一份就有了第二个真相源，
            #    集合改了这里不跟着改（漂移），而且**逆改集合驱动不到这条分支**
            #    —— 一个无法被驱动的分支 = 一个无法被验证的承诺。
            #    image 在上面单独处理（它同时出现在 mixed 子项里），故减去。
            for _mt in sorted(_WECOM_DOWNLOADABLE_MSGTYPES - {"image"}):
                if msgtype != _mt:
                    continue
                if isinstance(body.get(_mt), dict):
                    refs.append((_mt, body[_mt]))
                else:
                    # 🔴 P2-3 同形:``msgtype=file`` / ``video`` 但顶层对象
                    # 形状不对。协议里这个对象是**必填**的 ⇒ 缺了就是畸形,
                    # ⛔ 不是「用户发了条空消息」。
                    failures.append(MediaFailure(_mt, "payload_malformed"))
            # Handle appmsg (WeCom AI Bot attachments with PDF/Word/Excel)
            if msgtype == "appmsg" and isinstance(body.get("appmsg"), dict):
                appmsg = body["appmsg"]
                if isinstance(appmsg.get("file"), dict):
                    refs.append(("file", appmsg["file"]))
                elif isinstance(appmsg.get("image"), dict):
                    refs.append(("image", appmsg["image"]))
            elif msgtype and msgtype not in _WECOM_KNOWN_MSGTYPES:
                # ⭐ 闭集兜底：判据不是「我列到的形状」，而是「不管它是什么，
                # 没被任何分支接住就必须留痕」。⛔ 不许再有"没写到的就静默丢"。
                logger.warning(
                    "[%s] 未处理的 WeCom msgtype: %s（可能有媒体未被取到）",
                    self.name, msgtype,
                )

        quote = body.get("quote") if isinstance(body.get("quote"), dict) else {}
        quote_type = str(quote.get("msgtype") or "").lower()
        if quote_type in {"image", "file"}:
            if isinstance(quote.get(quote_type), dict):
                refs.append((quote_type, quote[quote_type]))
            else:
                failures.append(MediaFailure(quote_type, "payload_malformed"))
        elif quote_type == "mixed":
            # 🔴 官方 ``QuoteContent`` 允许 ``mixed`` —— 原先这里只认 image/file,
            # 引用一条图文混排消息时那张图**完全不可见**。
            # ⭐ 复用同一个 walker,⛔ 不为「引用」再写一套(两套必漂移)。
            _walk_mixed(quote.get("mixed"))

        for kind, ref in refs:
            cached = await self._cache_media(kind, ref, failures=failures)
            if cached:
                path, content_type = cached
                media_paths.append(path)
                media_types.append(content_type)
            # ⛔ 这里**不再** append —— 记账已收敛进 ``_media_intake_failed``。
            # 两处都记就会重复计数(「2 个附件」其实只有 1 个);
            # ⭐ 而它能安全去掉的前提是「_cache_media 的每一条 return None
            #   都先经过 _media_intake_failed」—— 这条由 test_wecom_media_
            #   failure_reasons.py::test_every_failure_path_records_a_reason
            #   驱动全部七种成因钉死,⛔ 不靠读代码保证。

        return media_paths, media_types, failures

    @staticmethod
    def _media_failure_note(failures: List[MediaFailure]) -> str:
        """给 **Agent** 看的一行提示（⛔ 不是给用户的文案）。

        Agent 需要知道「用户发过附件但系统没取到」，否则它连
        「你发的图我没收到」都说不出来，只会答非所问。

        ⚠️ 这条给 Agent 的提示**带 reason 原文**是对的:它要据此决定说什么
        （链接过期 ⇒ 让用户重发;存储满 ⇒ 别让用户白重发）。
        ⛔ 用户侧那条**不许**带 reason 码 —— 见 ``_reply_media_intake_failed``。
        """
        kinds = "、".join(
            f"{_WECOM_KIND_LABEL.get(f.kind, f.kind)}/{f.reason}"
            for f in sorted(set(failures))
        )
        return f"[系统提示：用户发送了 {len(failures)} 个附件（{kinds}），但未能取到内容]"

    @staticmethod
    def _media_failure_reply_text(failures: List[MediaFailure]) -> str:
        """把失败清单翻译成给**用户**的一句话:是什么 + 现在怎么办。

        🔴 上一版**只有一句写死的**「请稍后重新发送；如果是较大的文件，
        可以先压缩后再试」——对链接过期、解密失败、磁盘写满全是错误指引。
        ⭐ 「补错的建议比不补更坏」:它让用户去做一件确定无效的事,
          然后以为是自己的问题。
        ⛔ 不堆文案:名字一行、建议去重后最多两句,⛔ 无成功态/进度旁白。
        """
        kinds = "、".join(sorted({
            _WECOM_KIND_LABEL.get(f.kind, "附件") for f in failures}))
        advice: List[str] = []
        for f in failures:
            tip = _MEDIA_FAILURE_ADVICE.get(f.reason, _MEDIA_FAILURE_ADVICE_FALLBACK)
            if tip not in advice:
                advice.append(tip)
        return f"未能读取你发送的{kinds}。" + "".join(advice[:2])

    async def _reply_media_intake_failed(
        self, chat_id: str, failures: List[MediaFailure]
    ) -> None:
        """纯附件消息一个都没取到 ⇒ 给用户一条**可行动**的回复。

        ⭐ 判据「不写它用户会不会做错事」= **会**：他看不到任何反应，
        会以为机器人挂了或自己没发出去，然后反复重发。
        ⛔ 但一个字都不含内部路径 / 原始错误 / reason 码 —— 那些在日志里
        （``_media_intake_failed`` 已记 reason / url_host / 错误类型）。
        """
        try:
            res = await self.send(
                chat_id,
                self._media_failure_reply_text(failures),
                reply_to=None,
                metadata=None,
            )
        except Exception as exc:
            logger.error(
                "[%s] 连「附件读取失败」的回复都没发出去 (chat_id=%s): %s",
                self.name, chat_id, safe_exc(exc),
            )
            return
        # ⭐ 判据是 SendResult.success,⛔ 不是「没抛异常」——
        # 这条判据我今晚已经在三个地方犯过,写完就地再过一遍。
        if getattr(res, "success", True) is False:
            logger.error(
                "[%s] 「附件读取失败」的回复被平台拒收 (chat_id=%s): %s",
                self.name, chat_id, getattr(res, "error", None) or "unknown",
            )

    def _media_intake_failed(
        self,
        kind: str,
        reason: str,
        media: Dict[str, Any],
        exc: Optional[BaseException] = None,
        failures: Optional[List[MediaFailure]] = None,
    ) -> None:
        """入站媒体没拿到 —— 必须留下【可区分、可定位】的痕迹。

        ⛔ 原先这些分支只打 logger.debug（默认不输出）甚至什么都不打，于是用户
        看到的是「发了图没反应」，而日志里空空如也 —— 既是缺陷本身，也是它一直
        排查不出来的原因。⇒ 升到 warning，并让六种失败彼此可区分。

        ⛔ 不许打印 url 全文 / aeskey / base64 内容：企微媒体 url 带鉴权参数，
        原先 ``logger.debug("... from %s", url)`` 是会泄漏的。这里只记 host 与
        长度这类足以定位、又不含凭据的摘要。

        🔴 ⛔ 用 ``.hostname`` 而不是 ``.netloc``：``netloc`` **包含 userinfo**，
           ``https://user:secret@host/x`` 的 netloc 就是 ``user:secret@host``
           —— 我上一版正是用 netloc 去"脱敏"，等于换个地方继续泄漏凭据。
           ⭐ 修一个泄漏点时用的工具本身也要过一遍同一条判据。
        """
        url = str(media.get("url") or "")
        host = ""
        if url:
            try:
                parsed = urlparse(url)
                # hostname 已剥掉 userinfo 且小写化；端口单独取，⛔ 不回落 netloc。
                host = parsed.hostname or ""
                if host and parsed.port:
                    host = f"{host}:{parsed.port}"
            except Exception:
                host = "<unparsable>"
        logger.warning(
            "[%s] 入站媒体获取失败: kind=%s reason=%s has_url=%s url_host=%s "
            "has_base64=%s has_aeskey=%s err=%s",
            self.name, kind, reason, bool(url), host or "-",
            bool(media.get("base64")), bool(media.get("aeskey")),
            type(exc).__name__ if exc is not None else "-",
        )
        # ⭐ 记账收敛到这**一个**出口:七条失败分支已经全部经过这里,
        # 于是「日志里有、但调用方看不见」这种半条链结构上不可能再出现。
        # ⛔ 不在调用方各自 append —— 那就有第二个真相源,漏一处就静默丢。
        if failures is not None:
            failures.append(MediaFailure(kind=kind, reason=reason))

    def _write_cache_or_report(
        self,
        kind: str,
        media: Dict[str, Any],
        writer,
        *args,
        failures: Optional[List[MediaFailure]] = None,
    ) -> Optional[str]:
        """落盘那一步的统一出口 —— 写失败必须变成**消息级**失败。

        🔴 原先四个写盘点(base64/url × image/document)全都让 ``OSError`` 裸奔:
        磁盘满、只读挂载、权限不足时异常一路冒到 ``_on_message`` 再到监听循环
        ⇒ **整条连接断开**,而这条消息的 ID 已经进了 dedup ⇒ 重连后也不会重放,
        用户那条消息**永久消失**。一次磁盘故障变成一次静默丢消息。

        ⛔ 只捕 ``OSError``:磁盘/权限/路径这类**环境**故障才降级成消息级失败。
        编程错误(TypeError/AttributeError)照旧向上抛 —— fail fast,⛔ 不许在这
        里兜底成"看起来正常"。

        ⭐ 四个调用点收敛到这一个出口,⛔ 不在四处各写一遍 try —— 复制四份的
        结果就是下次新增第五个写盘点时漏掉它。
        """
        try:
            return writer(*args)
        except OSError as exc:
            self._media_intake_failed(
                kind, "cache_write_failed", media, exc, failures=failures)
            return None

    async def _cache_media(
        self,
        kind: str,
        media: Dict[str, Any],
        failures: Optional[List[MediaFailure]] = None,
    ) -> Optional[Tuple[str, str]]:
        """取一条入站媒体到本地缓存。失败时把 ``MediaFailure`` 记进 ``failures``。

        ⭐ ``failures`` 用**出参**而不是改返回值:返回值一改,12 个调用点
        (含 3 个 ``monkeypatch.setattr`` 打桩点)全要跟着动 —— 而这条缺陷
        只在「原因没带出来」这一点上,改动作用域应当**刚好等于**它。
        照抄 ``weixin._collect_media`` 的出参风格,⛔ 不自造第三种。
        """
        if "base64" in media and media.get("base64"):
            try:
                raw = self._decode_base64(media["base64"])
            except Exception as exc:
                self._media_intake_failed(
                    kind, "base64_decode_failed", media, exc, failures=failures)
                return None

            if kind == "image":
                ext = self._detect_image_ext(raw)
                try:
                    path = self._write_cache_or_report(
                        kind, media, cache_image_from_bytes, raw, ext, failures=failures)
                except ValueError as exc:
                    self._media_intake_failed(
                    kind, "not_an_image", media, exc, failures=failures)
                    return None
                if path is None:
                    return None
                return path, self._mime_for_ext(ext, fallback="image/jpeg")

            filename = str(media.get("filename") or media.get("name") or "wecom_file")
            path = self._write_cache_or_report(
                kind, media, cache_document_from_bytes, raw, filename, failures=failures)
            if path is None:
                return None
            return path, mimetypes.guess_type(filename)[0] or "application/octet-stream"

        url = str(media.get("url") or "").strip()
        if not url:
            # 既没有 base64 也没有 url —— 原先这里【一句日志都没有】,
            # 是六条失败分支里最不可观测的一条。
            self._media_intake_failed(
                kind, "no_media_reference", media, failures=failures)
            return None

        try:
            raw, headers = await self._download_remote_bytes(url, max_bytes=ABSOLUTE_MAX_BYTES)
        except WeComMediaTooLarge as exc:
            # 🔴 ⛔ 必须排在 download_failed 前面:超限是**确定性**失败,
            # 给「请重新发送」等于让用户做一件必然再失败的事。
            self._media_intake_failed(
                kind, "too_large", media, exc, failures=failures)
            return None
        except Exception as exc:
            # ⚠️ 企微媒体 url 仅 5 分钟有效,过期与网络故障在这里表现相同;
            # reason 统一为 download_failed,由 err 类型区分,⛔ 不臆断是哪种。
            self._media_intake_failed(
                    kind, "download_failed", media, exc, failures=failures)
            return None

        aes_key = str(media.get("aeskey") or "").strip()
        if aes_key:
            try:
                raw = self._decrypt_file_bytes(raw, aes_key)
            except Exception as exc:
                self._media_intake_failed(
                    kind, "decrypt_failed", media, exc, failures=failures)
                return None

        content_type = str(headers.get("content-type") or "").split(";", 1)[0].strip() or "application/octet-stream"
        if kind == "image":
            ext = self._guess_extension(url, content_type, fallback=self._detect_image_ext(raw))
            try:
                path = self._write_cache_or_report(
                    kind, media, cache_image_from_bytes, raw, ext, failures=failures)
            except ValueError as exc:
                self._media_intake_failed(
                    kind, "not_an_image", media, exc, failures=failures)
                return None
            if path is None:
                return None
            return path, content_type or self._mime_for_ext(ext, fallback="image/jpeg")

        filename = self._guess_filename(url, headers.get("content-disposition"), content_type)
        path = self._write_cache_or_report(
            kind, media, cache_document_from_bytes, raw, filename, failures=failures)
        if path is None:
            return None
        return path, content_type

    @staticmethod
    def _decode_base64(data: str) -> bytes:
        payload = data.split(",", 1)[-1].strip()
        return base64.b64decode(payload)

    @staticmethod
    def _detect_image_ext(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if data.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if data.startswith((b"GIF87a", b"GIF89a")):
            return ".gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return ".webp"
        return ".jpg"

    @staticmethod
    def _mime_for_ext(ext: str, fallback: str = "application/octet-stream") -> str:
        return mimetypes.types_map.get(ext.lower(), fallback)

    @staticmethod
    def _guess_extension(url: str, content_type: str, fallback: str) -> str:
        ext = mimetypes.guess_extension(content_type) if content_type else None
        if ext:
            return ext
        path_ext = Path(urlparse(url).path).suffix
        if path_ext:
            return path_ext
        return fallback

    @staticmethod
    def _guess_filename(url: str, content_disposition: Optional[str], content_type: str) -> str:
        if content_disposition:
            match = re.search(r'filename="?([^";]+)"?', content_disposition)
            if match:
                return match.group(1)

        name = Path(urlparse(url).path).name or "document"
        if "." not in name:
            ext = mimetypes.guess_extension(content_type) or ".bin"
            name = f"{name}{ext}"
        return name

    @staticmethod
    def _derive_message_type(body: Dict[str, Any], text: str, media_types: List[str]) -> MessageType:
        """Choose the normalized inbound message type."""
        if any(mtype.startswith(("application/", "text/")) for mtype in media_types):
            return MessageType.DOCUMENT
        if any(mtype.startswith("image/") for mtype in media_types):
            return MessageType.TEXT if text else MessageType.PHOTO
        if str(body.get("msgtype") or "").lower() == "voice":
            return MessageType.VOICE
        return MessageType.TEXT

    # ------------------------------------------------------------------
    # Policy helpers
    # ------------------------------------------------------------------

    @property
    def enforces_own_access_policy(self) -> bool:
        """WeCom gates DM/group access at intake via dm_policy/group_policy."""
        return True

    def _open_dm_opted_in(self) -> bool:
        if os.getenv("GATEWAY_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}:
            return True
        return os.getenv("WECOM_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}

    def _is_dm_allowed(self, sender_id: str) -> bool:
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, sender_id)
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_dm_intake_allowed(self, sender_id: str) -> bool:
        principal = str(sender_id or "").strip()
        if not principal:
            return False
        if self._dm_policy == "disabled":
            return False
        if self._dm_policy == "allowlist":
            return _entry_matches(self._allow_from, principal)
        if self._dm_policy == "pairing":
            return True
        if self._dm_policy == "open":
            return self._open_dm_opted_in()
        return False

    def _is_group_allowed(self, chat_id: str, sender_id: str) -> bool:
        if self._group_policy == "disabled":
            return False
        if self._group_policy == "pairing":
            return False
        if self._group_policy == "allowlist" and not _entry_matches(self._group_allow_from, chat_id):
            return False

        group_cfg = self._resolve_group_cfg(chat_id)
        sender_allow = _coerce_list(group_cfg.get("allow_from") or group_cfg.get("allowFrom"))
        if sender_allow:
            return _entry_matches(sender_allow, sender_id)
        return True

    def _resolve_group_cfg(self, chat_id: str) -> Dict[str, Any]:
        if not isinstance(self._groups, dict):
            return {}
        if chat_id in self._groups and isinstance(self._groups[chat_id], dict):
            return self._groups[chat_id]
        lowered = chat_id.lower()
        for key, value in self._groups.items():
            if isinstance(key, str) and key.lower() == lowered and isinstance(value, dict):
                return value
        wildcard = self._groups.get("*")
        return wildcard if isinstance(wildcard, dict) else {}

    def _remember_reply_req_id(self, message_id: str, req_id: str) -> None:
        normalized_message_id = str(message_id or "").strip()
        normalized_req_id = str(req_id or "").strip()
        if not normalized_message_id or not normalized_req_id:
            return
        self._reply_req_ids[normalized_message_id] = normalized_req_id
        while len(self._reply_req_ids) > DEDUP_MAX_SIZE:
            self._reply_req_ids.pop(next(iter(self._reply_req_ids)))

    def _remember_chat_req_id(self, chat_id: str, req_id: str) -> None:
        """Cache the most recent inbound req_id per chat.

        Used as a fallback reply target when we need to send into a group
        without an explicit ``reply_to`` — WeCom AI Bots are blocked from
        APP_CMD_SEND in groups and must use APP_CMD_RESPONSE bound to some
        prior req_id. Bounded like _reply_req_ids so long-running gateways
        don't leak memory across many chats.
        """
        normalized_chat_id = str(chat_id or "").strip()
        normalized_req_id = str(req_id or "").strip()
        if not normalized_chat_id or not normalized_req_id:
            return
        self._last_chat_req_ids[normalized_chat_id] = normalized_req_id
        while len(self._last_chat_req_ids) > DEDUP_MAX_SIZE:
            self._last_chat_req_ids.pop(next(iter(self._last_chat_req_ids)))

    def _reply_req_id_for_message(self, reply_to: Optional[str]) -> Optional[str]:
        normalized = str(reply_to or "").strip()
        if not normalized or normalized.startswith("quote:"):
            return None
        return self._reply_req_ids.get(normalized)

    # ------------------------------------------------------------------
    # Outbound messaging
    # ------------------------------------------------------------------

    @staticmethod
    def _guess_mime_type(filename: str) -> str:
        mime_type = mimetypes.guess_type(filename)[0]
        if mime_type:
            return mime_type
        if Path(filename).suffix.lower() == ".amr":
            return "audio/amr"
        return "application/octet-stream"

    @staticmethod
    def _normalize_content_type(content_type: str, filename: str) -> str:
        normalized = str(content_type or "").split(";", 1)[0].strip().lower()
        guessed = WeComAdapter._guess_mime_type(filename)
        if not normalized:
            return guessed
        if normalized in {"application/octet-stream", "text/plain"}:
            return guessed
        return normalized

    @staticmethod
    def _detect_wecom_media_type(content_type: str) -> str:
        mime_type = str(content_type or "").strip().lower()
        if mime_type.startswith("image/"):
            return "image"
        if mime_type.startswith("video/"):
            return "video"
        if mime_type.startswith("audio/") or mime_type == "application/ogg":
            return "voice"
        return "file"

    @staticmethod
    def _apply_file_size_limits(file_size: int, detected_type: str, content_type: Optional[str] = None) -> Dict[str, Any]:
        file_size_mb = file_size / (1024 * 1024)
        normalized_type = str(detected_type or "file").lower()
        normalized_content_type = str(content_type or "").strip().lower()

        if file_size > ABSOLUTE_MAX_BYTES:
            return {
                "final_type": normalized_type,
                "rejected": True,
                "reject_reason": (
                    f"文件大小 {file_size_mb:.2f}MB 超过了企业微信允许的最大限制 20MB，无法发送。"
                    "请尝试压缩文件或减小文件大小。"
                ),
                "downgraded": False,
                "downgrade_note": None,
            }

        if normalized_type == "image" and file_size > IMAGE_MAX_BYTES:
            return {
                "final_type": "file",
                "rejected": False,
                "reject_reason": None,
                "downgraded": True,
                "downgrade_note": f"图片大小 {file_size_mb:.2f}MB 超过 10MB 限制，已转为文件格式发送",
            }

        if normalized_type == "video" and file_size > VIDEO_MAX_BYTES:
            return {
                "final_type": "file",
                "rejected": False,
                "reject_reason": None,
                "downgraded": True,
                "downgrade_note": f"视频大小 {file_size_mb:.2f}MB 超过 10MB 限制，已转为文件格式发送",
            }

        if normalized_type == "voice":
            if normalized_content_type and normalized_content_type not in VOICE_SUPPORTED_MIMES:
                return {
                    "final_type": "file",
                    "rejected": False,
                    "reject_reason": None,
                    "downgraded": True,
                    "downgrade_note": (
                        f"语音格式 {normalized_content_type} 不支持，企微仅支持 AMR 格式，已转为文件格式发送"
                    ),
                }
            if file_size > VOICE_MAX_BYTES:
                return {
                    "final_type": "file",
                    "rejected": False,
                    "reject_reason": None,
                    "downgraded": True,
                    "downgrade_note": f"语音大小 {file_size_mb:.2f}MB 超过 2MB 限制，已转为文件格式发送",
                }

        return {
            "final_type": normalized_type,
            "rejected": False,
            "reject_reason": None,
            "downgraded": False,
            "downgrade_note": None,
        }

    @staticmethod
    def _response_error(response: Dict[str, Any]) -> Optional[str]:
        errcode = response.get("errcode", 0)
        if errcode in {0, None}:
            return None
        errmsg = str(response.get("errmsg") or "unknown error")
        return f"WeCom errcode {errcode}: {errmsg}"

    @classmethod
    def _raise_for_wecom_error(cls, response: Dict[str, Any], operation: str) -> None:
        error = cls._response_error(response)
        if error:
            raise RuntimeError(f"{operation} failed: {error}")

    @staticmethod
    def _decrypt_file_bytes(encrypted_data: bytes, aes_key: str) -> bytes:
        if not encrypted_data:
            raise ValueError("encrypted_data is empty")
        if not aes_key:
            raise ValueError("aes_key is required")

        # WeCom doesn't pad base64 keys; add padding if needed
        aes_key = aes_key + '=' * ((4 - len(aes_key) % 4) % 4)
        key = base64.b64decode(aes_key)
        if len(key) != 32:
            raise ValueError(f"Invalid WeCom AES key length: expected 32 bytes, got {len(key)}")

        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:  # pragma: no cover - dependency is environment-specific
            raise RuntimeError("cryptography is required for WeCom media decryption") from exc

        cipher = Cipher(algorithms.AES(key), modes.CBC(key[:16]))
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(encrypted_data) + decryptor.finalize()

        pad_len = decrypted[-1]
        if pad_len < 1 or pad_len > 32 or pad_len > len(decrypted):
            raise ValueError(f"Invalid PKCS#7 padding value: {pad_len}")
        if any(byte != pad_len for byte in decrypted[-pad_len:]):
            raise ValueError("Invalid PKCS#7 padding: padding bytes mismatch")

        return decrypted[:-pad_len]

    async def _download_remote_bytes(
        self,
        url: str,
        max_bytes: int,
    ) -> Tuple[bytes, Dict[str, str]]:
        from gateway.platforms.base import _ssrf_redirect_guard
        from tools.url_safety import create_ssrf_safe_async_client, is_safe_url

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe URL (SSRF protection): {url[:80]}")

        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx is required for WeCom media download")

        client = self._http_client or create_ssrf_safe_async_client(
            timeout=30.0,
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_guard]},
        )
        created_client = client is not self._http_client
        try:
            async with client.stream(
                "GET",
                url,
                headers={
                    "User-Agent": "HermesAgent/1.0",
                    "Accept": "*/*",
                },
            ) as response:
                response.raise_for_status()
                headers = {key.lower(): value for key, value in response.headers.items()}
                content_length = headers.get("content-length")
                if content_length and content_length.isdigit() and int(content_length) > max_bytes:
                    raise WeComMediaTooLarge(
                        f"Remote media exceeds WeCom limit: {int(content_length)} bytes > {max_bytes} bytes"
                    )

                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise WeComMediaTooLarge(
                            f"Remote media exceeds WeCom limit while downloading: {len(data)} bytes > {max_bytes} bytes"
                        )

                return bytes(data), headers
        finally:
            if created_client:
                await client.aclose()

    @staticmethod
    def _looks_like_url(media_source: str) -> bool:
        parsed = urlparse(str(media_source or ""))
        return parsed.scheme in {"http", "https"}

    async def _load_outbound_media(
        self,
        media_source: str,
        file_name: Optional[str] = None,
    ) -> Tuple[bytes, str, str]:
        source = str(media_source or "").strip()
        if not source:
            raise ValueError("media source is required")
        if re.fullmatch(r"<[^>\n]+>", source):
            raise ValueError(f"Media placeholder was not replaced with a real file path: {source}")

        parsed = urlparse(source)
        if parsed.scheme in {"http", "https"}:
            data, headers = await self._download_remote_bytes(source, max_bytes=ABSOLUTE_MAX_BYTES)
            content_disposition = headers.get("content-disposition")
            resolved_name = file_name or self._guess_filename(source, content_disposition, headers.get("content-type", ""))
            content_type = self._normalize_content_type(headers.get("content-type", ""), resolved_name)
            return data, content_type, resolved_name

        if parsed.scheme == "file":
            local_path = Path(unquote(parsed.path)).expanduser()
        else:
            local_path = Path(source).expanduser()

        if not local_path.is_absolute():
            local_path = (Path.cwd() / local_path).resolve()

        if not local_path.exists() or not local_path.is_file():
            raise FileNotFoundError(f"Media file not found: {local_path}")

        data = local_path.read_bytes()
        resolved_name = file_name or local_path.name
        content_type = self._normalize_content_type("", resolved_name)
        return data, content_type, resolved_name

    async def _prepare_outbound_media(
        self,
        media_source: str,
        file_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        data, content_type, resolved_name = await self._load_outbound_media(media_source, file_name=file_name)
        detected_type = self._detect_wecom_media_type(content_type)
        size_check = self._apply_file_size_limits(len(data), detected_type, content_type)
        return {
            "data": data,
            "content_type": content_type,
            "file_name": resolved_name,
            "detected_type": detected_type,
            **size_check,
        }

    async def _upload_media_bytes(self, data: bytes, media_type: str, filename: str) -> Dict[str, Any]:
        if not data:
            raise ValueError("Cannot upload empty media")

        total_size = len(data)
        total_chunks = (total_size + UPLOAD_CHUNK_SIZE - 1) // UPLOAD_CHUNK_SIZE
        if total_chunks > MAX_UPLOAD_CHUNKS:
            raise ValueError(
                f"File too large: {total_chunks} chunks exceeds maximum of {MAX_UPLOAD_CHUNKS} chunks"
            )

        init_response = await self._send_request(
            APP_CMD_UPLOAD_MEDIA_INIT,
            {
                "type": media_type,
                "filename": filename,
                "total_size": total_size,
                "total_chunks": total_chunks,
                "md5": hashlib.md5(data).hexdigest(),
            },
        )
        self._raise_for_wecom_error(init_response, "media upload init")

        init_body = init_response.get("body") if isinstance(init_response.get("body"), dict) else {}
        upload_id = str(init_body.get("upload_id") or "").strip()
        if not upload_id:
            raise RuntimeError(f"media upload init failed: missing upload_id in response {init_response}")

        for chunk_index, start in enumerate(range(0, total_size, UPLOAD_CHUNK_SIZE)):
            chunk = data[start : start + UPLOAD_CHUNK_SIZE]
            chunk_response = await self._send_request(
                APP_CMD_UPLOAD_MEDIA_CHUNK,
                {
                    "upload_id": upload_id,
                    # Match the official SDK implementation, which currently uses 0-based chunk indexes.
                    "chunk_index": chunk_index,
                    "base64_data": base64.b64encode(chunk).decode("ascii"),
                },
            )
            self._raise_for_wecom_error(chunk_response, f"media upload chunk {chunk_index}")

        finish_response = await self._send_request(
            APP_CMD_UPLOAD_MEDIA_FINISH,
            {"upload_id": upload_id},
        )
        self._raise_for_wecom_error(finish_response, "media upload finish")

        finish_body = finish_response.get("body") if isinstance(finish_response.get("body"), dict) else {}
        media_id = str(finish_body.get("media_id") or "").strip()
        if not media_id:
            raise RuntimeError(f"media upload finish failed: missing media_id in response {finish_response}")

        return {
            "type": str(finish_body.get("type") or media_type),
            "media_id": media_id,
            "created_at": finish_body.get("created_at"),
        }

    async def _send_media_message(self, chat_id: str, media_type: str, media_id: str) -> Dict[str, Any]:
        response = await self._send_request(
            APP_CMD_SEND,
            {
                "chatid": chat_id,
                "msgtype": media_type,
                media_type: {"media_id": media_id},
            },
        )
        self._raise_for_wecom_error(response, "send media message")
        return response

    async def _send_reply_markdown(self, reply_req_id: str, content: str) -> Dict[str, Any]:
        response = await self._send_reply_request(
            reply_req_id,
            {
                "msgtype": "markdown",
                "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]},
            },
        )
        self._raise_for_wecom_error(response, "send reply markdown")
        return response

    async def _send_reply_media_message(
        self,
        reply_req_id: str,
        media_type: str,
        media_id: str,
    ) -> Dict[str, Any]:
        response = await self._send_reply_request(
            reply_req_id,
            {
                "msgtype": media_type,
                media_type: {"media_id": media_id},
            },
        )
        self._raise_for_wecom_error(response, "send reply media message")
        return response

    async def _send_followup_markdown(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
    ) -> Optional[SendResult]:
        if not content:
            return None
        result = await self.send(chat_id=chat_id, content=content, reply_to=reply_to)
        if not result.success:
            logger.warning("[%s] Follow-up markdown send failed: %s", self.name, result.error)
        return result

    async def _send_media_source(
        self,
        chat_id: str,
        media_source: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
    ) -> SendResult:
        if not chat_id:
            return SendResult(success=False, error="chat_id is required")

        try:
            prepared = await self._prepare_outbound_media(media_source, file_name=file_name)
        except FileNotFoundError:
            return SendResult(
                success=False,
                error="WeCom media file was not found. Regenerate or re-upload it, then try again.",
            )
        except Exception as exc:
            logger.error(
                "[%s] Failed to prepare outbound media %s: %s",
                self.name,
                safe_url_for_log(media_source),
                safe_exc(exc),
            )
            return SendResult(
                success=False,
                error="WeCom could not prepare the media. Check the source and try again.",
            )

        if prepared["rejected"]:
            await self._send_followup_markdown(
                chat_id,
                f"⚠️ {prepared['reject_reason']}",
                reply_to=reply_to,
            )
            return SendResult(success=False, error=prepared["reject_reason"])

        reply_req_id = self._reply_req_id_for_message(reply_to)
        if not reply_req_id and chat_id in self._last_chat_req_ids:
            reply_req_id = self._last_chat_req_ids[chat_id]

        try:
            upload_result = await self._upload_media_bytes(
                prepared["data"],
                prepared["final_type"],
                prepared["file_name"],
            )
            if reply_req_id:
                media_response = await self._send_reply_media_message(
                    reply_req_id,
                    prepared["final_type"],
                    upload_result["media_id"],
                )
            else:
                media_response = await self._send_media_message(
                    chat_id,
                    prepared["final_type"],
                    upload_result["media_id"],
                )
        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending media to WeCom")
        except Exception as exc:
            logger.error(
                "[%s] Failed to send media %s: %s",
                self.name,
                safe_url_for_log(media_source),
                safe_exc(exc),
            )
            return SendResult(
                success=False,
                error="WeCom media delivery failed. Check the bot connection and try again.",
            )

        caption_result = None
        downgrade_result = None
        if caption:
            caption_result = await self._send_followup_markdown(
                chat_id,
                caption,
                reply_to=reply_to,
            )
        if prepared["downgraded"] and prepared["downgrade_note"]:
            downgrade_result = await self._send_followup_markdown(
                chat_id,
                f"ℹ️ {prepared['downgrade_note']}",
                reply_to=reply_to,
            )

        return SendResult(
            success=True,
            message_id=self._payload_req_id(media_response) or uuid.uuid4().hex[:12],
            raw_response={
                "upload": upload_result,
                "media": media_response,
                "caption": caption_result.raw_response if caption_result else None,
                "caption_error": caption_result.error if caption_result and not caption_result.success else None,
                "downgrade": downgrade_result.raw_response if downgrade_result else None,
                "downgrade_error": downgrade_result.error if downgrade_result and not downgrade_result.success else None,
            },
        )

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send markdown to a WeCom chat via proactive ``aibot_send_msg``."""
        del metadata

        if not chat_id:
            return SendResult(success=False, error="chat_id is required")

        try:
            reply_req_id = self._reply_req_id_for_message(reply_to)

            if not reply_req_id and chat_id in self._last_chat_req_ids:
                reply_req_id = self._last_chat_req_ids[chat_id]

            if reply_req_id:
                response = await self._send_reply_markdown(reply_req_id, content)
            else:
                response = await self._send_request(
                    APP_CMD_SEND,
                    {
                        "chatid": chat_id,
                        "msgtype": "markdown",
                        "markdown": {"content": content[:self.MAX_MESSAGE_LENGTH]},
                    },
                )
        except asyncio.TimeoutError:
            return SendResult(success=False, error="Timeout sending message to WeCom")
        except Exception as exc:
            logger.error("[%s] Send failed: %s", self.name, safe_exc(exc))
            return SendResult(success=False, error=str(exc))

        error = self._response_error(response)
        if error:
            return SendResult(success=False, error=error)

        return SendResult(
            success=True,
            message_id=self._payload_req_id(response) or uuid.uuid4().hex[:12],
            raw_response=response,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del metadata

        result = await self._send_media_source(
            chat_id=chat_id,
            media_source=image_url,
            caption=caption,
            reply_to=reply_to,
        )
        if result.success or not self._looks_like_url(image_url):
            return result

        logger.warning(
            "[%s] Falling back to text send for image URL %s: %s",
            self.name,
            safe_url_for_log(image_url),
            result.error,
        )
        fallback_text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id=chat_id, content=fallback_text, reply_to=reply_to)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=image_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=audio_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        del kwargs
        return await self._send_media_source(
            chat_id=chat_id,
            media_source=video_path,
            caption=caption,
            reply_to=reply_to,
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """WeCom does not expose typing indicators in this adapter."""
        del chat_id, metadata

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return minimal chat info."""
        return {
            "name": chat_id,
            "type": "group" if chat_id and chat_id.lower().startswith("group") else "dm",
        }


# ------------------------------------------------------------------
# QR code scan flow for obtaining bot credentials
# ------------------------------------------------------------------

_QR_GENERATE_URL = "https://work.weixin.qq.com/ai/qc/generate"
_QR_QUERY_URL = "https://work.weixin.qq.com/ai/qc/query_result"
_QR_CODE_PAGE = "https://work.weixin.qq.com/ai/qc/gen?source=hermes&scode="
_QR_POLL_INTERVAL = 3  # seconds
_QR_POLL_TIMEOUT = 300  # 5 minutes


def qr_scan_for_bot_info(
    *,
    timeout_seconds: int = _QR_POLL_TIMEOUT,
) -> Optional[Dict[str, str]]:
    """Run the WeCom QR scan flow to obtain bot_id and secret.

    Fetches a QR code from WeCom, renders it in the terminal, and polls
    until the user scans it or the timeout expires.

    Returns ``{"bot_id": ..., "secret": ...}`` on success, ``None`` on
    failure or timeout.

    Note: the ``work.weixin.qq.com/ai/qc/{generate,query_result}`` endpoints
    used here are not part of WeCom's public developer API — they back the
    admin-console web UI's bot-creation flow and may change without notice.
    The same pattern is used by the feishu/dingtalk QR setup wizards.
    """
    try:
        import urllib.request
        import urllib.parse
    except ImportError:  # pragma: no cover
        logger.error("urllib is required for WeCom QR scan")
        return None

    generate_url = f"{_QR_GENERATE_URL}?source=hermes"

    # ── Step 1: Fetch QR code ──
    print("  Connecting to WeCom...", end="", flush=True)
    try:
        req = urllib.request.Request(generate_url, headers={"User-Agent": "HermesAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("WeCom QR: failed to fetch QR code: %s", safe_exc(exc))
        print(f" failed: {exc}")
        return None

    data = raw.get("data") or {}
    scode = str(data.get("scode") or "").strip()
    auth_url = str(data.get("auth_url") or "").strip()

    if not scode or not auth_url:
        logger.error("WeCom QR: unexpected response format: %s", raw)
        print(" failed: unexpected response format")
        return None

    print(" done.")

    # ── Step 2: Render QR code in terminal ──
    print()
    qr_rendered = False
    try:
        import qrcode as _qrcode
        qr = _qrcode.QRCode()
        qr.add_data(auth_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        qr_rendered = True
    except ImportError:
        pass
    except Exception:
        pass

    page_url = f"{_QR_CODE_PAGE}{urllib.parse.quote(scode)}"
    if qr_rendered:
        print(f"\n  Scan the QR code above, or open this URL directly:\n  {page_url}")
    else:
        print(f"  Open this URL in WeCom on your phone:\n\n  {page_url}\n")
        print("  Tip: pip install qrcode  to display a scannable QR code here next time")
    print()
    print("  Fetching configuration results...", end="", flush=True)

    # ── Step 3: Poll for result ──
    deadline = time.monotonic() + timeout_seconds
    query_url = f"{_QR_QUERY_URL}?scode={urllib.parse.quote(scode)}"
    poll_count = 0

    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(query_url, headers={"User-Agent": "HermesAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            logger.debug("WeCom QR poll error: %s", safe_exc(exc))
            time.sleep(_QR_POLL_INTERVAL)
            continue

        poll_count += 1
        # Print a dot on every poll so progress is visible within 3s.
        print(".", end="", flush=True)

        result_data = result.get("data") or {}
        status = str(result_data.get("status") or "").lower()

        if status == "success":
            print()  # newline after "Fetching configuration results..." dots
            bot_info = result_data.get("bot_info") or {}
            bot_id = str(bot_info.get("botid") or bot_info.get("bot_id") or "").strip()
            secret = str(bot_info.get("secret") or "").strip()
            if bot_id and secret:
                return {"bot_id": bot_id, "secret": secret}
            logger.warning(
                "WeCom QR: scan reported success but bot_info missing or incomplete: %s",
                result_data,
            )
            print(
                "  QR scan reported success but no bot credentials were returned.\n"
                "  This usually means the bot was not actually created on the WeCom side.\n"
                "  Falling back to manual credential entry."
            )
            return None

        time.sleep(_QR_POLL_INTERVAL)

    print()  # newline after dots
    print(f"  QR scan timed out ({timeout_seconds // 60} minutes). Please try again.")
    return None


# ──────────────────────────────────────────────────────────────────────────
# Plugin migration glue (#41112 / #3823)
#
# Added when the WeCom adapters (wecom + wecom_callback, sharing the
# wecom_crypto satellite) moved from gateway/platforms/ into this bundled
# plugin. register() exposes BOTH platforms via the registry, replacing the
# Platform.WECOM / Platform.WECOM_CALLBACK elifs in gateway/run.py, the
# _PLATFORM_CONNECTED_CHECKERS entries in gateway/config.py, the _setup_wecom
# wizard + _PLATFORMS["wecom"] static dict in hermes_cli/gateway.py, and the
# _send_wecom dispatch in tools/send_message_tool.py. Env→PlatformConfig
# seeding stays in core, same as prior migrations.
# ──────────────────────────────────────────────────────────────────────────


async def _standalone_send(
    pconfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
):
    """Out-of-process WeCom delivery via the adapter's WebSocket send pipeline.

    Implements the standalone_sender_fn contract so deliver=wecom cron jobs
    succeed when cron runs separately from the gateway. Opens an ephemeral
    WeComAdapter, connects, sends, and disconnects. Replaces the legacy
    _send_wecom helper.
    """
    if not check_wecom_requirements():
        return {"error": "WeCom requirements not met. Need aiohttp + WECOM_BOT_ID/SECRET."}
    try:
        adapter = WeComAdapter(pconfig)
        connected = await adapter.connect()
        if not connected:
            return {"error": f"WeCom: failed to connect - {getattr(adapter, 'fatal_error_message', None) or 'unknown error'}"}
        try:
            result = await adapter.send(chat_id, message)
            if not result.success:
                return {"error": f"WeCom send failed: {result.error}"}
            return {
                "success": True,
                "platform": "wecom",
                "chat_id": chat_id,
                "message_id": result.message_id,
            }
        finally:
            await adapter.disconnect()
    except Exception as e:
        return {"error": f"WeCom send failed: {e}"}


def interactive_setup() -> None:
    """Interactive setup for WeCom — QR scan or manual credential input.

    Replaces hermes_cli/gateway.py::_setup_wecom and the static
    _PLATFORMS["wecom"] dict. CLI helpers are lazy-imported.
    """
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.setup import prompt_choice
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_info,
        print_success,
        print_warning,
        print_error,
    )

    print_header("WeCom (Enterprise WeChat)")
    existing_bot_id = get_env_value("WECOM_BOT_ID")
    existing_secret = get_env_value("WECOM_SECRET")
    if existing_bot_id and existing_secret:
        print_success("WeCom is already configured.")
        if not prompt_yes_no("Reconfigure WeCom?", False):
            return

    method_idx = prompt_choice(
        "How would you like to set up WeCom?",
        [
            "Scan QR code to obtain Bot ID and Secret automatically (recommended)",
            "Enter existing Bot ID and Secret manually",
        ],
        0,
    )

    bot_id = None
    secret = None

    if method_idx == 0:
        try:
            credentials = qr_scan_for_bot_info()
        except KeyboardInterrupt:
            print_warning("WeCom setup cancelled.")
            return
        except Exception as exc:
            print_warning(f"QR scan failed: {exc}")
            credentials = None
        if credentials:
            bot_id = credentials.get("bot_id", "")
            secret = credentials.get("secret", "")
            print_success("✔ QR scan successful! Bot ID and Secret obtained.")
        if not bot_id or not secret:
            print_info("QR scan did not complete. Continuing with manual input.")
            bot_id = None
            secret = None

    if not bot_id or not secret:
        print_info("1. Go to WeCom Application → Workspace → Smart Robot -> Create smart robots")
        print_info("2. Select API Mode")
        print_info("3. Copy the Bot ID and Secret from the bot's credentials info")
        print_info("4. The bot connects via WebSocket — no public endpoint needed")
        bot_id = prompt("Bot ID", password=False)
        if not bot_id:
            print_warning("Skipped — WeCom won't work without a Bot ID.")
            return
        secret = prompt("Secret", password=True)
        if not secret:
            print_warning("Skipped — WeCom won't work without a Secret.")
            return

    save_env_value("WECOM_BOT_ID", bot_id)
    save_env_value("WECOM_SECRET", secret)

    print_info("The gateway DENIES all users by default for security.")
    print_info("Enter user IDs to create an allowlist, or leave empty.")
    allowed = prompt("Allowed user IDs (comma-separated, or empty)", password=False)
    if allowed:
        save_env_value("WECOM_ALLOWED_USERS", allowed.replace(" ", ""))
        print_success("Saved — only these users can interact with the bot.")
    else:
        access_idx = prompt_choice(
            "How should unauthorized users be handled?",
            [
                "Enable open access (anyone can message the bot)",
                "Use DM pairing (unknown users request access, you approve with 'hermes pairing approve')",
                "Disable direct messages",
                "Skip for now (bot will deny all users until configured)",
            ],
            1,
        )
        if access_idx == 0:
            save_env_value("WECOM_DM_POLICY", "open")
            save_env_value("GATEWAY_ALLOW_ALL_USERS", "true")
            print_warning("Open access enabled — anyone can use your bot!")
        elif access_idx == 1:
            save_env_value("WECOM_DM_POLICY", "pairing")
            print_success("DM pairing mode — users will receive a code to request access.")
            print_info("Approve with: hermes pairing approve <platform> <code>")
        elif access_idx == 2:
            save_env_value("WECOM_DM_POLICY", "disabled")
            print_warning("Direct messages disabled.")
        else:
            print_info("Skipped — configure later with 'hermes gateway setup'")

    home = prompt("Home chat ID (optional, for cron/notifications)", password=False).strip()
    if home:
        save_env_value("WECOM_HOME_CHANNEL", home)
        print_success(f"Home channel set to {home}")
    else:
        if remove_env_value("WECOM_HOME_CHANNEL"):
            print_info("Home channel cleared.")

    print_success("💬 WeCom configured!")


def _is_connected(config) -> bool:
    """WeCom (Smart Robot) is connected when a bot_id is configured. Mirrors the
    legacy _PLATFORM_CONNECTED_CHECKERS[Platform.WECOM] entry."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("bot_id"))


def _callback_is_connected(config) -> bool:
    """WeCom callback mode is connected when corp_id (or a multi-app `apps`
    block) is configured. Mirrors the legacy
    _PLATFORM_CONNECTED_CHECKERS[Platform.WECOM_CALLBACK] entry."""
    extra = getattr(config, "extra", {}) or {}
    return bool(extra.get("corp_id") or extra.get("apps"))


def _build_adapter(config):
    """Factory wrapper that constructs WeComAdapter from a PlatformConfig."""
    return WeComAdapter(config)


def _build_callback_adapter(config):
    """Factory wrapper that constructs WecomCallbackAdapter from a PlatformConfig."""
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
    return WecomCallbackAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — registers both WeCom platforms."""
    ctx.register_platform(
        name="wecom",
        label="WeCom (Enterprise WeChat)",
        adapter_factory=_build_adapter,
        check_fn=check_wecom_requirements,
        is_connected=_is_connected,
        validate_config=_is_connected,
        required_env=["WECOM_BOT_ID", "WECOM_SECRET"],
        install_hint="Run `hermes setup` to install WeCom support.",
        setup_fn=interactive_setup,
        allowed_users_env="WECOM_ALLOWED_USERS",
        allow_all_env="WECOM_ALLOW_ALL_USERS",
        cron_deliver_env_var="WECOM_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4000,
        emoji="💼",
        allow_update_command=True,
    )

    from plugins.platforms.wecom.callback_adapter import check_wecom_callback_requirements
    ctx.register_platform(
        name="wecom_callback",
        label="WeCom Callback (self-built apps)",
        adapter_factory=_build_callback_adapter,
        check_fn=check_wecom_callback_requirements,
        is_connected=_callback_is_connected,
        validate_config=_callback_is_connected,
        required_env=["WECOM_CALLBACK_CORP_ID", "WECOM_CALLBACK_CORP_SECRET"],
        install_hint="Run `hermes setup` to install WeCom support.",
        allowed_users_env="WECOM_CALLBACK_ALLOWED_USERS",
        allow_all_env="WECOM_CALLBACK_ALLOW_ALL_USERS",
        emoji="💼",
        allow_update_command=True,
    )
