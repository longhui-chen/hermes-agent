"""Unit tests for the extracted turn prologue (``agent/turn_context.py``).

These exercise ``build_turn_context`` against a lightweight fake agent to
confirm the prologue produces the right ``TurnContext`` and applies the
``agent`` side effects the loop relies on — without spinning up a real
``AIAgent`` or hitting any provider.
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import types
from unittest.mock import MagicMock, patch

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import agent.zet_agent_response_mode as response_mode
import tools.skills_tool as skills_tool_module
from agent.context_compressor import ContextCompressor
from agent.conversation_loop import (
    _consume_trusted_skill_slug,
    _consume_trusted_skill_task_message,
    _hardware_scope_skill_slug,
)
from agent.turn_context import TurnContext, build_turn_context
from agent.zet_agent_response_mode import (
    apply_trusted_skill_execution,
    reset_trusted_skill_execution,
    trusted_skill_allowed_tool_names,
    trusted_skill_operation_block_message,
)
from gateway.session_context import clear_turn_vars, set_turn_vars
from hermes_state import SessionDB

_TEST_INTEGRITY_KEY_ID = "presets-test-202607"
_TEST_INTEGRITY_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)


def test_video_skill_slug_never_enters_hardware_trusted_scope():
    assert _hardware_scope_skill_slug("video-edit-workflow-mini") == ""
    assert _hardware_scope_skill_slug("/video_edit") == ""
    assert _hardware_scope_skill_slug("camsnap") == "camsnap"


class _FakeTodoStore:
    def has_items(self):
        return True

    def _hydrate(self, *_a, **_k):
        pass


class _FakeGuardrails:
    def __init__(self):
        self.reset_called = False

    def reset_for_turn(self):
        self.reset_called = True


def _write_presets_integrity_manifest(
    presets_dir,
    *,
    skill_bytes: bytes,
    monkeypatch,
    key_id: str = _TEST_INTEGRITY_KEY_ID,
    private_key: Ed25519PrivateKey = _TEST_INTEGRITY_PRIVATE_KEY,
    register_public_key: bool = True,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    files = {
        "skills/camsnap/SKILL.md": hashlib.sha256(
            skill_bytes
        ).hexdigest(),
    }
    files.update({
        path: hashlib.sha256(payload).hexdigest()
        for path, payload in (extra_files or {}).items()
    })
    manifest = {
        "schema": "zettlab.presets.integrity.v1",
        "files": files,
    }
    manifest_raw = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    public_key_b64 = base64.b64encode(
        private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    if register_public_key:
        monkeypatch.setitem(
            response_mode._TRUSTED_PRESETS_PUBLIC_KEYS_B64,
            key_id,
            public_key_b64,
        )
    signature = {
        "schema": "zettlab.presets.integrity-signature.v1",
        "key_id": key_id,
        "signature": base64.b64encode(
            private_key.sign(manifest_raw)
        ).decode("ascii"),
    }
    skills_dir = presets_dir / "skills"
    (skills_dir / ".zettlab-integrity.json").write_bytes(manifest_raw)
    (skills_dir / ".zettlab-integrity.sig.json").write_text(
        json.dumps(signature, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


class _FakeAgent:
    """Minimal stand-in covering only what the prologue touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.model = "test/model"
        self.provider = "openrouter"
        self.requested_provider = "openrouter"
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = "sk-x"
        self.api_mode = "chat_completions"
        self.platform = "cli"
        self.quiet_mode = True
        self.max_iterations = 90
        self.tools = []
        self.valid_tool_names = set()
        self.enabled_toolsets = None
        self.disabled_toolsets = None
        self._skip_mcp_refresh = False
        self.compression_enabled = False
        self.context_compressor = types.SimpleNamespace(
            protect_first_n=2, protect_last_n=2
        )
        # Make the fake compressor honour the ContextEngine contract that the
        # real code now relies on (should_compress_info returns a (bool, reason)
        # tuple). Without it build_turn_context raises AttributeError.
        def _fake_should_compress(tokens=None):
            return False

        def _fake_should_compress_info(tokens=None):
            return (False, None)

        self.context_compressor.should_compress = _fake_should_compress
        self.context_compressor.should_compress_info = _fake_should_compress_info
        self._cached_system_prompt = "SYSTEM"
        self._memory_store = None
        self._memory_manager = None
        self._memory_nudge_interval = 0
        self._turns_since_memory = 0
        self._user_turn_count = 0
        self._todo_store = _FakeTodoStore()
        self._tool_guardrails = _FakeGuardrails()
        self._compression_warning = None
        self._emit_warning = MagicMock()
        self._last_ctx_overflow_warn = None
        self._interrupt_requested = False
        self._memory_write_origin = "assistant_tool"
        self._stream_context_scrubber = None
        self._stream_think_scrubber = None
        # Attributes the prologue assigns; recorded for assertions.
        self._invalid_tool_retries = -1
        self._vision_supported = None
        self._persist_calls = 0
        self._session_messages = []
        self._pending_cli_user_message = None
        self._session_persist_lock = threading.RLock()
        # Records _cached_system_prompt at the moment _ensure_db_session()
        # is called (regression guard for #45499 turn-setup ordering).
        self._ensure_db_prompt_at_call = "<unset>"

    def _warn_context_overflow_blocked(self, reason, preflight_tokens, threshold_tokens):
        # Mirror the real AIAgent helper so tests can assert the warning fired.
        _warn_kind = (reason or "unknown").split(":", 1)[0]
        _warn_key = ("ctx_overflow_blocked", _warn_kind)
        if self._last_ctx_overflow_warn != _warn_key:
            self._last_ctx_overflow_warn = _warn_key
            self._emit_warning(
                f"⚠ Context is over the compression threshold "
                f"(~{preflight_tokens:,} tokens >= {threshold_tokens:,}) "
                f"but compression is currently blocked ({reason})."
            )

    def _clear_context_overflow_warn(self):
        self._last_ctx_overflow_warn = None

    # --- methods the prologue calls ---
    def _ensure_db_session(self):
        self._ensure_db_prompt_at_call = self._cached_system_prompt

    def _restore_primary_runtime(self):
        pass

    def _cleanup_dead_connections(self):
        return False

    def _emit_status(self, _msg):
        pass

    def _replay_compression_warning(self):
        pass

    def _hydrate_todo_store(self, *_a, **_k):
        pass

    def _safe_print(self, *_a, **_k):
        pass

    def _persist_session(self, *_a, **_k):
        self._persist_calls += 1


def _make_agent_with_cooldown(db_path, session_id, *, cooldown_until=None):
    agent = _FakeAgent()
    agent.compression_enabled = True
    agent._emit_status = MagicMock()
    agent._compress_context = MagicMock(
        side_effect=lambda messages, *_a, **_k: (messages, "SYSTEM")
    )

    db = SessionDB(db_path=db_path)
    db.create_session(session_id, source="cli")
    if cooldown_until is not None:
        db.record_compression_failure_cooldown(session_id, cooldown_until, "timeout")

    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
    compressor.bind_session_state(db, session_id)
    agent.context_compressor = compressor
    agent._session_db = db
    return agent


@pytest.fixture(autouse=True)
def _stub_runtime_main():
    """``build_turn_context`` calls ``auxiliary_client.set_runtime_main`` as a
    production side effect (telling aux tools the live main provider/model).
    That writes a module-level global these unit tests don't care about and
    which would otherwise leak into sibling tests (e.g. provider-parity
    resolution) when the per-test process isolation plugin is disabled. Stub
    it out so the prologue tests stay hermetic.
    """
    with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
         patch("agent.auxiliary_client.set_runtime_auxiliary_task_configs", lambda *a, **k: None):
        yield


def _build(agent, **overrides):
    kwargs = dict(
        agent=agent,
        user_message="hello",
        system_message=None,
        conversation_history=None,
        task_id=None,
        stream_callback=None,
        persist_user_message=None,
        restore_or_build_system_prompt=lambda *a, **k: None,
        install_safe_stdio=lambda: None,
        sanitize_surrogates=lambda s: s,
        summarize_user_message_for_log=lambda s: s,
        set_session_context=lambda _sid: None,
        set_current_write_origin=lambda _o: None,
        ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None),
    )
    kwargs.update(overrides)
    return build_turn_context(**kwargs)


def test_returns_turn_context_with_user_message_appended():
    agent = _FakeAgent()
    ctx = _build(agent)
    assert isinstance(ctx, TurnContext)
    assert ctx.user_message == "hello"
    # The user turn was appended and indexed.
    assert ctx.messages[-1] == {"role": "user", "content": "hello"}
    assert ctx.current_turn_user_idx == len(ctx.messages) - 1
    assert ctx.active_system_prompt == "SYSTEM"


# ── Trivial-prompt prefetch gate (PR #25350 salvage) ─────────────────────────
#
# The prologue is the ONLY place the per-turn synchronous
# memory_manager.prefetch_all() fires; a bare greeting must not block the
# turn on provider network round-trips, while a substantive question must
# still prefetch. These assert the gate at the call site (the classifier
# itself is covered in tests/agent/test_memory_provider.py).


def _agent_with_memory_manager():
    agent = _FakeAgent()
    mm = MagicMock()
    mm.prefetch_all.return_value = "REMEMBERED CONTEXT"
    agent._memory_manager = mm
    return agent, mm


def test_prefetch_skipped_for_trivial_user_message():
    agent, mm = _agent_with_memory_manager()
    ctx = _build(agent, user_message="hi!")
    mm.prefetch_all.assert_not_called()
    assert ctx.ext_prefetch_cache == ""


def test_prefetch_runs_for_substantive_user_message():
    agent, mm = _agent_with_memory_manager()
    query = "what did we decide about the deploy pipeline?"
    ctx = _build(agent, user_message=query)
    mm.prefetch_all.assert_called_once_with(query)
    assert ctx.ext_prefetch_cache == "REMEMBERED CONTEXT"


def test_turn_start_replaces_stale_parent_history_with_compression_child():
    agent = _FakeAgent()
    stale_history = [{"role": "user", "content": "stale parent"}]
    compacted_history = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "assistant", "content": "child tail"},
    ]

    def _recover(_agent):
        _agent.session_id = "compression-child"
        return compacted_history

    log_context = MagicMock()
    with patch(
        "agent.turn_context.recover_rotated_compression_session",
        side_effect=_recover,
    ):
        ctx = _build(
            agent,
            conversation_history=stale_history,
            set_session_context=log_context,
        )

    assert agent.session_id == "compression-child"
    assert agent._current_turn_id.startswith("compression-child:")
    log_context.assert_called_once_with("compression-child")
    assert ctx.conversation_history == compacted_history
    assert ctx.messages == compacted_history + [{"role": "user", "content": "hello"}]
    assert all(message.get("content") != "stale parent" for message in ctx.messages)
def test_governor_scope_follows_the_compression_child():
    """推荐卡存进哪个 scope，必须跟响应头回给客户端的 session 一致。

    压缩旋转恢复会把 agent.session_id 换成 canonical child。在恢复之前绑定
    governor scope，卡片就存进了父 scope，客户端照响应头提交动作时 governor 在
    子 scope 里找不到刚展示的 proposal，只能拒绝——那张卡从此点不动。
    """
    agent = _FakeAgent()

    def _recover(_agent):
        _agent.session_id = "compression-child"
        return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]

    with patch(
        "agent.turn_context.recover_rotated_compression_session",
        side_effect=_recover,
    ):
        _build(agent, conversation_history=[{"role": "user", "content": "stale parent"}])

    assert agent._creation_governor_conversation_session_id == "compression-child"


def test_governor_scope_binds_after_mid_turn_session_rotation():
    """绑定必须发生在本轮所有会旋转 session 的动作之后。

    turn-start 的旋转恢复、idle 压缩、preflight 压缩都可能把 agent.session_id
    换成 canonical child，而响应头回给客户端的是 child。绑早了，推荐卡就存进
    父 scope，客户端照响应头提交动作时 governor 在 child scope 里找不到刚展示
    的 proposal，只能拒绝——那张卡从此点不动。

    这里用「系统提示重建时旋转 session」模拟中途旋转：它排在 turn-start 恢复
    之后，绑定点如果还留在恢复旁边就会读到旧值。
    """
    agent = _FakeAgent()
    # 逼真实的系统提示重建路径跑起来（默认 fixture 直接给了缓存值就不调了）。
    agent._cached_system_prompt = None

    def _rotate_during_prompt_restore(*_args, **_kwargs):
        agent.session_id = "rotated-child"
        return "SYSTEM"

    _build(agent, restore_or_build_system_prompt=_rotate_during_prompt_restore)

    assert agent._creation_governor_conversation_session_id == "rotated-child"


def test_explicit_gateway_session_key_survives_the_compression_child():
    """对照：调用方显式指定的 gateway key 是稳定作用域，恢复不该动它。"""
    agent = _FakeAgent()
    agent._gateway_session_key = "app-conversation-42"

    def _recover(_agent):
        _agent.session_id = "compression-child"
        return [{"role": "user", "content": "[CONTEXT COMPACTION] summary"}]

    with patch(
        "agent.turn_context.recover_rotated_compression_session",
        side_effect=_recover,
    ):
        _build(agent, conversation_history=[{"role": "user", "content": "stale parent"}])

    assert agent._creation_governor_conversation_session_id == "app-conversation-42"


def test_records_trusted_current_user_and_previous_assistant_messages():
    agent = _FakeAgent()
    _build(
        agent,
        user_message="确认安装",
        conversation_history=[
            {"role": "user", "content": "帮我找技能"},
            {
                "role": "assistant",
                "content": "候选是 owner/repo/example，是否安装？",
            },
        ],
    )

    assert agent._current_user_message == "确认安装"
    assert (
        agent._previous_assistant_message
        == "候选是 owner/repo/example，是否安装？"
    )


def test_applies_agent_side_effects():
    agent = _FakeAgent()
    _build(agent)
    # Retry counters reset, guardrails reset, vision re-armed, turn counted.
    assert agent._invalid_tool_retries == 0
    assert agent._tool_guardrails.reset_called is True
    assert agent._vision_supported is True
    assert agent._user_turn_count == 1
    # Crash-resilience persistence fired once.
    assert agent._persist_calls == 1
    # task/turn ids assigned on the agent.
    assert agent._current_task_id
    assert agent._current_turn_id




def test_trusted_skill_fresh_read_context_clears_after_dispatch_error(
    monkeypatch,
):
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    monkeypatch.setattr(
        response_mode,
        "_trusted_skill_view_refresh_required",
        lambda _agent, _args: True,
    )

    def _fail():
        assert response_mode.trusted_skill_view_fresh_read_required()
        raise RuntimeError("skill read failed")

    with pytest.raises(RuntimeError, match="skill read failed"):
        response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="skill_view",
            function_args={"name": "trusted-skill"},
            dispatch=_fail,
        )

    assert not response_mode.trusted_skill_view_fresh_read_required()


def test_startup_rejects_dev_signing_key_for_release_directory(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "v0.7.99"
    skill_dir = presets_dir / "skills" / "camsnap"
    skill_dir.mkdir(parents=True)
    skill_bytes = b"# release directory with development signature\n"
    (skill_dir / "SKILL.md").write_bytes(skill_bytes)
    dev_key_id = "presets-dev-test-202607"
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=skill_bytes,
        monkeypatch=monkeypatch,
        key_id=dev_key_id,
        register_public_key=False,
    )
    public_key_b64 = base64.b64encode(
        _TEST_INTEGRITY_PRIVATE_KEY.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))
    monkeypatch.setenv("ZETTLAB_PRESETS_DEV_KEY_ID", dev_key_id)
    monkeypatch.setenv(
        "ZETTLAB_PRESETS_DEV_PUBLIC_KEY_B64",
        public_key_b64,
    )

    assert response_mode._capture_trusted_presets_snapshot() is None


def test_startup_rejects_incomplete_dev_signing_key_configuration(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "dev-feature-intl-20260724"
    skill_dir = presets_dir / "skills" / "camsnap"
    skill_dir.mkdir(parents=True)
    skill_bytes = b"# development signature missing public key config\n"
    (skill_dir / "SKILL.md").write_bytes(skill_bytes)
    dev_key_id = "presets-dev-test-202607"
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=skill_bytes,
        monkeypatch=monkeypatch,
        key_id=dev_key_id,
        register_public_key=False,
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))
    monkeypatch.setenv("ZETTLAB_PRESETS_DEV_KEY_ID", dev_key_id)
    monkeypatch.delenv("ZETTLAB_PRESETS_DEV_PUBLIC_KEY_B64", raising=False)

    assert response_mode._capture_trusted_presets_snapshot() is None


def test_startup_rejects_skill_bytes_not_pinned_by_release_manifest(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "presets"
    skill_dir = presets_dir / "skills" / "camsnap"
    skill_dir.mkdir(parents=True)
    released_bytes = b"# released skill\n"
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=released_bytes,
        monkeypatch=monkeypatch,
    )
    (skill_dir / "SKILL.md").write_bytes(b"# forged before startup\n")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))

    snapshot = response_mode._capture_trusted_presets_snapshot()

    assert snapshot is not None
    assert snapshot.skills == ()


def test_startup_rejects_forged_skill_and_manifest_without_release_key(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "presets"
    skill_dir = presets_dir / "skills" / "camsnap"
    skill_dir.mkdir(parents=True)
    released_bytes = b"# released skill\n"
    (skill_dir / "SKILL.md").write_bytes(released_bytes)
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=released_bytes,
        monkeypatch=monkeypatch,
    )

    forged_bytes = b"# forged skill and manifest before startup\n"
    (skill_dir / "SKILL.md").write_bytes(forged_bytes)
    forged_manifest = {
        "schema": "zettlab.presets.integrity.v1",
        "files": {
            "skills/camsnap/SKILL.md": hashlib.sha256(
                forged_bytes
            ).hexdigest(),
        },
    }
    (presets_dir / "skills" / ".zettlab-integrity.json").write_text(
        json.dumps(forged_manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))

    snapshot = response_mode._capture_trusted_presets_snapshot()

    assert snapshot is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("拍一张快照", True),
        ("请帮我拍张快照", True),
        ("获取当前快照", True),
        ("返回当前最新的一张图片", False),
        ("总结“拍一张快照”这句话", False),
        ("把这张图片压缩一下", False),
    ],
)
def test_camera_shortcut_only_accepts_bounded_snapshot_intent(message, expected):
    agent = _FakeAgent()

    task = response_mode._skill_direct_task_context(agent, message)

    assert task.camera_applicable is expected
    assert not task.camera_explicit


@pytest.mark.parametrize(
    "message",
    [
        "看下摄像头下现在有多少人",
        "获取摄像头最新画面并统计人数",
        "分析摄像头当前画面有没有人",
        "count the people in the current camera frame",
    ],
)
def test_camera_analysis_intent_uses_camera_scope(message):
    task = response_mode._skill_direct_task_context(_FakeAgent(), message)

    assert task.camera_applicable


@pytest.mark.parametrize(
    "message",
    [
        "按顺序检查摄像头、3D 打印机、电脑和网络位置的连接状态",
        "检查一下摄像头是否可用",
        "check camera connection status",
    ],
)
def test_camera_inventory_intent_uses_list_only_scope(message):
    task = response_mode._skill_direct_task_context(_FakeAgent(), message)

    assert task.camera_applicable
    assert task.camera_inventory_only


def test_broad_hardware_inventory_grants_camera_list_and_printer_read():
    task = response_mode._skill_direct_task_context(
        _FakeAgent(),
        "检查下当前已连接的硬件状态",
    )

    assert task.camera_applicable
    assert task.camera_inventory_only
    assert task.printer3d_applicable


def test_specific_pc_inventory_does_not_grant_camera_or_printer_scope():
    task = response_mode._skill_direct_task_context(
        _FakeAgent(),
        "查看硬件连接中的电脑",
    )

    assert not task.camera_applicable
    assert not task.printer3d_applicable


def test_camera_media_intent_is_not_inventory_only():
    task = response_mode._skill_direct_task_context(
        _FakeAgent(),
        "获取摄像头最新画面并统计人数",
    )

    assert task.camera_applicable
    assert not task.camera_inventory_only


@pytest.mark.parametrize("message", [
    "我已经授权了摄像头，再试下",
    "摄像头已经打开，重试",
    "摄像头开关打开了，重试刚才的操作",
])
def test_camera_authorization_retry_is_inventory_only(message):
    task = response_mode._skill_direct_task_context(_FakeAgent(), message)
    assert task.camera_applicable
    assert task.camera_inventory_only


@pytest.mark.parametrize("message", [
    "再试下",
    "总结“我已经授权了摄像头，再试下”这句话",
    "我已经授权了摄像头，再试下并开始持续监控",
    "我没有授权摄像头，再试下",
])
def test_camera_retry_does_not_infer_authority_from_unrelated_text(message):
    task = response_mode._skill_direct_task_context(_FakeAgent(), message)
    assert not task.camera_applicable


def test_camera_inventory_command_policy_allows_list_but_blocks_capture(
    monkeypatch,
):
    def camera_argv(args):
        command = str(args.get("command") or "")
        if command.endswith(" list"):
            return ["python3", "camera_connector.py", "list"]
        if " snap --camera-id " in command:
            return [
                "python3",
                "camera_connector.py",
                "snap",
                "--camera-id",
                command.rsplit(" ", 1)[-1],
            ]
        return None

    monkeypatch.setattr(response_mode, "_camera_runtime_argv", camera_argv)
    assert response_mode._camera_command_policy(
        {"command": "python3 camera_connector.py list"},
        inventory_only=True,
    )
    assert not response_mode._camera_command_policy(
        {
            "command": (
                "python3 camera_connector.py snap --camera-id cam_front"
            )
        },
        camera_ids=frozenset({"cam_front"}),
        inventory_only=True,
    )


def test_unrelated_people_count_does_not_use_camera_scope():
    task = response_mode._skill_direct_task_context(
        _FakeAgent(),
        "统计这份文档里提到了多少人",
    )

    assert not task.camera_applicable


def test_camera_continuation_requires_same_session_recent_snapshot():
    from gateway.session_context import clear_session_vars, set_session_vars

    response_mode._CAMERA_RESUME_SESSIONS.clear()
    session_tokens = set_session_vars(
        session_key="zettlab:user:main:camera-continuation",
        session_id="zettlab:user:main:camera-continuation",
    )
    first_turn = set_turn_vars(turn_id="camera-source-turn")
    try:
        turn_identity = response_mode._current_skill_direct_turn_identity()
        assert turn_identity is not None
        response_mode._remember_camera_resume_locked(
            turn_identity=turn_identity,
            now=response_mode.time.monotonic(),
        )
    finally:
        clear_turn_vars(first_turn)

    continuation_turn = set_turn_vars(turn_id="camera-continuation-turn")
    try:
        task = response_mode._skill_direct_task_context(
            _FakeAgent(),
            "再获取下最新的画面，统计下当前画面有多少个人",
        )
        assert task.camera_applicable
    finally:
        clear_turn_vars(continuation_turn)
        clear_session_vars(session_tokens)

    other_session_tokens = set_session_vars(
        session_key="zettlab:user:main:other-session",
        session_id="zettlab:user:main:other-session",
    )
    other_turn = set_turn_vars(turn_id="camera-other-turn")
    try:
        task = response_mode._skill_direct_task_context(
            _FakeAgent(),
            "再获取下最新的画面，统计下当前画面有多少个人",
        )
        assert not task.camera_applicable
    finally:
        clear_turn_vars(other_turn)
        clear_session_vars(other_session_tokens)
        response_mode._CAMERA_RESUME_SESSIONS.clear()


def test_camera_transport_selection_authorizes_ambiguous_display_text():
    agent = _FakeAgent()

    task = response_mode._skill_direct_task_context(
        agent,
        "返回当前最新的一张图片",
        explicit_skill_slug="camsnap",
    )

    assert task.camera_applicable
    assert task.camera_explicit


def test_camera_list_result_extracts_response_bounded_valid_id_snapshot():
    result = {
        "output": json.dumps(
            {
                "data": {
                    "action": "list",
                    "status": "ok",
                    "cameras": [
                        {"camera_id": "cam_front", "name": "Front"},
                        {"camera_id": "cam_back", "name": "Back"},
                    ],
                }
            }
        )
    }

    assert response_mode._camera_ids_from_terminal_result(result) == {
        "cam_front",
        "cam_back",
    }
    invalid = {
        "output": json.dumps(
            {
                "data": {
                    "action": "list",
                    "status": "ok",
                    "cameras": [{"camera_id": "../../secret"}],
                }
            }
        )
    }
    assert response_mode._camera_ids_from_terminal_result(invalid) is None
    many_cameras = {
        "output": json.dumps(
            {
                "data": {
                    "action": "list",
                    "status": "ok",
                    "cameras": [
                        {"camera_id": f"cam_{index}"}
                        for index in range(128)
                    ],
                }
            }
        )
    }
    assert len(response_mode._camera_ids_from_terminal_result(many_cameras) or ()) == 128
    oversized_output = {"output": "x" * (1024 * 1024 + 1)}
    assert response_mode._camera_ids_from_terminal_result(oversized_output) is None


def test_camera_snapshot_attachment_stays_under_active_output_root(tmp_path):
    from agent import secret_scope as secret_scope_module

    output_root = tmp_path / "output"
    output_root.mkdir()
    frame = output_root / "current.jpg"
    frame.write_bytes(b"jpeg")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"jpeg")
    symlink = output_root / "linked.jpg"
    symlink.symlink_to(outside)
    scope_token = secret_scope_module.set_secret_scope(
        {"ZET_AGENT_OUTPUT_DIR": str(output_root)}
    )
    try:
        result = {
            "output": json.dumps(
                {
                    "data": {
                        "action": "snap",
                        "status": "ok",
                        "attachment_path": str(frame),
                    }
                }
            )
        }
        assert response_mode._camera_attachment_path_from_terminal_result(
            result
        ) == str(frame.resolve())
        assert response_mode._trusted_camera_attachment_path(str(outside)) is None
        assert response_mode._trusted_camera_attachment_path(str(symlink)) is None
    finally:
        secret_scope_module.reset_secret_scope(scope_token)


def test_camera_vision_scope_is_bound_to_exact_current_attachment(monkeypatch):
    turn_tokens = set_turn_vars(turn_id="camera-vision-turn")
    try:
        agent = _FakeAgent()
        agent.platform = "zet_agent"
        task = response_mode._skill_direct_task_context(
            agent,
            "获取摄像头最新画面并统计人数",
        )
        exact_path = "/trusted/output/current.jpg"
        monkeypatch.setattr(
            response_mode,
            "_trusted_camera_attachment_path",
            lambda raw_path: str(raw_path) if raw_path == exact_path else None,
        )
        scope = response_mode._SkillDirectScope(
            relative_path=response_mode._CAMERA_SKILL_PATH,
            task_sha256=task.task_sha256,
            turn_identity=task.turn_identity,
            allowed_tools=frozenset({"terminal", "vision_analyze"}),
            execution_receipt=response_mode._TrustedExecutionReceipt(
                agent_id="main",
                action_token="action-token",
                hardware_execution_token="b" * 64,
                turn_id="camera-vision-turn",
                session_id="camera-session",
            ),
            camera_attachment_paths=frozenset({exact_path}),
        )
        agent._zet_agent_skill_direct_task = task
        agent._zet_agent_skill_direct_scope = scope
        agent._zet_agent_skill_direct_operation = None
        args = {
            "image_url": exact_path,
            "question": "统计画面中清晰可见的人数。",
        }

        assert trusted_skill_operation_block_message(
            agent,
            function_name="vision_analyze",
            function_args=args,
        ) is None
        result = response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="vision_analyze",
            function_args=args,
            dispatch=lambda: '{"people": 3}',
        )
        assert result == '{"people": 3}'
        assert trusted_skill_allowed_tool_names(agent) == frozenset()

        agent._zet_agent_skill_direct_scope = scope
        blocked = trusted_skill_operation_block_message(
            agent,
            function_name="vision_analyze",
            function_args={
                "image_url": "/trusted/output/older.jpg",
                "question": "统计人数",
            },
        )
        assert blocked is not None
        assert "blocked before execution" in blocked
    finally:
        clear_turn_vars(turn_tokens)


@pytest.mark.parametrize(
    ("message", "expected_types"),
    [
        ("帮我连接下摄像头", ("camera",)),
        ("添加一台 3D 打印机和一个电脑节点", ("printer3d", "pc_node")),
        ("发现附近可以连接的硬件设备", ("camera", "printer3d", "pc_node", "tv")),
        ("Connect a camera and a 3D printer", ("camera", "printer3d")),
        ("配对一个语音终端", ("voice_terminal",)),
        ("解释一下“帮我连接摄像头”这句话", ()),
        ("摄像头连接失败了", ()),
        ("查看摄像头", ()),
        ("查看下硬件连接中的电脑", ()),
        ("查看硬件连接中的电脑有哪些文件", ()),
        ("查看已连接的电脑", ()),
        ("Show connected computers", ()),
        ("Show the computer connection status", ()),
        ("帮我重新连接电脑", ("pc_node",)),
    ],
)
def test_hardware_enrollment_fallback_is_bounded(message, expected_types):
    assert response_mode._hardware_enrollment_requested_types(message) == expected_types


def test_private_subnet_hardware_enrollment_is_normalized_and_narrowed():
    assert response_mode._private_hardware_discovery_scope(
        "发现 192.168.8.27/24 网段里的硬件设备",
    ) == "192.168.8.0/24"
    assert response_mode._hardware_enrollment_requested_types(
        "发现 192.168.8.27/24 网段里的硬件设备",
        subnet_scoped=True,
    ) == ("camera", "tv")
    assert response_mode._private_hardware_discovery_scope(
        "发现 203.0.113.0/24 网段里的设备",
    ) == ""
    assert response_mode._private_hardware_discovery_scope(
        "发现 192.168.0.0/16 网段里的设备",
    ) == ""


def test_current_subnet_hardware_enrollment_emits_current_scope():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    response = response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="帮我扫描下当前网段有哪些硬件设备可以连接",
        response_text="将在卡片内发现 ONVIF 摄像头和 DLNA 电视。",
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    )

    assert response.count("```zettlab-connector-enrollment-intent") == 1
    assert '"resource_kind": "camera"' in response
    assert '"resource_kind": "tv"' in response
    assert '"network_scope": {\n    "mode": "current"' in response
    assert "printer3d" not in response
    assert "pc_node" not in response


def test_hardware_enrollment_fallback_emits_canonical_secret_free_intent():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    response = response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="帮我连接下摄像头",
        response_text=(
            "请前往设置页。\n\n"
            "```zettlab-hardware-enrollment-intent\n"
            '{"schema_version":"1","kind":"hardware",'
            '"requested_types":["camera"],"discovery_requested":true,'
            '"host":"192.0.2.1"}\n```'
        ),
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    )

    assert "```zettlab-hardware-enrollment-intent" not in response
    assert response.count("```zettlab-connector-enrollment-intent") == 1
    assert '"resource_kind": "camera"' in response
    assert "192.0.2.1" not in response
    assert '"host"' not in response


def test_hardware_enrollment_fallback_replaces_duplicate_v2_and_v1_with_scoped_v2():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    response = response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="发现 192.168.8.27/24 网段里的硬件设备",
        response_text=(
            "将生成本地发现预览。\n\n"
            "```zettlab-connector-enrollment-intent\n"
            '{"schema_version":"2","kind":"connector_enrollment","items":'
            '[{"resource_kind":"camera"},{"resource_kind":"tv"}],'
            '"setup_requested":true}\n```\n\n'
            "```zettlab-hardware-enrollment-intent\n"
            '{"schema_version":"1","kind":"hardware","requested_types":'
            '["camera","printer3d","pc_node"],"discovery_requested":true}\n```'
        ),
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    )

    assert response.count("```zettlab-connector-enrollment-intent") == 1
    assert "```zettlab-hardware-enrollment-intent" not in response
    assert '"resource_kind": "camera"' in response
    assert '"resource_kind": "tv"' in response
    assert '"network_scope": {\n    "cidr": "192.168.8.0/24"' in response
    assert "printer3d" not in response
    assert "pc_node" not in response


def test_invalid_or_unsupported_subnet_discovery_never_falls_back_to_broad_scan():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    for message in (
        "发现 203.0.113.0/24 网段里的硬件设备",
        "发现 192.168.0.0/16 网段里的硬件设备",
        "发现 192.168.8.0/24 网段里的 3D 打印机",
    ):
        response = response_mode.ensure_hardware_enrollment_intent(
            agent,
            user_message=message,
            response_text="当前请求无法生成受限发现卡。",
            completed=True,
            failed=False,
            interrupted=False,
            structured_output=False,
        )
        assert "zettlab-connector-enrollment-intent" not in response
        assert "zettlab-hardware-enrollment-intent" not in response


def test_mixed_protocol_v2_is_preserved_without_legacy_hardware_fallback():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    original = (
        "请确认摄像头和 SSH 连接。\n\n"
        "```zettlab-connector-enrollment-intent\n"
        '{"schema_version":"2","kind":"connector_enrollment","items":'
        '[{"resource_kind":"camera"},{"resource_kind":"protocol_endpoint",'
        '"adapter_id":"ssh"}],"setup_requested":true}\n```'
    )
    response = response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="添加一个摄像头和 SSH 连接",
        response_text=original,
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    )

    assert response == original
    assert response.count("```zettlab-connector-enrollment-intent") == 1
    assert "zettlab-hardware-enrollment-intent" not in response


def test_hardware_status_turn_strips_model_authored_enrollment_card():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    response = response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="按顺序检查所有已连接硬件的状态",
        response_text=(
            "摄像头在线，打印机当前会话未授权。\n\n"
            "```zettlab-hardware-enrollment-intent\n"
            '{"schema_version":"1","kind":"hardware",'
            '"requested_types":["camera","printer3d","pc_node"],'
            '"discovery_requested":true}\n```'
        ),
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    )

    assert response == "摄像头在线，打印机当前会话未授权。"


def test_direct_input_client_never_receives_legacy_enrollment_card():
    agent = _FakeAgent()
    agent.platform = "zet_agent"
    agent._zettlab_connector_direct_input = True
    for original in (
        "继续连接。",
        '继续连接。\n```zettlab-hardware-enrollment-intent\n{"schema_version":"1","kind":"hardware","requested_types":["camera"]}\n```',
    ):
        response = response_mode.ensure_hardware_enrollment_intent(
            agent, user_message="帮我连接摄像头", response_text=original,
            completed=True, failed=False, interrupted=False, structured_output=False,
        )
        assert response == "继续连接。"


def test_hardware_enrollment_fallback_ignores_non_app_and_failed_turns():
    agent = _FakeAgent()
    agent.platform = "telegram"
    original = "请前往设置页。"

    assert response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="帮我连接下摄像头",
        response_text=original,
        completed=True,
        failed=False,
        interrupted=False,
        structured_output=False,
    ) == original
    agent.platform = "zet_agent"
    assert response_mode.ensure_hardware_enrollment_intent(
        agent,
        user_message="帮我连接下摄像头",
        response_text=original,
        completed=False,
        failed=True,
        interrupted=False,
        structured_output=False,
    ) == original


@pytest.mark.parametrize(
    ("message", "explicit_skill_slug", "inventory_only"),
    [
        ("拍一张快照", "", False),
        ("返回当前最新的一张图片", "camsnap", False),
        ("我已经授权了摄像头，再试下", "", True),
    ],
)
def test_camera_runtime_receipt_requires_attested_camsnap_scope_flow(
    tmp_path, monkeypatch, message, explicit_skill_slug, inventory_only
):
    from agent import secret_scope as secret_scope_module
    from gateway.session_context import clear_session_vars, set_session_vars
    from tools.environments.local import build_camera_runtime_env

    presets_dir = tmp_path / "presets"
    camera_dir = presets_dir / "skills" / "camsnap"
    camera_dir.mkdir(parents=True)
    camera_bytes = b"# trusted camsnap skill\n"
    (camera_dir / "SKILL.md").write_bytes(camera_bytes)
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=camera_bytes,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))
    snapshot = response_mode._capture_trusted_presets_snapshot()
    assert snapshot is not None
    monkeypatch.setattr(response_mode, "_TRUSTED_PRESETS_SNAPSHOT", snapshot)
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", presets_dir / "skills")
    def _camera_argv(args):
        command = str(args.get("command") or "")
        if command.endswith(" list"):
            return ["python3", "camera_connector.py", "list"]
        if " snap --camera-id " in command:
            return [
                "python3",
                "camera_connector.py",
                "snap",
                "--camera-id",
                command.rsplit(" ", 1)[-1],
            ]
        return None

    monkeypatch.setattr(response_mode, "_camera_runtime_argv", _camera_argv)
    monkeypatch.setattr(
        response_mode,
        "_trusted_camera_attachment_path",
        lambda raw_path: str(raw_path) if raw_path else None,
    )

    secret_token = secret_scope_module.set_secret_scope(
        {
            "ZET_AGENT_ID": "main",
            "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token:camera/v2",
        }
    )
    session_tokens = set_session_vars(
        session_key="zettlab:user:main:camera-session",
        session_id="zettlab:user:main:camera-session",
    )
    turn_tokens = set_turn_vars(
        turn_id="camera-turn",
        hardware_execution_token="b" * 64,
    )
    try:
        with pytest.raises(PermissionError):
            build_camera_runtime_env()

        agent = _FakeAgent()
        agent.platform = "zet_agent"
        reset_trusted_skill_execution(
            agent,
            message,
            explicit_skill_slug=explicit_skill_slug,
        )
        result = skills_tool_module.skill_view("camsnap")
        assert apply_trusted_skill_execution(
            agent,
            function_name="skill_view",
            function_result=result,
        )
        assert trusted_skill_allowed_tool_names(agent) == {
            "terminal",
            "vision_analyze",
        }
        policy_error = trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args={
                "command": "true",
                "workdir": "agent_output",
            },
        )
        assert policy_error is not None
        assert "command-policy error" in policy_error

        list_args = {
            "command": "python3 camera_connector.py list",
            "workdir": "agent_output",
        }
        assert trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args=list_args,
        ) is None
        with pytest.raises(PermissionError):
            build_camera_runtime_env()

        def _frozen_camera_env():
            frozen = build_camera_runtime_env()
            assert frozen == {
                "ZET_AGENT_ID": "main",
                "ZETTLAB_AGENT_ACTION_TOKEN": "profile-token:camera/v2",
                "ZETTLAB_HARDWARE_EXECUTION_TOKEN": "b" * 64,
                "HERMES_TURN_ID": "camera-turn",
                "HERMES_SESSION_ID": "zettlab:user:main:camera-session",
                "HERMES_SESSION_KEY": "zettlab:user:main:camera-session",
            }

        def _dispatch_list():
            _frozen_camera_env()
            return json.dumps(
                {
                    "output": json.dumps(
                        {
                            "data": {
                                "action": "list",
                                "status": "ok",
                                "cameras": [
                                    {
                                        "camera_id": "cam_front",
                                        "name": "Front camera",
                                    }
                                ],
                            }
                        }
                    ),
                    "exit_code": 0,
                    "camera_runtime_direct": True,
                }
            )

        response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="terminal",
            function_args=list_args,
            dispatch=_dispatch_list,
        )

        if inventory_only:
            assert trusted_skill_operation_block_message(
                agent, function_name="terminal", function_args={
                    "command": "python3 camera_connector.py snap --camera-id cam_front",
                },
            ) is not None
            return

        invented_error = trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args={
                "command": "python3 camera_connector.py snap --camera-id 2",
                "workdir": "agent_output",
            },
        )
        assert invented_error is not None
        assert "camera_id returned by that list" in invented_error

        snap_args = {
            "command": (
                "python3 camera_connector.py snap --camera-id cam_front"
            ),
            "workdir": "agent_output",
        }
        assert trusted_skill_operation_block_message(
            agent,
            function_name="terminal",
            function_args=snap_args,
        ) is None

        def _dispatch_snap():
            _frozen_camera_env()
            return json.dumps(
                {
                    "output": json.dumps(
                        {
                            "data": {
                                "action": "snap",
                                "status": "ok",
                                "camera_id": "cam_front",
                                "attachment_path": "/trusted/output/current.jpg",
                            }
                        }
                    ),
                    "exit_code": 0,
                    "camera_runtime_direct": True,
                }
            )

        response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="terminal",
            function_args=snap_args,
            dispatch=_dispatch_snap,
        )
        vision_args = {
            "image_url": "/trusted/output/current.jpg",
            "question": "统计画面中清晰可见的人数。",
        }
        assert trusted_skill_operation_block_message(
            agent,
            function_name="vision_analyze",
            function_args=vision_args,
        ) is None
        assert response_mode.dispatch_trusted_skill_operation(
            agent,
            function_name="vision_analyze",
            function_args=vision_args,
            dispatch=lambda: '{"people": 2}',
        ) == '{"people": 2}'
        with pytest.raises(PermissionError):
            build_camera_runtime_env()
    finally:
        response_mode._TRUSTED_HARDWARE_RUNTIME_RECEIPT.set(None)
        clear_turn_vars(turn_tokens)
        clear_session_vars(session_tokens)
        secret_scope_module.reset_secret_scope(secret_token)


def test_camsnap_repeat_view_mints_fresh_scope_for_next_turn(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "presets"
    camera_dir = presets_dir / "skills" / "camsnap"
    camera_dir.mkdir(parents=True)
    camera_bytes = b"# trusted camsnap skill\n"
    (camera_dir / "SKILL.md").write_bytes(camera_bytes)
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=camera_bytes,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))
    snapshot = response_mode._capture_trusted_presets_snapshot()
    assert snapshot is not None
    monkeypatch.setattr(response_mode, "_TRUSTED_PRESETS_SNAPSHOT", snapshot)
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", presets_dir / "skills")
    monkeypatch.setattr(
        response_mode,
        "_camera_runtime_argv",
        lambda _args: ["python3", "camera_connector.py", "list"],
    )
    monkeypatch.setattr(
        response_mode,
        "_capture_trusted_execution_receipt",
        lambda turn_identity, _relative_path: response_mode._TrustedExecutionReceipt(
            agent_id="main",
            action_token="action-token",
            hardware_execution_token="b" * 64,
            turn_id=turn_identity[0],
            session_id="stable-camera-session",
        ),
    )

    agent = _FakeAgent()
    agent.platform = "zet_agent"
    task_id = "stable-camera-session"
    skills_tool_module.reset_skill_view_dedup(task_id)
    attestations = []
    try:
        for turn_id in ("camera-turn-1", "camera-turn-2"):
            turn_tokens = set_turn_vars(turn_id=turn_id)
            try:
                reset_trusted_skill_execution(
                    agent,
                    "查看摄像头最新快照",
                    explicit_skill_slug="camsnap",
                )
                args = {"name": "camsnap"}
                result = response_mode.dispatch_trusted_skill_operation(
                    agent,
                    function_name="skill_view",
                    function_args=args,
                    dispatch=lambda: skills_tool_module._skill_view_with_bump(
                        args,
                        task_id=task_id,
                    ),
                )
                payload = json.loads(result)
                assert payload.get("dedup") is None
                attestations.append(
                    payload[response_mode._ATTESTATION_FIELD]
                )
                assert apply_trusted_skill_execution(
                    agent,
                    function_name="skill_view",
                    function_result=result,
                )
                assert trusted_skill_operation_block_message(
                    agent,
                    function_name="terminal",
                    function_args={
                        "command": "python3 camera_connector.py list"
                    },
                ) is None
            finally:
                clear_turn_vars(turn_tokens)
    finally:
        skills_tool_module.reset_skill_view_dedup(task_id)

    assert len(set(attestations)) == 2


def test_camsnap_skill_cannot_activate_for_unrelated_task_flow(
    tmp_path, monkeypatch
):
    presets_dir = tmp_path / "presets"
    camera_dir = presets_dir / "skills" / "camsnap"
    camera_dir.mkdir(parents=True)
    camera_bytes = b"# trusted camsnap skill\n"
    (camera_dir / "SKILL.md").write_bytes(camera_bytes)
    _write_presets_integrity_manifest(
        presets_dir,
        skill_bytes=camera_bytes,
        monkeypatch=monkeypatch,
    )
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets_dir))
    snapshot = response_mode._capture_trusted_presets_snapshot()
    assert snapshot is not None
    monkeypatch.setattr(response_mode, "_TRUSTED_PRESETS_SNAPSHOT", snapshot)
    monkeypatch.setattr(skills_tool_module, "SKILLS_DIR", presets_dir / "skills")
    turn_tokens = set_turn_vars(turn_id="unrelated-turn")
    try:
        agent = _FakeAgent()
        agent.platform = "zet_agent"
        reset_trusted_skill_execution(agent, "总结今天的会议")
        result = skills_tool_module.skill_view("camsnap")
        assert not apply_trusted_skill_execution(
            agent,
            function_name="skill_view",
            function_result=result,
        )
        assert trusted_skill_allowed_tool_names(agent) == frozenset()
    finally:
        clear_turn_vars(turn_tokens)


@pytest.mark.parametrize(
    "token",
    ["", "has space", "line\nfeed", "\x7fdelete", "x" * 4097],
)
def test_camera_action_token_rejects_empty_control_or_oversized_values(token):
    assert not response_mode._is_opaque_action_token(token)


def test_camera_action_token_accepts_opaque_utf8_value():
    assert response_mode._is_opaque_action_token("profile-token:相机/v2")


def test_pre_llm_hook_receives_execution_origin_and_kanban_marker(monkeypatch):
    agent = _FakeAgent()
    agent._user_id = "transport-user"
    agent._user_id_alt = "canonical-user"
    agent._memory_write_origin = "background_review"
    agent._zet_agent_execution_policy = "silent_automation"
    agent.request_overrides = {"response_format": {"type": "json_schema"}}
    agent._supports_followup_turns = False
    agent._creation_action_receipt_transport = "canonical_final_v1"
    agent.stream_delta_callback = lambda _delta: None
    captured = {}

    def invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            captured.update(kwargs)
        return []

    monkeypatch.setenv("HERMES_KANBAN_TASK", "KAN-123")
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)
    _build(agent)

    assert captured["platform"] == "cli"
    assert captured["api_mode"] == "chat_completions"
    assert captured["sender_id"] == "canonical-user"
    assert captured["execution_origin"] == "background_review"
    assert captured["execution_policy"] == "silent_automation"
    assert captured["is_kanban_worker"] is True
    assert captured["structured_output"] is True
    assert captured["supports_followup_turns"] is False
    assert captured["streaming_output"] is True
    assert (
        captured["creation_action_receipt_transport"] == "canonical_final_v1"
    )


def test_creation_governor_pre_hook_uses_stable_gateway_conversation_scope(monkeypatch):
    agent = _FakeAgent()
    agent._gateway_session_key = "stable-app-conversation"
    captured = {}

    def invoke_hook(name, **kwargs):
        if name == "pre_llm_call":
            captured.update(kwargs)
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", invoke_hook)

    _build(agent)

    assert captured["conversation_session_id"] == "stable-app-conversation"
    assert captured["session_id"] == "sess-1"


def test_persist_user_message_becomes_original():
    agent = _FakeAgent()
    ctx = _build(agent, user_message="api-prefixed", persist_user_message="clean")
    # original_user_message tracks the clean persist override.
    assert ctx.original_user_message == "clean"
    # but the appended user turn carries the full (sanitized) message.
    assert ctx.messages[-1]["content"] == "api-prefixed"






def test_pending_cli_message_uses_clean_override_for_api_local_note():
    """A noted API message reuses the clean staged dict and its DB marker."""
    agent = _FakeAgent()
    staged = {"role": "user", "content": "clean prompt", "_db_persisted": True}
    agent._pending_cli_user_message = staged

    ctx = _build(
        agent,
        user_message="[MODEL NOTE]\n\nclean prompt",
        persist_user_message="clean prompt",
    )

    assert ctx.messages[-1] is staged
    assert ctx.messages[-1]["content"] == "[MODEL NOTE]\n\nclean prompt"
    assert ctx.messages[-1]["_db_persisted"] is True
    assert agent._pending_cli_user_message is None


def test_runtime_main_sync_happens_after_restore():
    agent = _FakeAgent()
    agent.model = "stale-fallback-model"
    agent.provider = "openai-codex"
    agent.base_url = "https://chatgpt.com/backend-api/codex"
    agent.api_key = "fallback-key"
    agent.api_mode = "codex_responses"

    def restore_primary():
        agent.model = "primary-model"
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"
        agent.api_key = "primary-key"
        agent.api_mode = "anthropic_messages"

    agent._restore_primary_runtime = restore_primary
    calls = []
    with patch(
        "agent.auxiliary_client.set_runtime_main",
        side_effect=lambda *args, **kwargs: calls.append((args, kwargs)),
    ):
        _build(agent)

    assert calls == [
        (
                ("anthropic", "primary-model"),
                {
                    "requested_provider": "openrouter",
                    "base_url": "https://api.anthropic.com",
                "api_key": "primary-key",
                "api_mode": "anthropic_messages",
                "auth_mode": "",
            },
        )
    ]






def test_ensure_db_session_runs_after_system_prompt_restore():
    """Regression for #45499.

    On a fresh API/gateway agent (``_cached_system_prompt is None``) the DB
    session row must be created AFTER the system prompt is restored/built, so
    the persisted snapshot is written non-NULL. If ``_ensure_db_session()``
    ran first it would insert ``system_prompt=NULL`` and trip the misleading
    "stored system prompt is null; rebuilding" warning plus a first-turn
    prefix cache miss.
    """
    agent = _FakeAgent()
    agent._cached_system_prompt = None  # fresh agent, no cached prompt yet

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"

    _build(agent, restore_or_build_system_prompt=_restore)

    # The prompt was populated before the DB row was created.
    assert agent._ensure_db_prompt_at_call == "REBUILT-SYSTEM"
    assert agent._cached_system_prompt == "REBUILT-SYSTEM"


def test_pending_system_prompt_is_persisted_after_session_row_creation():
    agent = _FakeAgent()
    agent._cached_system_prompt = None
    agent._session_db = MagicMock()
    agent._session_db.update_system_prompt.return_value = True

    def _restore(_agent, _system_message, _history):
        _agent._cached_system_prompt = "REBUILT-SYSTEM"
        _agent._system_prompt_persist_pending = True

    _build(agent, restore_or_build_system_prompt=_restore)

    agent._session_db.update_system_prompt.assert_called_once_with(
        agent.session_id, "REBUILT-SYSTEM"
    )
    assert agent._system_prompt_persist_pending is False


# ── Between-turns MCP refresh (cache-safe late-binding) ──────────────────────
#
# A slow MCP server that connects after the agent's build-time tool snapshot
# must become callable by the user's NEXT turn — without mutating an in-flight
# turn's cached request prefix. The prologue is exactly that boundary, so the
# refresh hook lives here. These assert the contract (R1/R2/R6 in the spec),
# not timing permutations.


def test_between_turns_refresh_adds_late_tool_when_servers_registered():
    """R1: a tool that registered since build lands in this turn's snapshot."""
    agent = _FakeAgent()

    new_def = {"type": "function", "function": {"name": "mcp_x_tool", "description": "", "parameters": {}}}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(model_tools, "get_tool_definitions", return_value=[new_def]):
        _build(agent)

    assert "mcp_x_tool" in agent.valid_tool_names
    assert any(t["function"]["name"] == "mcp_x_tool" for t in agent.tools)


def test_between_turns_refresh_requests_same_turn_snapshot_reuse():
    """The request-scoped fast path is enabled only by the turn prologue."""
    agent = _FakeAgent()

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch("tools.mcp_tool.refresh_agent_mcp_tools") as refresh:
        _build(agent)

    refresh.assert_called_once_with(
        agent,
        quiet_mode=True,
        reuse_current_turn_snapshot=True,
    )


def test_reused_zet_shell_refreshes_live_gates_with_no_registered_mcp_tools():
    """A last-tool removal or core grant change cannot preserve old tools."""
    agent = _FakeAgent()
    agent._zet_runtime_shell_force_tool_refresh = True

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=False), \
         patch("tools.mcp_tool.refresh_agent_mcp_tools") as refresh:
        _build(agent)

    refresh.assert_called_once_with(
        agent,
        quiet_mode=True,
        reuse_current_turn_snapshot=True,
    )
    assert agent._zet_runtime_shell_force_tool_refresh is False


def test_reused_zet_shell_tool_refresh_failure_exposes_no_stale_tools():
    agent = _FakeAgent()
    agent.tools = [
        {"type": "function", "function": {"name": "revoked_write"}}
    ]
    agent.valid_tool_names = {"revoked_write"}
    agent._zet_runtime_shell_force_tool_refresh = True

    with patch(
        "tools.mcp_tool.refresh_agent_mcp_tools",
        side_effect=RuntimeError("live authorization unavailable"),
    ):
        _build(agent)

    assert agent.tools == []
    assert agent.valid_tool_names == set()
    assert agent._tools_disabled_for_request is True






def test_api_server_tool_choice_none_skips_registered_mcp_refresh(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    class FakeAgent(_FakeAgent):
        def __init__(self, **_kwargs):
            super().__init__()
            self.platform = "api_server"
            self.tools = [
                {"type": "function", "function": {"name": "present_plan"}},
                {"type": "function", "function": {"name": "todo"}},
            ]
            self.valid_tool_names = {"present_plan", "todo"}

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openai",
            "base_url": "https://example.test/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "gpt-5")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda: {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    agent = adapter._create_agent(
        session_id="meeting-summary",
        request_overrides={"tool_choice": "none"},
    )

    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch("tools.mcp_tool.refresh_agent_mcp_tools") as refresh:
        _build(agent)

    refresh.assert_not_called()
    assert agent.tools == []
    assert agent.valid_tool_names == set()


def test_between_turns_refresh_no_churn_when_unchanged():
    """R2: an unchanged tool set leaves the snapshot object identity intact
    (no needless swap → nothing for the next request prefix to diff against)."""
    agent = _FakeAgent()
    same = [{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}]
    agent.tools = same
    agent.valid_tool_names = {"a"}

    import model_tools
    with patch("tools.mcp_tool.has_registered_mcp_tools", return_value=True), \
         patch.object(
             model_tools, "get_tool_definitions",
             return_value=[{"type": "function", "function": {"name": "a", "description": "", "parameters": {}}}],
         ):
        _build(agent)

    assert agent.tools is same  # not replaced → no churn






def test_expired_cooldown_allows_preflight(tmp_path):
    agent = _make_agent_with_cooldown(
        tmp_path / "state.db",
        "sess-1",
        cooldown_until=1.0,
    )

    with patch("agent.turn_context._should_run_preflight_estimate", return_value=True), \
         patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        ctx = _build(agent)

    assert isinstance(ctx, TurnContext)
    agent._emit_status.assert_called_once()
    agent._compress_context.assert_called()
