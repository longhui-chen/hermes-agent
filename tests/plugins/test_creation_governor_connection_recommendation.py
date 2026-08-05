"""Connection recommendations (channel/connector) — Phase 3a coverage.

覆盖需求 2.2 / 5.2 的策略硬约束：
* 目标必须逐字命中真实库存的 recommendable 集合（事后过滤，模型说了不算）；
* 库存缺失 → 该轮禁止连接类推荐；
* 交付走 ctx.emit_attachment（channel.connect / connector.connect 卡），不进文本信封；
* dismiss 回执（attachment_action hook）落 30 天闩锁，同目标不再推荐；
* 无发射通道时静默降级，正文不受影响。
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


PLUGIN_PATH = (
    Path(__file__).resolve().parents[2]
    / "plugins"
    / "creation-governor"
    / "__init__.py"
)


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "creation_governor_plugin_connection", PLUGIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._reset_state_for_tests()
    return module


class _FakeLlm:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def complete(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        parsed = self.results.pop(0)
        return SimpleNamespace(text=json.dumps(parsed))


class _Context:
    def __init__(self, llm=None, *, with_emitter=True):
        self.llm = llm
        self.tools = []
        self.hooks = []
        self.auxiliary_tasks = []
        self.emitted = []
        if with_emitter:
            self.emit_attachment = self._emit  # type: ignore[assignment]

    def _emit(self, attachment):
        self.emitted.append(attachment)
        return True

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_hook(self, *args, **kwargs):
        self.hooks.append((args, kwargs))

    def register_auxiliary_task(self, **kwargs):
        self.auxiliary_tasks.append(kwargs)


_INVENTORY = {
    "fetched": True,
    "channels_connected": ["feishu"],
    "channels_recommendable": ["telegram", "discord"],
    "connectors_connected": ["notion"],
    "connectors_recommendable": ["gmail", "github"],
}


def _connection_candidate(*, decision="channel", target="telegram"):
    return {
        "decision": decision,
        "target": target,
        "suggested_name": "每日提醒直达 Telegram" if decision == "channel" else "连接 Gmail",
        "reason": "提醒需要直达用户的 IM 才有长期价值。",
        "evidence_turn_ids": ["evidence-1"],
        "confidence": 0.8,
        "dedup_key": "model-made-up-key",
        "proposal_text": "要连接吗？",
    }


def _drive_turn(plugin, session_id, message, response="好的，已安排。"):
    plugin._on_pre_llm_call(
        session_id=session_id,
        turn_id="turn-1",
        user_message=message,
        conversation_history=[],
    )
    return plugin._transform_llm_output(
        session_id=session_id, response_text=response
    )


def test_channel_candidate_emits_channel_connect_attachment(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    transformed = _drive_turn(plugin, "s-channel", "数据日报出来后我经常在电脑前错过")

    # 正文不追加信封（连接推荐不进文本通道）
    assert transformed is None
    assert len(ctx.emitted) == 1
    attachment = ctx.emitted[0]
    assert attachment["kind"] == "channel.connect"
    assert attachment["state"] == "active"
    assert attachment["payload"] == {"channel_kind": "telegram"}
    assert [a["id"] for a in attachment["actions"]] == ["dismiss", "connect"]
    assert attachment["dedup_key"] == "channel:telegram"
    assert attachment["id"].startswith("cg-")
    assert attachment["expires_at"] > 0
    # 评估请求里带了真实库存行（不由模型猜测）
    evidence = ctx.llm.calls[0][0][1]["content"]
    assert "[connection-inventory]" in evidence
    assert "telegram" in evidence


def test_connector_candidate_emits_connector_connect_attachment(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate(decision="connector", target="gmail")]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    transformed = _drive_turn(plugin, "s-connector", "帮我整理最近的邮件待办")

    assert transformed is None
    assert len(ctx.emitted) == 1
    attachment = ctx.emitted[0]
    assert attachment["kind"] == "connector.connect"
    assert attachment["payload"] == {"provider": "gmail", "blocking": False}
    assert attachment["dedup_key"] == "connector:gmail"


def test_target_outside_inventory_is_rejected(monkeypatch):
    plugin = _load_plugin()
    # feishu 已连接、wechat 不在 recommendable —— 两种都必须拒绝
    ctx = _Context(_FakeLlm([
        _connection_candidate(target="feishu"),
        _connection_candidate(target="whatsapp"),
    ]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    assert _drive_turn(plugin, "s-connected", "日报结果发到我手机才有用") is None
    assert _drive_turn(plugin, "s-unsupported", "日报结果发到我手机才有用") is None
    assert ctx.emitted == []


def test_unfetched_inventory_forbids_connection_decisions(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(
        plugin,
        "_fetch_connection_inventory",
        lambda: {"fetched": False, "channels_connected": [], "channels_recommendable": [],
                 "connectors_connected": [], "connectors_recommendable": []},
    )

    assert _drive_turn(plugin, "s-nofetch", "日报结果发到我手机才有用") is None
    assert ctx.emitted == []
    evidence = ctx.llm.calls[0][0][1]["content"]
    assert "forbidden" in evidence


def test_dismiss_action_latches_target_for_the_session(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([
        _connection_candidate(),
        _connection_candidate(),
    ]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))
    # 冷却窗口不影响本测试：直接放开
    monkeypatch.setattr(plugin, "PROMPT_COOLDOWN_TURNS", -1)

    _drive_turn(plugin, "s-dismiss", "日报结果发到我手机才有用")
    assert len(ctx.emitted) == 1
    attachment_id = ctx.emitted[0]["id"]

    # 模拟 zet_agent 的 attachment_action 入站派发
    plugin._on_attachment_action(
        session_id="s-dismiss",
        attachment_id=attachment_id,
        action_id="dismiss",
        action_token="tok-1",
        turn_id="",
        payload=None,
        profile_name="default",
    )

    # 同目标再次成为候选 → 已闩锁，不再发射
    _drive_turn(plugin, "s-dismiss", "上次说的日报直达手机那事再看看")
    assert len(ctx.emitted) == 1


def test_missing_emitter_degrades_silently(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]), with_emitter=False)
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    # 无 emit_attachment 能力 → 不炸、不改正文、不出信封
    assert _drive_turn(plugin, "s-noemit", "日报结果发到我手机才有用") is None


def test_attachment_action_hook_registered():
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([]))
    plugin.register(ctx)
    hook_names = [args[0] for args, _ in ctx.hooks]
    assert "attachment_action" in hook_names
