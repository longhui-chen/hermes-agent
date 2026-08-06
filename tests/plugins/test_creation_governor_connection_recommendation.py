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


def _artifact_candidate():
    return {
        "decision": "artifact",
        "suggested_name": "华东销售季报页面",
        "reason": "这份分析整理成可打开的页面后可以反复查看和分享。",
        "evidence_turn_ids": ["evidence-1"],
        "confidence": 0.8,
        "dedup_key": "east-sales-quarterly",
        "proposal_text": "要做成一个页面吗？",
    }


def test_artifact_candidate_emits_artifact_recommendation(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_artifact_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    transformed = _drive_turn(plugin, "s-artifact", "帮我把这季度华东销售数据整理分析一下")

    assert transformed is None  # 不进文本信封
    assert len(ctx.emitted) == 1
    attachment = ctx.emitted[0]
    assert attachment["kind"] == "artifact.recommendation"
    assert attachment["payload"]["title"] == "华东销售季报页面"
    assert attachment["payload"]["reason"].startswith("这份分析")
    assert [a["id"] for a in attachment["actions"]] == ["dismiss", "accept"]
    assert attachment["dedup_key"] == "artifact:east-sales-quarterly"


def test_artifact_dismiss_latches(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_artifact_candidate(), _artifact_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))
    monkeypatch.setattr(plugin, "PROMPT_COOLDOWN_TURNS", -1)

    _drive_turn(plugin, "s-art-dismiss", "帮我把这季度华东销售数据整理分析一下")
    assert len(ctx.emitted) == 1
    plugin._on_attachment_action(
        session_id="s-art-dismiss",
        attachment_id=ctx.emitted[0]["id"],
        action_id="dismiss",
        action_token="tok-a",
        turn_id="",
        payload=None,
        profile_name="default",
    )
    _drive_turn(plugin, "s-art-dismiss", "这个分析结果再帮我看一眼")
    assert len(ctx.emitted) == 1


# ---------------------------------------------------------------------------
# 区域感知库存（available_kinds）+ 出卡文案衔接（同轮/后续轮上下文注入）
# ---------------------------------------------------------------------------


def _install_fake_channels_tool(monkeypatch, payload):
    import sys
    import types

    fake = types.ModuleType("tools.list_my_channels_tool")
    fake._check_list_my_channels = lambda: True
    fake.list_my_channels_tool = lambda args, **kw: json.dumps(payload, ensure_ascii=False)
    monkeypatch.setitem(sys.modules, "tools.list_my_channels_tool", fake)
    # connector 工具一并桩为不可用，隔离本组用例
    fake_conn = types.ModuleType("tools.list_my_connectors_tool")
    fake_conn._check_list_my_connectors = lambda: False
    fake_conn.list_my_connectors_tool = lambda args, **kw: json.dumps({"connectors": []})
    monkeypatch.setitem(sys.modules, "tools.list_my_connectors_tool", fake_conn)


def test_fetch_inventory_intersects_region_available_kinds(monkeypatch):
    """新版 local-server 回传 available_kinds（区域感知）：推荐范围 = 可连 ∩ 白名单。
    CN 设备场景：telegram 在白名单但不在区域可连清单 → 不得进入 recommendable。"""
    plugin = _load_plugin()
    _install_fake_channels_tool(
        monkeypatch,
        {
            "installed_channels": [
                {"kind": "feishu", "name": "飞书", "status": "online", "target_ref": "channel:feishu"}
            ],
            "available_kinds": ["wecom", "wechat", "dingtalk"],
        },
    )
    inventory = plugin._fetch_connection_inventory()
    assert inventory["fetched"] is True
    assert inventory["channels_connected"] == ["feishu"]
    # dingtalk 不在平台白名单、telegram/discord/slack 不在区域可连清单 —— 都被排除
    assert inventory["channels_recommendable"] == ["wechat", "wecom"]


def test_fetch_inventory_legacy_server_without_available_kinds(monkeypatch):
    """老版 local-server 无 available_kinds 字段：降级回「白名单 − 已连」旧公式。"""
    plugin = _load_plugin()
    _install_fake_channels_tool(
        monkeypatch,
        {
            "installed_channels": [
                {"kind": "feishu", "name": "飞书", "status": "online", "target_ref": "channel:feishu"}
            ]
        },
    )
    inventory = plugin._fetch_connection_inventory()
    assert inventory["fetched"] is True
    assert inventory["channels_recommendable"] == sorted(
        plugin.RECOMMENDABLE_CHANNEL_KINDS - {"feishu"}
    )


def test_channel_proposal_injects_same_turn_card_context(monkeypatch):
    """出卡当轮：主模型必须被告知「回复下方会附加连接卡」，禁止编造设置路径。"""
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    injected = plugin._on_pre_llm_call(
        session_id="s-ctx",
        turn_id="turn-1",
        user_message="数据日报出来后我经常在电脑前错过",
        conversation_history=[],
    )
    assert injected is not None
    context = injected["context"]
    assert "connect card" in context
    assert "telegram" in context
    assert "invent settings paths" in context


def test_channel_proposal_followup_context_points_to_card(monkeypatch):
    """后续轮 carry context：连接类不得引导走 Hermes 原生创建流程，必须指向已出的卡。"""
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    _drive_turn(plugin, "s-carry", "数据日报出来后我经常在电脑前错过")
    followup = plugin._on_pre_llm_call(
        session_id="s-carry",
        turn_id="turn-2",
        user_message="好的",
        conversation_history=[],
    )
    assert followup is not None
    context = followup["context"]
    assert "Connect button" in context
    assert "native creation flow" not in context


def test_creation_type_followup_context_keeps_native_flow(monkeypatch):
    """回归护栏：agent/skill/task 的后续轮话术保持原样（仍走原生创建流程）。"""
    plugin = _load_plugin()
    creation_candidate = {
        "decision": "task",
        "suggested_name": "每周销售汇总",
        "reason": "重复出现的整理需求。",
        "evidence_turn_ids": ["evidence-1"],
        "confidence": 0.8,
        "dedup_key": "task-weekly-sales",
        "proposal_text": "要沉淀成 Task 吗？",
    }
    ctx = _Context(_FakeLlm([creation_candidate]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    _drive_turn(plugin, "s-task-carry", "帮我整理这周的销售数据")
    followup = plugin._on_pre_llm_call(
        session_id="s-task-carry",
        turn_id="turn-2",
        user_message="好的",
        conversation_history=[],
    )
    assert followup is not None
    assert "native creation flow" in followup["context"]


def test_availability_context_grounds_main_model_every_turn(monkeypatch):
    """主模型口径接地：评估轮注入 [channel-availability]（含可连清单），
    非评估轮复用会话缓存继续注入；库存未取到则完全不注入。"""
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([{"decision": "none"}]))
    plugin.register(ctx)
    inventory = dict(_INVENTORY) | {"channels_available": ["feishu", "wechat", "wecom"]}
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(inventory))

    first = plugin._on_pre_llm_call(
        session_id="s-ground", turn_id="turn-1", user_message="随便聊聊", conversation_history=[]
    )
    assert "[channel-availability]" in first["context"]
    assert "feishu, wechat, wecom" in first["context"]
    assert "never suggest" in first["context"]

    second = plugin._on_pre_llm_call(
        session_id="s-ground", turn_id="turn-2", user_message="继续", conversation_history=[]
    )
    assert "[channel-availability]" in second["context"]


def test_availability_context_absent_when_inventory_unfetched(monkeypatch):
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([{"decision": "none"}]))
    plugin.register(ctx)
    monkeypatch.setattr(
        plugin,
        "_fetch_connection_inventory",
        lambda: {"fetched": False, "channels_connected": [], "channels_available": [],
                 "channels_recommendable": [], "connectors_connected": [],
                 "connectors_recommendable": []},
    )
    result = plugin._on_pre_llm_call(
        session_id="s-noground", turn_id="turn-1", user_message="随便聊聊", conversation_history=[]
    )
    assert "[channel-availability]" not in (result or {}).get("context", "")


def test_empty_target_falls_back_to_suggested_name_in_pool(monkeypatch):
    """真机实测：flash 档检测器把渠道 kind 填进 suggested_name 而漏掉 target。
    仅当 suggested_name 逐字命中库存池时回退采用；不命中仍拒绝。"""
    plugin = _load_plugin()
    hit = _connection_candidate() | {"target": "", "suggested_name": "telegram"}
    miss = _connection_candidate() | {"target": "", "suggested_name": "微信通知直达"}
    ctx = _Context(_FakeLlm([hit, miss]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))
    monkeypatch.setattr(plugin, "PROMPT_COOLDOWN_TURNS", -1)

    _drive_turn(plugin, "s-fallback-hit", "日报结果发到我手机才有用")
    assert len(ctx.emitted) == 1
    assert ctx.emitted[0]["payload"] == {"channel_kind": "telegram"}
    assert ctx.emitted[0]["dedup_key"] == "channel:telegram"

    _drive_turn(plugin, "s-fallback-miss", "日报结果发到我手机才有用")
    assert len(ctx.emitted) == 1  # 第二个候选仍被硬闸拒绝


def test_channel_card_emitted_at_pre_llm_stage(monkeypatch):
    """采集即发射（真机体验修复）：判定完成即出卡，不等 turn 收尾——用户不用
    等完整回复生成（Pro 模型 20s+）才看到卡；transform 钩子不重复发射。"""
    plugin = _load_plugin()
    ctx = _Context(_FakeLlm([_connection_candidate()]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    plugin._on_pre_llm_call(
        session_id="s-early",
        turn_id="turn-1",
        user_message="日报结果发到我手机才有用",
        conversation_history=[],
    )
    assert len(ctx.emitted) == 1
    assert ctx.emitted[0]["kind"] == "channel.connect"

    transformed = plugin._transform_llm_output(session_id="s-early", response_text="好的，已安排。")
    assert transformed is None
    assert len(ctx.emitted) == 1


def test_connection_candidate_missing_name_falls_back_to_target(monkeypatch):
    """flash 检测器强调 target 后偶发漏填 suggested_name：连接类用 target 兜底，
    不拒掉合法推荐（标题由客户端 i18n 渲染，此字段仅展示面冗余）。"""
    plugin = _load_plugin()
    candidate = _connection_candidate() | {"suggested_name": ""}
    ctx = _Context(_FakeLlm([candidate]))
    plugin.register(ctx)
    monkeypatch.setattr(plugin, "_fetch_connection_inventory", lambda: dict(_INVENTORY))

    plugin._on_pre_llm_call(
        session_id="s-noname",
        turn_id="turn-1",
        user_message="日报结果发到我手机才有用",
        conversation_history=[],
    )
    assert len(ctx.emitted) == 1
    assert ctx.emitted[0]["payload"] == {"channel_kind": "telegram"}
