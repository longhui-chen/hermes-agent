"""内容拦截的错误码必须传播到**所有**客户端形状，被拒轮次不得回到 Responses 链。

Codex 在 PR #257 上报的两条 P1。
"""

from gateway.platforms.api_server import APIServerAdapter, _hermes_error_code


REFUSAL = {
    "completed": False,
    "failed": True,
    "messages": [],
    "error": "content_policy_blocked: 内容不合规",
    "provider_error": {"code": "content_blocked", "message": "内容不合规"},
}


def test_non_streaming_and_finish_chunk_carry_the_real_code():
    """非流式响应和标准 SSE finish chunk 原来硬编码 agent_error。

    只消费标准 OpenAI 形状的客户端（不吃 Hermes 私有的 __hermes_error__ 事件）
    因此看不到 content_blocked，同一次审核拦截在它们那里降级成通用错误，用户
    拿不到合规提示。
    """
    assert _hermes_error_code(REFUSAL, "error") == "content_blocked"


def test_truncation_still_wins():
    """反向：截断仍然是 output_truncated，别把这条也改歪了。"""
    assert _hermes_error_code(REFUSAL, "length") == "output_truncated"


def test_generic_failure_falls_back_to_agent_error():
    """没有 provider_error 时仍然是通用码。"""
    assert _hermes_error_code({"failed": True, "error": "boom"}, "error") == "agent_error"


def test_refused_first_turn_is_not_resynthesised_into_the_responses_chain():
    """被拒首轮不得回到存储的 Responses transcript。

    被拒时 conversation_loop 把这一轮从 result["messages"] 裁掉，首轮被拒就裁成
    空列表 —— 而空列表原来被当成「旧调用形状」，于是 current_user 又被合成回去。
    之后 previous_response_id 从 response store 读回这段历史，被拒文本在链式会话
    里被重新提交给审核网关和模型。
    """
    history = APIServerAdapter._build_response_conversation_history(
        [], "违规内容", REFUSAL, "",
    )
    assert all(m.get("content") != "违规内容" for m in history), (
        "被拒内容仍留在 Responses transcript 里，下一轮会被重新提交"
    )


def test_refused_turn_keeps_prior_history_intact():
    """只丢被拒的那一轮，之前的对话不能跟着丢。"""
    prior = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
    history = APIServerAdapter._build_response_conversation_history(
        list(prior), "违规内容", REFUSAL, "",
    )
    assert history == prior


def test_normal_turn_is_still_stored():
    """反向：正常一轮仍然要把 user + assistant 存进去。"""
    ok = {"completed": True, "messages": []}
    history = APIServerAdapter._build_response_conversation_history([], "你好", ok, "你好呀")
    assert [m["content"] for m in history] == ["你好", "你好呀"]
