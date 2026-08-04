"""合规拦截绝不能走 fallback，也不能因为 error.type 没被读到而漏判。

两条都是 Codex 在 PR #257 上指出的 P1，两条的后果是同一个：合规网关已经拒绝
的请求被路由到用户自己配置的备用模型，并且可能真的拿到回答。
"""

import agent.error_classifier as ec
from agent.error_classifier import FailoverReason, classify_api_error


class _StubError(Exception):
    """带 body 的上游错误，形状照抄 OpenAI SDK 的 APIStatusError。"""

    def __init__(self, message: str, body: dict, status_code: int = 400):
        super().__init__(message)
        self.body = body
        self.status_code = status_code
        self.response = None


def _classify(body, message="request failed", status=400):
    return classify_api_error(_StubError(message, body, status))


def test_moderation_type_is_recognised_even_when_code_is_generic():
    """error.type 必须单独看。

    _extract_error_code 读的是 ``code or type``：一个真值但通用的 code（CN 网关
    某些路径返回 "400"）会短路 or，随后又被 != "400" 守卫丢掉，type 再也没机会
    被读到。结果这一帧落进 format_error / provider_bad_request，客户端拿不到
    content_blocked，fallback 也拦不住。
    """
    classified = _classify({
        "error": {
            "code": "400",
            "type": "content_policy_violation",
            "message": "内容不合规",
        }
    })
    assert classified.reason == FailoverReason.content_policy_blocked


def test_moderation_code_alone_still_recognised():
    """反向：code 里带机器码、message 是本地化句子的老形状不能被改坏。"""
    classified = _classify({
        "error": {"code": "moderation_input_blocked", "message": "内容不合规"}
    })
    assert classified.reason == FailoverReason.content_policy_blocked


def test_no_fallback_flag_is_honoured(monkeypatch):
    """开关打开时，分类结果必须明确说「不要 fallback」。"""
    monkeypatch.setattr(ec, "content_policy_fallback_disabled", lambda: True)
    classified = _classify({
        "error": {"code": "moderation_input_blocked", "message": "内容不合规"}
    })
    assert classified.reason == FailoverReason.content_policy_blocked
    assert classified.should_fallback is False


def test_fallback_still_allowed_when_switch_is_off(monkeypatch):
    """反向钉住：海外设备没开这个开关时，供应商安全过滤仍然可以换模型重试。

    没有这条，上面那条可以靠「永远不 fallback」通过。
    """
    monkeypatch.setattr(ec, "content_policy_fallback_disabled", lambda: False)
    classified = _classify({
        "error": {"code": "moderation_input_blocked", "message": "内容不合规"}
    })
    assert classified.should_fallback is True


# ── 被拒的一轮不得留在会话历史里 ────────────────────────────────────

from agent.conversation_loop import (  # noqa: E402
    _purge_refused_rows_from_session_db,
    _transcript_without_refused_turn,
)


class _FakeSessionDB:
    def __init__(self):
        self.deleted = []

    def delete_message(self, session_id, message_id):
        self.deleted.append((session_id, message_id))
        return True


class _FakeAgent:
    def __init__(self, db):
        self._session_db = db
        self.session_id = "s1"


def test_refused_rows_are_deleted_from_the_session_db():
    """内存里裁掉不够 —— DB 行是在首次模型调用之前就写好的。

    不删行的话，下一轮 get_messages_as_conversation() 会把被拒内容恢复回来
    重新提交给审核网关（按次计费）和模型，用户会被反复拦在同一段他已经看不到
    改法的文字上。
    """
    db = _FakeSessionDB()
    messages = [
        {"role": "user", "content": "hello", "_db_message_id": 1},
        {"role": "assistant", "content": "hi", "_db_message_id": 2},
        {"role": "user", "content": "违规内容", "_db_message_id": 3},
    ]
    kept = _transcript_without_refused_turn(messages, "违规内容", 2)
    assert len(kept) == 2

    _purge_refused_rows_from_session_db(_FakeAgent(db), messages, len(kept))
    assert db.deleted == [("s1", 3)]


def test_purge_is_best_effort_when_the_store_cannot_delete():
    """老 store 没有 delete_message 时不能把一次内容拒绝变成崩溃的 turn。"""

    class _NoDelete:
        pass

    _purge_refused_rows_from_session_db(
        _FakeAgent(_NoDelete()), [{"role": "user", "_db_message_id": 7}], 0
    )


def test_stale_index_pointing_at_another_user_row_is_reanchored():
    """压缩重建后，旧索引可能仍是一条 user 行，但不是本轮那条。

    只校验 role 就裁剪，会把被拒内容留在返回的 transcript 里。
    """
    messages = [
        {"role": "user", "content": "违规内容"},          # 本轮真正被拒的
        {"role": "assistant", "content": "..."},
        {"role": "user", "content": "todo snapshot"},   # 压缩追加，排在它之后
    ]
    # 索引指向 2：确实是一条 user 行（只校验 role 会直接采信），但那是压缩
    # 追加的快照，不是本轮。按它裁剪 → messages[:2]，被拒内容原样留下。
    kept = _transcript_without_refused_turn(messages, "违规内容", 2)
    assert all(m["content"] != "违规内容" for m in kept), (
        "被拒内容仍留在 transcript 里，下一轮会被重新提交"
    )


# ── reanchor 的 no-exact fallback 不能被采信 ──────────────────────────

def test_reanchor_fallback_to_a_synthetic_user_row_is_rejected():
    """reanchor 找不到 exact match 时会退回**最后一条** user 行。

    压缩的 merge-summary-into-tail 会改写当前 user 的 content，而 todo store 又
    可能在尾部追加一条 synthetic user —— 于是那个 fallback 指向的是 snapshot，
    排在被拒消息之后。按它裁剪就把被拒内容留下了，下一轮继续提交给审核网关和
    模型；_purge_refused_rows_from_session_db 也只会从 snapshot 开始删。

    正确做法是：锚点内容对不上就当没找到，走更保守的兜底（宁可多裁一轮）。
    """
    # 被拒的一轮拿不到助手回复，所以尾部是连续两条 user：被改写的本轮 + snapshot。
    messages = [
        {"role": "user", "content": "早先的正常提问"},
        {"role": "assistant", "content": "正常回答"},
        {"role": "user", "content": "被压缩改写过的违规内容"},   # 本轮，content 已被改写
        {"role": "user", "content": "[todo snapshot]"},          # 压缩追加的 synthetic
    ]
    kept = _transcript_without_refused_turn(messages, "原始违规内容", 0)
    assert all("违规" not in str(m.get("content", "")) for m in kept), (
        "被拒内容仍留在 transcript 里 —— reanchor 的 fallback 被错误采信了"
    )


def test_exact_match_reanchor_is_still_trusted():
    """反向：内容对得上时仍然按它裁，别把这条也改保守了。"""
    messages = [
        {"role": "user", "content": "早先的正常提问"},
        {"role": "assistant", "content": "正常回答"},
        {"role": "user", "content": "违规内容"},
    ]
    kept = _transcript_without_refused_turn(messages, "违规内容", 99)  # 索引越界，强制 reanchor
    assert [m["content"] for m in kept] == ["早先的正常提问", "正常回答"]
