"""⛔ 不许把内部执行状态播报进【回答正文】—— 提示词层约束 + 结构性闭集。

═══════════════════════════════════════════════════════════════════════
缺陷(用户真机截图,2026-08-17)
═══════════════════════════════════════════════════════════════════════
Telegram 会话里,模型回复正文带着::

    Progress
    ・✕ 👁 请识别并描述这张图片中的内容,回答用户"这是?" (failed)
    ———
    图片这次还是没有传到我这里,[图片] 只是占位符…

违反全局规则「⛔ 别堆用户文案 —— 这是产品不是 demo」里被点名的反模式:
**把「我实现了什么」写给用户看:进度步骤播报**。
判据「这句话不出现,用户会不会做错事 / 卡住 / 不知道下一步?」——**不会**。

🔴 **归属**:那段文本**是模型自己写的**,⛔ 不是任何一处代码拼的。
证据链(每条都带阳性对照):
  · 全仓 ``git grep -F "・"`` = 0、``"———"`` = 0;而同法 ``✕`` / ``👁``
    **有命中** ⇒ 量具有效,⛔ 不是假 0;
  · Hermes 里唯一把进度投递到 IM 的机制是 ``gateway/run.py`` 的
    ``progress_queue``,内容只有 ``💬 <思考原文>``(默认 off,设备未配置)
    或工具行 ``<emoji> <label>`` —— **没有 Progress 表头 / ・ / (failed)**;
  · ⚠️ 「设备日志里 Progress 全 0」这条**不作为证据** —— 阳性对照证明
    ``agent.log`` / ``gateway.log`` / ``errors.log`` **都不记录模型回复正文**
    (我自己制造并被模型答对的 ``ZEBRA-QUASAR-8815-VERIFY`` 在四个日志文件里
    全 0)。⇒ 那个 0 是**量具盲区**,⛔ 不能当证据。
⇒ 唯一能改的地方是**提示词层**。

═══════════════════════════════════════════════════════════════════════
⭐ 必须保持不变的行为(⛔ 先列这份,不许只列「要改什么」)
═══════════════════════════════════════════════════════════════════════
1. **失败时仍然给一句可行动的话** —— ⚠️ 「闭嘴」比旁白更坏。
   截图里那句「图片这次还是没有传到我这里…请重新上传」**一个字都不许砍**。
2. **错误提示的分类与可行动性** —— ⛔ 不许把不同根因压成一句「操作失败」。
3. **破坏性 / 不可逆动作的确认文案**。
4. **可访问性文本**。
⇒ 约束文本里必须显式写着这四条,本文件逐条钉死(``test_must_not_cut_*``)。

═══════════════════════════════════════════════════════════════════════
本门关住 / 仍开集(⭐ 分格声明)
═══════════════════════════════════════════════════════════════════════
**关住(确定性)**
  A. 约束**无条件**进入组装出的 system prompt —— 关掉所有可选开关、无工具、
     任意 platform/model 都在(⛔ 不许像 TASK_COMPLETION_GUIDANCE 那样可配置关掉);
  B. 「不许砍」的四格在约束文本里逐条存在;
  C. **结构性闭集**:Hermes 自己⛔ 不再有代码把 Progress 式区块拼进出站正文。

**⛔ 仍是开集(明说,⛔ 不假装钉住了)**
  提示词只能**要求**模型照做,⛔ 保证不了它照做。
  「用户最终看到的文本形状」这一层**只能真机观测** —— 已在云机上做四情形
  A/B,结果记在 ``~/Desktop/Test/DEPLOY-IMMEDIA-2026-08-17.md``;
  样本量小,⛔ 不许当成「已证明」。
"""

from __future__ import annotations

import ast
import pathlib
import re
from types import SimpleNamespace

import pytest

from agent.prompt_builder import USER_FACING_NARRATION_GUIDANCE as GUIDANCE
from agent.system_prompt import build_system_prompt_parts

_REPO = pathlib.Path(__file__).resolve().parents[2]

#: 约束的指纹 —— 用它在组装结果里定位,⛔ 不用整段比对(整段会随措辞漂移)。
_FINGERPRINT = "Your reply is a product surface, not a work log."


def _agent(**overrides):
    """把每一个可选开关都关到最小 —— ⭐ 判据是「关成这样它还在不在」。"""
    base = dict(
        load_soul_identity=False,
        skip_context_files=True,
        valid_tool_names=[],          # ← TASK_COMPLETION_GUIDANCE 会因此缺席
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable(**overrides) -> str:
    return build_system_prompt_parts(_agent(**overrides))["stable"]


# ═════════ A. 无条件、跨渠道跨模型 ═════════


def test_present_even_with_every_optional_switch_off() -> None:
    """⭐ 最小配置下仍在 —— 且**对照组**证明这个最小配置真的关掉了别的块。"""
    stable = _stable()
    assert _FINGERPRINT in stable, (
        "约束在最小配置下缺席 ⇒ 它被挂到了某个可选开关后面,一关就漏,"
        "而漏出去的正是给用户看的那一面。"
    )
    # 阳性对照:同一份最小配置里,可配置的那块**确实**不在
    # ⇒ 证明「最小配置」真的把东西关掉了,⛔ 不是配置没生效。
    from agent.prompt_builder import TASK_COMPLETION_GUIDANCE

    assert TASK_COMPLETION_GUIDANCE not in stable, (
        "对照组失效:连可配置的 TASK_COMPLETION_GUIDANCE 都还在 ⇒ "
        "「最小配置」没真的关掉任何东西,上面那条断言什么都没证明。"
    )


@pytest.mark.parametrize(
    "platform", ["", "telegram", "feishu", "wecom", "zet_agent", "slack", "discord"]
)
def test_present_on_every_channel(platform: str) -> None:
    assert _FINGERPRINT in _stable(platform=platform), (
        f"{platform}: 约束缺席 ⇒ 这条渠道上用户还会看到进度播报"
    )


@pytest.mark.parametrize(
    "model,provider",
    [("", ""), ("lite", "custom"), ("gpt-4o", "openai"), ("claude-opus-5", "anthropic")],
)
def test_present_for_every_model_family(model: str, provider: str) -> None:
    assert _FINGERPRINT in _stable(model=model, provider=provider)


# ═════════ B. 「不许砍」的四格逐条钉死 ═════════


class TestMustNotCutTheseFourCells:
    """⭐ 作用域**刚好等于**「进度旁白」这一格 —— 砍大了就是新缺陷。"""

    def test_failure_must_still_say_something_actionable(self) -> None:
        low = GUIDANCE.lower()
        assert "not permission to go quiet" in low, (
            "缺少「⛔ 不是让你失败时闭嘴」—— 静默比旁白更坏,"
            "截图里那句『图片没传到我这里…』正是必须留下的"
        )
        assert "silence is worse" in low
        for action in ("retry", "re-upload", "grant access", "rephrase"):
            assert action in low, f"没给出可行动的动词示例: {action}"

    def test_error_classification_stays(self) -> None:
        low = GUIDANCE.lower()
        assert "specific and actionable" in low
        assert "do not collapse distinct causes" in low, (
            "缺少「⛔ 不许把不同根因压成一句通用失败」—— "
            "这正是全局硬规则里点名的反模式"
        )

    def test_destructive_confirmations_stay(self) -> None:
        assert "destructive or irreversible" in GUIDANCE.lower()

    def test_accessibility_text_stays(self) -> None:
        assert "accessibility text stays" in GUIDANCE.lower()

    def test_unsolicited_internal_progress_is_still_forbidden(self) -> None:
        """原缺陷必须仍被挡住:未经请求的 Progress/内部状态不能回正文。"""
        low = GUIDANCE.lower()
        for target in ("did not ask for", '"progress" section',
                       "internal task/subtask names", "status markers"):
            assert target in low, f"没点名原缺陷: {target}"
        for state in ("queued", "running", "failed", "completed"):
            assert state in low, f"原截图里的状态形态没有被禁止: {state}"

    def test_user_requested_steps_tools_and_reports_are_explicitly_allowed(self) -> None:
        low = GUIDANCE.lower()
        for allowed in ("user explicitly asks", "tutorial", "plan", "step list",
                        "command or tool list", "final verification report"):
            assert allowed in low, f"用户明确请求的正常内容仍未被放行: {allowed}"
        for overbroad in ("no progress or step lists", "no tool names"):
            assert overbroad not in low, f"宽禁令仍在: {overbroad}"


# ═════════ C. 结构性闭集:Hermes 自己⛔ 不再拼进度块进出站正文 ═════════


def test_no_hermes_code_composes_a_progress_block_into_reply_text() -> None:
    """⭐ 这一条与提示词无关 —— 它防的是**将来有人**在代码里加回来。

    判据:全仓⛔ 不许出现「``Progress`` 表头字面量」与「``(failed)`` 状态后缀」
    的组合式拼装。⚠️ 这是**开集判据的一个特例**:它罩不住别的写法,
    但它罩得住**这一次这个形状**,并且在有人复制粘贴时会红。
    """
    offenders: list[str] = []
    pat_header = re.compile(r'["\']\s*Progress\s*(\\n|["\'])')
    for p in sorted(_REPO.glob("gateway/**/*.py")) + sorted(_REPO.glob("agent/**/*.py")):
        try:
            src = p.read_text()
        except Exception:
            continue
        for i, ln in enumerate(src.splitlines(), 1):
            if pat_header.search(ln):
                offenders.append(f"{p.relative_to(_REPO)}:{i}  {ln.strip()[:90]}")
    assert not offenders, (
        "有人在出站路径上拼了一个 `Progress` 表头 —— 那正是本轮要消除的形状:\n  "
        + "\n  ".join(offenders)
    )


def test_the_only_im_progress_channel_is_still_the_separate_bubble() -> None:
    """⭐ 「砍掉后 App/PC 的进度还看不看得到」的结构性回答。

    IM 侧进度走的是**独立气泡**(``progress_queue`` → ``_send_progress_text``),
    与回答正文是两条消息;App/PC 走的是结构化事件(``plan_seeding`` / 计划卡)。
    ⇒ 约束模型别在**正文**里播报,⛔ 不影响这两条既有通道。
    本条钉住那两条通道仍然存在 —— 若哪天没了,这条推理的前提就变了。
    """
    run_src = (_REPO / "gateway/run.py").read_text()
    assert "_send_progress_text" in run_src and "progress_queue" in run_src, (
        "IM 的独立进度气泡通道没了 ⇒ 「砍正文不影响进度」这个前提不再成立"
    )
    assert (_REPO / "agent/plan_seeding.py").exists(), (
        "App 侧计划卡播种没了 ⇒ 同上,必须重新判定"
    )


def test_guidance_is_appended_unconditionally_in_source() -> None:
    """⛔ 源码层面也不许把它挪到 if 里(⭐ 与 A 组互为独立判据)。"""
    tree = ast.parse((_REPO / "agent/system_prompt.py").read_text())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "append"):
            continue
        for a in node.args:
            if isinstance(a, ast.Name) and a.id == "USER_FACING_NARRATION_GUIDANCE":
                found.append(node)
    assert len(found) == 1, f"约束的 append 点不是恰好 1 个: {len(found)}"

    call = found[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.Try)):
            for child in ast.walk(node):
                if child is call:
                    pytest.fail(
                        "约束被挂进了 if/try ⇒ 某些配置下会缺席,"
                        "而缺席的正是给用户看的那一面"
                    )
