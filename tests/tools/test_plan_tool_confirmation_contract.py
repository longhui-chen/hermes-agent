"""ZET-3140：无卡片平台的计划确认，工具结果不许夹带面向用户的英文话术。

现场：飞书 Bot 的中文会话里，计划确认蹦出英文。

根因（四条分支实查后的判别式，⛔ 不是"看起来像"）：
`present_plan_with_meta` 有四条返回分支，只有「无 callback + 非 auto_execute」
这一条附的是**第一人称、面向用户的话术**（"…before **I** proceed."，还带
`"go"` / `"confirm"` 这种示例回复词）。模型极可能原样复述给用户。
另外三条附的是**祈使式内部指令**（"start carrying out…"、"Do NOT rebuild…"），
模型不会念给用户，⛔ 不在本轮作用域内。

⛔ 本文件**不声称**"改完模型就会用中文回" —— 那是模型行为，测试证明不了。
用我们自己写的桩去"证明模型会复述英文"是循环论证：桩吐什么由我们决定。
下面四条判据测的全是**我们能控制的东西**：工具结果里有什么、没有什么。
用户可见语言是否真的改善，归真机验证收口。
"""

from __future__ import annotations

import pytest

from tools.plan_tool import (
    PLAN_PRESENTED_RESULT,
    present_plan,
    present_plan_with_meta,
)

_TITLE = "部署计划"
_GROUPS = [
    {
        "icon": "🔧",
        "label": "准备",
        "count": 2,
        "items": ["检查现有配置", "备份凭据目录"],
    },
    {"icon": "🚀", "label": "Rollout", "count": 1, "items": ["Deploy to staging"]},
]

# 判据①的检测集：面向用户的自然语言话术特征。
# ⚠️ 自然语言无法闭集检测 —— 这里是**开集**，只钉住已知的那几种形态。
# 兜底靠判据③（其余分支逐字不变）和 review，⛔ 不假装这一条罩全了。
_USER_FACING_PHRASES = (
    "before I proceed",   # 第一人称承诺
    "I proceed",
    "I'll proceed",
    'e.g. "go"',          # 示例回复词
    '"confirm")',
    "reply to confirm",   # 直接对用户说话
)


def _no_callback_non_auto() -> str:
    return present_plan(_TITLE, _GROUPS, callback=None, auto_execute=False)


def test_criterion_1_no_user_facing_english_script_in_tool_result():
    """判据①：④ 分支的工具结果不含面向用户的自然语言话术。"""
    result = _no_callback_non_auto()
    assert result, "校准失败：④ 分支返回空，下面的断言会变成空转"
    leaked = [phrase for phrase in _USER_FACING_PHRASES if phrase in result]
    assert not leaked, (
        f"④ 分支仍在工具结果里夹带面向用户的英文话术 {leaked}；"
        "模型会把它原样复述给用户。这里只能放给模型看的指令。"
    )


def test_criterion_2_title_and_groups_are_preserved_verbatim():
    """判据②：任何语言的 title / groups 原文逐字保留，⛔ 不许被改写。"""
    result = _no_callback_non_auto()
    assert _TITLE in result, "中文 title 被改写或丢失"
    for group in _GROUPS:
        assert group["label"] in result, f"group label 丢失：{group['label']}"
        for item in group["items"]:
            assert item in result, f"计划条目丢失：{item}"


def test_criterion_3_the_other_three_branches_are_byte_for_byte_unchanged():
    """判据③：另外三条分支逐字不变 —— 本轮作用域只含 ④。"""
    called: list = []

    def _callback(*args):
        called.append(args)

    # ② callback + 非 auto
    result_2, meta_2 = present_plan_with_meta(
        title=_TITLE, groups=_GROUPS, callback=_callback
    )
    assert result_2 == PLAN_PRESENTED_RESULT
    # 校准：callback 是延迟的（包在 meta["emit"] 闭包里，由 plan_seeding 在
    # 落盘成功后才调）。⛔ 不能断言 present_plan 直接调过它 —— 那会把"分支
    # 没走到"和"设计如此"混为一谈。这里显式触发一次来证明确实进了该分支。
    assert meta_2 is not None, "校准失败：②没有进 callback 分支（meta 为 None）"
    meta_2["emit"]()

    # ① callback + auto_execute
    branch_1 = present_plan(_TITLE, _GROUPS, callback=_callback, auto_execute=True)
    assert branch_1 == (
        "Plan presented to the user's App as a read-only card. "
        "Auto-execute is enabled: start carrying out the plan now in "
        "this same turn. Do NOT ask the user to confirm or say you are "
        "waiting — just proceed. A task list has already been created "
        "from this plan (see the todo result below); update item "
        "statuses with the `todo` tool using merge=true as you work. "
        "Do NOT rebuild the list from scratch."
    )

    # ③ 无 callback + auto_execute
    branch_3 = present_plan(_TITLE, _GROUPS, callback=None, auto_execute=True)
    assert branch_3.endswith(
        "\n\nAuto-execute is enabled and there is no user available to "
        "confirm: start carrying out the plan now in this same turn. Do "
        "NOT wait for a reply — just proceed."
    )
    assert called, "校准失败：callback 从未被调用，①② 两条根本没走到"


def test_criterion_4_model_is_still_told_a_user_confirmation_is_required():
    """判据④ 🔴：砍话术不许连"需要用户确认"的语义一起砍掉。

    原文那一句同时承担两件事：面向用户的英文话术（要砍）+ 告诉模型"这里必须
    等用户确认"（⛔ 绝不能砍）。砍过头的后果是计划不等确认就自己往下执行，
    比返回英文严重得多。改动作用域必须**刚好等于**「英文用户话术」。
    """
    result = _no_callback_non_auto()
    lowered = result.lower()
    assert "confirm" in lowered, (
        "④ 分支不再告诉模型需要用户确认 —— 计划可能不等确认就被执行"
    )
    assert any(
        marker in lowered for marker in ("do not", "don't", "before")
    ), "④ 分支缺少「确认前不要执行」的约束语义"


@pytest.mark.parametrize("auto_execute", [False, True])
def test_plan_text_body_is_shared_by_both_no_callback_branches(auto_execute):
    """③④ 共用同一份计划正文；改 ④ 的后缀不许动到正文渲染。"""
    result = present_plan(_TITLE, _GROUPS, callback=None, auto_execute=auto_execute)
    assert result.startswith(f"📋 {_TITLE}")
