"""MCP tool timeout 的类型/域校验（RH 复审 P2-3）。

我上一版写的是 ``isinstance(timeout, (int, float))``，并在注释里声称
「照抄同文件 ``_parse_boolish``」。**那个声称不成立**：``_parse_boolish``
判的是布尔，没有数值域的概念，照它写出来的判据漏掉三类，全部可复现：

  · ``bool`` 是 ``int`` 的**子类** ⇒ ``True`` 通过，超时静静变成 1 秒
  · 负数通过 ⇒ ``asyncio.wait_for(-1)`` 立即超时，每次调用都失败
  · ``NaN`` / ``inf`` 通过 ⇒ NaN 的比较**恒为 False**，永远不超时，
    阻塞的 coroutine 再也收不回来（RH 实测：外部 3 秒仍不收敛）

仓内真正的先例是同文件 ``_safe_numeric``（:1375）。
⭐ 「照抄 X」是一条**可验证的断言**，⛔ 不是修辞 —— 说了就要能逐条对上。
"""
from __future__ import annotations

import math

import pytest

from tools.mcp_tool import (
    _DEFAULT_TOOL_TIMEOUT,
    _MIN_TOOL_TIMEOUT,
    _safe_optional_timeout,
)


def test_none_still_means_no_timeout():
    """⛔ 不许弄坏原来对的：``None`` 是合法的「不设超时」。"""
    assert _safe_optional_timeout(None, _DEFAULT_TOOL_TIMEOUT) is None


@pytest.mark.parametrize("good", [1, 30, 300.0, 0.5, "45"])
def test_usable_values_pass_through(good):
    """正常值（含 YAML 来的字符串、亚秒）必须原样可用。"""
    out = _safe_optional_timeout(good, _DEFAULT_TOOL_TIMEOUT)
    assert out == pytest.approx(float(good))


@pytest.mark.parametrize(
    "bad,why",
    [
        (True, "bool 是 int 子类 ⇒ 会被当成 1 秒"),
        (False, "bool 是 int 子类 ⇒ 会被当成 0 秒（立即超时）"),
        (-1, "负数 ⇒ wait_for 立即超时，每次调用都失败"),
        (0, "0 ⇒ 立即超时"),
        (float("nan"), "NaN 比较恒 False ⇒ 永远不超时，coroutine 收不回来"),
        (float("inf"), "inf ⇒ 实际上等于不超时，但配置写的不是 None"),
        (float("-inf"), "-inf ⇒ 立即超时"),
        ("abc", "非数字字符串"),
        (object(), "任意对象（属性替身返回的东西）"),
    ],
)
def test_unusable_values_fall_back_to_default(bad, why):
    out = _safe_optional_timeout(bad, _DEFAULT_TOOL_TIMEOUT)
    assert out == _DEFAULT_TOOL_TIMEOUT, f"{why}；实际={out!r}"


def test_result_is_always_finite_and_positive_or_none():
    """闭集兜底：⛔ 不管喂进来什么，出来的只能是 None 或可用的正有限数。

    ⭐ 判据不是「我列到的那几种坏值」——那是开集，给没列到的发免检。
    """
    for value in [None, True, False, -1, 0, 1, 0.5, "3", "x", object(),
                  float("nan"), float("inf"), float("-inf"), [], {}, 10**400]:
        out = _safe_optional_timeout(value, _DEFAULT_TOOL_TIMEOUT)
        assert out is None or (
            isinstance(out, (int, float))
            and math.isfinite(out)
            and out >= _MIN_TOOL_TIMEOUT
        ), f"{value!r} 产出了不可用的 timeout: {out!r}"


def test_sub_second_configs_are_not_silently_raised():
    """⛔ 下限不许照抄 ``_safe_numeric`` 默认的 1 —— 那会改掉既有亚秒配置。

    ⭐ 「修一个缺陷却弄坏原来对的东西」：这里只需要挡住 <= 0。
    """
    assert _safe_optional_timeout(0.25, _DEFAULT_TOOL_TIMEOUT) == pytest.approx(0.25)
