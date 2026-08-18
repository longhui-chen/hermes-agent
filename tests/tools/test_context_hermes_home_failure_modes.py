"""``_inject_context_hermes_home`` 的三种失效形态,一条一个测试函数。

⚠️ 这里**曾经**是一个 ``except Exception: pass``,把三件性质完全不同的事压成同一个
"静默通过",其中最毒的一件是连 ``from hermes_constants import`` 的 ImportError 也一起
吞掉 ⇒ 打包/部署一出问题,这个 pin **永久静默失效、全路径、全时间**,而日志上一切正常。

⭐ 为什么一条断言一个函数(而不是塞进一个测试):三个分支挤在一起时,任何一次红都要靠
``grep -c`` 去猜"红在谁身上"——第一版正是这样,逆改红了却红在**另一条**断言上。
拆开之后,任何一次红**必然**红在那个函数自己的断言上,判据不需要猜。
"""

from __future__ import annotations

import sys
import types

import pytest


# ① 没有 pin ⇒ 静默 no-op,⛔ 不许记 ERROR。
# 绝大多数正常路径(单 profile)走这里,记日志会把真信号淹掉。
def test_absent_pin_is_silent(monkeypatch, caplog):
    import hermes_constants
    from tools.environments.local import _inject_context_hermes_home

    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", lambda: None)
    env: dict[str, str] = {}
    with caplog.at_level("ERROR"):
        _inject_context_hermes_home(env)

    assert env == {}, "ABSENT_PIN_MUST_NOT_TOUCH_ENV: " + repr(env)
    assert not caplog.records, (
        "ABSENT_PIN_MUST_NOT_LOG: 无 pin 是正常路径,记 ERROR 会把真信号淹掉 —— "
        + repr([r.message for r in caplog.records])
    )


# ② 机制本身不可用(符号导不进来)⇒ 必须**响亮**:留下可定位的 ERROR 并上抛。
# ⛔ 静默继续 = 带着一个并不存在的安全边界在服务。
#
# ⚠️ 模拟方式刻意精确:往 sys.modules 塞一个**缺该符号**的替身模块,让那一行 import
# 自己失败。⛔ 不 patch builtins.__import__ —— 那会拦到 pytest 自身和任何延迟导入,
# 打击面太大,红了也说不清红在哪。
def test_unavailable_mechanism_is_loud(monkeypatch, caplog):
    from tools.environments.local import _inject_context_hermes_home

    stub = types.ModuleType("hermes_constants")  # 故意不带 get_hermes_home_override
    monkeypatch.setitem(sys.modules, "hermes_constants", stub)

    with caplog.at_level("ERROR"):
        with pytest.raises(ImportError):
            _inject_context_hermes_home({})

    assert any("profile pin unavailable" in r.message for r in caplog.records), (
        "UNAVAILABLE_MECHANISM_MUST_BE_LOUD: 没有留下可定位的 ERROR —— "
        + repr([r.message for r in caplog.records])
    )


# ③ 有 pin 但取用时抛异常 ⇒ **fail closed**:异常上抛、子进程不要起。
# ⛔ 不许接住 —— "明知该指向 A 却指向了 B 的凭据库"比"这次操作失败"严重得多:
# 各 profile 绑的是不同的真人身份。
# ⭐ 爆炸半径很窄:只有 pin 存在(多 profile 会话)才可能触发,单 profile 走 ①。
def test_failing_pin_lookup_fails_closed(monkeypatch):
    import hermes_constants
    from tools.environments.local import _inject_context_hermes_home

    def boom():
        raise RuntimeError("PIN_LOOKUP_EXPLODED")

    monkeypatch.setattr(hermes_constants, "get_hermes_home_override", boom)

    with pytest.raises(RuntimeError, match="PIN_LOOKUP_EXPLODED"):
        _inject_context_hermes_home({})
