"""凡有原生**批量**图片能力的 adapter，必须也有原生**单张**能力（RH 第五轮 P1）。

现场：kanban 交付物从批量改成逐张投递后，Email 因为只覆写了
``send_multiple_images``、没有 ``send_image_file``，落到基类兜底 ——
那条只发一句「native image send unavailable」的**文本**，而且发送成功
⇒ 调用方把它记成投递成功。用户既没拿到图，也不会出现在失败汇总里。

⭐ 这是个**闭集**判据：覆写者名单可以从类继承关系直接算出来，
⛔ 不是「我想到了 Email」。以后任何 adapter 加了 batch 覆写却忘了单张，
这条门当场红。
"""
from __future__ import annotations

import importlib
import pkgutil

import pytest

from gateway.platforms.base import BasePlatformAdapter


#: 允许 import 失败的模块（缺可选依赖）。⛔ 闭集 —— 新增一个必须显式登记，
#: 并说明为什么它可以逃过能力检查。当前为空:本环境 92 个模块全部可导入。
_IMPORT_EXEMPT: set[str] = set()


def _import_all_adapter_modules() -> dict[str, str]:
    """import 全部 adapter 模块，返回 {模块名: 失败原因}。"""
    import gateway.platforms
    import plugins.platforms

    failures: dict[str, str] = {}
    for pkg in (gateway.platforms, plugins.platforms):
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            if mod.name.endswith(("._base", ".helpers")):
                continue
            try:
                importlib.import_module(mod.name)
            except Exception as exc:  # noqa: BLE001
                # 🔴 ⛔ 不许静默吞掉:上一版在这里 `continue`，于是
                # **让某个 adapter import 失败就能让它逃过本门** ——
                # RH 实测:删掉 Email 单张能力时门会红，再让 Email import
                # 失败，同一个真实漏洞变成 `1 passed`。⭐ 假闭集。
                # ⇒ 记下来，由下面的门断言「失败集合 ⊆ 显式豁免集」。
                failures[mod.name] = f"{type(exc).__name__}: {exc}"
    return failures
def _all_adapter_classes():
    """收集 BasePlatformAdapter 的全部子类（模块须先 import）。"""
    seen, out = set(), []
    stack = [BasePlatformAdapter]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            out.append(sub)
            stack.append(sub)
    return out


def test_every_adapter_module_imports():
    """🔴 先证明**扫描范围完整** —— ⛔ import 失败不许静默变成"豁免"。

    ⭐ 这条是上面那条能力门的**前提**:范围不完整时，能力门对没扫到的
    adapter 恒绿。前提要单独断言，⛔ 不能藏在 try/except 里。
    """
    failures = _import_all_adapter_modules()
    unexpected = {k: v for k, v in failures.items() if k not in _IMPORT_EXEMPT}
    assert not unexpected, (
        "这些 adapter 模块 import 失败 ⇒ 它们逃过了能力检查：\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in sorted(unexpected.items()))
        + "\n若确实缺可选依赖，把模块名加进 _IMPORT_EXEMPT 并写明理由。")


def test_batch_image_capability_implies_single_image_capability():
    _import_all_adapter_modules()
    offenders = []
    checked = 0
    for cls in _all_adapter_classes():
        has_batch = "send_multiple_images" in cls.__dict__
        if not has_batch:
            continue
        checked += 1
        # 单张能力可以来自自己或任一非 base 的祖先
        has_single = any(
            "send_image_file" in k.__dict__
            for k in cls.__mro__ if k is not BasePlatformAdapter
        )
        if not has_single:
            offenders.append(f"{cls.__module__}.{cls.__name__}")

    # ⭐ 阳性对照：⛔ 一个都没扫到说明 import 全挂了，那不是"通过"
    assert checked >= 5, (
        f"只扫到 {checked} 个有 batch 能力的 adapter —— 扫描本身可能坏了")
    assert not offenders, (
        "这些 adapter 有原生批量图片能力、却没有单张能力 ⇒ 逐张投递会落到\n"
        "基类兜底（只发一句文本提示，还被记成投递成功）：\n  "
        + "\n  ".join(offenders))
