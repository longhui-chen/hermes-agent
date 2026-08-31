"""入站媒体占位文本里的路径，必须是【模型运行环境能读到的】那一条。

族 A（文件/附件给不到 Agent）在 Hermes 侧的一格：``media_urls`` 存的是
**宿主机**缓存路径。local 后端下 gateway 与模型同文件系统，原样给就能读；
docker 后端下这些目录是 bind mount 进容器的，宿主路径在容器里**不存在**
⇒ 模型拿到一条读不到的路径。

🔴 **本文件在「可读性契约接线」后重写过一次。**
接线前这些用例喂的是**不存在的假路径**（``/host/.hermes/cache/images/a.jpg``），
只验证「字符串被翻译了」。接线后 ``_build_media_placeholder`` 要求先拿到
``open + regular-file + read`` 的回执 —— 假路径当然拿不到，三条门全红。

⭐ 那不是功能被弄坏，是**这些门原来的前提太弱**：它们能在「文件根本不存在」
的情况下绿，而那恰恰是族 A 的现场。⇒ 改成真实临时文件，
**原来钉的两个性质（local 逐字不变 / docker 翻译）一条不少地保留**。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.run import _build_media_placeholder as _build_media_placeholder_async


def _build_media_placeholder(event):
    """⚠️ 签名**有意**改成 async(探测不许在事件循环上同步等待)。

    这里只改**调用方式**,下面每一条断言**逐字不变** —— 同样的输入
    必须得到同样的输出。
    """
    import asyncio

    return asyncio.run(_build_media_placeholder_async(event))


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """把 Hermes cache 根指到 tmp，并造出真实文件。"""
    import hermes_constants

    root = tmp_path / "hermes"
    for sub in ("cache/images", "cache/audio", "cache/videos", "cache/documents"):
        (root / sub).mkdir(parents=True)

    monkeypatch.setattr(
        hermes_constants, "get_hermes_dir",
        lambda new_subpath, old_name=None: root / new_subpath)
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return root


def _make(cache_root, rel: str, data: bytes = b"payload") -> str:
    p = cache_root / rel
    p.write_bytes(data)
    return str(p)


def _event(paths, types):
    return SimpleNamespace(media_urls=list(paths), media_types=list(types))


def test_local_backend_paths_are_unchanged(cache):
    """⛔ 正常功能不许弄坏：local 后端下输出与从前逐字节相同。"""
    src = _make(cache, "cache/images/a.jpg")
    out = _build_media_placeholder(_event([src], ["image/jpeg"]))
    assert out == f"[User sent an image: {src}]"


def test_docker_backend_path_is_translated_for_the_model(cache, monkeypatch):
    """🔴 docker 后端：给模型的必须是**容器内**路径，⛔ 不是宿主路径。"""
    src = _make(cache, "cache/images/a.jpg")
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    import tools.credential_files as cf

    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": "/root/.hermes/cache/images/a.jpg")

    out = _build_media_placeholder(_event([src], ["image/jpeg"]))
    assert src not in out, f"宿主路径原样进了给模型的文本 —— 容器里读不到它:{out}"
    assert "/root/.hermes/cache/images/a.jpg" in out, f"没有翻译成容器内路径:{out}"


def test_every_media_kind_goes_through_translation(cache, monkeypatch):
    """四类媒体（图/音/视/文件）都要翻译，⛔ 不许只修 image 那一支。"""
    monkeypatch.setenv("TERMINAL_ENV", "docker")

    import tools.credential_files as cf

    paths = [
        _make(cache, "cache/images/a.jpg"),
        _make(cache, "cache/audio/b.ogg"),
        _make(cache, "cache/videos/c.mp4"),
        _make(cache, "cache/documents/d.pdf"),
    ]
    mapping = {p: f"/root/.hermes/slot{i}" for i, p in enumerate(paths)}
    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": mapping.get(host))

    types = ["image/jpeg", "audio/ogg", "video/mp4", "application/pdf"]
    out = _build_media_placeholder(_event(paths, types))

    assert out.count("/root/.hermes/slot") == 4, f"四类媒体里有分支没走翻译:\n{out}"
    for host in paths:
        assert host not in out, f"未翻译的宿主路径漏进文本:{host}\n{out}"


# ───────────── 接线本身：⛔ 没有回执就不许把路径给模型 ─────────────

def test_unreadable_artifact_never_reaches_the_model(cache):
    """🔴 族 A 本体：路径读不到时，⛔ 不许把它写进提示。

    原先这里只做字符串翻译就拼进去 —— 翻译只回答「按声明的 mount 该映射成
    什么」，⛔ 不回答「模型那边真的打得开吗」。路径不存在时模型只会自己编，
    而链路上没有任何一层报错。
    """
    missing = str(cache / "cache/documents/gone.pdf")   # ⛔ 故意不创建
    out = _build_media_placeholder(_event([missing], ["application/pdf"]))

    assert missing not in out, (
        f"读不到的路径仍被塞给了模型 —— 它会假装读过:{out}")
    assert "could not be read" in out, f"没告诉模型这里本来有个附件:{out}"
    assert "attachment_expired" in out, f"没给出失败分类:{out}"


def test_unreadable_artifact_message_leaks_nothing(cache):
    """⛔ 给模型的那句话不许含绝对路径 / 原始 error。"""
    missing = str(cache / "cache/documents/secret-payroll.pdf")
    out = _build_media_placeholder(_event([missing], ["application/pdf"]))
    assert "secret-payroll" not in out and missing not in out, f"路径泄漏:{out}"
    assert "Errno" not in out and "Traceback" not in out, f"原始错误泄漏:{out}"


def test_readable_and_unreadable_are_reported_separately(cache):
    """一条能读、一条不能读 ⇒ 各报各的，⛔ 不许一坏全丢。

    ⭐ 孪生枚举：只测「全成功」和「全失败」会漏掉混合这一档，
    而混合恰恰是最容易写成「有一个失败就整批放弃」的地方。
    """
    good = _make(cache, "cache/images/ok.jpg")
    bad = str(cache / "cache/images/missing.jpg")

    out = _build_media_placeholder(_event([good, bad], ["image/jpeg", "image/jpeg"]))

    assert f"[User sent an image: {good}]" in out, f"能读的那条也被丢了:{out}"
    assert "could not be read" in out, f"读不到的那条没有留痕:{out}"
    assert bad not in out


def test_no_media_means_no_verification(cache):
    """⛔ 无附件的会话不许触发任何验证 —— 空事件必须原样返回空串。"""
    assert _build_media_placeholder(_event([], [])) == ""
