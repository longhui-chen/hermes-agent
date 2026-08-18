"""profile 维度的两条「半条链」:代次没被消费 · 锁没按 profile 派生。

两条都是**本 PR 自己造的**:
  · ``_profile_generations`` + ``cache_generation()`` 是本 PR 新增,
    但 ``model_tools`` 这个消费者没跟上 ⇒ ``/reload-mcp`` 后**新工具不生效**。
  · ``_current_mcp_profile_identity`` 也是本 PR 新增,是它让 per-profile
    discovery 成为真实路径,而锁仍然是**首个 profile** 的标量
    ⇒ 两个进程锁不同文件 ⇒ **同时拉起同一组 MCP 子进程**。

⭐ 判据落在**行为**上(key 变没变 / 锁路径解析成什么),⛔ 不是「源码里有没有这个词」。
"""

import os
import pathlib

import pytest


# ═════════ ① 具名 profile 的 MCP 变更必须让工具定义缓存失效 ═════════

class TestProfileGenerationReachesTheToolDefsCache:
    def test_a_named_profile_bump_changes_the_cache_key(self, monkeypatch):
        """🔴 ``_bump_generation(profile)`` **不动**全局 ``_generation``
        ⇒ 只看 ``_generation`` 的 key 一动不动 ⇒ 缓存永不失效。"""
        import model_tools
        from tools.registry import registry

        # ⚠️ ``_bump_generation`` 会 ``.expanduser().resolve()``(macOS 上
        # ``/tmp`` → ``/private/tmp``)⇒ 夹具的身份必须走**同一套规范化**,
        # 否则两边查的根本不是同一个 key,门会假红。
        home = str(pathlib.Path("/tmp/hermes-profile-A").expanduser().resolve())
        monkeypatch.setattr(
            registry, "_current_profile_identity", lambda: home, raising=False
        )

        before_global = registry._generation
        key_before, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        assert key_before is not None, "缓存被旁路了 ⇒ 这条断言什么也没验到"

        registry._bump_generation(home)

        assert registry._generation == before_global, (
            "前提变了:具名 profile 现在也会动全局代次 —— 请重判这条门是否还需要"
        )
        key_after, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        assert key_after != key_before, (
            "profile 的 MCP 变更没有改变缓存键 ⇒ /reload-mcp 后 Agent 仍拿到旧工具表,"
            "得等一次无关的全局注册或**进程重启**才恢复"
        )

    def test_an_unchanged_registry_still_hits_the_cache(self, monkeypatch):
        """🔴 **必须保持不变**:什么都没动时 key 必须稳定,⛔ 不许每次都 miss。"""
        import model_tools
        from tools.registry import registry

        # ⚠️ ``_bump_generation`` 会 ``.expanduser().resolve()``(macOS 上
        # ``/tmp`` → ``/private/tmp``)⇒ 夹具的身份必须走**同一套规范化**,
        # 否则两边查的根本不是同一个 key,门会假红。
        home = str(pathlib.Path("/tmp/hermes-profile-A").expanduser().resolve())
        monkeypatch.setattr(
            registry, "_current_profile_identity", lambda: home, raising=False
        )
        k1, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        k2, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        assert k1 == k2 and k1 is not None, "key 不稳定 ⇒ 缓存彻底失效,每次都重算 schema"

    def test_a_global_bump_still_changes_the_key(self, monkeypatch):
        """🔴 **必须保持不变**:原本靠全局代次失效的那一半不许丢。"""
        import model_tools
        from tools.registry import registry

        monkeypatch.setattr(
            registry, "_current_profile_identity", lambda: "/tmp/hermes-profile-A",
            raising=False,
        )
        k1, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        registry._bump_generation(None)
        k2, _ = model_tools._lookup_scoped_tool_defs_cache(None, None, False)
        assert k1 != k2, "全局注册不再让缓存失效 ⇒ 修复把原本对的东西弄坏了"


# ═════════ ② discovery 锁必须按 profile 派生 ═════════

class TestDiscoveryLockIsProfileScoped:
    @pytest.fixture(autouse=True)
    def _clear(self):
        import tools.mcp_tool as mcp_tool

        mcp_tool._MCP_DISCOVERY_LOCK_PATHS.clear()
        yield
        mcp_tool._MCP_DISCOVERY_LOCK_PATHS.clear()

    @staticmethod
    def _resolved_path_for(monkeypatch, identity, tmp_path):
        """跑真实解析路径,只把 ``open`` 换成记录器 —— ⛔ 不复制实现。"""
        import tools.mcp_tool as mcp_tool

        seen = []

        def _fake_open(path, *a, **kw):
            seen.append(path)
            return open(tmp_path / "sink", "w", encoding="utf-8")

        monkeypatch.setattr(mcp_tool, "open", _fake_open, raising=False)
        monkeypatch.setattr(mcp_tool, "_lock_file_exclusive", lambda fh: True, raising=False)
        mcp_tool._try_acquire_mcp_discovery_lock(identity)
        return seen

    def test_two_profiles_resolve_to_two_different_lock_files(self, monkeypatch, tmp_path):
        a, b = str(tmp_path / "home-a"), str(tmp_path / "home-b")
        seen_a = self._resolved_path_for(monkeypatch, a, tmp_path)
        seen_b = self._resolved_path_for(monkeypatch, b, tmp_path)
        assert seen_a and seen_b, "没走到 open ⇒ 这条断言什么也没验到"
        assert seen_a[0] != seen_b[0], (
            "第二个 profile 复用了第一个的锁文件 ⇒ 与服务同一 profile 的 CLI/TUI "
            "进程互斥不了 ⇒ 同时拉起同一组 MCP 子进程(凭据并发使用 + 重复进程)"
        )
        assert seen_b[0] == os.path.join(b, ".mcp-discovery.lock")

    def test_the_default_profile_path_is_byte_identical_to_before(self, monkeypatch, tmp_path):
        """🔴 **必须保持不变**:默认 profile 解析出的文件与旧版逐字相同。"""
        import tools.mcp_tool as mcp_tool

        home = tmp_path / "hermes-home"
        monkeypatch.setattr(
            mcp_tool, "_current_mcp_profile_identity",
            lambda: os.path.normcase(os.path.abspath(str(home))),
        )
        seen = self._resolved_path_for(monkeypatch, None, tmp_path)
        assert seen[0] == os.path.join(
            os.path.normcase(os.path.abspath(str(home))), ".mcp-discovery.lock"
        ), "默认路径变了 ⇒ 与既有进程互斥不上,等于把锁悄悄关了"

    def test_the_same_profile_reuses_its_cached_path(self, monkeypatch, tmp_path):
        """缓存仍然有效(每次重算路径不是缺陷,但白费;这里钉住 map 真的在用)。"""
        import tools.mcp_tool as mcp_tool

        a = str(tmp_path / "home-a")
        self._resolved_path_for(monkeypatch, a, tmp_path)
        assert mcp_tool._MCP_DISCOVERY_LOCK_PATHS.get(a) == os.path.join(
            a, ".mcp-discovery.lock"
        )
