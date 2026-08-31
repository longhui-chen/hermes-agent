"""Tests for tools.env_passthrough — skill and config env var passthrough."""

import os
import pytest
import yaml

from agent import secret_scope as ss
import tools.env_passthrough as _ep_mod
from tools.env_passthrough import (
    clear_env_passthrough,
    get_all_passthrough,
    is_env_passthrough,
    register_env_passthrough,
    resolve_passthrough_value,
)


@pytest.fixture(autouse=True)
def _clean_passthrough():
    """Ensure a clean passthrough state for every test."""
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    ss.set_multiplex_active(False)
    yield
    clear_env_passthrough()
    _ep_mod._config_passthrough = None
    ss.set_multiplex_active(False)


class TestSkillScopedPassthrough:
    def test_register_and_check(self):
        assert not is_env_passthrough("TENOR_API_KEY")
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")


    def test_skips_empty(self):
        register_env_passthrough(["", "  ", "VALID_KEY"])
        assert is_env_passthrough("VALID_KEY")
        assert not is_env_passthrough("")


class TestConfigPassthrough:
    def test_reads_from_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["MY_CUSTOM_KEY", "ANOTHER_TOKEN"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough = None

        assert is_env_passthrough("MY_CUSTOM_KEY")
        assert is_env_passthrough("ANOTHER_TOKEN")
        assert not is_env_passthrough("UNRELATED_VAR")


    def test_union_of_skill_and_config(self, tmp_path, monkeypatch):
        config = {"terminal": {"env_passthrough": ["CONFIG_KEY"]}}
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.dump(config), encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _ep_mod._config_passthrough = None

        register_env_passthrough(["SKILL_KEY"])
        all_pt = get_all_passthrough()
        assert "CONFIG_KEY" in all_pt
        assert "SKILL_KEY" in all_pt


class TestProfileScopedResolution:
    def test_active_scope_overrides_process_fallback(self):
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "profile-b"})
        try:
            assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") == "profile-b"
        finally:
            ss.reset_secret_scope(token)

    def test_active_scope_does_not_fall_back_to_another_profile(self):
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") is None
        finally:
            ss.reset_secret_scope(token)

    def test_unscoped_multiplex_read_fails_closed(self):
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            resolve_passthrough_value("SERVICE_TOKEN", "profile-a")

    def test_single_profile_keeps_callers_fallback(self):
        assert resolve_passthrough_value("SERVICE_TOKEN", "profile-a") == "profile-a"

    def test_active_scope_keeps_explicit_global_override(self, monkeypatch):
        """Global terminal settings still honor a caller-provided override."""
        monkeypatch.setenv("TERMINAL_CWD", "/default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            assert resolve_passthrough_value("TERMINAL_CWD", "/explicit") == "/explicit"
        finally:
            ss.reset_secret_scope(token)


class TestExecuteCodeIntegration:
    """Verify that the passthrough is checked in execute_code's env filtering."""

    def test_secret_substring_blocked_by_default(self):
        """TENOR_API_KEY should be blocked without passthrough."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" not in child_env

    def test_passthrough_allows_secret_through(self):
        """TENOR_API_KEY should pass through when registered."""
        _SAFE_ENV_PREFIXES = ("PATH", "HOME", "USER", "LANG", "LC_", "TERM",
                              "TMPDIR", "TMP", "TEMP", "SHELL", "LOGNAME",
                              "XDG_", "PYTHONPATH", "VIRTUAL_ENV", "CONDA")
        _SECRET_SUBSTRINGS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL",
                              "PASSWD", "AUTH")

        register_env_passthrough(["TENOR_API_KEY"])

        test_env = {"PATH": "/usr/bin", "TENOR_API_KEY": "test123", "HOME": "/home/user"}
        child_env = {}
        for k, v in test_env.items():
            if is_env_passthrough(k):
                child_env[k] = v
                continue
            if any(s in k.upper() for s in _SECRET_SUBSTRINGS):
                continue
            if any(k.startswith(p) for p in _SAFE_ENV_PREFIXES):
                child_env[k] = v

        assert "PATH" in child_env
        assert "HOME" in child_env
        assert "TENOR_API_KEY" in child_env
        assert child_env["TENOR_API_KEY"] == "test123"

    def test_execute_code_uses_active_profile_for_passthrough(self, monkeypatch):
        """The execute_code child must receive the routed profile's value."""
        from tools.code_execution_tool import _scrub_child_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            child_env = _scrub_child_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert child_env["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_execute_code_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not leak into the execute_code child."""
        from tools.code_execution_tool import _scrub_child_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            child_env = _scrub_child_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert "SERVICE_TOKEN" not in child_env


class TestTerminalIntegration:
    """Verify that the passthrough is checked in terminal's env sanitizers."""

    def test_background_terminal_uses_active_profile_for_passthrough(self, monkeypatch):
        """Background/PTY terminal children must use the routed profile value."""
        from tools.environments.local import _sanitize_subprocess_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-routed-profile"})
        try:
            child_env = _sanitize_subprocess_env(
                {"SERVICE_TOKEN": "token-for-default"},
                {"SERVICE_TOKEN": "token-for-default"},
            )
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert child_env["SERVICE_TOKEN"] == "token-for-routed-profile"

    def test_background_terminal_omits_missing_scoped_passthrough(self, monkeypatch):
        """A missing routed secret must not leak into background terminal work."""
        from tools.environments.local import _sanitize_subprocess_env

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        token = ss.set_secret_scope({})
        try:
            child_env = _sanitize_subprocess_env({"SERVICE_TOKEN": "token-for-default"})
        finally:
            ss.reset_secret_scope(token)
            ss.set_multiplex_active(False)

        assert "SERVICE_TOKEN" not in child_env

    def test_shared_local_snapshot_re_resolves_current_profile(self, monkeypatch, tmp_path):
        """A persistent shell snapshot must not retain the previous profile's value."""
        from tools.environments.local import LocalEnvironment

        register_env_passthrough(["SERVICE_TOKEN"])
        monkeypatch.setenv("SERVICE_TOKEN", "token-for-default")
        ss.set_multiplex_active(True)
        env = None
        token_b = None
        token_c = None
        try:
            token_a = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-profile-a"})
            try:
                env = LocalEnvironment(cwd=str(tmp_path))
                assert env.execute("printf '%s' \"$SERVICE_TOKEN\"")["output"] == "token-for-profile-a"
            finally:
                ss.reset_secret_scope(token_a)

            token_b = ss.set_secret_scope({"SERVICE_TOKEN": "token-for-profile-b"})
            result = env.execute("printf '%s' \"$SERVICE_TOKEN\"")
            ss.reset_secret_scope(token_b)
            token_b = None

            token_c = ss.set_secret_scope({})
            missing = env.execute("printf '%s' \"${SERVICE_TOKEN-unset}\"")
        finally:
            if token_b is not None:
                ss.reset_secret_scope(token_b)
            if token_c is not None:
                ss.reset_secret_scope(token_c)
            ss.set_multiplex_active(False)
            if env is not None:
                env.cleanup()

        assert result["output"] == "token-for-profile-b"
        assert missing["output"] == "unset"

    def test_blocklisted_var_blocked_by_default(self):
        from tools.environments.local import _sanitize_subprocess_env, _HERMES_PROVIDER_ENV_BLOCKLIST

        # Pick a var we know is in the blocklist
        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var not in result
        assert "PATH" in result

    def test_passthrough_cannot_override_provider_blocklist(self):
        """GHSA-rhgp-j443-p4rf: register_env_passthrough must NOT accept
        Hermes provider credentials — that was the bypass where a skill
        could declare ANTHROPIC_TOKEN / OPENAI_API_KEY as passthrough and
        defeat the execute_code sandbox scrubbing."""
        from tools.environments.local import (
            _sanitize_subprocess_env,
            _HERMES_PROVIDER_ENV_BLOCKLIST,
        )

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        # Attempt to register — must be silently refused (logged warning).
        register_env_passthrough([blocked_var])

        # is_env_passthrough must NOT report it as allowed
        assert not is_env_passthrough(blocked_var)

        # Sanitizer still strips the var from subprocess env
        env = {blocked_var: "secret_value", "PATH": "/usr/bin"}
        result = _sanitize_subprocess_env(env)
        assert blocked_var not in result
        assert "PATH" in result

    def test_passthrough_cannot_override_internal_dynamic_secret(self):
        """A skill must NOT be able to register dynamically-named Hermes
        secrets (AUXILIARY_*_API_KEY / _BASE_URL, GATEWAY_RELAY_* auth) as
        passthrough — they aren't in the static blocklist, so this is the
        defense-in-depth layer that keeps env_passthrough consistent with the
        unconditional strip in the sanitizers."""
        from tools.environments.local import _sanitize_subprocess_env

        for var in (
            "AUXILIARY_VISION_API_KEY",
            "AUXILIARY_VISION_BASE_URL",
            "GATEWAY_RELAY_SECRET",
            "GATEWAY_RELAY_DELIVERY_KEY",
        ):
            register_env_passthrough([var])
            assert not is_env_passthrough(var), (
                f"{var} should be refused passthrough registration"
            )
            result = _sanitize_subprocess_env({var: "secret", "PATH": "/usr/bin"})
            assert var not in result
            assert "PATH" in result

    def test_passthrough_cannot_override_connector_runtime_scope(self):
        """Connector runtime vars are dedicated-runner only, never passthrough.

        ⚠️ 这条原本对 ``PROFILE_SCOPED_SUBPROCESS_ENV_KEYS`` **整个并集**断言
        「键绝不出现在 run env」。但那个并集由 5 个语义不同的子集合拼成，而
        "dedicated-runner only" 的理由（来自 `159b9c199f fix(connectors):
        收紧连接器令牌执行边界`）只针对 **bearer token** 那几类。
        ``PROFILE_PUBLIC_RUNTIME_ENV_KEYS`` 在定义处写的恰恰相反 ——
        「平台拥有的、按 profile 隔离的路径能力……终端和 skills **需要随当前
        profile 重注入**」。⇒ 对这一类要求「键不出现」是把连接器令牌的规则
        套错了对象，也正是它挡住了 profile 隔离修复。

        ⇒ 拆成两类：bearer 类**原保护一条不减**；public 路径类换成成对断言
        （正向值对 / 反向外部值被丢弃 / profile 之间互不相等）。
        """
        from tools.environments.local import (
            PROFILE_PUBLIC_RUNTIME_ENV_KEYS,
            PROFILE_SCOPED_SUBPROCESS_ENV_KEYS,
            _make_run_env,
            _sanitize_subprocess_env,
        )

        bearer_keys = PROFILE_SCOPED_SUBPROCESS_ENV_KEYS - PROFILE_PUBLIC_RUNTIME_ENV_KEYS
        assert bearer_keys, "calibration: bearer 类为空,判据会恒真"

        for var in PROFILE_SCOPED_SUBPROCESS_ENV_KEYS:
            register_env_passthrough([var])
            # 无论哪一类,注册 passthrough 都不该让它成为 passthrough,
            # 也不该让外部值穿过 sanitize —— 这两条对所有键都保持不变。
            assert not is_env_passthrough(var)
            assert var not in _sanitize_subprocess_env({var: "secret", "PATH": "/usr/bin"})

        # ── bearer token:原契约,键绝不出现在通用子进程环境里 ──
        for var in sorted(bearer_keys):
            assert var not in _make_run_env({var: "secret"}), (
                f"{var} 是 bearer token,⛔ 不该进通用子进程环境")

        # ── public 路径能力:反向 —— 外部传入的值一律不被采纳 ──
        for var in sorted(PROFILE_PUBLIC_RUNTIME_ENV_KEYS):
            assert _make_run_env({var: "secret"}).get(var) != "secret", (
                f"{var} 采纳了调用方传入的外部值,越过了 profile 派生")

    def test_profile_public_runtime_keys_are_derived_per_profile(self, tmp_path,
                                                                 monkeypatch):
        """public 路径能力必须【按 profile 派生】,而且两个 profile 互不相等。

        ⭐ 这条是 P1 那个缺陷的直接反面：`WECOM_CLI_CONFIG_DIR` 曾经硬指
        `profiles/main`，于是 A profile 的企微 CLI 读到了 B profile 的配置。
        只断言"外部值没被采纳"不够 —— 那只证明某个特定字符串没进来，
        ⛔ 没证明进来的是**对的那个**。
        """
        from tools.environments.local import (
            PROFILE_PUBLIC_RUNTIME_ENV_KEYS,
            _make_run_env,
        )

        def _run_env_for(home):
            monkeypatch.setenv("HERMES_HOME", str(home))
            return _make_run_env({})

        env_a = _run_env_for(tmp_path / "profileA")
        env_b = _run_env_for(tmp_path / "profileB")

        derived = [k for k in sorted(PROFILE_PUBLIC_RUNTIME_ENV_KEYS) if k in env_a]
        assert derived, (
            "calibration: 一个 public 路径键都没被注入,下面的隔离断言会恒真")

        for var in derived:
            # 正向:值确实由当前 profile 派生(落在该 profile 目录下)
            assert str(tmp_path / "profileA") in env_a[var], (
                f"{var}={env_a[var]!r} 不是从当前 profile 派生的")
            # 隔离:换个 profile 必须拿到不同的值
            assert env_b.get(var) != env_a[var], (
                f"{var} 在两个 profile 下相同({env_a[var]!r}) —— profile 隔离失效")

    def test_passthrough_allows_auxiliary_non_secret_routing(self):
        """AUXILIARY_*_PROVIDER / _MODEL and GATEWAY_RELAY routing hints are not
        secrets, so a skill may still register them (they're not protected)."""
        register_env_passthrough([
            "AUXILIARY_VISION_PROVIDER",
            "AUXILIARY_VISION_MODEL",
            "GATEWAY_RELAY_URL",
        ])
        assert is_env_passthrough("AUXILIARY_VISION_PROVIDER")
        assert is_env_passthrough("AUXILIARY_VISION_MODEL")
        assert is_env_passthrough("GATEWAY_RELAY_URL")

    def test_make_run_env_blocklist_override_rejected(self):
        """_make_run_env must NOT expose a blocklisted var to subprocess env
        even after a skill attempts to register it via passthrough."""
        from tools.environments.local import (
            _make_run_env,
            _HERMES_PROVIDER_ENV_BLOCKLIST,
        )

        blocked_var = next(iter(_HERMES_PROVIDER_ENV_BLOCKLIST))
        os.environ[blocked_var] = "secret_value"
        try:
            # Without passthrough — blocked
            result_before = _make_run_env({})
            assert blocked_var not in result_before

            # Skill tries to register it — must be refused, so still blocked
            register_env_passthrough([blocked_var])
            result_after = _make_run_env({})
            assert blocked_var not in result_after
        finally:
            os.environ.pop(blocked_var, None)

    def test_non_hermes_api_key_still_registerable(self):
        """Third-party API keys (TENOR_API_KEY, NOTION_TOKEN, etc.) are NOT
        Hermes provider credentials and must still pass through — skills
        that legitimately wrap third-party APIs must keep working."""
        # TENOR_API_KEY is a real example — used by the gif-search skill
        register_env_passthrough(["TENOR_API_KEY"])
        assert is_env_passthrough("TENOR_API_KEY")

        # Arbitrary skill-specific var
        register_env_passthrough(["MY_SKILL_CUSTOM_CONFIG"])
        assert is_env_passthrough("MY_SKILL_CUSTOM_CONFIG")

    def test_provider_blocklist_import_failure_fails_closed(self, monkeypatch):
        """If the dynamic provider blocklist can't be imported, provider
        credentials must be treated as protected and refused passthrough —
        otherwise a skill could tunnel a Hermes credential into the
        execute_code child (regression for #37950 / GHSA-rhgp-j443-p4rf).

        Verifies the full path: _is_hermes_provider_credential returns True,
        register_env_passthrough refuses the var, and _scrub_child_env keeps
        it out of the child env. A non-Hermes key is also rejected here (the
        fallback is conservative: when we can't tell, we fail closed), which
        is the safe direction.
        """
        import builtins

        from tools.code_execution_tool import _scrub_child_env

        real_import = builtins.__import__

        def fail_local_import(name, *args, **kwargs):
            if name == "tools.environments.local":
                raise ImportError("synthetic blocklist import failure")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_local_import)

        # Every name is now treated as a protected provider credential.
        assert _ep_mod._is_hermes_provider_credential("OPENAI_API_KEY")
        assert _ep_mod._is_hermes_provider_credential("ANTHROPIC_API_KEY")
        assert _ep_mod._is_hermes_provider_credential("GH_TOKEN")

        # Registration is refused while the blocklist is unavailable.
        register_env_passthrough(["OPENAI_API_KEY", "ANTHROPIC_API_KEY"])
        assert not is_env_passthrough("OPENAI_API_KEY")
        assert not is_env_passthrough("ANTHROPIC_API_KEY")

        # And the credential never reaches the execute_code child.
        child_env = _scrub_child_env(
            {
                "OPENAI_API_KEY": "synthetic-secret",
                "ANTHROPIC_API_KEY": "synthetic-secret",
                "PATH": "/usr/bin",
            },
            is_passthrough=is_env_passthrough,
            is_windows=False,
        )
        assert "OPENAI_API_KEY" not in child_env
        assert "ANTHROPIC_API_KEY" not in child_env
        assert child_env["PATH"] == "/usr/bin"
