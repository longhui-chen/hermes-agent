"""Tests for toolsets.py — toolset resolution, validation, and composition."""

from tools.registry import ToolRegistry
from toolsets import (
    TOOLSETS,
    get_toolset,
    resolve_toolset,
    resolve_multiple_toolsets,
    get_all_toolsets,
    validate_toolset,
    create_custom_toolset,
    get_toolset_info,
)


def _dummy_handler(args, **kwargs):
    return "{}"


def _make_schema(name: str, description: str = "test tool"):
    return {
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": {}},
    }


class TestGetToolset:
    def test_known_toolset(self):
        ts = get_toolset("web")
        assert ts is not None
        assert "web_search" in ts["tools"]

    def test_x_search_toolset_marks_read_only_and_points_to_xurl(self):
        ts = get_toolset("x_search")
        assert ts is not None
        assert ts["tools"] == ["x_search"]
        description = ts["description"].lower()
        assert "read-only" in description
        assert "xurl" in description
        assert "authenticated" in description

    def test_merges_registry_tools_into_builtin_toolset(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="web_search_plus",
            toolset="web",
            schema=_make_schema("web_search_plus", "Plugin web search"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        ts = get_toolset("web")
        assert ts is not None
        assert set(ts["tools"]) == {"web_search", "web_extract", "web_search_plus"}



class TestResolveToolset:
    def test_leaf_toolset(self):
        tools = resolve_toolset("web")
        assert set(tools) == {"web_search", "web_extract"}

    def test_composite_toolset(self):
        tools = resolve_toolset("debugging")
        assert "terminal" in tools
        assert "web_search" in tools
        assert "web_extract" in tools

    def test_cycle_detection(self):
        # Create a cycle: A includes B, B includes A
        TOOLSETS["_cycle_a"] = {"description": "test", "tools": ["t1"], "includes": ["_cycle_b"]}
        TOOLSETS["_cycle_b"] = {"description": "test", "tools": ["t2"], "includes": ["_cycle_a"]}
        try:
            tools = resolve_toolset("_cycle_a")
            # Should not infinite loop — cycle is detected
            assert "t1" in tools
            assert "t2" in tools
        finally:
            del TOOLSETS["_cycle_a"]
            del TOOLSETS["_cycle_b"]


    def test_plugin_toolset_uses_registry_snapshot(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_b",
            toolset="plugin_example",
            schema=_make_schema("plugin_b", "B"),
            handler=_dummy_handler,
        )
        reg.register(
            name="plugin_a",
            toolset="plugin_example",
            schema=_make_schema("plugin_a", "A"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        assert resolve_toolset("plugin_example") == ["plugin_a", "plugin_b"]




class TestResolveMultipleToolsets:
    def test_combines_and_deduplicates(self):
        tools = resolve_multiple_toolsets(["web", "terminal"])
        assert "web_search" in tools
        assert "web_extract" in tools
        assert "terminal" in tools
        # No duplicates
        assert len(tools) == len(set(tools))



class TestValidateToolset:
    def test_valid(self):
        assert validate_toolset("web") is True
        assert validate_toolset("terminal") is True


    def test_invalid(self):
        assert validate_toolset("nonexistent") is False

    def test_mcp_alias_uses_live_registry(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="mcp__dynserver__ping",
            toolset="mcp-dynserver",
            schema=_make_schema("mcp__dynserver__ping", "Ping"),
            handler=_dummy_handler,
        )
        reg.register_toolset_alias("dynserver", "mcp-dynserver")

        monkeypatch.setattr("tools.registry.registry", reg)

        assert validate_toolset("dynserver") is True
        assert validate_toolset("mcp-dynserver") is True
        assert "mcp__dynserver__ping" in resolve_toolset("dynserver")


class TestGetToolsetInfo:
    def test_leaf(self):
        info = get_toolset_info("web")
        assert info["name"] == "web"
        assert info["is_composite"] is False
        assert info["tool_count"] == 2

    def test_composite(self):
        info = get_toolset_info("debugging")
        assert info["is_composite"] is True
        assert info["tool_count"] > len(info["direct_tools"])



class TestCreateCustomToolset:
    def test_runtime_creation(self):
        create_custom_toolset(
            name="_test_custom",
            description="Test toolset",
            tools=["web_search"],
            includes=["terminal"],
        )
        try:
            tools = resolve_toolset("_test_custom")
            assert "web_search" in tools
            assert "terminal" in tools
            assert validate_toolset("_test_custom") is True
        finally:
            del TOOLSETS["_test_custom"]


class TestRegistryOwnedToolsets:
    def test_registry_membership_is_live(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="test_live_toolset_tool",
            toolset="test-live-toolset",
            schema=_make_schema("test_live_toolset_tool", "Live"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        assert validate_toolset("test-live-toolset") is True
        assert get_toolset("test-live-toolset")["tools"] == ["test_live_toolset_tool"]
        assert resolve_toolset("test-live-toolset") == ["test_live_toolset_tool"]


class TestToolsetConsistency:
    """Verify structural integrity of the built-in TOOLSETS dict."""

    def test_all_toolsets_have_required_keys(self):
        for name, ts in TOOLSETS.items():
            assert "description" in ts, f"{name} missing description"
            assert "tools" in ts, f"{name} missing tools"
            assert "includes" in ts, f"{name} missing includes"


    def test_hermes_platforms_share_core_tools(self):
        """All hermes-* platform toolsets share the same core tools.

        Platform-specific additions (e.g. ``discord`` / ``discord_admin``
        on hermes-discord, gated on DISCORD_BOT_TOKEN) are allowed on top —
        the invariant is that the core set is identical across platforms.
        """
        platforms = ["hermes-cli", "hermes-telegram", "hermes-discord", "hermes-whatsapp", "hermes-slack", "hermes-signal", "hermes-homeassistant"]
        tool_sets = [set(TOOLSETS[p]["tools"]) for p in platforms]
        # All platforms must contain the shared core; platform-specific
        # extras are OK (subset check, not equality).
        core = set.intersection(*tool_sets)
        for name, ts in zip(platforms, tool_sets):
            assert core.issubset(ts), f"{name} is missing core tools: {core - ts}"
        # Sanity: the shared core must be non-trivial (i.e. we didn't
        # silently let a platform diverge so far that nothing is shared).
        assert len(core) > 20, f"Suspiciously small shared core: {len(core)} tools"


class TestPluginToolsets:
    def test_get_all_toolsets_includes_plugin_toolset(self, monkeypatch):
        reg = ToolRegistry()
        reg.register(
            name="plugin_tool",
            toolset="plugin_bundle",
            schema=_make_schema("plugin_tool", "Plugin tool"),
            handler=_dummy_handler,
        )

        monkeypatch.setattr("tools.registry.registry", reg)

        all_toolsets = get_all_toolsets()
        assert "plugin_bundle" in all_toolsets
        assert all_toolsets["plugin_bundle"]["tools"] == ["plugin_tool"]


class TestDefaultPlatformWebSearchCoverage:
    def test_hermes_whatsapp_toolset_includes_web_search(self):
        assert "web_search" in resolve_toolset("hermes-whatsapp")



class TestResolveToolsetIncludeRegistry:
    """include_registry flag exposes the static (pre-registry-merge) view used
    by platform reverse-mapping. Regression harness for issue #49622."""

    def test_include_registry_false_excludes_registry_tools(self):
        from tools.registry import discover_builtin_tools
        discover_builtin_tools()  # registers read_terminal into 'terminal'

        merged = set(resolve_toolset("terminal"))
        static = set(resolve_toolset("terminal", include_registry=False))

        assert static == {"terminal", "process"}, static
        # read_terminal is registered into 'terminal' but is desktop-only and
        # not part of the static definition — it must only appear in the merged view.
        assert "read_terminal" in merged
        assert "read_terminal" not in static


    def test_static_view_threads_through_includes(self):
        # 'debugging' has direct tools [terminal, process] and includes [web, file]
        static = set(resolve_toolset("debugging", include_registry=False))
        assert {"terminal", "process"} <= static
        assert "web_search" in static
        assert "read_file" in static


    def test_registry_only_toolset_static_view_is_empty(self):
        assert resolve_toolset("__definitely_not_a_real_toolset__", include_registry=False) == []


class TestZetAgentDeviceToolReachability:
    """Registration alone does not make a tool reachable. The REAL path on a
    device is: platform config -> _get_platform_tools (reverse-maps the
    hermes-zet-agent composite into catalog toolset names; non-configurable
    catalog entries are recovered by walking TOOLSETS) -> get_tool_definitions
    (resolves those names back to tool names, then applies check_fn gates).
    A tool registered under a toolset with no catalog entry is an orphan the
    reverse-mapping silently drops — registered, discovered, gate open, and
    still absent from the model's schema (found via live tooldump on a real
    device; two intermediate layers looked correct while the model got
    nothing). Assertions below therefore run the real path, not the layers.
    """

    # The shipped device config for the zet_agent platform (verified live).
    _DEVICE_CONFIG = {"platform_toolsets": {"zet_agent": ["hermes-zet-agent", "cronjob"]}}

    # Verbatim enabled-toolsets list dumped from a real device's gateway
    # ([E2E_TOOLDUMP] instrumentation, 2026-07-30) — the exact input the
    # gateway passed to get_tool_definitions before the catalog fix. The
    # gateway recomputes this list per session via _get_platform_tools
    # (gateway/platforms/zet_agent.py:1783); it is not persisted state, so
    # after the catalog fix a restarted gateway computes it WITH
    # zettlab_apphost added.
    _DEVICE_DUMP_ENABLED = [
        "agent_call", "browser", "clarify", "code_execution", "computer_use",
        "creation_governor", "cronjob", "delegation", "file", "image_gen",
        "kanban", "memory", "session_search", "skills", "terminal", "todo",
        "tts", "video_gen", "vision", "web",
    ]

    @staticmethod
    def _real_path_tool_names(config, platform):
        from hermes_cli.tools_config import _get_platform_tools

        enabled = sorted(
            _get_platform_tools(config, platform, include_default_mcp_servers=False)
        )
        universe = set()
        for ts in enabled:
            universe.update(resolve_toolset(ts))
        return universe

    def test_real_path_full_chain_exposes_app_host(self, monkeypatch):
        """End-to-end: device config -> _get_platform_tools ->
        get_tool_definitions (with the profile scope satisfying the gate) ->
        app_host present in the final model schema."""
        from hermes_cli.tools_config import _get_platform_tools
        from model_tools import get_tool_definitions
        from tests.tools._profile_scope import mux_profile_scope

        enabled = sorted(_get_platform_tools(
            self._DEVICE_CONFIG, "zet_agent", include_default_mcp_servers=False
        ))
        scope = {
            "ZET_APPHOST_BASE_URL": "http://127.0.0.1:18080/api/v1/internal/apphost",
            "ZETTLAB_AGENT_ACTION_TOKEN": "t",
        }
        with mux_profile_scope(monkeypatch, scope):
            defs = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True)
        names = {d["function"]["name"] for d in defs}
        assert "app_host" in names

    def test_device_dump_fixture_plus_catalog_entry_exposes_app_host(self, monkeypatch):
        """Live-dump boundary test: feed the exact enabled list a real device's
        gateway passes to get_tool_definitions — plus the catalog entry the
        fixed generator now appends — and assert app_host reaches the final
        model schema. Complements the generator test below: this one proves
        'once the list carries zettlab_apphost the schema has app_host', the
        generator test proves 'the recomputed list does carry it'."""
        from model_tools import get_tool_definitions
        from tests.tools._profile_scope import mux_profile_scope

        scope = {
            "ZET_APPHOST_BASE_URL": "http://127.0.0.1:18080/api/v1/internal/apphost",
            "ZETTLAB_AGENT_ACTION_TOKEN": "t",
        }
        with mux_profile_scope(monkeypatch, scope):
            defs = get_tool_definitions(
                enabled_toolsets=[*self._DEVICE_DUMP_ENABLED, "zettlab_apphost"],
                quiet_mode=True,
            )
        assert "app_host" in {d["function"]["name"] for d in defs}

    def test_generator_output_covers_device_dump_plus_apphost(self):
        """The enabled list is recomputed per session, so the dump fixture
        stays honest only if the generator's output covers it (minus entries
        that are device-conditional) and now includes zettlab_apphost."""
        from hermes_cli.tools_config import _get_platform_tools

        enabled = set(_get_platform_tools(
            self._DEVICE_CONFIG, "zet_agent", include_default_mcp_servers=False
        ))
        # video_gen on the device comes from the ai-gateway capability probe,
        # which this environment doesn't have — exempt it from the coverage
        # comparison, nothing else.
        device_conditional = {"video_gen"}
        assert set(self._DEVICE_DUMP_ENABLED) - device_conditional <= enabled
        assert "zettlab_apphost" in enabled

    def test_catalog_entry_and_composite_are_both_load_bearing(self):
        # Both halves are required by the recovery walk: the catalog entry is
        # what gets recovered, and its static tools must be a subset of the
        # platform composite's tool-name universe. Dropping either one makes
        # app_host unreachable (deletion experiments EV8a/EV8b).
        assert "app_host" in TOOLSETS["zettlab_apphost"]["tools"]
        assert "app_host" in resolve_toolset("hermes-zet-agent")

    def test_device_meetings_alias_keeps_chat_and_cron_contracts_separate(self):
        """Chat recovers the read-only bridge without changing Cron bindings."""
        assert TOOLSETS["zettlab_device_meetings"]["tools"] == ["device_meetings"]
        assert "device_meetings" in resolve_toolset("hermes-zet-agent")
        assert "skill_operation" not in TOOLSETS["zettlab_device_meetings"]["tools"]
        assert set(TOOLSETS["zettlab_skill_runtime"]["tools"]) == {
            "skill_operation",
            "device_meetings",
        }

    def test_app_host_stays_off_shared_and_cron_real_paths(self):
        # Deliberate scoping, same rationale as call_agent: the App Host
        # credentials only exist in a zet_agent profile, and installing
        # applications on the device is a device-agent capability —
        # messaging/cron schemas must not advertise it. Changing this is a
        # decision, not a drive-by.
        from toolsets import _HERMES_CORE_TOOLS
        assert "app_host" not in _HERMES_CORE_TOOLS
        # Real path for cron (no explicit config -> its default composite).
        assert "app_host" not in self._real_path_tool_names({}, "cron")

    def test_profile_scope_sensitive_tools_reachable_on_zet_agent_real_path(self):
        """Generalized guard for this class of omission: a tool whose check_fn
        is profile-scope-sensitive depends on device-profile credentials that
        only a zet_agent turn can resolve — if the real resolution path drops
        it, it is silently unreachable exactly where it is meant to work.

        Exemptions, each deliberate and documented:
        - explicitly non-Zet-Agent session surfaces: ``skill_operation`` is
          Cron-only and therefore must not be recovered into a zet_agent turn.
        - opt-in toolsets (_DEFAULT_OFF_TOOLSETS): injected via platform
          config when the user enables them (e.g. video_generate) — absent by
          decision, not lost.
        - KNOWN PRE-EXISTING GAPS, reported upstream and pending a decision:
          list_my_channels / send_channel_message (toolset zettlab_channels)
          and get_personal_calendar (toolset personal_calendar) are registered
          under toolsets with no catalog entry, so the real path drops them
          today. Do NOT add to this set — fix the catalog instead; remove an
          entry here when its gap is fixed.
        """
        from hermes_cli.tools_config import _DEFAULT_OFF_TOOLSETS
        from tools.registry import discover_builtin_tools, registry

        known_preexisting_gaps = {
            "list_my_channels", "send_channel_message", "get_personal_calendar",
        }
        non_zet_agent_session_tools = {"skill_operation"}
        discover_builtin_tools()
        scope_sensitive = {
            entry.name
            for entry in registry._tools.values()
            if getattr(entry.check_fn, "_profile_scope_sensitive", False)
            and entry.name not in non_zet_agent_session_tools
            and entry.toolset not in _DEFAULT_OFF_TOOLSETS
        }
        # Sanity: the guard must be looking at a non-empty set, otherwise a
        # marker rename would silently turn this test into a no-op.
        assert "app_host" in scope_sensitive
        assert "app_data" in scope_sensitive
        reachable = self._real_path_tool_names(self._DEVICE_CONFIG, "zet_agent")
        missing = scope_sensitive - reachable - known_preexisting_gaps
        assert not missing, (
            f"profile-scope-sensitive tools not reachable on zet_agent real path: {missing}"
        )
        # If a known gap becomes reachable, the exemption is stale — prune it
        # so the guard tightens instead of rotting.
        healed = known_preexisting_gaps & reachable
        assert not healed, f"known gaps now reachable — remove from exemptions: {healed}"
