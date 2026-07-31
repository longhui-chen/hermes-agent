"""Tests for tools.approval.request_tool_approval — the plugin pre_tool_call
``{"action": "approve"}`` escalation into the human-approval gate.

These verify that a plugin-driven approval reuses the SAME machinery as a
Tier-2 dangerous-command match: session/permanent allowlist, the CLI prompt,
the gateway submit_pending path, cron_mode, and fail-closed timeouts.
"""

import pytest

import tools.approval as approval
from tools.approval import request_tool_approval


@pytest.fixture(autouse=True)
def _isolate_approval_state(monkeypatch):
    """Give each test a clean session key and empty allowlists."""
    monkeypatch.setattr(
        approval, "get_current_session_key",
        lambda default="default": "test-session",
    )
    # Empty session + permanent approval stores so nothing pre-approves.
    monkeypatch.setattr(approval, "is_approved", lambda sk, pk: False)
    # Not a yolo session (the shared gate checks this first).
    monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: False)
    monkeypatch.setattr(approval, "_YOLO_MODE_FROZEN", False, raising=False)
    # No thread-registered CLI callback by default.
    monkeypatch.setattr(
        "tools.terminal_tool._get_approval_callback", lambda: None, raising=False
    )
    yield


class TestRequestToolApproval:
    def test_session_cached_approval_short_circuits(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        # Should NOT prompt at all.
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("should not prompt when already approved"),
        )
        res = request_tool_approval("write_file", "sensitive path", rule_key="ssh")
        assert res == {"approved": True, "message": None}

    def test_cli_approve_once(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "once")
        res = request_tool_approval("write_file", "writing ~/.ssh/authorized_keys")
        assert res["approved"] is True

    def test_cli_deny_blocks(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval("terminal", "curl PUT to external API")
        assert res["approved"] is False
        assert "denied" in res["message"].lower()
        assert res["pattern_key"].startswith("plugin_rule:")

    def test_cli_session_persists_session_only(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "session")
        calls = {"session": [], "permanent": []}
        monkeypatch.setattr(approval, "approve_session",
                            lambda sk, pk: calls["session"].append(pk))
        monkeypatch.setattr(approval, "approve_permanent",
                            lambda pk: calls["permanent"].append(pk))
        monkeypatch.setattr(approval, "save_permanent_allowlist", lambda x: None)
        res = request_tool_approval("write_file", "reason", rule_key="ssh-writes")
        assert res["approved"] is True
        assert calls["session"] == ["plugin_rule:ssh-writes"]
        assert calls["permanent"] == []  # session != always

    def test_cli_always_persists_permanent(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "always")
        persisted = {}
        monkeypatch.setattr(approval, "approve_session", lambda sk, pk: None)
        monkeypatch.setattr(approval, "approve_permanent",
                            lambda pk: persisted.setdefault("key", pk))
        monkeypatch.setattr(approval, "save_permanent_allowlist",
                            lambda x: persisted.setdefault("saved", True))
        res = request_tool_approval("write_file", "reason", rule_key="ssh-writes")
        assert res["approved"] is True
        assert persisted["key"] == "plugin_rule:ssh-writes"
        assert persisted["saved"] is True

    def test_gateway_path_submits_pending_and_defers(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        submitted = {}
        monkeypatch.setattr(approval, "submit_pending",
                            lambda sk, data: submitted.update(data)
                            or "test-approval-id-1234567890")
        res = request_tool_approval("browser_navigate", "external URL",
                                    rule_key="ext-nav")
        assert res["approved"] is False
        assert res["status"] == "approval_required"
        assert submitted["pattern_key"] == "plugin_rule:ext-nav"

    def test_cron_deny_mode_blocks(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "env_var_enabled",
                            lambda v: v == "HERMES_CRON_SESSION")
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "deny")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "cron" in res["message"].lower()

    def test_cron_approve_mode_allows(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "env_var_enabled",
                            lambda v: v == "HERMES_CRON_SESSION")
        monkeypatch.setattr(approval, "_get_cron_approval_mode", lambda: "approve")
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is True

    def test_rule_key_derived_from_tool_and_reason(self, monkeypatch):
        """With no explicit rule_key, the pattern key is derived from
        tool + a hash of the reason (so distinct reasons persist apart)."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval("patch", "reason")  # no rule_key
        assert res["pattern_key"].startswith("plugin_rule:patch:")

    def test_distinct_reasons_get_distinct_keys(self, monkeypatch):
        """Two different reasons on the SAME tool must not share an [a]lways
        allowlist entry (Finding 3: tool_name alone was too coarse)."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        k1 = request_tool_approval("write_file", "write to ~/.ssh")["pattern_key"]
        k2 = request_tool_approval("write_file", "send an email")["pattern_key"]
        assert k1 != k2

    def test_explicit_rule_key_overrides_derivation(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "prompt_dangerous_approval", lambda *a, **k: "deny")
        res = request_tool_approval("terminal", "any", rule_key="my-rule")
        assert res["pattern_key"] == "plugin_rule:my-rule"

    def test_no_human_non_cron_fails_closed(self, monkeypatch):
        """Non-interactive, non-gateway, NON-cron context blocks (fail-closed)
        — a plugin-flagged action never runs ungated without a human."""
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        monkeypatch.setattr(approval, "env_var_enabled", lambda v: False)  # not cron
        res = request_tool_approval("terminal", "smtp send")
        assert res["approved"] is False
        assert "no interactive user or gateway" in res["message"].lower()

    def test_yolo_session_bypasses_gate(self, monkeypatch):
        """A --yolo session skips the plugin approval gate (parity with the
        dangerous-command path, via the shared _run_approval_gate)."""
        monkeypatch.setattr(approval, "is_current_session_yolo_enabled", lambda: True)
        monkeypatch.setattr(
            approval, "prompt_dangerous_approval",
            lambda *a, **k: pytest.fail("yolo must not prompt"),
        )
        res = request_tool_approval("terminal", "curl PUT", rule_key="ext")
        assert res == {"approved": True, "message": None}

    def test_operation_specific_risk_can_disable_yolo_bypass(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            approval, "is_current_session_yolo_enabled", lambda: True
        )
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(
            approval, "_is_gateway_approval_context", lambda: False
        )
        prompts = []

        def deny(*args, **kwargs):
            prompts.append((args, kwargs))
            return "deny"

        monkeypatch.setattr(approval, "prompt_dangerous_approval", deny)
        res = request_tool_approval(
            "skillhub_install",
            "resolved scan findings",
            rule_key="external-skill-risk:owner/repo/example:sha256:test",
            one_shot=True,
            allow_yolo_bypass=False,
        )

        assert res["approved"] is False
        assert len(prompts) == 1

    def test_one_shot_ignores_cached_approval_and_cannot_persist(self, monkeypatch):
        monkeypatch.setattr(approval, "is_approved", lambda sk, pk: True)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: True)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: False)
        prompts = []
        persisted = []

        def approve_always(*args, **kwargs):
            prompts.append((args, kwargs))
            return "always"

        monkeypatch.setattr(approval, "prompt_dangerous_approval", approve_always)
        monkeypatch.setattr(
            approval, "approve_session", lambda *args: persisted.append(("session", args))
        )
        monkeypatch.setattr(
            approval, "approve_permanent", lambda *args: persisted.append(("always", args))
        )

        first = request_tool_approval(
            "skillhub_install",
            "identifier owner/repo/a hash aaa",
            rule_key="external-skill:owner/repo/a:aaa",
            one_shot=True,
        )
        second = request_tool_approval(
            "skillhub_install",
            "identifier owner/repo/a hash aaa",
            rule_key="external-skill:owner/repo/a:aaa",
            one_shot=True,
        )

        assert first["approved"] is True
        assert second["approved"] is True
        assert len(prompts) == 2
        assert all(call[1]["allow_permanent"] is False for call in prompts)
        assert persisted == []

    def test_one_shot_gateway_payload_disables_permanent_choice(self, monkeypatch):
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        submitted = {}
        monkeypatch.setattr(
            approval, "submit_pending", lambda sk, data: submitted.update(data)
            or "test-approval-id-1234567890",
        )

        result = request_tool_approval(
            "skillhub_install",
            "identifier owner/repo/a hash aaa",
            rule_key="external-skill:owner/repo/a:aaa",
            one_shot=True,
        )

        assert result["status"] == "approval_required"
        assert result["approval_id"] == "test-approval-id-1234567890"
        assert submitted["allow_permanent"] is False

    def test_deferred_one_shot_grant_is_exact_and_consumed_once(self, monkeypatch):
        session_key = "test-session"
        approval.clear_session(session_key)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(
            approval, "_is_gateway_approval_context", lambda: True
        )

        first = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/a.txt",
            rule_key="agentcomputer:file.delete:request-a",
            one_shot=True,
        )
        assert first["status"] == "approval_required"
        assert approval.resolve_gateway_approval(
            session_key, "once", approval_id=first["approval_id"]
        ) == 1

        different = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/b.txt",
            rule_key="agentcomputer:file.delete:request-b",
            one_shot=True,
        )
        assert different["status"] == "approval_required"

        replay = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/a.txt",
            rule_key="agentcomputer:file.delete:request-a",
            one_shot=True,
        )
        assert replay["approved"] is True
        assert replay["user_approved"] is True

        consumed = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/a.txt",
            rule_key="agentcomputer:file.delete:request-a",
            one_shot=True,
        )
        assert consumed["status"] == "approval_required"
        approval.clear_session(session_key)

    def test_deferred_queue_requires_exact_opaque_id(self, monkeypatch):
        session_key = "queued-approval-session"
        approval.clear_session(session_key)
        monkeypatch.setattr(approval, "_is_interactive_cli", lambda: False)
        monkeypatch.setattr(approval, "_is_gateway_approval_context", lambda: True)
        monkeypatch.setattr(
            approval, "get_current_session_key", lambda default="default": session_key
        )

        first = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/a.txt",
            rule_key="agentcomputer:file.delete:a",
            one_shot=True,
        )
        second = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/b.txt",
            rule_key="agentcomputer:file.delete:b",
            one_shot=True,
        )

        assert first["approval_id"] != second["approval_id"]
        assert approval.resolve_gateway_approval(session_key, "once") == 0
        assert approval.resolve_gateway_approval(
            session_key, "once", approval_id="A" * 32
        ) == 0
        assert approval.resolve_gateway_approval(
            session_key, "once", approval_id=first["approval_id"]
        ) == 1

        replay_first = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/a.txt",
            rule_key="agentcomputer:file.delete:a",
            one_shot=True,
        )
        assert replay_first["approved"] is True
        still_pending_second = request_tool_approval(
            "agentcomputer_cli",
            "delete notes/b.txt",
            rule_key="agentcomputer:file.delete:b",
            one_shot=True,
        )
        assert still_pending_second["status"] == "approval_required"
        approval.clear_session(session_key)
