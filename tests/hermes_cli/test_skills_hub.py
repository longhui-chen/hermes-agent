from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console
from types import SimpleNamespace

from cli import ChatConsole
from hermes_cli.skills_hub import (
    _DIRECT_USER_INSTALL_REQUEST,
    do_check,
    do_install,
    do_list,
    do_update,
    handle_skills_slash,
)


class _DummyLockFile:
    def __init__(self, installed):
        self._installed = installed

    def list_installed(self):
        return self._installed


@pytest.fixture()
def hub_env(monkeypatch, tmp_path):
    """Set up isolated hub directory paths and return (monkeypatch, tmp_path)."""
    import tools.skills_hub as hub

    hub_dir = tmp_path / "skills" / ".hub"
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

    return hub_dir


# ---------------------------------------------------------------------------
# Fixtures for common skill setups
# ---------------------------------------------------------------------------

_HUB_ENTRY = {"name": "hub-skill", "source": "github", "trust_level": "community"}

_ALL_THREE_SKILLS = [
    {"name": "hub-skill", "category": "x", "description": "hub"},
    {"name": "builtin-skill", "category": "x", "description": "builtin"},
    {"name": "local-skill", "category": "x", "description": "local"},
]

_BUILTIN_MANIFEST = {"builtin-skill": "abc123"}


@pytest.fixture()
def three_source_env(monkeypatch, hub_env):
    """Populate hub/builtin/local skills for source-classification tests."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([_HUB_ENTRY]))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_kwargs: list(_ALL_THREE_SKILLS))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: dict(_BUILTIN_MANIFEST))

    return hub_env


def _capture(source_filter: str = "all") -> str:
    """Run do_list into a string buffer and return the output."""
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(source_filter=source_filter, console=console)
    return sink.getvalue()


def _capture_check(monkeypatch, results, name=None) -> str:
    import tools.skills_hub as hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    do_check(name=name, console=console)
    return sink.getvalue()


def _capture_update(monkeypatch, results) -> tuple[str, list[tuple[str, str, bool]]]:
    import tools.skills_hub as hub
    import hermes_cli.skills_hub as cli_hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    installs = []

    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("L", (), {
        "get_installed": lambda self, name: {"install_path": "category/" + name}
    })())
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, category="", force=False, console=None, source_id=None: installs.append((identifier, category, force)))
    monkeypatch.setattr(
        cli_hub,
        "do_install",
        lambda identifier, category="", force=False, console=None, **_kwargs:
            installs.append((identifier, category, force)),
    )

    do_update(console=console)
    return sink.getvalue(), installs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------




def test_do_list_platform_env_is_ignored(three_source_env, monkeypatch):
    """`hermes skills list` reads the active profile's config via
    HERMES_HOME (swapped by -p), so it must NOT pass a platform arg to
    ``get_disabled_skill_names`` — otherwise per-platform overrides
    would silently leak in from HERMES_PLATFORM env."""
    from agent import skill_utils

    seen = {}

    def _fake(platform=None):
        seen["platform"] = platform
        return set()

    monkeypatch.setattr(skill_utils, "get_disabled_skill_names", _fake)
    _capture()

    assert seen["platform"] is None


# ---------------------------------------------------------------------------
# Cross-registry hijack regression tests
#
# An update must never change a skill's source registry. Skill names are not
# namespaced across registries, so an unconstrained name resolve can install a
# different author's same-named skill over the user's files.
# ---------------------------------------------------------------------------




def test_check_for_skill_updates_does_not_fall_back_across_registries():
    """An entry whose source has no adapter reports `unavailable`.

    Previously `candidate_sources ... or sources` fell back to every source, so
    a same-named skill in another registry could satisfy the fetch and be
    reported as this entry's update -- the step that preceded the overwrite.
    The foreign source here returns a *valid* bundle with a different hash, so
    the old code reports `update_available` (sourced from the wrong registry)
    while the fixed code reports `unavailable`.
    """
    from tools.skills_hub import check_for_skill_updates

    class _ForeignBundle:
        name = "reddit"
        files = {"SKILL.md": "# a different author's reddit skill"}
        source = "skills.sh"
        identifier = "skills-sh/someone-else/reddit"
        trust_level = "community"
        metadata: dict = {}

    class _ForeignSource:
        """skills-sh adapter; must NOT be consulted for a clawhub-locked entry."""

        def source_id(self):
            return "skills-sh"

        def fetch(self, identifier):
            return _ForeignBundle()

        def inspect(self, identifier):
            return _ForeignBundle()

    lock = _DummyLockFile([
        {"name": "reddit", "identifier": "reddit", "source": "clawhub",
         "content_hash": "hash-of-the-clawhub-copy"},
    ])

    results = check_for_skill_updates(
        lock=lock,  # type: ignore[arg-type]  # duck-typed double, matches _DummyLockFile usage above
        sources=[_ForeignSource()],  # type: ignore[list-item]
    )

    assert len(results) == 1
    assert results[0]["source"] == "clawhub", "provenance must be preserved"
    assert results[0]["status"] == "unavailable", (
        "a clawhub-locked skill must not be matched against a skills-sh bundle; "
        "reporting update_available here is the cross-registry hijack"
    )
    assert "bundle" not in results[0], "must not carry a foreign registry's bundle"




# ---------------------------------------------------------------------------
# UrlSource-specific install paths: --name override, interactive prompts,
# non-interactive error, existing-category scan.
# ---------------------------------------------------------------------------


def _make_url_bundle_fetcher(name="", awaiting_name=True, url="https://example.com/SKILL.md"):
    """Return a fake source that simulates ``UrlSource.fetch`` for a
    URL-sourced skill whose name hasn't been auto-resolved."""

    class _UrlSource:
        def inspect(self, identifier):
            return type("Meta", (), {
                "extra": {"url": url, "awaiting_name": awaiting_name},
                "identifier": url,
                "name": name,
                "path": name,
            })()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": name,
                "files": {"SKILL.md": "---\ndescription: ok\n---\n# body\n"},
                "source": "url",
                "identifier": url,
                "trust_level": "community",
                "metadata": {"url": url, "awaiting_name": awaiting_name},
            })()

    return _UrlSource


def _install_mocks(
    monkeypatch,
    tmp_path,
    source_factory,
    category_hint="",
    verdict="safe",
):
    """Wire the minimum set of monkeypatches for a do_install dry run."""
    import tools.skills_hub as hub
    import tools.skills_guard as guard

    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "pending"
    q_path.mkdir(parents=True)

    install_calls: list = []

    def _install_from_quarantine(q, name, category, bundle, result, **_kwargs):
        install_calls.append({"name": name, "category": category})
        install_dir = tmp_path / "skills" / (f"{category}/" if category else "") / name
        install_dir.mkdir(parents=True, exist_ok=True)
        return install_dir

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth: [source_factory()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: q_path)
    monkeypatch.setattr(hub, "install_from_quarantine", _install_from_quarantine)
    monkeypatch.setattr(
        hub, "HubLockFile",
        lambda: type("Lock", (), {"get_installed": lambda self, n: None})(),
    )
    def scan(skill_path, source="community"):
        findings = []
        if verdict != "safe":
            findings.append(
                guard.Finding(
                    pattern_id="test-risk",
                    severity="high",
                    category="network",
                    file="scripts/run.py",
                    line=1,
                    match="requests.post",
                    description="sends data to an external endpoint",
                )
            )
        return guard.ScanResult(
            skill_name="pending",
            source=source,
            trust_level="community",
            verdict=verdict,
            findings=findings,
        )

    monkeypatch.setattr(guard, "scan_skill", scan)
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))
    return install_calls




class _ExternalRegistrySource:
    def inspect(self, identifier):
        return type("Meta", (), {
            "extra": {"source_url": "https://github.com/owner/repo/tree/main/example"},
            "identifier": "owner/repo/example",
            "name": "example",
            "path": "example",
        })()

    def fetch(self, identifier):
        return type("Bundle", (), {
            "name": "example",
            "files": {"SKILL.md": "---\nname: example\n---\n# Example\n"},
            "source": "github",
            "identifier": "owner/repo/example",
            "trust_level": "community",
            "metadata": {
                "source_url": "https://github.com/owner/repo/tree/main/example",
            },
        })()


def test_agent_install_prepares_candidate_without_install_or_approval(
    monkeypatch, tmp_path, hub_env
):
    from hermes_cli.skills_hub import do_agent_install

    installs = _install_mocks(monkeypatch, tmp_path, _ExternalRegistrySource)
    approvals = []

    def approve(tool_name, reason, **kwargs):
        approvals.append((tool_name, reason, kwargs))
        return {"approved": True, "message": None}

    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    sink = StringIO()
    do_agent_install(
        "owner/repo/example",
        console=Console(file=sink, force_terminal=False, color_system=None),
    )

    assert installs == []
    assert approvals == []


def test_safe_agent_install_with_confirmed_intent_has_no_second_approval(
    monkeypatch, tmp_path, hub_env
):
    from hermes_cli.skills_hub import do_agent_install

    installs = _install_mocks(monkeypatch, tmp_path, _ExternalRegistrySource)
    monkeypatch.setattr(
        "tools.approval.request_tool_approval",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("safe confirmed candidate must not prompt again")
        ),
    )

    result = do_agent_install(
        "owner/repo/example",
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        intent_confirmed=True,
    )

    assert result["status"] == "installed"
    assert installs == [{"name": "example", "category": ""}]


def test_changed_candidate_requires_new_user_confirmation(
    monkeypatch, tmp_path, hub_env
):
    from hermes_cli.skills_hub import do_agent_install

    installs = _install_mocks(monkeypatch, tmp_path, _ExternalRegistrySource)
    monkeypatch.setattr(
        "tools.approval.request_tool_approval",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("changed candidate must stop before risk approval")
        ),
    )

    result = do_agent_install(
        "owner/repo/example",
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        intent_confirmed=True,
        expected_candidate={
            "identifier": "owner/repo/example",
            "source_url": "https://github.com/owner/repo/tree/main/example",
            "content_hash": "sha256:older-content",
        },
    )

    assert result["status"] == "candidate_changed"
    assert result["candidate"]["content_hash"] != "sha256:older-content"
    assert installs == []


@pytest.mark.parametrize("verdict", ["caution", "dangerous"])
def test_non_safe_agent_install_requires_risk_specific_one_shot_approval(
    monkeypatch, tmp_path, hub_env, verdict
):
    from hermes_cli.skills_hub import do_agent_install

    installs = _install_mocks(
        monkeypatch,
        tmp_path,
        _ExternalRegistrySource,
        verdict=verdict,
    )
    approvals = []

    def approve(tool_name, reason, **kwargs):
        approvals.append((tool_name, reason, kwargs))
        return {"approved": True, "message": None}

    monkeypatch.setattr("tools.approval.request_tool_approval", approve)
    result = do_agent_install(
        "owner/repo/example",
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        intent_confirmed=True,
    )

    assert result["status"] == "installed"
    assert installs == [{"name": "example", "category": ""}]
    assert len(approvals) == 1
    tool_name, reason, kwargs = approvals[0]
    assert tool_name == "skillhub_install"
    assert "Identifier: owner/repo/example" in reason
    assert "https://github.com/owner/repo/tree/main/example" in reason
    assert "resolved content hash:" in reason
    assert f"scan verdict: {verdict}" in reason
    assert "high/network scripts/run.py" in reason
    assert kwargs["one_shot"] is True
    assert kwargs["allow_yolo_bypass"] is False
    assert kwargs["rule_key"].startswith(
        "external-skill-risk:owner/repo/example:"
    )


def test_denied_risk_decision_does_not_install(
    monkeypatch, tmp_path, hub_env
):
    from hermes_cli.skills_hub import do_agent_install

    installs = _install_mocks(
        monkeypatch,
        tmp_path,
        _ExternalRegistrySource,
        verdict="dangerous",
    )
    monkeypatch.setattr(
        "tools.approval.request_tool_approval",
        lambda *args, **kwargs: {
            "approved": False,
            "message": "user denied risk",
        },
    )

    result = do_agent_install(
        "owner/repo/example",
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        intent_confirmed=True,
    )

    assert result["status"] == "risk_denied"
    assert installs == []


def test_cli_install_preserves_direct_user_install_policy(monkeypatch):
    from hermes_cli.skills_hub import skills_command

    observed = {}

    def capture(identifier, **kwargs):
        observed["identifier"] = identifier
        observed.update(kwargs)

    monkeypatch.setattr("hermes_cli.skills_hub.do_install", capture)
    skills_command(
        SimpleNamespace(
            skills_action="install",
            identifier="owner/repo/example",
            category="",
            force=False,
            yes=False,
            name="",
        )
    )

    assert observed["identifier"] == "owner/repo/example"
    assert observed["_agent_request"] is _DIRECT_USER_INSTALL_REQUEST


def test_raw_noninteractive_external_install_is_blocked_at_mutation_boundary(
    monkeypatch, tmp_path, hub_env
):
    installs = _install_mocks(monkeypatch, tmp_path, _ExternalRegistrySource)
    sink = StringIO()

    do_install(
        "owner/repo/example",
        console=Console(file=sink, force_terminal=False, color_system=None),
        skip_confirm=True,
    )

    assert installs == []
    assert "native skillhub_install approval flow" in sink.getvalue()


def test_direct_library_call_cannot_fake_interactive_confirmation(
    monkeypatch, tmp_path, hub_env
):
    installs = _install_mocks(monkeypatch, tmp_path, _ExternalRegistrySource)
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: pytest.fail(
            "untrusted library caller must not reach interactive confirmation"
        ),
    )
    sink = StringIO()

    do_install(
        "owner/repo/example",
        console=Console(file=sink, force_terminal=False, color_system=None),
    )

    assert installs == []
    assert "native skillhub_install approval flow" in sink.getvalue()


def test_url_install_uses_name_override_on_non_interactive_surface(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        name_override="my-url-skill",
        _agent_request=_DIRECT_USER_INSTALL_REQUEST,
    )

    assert installs == [{"name": "my-url-skill", "category": ""}]


def test_url_install_rejects_invalid_name_override(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        name_override="SKILL",  # rejected by _is_valid_installed_skill_name
    )

    assert installs == []  # did NOT install
    assert "Invalid --name" in sink.getvalue()


def test_url_install_actionable_error_on_non_interactive_with_no_name(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        # No name_override — should error out with a retry hint.
    )

    assert installs == []
    out = sink.getvalue()
    assert "Cannot install from URL" in out
    assert "--name <your-name>" in out


def test_url_install_prompts_interactively_when_tty(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    # Simulate user typing "my-interactive" to name prompt, then "" to category.
    answers = iter(["my-interactive", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=False,  # interactive
        force=True,  # skip the final confirm prompt (tested elsewhere)
        _agent_request=_DIRECT_USER_INSTALL_REQUEST,
    )

    assert installs == [{"name": "my-interactive", "category": ""}]


def test_url_install_prompts_category_and_uses_typed_value(monkeypatch, tmp_path, hub_env):
    import tools.skills_hub as hub
    installs = _install_mocks(
        monkeypatch, tmp_path,
        _make_url_bundle_fetcher(name="sharethis-chat", awaiting_name=False),
    )

    # Stage an existing category bucket so _existing_categories finds it.
    (hub.SKILLS_DIR / "productivity" / "notion").mkdir(parents=True)
    (hub.SKILLS_DIR / "productivity" / "notion" / "SKILL.md").write_text("# notion")

    # Name is already resolved (from frontmatter) → only category prompt fires.
    answers = iter(["productivity"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/sharethis-chat/SKILL.md",
        console=console, skip_confirm=False, force=True,
        _agent_request=_DIRECT_USER_INSTALL_REQUEST,
    )

    assert installs == [{"name": "sharethis-chat", "category": "productivity"}]
    assert "Existing: productivity" in sink.getvalue()


def test_url_install_cancel_name_prompt_aborts(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    # Empty input with no default → name prompt returns None → abort.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=False, force=True,
    )

    assert installs == []
    assert "Installation cancelled" in sink.getvalue()


# ── _existing_categories ────────────────────────────────────────────────────






# ---------------------------------------------------------------------------
# browse_skills — dedup by identifier, not name
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Regression: full identifier must be recoverable from `hermes skills search`
# even when the slug is too long to fit the terminal width (issue #33674).
# ---------------------------------------------------------------------------

# A real browse-sh-style slug whose trailing -XXXXXX hash matters for install
_LONG_SLUG = "browse-sh/weather.gov/get-forecast-1uezib"

_LONG_RESULT = type("R", (), {
    "name": "get-forecast",
    "description": "Fetch the forecast",
    "source": "browse-sh",
    "trust_level": "community",
    "identifier": _LONG_SLUG,
})()


def test_do_search_json_flag_emits_full_identifiers(capsys):
    """`--json` must print a parseable array with full identifiers and skip the table."""
    from hermes_cli.skills_hub import do_search

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=40)

    with patch("tools.skills_hub.unified_search", return_value=[_LONG_RESULT]), \
         patch("tools.skills_hub.create_source_router", return_value={}), \
         patch("tools.skills_hub.GitHubAuth"):
        do_search("weather", console=console, as_json=True)

    # JSON goes to stdout via print(), not the Rich console sink.
    captured = capsys.readouterr().out
    import json as _json
    payload = _json.loads(captured)
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["identifier"] == _LONG_SLUG
    assert payload[0]["name"] == "get-forecast"
    assert payload[0]["source"] == "browse-sh"
    # Table render must be suppressed — sink should be empty (no "Searching for:" header).
    assert "Searching for:" not in sink.getvalue()
