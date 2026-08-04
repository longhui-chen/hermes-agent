"""Runtime authorization regressions for markdown-vault mutations."""

import json

import pytest

from agent import secret_scope
from hermes_constants import (
    reset_hermes_home_override,
    set_hermes_home_override,
)
from plugins.markdown_vault import tools
import tools.registry as tool_registry
from tools.registry import ToolRegistry, invalidate_check_fn_cache


VAULT = "/tmp/mdvault-runtime-auth-test"


def _mock_health(monkeypatch, *, advertise_mutations: bool) -> None:
    payload = {"status": "ok"}
    if advertise_mutations:
        payload["capabilities"] = [tools._CONDITIONAL_MUTATION_CAPABILITY]

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _limit=-1):
            return json.dumps(payload).encode("utf-8")

    monkeypatch.setattr(
        tools.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(),
    )


@pytest.fixture(autouse=True)
def _runtime(monkeypatch):
    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(False)
    monkeypatch.setenv("MARKDOWN_VAULT_PATH", VAULT)
    monkeypatch.setenv("MARKDOWN_VAULT_WRITE", "1")
    monkeypatch.setenv(
        "ZETTLAB_FILE_API_URL",
        "http://127.0.0.1:9090/api/v1",
    )
    invalidate_check_fn_cache()
    try:
        yield
    finally:
        invalidate_check_fn_cache()
        secret_scope.set_multiplex_active(previous_multiplex)


def test_write_gate_requires_live_conditional_mutation_capability(monkeypatch):
    _mock_health(monkeypatch, advertise_mutations=False)
    assert tools.check_vault_requirements() is True
    assert tools.check_vault_write_requirements() is False

    _mock_health(monkeypatch, advertise_mutations=True)
    assert tools.check_vault_write_requirements() is True


@pytest.mark.parametrize(
    ("name", "schema", "handler", "args"),
    [
        (
            "vault_write",
            tools.VAULT_WRITE_SCHEMA,
            tools.handle_vault_write,
            {"note": "legacy.md", "content": "new"},
        ),
        (
            "vault_delete",
            tools.VAULT_DELETE_SCHEMA,
            tools.handle_vault_delete,
            {"note": "legacy.md"},
        ),
    ],
)
def test_mutation_dispatch_flow_rechecks_capability_after_cached_gate(
    monkeypatch,
    name,
    schema,
    handler,
    args,
):
    _mock_health(monkeypatch, advertise_mutations=True)
    registry = ToolRegistry()
    registry.register(
        name=name,
        toolset="markdown_vault_write",
        schema=schema,
        handler=handler,
        check_fn=tools.check_vault_write_requirements,
    )
    assert registry.get_definitions({name}), "capable server should expose tool"

    # Simulate a local-server rollback while the registry still serves its
    # cached successful check_fn result.
    _mock_health(monkeypatch, advertise_mutations=False)
    assert registry.get_definitions({name}), "cached check_fn should still be true"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy server dispatch touched vault data")

    for operation in (
        "_read_note",
        "_backup",
        "_upload",
        "_conditional_update",
        "_delete_abs",
    ):
        monkeypatch.setattr(tools, operation, forbidden)

    result = json.loads(registry.dispatch(name, args))
    assert "conditional-mutation capability" in result["error"]


def test_vault_root_fails_closed_without_multiplex_scope(monkeypatch):
    secret_scope.set_multiplex_active(True)
    monkeypatch.setenv("MARKDOWN_VAULT_PATH", "/foreign/global-vault")

    with pytest.raises(secret_scope.UnscopedSecretError):
        tools._vault_root()
    assert tools.check_vault_requirements() is False
    assert tools.check_vault_write_requirements() is False


def test_profile_scoped_write_gate_does_not_reuse_another_profiles_grant(
    monkeypatch,
    tmp_path,
):
    _mock_health(monkeypatch, advertise_mutations=True)
    monkeypatch.setenv("MARKDOWN_VAULT_PATH", "/foreign/global-vault")
    monkeypatch.setenv("MARKDOWN_VAULT_WRITE", "1")
    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / ".env").write_text(
        "MARKDOWN_VAULT_PATH=/profile-a/vault\nMARKDOWN_VAULT_WRITE=1\n",
        encoding="utf-8",
    )
    (profile_b / ".env").write_text(
        "MARKDOWN_VAULT_PATH=/profile-b/vault\n",
        encoding="utf-8",
    )

    secret_scope.set_multiplex_active(True)
    invalidate_check_fn_cache()

    def check(profile):
        home_token = set_hermes_home_override(str(profile))
        scope_token = secret_scope.set_secret_scope(
            secret_scope.build_profile_secret_scope(profile)
        )
        try:
            return (
                tools._vault_root(),
                tool_registry._check_fn_cached(tools.check_vault_write_requirements),
            )
        finally:
            secret_scope.reset_secret_scope(scope_token)
            reset_hermes_home_override(home_token)

    assert check(profile_a) == ("/profile-a/vault", True)
    assert check(profile_b) == ("/profile-b/vault", False)


@pytest.mark.parametrize(
    ("name", "schema", "handler", "args"),
    [
        (
            "vault_write",
            tools.VAULT_WRITE_SCHEMA,
            tools.handle_vault_write,
            {"note": "revoked.md", "content": "new"},
        ),
        (
            "vault_delete",
            tools.VAULT_DELETE_SCHEMA,
            tools.handle_vault_delete,
            {"note": "revoked.md"},
        ),
    ],
)
def test_mutation_dispatch_flow_rechecks_revoked_profile_after_cached_gate(
    monkeypatch,
    tmp_path,
    name,
    schema,
    handler,
    args,
):
    _mock_health(monkeypatch, advertise_mutations=True)
    env_path = tmp_path / ".env"
    env_path.write_text(
        f"MARKDOWN_VAULT_PATH={VAULT}\nMARKDOWN_VAULT_WRITE=1\n",
        encoding="utf-8",
    )

    previous_multiplex = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)
    home_token = set_hermes_home_override(str(tmp_path))
    scope_token = secret_scope.set_secret_scope(
        secret_scope.build_profile_secret_scope(tmp_path)
    )
    try:
        registry = ToolRegistry()
        registry.register(
            name=name,
            toolset="markdown_vault_write",
            schema=schema,
            handler=handler,
            check_fn=tools.check_vault_write_requirements,
        )
        assert registry.get_definitions({name}), "initial write grant should expose tool"

        # The turn scope is now stale/writeable, while the source-of-truth
        # profile file has been revoked by local-server.
        env_path.write_text(f"MARKDOWN_VAULT_PATH={VAULT}\n", encoding="utf-8")
        assert not registry.get_definitions({name}), "revoked grant must not stay cached"

        def forbidden(*_args, **_kwargs):
            raise AssertionError("revoked dispatch touched vault data")

        for operation in (
            "_read_note",
            "_backup",
            "_upload",
            "_conditional_update",
            "_delete_abs",
        ):
            monkeypatch.setattr(tools, operation, forbidden)

        result = json.loads(registry.dispatch(name, args))
        assert "write access is not granted" in result["error"]
    finally:
        secret_scope.reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)
        secret_scope.set_multiplex_active(previous_multiplex)
