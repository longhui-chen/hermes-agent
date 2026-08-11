"""Flow coverage for profile-local Cron Skill operations over real loopback."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from cron.scheduler import _resolve_cron_enabled_toolsets
from gateway.session_context import _VAR_MAP
from model_tools import _clear_tool_defs_cache, get_tool_definitions
from tests.tools._profile_scope import mux_profile_scope
from tools.registry import discover_builtin_tools, registry
from tools.skill_operation_tool import skill_operation_tool


_SKILL = "application-create"
_MAINTENANCE_SKILL = "dashboard-maintenance"
_SLUG = "action-dashboard"
_READ = "maintenance.read"
_WRITE = "maintenance.apply"
_DIGEST = "b" * 64


def _write_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "profile"
    skill = profile / "skills" / "common" / _SKILL
    runtime = skill / "runtime"
    runtime.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: application-create\n---\nCreate apps.\n", encoding="utf-8"
    )
    (runtime / "app_operations.json").write_text(
        json.dumps(
            {
                "schema_version": "hermes.skill_app_operations.v1",
                "operations": [
                    {
                        "name": "maintenance.inspect",
                        "mode": "read",
                        "app_slug": _SLUG,
                        "app_operation": _READ,
                    },
                    {
                        "name": "maintenance.apply",
                        "mode": "mutation",
                        "app_slug": _SLUG,
                        "app_operation": _WRITE,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    maintenance = profile / "skills" / "common" / _MAINTENANCE_SKILL
    maintenance.mkdir(parents=True)
    (maintenance / "SKILL.md").write_text(
        "---\n"
        "name: dashboard-maintenance\n"
        "---\n"
        "Maintain the dashboard through application-create operations.\n",
        encoding="utf-8",
    )
    (profile / "config.yaml").write_text(
        "skills:\n  disabled: []\n  platform_disabled:\n    cron: []\n",
        encoding="utf-8",
    )
    return profile


class _Handler(BaseHTTPRequestHandler):
    calls = []

    def _send(self, payload, status=200):
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.__class__.calls.append(("GET", self.path, self.headers, None))
        assert self.path == f"/api/v1/internal/apps/{_SLUG}/capabilities"
        self._send(
            {
                "version": 1,
                "operations": [
                    {"name": _READ, "mode": "read"},
                    {"name": _WRITE, "mode": "mutation"},
                ],
                "capability_digest": _DIGEST,
            }
        )

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.__class__.calls.append(("POST", self.path, self.headers, body))
        assert body["capability_digest"] == _DIGEST
        self._send({"operation": self.path.rsplit("/", 1)[-1], "received": body})

    def log_message(self, _format, *_args):
        return


@contextmanager
def _server():
    _Handler.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}/api/v1/internal/apps", _Handler.calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextmanager
def _cron_scope(value="1", skills=(_MAINTENANCE_SKILL, _SKILL)):
    from gateway.session_context import (
        pop_cron_attached_skills,
        push_cron_attached_skills,
    )
    from tools.skill_operation_tool import (
        pop_cron_manifest_snapshot,
        push_cron_manifest_snapshot,
    )

    cron_var = _VAR_MAP["HERMES_CRON_SESSION"]
    token = cron_var.set(value)
    skills_token = push_cron_attached_skills(skills)
    snapshot_token = push_cron_manifest_snapshot()
    try:
        yield
    finally:
        pop_cron_manifest_snapshot(snapshot_token)
        pop_cron_attached_skills(skills_token)
        cron_var.reset(token)


def test_profile_local_skill_operation_real_transport_flow(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    with _server() as (base_url, calls), _cron_scope(), mux_profile_scope(
        monkeypatch,
        {
            "ZET_APPHOST_BASE_URL": base_url,
            "ZETTLAB_AGENT_ACTION_TOKEN": "flow-skill-token",
            "ZET_AGENT_ID": "flow-skill-agent",
        },
        poison_environ=True,
    ):
        capability = json.loads(
            skill_operation_tool({"action": "capabilities"})
        )
        read = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": "maintenance.inspect",
                    "query": {"limit": "10"},
                }
            )
        )
        mutation = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": "maintenance.apply",
                    "payload": {
                        "signals": [
                            {
                                "source_ids": ["entry-1"],
                                "source_url": "https://example.com/feed",
                                "confidence": 0.8,
                            }
                        ]
                    },
                    "idempotency_key": "digest:run-1:revision-1",
                }
            )
        )

    assert capability["data"]["operations"] == [
        {"name": "maintenance.inspect", "mode": "read"},
        {"name": "maintenance.apply", "mode": "mutation"},
    ]
    assert read["data"]["operation"] == _READ
    assert mutation["data"]["operation"] == _WRITE
    assert [method for method, *_rest in calls] == ["GET", "POST", "GET", "POST"]
    assert calls[1][3] == {
        "capability_digest": _DIGEST,
        "query": {"limit": "10"},
    }
    assert calls[3][3]["idempotency_key"] == "digest:run-1:revision-1"
    assert all(
        headers.get("X-Zettlab-Agent-Action-Token") == "flow-skill-token"
        for _method, _path, headers, _body in calls
    )
    assert all("flow-skill-agent" not in json.dumps(body) for *_prefix, body in calls if body)


def test_real_transport_is_not_called_after_same_run_manifest_change(
    monkeypatch, tmp_path
):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    manifest = (
        profile
        / "skills"
        / "common"
        / _SKILL
        / "runtime"
        / "app_operations.json"
    )
    with _server() as (base_url, calls), _cron_scope(), mux_profile_scope(
        monkeypatch,
        {
            "ZET_APPHOST_BASE_URL": base_url,
            "ZETTLAB_AGENT_ACTION_TOKEN": "flow-skill-token",
            "ZET_AGENT_ID": "flow-skill-agent",
        },
        poison_environ=True,
    ):
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "hermes.skill_app_operations.v1",
                    "operations": [
                        {
                            "name": "maintenance.apply",
                            "mode": "mutation",
                            "app_slug": "other-dashboard",
                            "app_operation": "records.apply",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        output = json.loads(
            skill_operation_tool(
                {
                    "action": "invoke",
                    "operation": "maintenance.apply",
                    "idempotency_key": "flow-tamper:1",
                }
            )
        )

    assert output["error"]["code"] == "skill_operation_unavailable"
    assert calls == []


def test_tool_discovery_and_non_cron_dispatch_fail_closed(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    discover_builtin_tools()
    entry = registry.get_entry("skill_operation")
    assert entry is not None
    assert entry.toolset == "zettlab_skill_runtime"
    assert getattr(entry.check_fn, "_session_scope_sensitive") is True

    with _cron_scope(""), patch("tools.app_data_tool._urlopen") as urlopen:
        output = json.loads(
            registry.dispatch(
                "skill_operation", {"action": "capabilities"}
            )
        )
    assert output["error"]["code"] == "cron_scope_required"
    urlopen.assert_not_called()


def test_default_cron_toolset_exposes_only_bound_skill_operation(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    discover_builtin_tools()

    with _cron_scope(), mux_profile_scope(
        monkeypatch,
        {
            "ZET_APPHOST_BASE_URL": "http://127.0.0.1:18080/api/v1/internal/apps",
            "ZETTLAB_AGENT_ACTION_TOKEN": "flow-skill-token",
            "ZET_AGENT_ID": "flow-skill-agent",
        },
        poison_environ=True,
    ):
        enabled = _resolve_cron_enabled_toolsets({}, {})
        assert enabled is not None
        names = {
            item["function"]["name"]
            for item in get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
    assert "zettlab_skill_runtime" in enabled
    assert "skill_operation" in names

    with _cron_scope(""), mux_profile_scope(
        monkeypatch,
        {
            "ZET_APPHOST_BASE_URL": "http://127.0.0.1:18080/api/v1/internal/apps",
            "ZETTLAB_AGENT_ACTION_TOKEN": "flow-skill-token",
            "ZET_AGENT_ID": "flow-skill-agent",
        },
        poison_environ=True,
    ):
        non_cron_names = {
            item["function"]["name"]
            for item in get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }
    assert "skill_operation" not in non_cron_names


def test_single_profile_schema_cache_tracks_job_skills_and_manifest(monkeypatch, tmp_path):
    profile = _write_profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    monkeypatch.setenv(
        "ZET_APPHOST_BASE_URL", "http://127.0.0.1:18080/api/v1/internal/apps"
    )
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "flow-skill-token")
    monkeypatch.setenv("ZET_AGENT_ID", "flow-skill-agent")
    discover_builtin_tools()
    enabled = _resolve_cron_enabled_toolsets({}, {})
    assert enabled is not None

    def visible_names():
        return {
            item["function"]["name"]
            for item in get_tool_definitions(
                enabled_toolsets=enabled,
                quiet_mode=True,
                skip_tool_search_assembly=True,
            )
        }

    # Both orderings matter: no first Cron job may decide availability for the
    # next job in the same long-lived single-profile process.
    _clear_tool_defs_cache()
    with _cron_scope(skills=()):
        assert "skill_operation" not in visible_names()
    with _cron_scope():
        assert "skill_operation" in visible_names()

    _clear_tool_defs_cache()
    with _cron_scope():
        assert "skill_operation" in visible_names()
    with _cron_scope(skills=()):
        assert "skill_operation" not in visible_names()

    # The attached name is unchanged, so only the manifest digest can
    # invalidate this previously-available cache entry.
    _clear_tool_defs_cache()
    with _cron_scope():
        assert "skill_operation" in visible_names()
    maintenance_index = (
        profile / "skills" / "common" / _MAINTENANCE_SKILL / "SKILL.md"
    )
    maintenance_content = maintenance_index.read_text(encoding="utf-8")
    maintenance_index.unlink()
    with _cron_scope():
        assert "skill_operation" not in visible_names()
    maintenance_index.write_text(maintenance_content, encoding="utf-8")
    with _cron_scope():
        assert "skill_operation" in visible_names()

    manifest = (
        profile
        / "skills"
        / "common"
        / _SKILL
        / "runtime"
        / "app_operations.json"
    )
    manifest.unlink()
    with _cron_scope():
        assert "skill_operation" not in visible_names()
    _clear_tool_defs_cache()
