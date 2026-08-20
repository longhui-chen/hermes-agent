from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.video_edit import client, normalizer, paths, preferences, tools
import plugins.video_edit as video_plugin


class _Context:
    def __init__(self):
        self.registrations = []

    def register_tool(self, **kwargs):
        self.registrations.append(kwargs)


@pytest.fixture
def isolated_video_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    monkeypatch.setattr("tools.runtime_workdir.agent_output_dir", lambda: str(output))
    return home, output


def test_registers_all_video_tools_without_capability_gate():
    context = _Context()
    video_plugin.register(context)

    names = {item["name"] for item in context.registrations}
    assert names == set(tools.HANDLERS)
    assert len(names) == 9
    assert all(item["toolset"] == "video_edit" for item in context.registrations)
    upload = next(item for item in context.registrations if item["name"] == "video_edit_upload_assets")
    assert upload["schema"]["parameters"]["required"] == ["workflow_id"]
    serialized = json.dumps(
        [{key: value for key, value in item.items() if key != "handler"} for item in context.registrations],
        ensure_ascii=False,
    )
    assert "ActionV1" not in serialized
    assert "BusinessExecution" not in serialized


def test_bundled_manifest_is_discovered_and_executes_runtime_tools(monkeypatch, isolated_video_home):
    import hermes_cli.plugins as plugins_module
    from hermes_cli.plugins import PluginManager
    import tools.registry as registry_module

    fresh_registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", fresh_registry)
    # Exercise the same discovery path used by Hermes, while isolating the
    # scan to this bundled plugin so optional third-party backends cannot make
    # the test environment-dependent.
    isolated_plugins = isolated_video_home[0] / "bundled-plugins"
    isolated_plugins.mkdir(parents=True)
    (isolated_plugins / "video_edit").symlink_to(
        Path(video_plugin.__file__).resolve().parent,
        target_is_directory=True,
    )
    monkeypatch.setattr(plugins_module, "get_bundled_plugins_dir", lambda: isolated_plugins)
    monkeypatch.setattr(PluginManager, "_scan_entry_points", lambda self: [])
    manager = PluginManager()
    manager.discover_and_load(force=True)

    loaded = manager._plugins["video_edit"]
    manifest = loaded.manifest
    assert manifest.kind == "backend"
    assert loaded.enabled is True
    assert set(loaded.tools_registered) == set(tools.HANDLERS)
    assert fresh_registry.get_tool_names_for_toolset("video_edit") == sorted(tools.HANDLERS)

    result = json.loads(
        fresh_registry.dispatch(
            "video_edit_preferences_resolve",
            {"task_id": "discovery-flow", "scene": "travel", "silent": False},
            agent_id="agent-a",
        )
    )
    assert result["ok"] is True
    assert result["next"] == "video_edit_upload_assets"


def test_normalizer_is_a_separate_bounded_plugin_adapter(isolated_video_home, monkeypatch, tmp_path):
    source = tmp_path / "input.mov"
    source.write_bytes(b"source")
    script = tmp_path / "normalize.py"
    script.write_text("# test helper", encoding="utf-8")
    monkeypatch.setenv("ZETTLAB_VIDEO_NORMALIZER", str(script))
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "must-not-reach-helper")
    monkeypatch.setenv("ZETTLAB_BUSINESS_EXECUTION_TOKEN", "retired")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        target = Path(command[command.index("--output") + 1])
        target.write_bytes(b"normalized")
        return type("Completed", (), {"returncode": 0, "stdout": '{"output": "' + str(target) + '"}\n', "stderr": ""})()

    monkeypatch.setattr(normalizer.subprocess, "run", fake_run)
    result = normalizer.normalize_file(source, "workflow-1", 0)
    assert result.read_bytes() == b"normalized"
    assert calls and calls[0][0][:2] == [normalizer.sys.executable, str(script)]
    assert calls[0][1]["check"] is False
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in calls[0][1]["env"]
    assert "ZETTLAB_BUSINESS_EXECUTION_TOKEN" not in calls[0][1]["env"]
    normalizer.cleanup([result], "workflow-1")
    assert not result.exists()


def test_normalizer_resolves_the_rerooted_presets_bundle(monkeypatch, tmp_path):
    presets = tmp_path / "presets"
    script = presets / "skills" / "video-edit-workflow-mini" / "scripts" / "normalize.py"
    script.parent.mkdir(parents=True)
    script.write_text("# installed helper", encoding="utf-8")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets))
    monkeypatch.delenv("ZETTLAB_VIDEO_NORMALIZER", raising=False)

    assert normalizer.normalizer_script() == script.resolve()


def test_preferences_are_profile_scoped_and_memory_hit_does_not_block(isolated_video_home):
    result = json.loads(
        tools.handle_preferences_update(
            {
                "scope": "global",
                "kind": "hard",
                "action": "set",
                "preferences": {"aspect_ratio": "16:9", "style": "travel"},
            },
            agent_id="agent-a",
        )
    )
    assert result["ok"] is True

    resolved = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "task-1", "scene": "hangzhou", "silent": True},
            agent_id="agent-a",
        )
    )
    assert resolved["ok"] is True
    assert resolved["memory_hit"] is True
    assert resolved["preferences"]["aspect_ratio"] == "16:9"
    assert resolved["preferences"]["style"] == "travel"
    assert Path(isolated_video_home[0] / "video_edit" / "agent-a" / "preferences.json").is_file()


def test_preferences_and_workflows_are_isolated_between_agents(isolated_video_home):
    updated = json.loads(
        tools.handle_preferences_update(
            {
                "scope": "global",
                "kind": "hard",
                "action": "set",
                "preferences": {"style": "travel"},
            },
            agent_id="agent-a",
        )
    )
    assert updated["ok"] is True

    agent_a = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "shared-task", "scene": "hangzhou", "silent": True},
            agent_id="agent-a",
        )
    )
    agent_b = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "shared-task", "scene": "hangzhou", "silent": True},
            agent_id="agent-b",
        )
    )

    assert agent_a["memory_hit"] is True
    assert agent_a["preferences"]["style"] == "travel"
    assert agent_b["memory_hit"] is False
    assert agent_b["preferences"]["style"] == "freestyle"
    assert agent_a["workflow_id"] != agent_b["workflow_id"]

    cross_agent = json.loads(
        tools.handle_create_project(
            {"workflow_id": agent_a["workflow_id"]},
            agent_id="agent-b",
        )
    )
    assert "workflow not found" in cross_agent["error"]


def test_preference_memory_is_bounded_and_keeps_recent_scene_updates(isolated_video_home):
    for index in range(preferences.MAX_SCENES + 9):
        result = preferences.update(
            "agent-a",
            "scene",
            f"scene-{index}",
            "hard",
            "set",
            {"style": f"style-{index}"},
        )
        assert result["ok"] is True

    path = isolated_video_home[0] / "video_edit" / "agent-a" / "preferences.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert len(payload["scenes"]) <= preferences.MAX_SCENES
    assert len(path.read_bytes()) <= preferences.MAX_PREFERENCE_BYTES
    assert "scene-0" not in payload["scenes"]
    assert payload["scenes"][f"scene-{preferences.MAX_SCENES + 8}"]["hard"]["style"] == (
        f"style-{preferences.MAX_SCENES + 8}"
    )

    preferences.update(
        "agent-a", "scene", "scene-9", "hard", "set", {"style": "touched"}
    )
    preferences.update(
        "agent-a",
        "scene",
        f"scene-{preferences.MAX_SCENES + 9}",
        "hard",
        "set",
        {"style": "newest"},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "scene-9" in payload["scenes"]
    assert "scene-10" not in payload["scenes"]


def test_oversized_existing_scene_is_dropped_during_read(isolated_video_home, monkeypatch):
    monkeypatch.setattr(preferences, "MAX_SCENE_BYTES", 64)
    path = isolated_video_home[0] / "video_edit" / "agent-a" / "preferences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "global": {"hard": {}, "soft": {}},
                "scenes": {
                    "too-large": {
                        "hard": {"user_prompt": "x" * 512},
                        "soft": {"user_prompt": "y" * 512},
                    },
                    "valid": {"hard": {"style": "travel"}, "soft": {}},
                },
            }
        ),
        encoding="utf-8",
    )

    resolved = preferences.resolve("agent-a", "valid", {}, silent=True)
    assert resolved["preferences"]["style"] == "travel"
    assert preferences.resolve("agent-a", "too-large", {}, silent=True)["memory_hit"] is False


def test_l1_chain_is_idempotent_and_uses_no_video_authorization_headers(isolated_video_home, monkeypatch, tmp_path):
    source = isolated_video_home[1] / "agent-a" / "input.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.delenv("ZETTLAB_AGENT_ACTION_TOKEN", raising=False)
    headers = client._headers("projects", {"x": 1}, agent_id="agent-a")
    assert "Authorization" not in headers
    assert "X-Zettlab-Agent-Action-Token" not in headers
    assert not any("Business-Execution" in key or "Action-Version" in key for key in headers)
    other_profile = client._headers("projects", {"x": 1}, agent_id="agent-b")
    assert headers["Idempotency-Key"] != other_profile["Idempotency-Key"]
    first_edit = client._headers(
        "projects", {"x": 1}, agent_id="agent-a", replay_scope="workflow-1"
    )
    resumed_edit = client._headers(
        "projects", {"x": 1}, agent_id="agent-a", replay_scope="workflow-1"
    )
    later_reedit = client._headers(
        "projects", {"x": 1}, agent_id="agent-a", replay_scope="workflow-2"
    )
    assert first_edit["Idempotency-Key"] == resumed_edit["Idempotency-Key"]
    assert first_edit["Idempotency-Key"] != later_reedit["Idempotency-Key"]

    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "task-chain", "preferences": {"style": "vlog"}},
            agent_id="agent-a",
        )
    )["workflow_id"]
    monkeypatch.setattr(client, "upload", lambda files, **kwargs: {"data": {"uploads": [{"object_key": "obj-1"}]}})
    uploaded = json.loads(
        tools.handle_upload_assets({"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a")
    )
    assert uploaded["uploaded"] == 1
    project_scopes = []
    monkeypatch.setattr(
        client,
        "create_project",
        lambda keys, prefs, user_prompt="", **kwargs: (
            project_scopes.append(kwargs.get("workflow_id"))
            or {"project_id": "project-1", "status": "queued"}
        ),
    )
    created = json.loads(tools.handle_create_project({"workflow_id": workflow, "user_prompt": "make a vlog"}, agent_id="agent-a"))
    assert created["project_id"] == "project-1"
    assert project_scopes == [workflow]
    monkeypatch.setattr(client, "poll_project", lambda project_id, timeout=120, **kwargs: {"project_id": project_id, "status": "completed", "result_url": "https://cdn.example.test/result.mp4"})
    waited = json.loads(tools.handle_wait_project({"workflow_id": workflow, "max_wait_seconds": 15}, agent_id="agent-a"))
    assert waited["continue_required"] is False
    assert "result_url" not in waited

    def fake_download(url, target):
        target.write_bytes(b"rendered")
        return {"path": str(target), "size": 8, "sha256": "digest"}

    monkeypatch.setattr(client, "download", fake_download)
    delivered = json.loads(tools.handle_download_result({"workflow_id": workflow, "filename": "hangzhou.mp4"}, agent_id="agent-a"))
    assert delivered["ok"] is True
    assert Path(delivered["output"]).read_bytes() == b"rendered"
    assert "authorization" not in json.dumps(delivered).lower()


def test_workflow_rejects_source_selection_changes_instead_of_reusing_old_project(isolated_video_home, monkeypatch):
    first = isolated_video_home[1] / "agent-a" / "first.mov"
    second = isolated_video_home[1] / "agent-a" / "second.mov"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "source-selection"}, agent_id="agent-a"
        )
    )["workflow_id"]
    monkeypatch.setattr(
        client,
        "upload",
        lambda files, **kwargs: {"data": {"uploads": [{"object_key": "old-object"}]}},
    )
    assert json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(first)]}, agent_id="agent-a"
        )
    )["uploaded"] == 1

    rejected = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(second)]}, agent_id="agent-a"
        )
    )
    assert "source selection changed" in rejected["error"]
    assert tools.state.get(workflow, "agent-a")["object_keys"] == ["old-object"]


def test_upload_batch_must_return_one_object_for_each_source(isolated_video_home, monkeypatch):
    source = isolated_video_home[1] / "agent-a" / "missing-result.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "upload-cardinality"}, agent_id="agent-a"
        )
    )["workflow_id"]
    monkeypatch.setattr(
        client,
        "upload",
        lambda files, **kwargs: {"data": {"uploads": []}},
    )

    result = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a"
        )
    )

    assert "does not match the requested batch" in result["error"]
    entry = tools.state.get(workflow, "agent-a")
    assert entry.get("object_keys", []) == []


def test_download_recovers_file_committed_before_state_checkpoint(isolated_video_home, monkeypatch):
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "download-crash-window"},
            agent_id="agent-a",
        )
    )["workflow_id"]
    target = paths.result_path("agent-a", "recovered.mp4")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "result_url": "https://cdn.example.test/result.mp4",
            "pending_output_path": str(target),
            "status": "downloading",
        },
    )
    target.write_bytes(b"already-downloaded")
    monkeypatch.setattr(
        client,
        "download",
        lambda *_args, **_kwargs: pytest.fail("recovery must not download twice"),
    )

    delivered = json.loads(
        tools.handle_download_result(
            {"workflow_id": workflow, "filename": "ignored.mp4"},
            agent_id="agent-a",
        )
    )

    assert delivered["ok"] is True
    assert delivered["recovered"] is True
    assert delivered["output"] == str(target)
    assert delivered["size"] == len(b"already-downloaded")


def test_persisted_output_path_is_revalidated_inside_agent_bucket(isolated_video_home):
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "outside-output"},
            agent_id="agent-a",
        )
    )["workflow_id"]
    outside = isolated_video_home[0] / "outside.mp4"
    outside.write_bytes(b"not an agent artifact")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(outside),
            "status": "delivered",
        },
    )

    delivered = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )

    assert "error" in delivered
    assert "checkpoint is invalid" in delivered["error"]


def test_output_root_rejects_a_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real-output"
    real.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr("tools.runtime_workdir.agent_output_dir", lambda: str(link))

    with pytest.raises(paths.VideoPathError, match="output directory is a symlink"):
        paths.output_root("agent-a")


def test_upload_normalized_intermediates_are_cleaned_after_each_batch(isolated_video_home, monkeypatch, tmp_path):
    source = isolated_video_home[1] / "agent-a" / "input.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "normalized", "preferences": {"upload_preference": "normalized"}},
            agent_id="agent-a",
        )
    )["workflow_id"]
    normalized = tmp_path / "normalized.mp4"
    normalized.write_bytes(b"normalized")
    monkeypatch.setattr(normalizer, "normalize_files", lambda files, workflow_id: [normalized])
    monkeypatch.setattr(normalizer, "cleanup", lambda paths, workflow_id: [path.unlink(missing_ok=True) for path in paths])
    monkeypatch.setattr(client, "upload", lambda files, **kwargs: {"data": {"uploads": [{"object_key": "obj-1"}]}})
    uploaded = json.loads(tools.handle_upload_assets({"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a"))
    assert uploaded["strategy"] == "normalized"
    assert uploaded["uploaded"] == 1
    assert not normalized.exists()


def test_normalizer_failure_falls_back_to_direct_without_new_workflow(isolated_video_home, monkeypatch):
    source = isolated_video_home[1] / "agent-a" / "input.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "normalize-fallback", "preferences": {"upload_preference": "normalized"}},
            agent_id="agent-a",
        )
    )["workflow_id"]
    monkeypatch.setattr(
        normalizer,
        "normalize_files",
        lambda files, workflow_id: (_ for _ in ()).throw(normalizer.NormalizeError("unsupported")),
    )
    seen = []
    monkeypatch.setattr(
        client,
        "upload",
        lambda files, **kwargs: seen.extend(files) or {"data": {"uploads": [{"object_key": "obj-1"}] }},
    )

    uploaded = json.loads(
        tools.handle_upload_assets({"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a")
    )

    assert uploaded["ok"] is True
    assert uploaded["workflow_id"] == workflow
    assert uploaded["strategy"] == "raw_direct"
    assert seen == [source.resolve()]


def test_proactive_report_is_exactly_once(isolated_video_home, monkeypatch, tmp_path):
    source = isolated_video_home[1] / "agent-a" / "weekly.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **kwargs: {
            "data": {
                "trigger_id": "pvm-report-exactly-once",
                "scene": "weekly",
                "files": [{"path": str(source)}],
            }
        },
    )
    first = json.loads(tools.handle_proactive_resolve({"manifest_id": "manifest-1", "task_id": "weekly-1"}, agent_id="agent-a"))
    assert first["silent"] is True
    workflow = first["workflow_id"]
    monkeypatch.setattr(client, "upload", lambda files, **kwargs: {"data": {"uploads": [{"object_key": "weekly-object"}]}})
    uploaded = json.loads(
        tools.handle_upload_assets({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert uploaded["uploaded"] == 1
    output = tmp_path / "weekly.mp4"
    output.write_bytes(b"rendered")
    tools.state.update(workflow, "agent-a", {"output_path": str(output), "proactive": True})
    calls = []
    monkeypatch.setattr(client, "proactive_report", lambda manifest_id, path, **kwargs: calls.append((manifest_id, path)) or {"ok": True})
    assert json.loads(tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a"))["reported"] is True
    assert json.loads(tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a"))["reused"] is True
    assert len(calls) == 1


def test_proactive_download_uses_server_trigger_output_bucket(
    isolated_video_home, monkeypatch
):
    """Weekly artifacts must land in the same bucket local-server validates."""
    source = isolated_video_home[1] / "agent-a" / "weekly.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **kwargs: {
            "trigger_mode": "proactive_silent",
            "trigger_id": "pvm-trigger-bucket-test",
            "scene": "weekly",
            "files": [{"path": str(source)}],
        },
    )
    first = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-1", "task_id": "weekly-bucket"},
            agent_id="agent-a",
        )
    )
    workflow = first["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {"result_url": "https://cdn.example.test/result.mp4", "status": "completed"},
    )

    def fake_download(_url, target):
        target.write_bytes(b"rendered")
        return {"path": str(target), "size": 8, "sha256": "digest"}

    monkeypatch.setattr(client, "download", fake_download)
    delivered = json.loads(
        tools.handle_download_result(
            {"workflow_id": workflow, "filename": "weekly.mp4"},
            agent_id="agent-a",
        )
    )

    assert delivered["ok"] is True
    assert delivered["output"].endswith(
        "/output/agent-a/proactive-pvm-trigger-bucket-test/weekly.mp4"
    )
