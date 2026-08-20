from __future__ import annotations

import json
import io
import os
from pathlib import Path
from types import SimpleNamespace

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
        input_path = Path(command[command.index("--input") + 1])
        calls.append((command, kwargs, input_path.stat().st_ino))
        target = Path(command[command.index("--output") + 1])
        target.write_bytes(b"normalized")
        completed = type(
            "Completed",
            (),
            {"returncode": 0, "stdout": '{"output": "' + str(target) + '"}\n', "stderr": ""},
        )()
        return completed, False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", fake_run)
    result = normalizer.normalize_file(source, "workflow-1", 0)
    assert result.read_bytes() == b"normalized"
    assert calls and calls[0][0][:2] == [normalizer.sys.executable, str(script)]
    input_path = Path(calls[0][0][calls[0][0].index("--input") + 1])
    assert input_path != source
    assert calls[0][2] == source.stat().st_ino
    assert "ZETTLAB_AGENT_ACTION_TOKEN" not in calls[0][1]["env"]
    assert "ZETTLAB_BUSINESS_EXECUTION_TOKEN" not in calls[0][1]["env"]
    normalizer.cleanup([result], "workflow-1")
    assert not result.exists()
    assert not input_path.exists()


def test_normalizer_subprocess_output_is_bounded(tmp_path):
    code = (
        "import sys; sys.stderr.write('x' * %d); "
        "sys.stdout.write('{\\\"output\\\": \\\"ok\\\"}\\n')"
    ) % (normalizer.MAX_STDERR_BYTES + 4096)

    completed, overflowed = normalizer._run_bounded_subprocess(
        [normalizer.sys.executable, "-c", code],
        env={},
        timeout=5,
    )

    assert overflowed is True
    assert len(completed.stderr.encode()) <= normalizer.MAX_STDERR_BYTES


def test_normalizer_uses_a_process_group_for_bounded_kill(monkeypatch):
    calls = []

    class FakeProcess:
        pid = 12345

        def __init__(self):
            self.stdout = io.BytesIO(b"")
            self.stderr = io.BytesIO(b"x" * (normalizer.MAX_STDERR_BYTES + 1))
            self.killed = False

        def poll(self):
            return -9 if self.killed else None

        def wait(self):
            self.killed = True
            return -9

        def kill(self):
            self.killed = True

    process = FakeProcess()

    def fake_popen(*_args, **kwargs):
        calls.append(kwargs)
        return process

    monkeypatch.setattr(normalizer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(normalizer.os, "killpg", lambda pid, signal: calls.append((pid, signal)))

    completed, overflowed = normalizer._run_bounded_subprocess(
        ["normalizer"], env={}, timeout=5
    )

    assert overflowed is True
    assert completed.returncode == -9
    if normalizer.os.name == "posix":
        assert calls[-1][0] == process.pid
        assert calls[0]["start_new_session"] is True


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


def test_oversized_preference_file_is_rejected_before_read(isolated_video_home):
    path = isolated_video_home[0] / "video_edit" / "agent-a" / "preferences.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{" + b" " * preferences.MAX_PREFERENCE_BYTES)

    with pytest.raises(preferences.PreferenceError, match="too large"):
        preferences.resolve("agent-a", "general", {}, silent=True)


def test_oversized_workflow_file_is_rejected_before_read(isolated_video_home):
    path = isolated_video_home[0] / "video_edit" / "agent-a" / "workflows.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"{" + b" " * tools.state.MAX_WORKFLOW_BYTES)

    with pytest.raises(tools.state.WorkflowError, match="too large"):
        tools.state.get("vew_missing", "agent-a")


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


def test_same_source_path_replacement_gets_a_new_upload_replay_scope(
    isolated_video_home, monkeypatch
):
    source = isolated_video_home[1] / "agent-a" / "same-path.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"version-one")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "source-replacement"}, agent_id="agent-a"
        )
    )["workflow_id"]
    scopes = []

    def fake_upload(files, **kwargs):
        scopes.append(kwargs["replay_scope"])
        return {
            "data": {
                "uploads": [
                    {"object_key": f"object-{len(scopes)}"} for _ in files
                ]
            }
        }

    monkeypatch.setattr(client, "upload", fake_upload)
    assert json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a"
        )
    )["uploaded"] == 1

    source.write_bytes(b"version-two")
    assert json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a"
        )
    )["uploaded"] == 1
    assert len(scopes) == 2
    assert scopes[0] != scopes[1]


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


def test_raw_media_is_not_accepted_as_output_checkpoint(isolated_video_home, monkeypatch, tmp_path):
    raw_root = tmp_path / "raw-media"
    raw_root.mkdir()
    raw_file = raw_root / "source.mp4"
    raw_file.write_bytes(b"raw media")
    monkeypatch.setattr(paths, "_RAW_ROOTS", (str(raw_root),))

    with pytest.raises(paths.VideoPathError, match="output"):
        paths.validate_output_file(str(raw_file), "agent-a")


def test_output_root_rejects_a_symlink(tmp_path, monkeypatch):
    real = tmp_path / "real-output"
    real.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr("tools.runtime_workdir.agent_output_dir", lambda: str(link))

    with pytest.raises(paths.VideoPathError, match="output directory is a symlink"):
        paths.output_root("agent-a")


def test_output_root_does_not_duplicate_profile_bucket(isolated_video_home):
    """ZET_AGENT_OUTPUT_DIR is already scoped to the active agent profile."""
    assert paths.output_root("agent-a") == isolated_video_home[1].resolve()


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
    second_source = isolated_video_home[1] / "agent-a" / "weekly-2.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    second_source.write_bytes(b"video-2")
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **kwargs: {
            "data": {
                "trigger_id": "pvm-report-exactly-once",
                "scene": "weekly",
                "files": [{"path": str(source)}, {"path": str(second_source)}],
            }
        },
    )
    first = json.loads(tools.handle_proactive_resolve({"manifest_id": "manifest-1", "task_id": "weekly-1"}, agent_id="agent-a"))
    assert first["silent"] is True
    workflow = first["workflow_id"]
    monkeypatch.setattr(
        client,
        "upload",
        lambda files, **kwargs: {
            "data": {
                "uploads": [
                    {"object_key": f"weekly-object-{index}"}
                    for index, _ in enumerate(files)
                ]
            }
        },
    )
    uploaded = json.loads(
        tools.handle_upload_assets({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert uploaded["uploaded"] == 2
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
    second_source = isolated_video_home[1] / "agent-a" / "weekly-2.mov"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"video")
    second_source.write_bytes(b"video-2")
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **kwargs: {
            "trigger_mode": "proactive_silent",
            "trigger_id": "pvm-trigger-bucket-test",
            "scene": "weekly",
            "files": [{"path": str(source)}, {"path": str(second_source)}],
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
    assert delivered["next"] == "video_edit_proactive_report"
    assert delivered["output"].endswith(
        "/output/proactive-pvm-trigger-bucket-test/weekly.mp4"
    )


@pytest.mark.parametrize(
    "files",
    [
        [{"path": "/volume1/subvol/data/only-one.mov"}],
        [{"path": f"/volume1/subvol/data/{index}.mov"} for index in range(tools.state.MAX_FILES + 1)],
        [{"path": "/volume1/subvol/data/one.mov"}, {"path": ""}],
    ],
)
def test_proactive_manifest_rejects_truncated_or_missing_entries(
    isolated_video_home, monkeypatch, files
):
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **kwargs: {
            "trigger_id": "pvm-invalid-manifest",
            "scene": "weekly",
            "files": files,
        },
    )

    result = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-invalid", "task_id": "weekly-invalid"},
            agent_id="agent-a",
        )
    )

    assert "error" in result
    assert "manifest" in result["error"]


def test_poll_project_selects_the_requested_project(monkeypatch):
    monkeypatch.setattr(
        client,
        "post_json",
        lambda *_args, **_kwargs: {
            "data": {
                "projects": [
                    {"project_id": "other", "status": "completed"},
                    {"project_id": "wanted", "status": "processing"},
                ]
            }
        },
    )

    assert client.poll_project("wanted")["project_id"] == "wanted"


def test_post_json_uses_direct_loopback_opener(monkeypatch):
    seen = []

    class Response:
        status = 200

        def getcode(self):
            return 200

        def read(self, _size):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    def build_opener(*handlers):
        seen.extend(handlers)
        return Opener()

    monkeypatch.setattr(client.urllib.request, "build_opener", build_opener)

    client.post_json("projects", {}, agent_id="agent-a")

    assert any(
        isinstance(handler, client.urllib.request.ProxyHandler)
        and handler.proxies == {}
        for handler in seen
    )


def test_download_rejects_numeric_host_resolving_to_loopback(monkeypatch):
    monkeypatch.setattr(
        client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (client.socket.AF_INET, client.socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))
        ],
    )

    assert not client._download_allowed("https://2130706433/result.mp4")


def test_download_connection_uses_resolved_ip_without_dns_retry(monkeypatch):
    dialed = []
    monkeypatch.setattr(
        client.socket,
        "create_connection",
        lambda address, *_args: dialed.append(address) or object(),
    )
    connection = client._PinnedHTTPSConnection(
        "cdn.example.test",
        pinned_addresses=("93.184.216.34",),
    )

    connection._create_connection(("cdn.example.test", 443), 5, None)

    assert dialed == [("93.184.216.34", 443)]


def test_create_project_rejects_partial_upload_checkpoint(isolated_video_home, monkeypatch):
    workflow = "vew_partial-upload-checkpoint"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": [
                "/volume1/subvol/data/one.mov",
                "/volume1/subvol/data/two.mov",
            ],
            "object_keys": ["assets/one"],
        },
    )
    called = False

    def fake_create_project(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"project_id": "unexpected"}

    monkeypatch.setattr(client, "create_project", fake_create_project)

    result = json.loads(
        tools.handle_create_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert "upload" in result["error"]
    assert called is False


def test_upload_idempotency_tracks_source_scope_not_regenerated_temp_file(
    monkeypatch, tmp_path
):
    source = tmp_path / "vewm_0.mp4"
    source.write_bytes(b"first-normalized-version")
    captured_keys = []

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"object-1"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.headers = {}

        def putrequest(self, *_args):
            return None

        def putheader(self, key, value):
            self.headers[key] = value

        def endheaders(self):
            captured_keys.append(self.headers["Idempotency-Key"])

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    client.upload([source], agent_id="agent-a", replay_scope="source-v1")
    stat_result = source.stat()
    os.utime(
        source,
        ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 1_000_000),
    )
    client.upload([source], agent_id="agent-a", replay_scope="source-v1")
    client.upload([source], agent_id="agent-a", replay_scope="source-v2")

    assert captured_keys[0] == captured_keys[1]
    assert captured_keys[1] != captured_keys[2]


def test_upload_rejects_path_replacement_after_validation(monkeypatch, tmp_path):
    source = tmp_path / "source.mov"
    victim = tmp_path / "victim.txt"
    source.write_bytes(b"safe")
    victim.write_bytes(b"secret")

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"object-1"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            source.unlink()
            source.symlink_to(victim)

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    with pytest.raises(client.VideoClientError):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")


def test_upload_closes_sources_when_preconnect_validation_fails(monkeypatch, tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"safe")
    descriptor = 42
    info = SimpleNamespace(
        st_size=4,
        st_mode=0,
        st_dev=1,
        st_ino=2,
        st_mtime_ns=3,
        st_ctime_ns=4,
    )
    closed = []
    monkeypatch.setattr(
        client,
        "_open_upload_sources",
        lambda _files: [(source, descriptor, info)],
    )
    monkeypatch.setattr(
        client,
        "_base_url",
        lambda: (_ for _ in ()).throw(
            client.VideoClientError("invalid base")
        ),
    )
    monkeypatch.setattr(client.os, "close", closed.append)

    with pytest.raises(client.VideoClientError):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    assert closed == [descriptor]


def test_upload_closes_current_descriptor_when_fstat_fails(monkeypatch, tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"safe")
    descriptor = 43
    closed = []
    monkeypatch.setattr(client.os, "open", lambda *_args, **_kwargs: descriptor)
    monkeypatch.setattr(
        client.os,
        "fstat",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fstat failed")),
    )
    monkeypatch.setattr(client.os, "close", closed.append)

    with pytest.raises(OSError):
        client._open_upload_sources([source])

    assert closed == [descriptor]


def test_download_uses_random_no_follow_part_file(monkeypatch, tmp_path):
    target = tmp_path / "result.mp4"
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"keep")
    (tmp_path / "result.mp4.part").symlink_to(victim)

    class Response:
        status = 200

        def __init__(self):
            self.sent = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            if self.sent:
                return b""
            self.sent = True
            return b"rendered"

    class Opener:
        def open(self, *_args, **_kwargs):
            return Response()

    monkeypatch.setattr(client.urllib.request, "build_opener", lambda *_args: Opener())
    monkeypatch.setattr(
        client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (client.socket.AF_INET, client.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))
        ],
    )

    evidence = client.download("https://cdn.example.test/result.mp4", target)

    assert target.read_bytes() == b"rendered"
    assert victim.read_bytes() == b"keep"
    assert evidence["size"] == len(b"rendered")
