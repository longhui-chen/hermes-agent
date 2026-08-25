from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import io
import os
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from plugins.video_edit import client, normalizer, paths, preferences, schemas, tools
import plugins.video_edit as video_plugin


_REAL_INSPECT_FILES = normalizer.inspect_files


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


@pytest.fixture(autouse=True)
def trusted_upload_media_probe(tmp_path, monkeypatch):
    _install_trusted_normalizer(tmp_path / "fixture", monkeypatch)

    def inspect(sources, _workflow_id):
        identities = []
        for source in sources:
            info = source.stat()
            identities.append(
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
            )
        return identities

    monkeypatch.setattr(normalizer, "inspect_files", inspect)


def _video_sample(suffix: str, marker: bytes = b"") -> bytes:
    suffix = suffix.lower()
    if suffix in {".3g2", ".3gp", ".m4v", ".mov", ".mp4"}:
        brand = b"qt  " if suffix == ".mov" else b"isom"
        payload = brand + b"\x00\x00\x00\x00" + brand + b"mp42"
        return (len(payload) + 8).to_bytes(4, "big") + b"ftyp" + payload + marker
    if suffix in {".mkv", ".webm"}:
        doctype = b"matroska" if suffix == ".mkv" else b"webm"
        return (
            b"\x1a\x45\xdf\xa3"
            + b"\x42\x82"
            + bytes([0x80 | len(doctype)])
            + doctype
            + b"\x83\x81\x01"
            + marker
        )
    if suffix == ".avi":
        body = b"AVI LIST\x00\x00\x00\x00strhvids" + marker
        return b"RIFF" + len(body).to_bytes(4, "little") + body
    if suffix == ".flv":
        return b"FLV\x01\x01\x00\x00\x00\x09" + marker
    if suffix in {".asf", ".wmv"}:
        return (
            bytes.fromhex("3026b2758e66cf11a6d900aa0062ce6c")
            + bytes.fromhex("c0ef19bc4d5bcf11a8fd00805f5c442b")
            + marker
        )
    if suffix == ".mxf":
        return bytes.fromhex("060e2b34020501010d01020101020400") + marker
    if suffix == ".ogv":
        return b"OggS\x00" + b"\x00" * 20 + b"\x80theora" + marker
    if suffix in {".mpeg", ".mpg"}:
        return b"\x00\x00\x01\xba" + b"\x00" * 12 + b"\x00\x00\x01\xe0" + marker
    if suffix in {".m2ts", ".mts", ".ts"}:
        packet_size = 192 if suffix == ".m2ts" else 188
        sync_offset = 4 if suffix == ".m2ts" else 0
        sample = bytearray(packet_size * 3)
        for index in range(3):
            sample[sync_offset + packet_size * index] = 0x47
        sample[sync_offset + 8 : sync_offset + 12] = b"\x00\x00\x01\xe0"
        sample.extend(marker)
        return bytes(sample)
    raise AssertionError(f"missing video sample for {suffix}")


def _write_video(path: Path, marker: bytes = b"") -> None:
    path.write_bytes(_video_sample(path.suffix, marker))


def _install_trusted_normalizer(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(normalizer, "_PRESETS_ANCHOR", None)
    presets = tmp_path / "presets"
    script = (
        presets
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / "normalize.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# trusted test adapter\n", encoding="utf-8")
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(presets))
    return script


def test_registers_all_video_tools_without_capability_gate():
    context = _Context()
    video_plugin.register(context)

    names = {item["name"] for item in context.registrations}
    assert names == set(tools.HANDLERS)
    assert len(names) == 9
    assert all(item["toolset"] == "video_edit" for item in context.registrations)
    upload = next(item for item in context.registrations if item["name"] == "video_edit_upload_assets")
    assert upload["schema"]["parameters"]["required"] == []
    assert tools.schemas.business_required_names("video_edit_upload_assets") == [
        "workflow_id"
    ]
    serialized = json.dumps(
        [{key: value for key, value in item.items() if key != "handler"} for item in context.registrations],
        ensure_ascii=False,
    )
    assert "ActionV1" not in serialized
    assert "BusinessExecution" not in serialized


def test_register_pins_unavailable_release_without_disabling_tools_or_help(
    tmp_path,
    monkeypatch,
):
    unavailable = tmp_path / "unavailable-presets"
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(unavailable))
    monkeypatch.setattr(normalizer, "_PRESETS_ANCHOR", None)
    context = _Context()

    video_plugin.register(context)

    assert normalizer._PRESETS_ANCHOR is normalizer._PRESETS_UNAVAILABLE
    assert len(context.registrations) == 9
    upload = next(
        item
        for item in context.registrations
        if item["name"] == "video_edit_upload_assets"
    )
    help_result = json.loads(upload["handler"]({"help": True}))
    assert help_result["tool"] == "video_edit_upload_assets"
    assert help_result["side_effects"] == "none"

    script = (
        unavailable
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / "normalize.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# appeared after registration\n", encoding="utf-8")
    with pytest.raises(normalizer.NormalizeError, match="unavailable"):
        normalizer.normalizer_script()


def test_bundled_manifest_is_discovered_and_executes_runtime_tools(monkeypatch, isolated_video_home):
    import hermes_cli.plugins as plugins_module
    from hermes_cli.plugins import PluginManager
    import tools.registry as registry_module

    fresh_registry = registry_module.ToolRegistry()
    monkeypatch.setattr(registry_module, "registry", fresh_registry)
    monkeypatch.setenv(
        "ZETTLAB_PRESETS_DIR",
        str(isolated_video_home[0] / "missing-presets"),
    )
    monkeypatch.setattr(normalizer, "_PRESETS_ANCHOR", None)
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

    help_result = json.loads(
        fresh_registry.dispatch(
            "video_edit_upload_assets",
            {"help": True},
            agent_id="agent-a",
        )
    )
    assert help_result["tool"] == "video_edit_upload_assets"
    assert help_result["side_effects"] == "none"

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
    _write_video(source, b"source")
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setenv("ZETTLAB_AGENT_ACTION_TOKEN", "must-not-reach-helper")
    monkeypatch.setenv("ZETTLAB_BUSINESS_EXECUTION_TOKEN", "retired")
    calls = []

    def fake_run(command, **kwargs):
        input_path = Path(command[command.index("--input") + 1])
        script_descriptor = kwargs["pass_fds"][0]
        calls.append(
            (
                command,
                kwargs,
                input_path.stat().st_ino,
                normalizer.os.fstat(script_descriptor).st_ino,
            )
        )
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
    assert calls and calls[0][0][0] == normalizer.sys.executable
    assert calls[0][0][1] != str(script)
    assert calls[0][0][1].endswith(f"/{calls[0][1]['pass_fds'][0]}")
    assert calls[0][3] == script.stat().st_ino
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
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    untrusted = tmp_path / "untrusted" / "normalize.py"
    untrusted.parent.mkdir()
    untrusted.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
    monkeypatch.setenv("ZETTLAB_VIDEO_NORMALIZER", str(untrusted))

    assert normalizer.normalizer_script() == script.resolve()


def test_register_pins_the_version_before_the_first_business_call(monkeypatch, tmp_path):
    presets = tmp_path / "presets"
    release = presets / "v3.1.0"
    next_release = presets / "v3.1.1"
    script = (
        release
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / "normalize.py"
    )
    script.parent.mkdir(parents=True)
    script.write_text("# versioned test adapter\n", encoding="utf-8")
    next_script = (
        next_release
        / "skills"
        / "video-edit-workflow-mini"
        / "scripts"
        / "normalize.py"
    )
    next_script.parent.mkdir(parents=True)
    next_script.write_text("# next version adapter\n", encoding="utf-8")
    current = presets / "current"
    current.symlink_to(release.name, target_is_directory=True)
    monkeypatch.setenv("ZETTLAB_PRESETS_DIR", str(current))

    context = _Context()
    video_plugin.register(context)

    current.unlink()
    current.symlink_to(next_release.name, target_is_directory=True)

    assert normalizer.normalizer_script() == script.resolve()
    first_generation = normalizer.generation()
    assert normalizer.normalizer_script() == script.resolve()
    assert normalizer.generation() == first_generation

    normalizer._PRESETS_ANCHOR = None
    assert normalizer.normalizer_script() == next_script.resolve()
    assert normalizer.generation() != first_generation


def test_normalizer_rejects_script_replacement_after_pinning(monkeypatch, tmp_path):
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    assert normalizer.normalizer_script() == script.resolve()

    script.unlink()
    script.write_text("# replaced adapter\n", encoding="utf-8")

    with pytest.raises(normalizer.NormalizeError, match="unavailable"):
        normalizer.normalizer_script()


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "O_NOFOLLOW"),
    reason="read-only inherited descriptors require POSIX O_NOFOLLOW",
)
def test_normalizer_executes_opened_inode_when_script_path_is_replaced(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.mov"
    _write_video(source, b"source")
    script = _install_trusted_normalizer(tmp_path, monkeypatch)
    script.write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--input', required=True)\n"
        "parser.add_argument('--output', required=True)\n"
        "args = parser.parse_args()\n"
        "Path(args.output).write_bytes(b'anchored-inode')\n"
        "print(json.dumps({'output': args.output}))\n",
        encoding="utf-8",
    )
    assert normalizer.initialize_runtime() is True
    real_popen = normalizer.subprocess.Popen
    replaced = False
    inherited: list[tuple[str, tuple[int, ...]]] = []

    def replace_then_spawn(command, **kwargs):
        nonlocal replaced
        inherited.append((command[1], kwargs.get("pass_fds", ())))
        if not replaced:
            replaced = True
            script.unlink()
            script.write_text(
                "raise RuntimeError('replacement must not execute')\n",
                encoding="utf-8",
            )
        return real_popen(command, **kwargs)

    monkeypatch.setattr(normalizer.subprocess, "Popen", replace_then_spawn)

    result = normalizer.normalize_file(source, "workflow-fd-pinning", 0)

    assert result.read_bytes() == b"anchored-inode"
    assert replaced is True
    assert len(inherited) == 1
    script_path, pass_fds = inherited[0]
    assert len(pass_fds) == 1
    assert script_path.endswith(f"/{pass_fds[0]}")
    normalizer.cleanup([result], "workflow-fd-pinning")


@pytest.mark.parametrize("normalize", [True, False])
def test_checkpoint_normalize_strategy_accepts_only_boolean_values(normalize):
    assert tools._checkpoint_normalize_strategy({"normalize": normalize}) is normalize


@pytest.mark.parametrize(
    "entry",
    [{}, {"normalize": None}, {"normalize": 0}, {"normalize": 1}, {"normalize": "raw_direct"}],
)
def test_checkpoint_normalize_strategy_rejects_missing_or_invalid_values(entry):
    with pytest.raises(tools._WorkflowUnavailable, match="checkpoint is invalid"):
        tools._checkpoint_normalize_strategy(entry)


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
    _write_video(source, b"video")
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
    report_payload = {
        "manifest_id": "manifest-1",
        "output_path": "/trusted/output.mp4",
    }
    first_report = client._headers("result", report_payload, agent_id="agent-a")
    crash_retry = client._headers("result", report_payload, agent_id="agent-a")
    assert first_report["Idempotency-Key"] == crash_retry["Idempotency-Key"]
    assert "Authorization" not in first_report

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
    _write_video(first, b"first")
    _write_video(second, b"second")
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
    assert rejected["reason_code"] == "workflow_unavailable"
    assert rejected["retryable"] is False
    assert rejected["next"] is None
    assert tools.state.get(workflow, "agent-a")["object_keys"] == ["old-object"]


def test_source_fingerprint_ignores_ctime_only_but_detects_inode_replacement():
    info = SimpleNamespace(
        st_dev=7,
        st_ino=11,
        st_size=4096,
        st_mtime_ns=1_700_000_000_000_000_000,
        st_ctime_ns=1_700_000_000_000_000_001,
    )

    class StablePath:
        def __str__(self):
            return "/volume1/subvol/data/same.mov"

        def stat(self):
            return info

    source = StablePath()
    original = tools._source_fingerprint([source])
    info.st_ctime_ns += 1

    assert tools._source_fingerprint([source]) == original

    info.st_ino += 1
    assert tools._source_fingerprint([source]) != original


def test_upload_retry_survives_admission_ctime_change_after_transient_failure(
    isolated_video_home,
    monkeypatch,
    tmp_path,
):
    source = isolated_video_home[1] / "agent-a" / "retry-after-admission.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"retry-after-admission")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "retry-after-admission"}, agent_id="agent-a"
        )
    )["workflow_id"]

    _install_trusted_normalizer(tmp_path, monkeypatch)
    monkeypatch.setattr(normalizer, "inspect_files", _REAL_INSPECT_FILES)
    admission_ctimes = []

    def inspect(command, *, env, timeout, pass_fds):
        assert len(pass_fds) == 1
        assert command[1].endswith(f"/{pass_fds[0]}")
        admission_ctimes.append(source.stat().st_ctime_ns)
        payload = {
            "ok": True,
            "items": [{"category": "direct_only", "direct_ok": True}],
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), ""), False

    monkeypatch.setattr(normalizer, "_run_bounded_subprocess", inspect)
    connection_attempts = 0

    class Response:
        status = 200

        def read(self, _size):
            return b'{"data":{"uploads":[{"object_key":"asset-1"}]}}'

    class Connection:
        def __init__(self, *_args, **_kwargs):
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts == 1:
                raise OSError("transient connection failure")

        def putrequest(self, *_args):
            return None

        def putheader(self, *_args):
            return None

        def endheaders(self):
            return None

        def send(self, _chunk):
            return None

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)
    initial_ctime = source.stat().st_ctime_ns
    time.sleep(0.01)

    transient = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]},
            agent_id="agent-a",
        )
    )

    assert transient["reason_code"] == "transient_failure"
    assert transient["retryable"] is True
    assert source.stat().st_ctime_ns != initial_ctime
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["source_fingerprint"] == tools._source_fingerprint([source])

    retried = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]},
            agent_id="agent-a",
        )
    )

    assert retried["ok"] is True
    assert retried["uploaded"] == 1
    assert admission_ctimes and len(admission_ctimes) == 2
    assert connection_attempts == 2
    assert tools.state.get(workflow, "agent-a")["object_keys"] == ["asset-1"]


def test_incomplete_upload_same_source_path_replacement_requires_new_task(
    isolated_video_home, monkeypatch
):
    source = isolated_video_home[1] / "agent-a" / "same-path.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"version-one")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "source-replacement"}, agent_id="agent-a"
        )
    )["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": [str(source)],
            "source_names": [source.name],
            "source_fingerprint": tools._source_fingerprint([source]),
            "normalize": False,
            "status": "uploading",
        },
    )
    checkpoint = tools.state.get(workflow, "agent-a")
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail(
            "changed media must not resume an incomplete upload"
        ),
    )

    _write_video(source, b"version-two")
    rejected = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]}, agent_id="agent-a"
        )
    )

    assert rejected["reason_code"] == "workflow_unavailable"
    assert rejected["retryable"] is False
    assert rejected["next"] is None
    assert "new task_id" in rejected["error"]
    assert tools.state.get(workflow, "agent-a") == checkpoint


def test_complete_upload_checkpoint_reuses_after_sources_are_removed_without_media_access(
    isolated_video_home, monkeypatch
):
    sources = [
        isolated_video_home[1] / "agent-a" / "complete-a.mov",
        isolated_video_home[1] / "agent-a" / "complete-b.mov",
    ]
    sources[0].parent.mkdir(parents=True)
    for index, source in enumerate(sources):
        _write_video(source, f"video-{index}".encode())
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "complete-upload-retry"}, agent_id="agent-a"
        )
    )["workflow_id"]
    monkeypatch.setattr(
        client,
        "upload",
        lambda files, **_kwargs: {
            "data": {
                "uploads": [
                    {"object_key": f"assets/complete-{index}"}
                    for index, _source in enumerate(files)
                ]
            }
        },
    )
    first = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(path) for path in sources]},
            agent_id="agent-a",
        )
    )
    assert first["uploaded"] == len(sources), first
    checkpoint = tools.state.get(workflow, "agent-a")
    for source in sources:
        source.unlink()

    def unexpected_media_access(*_args, **_kwargs):
        pytest.fail("a complete upload checkpoint must not access local media")

    monkeypatch.setattr(tools, "_files_for_upload", unexpected_media_access)
    monkeypatch.setattr(tools, "_source_fingerprint", unexpected_media_access)
    monkeypatch.setattr(tools, "_should_normalize", unexpected_media_access)
    monkeypatch.setattr(client, "upload", unexpected_media_access)

    for retry_args in (
        {"workflow_id": workflow},
        {"workflow_id": workflow, "files": [str(path) for path in sources]},
    ):
        reused = json.loads(
            tools.handle_upload_assets(retry_args, agent_id="agent-a")
        )
        assert reused["ok"] is True
        assert reused["uploaded"] == len(sources)
        assert reused["reused"] is True
        assert reused["strategy"] == "raw_direct"
        assert reused["next"] == "video_edit_create_project"

    assert tools.state.get(workflow, "agent-a") == checkpoint


def test_complete_normalized_upload_reuses_without_current_generation(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_complete-normalized-generation"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/removed/source.mov"],
            "source_fingerprint": "completed-source",
            "object_keys": ["assets/completed-source"],
            "normalize": True,
            "normalizer_generation": "previous-generation",
            "status": "assets_uploaded",
        },
    )
    monkeypatch.setattr(
        normalizer,
        "generation",
        lambda: pytest.fail("completed upload must not inspect runtime generation"),
    )
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail("completed upload must not replay"),
    )

    reused = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert reused["ok"] is True
    assert reused["reused"] is True
    assert reused["strategy"] == "normalized"
    assert reused["next"] == "video_edit_create_project"


def test_upload_batch_must_return_one_object_for_each_source(isolated_video_home, monkeypatch):
    source = isolated_video_home[1] / "agent-a" / "missing-result.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"video")
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

    assert result["reason_code"] == "service_request_rejected"
    assert result["retryable"] is False
    assert result["next"] is None
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
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-download-crash-window",
            "project": {"status": "completed"},
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


def test_expired_result_url_repolls_existing_project_and_downloads_refreshed_url(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_expired-result-url"
    project_id = "project-expired-result-url"
    stale_url = "https://cdn.example.test/stale.mp4"
    refreshed_url = "https://cdn.example.test/refreshed.mp4"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "task_id": "expired-result-url",
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "source_fingerprint": "stable-source",
            "object_keys": ["assets/source"],
            "project_id": project_id,
            "project": {
                "project_id": project_id,
                "status": "completed",
                "result_url": stale_url,
            },
            "result_url": stale_url,
            "status": "completed",
        },
    )
    downloads = []
    polls = []

    def fake_download(url, target):
        downloads.append(url)
        if url == stale_url:
            raise client.ResultURLUnavailable(
                "expired signed URL",
                status=404,
                transient=True,
            )
        target.write_bytes(b"refreshed-render")
        return client.file_evidence(target)

    def fake_poll(requested_project_id, **_kwargs):
        polls.append(requested_project_id)
        return {
            "project_id": requested_project_id,
            "status": "completed",
            "result_url": refreshed_url,
        }

    monkeypatch.setattr(client, "download", fake_download)
    monkeypatch.setattr(client, "poll_project", fake_poll)
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail("recovery must not reupload"),
    )
    monkeypatch.setattr(
        client,
        "create_project",
        lambda *_args, **_kwargs: pytest.fail("recovery must not rebuild"),
    )

    delivered = json.loads(
        tools.handle_download_result(
            {"workflow_id": workflow, "filename": "recovered.mp4"},
            agent_id="agent-a",
        )
    )

    assert delivered["ok"] is True
    assert downloads == [stale_url, refreshed_url]
    assert polls == [project_id]
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["task_id"] == "expired-result-url"
    assert checkpoint["project_id"] == project_id
    assert checkpoint["object_keys"] == ["assets/source"]
    assert checkpoint["result_url"] == refreshed_url
    assert checkpoint["result_url_refresh_required"] is False
    assert stale_url not in json.dumps(delivered)
    assert refreshed_url not in json.dumps(delivered)


def test_unrenewed_result_url_is_not_downloaded_again_before_next_repoll(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_result-url-refresh-pending"
    project_id = "project-result-url-refresh-pending"
    stale_url = "https://cdn.example.test/still-stale.mp4"
    refreshed_url = "https://cdn.example.test/finally-refreshed.mp4"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": project_id,
            "project": {
                "project_id": project_id,
                "status": "completed",
                "result_url": stale_url,
            },
            "result_url": stale_url,
            "status": "completed",
        },
    )
    downloads = []
    polled_urls = iter([stale_url, refreshed_url])
    polls = []

    def fake_download(url, target):
        downloads.append(url)
        if url == stale_url:
            raise client.ResultURLUnavailable(
                "expired signed URL",
                status=404,
                transient=True,
            )
        target.write_bytes(b"rendered-after-refresh")
        return client.file_evidence(target)

    def fake_poll(requested_project_id, **_kwargs):
        polls.append(requested_project_id)
        return {
            "project_id": requested_project_id,
            "status": "completed",
            "result_url": next(polled_urls),
        }

    monkeypatch.setattr(client, "download", fake_download)
    monkeypatch.setattr(client, "poll_project", fake_poll)

    pending = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert pending["reason_code"] == "transient_failure"
    assert pending["next"] == "video_edit_download_result"
    assert downloads == [stale_url]
    pending_checkpoint = tools.state.get(workflow, "agent-a")
    assert pending_checkpoint["result_url_refresh_required"] is True
    assert pending_checkpoint["project_id"] == project_id
    assert pending_checkpoint["object_keys"] == ["assets/source"]

    delivered = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert delivered["ok"] is True
    assert downloads == [stale_url, refreshed_url]
    assert polls == [project_id, project_id]
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["project_id"] == project_id
    assert checkpoint["result_url"] == refreshed_url
    assert checkpoint["result_url_refresh_required"] is False


def test_result_url_repoll_failure_preserves_project_and_upload_checkpoints(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_result-url-repoll-failure"
    project_id = "project-result-url-repoll-failure"
    stale_url = "https://cdn.example.test/expired-before-repoll.mp4"
    object_keys = ["assets/source-a", "assets/source-b"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": [
                "/volume1/subvol/data/source-a.mov",
                "/volume1/subvol/data/source-b.mov",
            ],
            "object_keys": object_keys,
            "project_id": project_id,
            "project": {
                "project_id": project_id,
                "status": "completed",
                "result_url": stale_url,
            },
            "result_url": stale_url,
            "status": "completed",
        },
    )
    monkeypatch.setattr(
        client,
        "download",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            client.ResultURLUnavailable(
                "expired signed URL",
                status=404,
                transient=True,
            )
        ),
    )
    monkeypatch.setattr(
        client,
        "poll_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            client.VideoClientError("project poll unavailable", transient=True)
        ),
    )

    result = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )

    assert result["reason_code"] == "transient_failure"
    assert result["next"] == "video_edit_download_result"
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["project_id"] == project_id
    assert checkpoint["object_keys"] == object_keys
    assert checkpoint["result_url"] == stale_url
    assert checkpoint["result_url_refresh_required"] is True


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
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-outside-output",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(outside),
            "status": "delivered",
        },
    )

    delivered = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )

    assert "error" in delivered
    assert delivered["reason_code"] == "workflow_unavailable"
    assert delivered["retryable"] is False
    assert delivered["next"] is None
    assert delivered["recovery"] == tools.schemas.error_contract(
        "video_edit_download_result", "workflow_unavailable"
    )["recovery"]


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


@pytest.mark.parametrize(
    ("suffix", "expected_mime"),
    [
        (".3g2", "video/3gpp2"),
        (".3gp", "video/3gpp"),
        (".asf", "video/x-ms-asf"),
        (".avi", "video/x-msvideo"),
        (".flv", "video/x-flv"),
        (".m2ts", "video/mp2t"),
        (".m4v", "video/mp4"),
        (".mkv", "video/x-matroska"),
        (".mov", "video/quicktime"),
        (".mp4", "video/mp4"),
        (".mpeg", "video/mpeg"),
        (".mpg", "video/mpeg"),
        (".mts", "video/mp2t"),
        (".mxf", "application/mxf"),
        (".ogv", "video/ogg"),
        (".ts", "video/mp2t"),
        (".webm", "video/webm"),
        (".wmv", "video/x-ms-wmv"),
    ],
)
def test_video_media_admission_has_a_fixed_container_and_mime_contract(
    suffix,
    expected_mime,
):
    path = Path(f"/trusted/source{suffix}")

    assert paths.validate_video_sample(path, _video_sample(suffix)) == expected_mime
    assert paths.video_media_type(path) == expected_mime
    assert expected_mime != "application/octet-stream"


@pytest.mark.parametrize(
    ("name", "sample"),
    [
        ("document.txt", _video_sample(".mp4")),
        ("renamed.mp4", b"%PDF-1.7\nnot video"),
        ("wrong-container.mp4", _video_sample(".webm")),
        ("audio.ogv", b"OggS\x00" + b"\x00" * 20 + b"\x01vorbis"),
        (
            "audio.mp4",
            b"\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00isommp42",
        ),
        (
            "audio.mkv",
            b"\x1a\x45\xdf\xa3\x42\x82\x88matroska\x83\x81\x02",
        ),
        ("audio.flv", b"FLV\x01\x04\x00\x00\x00\x09"),
        ("audio.avi", b"RIFF\x10\x00\x00\x00AVI LISTstrhauds"),
        (
            "audio.mpeg",
            b"\x00\x00\x01\xba" + b"\x00" * 12 + b"\x00\x00\x01\xc0",
        ),
        ("audio.m4a", _video_sample(".mp4")),
    ],
)
def test_video_media_admission_rejects_unknown_mismatched_and_non_video_files(
    name,
    sample,
):
    with pytest.raises(paths.VideoPathError, match="supported video"):
        paths.validate_video_sample(Path(name), sample)


def test_video_descriptor_sniff_reads_only_the_bounded_header(monkeypatch, tmp_path):
    source = tmp_path / "bounded.mp4"
    _write_video(source, b"x" * (paths.VIDEO_HEADER_BYTES * 2))
    descriptor = os.open(source, os.O_RDONLY)
    read_sizes = []
    real_read = os.read
    monkeypatch.setattr(
        paths.os,
        "read",
        lambda fd, size: read_sizes.append(size) or real_read(fd, size),
    )
    try:
        paths.validate_video_descriptor(source, descriptor)
    finally:
        os.close(descriptor)

    assert read_sizes == [paths.VIDEO_HEADER_BYTES]


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("search-result.txt", b"plain document"),
        ("renamed-result.mp4", b"%PDF-1.7\nrenamed document"),
    ],
)
def test_search_handoff_non_video_is_rejected_before_prepare_or_upload(
    isolated_video_home,
    monkeypatch,
    filename,
    content,
):
    source = isolated_video_home[1] / "agent-a" / filename
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": f"non-video-{filename}"},
            agent_id="agent-a",
        )
    )["workflow_id"]
    monkeypatch.setattr(
        normalizer,
        "normalize_files",
        lambda *_args, **_kwargs: pytest.fail("non-video input must not be normalized"),
    )
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail("non-video input must not be uploaded"),
    )

    result = json.loads(
        tools.handle_upload_assets(
            {
                "workflow_id": workflow,
                "files": [str(source)],
                "normalize": True,
            },
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "invalid_arguments"
    assert result["retryable"] is True
    assert result["next"] == "video_edit_upload_assets"
    assert "supported video" in result["error"]
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint.get("source_paths") in (None, [])
    assert checkpoint.get("object_keys") in (None, [])


@pytest.mark.parametrize("failure_mode", ["non_video", "missing"])
def test_persisted_source_becoming_invalid_is_terminal_and_preserves_checkpoint(
    isolated_video_home,
    monkeypatch,
    failure_mode,
):
    source = isolated_video_home[1] / "agent-a" / "remembered.mp4"
    source.parent.mkdir(parents=True)
    _write_video(source, b"original-video")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "persisted-source-became-invalid"},
            agent_id="agent-a",
        )
    )["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": [str(source)],
            "source_names": [source.name],
            "source_fingerprint": tools._source_fingerprint([source]),
            "normalize": False,
            "status": "uploading",
        },
    )
    checkpoint = tools.state.get(workflow, "agent-a")
    if failure_mode == "non_video":
        source.write_bytes(b"%PDF-1.7\nreplaced document")
    else:
        source.unlink()
    monkeypatch.setattr(
        normalizer,
        "normalize_files",
        lambda *_args, **_kwargs: pytest.fail("invalid resume source must not be normalized"),
    )
    monkeypatch.setattr(
        client,
        "upload",
        lambda *_args, **_kwargs: pytest.fail("invalid resume source must not be uploaded"),
    )

    result = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None
    assert result["recovery"] == schemas.error_contract(
        "video_edit_upload_assets",
        "workflow_unavailable",
    )["recovery"]
    assert tools.state.get(workflow, "agent-a") == checkpoint


def test_upload_normalized_intermediates_are_cleaned_after_each_batch(isolated_video_home, monkeypatch, tmp_path):
    source = isolated_video_home[1] / "agent-a" / "input.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"video")
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
    _write_video(source, b"video")
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


def test_partial_normalized_upload_retry_keeps_normalized_strategy(
    isolated_video_home,
    monkeypatch,
    tmp_path,
):
    sources = [
        isolated_video_home[1] / "agent-a" / f"normalized-retry-{index}.mov"
        for index in range(4)
    ]
    sources[0].parent.mkdir(parents=True)
    for index, source in enumerate(sources):
        _write_video(source, f"source-{index}".encode())
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {
                "task_id": "partial-normalized-retry",
                "preferences": {"upload_preference": "normalized"},
            },
            agent_id="agent-a",
        )
    )["workflow_id"]
    normalize_calls = 0

    def fake_normalize(files, _workflow_id):
        nonlocal normalize_calls
        normalize_calls += 1
        outputs = []
        for index, _source in enumerate(files):
            target = tmp_path / f"vewm_{index}.mp4"
            target.write_bytes(f"normalized-{index}".encode())
            outputs.append(target)
        return outputs

    monkeypatch.setattr(normalizer, "normalize_files", fake_normalize)
    monkeypatch.setattr(
        normalizer,
        "cleanup",
        lambda items, _workflow_id: [item.unlink(missing_ok=True) for item in items],
    )
    upload_calls = 0
    uploaded_batches = []

    def upload(files, **_kwargs):
        nonlocal upload_calls
        upload_calls += 1
        uploaded_batches.append([path.name for path in files])
        if upload_calls == 2:
            raise client.VideoClientError("temporary upload failure", transient=True)
        return {
            "data": {
                "uploads": [
                    {"object_key": f"assets/{path.name}"} for path in files
                ]
            }
        }

    monkeypatch.setattr(client, "upload", upload)

    first = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(path) for path in sources]},
            agent_id="agent-a",
        )
    )

    assert first["reason_code"] == "transient_failure"
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["normalize"] is True
    assert checkpoint["normalizer_generation"] == normalizer.generation()
    assert len(checkpoint["object_keys"]) == 3

    retried = json.loads(
        tools.handle_upload_assets(
            {
                "workflow_id": workflow,
                "files": [str(path) for path in sources],
                "normalize": False,
            },
            agent_id="agent-a",
        )
    )

    assert retried["ok"] is True
    assert retried["strategy"] == "normalized"
    assert retried["uploaded"] == 4
    assert normalize_calls == 2
    assert uploaded_batches == [
        ["vewm_0.mp4", "vewm_1.mp4", "vewm_2.mp4"],
        ["vewm_3.mp4"],
        ["vewm_3.mp4"],
    ]
    assert tools.state.get(workflow, "agent-a")["normalize"] is True


def test_partial_raw_fallback_retry_keeps_raw_strategy(
    isolated_video_home,
    monkeypatch,
):
    sources = [
        isolated_video_home[1] / "agent-a" / f"raw-retry-{index}.mov"
        for index in range(4)
    ]
    sources[0].parent.mkdir(parents=True)
    for index, source in enumerate(sources):
        _write_video(source, f"source-{index}".encode())
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {
                "task_id": "partial-raw-fallback-retry",
                "preferences": {"upload_preference": "normalized"},
            },
            agent_id="agent-a",
        )
    )["workflow_id"]
    normalize_calls = 0

    def unavailable_normalizer(_files, _workflow_id):
        nonlocal normalize_calls
        normalize_calls += 1
        raise normalizer.NormalizeError("unsupported media")

    monkeypatch.setattr(normalizer, "normalize_files", unavailable_normalizer)
    upload_calls = 0
    uploaded_batches = []

    def upload(files, **_kwargs):
        nonlocal upload_calls
        upload_calls += 1
        uploaded_batches.append([path.name for path in files])
        if upload_calls == 2:
            raise client.VideoClientError("temporary upload failure", transient=True)
        return {
            "data": {
                "uploads": [
                    {"object_key": f"assets/{path.name}"} for path in files
                ]
            }
        }

    monkeypatch.setattr(client, "upload", upload)

    first = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(path) for path in sources]},
            agent_id="agent-a",
        )
    )

    assert first["reason_code"] == "transient_failure"
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["normalize"] is False
    assert checkpoint["normalizer_generation"] == ""
    assert len(checkpoint["object_keys"]) == 3

    retried = json.loads(
        tools.handle_upload_assets(
            {
                "workflow_id": workflow,
                "files": [str(path) for path in sources],
                "normalize": True,
            },
            agent_id="agent-a",
        )
    )

    assert retried["ok"] is True
    assert retried["strategy"] == "raw_direct"
    assert retried["uploaded"] == 4
    assert normalize_calls == 1
    assert uploaded_batches == [
        ["raw-retry-0.mov", "raw-retry-1.mov", "raw-retry-2.mov"],
        ["raw-retry-3.mov"],
        ["raw-retry-3.mov"],
    ]
    assert tools.state.get(workflow, "agent-a")["normalize"] is False


def test_partial_normalized_upload_rejects_a_new_runtime_generation(
    isolated_video_home,
    monkeypatch,
):
    sources = [
        isolated_video_home[1] / "agent-a" / f"generation-retry-{index}.mov"
        for index in range(4)
    ]
    sources[0].parent.mkdir(parents=True)
    for index, source in enumerate(sources):
        _write_video(source, f"source-{index}".encode())
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {
                "task_id": "partial-generation-retry",
                "preferences": {"upload_preference": "normalized"},
            },
            agent_id="agent-a",
        )
    )["workflow_id"]
    generation = "generation-a"
    monkeypatch.setattr(normalizer, "generation", lambda: generation)
    monkeypatch.setattr(normalizer, "normalize_files", lambda files, _workflow_id: list(files))
    monkeypatch.setattr(normalizer, "cleanup", lambda _files, _workflow_id: None)
    upload_calls = 0

    def upload(files, **_kwargs):
        nonlocal upload_calls
        upload_calls += 1
        if upload_calls == 2:
            raise client.VideoClientError("temporary upload failure", transient=True)
        return {
            "data": {
                "uploads": [
                    {"object_key": f"assets/{path.name}"} for path in files
                ]
            }
        }

    monkeypatch.setattr(client, "upload", upload)
    first = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(path) for path in sources]},
            agent_id="agent-a",
        )
    )
    assert first["reason_code"] == "transient_failure"
    assert tools.state.get(workflow, "agent-a")["normalizer_generation"] == "generation-a"

    generation = "generation-b"
    retried = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(path) for path in sources]},
            agent_id="agent-a",
        )
    )

    assert retried["reason_code"] == "workflow_unavailable"
    assert retried["retryable"] is False
    assert retried["next"] is None
    assert upload_calls == 2


def test_accepted_normalized_upload_without_checkpoint_rejects_new_generation(
    isolated_video_home,
    monkeypatch,
):
    source = isolated_video_home[1] / "agent-a" / "accepted-response-lost.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"accepted-response-lost")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {
                "task_id": "accepted-response-lost",
                "preferences": {"upload_preference": "normalized"},
            },
            agent_id="agent-a",
        )
    )["workflow_id"]
    runtime_generation = "generation-before-restart"
    monkeypatch.setattr(
        normalizer,
        "generation",
        lambda: runtime_generation,
    )
    normalize_calls = 0

    def normalize(files, _workflow_id):
        nonlocal normalize_calls
        normalize_calls += 1
        return list(files)

    monkeypatch.setattr(normalizer, "normalize_files", normalize)
    monkeypatch.setattr(normalizer, "cleanup", lambda *_args, **_kwargs: None)
    accepted_batches: list[tuple[list[Path], str]] = []

    def accepted_but_response_lost(files, **kwargs):
        accepted_batches.append((list(files), kwargs["replay_scope"]))
        raise client.VideoClientError("upload response lost", transient=True)

    monkeypatch.setattr(client, "upload", accepted_but_response_lost)

    first = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow, "files": [str(source)]},
            agent_id="agent-a",
        )
    )

    assert first["reason_code"] == "transient_failure"
    checkpoint = tools.state.get(workflow, "agent-a")
    assert checkpoint["source_fingerprint"]
    assert checkpoint["normalize"] is True
    assert checkpoint["normalizer_generation"] == "generation-before-restart"
    assert checkpoint.get("object_keys", []) == []
    assert accepted_batches == [([source.resolve()], checkpoint["source_fingerprint"])]

    runtime_generation = "generation-after-restart"
    retried = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert retried["reason_code"] == "workflow_unavailable"
    assert retried["retryable"] is False
    assert retried["next"] is None
    assert normalize_calls == 1
    assert len(accepted_batches) == 1
    assert tools.state.get(workflow, "agent-a") == checkpoint


def test_proactive_report_is_exactly_once(isolated_video_home, monkeypatch, tmp_path):
    source = isolated_video_home[1] / "agent-a" / "weekly.mov"
    second_source = isolated_video_home[1] / "agent-a" / "weekly-2.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"video")
    _write_video(second_source, b"video-2")
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
    output = paths.result_path(
        "agent-a",
        "weekly.mp4",
        session_id="proactive-pvm-report-exactly-once",
    )
    output.write_bytes(b"rendered")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "project_id": "project-report-exactly-once",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(output),
            "status": "delivered",
            "proactive": True,
        },
    )
    calls = []
    monkeypatch.setattr(client, "proactive_report", lambda manifest_id, path, **kwargs: calls.append((manifest_id, path)) or {"ok": True})
    assert json.loads(tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a"))["reported"] is True
    assert json.loads(tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a"))["reused"] is True
    assert len(calls) == 1


def test_proactive_report_is_exactly_once_under_concurrent_calls(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_concurrent-proactive-report"
    trigger_id = "pvm-concurrent-report"
    output = paths.result_path(
        "agent-a",
        "weekly.mp4",
        session_id=f"proactive-{trigger_id}",
    )
    output.write_bytes(b"rendered")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/source-a.mov", "/source-b.mov"],
            "object_keys": ["assets/source-a", "assets/source-b"],
            "project_id": "project-concurrent-report",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(output),
            "manifest_id": "manifest-concurrent-report",
            "proactive_trigger_id": trigger_id,
            "proactive": True,
            "status": "delivered",
        },
    )
    start = threading.Barrier(3)
    report_entered = threading.Event()
    release_report = threading.Event()
    calls: list[tuple[str, str]] = []
    calls_lock = threading.Lock()

    def slow_report(manifest_id, path, **_kwargs):
        with calls_lock:
            calls.append((manifest_id, path))
        report_entered.set()
        assert release_report.wait(timeout=2)
        return {"ok": True}

    def invoke():
        start.wait(timeout=2)
        return json.loads(
            tools.handle_proactive_report(
                {"workflow_id": workflow},
                agent_id="agent-a",
            )
        )

    monkeypatch.setattr(client, "proactive_report", slow_report)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke) for _ in range(2)]
        start.wait(timeout=2)
        assert report_entered.wait(timeout=2)
        time.sleep(0.1)
        release_report.set()
        results = [future.result(timeout=2) for future in futures]

    assert calls == [("manifest-concurrent-report", str(output))]
    assert all(result["reported"] is True for result in results)
    assert sorted(bool(result.get("reused")) for result in results) == [False, True]


def test_proactive_report_rejects_symlinked_bounded_lock_file(
    isolated_video_home,
    monkeypatch,
    tmp_path,
):
    workflow = "vew_symlinked-report-lock"
    tools.state.update(
        workflow,
        "agent-a",
        {"proactive": True, "status": "delivered"},
    )
    bucket = (
        int.from_bytes(hashlib.sha256(workflow.encode()).digest()[:2], "big")
        % tools.state.REPORT_LOCK_BUCKETS
    )
    lock_path = (
        isolated_video_home[0]
        / "video_edit"
        / "agent-a"
        / f".report-{bucket:02d}.lock"
    )
    outside = tmp_path / "outside-lock-target"
    outside.write_text("sentinel", encoding="utf-8")
    lock_path.symlink_to(outside)
    monkeypatch.setattr(
        client,
        "proactive_report",
        lambda *_args, **_kwargs: pytest.fail("symlinked lock must fail closed"),
    )

    result = json.loads(
        tools.handle_proactive_report(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None
    assert outside.read_text(encoding="utf-8") == "sentinel"


def test_proactive_resolve_rejects_changed_identity_without_mutating_checkpoint(
    isolated_video_home,
    monkeypatch,
):
    first_source = isolated_video_home[1] / "agent-a" / "weekly-a.mov"
    first_extra = isolated_video_home[1] / "agent-a" / "weekly-a-extra.mov"
    first_source.parent.mkdir(parents=True)
    for source, payload in (
        (first_source, b"weekly-a"),
        (first_extra, b"weekly-a-extra"),
    ):
        _write_video(source, payload)

    manifests = {
        "manifest-a": {
            "trigger_id": "pvm-weekly-a",
            "scene": "weekly",
            "files": [{"path": str(first_source)}, {"path": str(first_extra)}],
        },
        "manifest-b": {
            "trigger_id": "pvm-weekly-a",
            "scene": "weekly",
            "files": [{"path": str(first_source)}, {"path": str(first_extra)}],
        },
    }
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda manifest_id, **_kwargs: {"data": manifests[manifest_id]},
    )

    first = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-a", "task_id": "weekly-shared"},
            agent_id="agent-a",
        )
    )
    workflow = first["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "object_keys": ["assets/weekly-a", "assets/weekly-a-extra"],
            "project_id": "project-weekly-a",
            "reported": True,
            "status": "reported",
        },
    )
    checkpoint = tools.state.get(workflow, "agent-a")

    changed = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-b", "task_id": "weekly-shared"},
            agent_id="agent-a",
        )
    )

    assert changed["reason_code"] == "workflow_unavailable"
    assert changed["retryable"] is False
    assert changed["next"] is None
    assert tools.state.get(workflow, "agent-a") == checkpoint


@pytest.mark.parametrize("changed_field", ["trigger", "sources", "scene"])
def test_proactive_resolve_rejects_mutated_manifest_payload_identity(
    isolated_video_home,
    monkeypatch,
    changed_field,
):
    source_a = isolated_video_home[1] / "agent-a" / "source-a.mov"
    source_b = isolated_video_home[1] / "agent-a" / "source-b.mov"
    replacement_a = isolated_video_home[1] / "agent-a" / "replacement-a.mov"
    replacement_b = isolated_video_home[1] / "agent-a" / "replacement-b.mov"
    source_a.parent.mkdir(parents=True)
    for source in (source_a, source_b, replacement_a, replacement_b):
        _write_video(source, source.name.encode())

    initial_payload = {
        "trigger_id": "pvm-stable-trigger",
        "scene": "weekly",
        "files": [{"path": str(source_a)}, {"path": str(source_b)}],
    }
    changed_payload = {
        "trigger_id": initial_payload["trigger_id"],
        "scene": initial_payload["scene"],
        "files": list(initial_payload["files"]),
    }
    if changed_field == "trigger":
        changed_payload["trigger_id"] = "pvm-replacement-trigger"
    elif changed_field == "sources":
        changed_payload["files"] = [
            {"path": str(replacement_a)},
            {"path": str(replacement_b)},
        ]
    else:
        changed_payload["scene"] = "travel"

    responses = iter((initial_payload, changed_payload))
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda _manifest_id, **_kwargs: {"data": next(responses)},
    )
    first = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-stable", "task_id": "weekly-stable"},
            agent_id="agent-a",
        )
    )
    workflow = first["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "object_keys": ["assets/source-a", "assets/source-b"],
            "project_id": "project-stable",
            "reported": True,
            "status": "reported",
        },
    )
    checkpoint = tools.state.get(workflow, "agent-a")

    changed = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-stable", "task_id": "weekly-stable"},
            agent_id="agent-a",
        )
    )

    assert changed["reason_code"] == "workflow_unavailable"
    assert changed["retryable"] is False
    assert changed["next"] is None
    assert tools.state.get(workflow, "agent-a") == checkpoint


def test_proactive_resolve_same_identity_preserves_existing_checkpoint(
    isolated_video_home,
    monkeypatch,
):
    source_a = isolated_video_home[1] / "agent-a" / "stable-a.mov"
    source_b = isolated_video_home[1] / "agent-a" / "stable-b.mov"
    source_a.parent.mkdir(parents=True)
    _write_video(source_a, b"stable-a")
    _write_video(source_b, b"stable-b")
    payload = {
        "trigger_id": "pvm-stable-resume",
        "scene": "weekly",
        "files": [{"path": str(source_a)}, {"path": str(source_b)}],
    }
    monkeypatch.setattr(
        client,
        "proactive_resolve",
        lambda _manifest_id, **_kwargs: {"data": payload},
    )
    first = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-resume", "task_id": "weekly-resume"},
            agent_id="agent-a",
        )
    )
    workflow = first["workflow_id"]
    tools.state.update(
        workflow,
        "agent-a",
        {
            "object_keys": ["assets/stable-a", "assets/stable-b"],
            "project_id": "project-resume",
            "reported": True,
            "status": "reported",
        },
    )

    resumed = json.loads(
        tools.handle_proactive_resolve(
            {"manifest_id": "manifest-resume", "task_id": "weekly-resume"},
            agent_id="agent-a",
        )
    )
    checkpoint = tools.state.get(workflow, "agent-a")

    assert resumed["ok"] is True
    assert resumed["workflow_id"] == workflow
    assert checkpoint["object_keys"] == ["assets/stable-a", "assets/stable-b"]
    assert checkpoint["project_id"] == "project-resume"
    assert checkpoint["reported"] is True
    assert checkpoint["status"] == "reported"


@pytest.mark.parametrize(
    "handler_name",
    ["video_edit_download_result", "video_edit_proactive_report"],
)
def test_missing_proactive_trigger_is_terminal_without_provider_call(
    isolated_video_home,
    monkeypatch,
    handler_name,
):
    workflow = f"vew_missing-trigger-{handler_name}"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-missing-trigger",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": "/untrusted/result.mp4",
            "manifest_id": "manifest-missing-trigger",
            "proactive": True,
            "status": "delivered",
        },
    )
    monkeypatch.setattr(
        client,
        "download",
        lambda *_args, **_kwargs: pytest.fail("terminal checkpoint must not download"),
    )
    monkeypatch.setattr(
        client,
        "proactive_report",
        lambda *_args, **_kwargs: pytest.fail("terminal checkpoint must not report"),
    )

    result = json.loads(
        tools.HANDLERS[handler_name]({"workflow_id": workflow}, agent_id="agent-a")
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None
    assert result["recovery"] == tools.schemas.error_contract(
        handler_name, "workflow_unavailable"
    )["recovery"]


def test_damaged_proactive_output_recovers_in_same_workflow_then_reports_once(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_damaged-proactive-output"
    trigger_id = "pvm-damaged-output"
    output = paths.result_path(
        "agent-a",
        "damaged.mp4",
        session_id=f"proactive-{trigger_id}",
    )
    output.write_bytes(b"")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-damaged-output",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(output),
            "manifest_id": "manifest-damaged-output",
            "proactive_trigger_id": trigger_id,
            "proactive": True,
            "status": "delivered",
        },
    )

    needs_download = json.loads(
        tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert needs_download["reason_code"] == "result_not_downloaded"
    assert needs_download["next"] == "video_edit_download_result"

    download_calls = []

    def flaky_download(_url, target):
        download_calls.append(target)
        if len(download_calls) == 1:
            raise client.VideoClientError(
                "temporary download failure",
                transient=True,
            )
        target.write_bytes(b"recovered-render")
        return client.file_evidence(target)

    monkeypatch.setattr(client, "download", flaky_download)
    first = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert first["reason_code"] == "transient_failure"
    assert first["next"] == "video_edit_download_result"

    delivered = json.loads(
        tools.handle_download_result({"workflow_id": workflow}, agent_id="agent-a")
    )
    assert delivered["ok"] is True
    assert delivered["workflow_id"] == workflow
    assert delivered["output"] == str(output)
    assert output.read_bytes() == b"recovered-render"
    assert download_calls == [output, output]

    reports = []
    monkeypatch.setattr(
        client,
        "proactive_report",
        lambda manifest_id, path, **_kwargs: reports.append((manifest_id, path))
        or {"ok": True},
    )
    assert json.loads(
        tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a")
    )["reported"] is True
    assert json.loads(
        tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a")
    )["reused"] is True
    assert reports == [("manifest-damaged-output", str(output))]


def test_proactive_report_rejects_output_outside_its_bucket(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_outside-proactive-output"
    isolated_video_home[0].mkdir(parents=True, exist_ok=True)
    outside = isolated_video_home[0] / "outside-proactive.mp4"
    outside.write_bytes(b"outside")
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/source.mov"],
            "object_keys": ["assets/source"],
            "project_id": "project-outside-proactive",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "output_path": str(outside),
            "manifest_id": "manifest-outside-proactive",
            "proactive_trigger_id": "pvm-outside-proactive",
            "proactive": True,
            "status": "delivered",
        },
    )
    monkeypatch.setattr(
        client,
        "proactive_report",
        lambda *_args, **_kwargs: pytest.fail("outside output must not report"),
    )

    result = json.loads(
        tools.handle_proactive_report({"workflow_id": workflow}, agent_id="agent-a")
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None


def test_proactive_download_uses_server_trigger_output_bucket(
    isolated_video_home, monkeypatch
):
    """Weekly artifacts must land in the same bucket local-server validates."""
    source = isolated_video_home[1] / "agent-a" / "weekly.mov"
    second_source = isolated_video_home[1] / "agent-a" / "weekly-2.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"video")
    _write_video(second_source, b"video-2")
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
        {
            "object_keys": ["assets/weekly", "assets/weekly-2"],
            "project_id": "project-trigger-bucket",
            "project": {"status": "completed"},
            "result_url": "https://cdn.example.test/result.mp4",
            "status": "completed",
        },
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
    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert result["next"] is None


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


@pytest.mark.parametrize("status", [302, 401, 403, 404, 410])
def test_download_classifies_expired_or_redirected_result_url_as_unavailable(
    monkeypatch,
    tmp_path,
    status,
):
    target = tmp_path / "result.mp4"

    class Opener:
        def open(self, request, **_kwargs):
            raise client.urllib.error.HTTPError(
                request.full_url,
                status,
                "result unavailable",
                {},
                None,
            )

    monkeypatch.setattr(
        client.urllib.request,
        "build_opener",
        lambda *_args: Opener(),
    )
    monkeypatch.setattr(
        client.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                client.socket.AF_INET,
                client.socket.SOCK_STREAM,
                0,
                "",
                ("93.184.216.34", 443),
            )
        ],
    )

    with pytest.raises(client.ResultURLUnavailable) as captured:
        client.download("https://cdn.example.test/result.mp4", target)

    assert captured.value.status == status
    assert captured.value.transient is True
    assert not target.exists()
    assert not list(tmp_path.glob(f".{target.name}.*.part"))


@pytest.mark.parametrize(
    ("source_paths", "object_keys"),
    [
        (
            [
                "/volume1/subvol/data/one.mov",
                "/volume1/subvol/data/two.mov",
            ],
            ["assets/one"],
        ),
        (["/volume1/subvol/data/one.mov"], ["assets/one", "assets/two"]),
        (["/volume1/subvol/data/one.mov"], [""]),
        ([""], ["assets/one"]),
        (
            ["/volume1/subvol/data/one.mov", ""],
            ["assets/one"],
        ),
        ({"unexpected": "source"}, ["assets/one"]),
        (["/volume1/subvol/data/one.mov"], 1),
    ],
)
def test_create_project_rejects_incomplete_upload_checkpoint(
    isolated_video_home,
    monkeypatch,
    source_paths,
    object_keys,
):
    workflow = "vew_partial-upload-checkpoint"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": source_paths,
            "object_keys": object_keys,
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
    assert result["reason_code"] == "upload_incomplete"
    assert called is False


def test_upload_rejects_extra_object_key_checkpoint_without_provider_call(
    isolated_video_home,
    monkeypatch,
):
    source = isolated_video_home[1] / "agent-a" / "extra-key.mov"
    source.parent.mkdir(parents=True)
    _write_video(source, b"video")
    workflow = json.loads(
        tools.handle_preferences_resolve(
            {"task_id": "extra-object-key"},
            agent_id="agent-a",
        )
    )["workflow_id"]
    source = source.resolve()
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": [str(source)],
            "source_fingerprint": tools._source_fingerprint([source]),
            "object_keys": ["assets/source", "assets/untrusted-extra"],
        },
    )
    called = False

    def fake_upload(*_args, **_kwargs):
        nonlocal called
        called = True
        return {"data": {"uploads": [{"object_key": "unexpected"}]}}

    monkeypatch.setattr(client, "upload", fake_upload)

    result = json.loads(
        tools.handle_upload_assets(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "workflow_unavailable"
    assert result["retryable"] is False
    assert called is False


def test_wait_preserves_terminal_status_for_corrupt_project_checkpoint(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_corrupt-terminal-project"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
            "project_id": "project-corrupt-terminal",
            "project": "corrupt",
            "status": "failed",
        },
    )
    monkeypatch.setattr(
        client,
        "poll_project",
        lambda *_args, **_kwargs: pytest.fail("terminal project must not be polled"),
    )

    result = json.loads(
        tools.handle_wait_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "project_terminal"
    assert result["status"] == "failed"


def test_wait_polls_active_corrupt_project_checkpoint(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_corrupt-active-project"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
            "project_id": "project-corrupt-active",
            "project": "corrupt",
            "status": "processing",
        },
    )
    polled = []

    def fake_poll(project_id, **_kwargs):
        polled.append(project_id)
        return {
            "project_id": project_id,
            "status": "completed",
            "result_url": "https://cdn.example.test/result.mp4",
        }

    monkeypatch.setattr(client, "poll_project", fake_poll)

    result = json.loads(
        tools.handle_wait_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["ok"] is True
    assert result["status"] == "completed"
    assert polled == ["project-corrupt-active"]


def test_wait_declares_completed_project_without_result_url_as_incomplete(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew_completed-without-result-url"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
            "project_id": "project-without-result-url",
            "project": {"status": "processing"},
            "status": "processing",
        },
    )
    monkeypatch.setattr(
        client,
        "poll_project",
        lambda project_id, **_kwargs: {
            "project_id": project_id,
            "status": "completed",
        },
    )

    result = json.loads(
        tools.handle_wait_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "project_not_completed"
    assert result["retryable"] is True
    assert result["next"] == "video_edit_wait_project"


def test_service_admission_failures_have_one_neutral_contract(
    isolated_video_home,
    monkeypatch,
):
    results = []
    for status in (401, 403):
        workflow = f"vew_service-admission-{status}"
        tools.state.update(
            workflow,
            "agent-a",
            {
                "source_paths": ["/volume1/subvol/data/one.mov"],
                "object_keys": ["assets/one"],
            },
        )

        def reject(*_args, _status=status, **_kwargs):
            raise client.VideoClientError(
                f"provider status {_status}",
                status=_status,
                body={"token": "secret-provider-body"},
            )

        monkeypatch.setattr(client, "create_project", reject)
        results.append(
            json.loads(
                tools.handle_create_project(
                    {"workflow_id": workflow},
                    agent_id="agent-a",
                )
            )
        )

    expected = {
        "error": "video service admission failed",
        "code": "create_project_failed",
        "reason_code": "service_admission_failed",
        "retryable": False,
        "next": None,
        "recovery": (
            "Stop. Surface the video service admission failure without retrying "
            "or bypassing the plugin."
        ),
    }
    assert results == [expected, expected]
    rendered = json.dumps(results, sort_keys=True).lower()
    assert not any(
        forbidden in rendered
        for forbidden in (
            "401",
            "403",
            "authorization",
            "credential",
            "permission",
            "profile",
            "provider status",
            "scope",
            "secret-provider-body",
            "token",
        )
    )


@pytest.mark.parametrize("status", [400, 404, 409, 413, 422])
def test_non_retryable_service_rejections_stop_without_replay(
    isolated_video_home,
    monkeypatch,
    status,
):
    workflow = f"vew-service-rejected-{status}"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
        },
    )
    monkeypatch.setattr(
        client,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            client.VideoClientError(
                "provider rejected request",
                status=status,
                body={"private": "must-not-leak"},
            )
        ),
    )

    result = json.loads(
        tools.handle_create_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "service_request_rejected"
    assert result["retryable"] is False
    assert result["next"] is None
    assert result["recovery"] == schemas.error_contract(
        "video_edit_create_project", "service_request_rejected"
    )["recovery"]
    rendered = json.dumps(result, sort_keys=True).lower()
    assert str(status) not in rendered
    assert "must-not-leak" not in rendered


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_retryable_service_statuses_keep_same_workflow_recovery(
    isolated_video_home,
    monkeypatch,
    status,
):
    workflow = f"vew-service-transient-{status}"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
        },
    )
    monkeypatch.setattr(
        client,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            client.VideoClientError("temporary service failure", status=status)
        ),
    )

    result = json.loads(
        tools.handle_create_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "transient_failure"
    assert result["retryable"] is True
    assert result["next"] == "video_edit_create_project"


def test_explicit_network_failure_remains_retryable(
    isolated_video_home,
    monkeypatch,
):
    workflow = "vew-explicit-network-failure"
    tools.state.update(
        workflow,
        "agent-a",
        {
            "source_paths": ["/volume1/subvol/data/one.mov"],
            "object_keys": ["assets/one"],
        },
    )
    monkeypatch.setattr(
        client,
        "create_project",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            client.VideoClientError("network unavailable", transient=True)
        ),
    )

    result = json.loads(
        tools.handle_create_project(
            {"workflow_id": workflow},
            agent_id="agent-a",
        )
    )

    assert result["reason_code"] == "transient_failure"
    assert result["retryable"] is True


def test_upload_stream_declares_the_admitted_video_mime(monkeypatch, tmp_path):
    source = tmp_path / "source.mkv"
    _write_video(source, b"payload")
    sent = []

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
            return None

        def send(self, chunk):
            sent.append(chunk)

        def getresponse(self):
            return Response()

        def close(self):
            return None

    monkeypatch.setattr(client.http.client, "HTTPConnection", Connection)

    client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    request_body = b"".join(sent)
    assert b"Content-Type: video/x-matroska" in request_body
    assert b"application/octet-stream" not in request_body


def test_upload_rechecks_the_opened_descriptor_before_connect(monkeypatch, tmp_path):
    source = tmp_path / "renamed.mp4"
    source.write_bytes(b"%PDF-1.7\nnot video")
    connections = []
    monkeypatch.setattr(
        client.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connections.append(True),
    )

    with pytest.raises(paths.VideoPathError, match="supported video"):
        client.upload([source], agent_id="agent-a", replay_scope="source-v1")

    assert connections == []


def test_upload_idempotency_tracks_source_scope_not_regenerated_temp_file(
    monkeypatch, tmp_path
):
    source = tmp_path / "vewm_0.mp4"
    _write_video(source, b"first-normalized-version")
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
    _write_video(source, b"safe")
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
    _write_video(source, b"safe")
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
        "_inspect_opened_sources",
        lambda opened, _replay_scope: opened,
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
    _write_video(source, b"safe")
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
