from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plugins.video_edit as video_plugin
from plugins.video_edit import client


class _Context:
    def __init__(self) -> None:
        self.handlers: dict[str, Any] = {}

    def register_tool(self, *, name, handler, **_kwargs) -> None:
        self.handlers[name] = handler


def _video_sample(marker: bytes) -> bytes:
    def box(kind: bytes, payload: bytes) -> bytes:
        return (len(payload) + 8).to_bytes(4, "big") + kind + payload

    hdlr = box(b"hdlr", b"\x00" * 8 + b"vide" + b"\x00" * 12)
    return (
        box(b"ftyp", b"qt  \x00\x00\x00\x00qt  ")
        + box(b"moov", box(b"trak", box(b"mdia", hdlr)))
        + marker
    )


def test_registered_tools_keep_one_selection_across_internal_upload_batches(
    tmp_path,
    monkeypatch,
):
    hermes_home = tmp_path / "hermes"
    output = tmp_path / "output"
    media_root = output / "agent-a"
    media_root.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("ZET_AGENT_OUTPUT_DIR", str(output))
    monkeypatch.setattr(
        "tools.runtime_workdir.agent_output_dir",
        lambda: str(output),
    )

    sources: list[Path] = []
    for index in range(14):
        source = media_root / f"clip-{index:02d}.mov"
        source.write_bytes(_video_sample(f"clip-{index}".encode()))
        sources.append(source)

    batches: list[list[str]] = []
    replay_scopes: list[str] = []

    def upload(files, **kwargs):
        names = [path.name for path in files]
        batches.append(names)
        replay_scopes.append(kwargs["replay_scope"])
        return {
            "data": {"uploads": [{"object_key": f"asset/{name}"} for name in names]}
        }

    project_keys: list[str] = []

    def create_project(object_keys, _preferences, **_kwargs):
        project_keys.extend(object_keys)
        return {"project_id": "project-batched", "status": "queued"}

    monkeypatch.setattr(client, "upload", upload)
    monkeypatch.setattr(client, "create_project", create_project)

    context = _Context()
    video_plugin.register(context)
    resolve = context.handlers["video_edit_preferences_resolve"]
    upload_assets = context.handlers["video_edit_upload_assets"]
    create = context.handlers["video_edit_create_project"]

    resolved = json.loads(
        resolve(
            {
                "scene": "family_outing",
                "preferences": {
                    "upload_preference": "raw_direct",
                    "user_prompt": "Make one coherent recap from every selected clip.",
                },
            },
            agent_id="agent-a",
            turn_id="turn-batched",
        )
    )
    uploaded = json.loads(
        upload_assets(
            {
                "workflow_id": resolved["workflow_id"],
                "files": [str(source) for source in sources],
            },
            agent_id="agent-a",
        )
    )
    created = json.loads(
        create(
            {"workflow_id": resolved["workflow_id"]},
            agent_id="agent-a",
        )
    )

    assert uploaded["uploaded"] == 14
    assert [len(batch) for batch in batches] == [10, 4]
    assert replay_scopes[0].endswith(":0")
    assert replay_scopes[1].endswith(":10")
    assert replay_scopes[0] != replay_scopes[1]
    assert project_keys == [f"asset/{source.name}" for source in sources]
    assert created["project_id"] == "project-batched"
