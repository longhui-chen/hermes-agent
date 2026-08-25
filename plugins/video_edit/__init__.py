"""Native Zettlab Memo video-editing tools.

The plugin is deliberately an ordinary Hermes tool provider.  It owns the
provider HTTP calls and profile-scoped workflow/preferences state; the skill
only chooses the next tool and supplies the user's creative intent.  There is
no video-specific token, receipt, check, or skill authorization protocol here.
The loopback local-server owns the device-side connection and upstream
credential; this plugin sends only bounded request identity for replay.
"""

from __future__ import annotations

from plugins.video_edit import normalizer
from plugins.video_edit.schemas import TOOL_DEFINITIONS
from plugins.video_edit.tools import HANDLERS


def register(ctx) -> None:
    # Fix either the current release or its absence without disabling Help/tools.
    normalizer.initialize_runtime()
    for definition in TOOL_DEFINITIONS:
        name = definition["name"]
        ctx.register_tool(
            name=name,
            toolset="video_edit",
            schema=definition,
            handler=HANDLERS[name],
            emoji=definition.get("emoji", "🎬"),
            description=definition.get("description", ""),
        )
