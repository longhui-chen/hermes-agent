"""Native Zettlab Memo video-editing tools.

The plugin is deliberately an ordinary Hermes tool provider.  It owns the
provider HTTP calls and profile-scoped workflow/preferences state; the skill
only chooses the next tool and supplies the user's creative intent.  There is
no video-specific token, receipt, check, or skill authorization protocol here.
The existing profile-scoped platform token is used internally
when the loopback local-server requires its normal caller identity.
"""

from __future__ import annotations

from plugins.video_edit.schemas import TOOL_DEFINITIONS
from plugins.video_edit.tools import HANDLERS


def register(ctx) -> None:
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
