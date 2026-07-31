"""Early trusted video-edit runtime preparation for gateway processes."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Sequence

logger = logging.getLogger(__name__)


def _is_gateway_run_request(argv: Sequence[str]) -> bool:
    args = list(argv[1:])
    return any(
        args[index : index + 2] == ["gateway", "run"]
        for index in range(max(len(args) - 1, 0))
    )


def prepare_trusted_video_edit_runtime_before_cli_logging(
    argv: Sequence[str] | None = None,
) -> bool:
    """Prepare the trusted worker before centralized logging starts a thread."""
    effective_argv = sys.argv if argv is None else argv
    if not os.environ.get("ZETTLAB_PRESETS_DIR") or not _is_gateway_run_request(
        effective_argv
    ):
        return False
    try:
        terminal_tool = importlib.import_module("tools.terminal_tool")
        terminal_tool._late_prepare_video_edit_worker_before_terminal()
        return True
    except Exception as exc:
        logger.warning(
            "Trusted video-edit runtime unavailable before CLI logging: %s",
            type(exc).__name__,
        )
        return False
