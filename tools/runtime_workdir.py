"""Resolve platform-owned semantic working-directory aliases."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional


AGENT_OUTPUT_WORKDIR = "agent_output"
AGENT_OUTPUT_ENV = "ZET_AGENT_OUTPUT_DIR"


class RuntimeWorkdirError(ValueError):
    """A semantic workdir alias cannot be resolved safely."""


def resolve_runtime_workdir(
    workdir: Optional[str],
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Resolve a supported workdir alias to a validated platform path.

    Ordinary filesystem paths pass through unchanged. Only the exact
    ``agent_output`` sentinel consults the active profile scope (or the process
    environment outside multiplex mode); arbitrary variables embedded in
    either the argument or platform value are never expanded.
    """
    if workdir != AGENT_OUTPUT_WORKDIR:
        return workdir

    if environ is None:
        try:
            from agent.secret_scope import get_secret

            output_value = get_secret(AGENT_OUTPUT_ENV, "")
        except RuntimeError as exc:
            raise RuntimeWorkdirError(
                f"workdir '{AGENT_OUTPUT_WORKDIR}' is unavailable: "
                "no active platform profile scope"
            ) from exc
    else:
        output_value = environ.get(AGENT_OUTPUT_ENV, "")
    output_dir = str(output_value or "").strip()
    if not output_dir:
        raise RuntimeWorkdirError(
            f"workdir '{AGENT_OUTPUT_WORKDIR}' is unavailable: "
            f"{AGENT_OUTPUT_ENV} is not set by the platform"
        )
    if not os.path.isabs(output_dir):
        raise RuntimeWorkdirError(
            f"workdir '{AGENT_OUTPUT_WORKDIR}' is unavailable: "
            f"{AGENT_OUTPUT_ENV} must be an absolute path"
        )
    if not os.path.isdir(output_dir):
        raise RuntimeWorkdirError(
            f"workdir '{AGENT_OUTPUT_WORKDIR}' is unavailable: "
            f"{AGENT_OUTPUT_ENV} is not an existing directory"
        )
    return os.path.normpath(output_dir)
