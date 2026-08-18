"""Resolve platform-owned semantic working-directory aliases."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any, Optional


AGENT_OUTPUT_WORKDIR = "agent_output"
AGENT_OUTPUT_ENV = "ZET_AGENT_OUTPUT_DIR"
AGENT_OUTPUT_ARG = "_zettlab_agent_output_workdir"

# origin.chat_id is untrusted persisted data; suffixes failing this never join.
_CRON_SESSION_SUFFIX_RE = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


class RuntimeWorkdirError(ValueError):
    """A semantic workdir alias cannot be resolved safely."""


def agent_output_dir(
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Return the validated platform output directory, or ``None``.

    Non-raising companion to :func:`resolve_runtime_workdir`. The snapshot guard
    and the local terminal backend both anchor an out-of-scope fallback cwd here,
    and they MUST agree on the value: protecting one directory while the command
    runs in another is exactly the split the guard exists to prevent.
    """

    try:
        return resolve_runtime_workdir(AGENT_OUTPUT_WORKDIR, environ=environ)
    except RuntimeWorkdirError:
        return None


def cron_session_suffix(origin_chat_id: Any) -> Optional[str]:
    """Session suffix (text after the LAST colon), or ``None``.

    Must match local-server sessionscope.ShortID so both sides bucket the
    same run under the same directory.
    """
    if not isinstance(origin_chat_id, str):
        return None
    _, sep, suffix = origin_chat_id.rpartition(":")
    if not sep:
        return None
    return suffix if _CRON_SESSION_SUFFIX_RE.match(suffix) else None


def prepare_cron_session_output_dir(origin_chat_id: Any) -> Optional[str]:
    """Resolve and create ``<ZET_AGENT_OUTPUT_DIR>/<session>`` for one cron run.

    Falls back to the agent output root on underivable suffix or mkdir failure;
    ``None`` when the platform exposes no output dir (upstream deployments).
    """
    base = agent_output_dir()
    if not base:
        return None
    suffix = cron_session_suffix(origin_chat_id)
    if not suffix:
        return base
    session_dir = os.path.join(base, suffix)
    try:
        os.makedirs(session_dir, exist_ok=True)
    except OSError:
        return base
    # agent 对 output 可写，预埋同名 symlink 能把整个 run 的锚点引出沙箱；
    # 桶必须是 base 下的真实目录，否则回落 base。
    try:
        if os.path.islink(session_dir):
            return base
        real_base = os.path.realpath(base)
        if not os.path.realpath(session_dir).startswith(real_base + os.sep):
            return base
    except OSError:
        return base
    return session_dir


def push_cron_output_scope(session_dir: str):
    """Overlay ``ZET_AGENT_OUTPUT_DIR`` for the current cron run (contextvar
    copy, never ``os.environ``). Returns a token for
    :func:`pop_cron_output_scope`; ``None`` when multiplex is on with no scope
    active — installing one there would hide other profiles' secrets.
    """
    from agent.secret_scope import (
        current_secret_scope,
        is_multiplex_active,
        set_secret_scope,
    )

    prev = current_secret_scope()
    if prev is not None:
        overlay = dict(prev)
    elif not is_multiplex_active():
        # Scope miss falls through to os.environ when multiplexing is off,
        # so a single-key scope is a pure overlay here.
        overlay = {}
    else:
        return None
    overlay[AGENT_OUTPUT_ENV] = session_dir
    return set_secret_scope(overlay)


def pop_cron_output_scope(token) -> None:
    """Restore the secret scope replaced by :func:`push_cron_output_scope`."""
    if token is None:
        return
    from agent.secret_scope import reset_secret_scope

    reset_secret_scope(token)


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
