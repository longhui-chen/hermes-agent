"""Resolve gateway ``terminal.cwd`` placeholder values to ``TERMINAL_CWD``.

When ``terminal.cwd`` is unset or a placeholder (``.``, ``auto``, ``cwd``),
the gateway must not blindly map host ``Path.home()`` into container backends.
Docker with workspace mounting still needs an explicit host path signal
(``MESSAGING_CWD`` or an absolute config path) for ``terminal_tool`` to map
``/host/project`` → ``/workspace``.
"""

from __future__ import annotations

CWD_PLACEHOLDERS = frozenset({".", "auto", "cwd"})


def _truthy_env(value: str | None) -> bool:
    return (value or "").strip().lower() in {"true", "1", "yes"}


def resolve_placeholder_terminal_cwd(
    *,
    configured_cwd: str,
    terminal_backend: str,
    messaging_cwd: str | None,
    docker_mount_cwd_to_workspace: bool,
    home_fallback: str,
    managed_gateway: bool = False,
) -> str | None:
    """Return the ``TERMINAL_CWD`` value to set, or ``None`` to leave it unset.

    Cases:
      - **local** + placeholder + managed gateway → ``MESSAGING_CWD`` or ``None``
      - **local** + placeholder → ``MESSAGING_CWD`` or ``home_fallback``
      - **docker** + placeholder + mount on + host ``MESSAGING_CWD`` → host path
        (for ``terminal_tool`` ``/workspace`` mapping)
      - **docker** + placeholder + mount off → ``None`` (sandbox default)
      - other non-local backends + placeholder → ``None``

    The managed-gateway carve-out exists because ``home_fallback`` there is the
    root daemon's ``/root``: a directory the model shell cannot traverse and
    that sits outside every snapshot target. Planting it in ``TERMINAL_CWD``
    makes it look like a deliberate anchor to everything downstream — the file
    tools stop at it before reaching their platform-output fallback, and every
    relative write becomes an out-of-scope block. Leaving the variable unset is
    the honest answer: the platform, not this process's HOME, decides where a
    managed agent works.
    """
    if configured_cwd and configured_cwd not in CWD_PLACEHOLDERS:
        return configured_cwd

    backend = (terminal_backend or "local").strip().lower()
    if backend == "local":
        messaging = (messaging_cwd or "").strip()
        if messaging:
            return messaging
        return None if managed_gateway else home_fallback

    if backend == "docker" and docker_mount_cwd_to_workspace:
        messaging = (messaging_cwd or "").strip()
        if messaging and messaging not in CWD_PLACEHOLDERS:
            return messaging

    return None
