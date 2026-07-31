"""One-shot process-inspection boundary for the ZetAgent token surface."""

import threading

from tools.process_security import harden_sensitive_process


_UNINITIALIZED = object()
_GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY: object | bool = _UNINITIALIZED
_GATEWAY_SENSITIVE_PROCESS_BOUNDARY_LOCK = threading.Lock()


def initialize_gateway_sensitive_process_boundary() -> bool:
    """Establish the immutable ZetAgent boundary exactly once.

    This intentionally does not set ``no_new_privs``: ordinary gateway tools
    may still execute an approved sudo flow. ``PR_SET_DUMPABLE=0`` plus the
    ptrace/resource capability drop protects the parent from same-UID model
    children without changing the process-wide exec privilege contract.
    """
    global _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY

    if _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY is _UNINITIALIZED:
        with _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_LOCK:
            if _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY is _UNINITIALIZED:
                try:
                    ready = harden_sensitive_process(
                        no_new_privs=False,
                        drop_ptrace=True,
                    )
                except Exception:
                    ready = False
                _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY = bool(ready)
    return _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY is True


def gateway_sensitive_process_boundary_ready() -> bool:
    """Return the startup result without initializing or retrying late."""
    return _GATEWAY_SENSITIVE_PROCESS_BOUNDARY_READY is True
