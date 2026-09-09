"""Private scheduler identity, not an authorization grant or shell environment.

Only run_one_job installs this after the durable running transition. Consumers
must still validate the active execution and their resource-specific policy at
the service boundary. Copied worker contexts do not extend an execution lease.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class CronExecutionContext:
    job_id: str
    execution_id: str
    profile_home: Path


_execution: ContextVar[CronExecutionContext | None] = ContextVar(
    "_cron_execution_identity", default=None
)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")


def current_execution() -> CronExecutionContext | None:
    """No fallback to job metadata, process environment, or Chat identity."""
    return _execution.get()


@contextmanager
def execution_scope(job_id: str, execution_id: str, profile_home: Path):
    # Legacy jobs with incompatible IDs still run, but cannot use scoped
    # hardware execution. Always shadow an outer context, including on failure.
    identity = None
    if (
        isinstance(job_id, str)
        and isinstance(execution_id, str)
        and _ID.fullmatch(job_id)
        and _ID.fullmatch(execution_id)
    ):
        identity = CronExecutionContext(job_id, execution_id, profile_home)
    token = _execution.set(identity)
    try:
        yield identity
    finally:
        _execution.reset(token)
