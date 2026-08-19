"""Process-local metadata for tool results produced by trusted runtimes."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping


class TrustedToolResult(str):
    """A model-visible string carrying immutable process-local control metadata."""

    def __new__(
        cls,
        value: str,
        *,
        terminal_failure_reason: str = "",
        metadata: Mapping[str, str] | None = None,
    ) -> "TrustedToolResult":
        obj = super().__new__(cls, value)
        trusted_metadata = dict(metadata or {})
        if terminal_failure_reason:
            trusted_metadata["terminal_failure_reason"] = terminal_failure_reason
        object.__setattr__(obj, "_trusted_metadata", MappingProxyType(trusted_metadata))
        return obj

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("TrustedToolResult metadata is immutable")

    @property
    def trusted_metadata(self) -> Mapping[str, str]:
        return self._trusted_metadata

    @property
    def terminal_failure_reason(self) -> str:
        return self._trusted_metadata.get("terminal_failure_reason", "")

    def with_text(self, value: str) -> "TrustedToolResult":
        return type(self)(value, metadata=self._trusted_metadata)


def preserve_trusted_tool_result(original: object, projected: object) -> object:
    """Reattach trusted metadata after middleware changes displayed text."""
    if isinstance(original, TrustedToolResult) and isinstance(projected, str):
        return original.with_text(projected)
    return projected
