"""Cron-only bridge for profile-local Skill-declared App operations.

Domain Skills may map a small logical operation catalog to owner-scoped
Generated App operations in ``runtime/app_operations.json``.  The model never
controls the target app, target operation, transport URL, or credentials.  The
same Local Server capability CAS, payload validation, response projection, and
application-level idempotency contract used by ``app_data`` remain authoritative.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import stat
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

from hermes_constants import get_skills_dir
from tools import app_data_tool as _app_data
from tools.registry import registry


logger = logging.getLogger(__name__)

_MANIFEST_RELATIVE_PATH = Path("runtime") / "app_operations.json"
_SCHEMA_VERSION = "zettlab.agent_app_operations.v1"
_SUPPORTED_SCHEMA_VERSIONS = frozenset(
    {
        _SCHEMA_VERSION,
        # Existing marketplace packages remain valid during the runtime-
        # neutral contract migration.
        "hermes.skill_app_operations.v1",
    }
)
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_SKILL_FRONTMATTER_BYTES = 64 * 1024
_MAX_SKILL_SCAN_ENTRIES = 1024
_MAX_ATTACHED_SKILLS = 16
_MAX_OPERATIONS = 32
_SKILL_RE = re.compile(r"^[a-z][a-z0-9-]{2,63}$")
_MANIFEST_KEYS = frozenset({"schema_version", "operations"})
_OPERATION_KEYS = frozenset(
    {"name", "mode", "app_slug", "app_operation"}
)
_REQUEST_KEYS = frozenset(
    {"action", "operation", "query", "payload", "idempotency_key"}
)


class _ManifestError(ValueError):
    """A fail-closed profile-local runtime manifest error."""


@dataclass(frozen=True)
class _DeclaredOperation:
    name: str
    mode: str
    app_slug: str
    app_operation: str


@dataclass(frozen=True)
class _ManifestSnapshot:
    authority: str
    operations: tuple[_DeclaredOperation, ...]
    error: str = ""


_SNAPSHOT_UNSET = object()
_CRON_MANIFEST_SNAPSHOT: ContextVar[object] = ContextVar(
    "hermes_cron_skill_operation_manifest_snapshot",
    default=_SNAPSHOT_UNSET,
)


def _is_regular_without_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode)


def _is_directory_without_symlink(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode)


def _read_regular_limited(path: Path, limit: int) -> bytes:
    if not _is_regular_without_symlink(path):
        raise _ManifestError(f"{path.name} is not a regular file")
    try:
        before = path.lstat()
        if before.st_size > limit:
            raise _ManifestError(f"{path.name} exceeds the size limit")
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
                raise _ManifestError(f"{path.name} changed while opening")
            raw = handle.read(limit + 1)
        after = path.lstat()
    except _ManifestError:
        raise
    except OSError as exc:
        raise _ManifestError(f"unable to read {path.name}") from exc
    if len(raw) > limit:
        raise _ManifestError(f"{path.name} exceeds the size limit")
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise _ManifestError(f"{path.name} changed while reading")
    return raw


def _skill_frontmatter(skill_root: Path) -> dict[str, object]:
    from agent.skill_utils import parse_frontmatter

    raw = _read_regular_limited(
        skill_root / "SKILL.md", _MAX_SKILL_FRONTMATTER_BYTES
    )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _ManifestError("SKILL.md is not UTF-8") from exc
    try:
        frontmatter, _body = parse_frontmatter(text)
    except Exception as exc:
        raise _ManifestError("SKILL.md frontmatter is invalid") from exc
    if not isinstance(frontmatter, dict):
        raise _ManifestError("SKILL.md frontmatter must be an object")
    return frontmatter


def _is_profile_local_skill_path(skills_root: Path, skill_md: Path) -> bool:
    try:
        relative = skill_md.relative_to(skills_root)
    except ValueError:
        return False
    # Hermes supports root-level and one-category-deep profile Skills. Runtime
    # manifests under support packages never become independent authorities.
    if len(relative.parts) not in {2, 3} or relative.name != "SKILL.md":
        return False
    current = skills_root
    if not _is_directory_without_symlink(current):
        return False
    for part in relative.parts:
        current = current / part
        if current == skill_md:
            return _is_regular_without_symlink(current)
        if not _is_directory_without_symlink(current):
            return False
    return False


def _profile_local_skill_files(skills_root: Path) -> list[Path]:
    """Return bounded root/category Skill indexes without following links."""
    scanned = 0
    result: list[Path] = []

    def _directories(path: Path) -> list[Path]:
        nonlocal scanned
        directories: list[Path] = []
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    scanned += 1
                    if scanned > _MAX_SKILL_SCAN_ENTRIES:
                        raise _ManifestError(
                            "profile-local Skill tree exceeds the scan limit"
                        )
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            directories.append(Path(entry.path))
                    except OSError:
                        continue
        except _ManifestError:
            raise
        except OSError as exc:
            raise _ManifestError("unable to scan profile-local Skills") from exc
        return directories

    for first_level in _directories(skills_root):
        direct_index = first_level / "SKILL.md"
        if _is_regular_without_symlink(direct_index):
            result.append(direct_index)
        for second_level in _directories(first_level):
            categorized_index = second_level / "SKILL.md"
            if _is_regular_without_symlink(categorized_index):
                result.append(categorized_index)
    return result


def _skill_roots(skill: str, *, require_manifest: bool = True) -> list[Path]:
    try:
        from agent.skill_utils import (
            is_excluded_skill_path,
            skill_matches_platform,
        )

        skills_root = get_skills_dir()
        if not _is_directory_without_symlink(skills_root):
            return []
        roots: list[Path] = []
        for skill_md in _profile_local_skill_files(skills_root):
            if is_excluded_skill_path(skill_md, root=skills_root):
                continue
            if not _is_profile_local_skill_path(skills_root, skill_md):
                continue
            candidate = skill_md.parent
            try:
                frontmatter = _skill_frontmatter(candidate)
            except _ManifestError:
                continue
            name = frontmatter.get("name")
            if not isinstance(name, str) or name.strip() != skill:
                continue
            if not skill_matches_platform(frontmatter):
                continue
            if require_manifest:
                runtime_dir = candidate / _MANIFEST_RELATIVE_PATH.parent
                manifest = candidate / _MANIFEST_RELATIVE_PATH
                if not _is_directory_without_symlink(runtime_dir):
                    continue
                if not _is_regular_without_symlink(manifest):
                    continue
            roots.append(candidate)
    except Exception:
        logger.warning(
            "Unable to resolve profile-local Skill runtime manifest; denying Skill operation"
        )
        return []
    return roots


def _resolve_bound_manifest() -> tuple[str, tuple[_DeclaredOperation, ...]]:
    try:
        from gateway.session_context import cron_attached_skills

        attached = cron_attached_skills()
    except Exception as exc:
        raise _ManifestError("scheduled-job Skill binding is unavailable") from exc
    if not attached or len(attached) > _MAX_ATTACHED_SKILLS:
        raise _ManifestError("scheduled job has no valid Skill binding")

    unique = tuple(dict.fromkeys(attached))
    authorities: list[str] = []
    for raw_skill in unique:
        if not isinstance(raw_skill, str) or not _SKILL_RE.fullmatch(raw_skill):
            raise _ManifestError("scheduled job has an invalid Skill binding")
        if _skill_disabled(raw_skill):
            raise _ManifestError(
                f"scheduled-job Skill '{raw_skill}' is disabled for cron"
            )
        roots = _skill_roots(raw_skill, require_manifest=False)
        if not roots:
            raise _ManifestError(
                f"scheduled-job Skill '{raw_skill}' is not an enabled profile-local Skill"
            )
        if len(roots) != 1:
            raise _ManifestError(
                f"scheduled-job Skill '{raw_skill}' is ambiguous in this profile"
            )
        root = roots[0]
        runtime_dir = root / _MANIFEST_RELATIVE_PATH.parent
        manifest = root / _MANIFEST_RELATIVE_PATH
        if _is_directory_without_symlink(runtime_dir) and _is_regular_without_symlink(
            manifest
        ):
            authorities.append(raw_skill)
            continue
        if os.path.lexists(manifest) or (
            os.path.lexists(runtime_dir)
            and not _is_directory_without_symlink(runtime_dir)
        ):
            raise _ManifestError(
                f"scheduled-job Skill '{raw_skill}' has an invalid runtime manifest"
            )
    if len(authorities) != 1:
        raise _ManifestError(
            "scheduled job must bind exactly one runtime operation manifest"
        )
    authority = authorities[0]
    return authority, _load_manifest(authority)


def push_cron_manifest_snapshot() -> object:
    """Freeze this Cron run's profile-local operation authority."""
    try:
        authority, operations = _resolve_bound_manifest()
        snapshot = _ManifestSnapshot(authority, operations)
    except _ManifestError as exc:
        # An unrelated Cron job must still run when it has no operation
        # authority. Preserve the fail-closed reason for this run only.
        snapshot = _ManifestSnapshot("", (), str(exc))
    return _CRON_MANIFEST_SNAPSHOT.set(snapshot)


def push_unavailable_cron_manifest_snapshot(reason: str) -> object:
    """Bind an explicitly unavailable authority for optional-bridge fallback."""
    message = str(reason or "scheduled-job operation snapshot is unavailable")
    return _CRON_MANIFEST_SNAPSHOT.set(_ManifestSnapshot("", (), message[:256]))


def clear_cron_manifest_snapshot() -> None:
    """Fail-close the current task after snapshot cleanup cannot restore it."""
    _CRON_MANIFEST_SNAPSHOT.set(_SNAPSHOT_UNSET)


def pop_cron_manifest_snapshot(token: object) -> None:
    """Restore the operation authority that preceded this Cron run."""
    _CRON_MANIFEST_SNAPSHOT.reset(token)


class CronSkillOperationScope:
    """Optional Cron bridge scope whose faults never escape into scheduling."""

    def __init__(self, job_id: str):
        self._job_id = job_id
        self._attached_token = None
        self._snapshot_token = None

    def bind(self, skills: object) -> None:
        binding_ready = False
        try:
            from gateway.session_context import push_cron_attached_skills

            self._attached_token = push_cron_attached_skills(skills)
            binding_ready = True
        except Exception:
            logger.warning(
                "Job '%s': unable to bind attached Skills; disabling Skill operations",
                self._job_id,
                exc_info=True,
            )
            try:
                from gateway.session_context import push_cron_attached_skills

                self._attached_token = push_cron_attached_skills([])
            except Exception:
                logger.warning(
                    "Job '%s': unable to install empty Skill fallback scope",
                    self._job_id,
                    exc_info=True,
                )

        try:
            if binding_ready:
                self._snapshot_token = push_cron_manifest_snapshot()
            else:
                self._snapshot_token = push_unavailable_cron_manifest_snapshot(
                    "scheduled-job Skill binding is unavailable"
                )
        except Exception:
            logger.warning(
                "Job '%s': unable to bind Skill operation snapshot",
                self._job_id,
                exc_info=True,
            )
            try:
                self._snapshot_token = push_unavailable_cron_manifest_snapshot(
                    "scheduled-job operation snapshot binding failed"
                )
            except Exception:
                logger.warning(
                    "Job '%s': unable to install unavailable operation snapshot",
                    self._job_id,
                    exc_info=True,
                )

    def close(self) -> None:
        if self._snapshot_token is not None:
            try:
                pop_cron_manifest_snapshot(self._snapshot_token)
            except Exception:
                logger.warning(
                    "Job '%s': unable to restore Skill operation snapshot",
                    self._job_id,
                    exc_info=True,
                )
                try:
                    clear_cron_manifest_snapshot()
                except Exception:
                    logger.warning(
                        "Job '%s': unable to fail-close operation snapshot cleanup",
                        self._job_id,
                        exc_info=True,
                    )

        if self._attached_token is not None:
            try:
                from gateway.session_context import pop_cron_attached_skills

                pop_cron_attached_skills(self._attached_token)
            except Exception:
                logger.warning(
                    "Job '%s': unable to restore attached Skill scope",
                    self._job_id,
                    exc_info=True,
                )
                try:
                    from gateway.session_context import push_cron_attached_skills

                    push_cron_attached_skills([])
                except Exception:
                    logger.warning(
                        "Job '%s': unable to fail-close attached Skill cleanup",
                        self._job_id,
                        exc_info=True,
                    )


def bind_cron_skill_operation_scope(
    skills: object, *, job_id: str
) -> CronSkillOperationScope:
    """Create a fail-closed optional bridge for one scheduled job."""
    scope = CronSkillOperationScope(job_id)
    scope.bind(skills)
    return scope


def _bound_manifest() -> tuple[str, tuple[_DeclaredOperation, ...]]:
    """Return the run-start authority only while live state still matches it."""
    snapshot = _CRON_MANIFEST_SNAPSHOT.get()
    if not isinstance(snapshot, _ManifestSnapshot):
        raise _ManifestError("scheduled-job operation snapshot is unavailable")
    if snapshot.error:
        raise _ManifestError(snapshot.error)
    try:
        authority, operations = _resolve_bound_manifest()
    except _ManifestError as exc:
        raise _ManifestError(
            "scheduled-job operation manifest changed or became unavailable"
        ) from exc
    if authority != snapshot.authority or operations != snapshot.operations:
        raise _ManifestError(
            "scheduled-job operation manifest changed after the run started"
        )
    return snapshot.authority, snapshot.operations


def _skill_disabled(skill: str) -> bool:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.skills_config import get_disabled_skills

        config = load_config()
        if not isinstance(config, dict):
            raise _ManifestError("skills config root is invalid")
        skills_config = config.get("skills")
        if skills_config is None:
            skills_config = {}
        if not isinstance(skills_config, dict):
            raise _ManifestError("skills config is invalid")

        def _valid_disabled_list(value: object) -> bool:
            # A single string is an established legacy spelling. Otherwise the
            # security boundary accepts only a YAML sequence of Skill names.
            if value is None or isinstance(value, str):
                return True
            return isinstance(value, list) and all(
                isinstance(item, str) for item in value
            )

        if not _valid_disabled_list(skills_config.get("disabled")):
            raise _ManifestError("skills.disabled config is invalid")
        platform_disabled = skills_config.get("platform_disabled")
        if platform_disabled is None:
            platform_disabled = {}
        if not isinstance(platform_disabled, dict):
            raise _ManifestError("skills.platform_disabled config is invalid")
        if not _valid_disabled_list(platform_disabled.get("cron")):
            raise _ManifestError("skills.platform_disabled.cron config is invalid")

        return skill in get_disabled_skills(config, "cron")
    except Exception:
        logger.warning(
            "Unable to resolve profile-local Skill disabled state; denying Skill operation"
        )
        return True


def skill_operation_manifest_cache_scope() -> str:
    """Fingerprint the exact Cron Skill binding and its parsed manifest.

    The model tool-schema cache is process-wide in a single-profile gateway.
    This semantic digest prevents one scheduled job's availability verdict from
    being reused by another job, and invalidates a cached verdict immediately
    after a manifest is added, removed, disabled, or changed.
    """
    from gateway.session_context import cron_attached_skills

    attached = cron_attached_skills()
    scope: dict[str, object] = {"attached_skills": list(attached)}
    try:
        bound_skill, operations = _bound_manifest()
    except _ManifestError:
        scope["manifest"] = None
    else:
        scope["manifest"] = {
            "skill": bound_skill,
            "operations": [
                {
                    "name": operation.name,
                    "mode": operation.mode,
                    "app_slug": operation.app_slug,
                    "app_operation": operation.app_operation,
                }
                for operation in operations
            ],
        }
    encoded = json.dumps(
        scope, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _ManifestError(f"duplicate manifest field {key!r}")
        result[key] = value
    return result


def _load_manifest(skill: str) -> tuple[_DeclaredOperation, ...]:
    if not _SKILL_RE.fullmatch(skill):
        raise _ManifestError("skill name is invalid")
    if _skill_disabled(skill):
        raise _ManifestError("skill is disabled for cron")
    roots = _skill_roots(skill)
    if len(roots) != 1:
        if not roots:
            raise _ManifestError("profile-local skill runtime manifest is missing")
        raise _ManifestError("profile-local skill runtime manifest is ambiguous")

    raw = _read_regular_limited(
        roots[0] / _MANIFEST_RELATIVE_PATH, _MAX_MANIFEST_BYTES
    )
    try:
        document = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except _ManifestError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _ManifestError("runtime manifest is invalid JSON") from exc
    if not isinstance(document, dict) or set(document) != _MANIFEST_KEYS:
        raise _ManifestError("runtime manifest fields are invalid")
    if document.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        raise _ManifestError("runtime manifest schema is unsupported")
    raw_operations = document.get("operations")
    if (
        not isinstance(raw_operations, list)
        or not raw_operations
        or len(raw_operations) > _MAX_OPERATIONS
    ):
        raise _ManifestError("runtime manifest operations are invalid")

    operations: list[_DeclaredOperation] = []
    seen: set[str] = set()
    for raw_operation in raw_operations:
        if not isinstance(raw_operation, dict) or set(raw_operation) != _OPERATION_KEYS:
            raise _ManifestError("runtime operation fields are invalid")
        name = raw_operation.get("name")
        mode = raw_operation.get("mode")
        app_slug = raw_operation.get("app_slug")
        app_operation = raw_operation.get("app_operation")
        if (
            not isinstance(name, str)
            or not _app_data._OPERATION_RE.fullmatch(name)
            or name in seen
        ):
            raise _ManifestError("runtime operation name is invalid")
        if mode not in {"read", "mutation"}:
            raise _ManifestError("runtime operation mode is invalid")
        try:
            target_slug = _app_data._validate_slug(app_slug)
            target_operation = _app_data._validate_operation(app_operation)
        except _app_data._BadRequest as exc:
            raise _ManifestError("runtime operation target is invalid") from exc
        seen.add(name)
        operations.append(
            _DeclaredOperation(
                name=name,
                mode=mode,
                app_slug=target_slug,
                app_operation=target_operation,
            )
        )
    return tuple(operations)


def _is_cron_session() -> bool:
    return _app_data._is_cron_session()


def _is_delegated_child_context() -> bool:
    return _app_data._is_delegated_child_context()


def _has_runtime_manifest() -> bool:
    try:
        _bound_manifest()
        return True
    except _ManifestError:
        return False


def _check_skill_operation() -> bool:
    if not _is_cron_session() or _is_delegated_child_context():
        return False
    return bool(
        _app_data._base_url()
        and _app_data._secret(_app_data._ACTION_TOKEN_SECRET)
        and _app_data._secret(_app_data._AGENT_ID_SECRET)
        and _has_runtime_manifest()
    )


_check_skill_operation._profile_scope_sensitive = True  # type: ignore[attr-defined]
_check_skill_operation._session_scope_sensitive = True  # type: ignore[attr-defined]


def _manifest_failure(message: str) -> str:
    return _app_data._failure(
        "skill_operation_unavailable", message[:_app_data._MAX_ERROR_MESSAGE_CHARS], status=0
    )


def _find_operation(
    operations: tuple[_DeclaredOperation, ...], name: str
) -> _DeclaredOperation | None:
    for operation in operations:
        if operation.name == name:
            return operation
    return None


def _run_skill_operation(args: object) -> str:
    if not _is_cron_session():
        return _app_data._failure(
            "cron_scope_required",
            "Skill operation 仅允许在 Cron 会话中调用",
            status=0,
        )
    if _is_delegated_child_context():
        return _app_data._failure(
            "delegated_child_scope_denied",
            "delegate_task 子 Agent 无权调用 Skill operation",
            status=0,
        )
    values = args if isinstance(args, dict) else {}
    try:
        if any(not isinstance(key, str) for key in values) or not set(
            values
        ).issubset(_REQUEST_KEYS):
            raise _app_data._BadRequest("请求包含未声明字段")
        action = _app_data._validate_action(values.get("action"))
        operation_raw = values.get("operation")
        payload_raw = values.get("payload")
        query_raw = values.get("query")
        key_raw = values.get("idempotency_key")
        if action == "capabilities":
            if any(
                value not in (None, "")
                for value in (operation_raw, payload_raw, query_raw, key_raw)
            ):
                raise _app_data._BadRequest(
                    "capabilities 不接受 operation、payload、query 或 idempotency_key"
                )
            operation_name = ""
            payload = None
            query = None
            key = ""
        else:
            operation_name = _app_data._validate_operation(operation_raw)
            payload = _app_data._validate_document("payload", payload_raw)
            query = _app_data._validate_query(query_raw)
            key = _app_data._validate_key(key_raw)
    except _app_data._BadRequest as exc:
        return _app_data._failure("invalid_request", str(exc), status=0)

    try:
        _bound_skill, operations = _bound_manifest()
    except _ManifestError as exc:
        return _manifest_failure(str(exc))
    if action == "capabilities":
        return _app_data._success(
            {
                "version": 1,
                "operations": [
                    {"name": operation.name, "mode": operation.mode}
                    for operation in operations
                ],
            }
        )

    declared = _find_operation(operations, operation_name)
    if declared is None:
        return _app_data._failure(
            "operation_not_declared",
            "此 profile-local Skill 未声明该 Cron operation",
            status=0,
        )
    if declared.mode == "mutation" and not key:
        return _app_data._failure(
            "invalid_request", "mutation 必须提供 idempotency_key", status=0
        )

    base = _app_data._base_url()
    token = _app_data._secret(_app_data._ACTION_TOKEN_SECRET)
    agent_id = _app_data._secret(_app_data._AGENT_ID_SECRET)
    if not base or not token or not agent_id:
        return _app_data._failure(
            "unsupported", "当前 Agent 未配置本地 App operation 数据桥", status=0
        )
    try:
        capabilities = _app_data._load_capabilities(
            base, token, declared.app_slug
        )
    except _app_data._BridgeError as exc:
        return _app_data._bridge_error(exc)
    target_mode = _app_data._declared_mode(
        capabilities, declared.app_operation
    )
    if target_mode is None:
        return _app_data._failure(
            "operation_not_declared",
            "目标应用未声明此 operation",
            status=0,
        )
    if target_mode != declared.mode:
        return _app_data._failure(
            "operation_contract_mismatch",
            "Skill operation 与目标应用 mode 不一致",
            status=0,
        )

    envelope: dict[str, object] = {
        "capability_digest": capabilities["capability_digest"]
    }
    if payload is not None:
        envelope["payload"] = payload
    if query is not None:
        envelope["query"] = query
    if key:
        envelope["idempotency_key"] = key
    path = (
        f"/{quote(declared.app_slug, safe='')}/operations/"
        f"{quote(declared.app_operation, safe='')}"
    )
    try:
        result = _app_data._request_json(
            base=base,
            token=token,
            method="POST",
            path=path,
            body=envelope,
            retry_read=False,
            timeout=(
                _app_data._READ_TIMEOUT
                if declared.mode == "read"
                else _app_data._MUTATION_TIMEOUT
            ),
        )
    except _app_data._BridgeError as exc:
        return _app_data._bridge_error(exc)
    return _app_data._success(result)


def skill_operation_tool(args, **_kw) -> str:
    """Invoke a Cron-safe operation declared by a profile-local Skill."""
    return _run_skill_operation(args)


SKILL_OPERATION_SCHEMA = {
    "name": "skill_operation",
    "description": (
        "Discover or invoke Cron-safe App operations declared by a profile-local "
        "Skill. Target apps and transport details are fixed by the Skill manifest."
    ),
    "parameters": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "action": {
                "type": "string",
                "enum": ["capabilities", "invoke"],
            },
            "operation": {
                "type": "string",
                "pattern": "^[a-z][a-z0-9_]*(?:\\.[a-z][a-z0-9_]*){1,7}$",
                "description": "Logical operation declared by the Skill runtime manifest.",
            },
            "query": {
                "type": "object",
                "maxProperties": 16,
                "propertyNames": {"pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "additionalProperties": {"type": "string", "maxLength": 1024},
            },
            "payload": {"type": "object"},
            "idempotency_key": {
                "type": "string",
                "pattern": "^[A-Za-z0-9._:-]{1,128}$",
                "description": "Required for mutations; reuse after an unknown result.",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="skill_operation",
    toolset="zettlab_skill_runtime",
    schema=SKILL_OPERATION_SCHEMA,
    handler=skill_operation_tool,
    check_fn=_check_skill_operation,
    emoji="data",
    max_result_size_chars=_app_data._MAX_RESPONSE_BYTES,
    defer_to_tool_search=False,
)
