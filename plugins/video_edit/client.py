"""Small bounded HTTP client for the local-server video business surface."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from plugins.video_edit import normalizer
from plugins.video_edit.paths import (
    VideoPathError,
    agent_id_from_kwargs,
    safe_id,
    validate_video_descriptor,
    video_media_type,
)

DEFAULT_BASE = "http://127.0.0.1:19090/api/v1/ai-proxy/business"
DEFAULT_INTERNAL_BASE = "http://127.0.0.1:19090/api/v1/internal/proactive-video"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_FILES = 8
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
RESULT_URL_UNAVAILABLE_STATUSES = frozenset(
    {301, 302, 303, 307, 308, 401, 403, 404, 410}
)


class VideoClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int = 0,
        body: Any = None,
        transient: bool = False,
    ):
        super().__init__(message)
        self.status = status
        self.body = body
        self.transient = transient


class ResultURLUnavailable(VideoClientError):
    """The signed result URL must be refreshed from its existing project."""


def _base_url() -> str:
    raw = str(os.environ.get("ZETTLAB_VIDEO_EDIT_BASE_URL", DEFAULT_BASE) or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise VideoClientError("video edit loopback endpoint is invalid")
    if parsed.port not in {19090, None}:
        raise VideoClientError("video edit loopback endpoint is invalid")
    return raw


def _internal_base() -> str:
    raw = str(os.environ.get("ZETTLAB_VIDEO_EDIT_INTERNAL_BASE_URL", DEFAULT_INTERNAL_BASE) or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise VideoClientError("video edit internal endpoint is invalid")
    return raw


def _replay_partition(agent_id: str = "") -> str:
    """Return a non-secret profile namespace for deterministic retries.

    This value is hashed into request IDs only. It is deliberately not sent as
    a credential or used by local-server for admission; it merely prevents two
    profiles with identical payloads from sharing an in-memory replay record.
    """
    return safe_id(agent_id.strip() or agent_id_from_kwargs())


def _request_key(
    operation: str,
    payload: Any,
    *,
    agent_id: str = "",
    replay_scope: str = "",
) -> str:
    encoded = json.dumps(
        {"payload": payload, "replay_scope": str(replay_scope or "").strip()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    partition = _replay_partition(agent_id)
    return hashlib.sha256(f"video-edit\x00{partition}\x00{operation}\x00".encode() + encoded).hexdigest()


def _headers(
    operation: str,
    payload: Any,
    *,
    content_type: str = "application/json",
    agent_id: str = "",
    replay_scope: str = "",
) -> dict[str, str]:
    key = _request_key(
        operation,
        payload,
        agent_id=agent_id,
        replay_scope=replay_scope,
    )
    return {
        "Accept": "application/json",
        "Content-Type": content_type,
        "X-Request-Id": key,
        "Idempotency-Key": key,
        "User-Agent": "hermes-video-edit-plugin/1",
    }


def _decode_response(response: Any) -> Any:
    body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise VideoClientError("video service response is too large", status=getattr(response, "status", 0))
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise VideoClientError("video service response is invalid", status=getattr(response, "status", 0)) from exc


def post_json(
    path: str,
    payload: Any,
    *,
    timeout: float = 1800.0,
    internal: bool = False,
    agent_id: str = "",
    replay_scope: str = "",
) -> Any:
    base = _internal_base() if internal else _base_url()
    url = f"{base}/{path.lstrip('/')}"
    headers = _headers(
        path,
        payload,
        agent_id=agent_id,
        replay_scope=replay_scope,
    )
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            status = int(getattr(response, "status", response.getcode()))
            body = _decode_response(response)
            if status >= 400:
                raise VideoClientError("video service rejected request", status=status, body=body)
            return body
    except urllib.error.HTTPError as exc:
        try:
            body = _decode_response(exc)
        except Exception:
            body = None
        # A 401 is a provider/device admission failure, not a transient video
        # caller capability. Retrying the same request only repeats a side effect
        # and hides the real upstream failure.
        raise VideoClientError("video service rejected request", status=exc.code, body=body) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise VideoClientError("video service request failed", transient=True) from exc


def _multipart_parts(
    files: list[Path],
    sizes: list[int],
    boundary: str,
) -> tuple[list[bytes], list[bytes], int]:
    preambles: list[bytes] = []
    epilogues: list[bytes] = []
    total = 0
    for path, size in zip(files, sizes):
        mime = video_media_type(path)
        safe_name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        preamble = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{safe_name}\"\r\nContent-Type: {mime}\r\n\r\n"
        ).encode()
        preambles.append(preamble)
        epilogues.append(b"\r\n")
        total += len(preamble) + size + 2
    return preambles, epilogues, total


def _open_upload_sources(
    files: list[Path],
) -> list[tuple[Path, int, os.stat_result]]:
    opened: list[tuple[Path, int, os.stat_result]] = []
    try:
        for path in files:
            descriptor = os.open(
                path,
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
                    raise VideoClientError("video upload source is invalid")
                validate_video_descriptor(path, descriptor)
            except Exception:
                os.close(descriptor)
                raise
            opened.append((path, descriptor, info))
        return opened
    except Exception:
        for _, descriptor, _ in opened:
            os.close(descriptor)
        raise


def _inspect_opened_sources(
    opened: list[tuple[Path, int, os.stat_result]],
    replay_scope: str,
) -> list[tuple[Path, int, os.stat_result]]:
    identities = normalizer.inspect_files(
        [path for path, _, _ in opened],
        replay_scope,
    )
    if len(identities) != len(opened):
        raise VideoClientError("video media inspection is invalid")

    refreshed: list[tuple[Path, int, os.stat_result]] = []
    for (path, descriptor, _), identity in zip(opened, identities):
        current = os.fstat(descriptor)
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        if current_identity != identity:
            raise VideoClientError("video upload source changed")
        refreshed.append((path, descriptor, current))
    return refreshed


def upload(
    files: list[Path],
    *,
    timeout: float = 1800.0,
    agent_id: str = "",
    replay_scope: str = "",
) -> Any:
    if not files or len(files) > MAX_UPLOAD_FILES:
        raise VideoClientError("invalid upload file count")
    try:
        opened = _open_upload_sources(files)
    except OSError as exc:
        raise VideoClientError("video upload source is unavailable") from exc
    connection: http.client.HTTPConnection | None = None
    try:
        sizes = [info.st_size for _, _, info in opened]
        total_size = sum(sizes)
        if total_size > MAX_UPLOAD_BYTES:
            raise VideoClientError("upload exceeds size limit")
        opened = _inspect_opened_sources(opened, replay_scope)
        boundary = "----hermes-video-edit-" + hashlib.sha256(
            "\x00".join(str(p) for p in files).encode()
        ).hexdigest()[:24]
        preambles, epilogues, body_size = _multipart_parts(
            files,
            sizes,
            boundary,
        )
        closing = f"--{boundary}--\r\n".encode()
        body_size += len(closing)
        base = urllib.parse.urlparse(_base_url())
        headers = _headers(
            "assets/upload",
            [str(path) for path in files],
            content_type=f"multipart/form-data; boundary={boundary}",
            agent_id=agent_id,
            replay_scope=replay_scope,
        )
        headers["Content-Length"] = str(body_size)
        connection = http.client.HTTPConnection(
            base.hostname,
            base.port or 80,
            timeout=timeout,
        )
        connection.putrequest("POST", base.path.rstrip("/") + "/assets/upload")
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        for (path, descriptor, opened_info), preamble, epilogue in zip(
            opened,
            preambles,
            epilogues,
        ):
            try:
                current_info = os.stat(path, follow_symlinks=False)
            except OSError as exc:
                raise VideoClientError("video upload source changed") from exc
            if (
                not stat.S_ISREG(current_info.st_mode)
                or current_info.st_dev != opened_info.st_dev
                or current_info.st_ino != opened_info.st_ino
                or current_info.st_size != opened_info.st_size
            ):
                raise VideoClientError("video upload source changed")
            connection.send(preamble)
            os.lseek(descriptor, 0, os.SEEK_SET)
            remaining = opened_info.st_size
            while remaining > 0:
                chunk = os.read(descriptor, min(CHUNK_BYTES, remaining))
                if not chunk:
                    raise VideoClientError("video upload source changed")
                connection.send(chunk)
                remaining -= len(chunk)
            finished_info = os.fstat(descriptor)
            if (
                finished_info.st_size != opened_info.st_size
                or finished_info.st_mtime_ns != opened_info.st_mtime_ns
                or finished_info.st_ctime_ns != opened_info.st_ctime_ns
            ):
                raise VideoClientError("video upload source changed")
            connection.send(epilogue)
        connection.send(closing)
        response = connection.getresponse()
        body = response.read(MAX_RESPONSE_BYTES + 1)
        if len(body) > MAX_RESPONSE_BYTES:
            raise VideoClientError("video service response is too large", status=response.status)
        try:
            decoded = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, ValueError) as exc:
            raise VideoClientError("video service response is invalid", status=response.status) from exc
        if response.status >= 400:
            raise VideoClientError("video service rejected upload", status=response.status, body=decoded)
        return decoded
    except (OSError, socket.timeout) as exc:
        raise VideoClientError("video upload failed", transient=True) from exc
    finally:
        for _, descriptor, _ in opened:
            os.close(descriptor)
        if connection is not None:
            connection.close()


def extract_upload_keys(body: Any) -> list[str]:
    data = body.get("data") if isinstance(body, dict) else None
    uploads = data.get("uploads") if isinstance(data, dict) else None
    if not isinstance(uploads, list):
        raise VideoClientError("video upload response is invalid")
    keys: list[str] = []
    for item in uploads:
        if not isinstance(item, dict) or not isinstance(item.get("object_key"), str) or not item["object_key"].strip():
            raise VideoClientError("video upload response is invalid")
        keys.append(item["object_key"].strip())
    return keys


def extract_project(body: Any) -> dict[str, Any]:
    candidates: list[Any] = []
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, dict):
            candidates.append(data.get("project"))
            candidates.append(data)
        candidates.extend([body.get("project"), body])
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("project_id") or "").strip():
            return dict(candidate)
    raise VideoClientError("video project response is invalid")


def create_project(
    object_keys: list[str],
    preferences: dict[str, Any],
    *,
    user_prompt: str = "",
    agent_id: str = "",
    workflow_id: str = "",
) -> dict[str, Any]:
    payload = {
        "object_keys": object_keys,
        "mode": "rendered",
        "aspect_ratio": preferences.get("aspect_ratio", "9:16"),
        "tips": [preferences.get("style", "freestyle")],
        "duration": int(preferences.get("duration", 60)),
    }
    prompt = str(user_prompt or preferences.get("user_prompt") or "").strip()
    if prompt:
        payload["user_prompt"] = prompt[:512]
    return extract_project(
        post_json(
            "projects",
            payload,
            timeout=1800.0,
            agent_id=agent_id,
            # Retry/resume of one workflow must replay, while a later explicit
            # re-edit of identical material must create a new cloud project.
            replay_scope=str(workflow_id or "").strip(),
        )
    )


def poll_project(project_id: str, *, timeout: float = 120.0, agent_id: str = "") -> dict[str, Any]:
    body = post_json("projects/batch", [project_id], timeout=timeout, agent_id=agent_id)
    projects: list[Any] = []
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, dict) and isinstance(data.get("projects"), list):
            projects = data["projects"]
        elif isinstance(data, list):
            projects = data
    for project in projects:
        if isinstance(project, dict) and str(project.get("project_id") or "").strip() == project_id:
            return dict(project)
    raise VideoClientError("video project poll response is invalid")


def _resolve_download_target(raw: str) -> tuple[urllib.parse.ParseResult, tuple[str, ...]] | None:
    parsed = urllib.parse.urlparse(str(raw or ""))
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        return None
    host = parsed.hostname.lower()
    if host in {"localhost"} or host.endswith((".local", ".lan", ".internal")):
        return None
    try:
        port = parsed.port or 443
    except ValueError:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            infos = socket.getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except OSError:
            return None
        addresses: list[str] = []
        for info in infos:
            sockaddr = info[4] if len(info) > 4 else ()
            raw_address = sockaddr[0] if sockaddr else ""
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError:
                return None
            if address.compressed not in addresses:
                addresses.append(address.compressed)
        if not addresses:
            return None
    else:
        addresses = [address.compressed]
    parsed_addresses = [ipaddress.ip_address(item) for item in addresses]
    if any(
        item.is_private
        or item.is_loopback
        or item.is_link_local
        or item.is_multicast
        or item.is_reserved
        or item.is_unspecified
        for item in parsed_addresses
    ):
        return None
    return parsed, tuple(addresses)


def _download_allowed(raw: str) -> bool:
    """Validate a result URL without performing a second DNS lookup."""
    return _resolve_download_target(raw) is not None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect is not allowed", headers, fp)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that only dials the addresses admitted before connect."""

    def __init__(self, host: str, *, pinned_addresses: tuple[str, ...], **kwargs: Any):
        self._pinned_addresses = pinned_addresses
        super().__init__(host, **kwargs)
        # HTTPConnection stores the module-level resolver on the instance;
        # replace that callback after the base constructor has run.
        self._create_connection = self._pinned_create_connection

    def _pinned_create_connection(
        self,
        address: tuple[str, int],
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        last_error: OSError | None = None
        for pinned in self._pinned_addresses:
            try:
                return socket.create_connection(
                    (pinned, address[1]),
                    timeout,
                    source_address,
                )
            except OSError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("video result host has no pinned address")


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, pinned_addresses: tuple[str, ...]):
        super().__init__()
        self._pinned_addresses = pinned_addresses

    def https_open(self, request: urllib.request.Request):  # type: ignore[override]
        return self.do_open(
            lambda host, **kwargs: _PinnedHTTPSConnection(
                host,
                pinned_addresses=self._pinned_addresses,
                **kwargs,
            ),
            request,
            context=self._context,
        )


def download(result_url: str, target: Path, *, timeout: float = 1800.0) -> dict[str, Any]:
    resolved_target = _resolve_download_target(result_url)
    if resolved_target is None:
        raise VideoClientError("video result URL is not allowed")
    _, pinned_addresses = resolved_target
    request = urllib.request.Request(result_url, headers={"User-Agent": "hermes-video-edit-plugin/1"})
    fd, part_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=str(target.parent)
    )
    part = Path(part_name)
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        _PinnedHTTPSHandler(pinned_addresses),
        _NoRedirect(),
    )
    total = 0
    digest = hashlib.sha256()
    try:
        with opener.open(request, timeout=timeout) as response, os.fdopen(fd, "wb") as stream:
            fd = -1
            while True:
                chunk = response.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > 4 * 1024 * 1024 * 1024:
                    raise VideoClientError("video result exceeds size limit")
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, target)
    except urllib.error.HTTPError as exc:
        if fd >= 0:
            os.close(fd)
        try:
            part.unlink()
        except OSError:
            pass
        if exc.code in RESULT_URL_UNAVAILABLE_STATUSES:
            raise ResultURLUnavailable(
                "video result URL is unavailable",
                status=exc.code,
                transient=True,
            ) from exc
        raise
    except Exception:
        if fd >= 0:
            os.close(fd)
        try:
            part.unlink()
        except OSError:
            pass
        raise
    return {"path": str(target), "size": total, "sha256": digest.hexdigest()}


def file_evidence(target: Path) -> dict[str, Any]:
    """Rebuild bounded evidence for a result committed before state persistence."""
    if target.is_symlink() or not target.is_file():
        raise VideoClientError("video result is unavailable")
    total = 0
    digest = hashlib.sha256()
    try:
        with target.open("rb") as stream:
            while True:
                chunk = stream.read(CHUNK_BYTES)
                if not chunk:
                    break
                total += len(chunk)
                if total > 4 * 1024 * 1024 * 1024:
                    raise VideoClientError("video result exceeds size limit")
                digest.update(chunk)
    except OSError as exc:
        raise VideoClientError("video result is unavailable") from exc
    if total <= 0:
        raise VideoClientError("video result is unavailable")
    return {"path": str(target), "size": total, "sha256": digest.hexdigest()}


def proactive_resolve(manifest_id: str, *, agent_id: str = "") -> dict[str, Any]:
    body = post_json(
        "manifest/resolve",
        {"manifest_id": manifest_id},
        timeout=30.0,
        internal=True,
        agent_id=agent_id,
    )
    if not isinstance(body, dict):
        raise VideoClientError("proactive manifest response is invalid")
    return body


def proactive_report(manifest_id: str, output_path: str, *, agent_id: str = "") -> dict[str, Any]:
    body = post_json(
        "result",
        {"manifest_id": manifest_id, "output_path": output_path},
        timeout=30.0,
        internal=True,
        agent_id=agent_id,
    )
    return body if isinstance(body, dict) else {"ok": True}
