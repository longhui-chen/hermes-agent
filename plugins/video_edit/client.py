"""Small bounded HTTP client for the local-server video business surface."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import mimetypes
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from plugins.video_edit.paths import VideoPathError

DEFAULT_BASE = "http://127.0.0.1:19090/api/v1/ai-proxy/business"
DEFAULT_INTERNAL_BASE = "http://127.0.0.1:19090/api/v1/internal/proactive-video"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_FILES = 8
MAX_UPLOAD_BYTES = 3 * 1024 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024


class VideoClientError(RuntimeError):
    def __init__(self, message: str, *, status: int = 0, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


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


def _platform_token() -> str:
    try:
        from agent.secret_scope import get_secret

        token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    except Exception as exc:
        raise VideoClientError("video edit platform identity is unavailable") from exc
    if not token:
        raise VideoClientError("video edit platform identity is unavailable")
    return token


def _request_key(operation: str, payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(f"video-edit\x00{operation}\x00".encode() + encoded).hexdigest()


def _headers(operation: str, payload: Any, *, content_type: str = "application/json") -> dict[str, str]:
    key = _request_key(operation, payload)
    return {
        "Accept": "application/json",
        "Content-Type": content_type,
        "X-Request-Id": key,
        "Idempotency-Key": key,
        # This is the pre-existing Hermes platform identity. It is not a
        # video capability and is never exposed as a model argument.
        "X-Zettlab-Agent-Action-Token": _platform_token(),
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


def post_json(path: str, payload: Any, *, timeout: float = 1800.0, internal: bool = False) -> Any:
    base = _internal_base() if internal else _base_url()
    url = f"{base}/{path.lstrip('/')}"
    headers = _headers(path, payload)
    request = urllib.request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
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
        # A 401 is a stable caller-identity failure, not a transient turn
        # capability. Retrying the same request only repeats a side effect and
        # recreates the authorization loop this plugin is meant to remove.
        raise VideoClientError("video service rejected request", status=exc.code, body=body) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise VideoClientError("video service request failed") from exc


def _multipart_parts(files: list[Path], boundary: str) -> tuple[list[bytes], list[bytes], int]:
    preambles: list[bytes] = []
    epilogues: list[bytes] = []
    total = 0
    for path in files:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        safe_name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        preamble = (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; "
            f"filename=\"{safe_name}\"\r\nContent-Type: {mime}\r\n\r\n"
        ).encode()
        preambles.append(preamble)
        epilogues.append(b"\r\n")
        total += len(preamble) + path.stat().st_size + 2
    return preambles, epilogues, total


def upload(files: list[Path], *, timeout: float = 1800.0) -> Any:
    if not files or len(files) > MAX_UPLOAD_FILES:
        raise VideoClientError("invalid upload file count")
    total_size = sum(path.stat().st_size for path in files)
    if total_size > MAX_UPLOAD_BYTES:
        raise VideoClientError("upload exceeds size limit")
    boundary = "----hermes-video-edit-" + hashlib.sha256("\x00".join(str(p) for p in files).encode()).hexdigest()[:24]
    preambles, epilogues, body_size = _multipart_parts(files, boundary)
    closing = f"--{boundary}--\r\n".encode()
    body_size += len(closing)
    base = urllib.parse.urlparse(_base_url())
    headers = _headers("assets/upload", [str(path) for path in files], content_type=f"multipart/form-data; boundary={boundary}")
    headers["Content-Length"] = str(body_size)
    connection = http.client.HTTPConnection(base.hostname, base.port or 80, timeout=timeout)
    try:
        connection.putrequest("POST", base.path.rstrip("/") + "/assets/upload")
        for key, value in headers.items():
            connection.putheader(key, value)
        connection.endheaders()
        for path, preamble, epilogue in zip(files, preambles, epilogues):
            connection.send(preamble)
            with path.open("rb") as stream:
                while True:
                    chunk = stream.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    connection.send(chunk)
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
        raise VideoClientError("video upload failed") from exc
    finally:
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


def create_project(object_keys: list[str], preferences: dict[str, Any], *, user_prompt: str = "") -> dict[str, Any]:
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
    return extract_project(post_json("projects", payload, timeout=1800.0))


def poll_project(project_id: str, *, timeout: float = 120.0) -> dict[str, Any]:
    body = post_json("projects/batch", [project_id], timeout=timeout)
    if isinstance(body, dict):
        data = body.get("data")
        projects = data.get("projects") if isinstance(data, dict) else None
        if isinstance(projects, list) and projects and isinstance(projects[0], dict):
            return dict(projects[0])
        if isinstance(data, list) and data and isinstance(data[0], dict):
            return dict(data[0])
    raise VideoClientError("video project poll response is invalid")


def _download_allowed(raw: str) -> bool:
    parsed = urllib.parse.urlparse(str(raw or ""))
    if parsed.scheme != "https" or parsed.username or parsed.password or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost"} or host.endswith((".local", ".lan", ".internal")):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (address.is_private or address.is_loopback or address.is_link_local or address.is_multicast or address.is_reserved)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise urllib.error.HTTPError(req.full_url, code, "redirect is not allowed", headers, fp)


def download(result_url: str, target: Path, *, timeout: float = 1800.0) -> dict[str, Any]:
    if not _download_allowed(result_url):
        raise VideoClientError("video result URL is not allowed")
    request = urllib.request.Request(result_url, headers={"User-Agent": "hermes-video-edit-plugin/1"})
    part = target.with_name(target.name + ".part")
    opener = urllib.request.build_opener(_NoRedirect())
    total = 0
    digest = hashlib.sha256()
    try:
        with opener.open(request, timeout=timeout) as response, part.open("wb") as stream:
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
    except Exception:
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


def proactive_resolve(manifest_id: str) -> dict[str, Any]:
    body = post_json("manifest/resolve", {"manifest_id": manifest_id}, timeout=30.0, internal=True)
    if not isinstance(body, dict):
        raise VideoClientError("proactive manifest response is invalid")
    return body


def proactive_report(manifest_id: str, output_path: str) -> dict[str, Any]:
    body = post_json("result", {"manifest_id": manifest_id, "output_path": output_path}, timeout=30.0, internal=True)
    return body if isinstance(body, dict) else {"ok": True}
