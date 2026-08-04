"""Shared client for Zettlab media generation via local-server ai-proxy.

Hermes runs on the device and must not hold a cloud user token. The local-server
loopback ai-proxy mints the device IoT credential and forwards these calls to
zettlab-ai-gateway.
"""

from __future__ import annotations

import atexit
import base64
import binascii
import io
import ipaddress
import json
import multiprocessing
import os
import queue
import stat
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse, urlunparse
from urllib.request import url2pathname

import requests
from agent.secret_scope import get_secret
from tools.interrupt import is_interrupted

DEFAULT_BASE_URL = "http://127.0.0.1:9090/api/v1/ai-proxy/v1"
CAPABILITY_TIMEOUT = 5.0
REQUEST_TIMEOUT = 30.0
MAX_CAPABILITY_RESPONSE_BYTES = 256 * 1024
MAX_ERROR_RESPONSE_BYTES = 64 * 1024
MAX_MEDIA_REQUEST_BYTES = 7 * 1024 * 1024
MAX_MEDIA_RESPONSE_BYTES = 1024 * 1024
MAX_INLINE_IMAGE_BYTES = 5 * 1024 * 1024
MAX_INPUT_IMAGE_URL_BYTES = 8 * 1024
LOCAL_IMAGE_READ_TIMEOUT = 30.0
ACTION_TOKEN_HEADER = "X-Zettlab-Agent-Action-Token"
ARTIFACT_SESSION_HEADER = "X-Zettlab-Artifact-Session-Id"
MAX_MEDIA_HTTP_WORKERS = 2
_STARTER_CAPACITY = threading.BoundedSemaphore(value=MAX_MEDIA_HTTP_WORKERS)
_SUPPORTED_IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp"}
_IMAGE_READ_CHUNK_BYTES = 48 * 1024
_MAX_LOCAL_IMAGE_PATH_CHARS = 4096


class ZettlabMediaError(RuntimeError):
    """Raised when the local media generation proxy cannot satisfy a request."""


class ZettlabMediaDeadlineError(ZettlabMediaError):
    """Raised when one bounded HTTP attempt exceeds its wall-clock deadline."""


def _watch_parent(parent_pid: int) -> None:
    parent = multiprocessing.parent_process()
    if parent is not None:
        while parent.is_alive():
            time.sleep(0.2)
        os._exit(1)
    while os.getppid() == parent_pid:
        time.sleep(0.2)
    os._exit(1)


def _dispose_worker_process(process: Any, *, force: bool) -> None:
    alive = False
    try:
        alive = process.is_alive()
    except (AssertionError, OSError, ValueError):
        pass
    if force and alive:
        try:
            process.terminate()
        except (AssertionError, OSError, ValueError):
            pass
    try:
        process.join(timeout=0.2)
    except (AssertionError, OSError, ValueError):
        pass
    try:
        alive = process.is_alive()
    except (AssertionError, OSError, ValueError):
        alive = False
    if alive:
        try:
            process.kill()
        except (AssertionError, OSError, ValueError):
            pass
        try:
            process.join(timeout=0.2)
        except (AssertionError, OSError, ValueError):
            pass
    try:
        process.close()
    except (AssertionError, OSError, ValueError):
        pass


def _safe_close_worker_resource(resource: Any) -> None:
    try:
        resource.close()
    except (OSError, ValueError):
        pass


def _media_http_worker(connection: Any, parent_pid: int, request_buffer: Any, request_length: int) -> None:
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    request = json.loads(bytes(request_buffer[:request_length]).decode("utf-8"))
    session = requests.Session()
    session.trust_env = False
    response = None
    try:
        response = session.request(
            method=request["method"],
            url=request["url"],
            json=request.get("json"),
            headers=request.get("headers"),
            timeout=request["timeout"],
            allow_redirects=False,
            stream=True,
        )
        if response.status_code >= 400:
            limit = MAX_ERROR_RESPONSE_BYTES
        elif request["url"].endswith("/media/generation-capabilities"):
            limit = MAX_CAPABILITY_RESPONSE_BYTES
        else:
            limit = MAX_MEDIA_RESPONSE_BYTES
        try:
            body = response.raw.read(limit + 1, decode_content=True)
        except TypeError:
            body = response.raw.read(limit + 1)
        if len(body) > limit:
            connection.send({"error": "response_too_large"})
        else:
            connection.send({"status_code": response.status_code, "body": body})
    except requests.Timeout as exc:
        connection.send({"error": "timeout", "message": str(exc)})
    except requests.RequestException as exc:
        connection.send({"error": "request", "message": str(exc)})
    except Exception as exc:
        connection.send({"error": "internal", "message": str(exc)})
    finally:
        if response is not None:
            response.close()
        session.close()
        connection.close()


# requests enforces socket-idle timeouts, not a wall-clock deadline. Keeping the
# client in a short-lived helper process lets the parent kill it when a loopback
# peer stalls while sending headers or a response body.
class _MediaHTTPWorker:
    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None
        self._payload: Any = None

    def request(self, method: str, url: str, *, deadline: float, **kwargs: Any) -> requests.Response:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise ZettlabMediaDeadlineError("media HTTP request deadline exceeded while waiting")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ZettlabMediaDeadlineError("media HTTP request deadline exceeded before send")
            self._ensure_started({
                "method": method,
                "url": url,
                "json": kwargs.get("json"),
                "headers": kwargs.get("headers"),
                "timeout": max(0.2, remaining),
            }, deadline)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._reset(force=True)
                    raise ZettlabMediaDeadlineError("media HTTP request deadline exceeded")
                if is_interrupted():
                    self._reset(force=True)
                    raise ZettlabMediaError("media generation interrupted")
                try:
                    ready = self._connection.poll(min(0.1, remaining))
                except (EOFError, OSError) as exc:
                    self._reset(force=True)
                    raise requests.ConnectionError("media HTTP worker poll failed") from exc
                if not ready:
                    if not self._process.is_alive():
                        try:
                            if self._connection.poll(0):
                                result = self._connection.recv()
                                try:
                                    return self._response_from_result(url, result)
                                finally:
                                    self._reset(force=False)
                        except (EOFError, OSError) as exc:
                            self._reset(force=True)
                            raise requests.ConnectionError("media HTTP worker result read failed") from exc
                        self._reset(force=True)
                        raise requests.ConnectionError("media HTTP worker exited unexpectedly")
                    continue
                try:
                    result = self._connection.recv()
                except (EOFError, OSError) as exc:
                    self._reset(force=True)
                    raise requests.ConnectionError("media HTTP worker closed unexpectedly") from exc
                try:
                    return self._response_from_result(url, result)
                finally:
                    self._reset(force=False)
        finally:
            self._lock.release()

    def close(self) -> None:
        self._reset(force=True)

    def _ensure_started(self, request: Dict[str, Any], deadline: float) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._reset(force=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ZettlabMediaDeadlineError("media HTTP process start deadline exceeded")
        finished = threading.Event()
        cancelled = threading.Event()
        state_lock = threading.Lock()
        state: Dict[str, Any] = {}

        def cleanup(resources: Dict[str, Any]) -> None:
            parent = resources.get("parent")
            child = resources.get("child")
            process = resources.get("process")
            if parent is not None:
                self._safe_close(parent)
            if child is not None:
                self._safe_close(child)
            if process is not None:
                self._dispose_process(process, force=True)

        def start_process() -> None:
            resources: Dict[str, Any] = {}
            try:
                payload = json.dumps(request, separators=(",", ":")).encode("utf-8")
                request_buffer = self._context.RawArray("B", len(payload))
                request_buffer[:len(payload)] = payload
                parent, child = self._context.Pipe()
                resources.update(parent=parent, child=child, payload=request_buffer)
                process = self._context.Process(
                    target=_media_http_worker,
                    args=(child, os.getpid(), request_buffer, len(payload)),
                    daemon=True,
                )
                resources["process"] = process
                process.start()
            except Exception as exc:
                resources["failure"] = exc
            finally:
                try:
                    with state_lock:
                        should_cleanup = cancelled.is_set()
                        if not should_cleanup:
                            state.update(resources)
                        finished.set()
                    if should_cleanup or resources.get("failure") is not None:
                        cleanup(resources)
                finally:
                    _STARTER_CAPACITY.release()

        if not _STARTER_CAPACITY.acquire(blocking=False):
            raise ZettlabMediaDeadlineError("media HTTP process starter capacity exhausted")
        try:
            threading.Thread(target=start_process, name="zettlab-media-spawn", daemon=True).start()
        except Exception:
            _STARTER_CAPACITY.release()
            raise
        while not finished.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                with state_lock:
                    cancelled.set()
                    cleanup_now = dict(state) if finished.is_set() else None
                if cleanup_now:
                    cleanup(cleanup_now)
                raise ZettlabMediaDeadlineError("media HTTP process start deadline exceeded")
            if is_interrupted():
                with state_lock:
                    cancelled.set()
                    cleanup_now = dict(state) if finished.is_set() else None
                if cleanup_now:
                    cleanup(cleanup_now)
                raise ZettlabMediaError("media generation interrupted")
        if time.monotonic() >= deadline:
            cleanup(state)
            raise ZettlabMediaDeadlineError("media HTTP process start deadline exceeded")
        if state.get("failure") is not None:
            raise requests.ConnectionError("media HTTP process failed to start") from state["failure"]
        self._safe_close(state["child"])
        self._connection = state["parent"]
        self._process = state["process"]
        self._payload = state["payload"]

    def _reset(self, *, force: bool) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        self._payload = None
        if connection is not None:
            self._safe_close(connection)
        if process is None:
            return
        self._dispose_process(process, force=force)

    @staticmethod
    def _dispose_process(process: Any, *, force: bool) -> None:
        _dispose_worker_process(process, force=force)

    @staticmethod
    def _safe_close(resource: Any) -> None:
        _safe_close_worker_resource(resource)

    @staticmethod
    def _response_from_result(url: str, result: Dict[str, Any]) -> requests.Response:
        error = result.get("error")
        if error == "response_too_large":
            raise ZettlabMediaError("media generation response exceeds maximum size")
        if error == "timeout":
            raise requests.Timeout(result.get("message") or "media HTTP request timed out")
        if error:
            raise requests.ConnectionError(result.get("message") or "media HTTP request failed")
        response = requests.Response()
        response.status_code = int(result.get("status_code") or 0)
        response.url = url
        response.raw = io.BytesIO(result.get("body") or b"")
        return response


def _prepare_local_image_path(
    source: str,
    task_id: Optional[str],
) -> tuple[str, str, str, tuple[str, ...], Optional[str]]:
    raw = str(source or "").strip()
    if not raw or len(raw) > _MAX_LOCAL_IMAGE_PATH_CHARS or "\x00" in raw:
        raise ZettlabMediaError("local image input path is invalid")
    _reject_windows_network_or_device_path(raw)

    if os.path.isabs(raw):
        candidate = raw
    else:
        parsed = urlparse(raw)
        if parsed.scheme.casefold() != "file":
            if parsed.scheme or parsed.netloc:
                raise ZettlabMediaError("image input URL must use HTTP or HTTPS")
            raise ZettlabMediaError("local image input must use an absolute path")
        if parsed.netloc.casefold() not in {"", "localhost"}:
            raise ZettlabMediaError("network file URLs are not supported for image input")
        if parsed.query or parsed.fragment:
            raise ZettlabMediaError("local image file URL must not include query or fragment")
        candidate = url2pathname(unquote(parsed.path))
        if not candidate:
            raise ZettlabMediaError("local image input path is invalid")

    _reject_windows_network_or_device_path(candidate)
    if not os.path.isabs(candidate):
        raise ZettlabMediaError("local image input must use an absolute path")
    normalized_task_id = str(task_id or "default").strip() or "default"
    try:
        from hermes_constants import get_hermes_home_override
        from tools.file_tools import local_host_read_context_for_task

        terminal_backend, managed_hermes_roots = local_host_read_context_for_task(
            normalized_task_id
        )
        hermes_home_override = get_hermes_home_override()
    except ValueError as exc:
        raise ZettlabMediaError(str(exc)) from exc
    return (
        os.path.normpath(candidate),
        normalized_task_id,
        terminal_backend,
        managed_hermes_roots,
        hermes_home_override,
    )


def _read_authorized_media_file(
    path: str,
    limit: int,
    expected_identity: tuple[int, int],
    result_buffer: Any,
) -> int:
    """Open one authorized regular file and copy it into the bounded buffer."""
    from agent.file_safety import raise_if_read_blocked

    if (
        not path
        or len(path) > _MAX_LOCAL_IMAGE_PATH_CHARS
        or "\x00" in path
        or not os.path.isabs(path)
    ):
        raise ZettlabMediaError("local image worker requires an absolute path")
    _reject_windows_network_or_device_path(path)
    raise_if_read_blocked(path)
    source_stat = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(source_stat.st_mode):
        raise ZettlabMediaError("local image input must be a regular file")
    source_identity = (source_stat.st_dev, source_stat.st_ino)
    if source_identity != expected_identity:
        raise ZettlabMediaError("local image input changed after authorization")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as image_file:
        file_stat = os.fstat(image_file.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise ZettlabMediaError("local image input must be a regular file")
        if (file_stat.st_dev, file_stat.st_ino) != source_identity:
            raise ZettlabMediaError("local image input changed while opening")
        if (file_stat.st_dev, file_stat.st_ino) != expected_identity:
            raise ZettlabMediaError("local image input changed after authorization")
        if file_stat.st_size <= 0:
            raise ZettlabMediaError("local image input must contain image bytes")
        if file_stat.st_size > limit:
            raise ZettlabMediaError("inline image input exceeds maximum size")
        total = 0
        while True:
            chunk = image_file.read(min(_IMAGE_READ_CHUNK_BYTES, limit - total + 1))
            if not chunk:
                break
            new_total = total + len(chunk)
            if new_total > limit:
                raise ZettlabMediaError("inline image input exceeds maximum size")
            result_buffer[total:new_total] = chunk
            total = new_total
        final_stat = os.fstat(image_file.fileno())
        initial_signature = (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
        )
        final_signature = (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_mtime_ns,
        )
        if final_signature != initial_signature:
            raise ZettlabMediaError("local image input changed while reading")
    if total <= 0:
        raise ZettlabMediaError("local image input must contain image bytes")
    return total


def _media_file_worker(
    connection: Any,
    parent_pid: int,
    source: str,
    limit: int,
    task_id: str,
    terminal_backend: str,
    managed_hermes_roots: tuple[str, ...],
    hermes_home_override: Optional[str],
    result_buffer: Any,
) -> None:
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    result: Dict[str, Any]
    override_token: Any = None
    try:
        from hermes_constants import (
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from tools.file_tools import resolve_host_read_path_for_task

        if hermes_home_override:
            override_token = set_hermes_home_override(hermes_home_override)
        resolved, expected_identity = resolve_host_read_path_for_task(
            source,
            task_id,
            terminal_backend=terminal_backend,
            managed_hermes_roots=managed_hermes_roots,
        )
        path = str(resolved)
        total = _read_authorized_media_file(
            path,
            limit,
            expected_identity,
            result_buffer,
        )
        result = {"length": total}
    except ValueError as exc:
        result = {"error": "blocked", "message": str(exc)}
    except ZettlabMediaError as exc:
        result = {"error": "media", "message": str(exc)}
    except OSError as exc:
        result = {"error": "media", "message": f"unable to read local image input: {exc}"}
    except Exception as exc:
        result = {"error": "internal", "message": str(exc)}
    finally:
        if override_token is not None:
            try:
                reset_hermes_home_override(override_token)
            except Exception:
                pass
    try:
        connection.send(result)
    except (BrokenPipeError, EOFError, OSError):
        pass
    finally:
        connection.close()


class _MediaFileWorker:
    def __init__(self) -> None:
        self._context = multiprocessing.get_context("spawn")
        self._lock = threading.Lock()
        self._process: Any = None
        self._connection: Any = None
        self._buffer: Any = None

    def read(
        self,
        source: str,
        limit: int,
        *,
        deadline: float,
        task_id: str = "default",
        terminal_backend: str = "local",
        managed_hermes_roots: tuple[str, ...] = (),
        hermes_home_override: Optional[str] = None,
    ) -> bytes:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self._lock.acquire(timeout=remaining):
            raise ZettlabMediaDeadlineError(
                "local image read deadline exceeded while waiting"
            )
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ZettlabMediaDeadlineError(
                    "local image read deadline exceeded before start"
                )
            self._ensure_started(
                source,
                limit,
                task_id,
                terminal_backend,
                managed_hermes_roots,
                hermes_home_override,
                deadline,
            )
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._reset(force=True)
                    raise ZettlabMediaDeadlineError("local image read deadline exceeded")
                if is_interrupted():
                    self._reset(force=True)
                    raise ZettlabMediaError("media generation interrupted")
                try:
                    ready = self._connection.poll(min(0.1, remaining))
                except (EOFError, OSError) as exc:
                    self._reset(force=True)
                    raise ZettlabMediaError("local image worker poll failed") from exc
                if not ready:
                    if not self._process.is_alive():
                        try:
                            if self._connection.poll(0):
                                result = self._connection.recv()
                                try:
                                    return self._value_from_result(result, limit)
                                finally:
                                    self._reset(force=False)
                        except (EOFError, OSError) as exc:
                            self._reset(force=True)
                            raise ZettlabMediaError(
                                "local image worker result read failed"
                            ) from exc
                        self._reset(force=True)
                        raise ZettlabMediaError("local image worker exited unexpectedly")
                    continue
                try:
                    result = self._connection.recv()
                except (EOFError, OSError) as exc:
                    self._reset(force=True)
                    raise ZettlabMediaError("local image worker closed unexpectedly") from exc
                try:
                    return self._value_from_result(result, limit)
                finally:
                    self._reset(force=False)
        finally:
            self._lock.release()

    def close(self) -> None:
        self._reset(force=True)

    def _ensure_started(
        self,
        source: str,
        limit: int,
        task_id: str,
        terminal_backend: str,
        managed_hermes_roots: tuple[str, ...],
        hermes_home_override: Optional[str],
        deadline: float,
    ) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._reset(force=True)
        if limit <= 0 or time.monotonic() >= deadline:
            raise ZettlabMediaDeadlineError("local image process start deadline exceeded")
        finished = threading.Event()
        cancelled = threading.Event()
        state_lock = threading.Lock()
        state: Dict[str, Any] = {}

        def cleanup(resources: Dict[str, Any]) -> None:
            parent = resources.get("parent")
            child = resources.get("child")
            process = resources.get("process")
            if parent is not None:
                _safe_close_worker_resource(parent)
            if child is not None:
                _safe_close_worker_resource(child)
            if process is not None:
                _dispose_worker_process(process, force=True)

        def start_process() -> None:
            resources: Dict[str, Any] = {}
            try:
                result_buffer = self._context.RawArray("B", limit)
                parent, child = self._context.Pipe()
                resources.update(parent=parent, child=child, buffer=result_buffer)
                process = self._context.Process(
                    target=_media_file_worker,
                    args=(
                        child,
                        os.getpid(),
                        source,
                        limit,
                        task_id,
                        terminal_backend,
                        managed_hermes_roots,
                        hermes_home_override,
                        result_buffer,
                    ),
                    daemon=True,
                )
                resources["process"] = process
                process.start()
            except Exception as exc:
                resources["failure"] = exc
            finally:
                try:
                    with state_lock:
                        should_cleanup = cancelled.is_set()
                        if not should_cleanup:
                            state.update(resources)
                        finished.set()
                    if should_cleanup or resources.get("failure") is not None:
                        cleanup(resources)
                finally:
                    _STARTER_CAPACITY.release()

        if not _STARTER_CAPACITY.acquire(blocking=False):
            raise ZettlabMediaDeadlineError("local image process starter capacity exhausted")
        try:
            threading.Thread(
                target=start_process,
                name="zettlab-media-file-spawn",
                daemon=True,
            ).start()
        except Exception:
            _STARTER_CAPACITY.release()
            raise
        while not finished.wait(timeout=min(0.05, max(0.0, deadline - time.monotonic()))):
            if time.monotonic() >= deadline or is_interrupted():
                with state_lock:
                    cancelled.set()
                    cleanup_now = dict(state) if finished.is_set() else None
                if cleanup_now:
                    cleanup(cleanup_now)
                if is_interrupted():
                    raise ZettlabMediaError("media generation interrupted")
                raise ZettlabMediaDeadlineError(
                    "local image process start deadline exceeded"
                )
        if time.monotonic() >= deadline:
            cleanup(state)
            raise ZettlabMediaDeadlineError("local image process start deadline exceeded")
        if state.get("failure") is not None:
            raise ZettlabMediaError("local image process failed to start") from state["failure"]
        _safe_close_worker_resource(state["child"])
        self._connection = state["parent"]
        self._process = state["process"]
        self._buffer = state["buffer"]

    def _reset(self, *, force: bool) -> None:
        connection, process = self._connection, self._process
        self._connection = None
        self._process = None
        self._buffer = None
        if connection is not None:
            _safe_close_worker_resource(connection)
        if process is not None:
            _dispose_worker_process(process, force=force)

    def _value_from_result(
        self,
        result: Dict[str, Any],
        limit: int,
    ) -> bytes:
        error = result.get("error")
        if error == "blocked":
            raise ValueError(result.get("message") or "local image read denied")
        if error:
            raise ZettlabMediaError(result.get("message") or "local image read failed")
        length = result.get("length")
        if (
            not isinstance(length, int)
            or isinstance(length, bool)
            or length <= 0
            or length > limit
            or self._buffer is None
        ):
            raise ZettlabMediaError("local image worker returned an invalid result")
        return bytes(self._buffer[:length])


class _MediaHTTPSession:
    trust_env = False

    def __init__(self, workers: Optional[List[_MediaHTTPWorker]] = None) -> None:
        self._workers = (
            workers
            if workers is not None
            else [_MediaHTTPWorker() for _ in range(MAX_MEDIA_HTTP_WORKERS)]
        )
        if not self._workers:
            raise ValueError("media HTTP session requires at least one worker")
        self._available: queue.LifoQueue[_MediaHTTPWorker] = queue.LifoQueue(
            maxsize=len(self._workers)
        )
        for worker in self._workers:
            self._available.put_nowait(worker)
        self._closed = False
        self._state_lock = threading.Lock()

    def request(self, method: str, url: str, *, timeout: float, allow_redirects: bool, **kwargs: Any) -> requests.Response:
        if allow_redirects:
            raise ZettlabMediaError("media HTTP redirects are not allowed")
        payload = kwargs.get("json")
        if payload is not None and len(json.dumps(payload).encode("utf-8")) > MAX_MEDIA_REQUEST_BYTES:
            raise ZettlabMediaError("media generation request exceeds maximum size")
        deadline = time.monotonic() + timeout
        with self._state_lock:
            if self._closed:
                raise ZettlabMediaError("media HTTP session is closed")
        try:
            worker = self._available.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty as exc:
            raise ZettlabMediaDeadlineError("media HTTP worker capacity exhausted") from exc
        with self._state_lock:
            if self._closed:
                self._available.put_nowait(worker)
                raise ZettlabMediaError("media HTTP session is closed")
        try:
            return worker.request(method, url, deadline=deadline, **kwargs)
        finally:
            self._available.put_nowait(worker)

    def get(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> requests.Response:
        return self.request("DELETE", url, **kwargs)

    def merge_environment_settings(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"proxies": {}}

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        for worker in self._workers:
            worker.close()


_FILE_WORKER = _MediaFileWorker()
_SESSION = _MediaHTTPSession()
atexit.register(_SESSION.close)
atexit.register(_FILE_WORKER.close)


def _config_section(media_type: str) -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get(f"{media_type}_gen") if isinstance(cfg, dict) else None
        if not isinstance(section, dict):
            return {}
        zettlab = section.get("zettlab")
        if isinstance(zettlab, dict):
            merged = dict(section)
            merged.update(zettlab)
            return merged
        return section
    except Exception:
        return {}


def base_url(media_type: str) -> str:
    explicit = str(get_secret("ZETTLAB_AI_PROXY_BASE_URL", "") or "").strip()
    configured = explicit or _config_section(media_type).get("base_url")
    if configured:
        raw = str(configured).strip().rstrip("/")
    else:
        append_url = str(get_secret("ZET_CHAT_APPEND_URL", "") or "").strip()
        if append_url:
            parsed_append = urlparse(append_url)
            raw = urlunparse((
                parsed_append.scheme,
                parsed_append.netloc,
                "/api/v1/ai-proxy/v1",
                "",
                "",
                "",
            ))
        else:
            raw = DEFAULT_BASE_URL
    raw = raw or DEFAULT_BASE_URL
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        port = parsed.port
    except ValueError:
        port = -1
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (
        parsed.scheme not in {"http", "https"}
        or not loopback
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or port == -1
    ):
        raise ZettlabMediaError("Zettlab ai-proxy base_url must be a plain loopback HTTP(S) URL")
    return raw


def get_capabilities(media_type: Optional[str] = None) -> Dict[str, Any]:
    mt = media_type or "image"
    deadline = time.monotonic() + CAPABILITY_TIMEOUT
    resp = _SESSION.get(
        f"{base_url(mt)}/media/generation-capabilities",
        timeout=max(0.2, deadline - time.monotonic()),
        allow_redirects=False,
        stream=True,
    )
    try:
        _raise_for_status(resp)
        data = _bounded_response_json(resp, MAX_CAPABILITY_RESPONSE_BYTES)
    finally:
        _close_response(resp)
    if not isinstance(data, dict):
        raise ZettlabMediaError("media capability response is not a JSON object")
    return data


def action_headers() -> Dict[str, str]:
    token = str(get_secret("ZETTLAB_AGENT_ACTION_TOKEN", "") or "").strip()
    if not token:
        raise ZettlabMediaError("ZETTLAB_AGENT_ACTION_TOKEN is required for media generation")
    return {ACTION_TOKEN_HEADER: token}


def _bounded_response_json(resp: requests.Response, limit: int) -> Any:
    if not isinstance(resp, requests.Response):
        try:
            return resp.json()
        except (ValueError, UnicodeError) as exc:
            raise ZettlabMediaError("media generation response is not valid JSON") from exc
    raw = getattr(resp, "raw", None)
    if raw is None or not hasattr(raw, "read"):
        raise ZettlabMediaError("media generation response body is unavailable")
    try:
        body = raw.read(limit + 1, decode_content=True)
    except TypeError:
        body = raw.read(limit + 1)
    if len(body) > limit:
        raise ZettlabMediaError("media generation response exceeds maximum size")
    try:
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ZettlabMediaError("media generation response is not valid JSON") from exc


def _close_response(resp: Any) -> None:
    close = getattr(resp, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _response_error(resp: requests.Response, *, job_id: Optional[str] = None) -> ZettlabMediaError:
    status = int(getattr(resp, "status_code", 0) or 0)
    payload: Dict[str, Any] = {}
    try:
        candidate = _bounded_response_json(resp, MAX_ERROR_RESPONSE_BYTES)
        if isinstance(candidate, dict):
            payload = candidate
    except Exception:
        pass
    error = payload.get("error")
    error_data = error if isinstance(error, dict) else {}
    code = str(error_data.get("code") or payload.get("error_code") or "").strip()
    message = str(error_data.get("message") or payload.get("error_message") or (error if isinstance(error, str) else "")).strip()
    retryable = error_data.get("retryable", payload.get("retryable"))
    resolved_job_id = str(payload.get("job_id") or job_id or "").strip()
    details = [f"HTTP {status}" if status else "HTTP request failed"]
    if code:
        details.append(f"code={code}")
    if message:
        details.append(f"message={message}")
    if isinstance(retryable, bool):
        details.append(f"retryable={str(retryable).lower()}")
    if resolved_job_id:
        details.append(f"job_id={resolved_job_id}")
    return ZettlabMediaError("media generation request failed: " + "; ".join(details))


def _raise_for_status(resp: requests.Response, *, job_id: Optional[str] = None) -> None:
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise _response_error(resp, job_id=job_id) from exc


def _failed_job_error(job: Dict[str, Any]) -> ZettlabMediaError:
    code = str(job.get("error_code") or "").strip()
    message = str(job.get("error_message") or job.get("status") or "failed").strip()
    retryable = bool(job.get("retryable"))
    job_id = str(job.get("job_id") or "").strip()
    return ZettlabMediaError(
        f"media generation job failed: code={code or 'unknown'}; message={message}; "
        f"retryable={str(retryable).lower()}; job_id={job_id or 'unknown'}"
    )


def type_capability(media_type: str) -> Dict[str, Any]:
    caps = get_capabilities(media_type)
    section = caps.get(media_type)
    return section if isinstance(section, dict) else {}


def list_models(media_type: str) -> List[Dict[str, Any]]:
    section = type_capability(media_type)
    if section.get("enabled") is False:
        return []
    models = section.get("models")
    if not isinstance(models, list):
        return []
    out: List[Dict[str, Any]] = []
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        normalized = dict(model)
        normalized["modalities"] = supported_modalities(section, model)
        out.append(normalized)
    return out


def default_model(media_type: str) -> Optional[str]:
    section = type_capability(media_type)
    return _resolve_model_from_section(media_type, section)


def resolve_model(media_type: str, requested: Optional[str] = None) -> Optional[str]:
    model_id, _ = resolve_model_with_capability(media_type, requested)
    return model_id


def resolve_model_with_capability(
    media_type: str,
    requested: Optional[str] = None,
) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    section = type_capability(media_type)
    model_id = _resolve_model_from_section(media_type, section, requested)
    models = section.get("models")
    if model_id and isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
                model_with_limits = dict(model)
                model_with_limits["_type_limits"] = section.get("limits")
                return model_id, model_with_limits
    return model_id, None


def selected_model_capability(media_type: str) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    section = type_capability(media_type)
    model_id = _resolve_model_from_section(media_type, section)
    models = section.get("models")
    if model_id and isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
                return section, model
    return section, None


def _resolve_model_from_section(
    media_type: str,
    section: Dict[str, Any],
    requested: Optional[str] = None,
) -> Optional[str]:
    if section.get("enabled") is False:
        return None
    gateway_default = section.get("default_model")
    models = section.get("models")
    if isinstance(models, list):
        model_ids = [
            model.get("id").strip()
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("id"), str)
            and model.get("id").strip()
        ]
        explicit = str(requested or "").strip()
        if explicit:
            return explicit if explicit in model_ids else None
        configured = _config_section(media_type).get("model")
        if isinstance(configured, str) and configured.strip():
            configured_id = configured.strip()
            return configured_id if configured_id in model_ids else None
        if "default_model" in section:
            if isinstance(gateway_default, str) and gateway_default.strip() in model_ids:
                return gateway_default.strip()
            return None
        # Compatibility with gateways that predate the explicit default_model
        # field. Reuse this same capability snapshot so a transient second GET
        # cannot invalidate an otherwise usable response.
        if model_ids:
            return model_ids[0]
    return None


def is_available(media_type: str) -> bool:
    try:
        section = type_capability(media_type)
    except Exception:
        return False
    models = section.get("models")
    return (
        bool(section.get("enabled"))
        and isinstance(models, list)
        and any(
            supported_modalities(section, model)
            for model in models
            if isinstance(model, dict)
        )
    )


def validate_remote_url(value: Optional[str], *, label: str) -> Optional[str]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
        host = parsed.hostname
    except ValueError as exc:
        raise ZettlabMediaError(
            f"{label} must be a valid HTTPS URL for Zettlab media generation"
        ) from exc
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or any(character.isspace() for character in raw)
    ):
        raise ZettlabMediaError(
            f"{label} must be a valid HTTPS URL for Zettlab media generation"
        )
    return raw


def remote_inputs(
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    primary = validate_remote_url(image_url, label="image_url")
    if primary:
        out.append({"url": primary, "role": "source"})
    for idx, ref in enumerate(reference_image_urls or []):
        normalized = validate_remote_url(ref, label=f"reference_image_urls[{idx}]")
        if normalized:
            out.append({"url": normalized, "role": "reference"})
    return out


def _effective_inline_image_limit(limits: Any) -> Optional[int]:
    limit = limits.get("max_inline_image_bytes") if isinstance(limits, dict) else None
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        return None
    return min(limit, MAX_INLINE_IMAGE_BYTES)


def normalized_modalities(model_capability: Optional[Dict[str, Any]]) -> List[str]:
    raw = model_capability.get("modalities") if isinstance(model_capability, dict) else None
    modalities: List[str] = []
    if isinstance(raw, list):
        for value in raw:
            normalized = value.strip().casefold() if isinstance(value, str) else ""
            if normalized in {"text", "image"} and normalized not in modalities:
                modalities.append(normalized)
    elif raw is None:
        modalities.append("text")
    return modalities


def supports_inline_image_input(
    type_section: Optional[Dict[str, Any]],
    model_capability: Optional[Dict[str, Any]],
) -> bool:
    limits = type_section.get("limits") if isinstance(type_section, dict) else None
    return (
        "image" in normalized_modalities(model_capability)
        and _effective_inline_image_limit(limits) is not None
    )


def supports_input_image_url(model_capability: Optional[Dict[str, Any]]) -> bool:
    return (
        "image" in normalized_modalities(model_capability)
        and isinstance(model_capability, dict)
        and model_capability.get("supports_input_image_url") is True
    )


def image_input_description(
    type_section: Optional[Dict[str, Any]],
    model_capability: Optional[Dict[str, Any]],
) -> str:
    supports_inline = supports_inline_image_input(type_section, model_capability)
    supports_url = supports_input_image_url(model_capability)
    if supports_inline and supports_url:
        return (
            "Pass one PNG, JPEG, or WebP image as a base64 Data URI, absolute "
            "local file path, file URL, or HTTPS URL. HTTPS URLs are passed "
            "through without being downloaded by Hermes."
        )
    if supports_url:
        return (
            "Pass one HTTPS image URL. It is passed through without being "
            "downloaded by Hermes."
        )
    return (
        "Pass one PNG, JPEG, or WebP image as a base64 Data URI, absolute "
        "local file path, or file URL."
    )


def supported_modalities(
    type_section: Optional[Dict[str, Any]],
    model_capability: Optional[Dict[str, Any]],
) -> List[str]:
    modalities = normalized_modalities(model_capability)
    if (
        "image" in modalities
        and not supports_inline_image_input(type_section, model_capability)
        and not supports_input_image_url(model_capability)
    ):
        modalities.remove("image")
    return modalities


def _inline_image_limit(model_capability: Optional[Dict[str, Any]]) -> int:
    limits = model_capability.get("_type_limits") if isinstance(model_capability, dict) else None
    limit = _effective_inline_image_limit(limits)
    if "image" not in normalized_modalities(model_capability) or limit is None:
        raise ZettlabMediaError(
            "Inline image input is not enabled for this Zettlab media generation model"
        )
    return limit


def validate_input_image_url(
    value: str,
    model_capability: Optional[Dict[str, Any]],
) -> str:
    """Validate the wire shape only; ai-api owns fetching the image."""
    if not supports_input_image_url(model_capability):
        raise ZettlabMediaError(
            "HTTPS image URLs are not enabled for this Zettlab media generation model"
        )
    raw = str(value or "").strip()
    if len(raw.encode("utf-8")) > MAX_INPUT_IMAGE_URL_BYTES:
        raise ZettlabMediaError("image input URL exceeds maximum size")
    return validate_remote_url(raw, label="image input") or ""


def _sniff_image_mime(raw: bytes) -> Optional[str]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_inline_image_data_uri(value: str, *, max_bytes: int) -> str:
    """Validate one bounded PNG/JPEG/WebP base64 data URI."""
    normalized = str(value or "").strip()
    header, separator, encoded = normalized.partition(",")
    if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
        raise ZettlabMediaError(
            "image input must be a local image path or data URI using PNG, JPEG, or WebP"
        )
    declared_mime = header[len("data:"):-len(";base64")].lower()
    if declared_mime not in _SUPPORTED_IMAGE_MIMES:
        raise ZettlabMediaError("image data URI must use PNG, JPEG, or WebP")
    if not encoded or len(encoded) % 4 != 0:
        raise ZettlabMediaError("image data URI must contain valid base64")

    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    decoded_length = len(encoded) // 4 * 3 - padding
    if decoded_length <= 0:
        raise ZettlabMediaError("image data URI must contain image bytes")
    if decoded_length > min(max_bytes, MAX_INLINE_IMAGE_BYTES):
        raise ZettlabMediaError("inline image input exceeds maximum size")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ZettlabMediaError("image data URI must contain valid base64") from exc
    detected_mime = _sniff_image_mime(raw)
    if detected_mime is None:
        raise ZettlabMediaError("image data URI does not contain a supported image")
    if detected_mime != declared_mime:
        raise ZettlabMediaError("image data URI MIME type does not match its bytes")
    return normalized


def _validate_image_data_uri(value: str, limit: int) -> str:
    return validate_inline_image_data_uri(value, max_bytes=limit)


def _reject_windows_network_or_device_path(source: str) -> None:
    normalized = source.replace("/", "\\")
    folded = normalized.casefold()
    if (
        normalized.startswith("\\\\")
        or folded.startswith("\\??\\")
        or folded.startswith("\\device\\")
        or folded.startswith("\\global??\\")
    ):
        raise ZettlabMediaError(
            "Windows network and device paths are not supported for image input"
        )


def _local_image_data_uri(
    source: str,
    limit: int,
    task_id: str,
    terminal_backend: str,
    managed_hermes_roots: tuple[str, ...],
    hermes_home_override: Optional[str],
) -> str:
    try:
        raw = _FILE_WORKER.read(
            source,
            limit,
            deadline=time.monotonic() + LOCAL_IMAGE_READ_TIMEOUT,
            task_id=task_id,
            terminal_backend=terminal_backend,
            managed_hermes_roots=managed_hermes_roots,
            hermes_home_override=hermes_home_override,
        )
    except ValueError as exc:
        raise ZettlabMediaError(str(exc)) from exc
    mime = _sniff_image_mime(raw[:16])
    if mime is None:
        raise ZettlabMediaError("local image input must be a PNG, JPEG, or WebP file")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def inline_image_input(
    image_url: Optional[str],
    reference_image_urls: Optional[List[str]],
    model_capability: Optional[Dict[str, Any]],
    *,
    task_id: Optional[str] = None,
) -> Optional[str]:
    """Return one bounded inline image accepted by the gateway's v1 contract."""
    if isinstance(reference_image_urls, str):
        references = [reference_image_urls]
    elif isinstance(reference_image_urls, (list, tuple)):
        references = list(reference_image_urls)
    else:
        references = []
    candidates = [
        str(value).strip()
        for value in [image_url, *references]
        if value is not None and str(value).strip()
    ]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ZettlabMediaError("exactly one image input is supported")
    source = candidates[0]
    if source.startswith("data:"):
        limit = _inline_image_limit(model_capability)
        return _validate_image_data_uri(source, limit)
    _reject_windows_network_or_device_path(source)
    try:
        parsed = urlparse(source)
    except ValueError as exc:
        raise ZettlabMediaError("image input must be a valid HTTPS URL") from exc
    if parsed.scheme.casefold() in {"http", "https"}:
        return validate_input_image_url(source, model_capability)
    limit = _inline_image_limit(model_capability)
    (
        prepared_source,
        normalized_task_id,
        terminal_backend,
        managed_hermes_roots,
        hermes_home_override,
    ) = _prepare_local_image_path(source, task_id)
    return _local_image_data_uri(
        prepared_source,
        limit,
        normalized_task_id,
        terminal_backend,
        managed_hermes_roots,
        hermes_home_override,
    )


def image_path_data_uri(path: os.PathLike[str] | str, *, max_bytes: int = MAX_INLINE_IMAGE_BYTES) -> str:
    """Encode one runtime-owned regular image file for the inline gateway field."""
    source = os.fspath(path)
    limit = min(max_bytes, MAX_INLINE_IMAGE_BYTES)
    try:
        if stat.S_ISLNK(os.lstat(source).st_mode):
            raise ZettlabMediaError("generated image input must not be a symbolic link")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(source, flags)
        with os.fdopen(descriptor, "rb") as handle:
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                raise ZettlabMediaError("generated image input must be a regular file")
            if metadata.st_size <= 0 or metadata.st_size > limit:
                raise ZettlabMediaError("generated image input exceeds maximum size")
            raw = handle.read(limit + 1)
    except OSError as exc:
        raise ZettlabMediaError(f"unable to read generated image input: {exc}") from exc
    if not raw or len(raw) > limit or len(raw) != metadata.st_size:
        raise ZettlabMediaError("generated image input exceeds maximum size")
    mime = _sniff_image_mime(raw)
    if mime is None:
        raise ZettlabMediaError("generated image input must be PNG, JPEG, or WebP")
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def create_and_wait(
    *,
    media_type: str,
    model: str,
    prompt: str,
    payload: Dict[str, Any],
    session_id: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
    poll_interval: float = 2.0,
) -> Dict[str, Any]:
    operation_timeout = float(timeout_seconds or _timeout_from_capability(media_type))
    deadline = time.monotonic() + operation_timeout
    body = {
        "media_type": media_type,
        "model": model,
        "prompt": prompt,
        **payload,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Scene-Type": "media_generation",
        "X-Step-Title": "media_generation",
        **action_headers(),
    }
    normalized_session_id = str(session_id or "").strip()
    if normalized_session_id:
        headers["X-Task-Id"] = normalized_session_id
    artifact_headers = dict(headers)
    if normalized_session_id:
        artifact_headers[ARTIFACT_SESSION_HEADER] = normalized_session_id
    create_deadline = deadline
    resp = _SESSION.post(
        f"{base_url(media_type)}/media/generation-jobs",
        json=body,
        headers=headers,
        timeout=max(0.2, create_deadline - time.monotonic()),
        allow_redirects=False,
        stream=True,
    )
    try:
        _raise_for_status(resp)
        job = _bounded_response_json(resp, MAX_MEDIA_RESPONSE_BYTES)
    finally:
        _close_response(resp)
    if not isinstance(job, dict):
        raise ZettlabMediaError("media generation job response is not a JSON object")
    if job.get("status") == "done":
        job_id = str(job.get("job_id") or "").strip()
        if normalized_session_id and job_id:
            return _finalize_artifact_or_fallback(
                media_type=media_type,
                job_id=job_id,
                headers=artifact_headers,
                deadline=deadline,
                fallback_job=job,
            )
        return job
    if job.get("status") in {"failed", "cancelled"}:
        raise _failed_job_error(job)

    job_id = str(job.get("job_id") or "").strip()
    if not job_id:
        raise ZettlabMediaError("media generation job response did not include job_id")

    last_poll_error: Optional[Exception] = None
    poll_delay = max(0.2, poll_interval)
    try:
        while time.monotonic() < deadline:
            _interruptible_sleep(min(poll_delay, max(0.0, deadline - time.monotonic())))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                poll_deadline = min(deadline, time.monotonic() + REQUEST_TIMEOUT)
                resp = _SESSION.get(
                    f"{base_url(media_type)}/media/generation-jobs/{job_id}",
                    headers=headers,
                    timeout=max(0.2, poll_deadline - time.monotonic()),
                    allow_redirects=False,
                    stream=True,
                )
                try:
                    try:
                        resp.raise_for_status()
                    except requests.HTTPError as exc:
                        status_code = exc.response.status_code if exc.response is not None else 0
                        if 400 <= status_code < 500 and status_code not in {408, 429}:
                            raise _response_error(resp, job_id=job_id) from exc
                        raise
                    job = _bounded_response_json(resp, MAX_MEDIA_RESPONSE_BYTES)
                finally:
                    _close_response(resp)
            except ZettlabMediaDeadlineError as exc:
                last_poll_error = exc
                poll_delay = min(15.0, poll_delay * 2)
                continue
            except requests.HTTPError as exc:
                last_poll_error = exc
                poll_delay = min(15.0, poll_delay * 2)
                continue
            except requests.RequestException as exc:
                last_poll_error = exc
                poll_delay = min(15.0, poll_delay * 2)
                continue
            last_poll_error = None
            poll_delay = max(0.2, poll_interval)
            if not isinstance(job, dict):
                raise ZettlabMediaError("media generation poll response is not a JSON object")
            status = str(job.get("status") or "")
            if status == "done":
                if normalized_session_id:
                    job = _finalize_artifact_or_fallback(
                        media_type=media_type,
                        job_id=job_id,
                        headers=artifact_headers,
                        deadline=deadline,
                        fallback_job=job,
                    )
                return job
            if status in {"failed", "cancelled"}:
                raise _failed_job_error(job)
        detail = f": {last_poll_error}" if last_poll_error is not None else ""
        raise ZettlabMediaError(f"media generation timed out{detail}; job_id={job_id}")
    except ZettlabMediaError as exc:
        if f"job_id={job_id}" in str(exc):
            raise
        raise ZettlabMediaError(f"{exc}; job_id={job_id}") from exc


def _finalize_artifact_or_fallback(
    *,
    media_type: str,
    job_id: str,
    headers: Dict[str, str],
    deadline: float,
    fallback_job: Dict[str, Any],
) -> Dict[str, Any]:
    retry_delay = 0.5
    for attempt in range(3):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return fallback_job
        try:
            resp = _SESSION.get(
                f"{base_url(media_type)}/media/generation-jobs/{job_id}",
                headers=headers,
                timeout=max(0.2, remaining),
                allow_redirects=False,
                stream=True,
            )
            try:
                status_code = int(getattr(resp, "status_code", 0) or 0)
                if status_code == 408 or status_code == 429 or status_code >= 500:
                    raise requests.HTTPError(
                        f"retryable artifact finalization HTTP {status_code}",
                        response=resp,
                    )
                _raise_for_status(resp, job_id=job_id)
                job = _bounded_response_json(resp, MAX_MEDIA_RESPONSE_BYTES)
            finally:
                _close_response(resp)
        except (ZettlabMediaDeadlineError, requests.RequestException):
            if attempt == 2:
                return fallback_job
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return fallback_job
            _interruptible_sleep(min(retry_delay, remaining))
            retry_delay = min(2.0, retry_delay * 2)
            continue
        if not isinstance(job, dict):
            raise ZettlabMediaError(
                f"media artifact finalization response is not a JSON object; job_id={job_id}"
            )
        status = str(job.get("status") or "")
        if status in {"failed", "cancelled"}:
            raise _failed_job_error(job)
        if status != "done":
            raise ZettlabMediaError(
                f"media artifact finalization returned status={status or 'unknown'}; job_id={job_id}"
            )
        return job
    return fallback_job


def _interruptible_sleep(delay: float) -> None:
    deadline = time.monotonic() + delay
    while time.monotonic() < deadline:
        if is_interrupted():
            raise ZettlabMediaError("media generation interrupted")
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


def first_asset_url(job: Dict[str, Any]) -> str:
    job_id = str(job.get("job_id") or "unknown").strip() or "unknown"
    assets = job.get("assets")
    if isinstance(assets, list) and assets and isinstance(assets[0], dict):
        url = assets[0].get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
    shortcut = job.get("image") or job.get("video")
    if isinstance(shortcut, str) and shortcut.strip():
        return shortcut.strip()
    if not isinstance(assets, list) or not assets:
        raise ZettlabMediaError(f"media generation completed without assets; job_id={job_id}")
    if not isinstance(assets[0], dict):
        raise ZettlabMediaError(f"media generation asset has invalid shape; job_id={job_id}")
    raise ZettlabMediaError(f"media generation asset has no retrievable URL; job_id={job_id}")


def first_asset_local_path(job: Dict[str, Any]) -> str:
    job_id = str(job.get("job_id") or "unknown").strip() or "unknown"
    assets = job.get("assets")
    if isinstance(assets, list) and assets and isinstance(assets[0], dict):
        path = assets[0].get("local_path")
        if isinstance(path, str) and path.strip() and assets[0].get("persisted") is True:
            return path.strip()
    raise ZettlabMediaError(
        f"media generation completed but the asset was not persisted on the device; "
        f"job_id={job_id}"
    )


def first_asset_location(
    job: Dict[str, Any],
    *,
    prefer_local: bool,
) -> str:
    if prefer_local:
        try:
            return first_asset_local_path(job)
        except ZettlabMediaError:
            pass
    return first_asset_url(job)


def _timeout_from_capability(media_type: str) -> int:
    try:
        return timeout_from_model_capability(media_type, {"_type_limits": type_capability(media_type).get("limits")})
    except Exception:
        return 30 * 60 if media_type == "video" else 10 * 60


def timeout_from_model_capability(media_type: str, model: Optional[Dict[str, Any]]) -> int:
    limits = model.get("_type_limits") if isinstance(model, dict) else None
    if isinstance(limits, dict):
        provider_timeout = int(limits.get("provider_timeout_seconds") or 0)
        finalization_timeout = int(limits.get("finalization_timeout_seconds") or 0)
        total = provider_timeout + finalization_timeout
        if total > 0:
            return total
    return 30 * 60 if media_type == "video" else 10 * 60
