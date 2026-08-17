"""Service-level orchestration for LSP clients.

The :class:`LSPService` is the bridge between the synchronous
file_operations layer and the async :class:`agent.lsp.client.LSPClient`.

Design choices:

- A **single asyncio event loop** runs in a background thread.  All
  client work happens on that loop.  Synchronous callers from
  ``tools/file_operations.py`` use :meth:`get_diagnostics_sync` to
  open + wait + drain in one blocking call.

- One client per ``(server_id, workspace_root)`` key.  Lazy spawn:
  the first request for a key spawns the client; subsequent requests
  re-use it.

- A **broken-set** records ``(server_id, workspace_root)`` pairs that
  failed to spawn or initialize.  These are never retried for the
  life of the service.  Mirrors OpenCode's design.

- A **delta baseline** map keeps "diagnostics-as-of-the-last-snapshot"
  per file.  ``snapshot_baseline()`` is called BEFORE a write; the
  next ``get_diagnostics_sync()`` returns only diagnostics that
  weren't in the baseline.  This is the lift from Claude Code's
  ``beforeFileEdited`` / ``getNewDiagnostics`` pattern, except wired
  to the local LSP layer instead of MCP IDE RPC.

The service is **off by default** — call :meth:`is_active` to check
whether it's actually doing anything.  When LSP is disabled in
config, when no git workspace can be detected, when all configured
servers are missing binaries and auto-install is off, ``is_active``
returns False and the file_operations layer falls through to the
in-process syntax check.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from agent.lsp import eventlog
from agent.lsp.client import (
    DIAGNOSTICS_DOCUMENT_WAIT,
    LSPClient,
)
from agent.lsp.servers import (
    ServerContext,
    find_server_for_file,
    language_id_for,
)
from agent.lsp.workspace import (
    clear_cache,
    resolve_workspace_for_file,
)

logger = logging.getLogger("agent.lsp.manager")

DEFAULT_IDLE_TIMEOUT = 600  # seconds; servers idle for >10min get reaped
MIN_IDLE_TIMEOUT = 30  # floor for config values; must exceed any per-op wait budget


class _BackgroundLoop:
    """A daemon thread that owns one asyncio event loop.

    Provides :meth:`run` for synchronous callers — submits a coroutine
    to the loop and blocks until it finishes (or a timeout fires).
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_forever,
            name="hermes-lsp-loop",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                loop.close()
            except Exception:  # noqa: BLE001
                pass

    def run(self, coro, *, timeout: Optional[float] = None) -> Any:
        """Submit a coroutine to the loop and block until done.

        Returns the coroutine's result, or raises its exception.
        """
        from agent.async_utils import safe_schedule_threadsafe
        if self._loop is None:
            if asyncio.iscoroutine(coro):
                coro.close()
            raise RuntimeError("background loop not started")
        fut = safe_schedule_threadsafe(coro, self._loop)
        if fut is None:
            raise RuntimeError("background loop not running")
        try:
            return fut.result(timeout=timeout)
        except Exception:
            fut.cancel()
            raise

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._loop = None
        self._thread = None


#: 第二个 waiter 加入在飞 shutdown 时的等待上限。
#: ⛔ 不许拍脑袋:照抄同文件既有的 shutdown 硬超时
#: (``self._loop.run(self._shutdown_async(), timeout=10.0)``)—— 同一量纲。
#: ⭐ 关键不是数字,而是它**有界**:H⑤ 把「第二个调用者快速失败」换成了
#: 「加入等待」,若不设上限,那次改动就把 fast-fail 变成了永久挂起。
_SHUTDOWN_JOIN_TIMEOUT_SECONDS = 10.0


class LSPService:
    """One profile's LSP service.

    Created once per profile via :meth:`create_from_config`; the
    :func:`agent.lsp.get_service` accessor manages profile-local reuse.
    Most callers should use that accessor rather than constructing
    :class:`LSPService` directly.
    """

    # ------------------------------------------------------------------
    # construction + factory
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        enabled: bool,
        wait_mode: str,
        wait_timeout: float,
        install_strategy: str,
        binary_overrides: Optional[Dict[str, List[str]]] = None,
        env_overrides: Optional[Dict[str, Dict[str, str]]] = None,
        init_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        disabled_servers: Optional[List[str]] = None,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._enabled = enabled
        self._wait_mode = wait_mode if wait_mode in {"document", "full"} else "document"
        self._wait_timeout = wait_timeout
        self._install_strategy = install_strategy
        self._binary_overrides = binary_overrides or {}
        self._env_overrides = env_overrides or {}
        self._init_overrides = init_overrides or {}
        self._disabled_servers = set(disabled_servers or [])
        self._idle_timeout = idle_timeout

        self._loop = _BackgroundLoop()
        if self._enabled:
            self._loop.start()

        # Per-(server_id, workspace_root) state
        self._clients: Dict[Tuple[str, str], LSPClient] = {}
        self._broken: set = set()
        self._spawning: Dict[Tuple[str, str], asyncio.Future] = {}
        self._last_used: Dict[Tuple[str, str], float] = {}
        self._retiring_clients: Dict[Tuple[str, str], LSPClient] = {}
        self._cleanup_retry_clients: Dict[Tuple[str, str], LSPClient] = {}
        self._cleanup_tasks: Dict[Tuple[str, str], asyncio.Task] = {}
        self._shutdown_in_progress = False
        #: 复用中的 shutdown task —— 见 ``_shutdown_async``。
        self._shutdown_task = None
        self._state_lock = threading.Lock()
        self._idle_reaper_task: Optional[asyncio.Task] = None

        # Delta baseline: file path → snapshot of diagnostics taken
        # immediately before a write.  ``get_diagnostics_sync`` filters
        # out anything in the baseline so the agent only sees errors
        # introduced by the current edit.
        self._delta_baseline: Dict[str, List[Dict[str, Any]]] = {}

        if self._enabled and self._idle_timeout > 0:
            self._loop.run(self._start_idle_reaper(), timeout=2.0)

    @classmethod
    def create_from_config(cls) -> Optional["LSPService"]:
        """Build a service from ``hermes_cli.config`` settings.

        Returns ``None`` if the config can't be loaded.  The service
        itself returns ``is_active()`` False when LSP is disabled.
        """
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP config load failed: %s", e)
            return None

        lsp_cfg = (cfg.get("lsp") or {}) if isinstance(cfg, dict) else {}
        if not isinstance(lsp_cfg, dict):
            lsp_cfg = {}

        enabled = bool(lsp_cfg.get("enabled", True))
        wait_mode = lsp_cfg.get("wait_mode", "document")
        wait_timeout = float(lsp_cfg.get("wait_timeout", DIAGNOSTICS_DOCUMENT_WAIT))
        install_strategy = lsp_cfg.get("install_strategy", "auto")
        try:
            idle_timeout = float(lsp_cfg.get("idle_timeout", DEFAULT_IDLE_TIMEOUT))
        except (TypeError, ValueError):
            idle_timeout = DEFAULT_IDLE_TIMEOUT
        if 0 < idle_timeout < MIN_IDLE_TIMEOUT:
            # A timeout below the per-operation wait budget could reap a
            # client mid-flight; the resulting outer timeout would then
            # mark the (server, workspace) pair broken for the process
            # lifetime.  Clamp to a safe floor (0 still disables).
            idle_timeout = MIN_IDLE_TIMEOUT
        servers_cfg = lsp_cfg.get("servers") or {}
        disabled = []
        binary_overrides: Dict[str, List[str]] = {}
        env_overrides: Dict[str, Dict[str, str]] = {}
        init_overrides: Dict[str, Dict[str, Any]] = {}
        if isinstance(servers_cfg, dict):
            for name, sub in servers_cfg.items():
                if not isinstance(sub, dict):
                    continue
                if sub.get("disabled"):
                    disabled.append(name)
                cmd = sub.get("command")
                if isinstance(cmd, list) and cmd:
                    binary_overrides[name] = cmd
                env = sub.get("env")
                if isinstance(env, dict):
                    env_overrides[name] = {k: str(v) for k, v in env.items()}
                init = sub.get("initialization_options")
                if isinstance(init, dict):
                    init_overrides[name] = init

        return cls(
            enabled=enabled,
            wait_mode=wait_mode,
            wait_timeout=wait_timeout,
            install_strategy=install_strategy,
            binary_overrides=binary_overrides,
            env_overrides=env_overrides,
            init_overrides=init_overrides,
            disabled_servers=disabled,
            idle_timeout=idle_timeout,
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True iff this service should be consulted at all."""
        return self._enabled

    def enabled_for(self, file_path: str) -> bool:
        """Return True iff LSP should run for this specific file.

        Gates on workspace detection (file or cwd inside a git worktree),
        on whether any registered server matches the extension, and
        on whether the (server_id, workspace_root) pair is in the
        broken-set from a previous spawn failure.

        Files in already-broken pairs return False so the file_operations
        layer skips the LSP path entirely — no spawn attempts, no
        timeout cost — until the service is restarted (``hermes lsp
        restart``) or the process exits.
        """
        if not self._enabled:
            return False
        srv = find_server_for_file(file_path)
        if srv is None or srv.server_id in self._disabled_servers:
            return False
        ws_root, gated_in = resolve_workspace_for_file(file_path)
        if not (ws_root and gated_in):
            return False
        # Broken-set short-circuit.  Use the per-server root if we can
        # compute one cheaply; otherwise fall back to the workspace
        # root as the broken key (which is what _get_or_spawn would
        # have used anyway when it failed).
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        if (srv.server_id, per_server_root) in self._broken:
            return False
        return True

    def snapshot_baseline(self, file_path: str) -> None:
        """Snapshot current diagnostics for ``file_path`` as the delta baseline.

        Called BEFORE a write so the next ``get_diagnostics_sync()``
        can filter out pre-existing errors.  Best-effort — failures
        are silently swallowed so a flaky server can't break a write.

        Outer timeouts (e.g. server hangs during initialize) mark the
        (server_id, workspace_root) pair as broken so subsequent edits
        skip it instantly instead of re-paying the timeout cost.
        """
        if not self.enabled_for(file_path):
            return
        try:
            # Outer join budget must exceed the inner wait budget or a
            # slow-but-alive server gets falsely marked broken.
            t = max(8.0, self._wait_timeout + 3.0)
            diags = self._loop.run(self._snapshot_async(file_path), timeout=t)
            self._delta_baseline[os.path.abspath(file_path)] = diags or []
        except Exception as e:  # noqa: BLE001
            logger.debug("baseline snapshot failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            self._delta_baseline[os.path.abspath(file_path)] = []

    def get_diagnostics_sync(
        self,
        file_path: str,
        *,
        delta: bool = True,
        timeout: Optional[float] = None,
        line_shift: Optional[Callable[[int], Optional[int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronously open ``file_path`` in the right server, wait for
        diagnostics, return them.

        If ``delta`` is True (default), the result is filtered against
        any baseline previously captured via :meth:`snapshot_baseline`.
        Diagnostics present in the baseline are removed so the caller
        only sees errors introduced by the current edit.

        When ``line_shift`` is provided, baseline diagnostics are
        remapped through it before the set-difference.  This handles
        the case where the edit deleted or inserted lines, causing
        pre-existing diagnostics below the edit point to surface at
        different line numbers in the post-edit snapshot — without
        the shift, they'd all look "introduced by this edit".  Pass
        a callable built by
        :func:`agent.lsp.range_shift.build_line_shift` (pre_text,
        post_text).  Omit when pre/post content isn't available;
        the unshifted comparison still catches diagnostics that
        didn't move.

        Returns an empty list when LSP is disabled, when no workspace
        can be detected, when no server matches, or when the server
        can't be spawned.  Never raises.
        """
        if not self.enabled_for(file_path):
            return []

        # Resolve server_id eagerly so we can emit structured logs even
        # when the request errors out below.
        srv = find_server_for_file(file_path)
        server_id = srv.server_id if srv else "?"

        try:
            t = timeout if timeout is not None else self._wait_timeout + 2.0
            diags = self._loop.run(self._open_and_wait_async(file_path), timeout=t)
        except asyncio.TimeoutError as e:
            eventlog.log_timeout(server_id, file_path)
            logger.debug("LSP diagnostics timeout for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []
        except Exception as e:  # noqa: BLE001
            eventlog.log_server_error(server_id, file_path, e)
            logger.debug("LSP diagnostics fetch failed for %s: %s", file_path, e)
            self._mark_broken_for_file(file_path, e)
            return []

        if diags is None:
            # The server is alive but never produced diagnostics for the
            # post-edit content within the wait budget (common for
            # tsserver on large projects).  Report "no data" rather than
            # whatever stale state is in the stores — surfacing the
            # previous edit's errors as if they were current is the
            # ghost-diagnostics bug.  The server is NOT marked broken:
            # slow is not dead, and the next edit may well succeed.
            eventlog.log_timeout(server_id, file_path, kind="fresh diagnostics")
            return []

        abs_path = os.path.abspath(file_path)
        if delta:
            baseline = self._delta_baseline.get(abs_path) or []
            if baseline:
                if line_shift is not None:
                    # Remap baseline diagnostics into post-edit
                    # coordinates so shifted-but-otherwise-identical
                    # entries hash equal under _diag_key.  Entries
                    # that mapped into a deleted region drop out
                    # silently — they no longer apply.
                    from agent.lsp.range_shift import shift_baseline
                    baseline = shift_baseline(baseline, line_shift)
                seen = {_diag_key(d) for d in baseline}
                diags = [d for d in diags if _diag_key(d) not in seen]
            # Roll baseline forward — next call returns deltas relative
            # to the just-emitted state, mirroring claude-code's
            # diagnosticTracking.
            try:
                fresh = self._loop.run(self._current_diags_async(file_path), timeout=2.0) or []
            except Exception:  # noqa: BLE001
                fresh = []
            if fresh:
                self._delta_baseline[abs_path] = fresh

        if diags:
            eventlog.log_diagnostics(server_id, file_path, len(diags))
        else:
            eventlog.log_clean(server_id, file_path)
        return diags

    def _mark_broken_for_file(self, file_path: str, exc: BaseException) -> None:
        """Mark the (server_id, workspace_root) pair as broken so subsequent
        edits skip it instantly instead of re-paying timeout cost.

        Called when the outer ``_loop.run`` timeout cancels an in-flight
        spawn/initialize that the inner ``_get_or_spawn`` task was still
        holding open.  Without this, every subsequent write would re-enter
        the spawn path and re-pay the full ``snapshot_baseline``
        timeout (8s) until the binary is fixed.

        Also kills any orphan client process that survived the cancelled
        future, and emits a single eventlog WARNING so the user knows
        which server gave up.

        ``exc`` 是外层捕获并用于日志的原始异常。子进程清理失败会显式抛出，
        并保留 owner 供重试。
        """
        srv = find_server_for_file(file_path)
        if srv is None:
            return
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            return
        try:
            per_server_root = srv.resolve_root(file_path, ws_root) or ws_root
        except Exception:  # noqa: BLE001
            per_server_root = ws_root
        key = (srv.server_id, per_server_root)
        already_broken = key in self._broken
        self._broken.add(key)
        if not already_broken:
            eventlog.log_spawn_failed(srv.server_id, per_server_root, exc)

        # 先标记 retiring，shutdown 成功后再按精确对象 CAS 删除。
        with self._state_lock:
            client = self._clients.get(key)
            cleanup_in_flight = (
                client is not None
                and self._retiring_clients.get(key) is client
            )
            if client is not None and not cleanup_in_flight:
                self._retiring_clients[key] = client
        if cleanup_in_flight:
            raise RuntimeError(
                f"LSP broken-client shutdown already in progress for {srv.server_id}"
            )
        if client is not None:
            try:
                self._loop.run(
                    self._cleanup_client_with_barrier(key, client), timeout=1.0
                )
            except Exception as shutdown_exc:  # noqa: BLE001
                with self._state_lock:
                    cleanup_task = self._cleanup_tasks.get(key)
                    cleanup_in_flight = (
                        cleanup_task is not None and not cleanup_task.done()
                    )
                    if not cleanup_in_flight and self._clients.get(key) is client:
                        self._cleanup_retry_clients[key] = client
                    if (
                        not cleanup_in_flight
                        and self._retiring_clients.get(key) is client
                    ):
                        self._retiring_clients.pop(key, None)
                raise RuntimeError(
                    f"LSP broken-client shutdown failed for {srv.server_id}"
                ) from shutdown_exc
            with self._state_lock:
                if self._clients.get(key) is client:
                    self._clients.pop(key, None)
                    self._last_used.pop(key, None)
                    self._cleanup_retry_clients.pop(key, None)
                if self._retiring_clients.get(key) is client:
                    self._retiring_clients.pop(key, None)

    @staticmethod
    async def _shutdown_client_for_retry(client: LSPClient) -> None:
        """严格关闭 client；失败时恢复仍活的 process handle 供下次重试。"""
        process = getattr(client, "_proc", None)

        def _restore_live_process() -> bool:
            if process is None or getattr(process, "returncode", None) is not None:
                return False
            current = getattr(client, "_proc", None)
            if current is not None and current is not process:
                raise RuntimeError("LSP process ownership changed during shutdown")
            client._proc = process
            client._stopping = False
            return True

        try:
            await client.shutdown()
        except BaseException:
            _restore_live_process()
            raise
        if _restore_live_process():
            raise RuntimeError("LSP client shutdown returned while process is alive")

    async def _cleanup_client_with_barrier(
        self, key: Tuple[str, str], client: LSPClient
    ) -> None:
        """共享一次 cleanup；重复取消只延迟调用方，不释放 owner fence。"""
        with self._state_lock:
            task = self._cleanup_tasks.get(key)
        if task is None:
            task = asyncio.create_task(self._shutdown_client_for_retry(client))
            with self._state_lock:
                existing = self._cleanup_tasks.get(key)
                if existing is None:
                    self._cleanup_tasks[key] = task
                else:
                    task.cancel()
                    task = existing

        cancelled = None
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError as exc:
                if cancelled is None:
                    cancelled = exc
                continue

        failure = None
        if task.cancelled():
            failure = asyncio.CancelledError()
        else:
            try:
                failure = task.exception()
            except BaseException as exc:  # noqa: BLE001
                failure = exc
        with self._state_lock:
            if failure is not None:
                if self._clients.get(key) is client:
                    self._cleanup_retry_clients[key] = client
            elif self._clients.get(key) is client:
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
                self._cleanup_retry_clients.pop(key, None)
            if self._cleanup_tasks.get(key) is task:
                self._cleanup_tasks.pop(key, None)
            if self._retiring_clients.get(key) is client:
                self._retiring_clients.pop(key, None)
        if failure is not None:
            raise failure
        if cancelled is not None:
            raise cancelled

    @staticmethod
    def _install_process_cleanup_owner_fence(client: LSPClient) -> None:
        """底层 cleanup 先清空句柄再失败时，恢复仍存活的 exact process。"""
        cleanup = getattr(client, "_cleanup_process", None)
        if not callable(cleanup) or getattr(
            client, "_hermes_cleanup_owner_fenced", False
        ):
            return

        async def _cleanup_with_owner_restore() -> None:
            process = getattr(client, "_proc", None)
            try:
                await cleanup()
            except BaseException:
                if (
                    process is not None
                    and getattr(process, "returncode", None) is None
                    and getattr(client, "_proc", None) is None
                ):
                    client._proc = process
                    client._stopping = False
                raise

        client._cleanup_process = _cleanup_with_owner_restore
        client._hermes_cleanup_owner_fenced = True

    def shutdown(self, *, raise_on_error: bool = False) -> None:
        """Tear down all clients and stop the background loop."""
        if not self._enabled:
            return
        try:
            self._loop.run(self._shutdown_async(), timeout=10.0)
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)
            if raise_on_error:
                raise
            return
        self._loop.stop()
        clear_cache()

    # ------------------------------------------------------------------
    # async internals
    # ------------------------------------------------------------------

    async def _snapshot_async(self, file_path: str) -> List[Dict[str, Any]]:
        client = await self._get_or_spawn(file_path)
        if client is None:
            return []
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            fresh = await client.wait_for_diagnostics(file_path, version, mode=self._wait_mode)
        except Exception as e:  # noqa: BLE001
            logger.debug("snapshot open/wait failed: %s", e)
            return []
        self._touch(client)
        if not fresh:
            # No fresh data for the pre-edit content — an empty baseline
            # is safe: worst case the delta filter removes less, never
            # more.  Never seed the baseline from stale stores.
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _open_and_wait_async(self, file_path: str) -> Optional[List[Dict[str, Any]]]:
        """Open + wait for FRESH diagnostics.

        Returns the fresh diagnostic list, or ``None`` when the server
        never produced post-change data within the wait budget.  The
        distinction matters: ``[]`` means "server checked the new
        content, it's clean", ``None`` means "no verdict" — the caller
        must not substitute stale data for either.
        """
        client = await self._get_or_spawn(file_path)
        if client is None:
            return None
        try:
            version = await client.open_file(file_path, language_id=language_id_for(file_path))
            await client.save_file(file_path)
            fresh = await client.wait_for_diagnostics(
                file_path, version, mode=self._wait_mode, timeout=self._wait_timeout
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("open/wait failed for %s: %s", file_path, e)
            return None
        self._touch(client)
        if not fresh:
            return None
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _current_diags_async(self, file_path: str) -> List[Dict[str, Any]]:
        ws, gated = resolve_workspace_for_file(file_path)
        srv = find_server_for_file(file_path)
        if not (ws and gated and srv):
            return []
        with self._state_lock:
            client = self._clients.get((srv.server_id, ws))
        if client is None:
            return []
        return list(client.diagnostics_for(file_path, fresh_only=True))

    async def _get_or_spawn(self, file_path: str) -> Optional[LSPClient]:
        srv = find_server_for_file(file_path)
        if srv is None:
            return None
        if srv.server_id in self._disabled_servers:
            eventlog.log_disabled(srv.server_id, file_path, "disabled in config")
            return None
        ws_root, gated = resolve_workspace_for_file(file_path)
        if not (ws_root and gated):
            eventlog.log_no_project_root(srv.server_id, file_path)
            return None
        per_server_root = srv.resolve_root(file_path, ws_root)
        if per_server_root is None:
            eventlog.log_disabled(
                srv.server_id, file_path, "exclude marker hit (server gated off)"
            )
            return None  # exclude marker hit, server gated off

        key = (srv.server_id, per_server_root)
        if key in self._broken:
            return None
        with self._state_lock:
            if self._shutdown_in_progress:
                return None
            retry_client = self._cleanup_retry_clients.get(key)
            if (
                key in self._retiring_clients
                or (
                    retry_client is not None
                    and retry_client is self._clients.get(key)
                )
            ):
                return None
            client = self._clients.get(key)
            if client is not None and client.is_running:
                self._last_used[key] = time.time()
                eventlog.log_active(srv.server_id, per_server_root)
                return client
            spawning = self._spawning.get(key)
        if spawning is not None:
            try:
                return await asyncio.shield(spawning)
            except Exception:  # noqa: BLE001
                return None

        # Begin spawn
        loop = asyncio.get_running_loop()
        spawn_future: asyncio.Future = loop.create_future()
        with self._state_lock:
            if self._shutdown_in_progress:
                return None
            existing_spawn = self._spawning.get(key)
            if existing_spawn is not None:
                spawn_future = existing_spawn
            else:
                self._spawning[key] = spawn_future
        if existing_spawn is not None:
            try:
                return await asyncio.shield(existing_spawn)
            except Exception:  # noqa: BLE001
                return None
        try:
            ctx = ServerContext(
                workspace_root=per_server_root,
                install_strategy=self._install_strategy,
                binary_overrides=self._binary_overrides,
                env_overrides=self._env_overrides,
                init_overrides=self._init_overrides,
            )
            spec = srv.build_spawn(per_server_root, ctx)
            if spec is None:
                # ``build_spawn`` returns None when the binary can't be
                # located (auto-install disabled, manual-only server,
                # or install attempt failed).  Surface this once via
                # the structured logger so the user can act on it.
                eventlog.log_server_unavailable(srv.server_id, srv.server_id)
                self._broken.add(key)
                spawn_future.set_result(None)
                return None
            client = LSPClient(
                server_id=srv.server_id,
                workspace_root=spec.workspace_root,
                command=spec.command,
                env=spec.env,
                cwd=spec.cwd,
                initialization_options=spec.initialization_options,
                seed_diagnostics_on_first_push=spec.seed_diagnostics_on_first_push or srv.seed_first_push,
            )
            with self._state_lock:
                if self._shutdown_in_progress:
                    spawn_future.set_result(None)
                    return None
                self._clients[key] = client
            self._install_process_cleanup_owner_fence(client)
            try:
                await client.start()
            except asyncio.CancelledError:
                with self._state_lock:
                    cleanup_owned = (
                        self._retiring_clients.get(key) is client
                        or self._shutdown_in_progress
                    )
                    if not cleanup_owned and self._clients.get(key) is client:
                        self._retiring_clients[key] = client
                if cleanup_owned:
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    raise
                try:
                    await self._cleanup_client_with_barrier(key, client)
                except asyncio.CancelledError:
                    with self._state_lock:
                        cleanup_failed = self._cleanup_retry_clients.get(key) is client
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    if cleanup_failed:
                        raise RuntimeError(
                            "LSP cancelled spawn cleanup failed"
                        )
                    raise
                except BaseException as cleanup_exc:
                    with self._state_lock:
                        if self._clients.get(key) is client:
                            self._cleanup_retry_clients[key] = client
                        if self._retiring_clients.get(key) is client:
                            self._retiring_clients.pop(key, None)
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    raise RuntimeError(
                        "LSP cancelled spawn cleanup failed"
                    ) from cleanup_exc
                with self._state_lock:
                    if self._clients.get(key) is client:
                        self._clients.pop(key, None)
                        self._last_used.pop(key, None)
                        self._cleanup_retry_clients.pop(key, None)
                    if self._retiring_clients.get(key) is client:
                        self._retiring_clients.pop(key, None)
                if not spawn_future.done():
                    spawn_future.set_result(None)
                raise
            except Exception as e:  # noqa: BLE001
                eventlog.log_spawn_failed(srv.server_id, per_server_root, e)
                self._broken.add(key)
                with self._state_lock:
                    cleanup_owned = (
                        self._retiring_clients.get(key) is client
                        or self._shutdown_in_progress
                    )
                    if not cleanup_owned and self._clients.get(key) is client:
                        self._retiring_clients[key] = client
                if cleanup_owned:
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    return None
                try:
                    await self._cleanup_client_with_barrier(key, client)
                except BaseException as cleanup_exc:
                    with self._state_lock:
                        if self._clients.get(key) is client:
                            self._cleanup_retry_clients[key] = client
                        if self._retiring_clients.get(key) is client:
                            self._retiring_clients.pop(key, None)
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    raise RuntimeError("LSP spawn cleanup failed") from cleanup_exc
                with self._state_lock:
                    if self._clients.get(key) is client:
                        self._clients.pop(key, None)
                        self._last_used.pop(key, None)
                        self._cleanup_retry_clients.pop(key, None)
                    if self._retiring_clients.get(key) is client:
                        self._retiring_clients.pop(key, None)
                if not spawn_future.done():
                    spawn_future.set_result(None)
                return None
            with self._state_lock:
                owner_current = (
                    not self._shutdown_in_progress
                    and self._clients.get(key) is client
                )
                if owner_current:
                    self._last_used[key] = time.time()
            if not owner_current:
                try:
                    await self._shutdown_client_for_retry(client)
                except BaseException:
                    with self._state_lock:
                        if self._clients.get(key) is client:
                            self._cleanup_retry_clients[key] = client
                    if not spawn_future.done():
                        spawn_future.set_result(None)
                    raise
                with self._state_lock:
                    if self._clients.get(key) is client:
                        self._clients.pop(key, None)
                        self._last_used.pop(key, None)
                        self._cleanup_retry_clients.pop(key, None)
                if not spawn_future.done():
                    spawn_future.set_result(None)
                return None
            eventlog.log_active(srv.server_id, per_server_root)
            if not spawn_future.done():
                spawn_future.set_result(client)
            return client
        finally:
            with self._state_lock:
                if self._spawning.get(key) is spawn_future:
                    self._spawning.pop(key, None)

    async def _start_idle_reaper(self) -> None:
        self._idle_reaper_task = asyncio.create_task(self._idle_reaper_loop())

    def _touch(self, client: LSPClient) -> None:
        """Refresh the last-used timestamp for a client we just used.

        Guarded on membership so a reaped-mid-operation client can't
        resurrect an orphan ``_last_used`` entry after the reaper popped
        the key.  All writers and the reaper run on the background loop
        thread; the lock keeps this consistent with the reader anyway.
        """
        key = (client.server_id, client.workspace_root)
        with self._state_lock:
            if key in self._clients:
                self._last_used[key] = time.time()

    async def _idle_reaper_loop(self) -> None:
        interval = min(60.0, self._idle_timeout)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._reap_idle_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # A transient sweep error must not kill the reaper —
                # otherwise one bad shutdown permanently re-opens the
                # unbounded-accumulation leak this loop exists to fix.
                logger.warning("LSP idle reaper sweep error: %s", e, exc_info=True)

    async def _reap_idle_once(self) -> None:
        cutoff = time.time() - self._idle_timeout
        with self._state_lock:
            snapshot = [
                (key, client)
                for key, client in self._clients.items()
                if key not in self._retiring_clients
                and key not in self._spawning
                and self._last_used.get(key, 0) < cutoff
            ]
            for key, client in snapshot:
                self._retiring_clients[key] = client
        if snapshot:
            try:
                results = await asyncio.gather(
                    *(
                        self._shutdown_client_for_retry(client)
                        for _, client in snapshot
                    ),
                    return_exceptions=True,
                )
            except BaseException:
                with self._state_lock:
                    for key, client in snapshot:
                        if self._clients.get(key) is client:
                            self._cleanup_retry_clients[key] = client
                        if self._retiring_clients.get(key) is client:
                            self._retiring_clients.pop(key, None)
                raise
            released = []
            failures = []
            with self._state_lock:
                for (key, client), result in zip(snapshot, results):
                    if isinstance(result, BaseException):
                        failures.append(result)
                        if self._clients.get(key) is client:
                            self._cleanup_retry_clients[key] = client
                    elif self._clients.get(key) is client:
                        self._clients.pop(key, None)
                        self._last_used.pop(key, None)
                        self._cleanup_retry_clients.pop(key, None)
                        released.append(client)
                    if self._retiring_clients.get(key) is client:
                        self._retiring_clients.pop(key, None)
            if released:
                eventlog.log_reaped(
                    [(c.server_id, c.workspace_root) for c in released],
                    self._idle_timeout,
                )
            if failures:
                raise RuntimeError(
                    f"LSP idle reaper failed for {len(failures)} client(s)"
                ) from failures[0]

    async def _shutdown_async(self) -> None:
        """严格等待 shutdown owner；重复取消不能提前释放 fence。"""
        # 🔴 **复用同一个 task,⛔ 不许每次调用都新建。**
        # 外层是 10 秒硬超时(``self._loop.run(self._shutdown_async(), timeout=10.0)``):
        #   ① 关闭超过 10 秒 ⇒ 外层 future 被取消,调用方拿到超时;
        #   ② 但 worker 是 **shield 住的**,它继续跑到成功,而成功路径
        #      **不重置** ``_shutdown_in_progress``(只有各异常路径重置);
        #   ③ 下次 ``shutdown_service(raise_on_error=True)`` 新建第二个 worker,
        #      一进 owner 就撞 ``RuntimeError("LSP shutdown already in progress")``。
        # ⇒ **一次慢关闭 ⇒ 该 profile 后续 unload / reload 永久失败。**
        # ⭐ 典型「一次性故障变永久降级」:闩上了,没有任何路径解得开。
        # ⇒ 第二次调用**加入在飞的那一趟**。⛔ 不去动 ``_shutdown_in_progress``
        # 的置位/归零 —— 那个闩在 owner 内部自洽,缺的是外面没人复用那一趟。
        worker = self._shutdown_task
        if worker is None or worker.done():
            worker = asyncio.create_task(self._shutdown_async_owned())
            self._shutdown_task = worker
            worker.add_done_callback(self._clear_shutdown_task)
        # 🔴 **本 PR 让这条无界等待【变得可达】⇒ 它不再是存量问题。**
        #
        # 改之前:第二个并发调用者会**自己新建**一个 worker,那个 worker 一进
        # ``_shutdown_async_owned`` 就撞
        # ``RuntimeError("LSP shutdown already in progress")`` ⇒ **快速失败**。
        # 改之后(H⑤ 复用唯一 task):第二个调用者**加入在飞的那一趟**并
        # ``await asyncio.shield(worker)`` ⇒ owner 卡住时,**原本快速失败的调用
        # 现在会挂住**。⭐ 我把 fast-fail 换成了 join,而那个 join 是无界的。
        #
        # ⇒ 与 ``_cleanup_unpublished_adapter`` 的重入路径同解:等待有界,
        # 超时**显式失败**;⛔ 不取消 owner(第二个 waiter 不拥有它),
        # ⛔ 不清 ``_shutdown_task``(清了下次又会新建、重新撞闩)。
        deadline = _SHUTDOWN_JOIN_TIMEOUT_SECONDS
        cancelled = None
        while True:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(worker), timeout=deadline
                )
            except asyncio.TimeoutError:
                raise TimeoutError(
                    f"LSP shutdown still running after {deadline:.1f}s"
                ) from None
            except asyncio.CancelledError as exc:
                if worker.cancelled():
                    raise
                if cancelled is None:
                    cancelled = exc
            else:
                if cancelled is not None:
                    raise cancelled
                return result

    def _clear_shutdown_task(self, task) -> None:
        """任务收尾后解除复用引用 —— **但必须先把闩一起放掉**。

        🔴 上一版只清引用。而 ``_shutdown_async_owned`` 的**成功路径刻意不重置**
        ``_shutdown_in_progress``(语义是「已经关掉了」)⇒ 一旦首次 shutdown 超过
        外层 10 秒期限、shield 住的 owner 随后**成功**结束:
          · 闩留在 True
          · 引用被这里清掉
        ⇒ 下一次 profile unload 新建 owner,一进去就**稳定**撞
        ``RuntimeError("LSP shutdown already in progress")`` ⇒ **该 profile 后续
        卸载/重载永久失败**。⭐ 半条链:我复用了 task,却让它的可观察结果先蒸发。

        ⇒ **成功完成时原子地把闩一起放掉**,再清引用。
        ⛔ 失败/取消时不动闩 —— 那些路径 owner 自己已经重置过了,
        再动一次会把「正在跑的另一趟」误判成空闲。
        """
        with self._state_lock:
            if self._shutdown_task is not task:
                return
            self._shutdown_task = None
            # 只有**干净完成**才放闩:异常与取消由 owner 自己的各分支负责。
            if not task.cancelled() and task.exception() is None:
                self._shutdown_in_progress = False

    async def _shutdown_async_owned(self) -> None:
        with self._state_lock:
            if self._shutdown_in_progress:
                raise RuntimeError("LSP shutdown already in progress")
            self._shutdown_in_progress = True
        reaper = self._idle_reaper_task
        self._idle_reaper_task = None
        try:
            if reaper is not None:
                reaper.cancel()
                await asyncio.gather(reaper, return_exceptions=True)
        except BaseException:
            with self._state_lock:
                if reaper is not None and not reaper.done():
                    self._idle_reaper_task = reaper
                self._shutdown_in_progress = False
            raise
        with self._state_lock:
            spawning = list(self._spawning.values())
        if spawning:
            try:
                await asyncio.gather(
                    *(asyncio.shield(future) for future in spawning),
                    return_exceptions=True,
                )
            except BaseException:
                with self._state_lock:
                    self._shutdown_in_progress = False
                raise
        with self._state_lock:
            pending = [
                (key, client)
                for key, client in self._clients.items()
                if self._retiring_clients.get(key) is client
            ]
            snapshot = [
                (key, client)
                for key, client in self._clients.items()
                if self._retiring_clients.get(key) is not client
            ]
            for key, client in snapshot:
                self._retiring_clients[key] = client
        if pending:
            with self._state_lock:
                for key, client in snapshot:
                    if self._retiring_clients.get(key) is client:
                        self._retiring_clients.pop(key, None)
                self._shutdown_in_progress = False
            raise RuntimeError(
                f"LSP shutdown already in progress for {len(pending)} client(s)"
            )
        try:
            results = await asyncio.gather(
                *(
                    self._shutdown_client_for_retry(client)
                    for _, client in snapshot
                ),
                return_exceptions=True,
            )
        except BaseException:
            with self._state_lock:
                for key, client in snapshot:
                    if self._clients.get(key) is client:
                        self._cleanup_retry_clients[key] = client
                    if self._retiring_clients.get(key) is client:
                        self._retiring_clients.pop(key, None)
                self._shutdown_in_progress = False
            raise
        failures = []
        with self._state_lock:
            for (key, client), result in zip(snapshot, results):
                if isinstance(result, BaseException):
                    failures.append(result)
                    if self._clients.get(key) is client:
                        self._cleanup_retry_clients[key] = client
                elif self._clients.get(key) is client:
                    self._clients.pop(key, None)
                    self._last_used.pop(key, None)
                    self._cleanup_retry_clients.pop(key, None)
                if self._retiring_clients.get(key) is client:
                    self._retiring_clients.pop(key, None)
            if failures:
                self._shutdown_in_progress = False
        if failures:
            raise RuntimeError(
                f"LSP shutdown failed for {len(failures)} client(s)"
            ) from failures[0]
        with self._state_lock:
            self._broken.clear()

    # ------------------------------------------------------------------
    # status / introspection (used by ``hermes lsp status``)
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the service for the CLI status command."""
        with self._state_lock:
            clients = [
                {
                    "server_id": k[0],
                    "workspace_root": k[1],
                    "state": c.state,
                    "running": c.is_running,
                }
                for k, c in self._clients.items()
            ]
            broken = list(self._broken)
        return {
            "enabled": self._enabled,
            "wait_mode": self._wait_mode,
            "wait_timeout": self._wait_timeout,
            "install_strategy": self._install_strategy,
            "clients": clients,
            "broken": broken,
            "disabled_servers": sorted(self._disabled_servers),
        }


def _diag_key(d: Dict[str, Any]) -> str:
    """Content equality key used for cross-edit delta filtering.

    Includes the diagnostic's position range — when used together
    with :func:`agent.lsp.range_shift.shift_baseline`, the baseline
    is line-shifted into post-edit coordinates BEFORE this key is
    computed, so identical-but-shifted diagnostics hash equal.  Two
    genuinely distinct diagnostics at different lines (e.g. the same
    error class introduced at a second site) hash differently and
    are surfaced as new.

    Mirrors :func:`agent.lsp.client._diagnostic_key`; intentionally
    identical so the two layers agree on diagnostic identity.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = ["LSPService"]
