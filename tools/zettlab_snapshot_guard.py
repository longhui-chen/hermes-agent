"""Zettlab Agent 文件变更保护快照的 pre-mutation hook（fork-only 模块）。

在 Agent 对用户文件做破坏性操作**之前**，先让设备端的 local-server 为受影响的
主文件夹建一张 btrfs 保护快照；拿不到快照就不执行操作（fail-closed）。这样用户
永远有一个「Agent 动手之前」的恢复点。

设计要点：

* **调用方不做语义判定**。哪些路径算破坏性、属于哪个受保护目录、要不要拍快照，
  全部由 local-server 按真实文件状态复核。这里只负责把涉及的绝对路径报上去。
* **一轮任务 × 一个主文件夹最多一张快照**，同轮复用。轮标识取 tool 调度层传下来
  的 ``turn_id``（chat 与 cron 都有值），服务端按 agent × turn × target 幂等。
* **轮状态按 turn_id 键控**（上限 8 个并发轮）：同一 gateway 进程里多轮并发时
  互不覆盖，finish 只收自己的轮；溢出被逐出的轮由服务端 pin TTL 自愈。
* **嵌套 dispatch 的轮继承**：execute_code 沙箱 RPC / MCP bridge 二次进入
  ``handle_function_call`` 时不带 ``turn_id``，靠 task_id → turn_id 映射回落到
  外层轮；映射在每次带 turn 的 dispatch 时登记。
* **同轮自建文件豁免**：本轮由 Agent 自己创建的文件，其后续修改不再触发快照——
  它的原始状态就是「不存在」，恢复手段是删掉。集合有上限，溢出即回退到正常拍
  快照。
* **老设备降级**：local-server 上没有这个接口（404）时放行并记一次日志，不假装
  已经建过快照。非设备环境（无回调地址 / token）完全不介入。
* 通道是 loopback + registry 注入的 per-agent action token，URL 从既有的回调地址
  派生，且**只信任 loopback origin**——非 loopback 的回调地址一律不发 token。

跨仓合同见 zettlab-local-server ``docs/agent-file-protection-internal-api.md``。
"""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

# local-server 注入 hermes 子进程的回调地址，用来反推 origin。故意不引入独立的
# ZETTLAB_LOCAL_SERVER_URL：从既有回调派生可以保证 action token 只发往 loopback。
_ORIGIN_ANCHOR_ENVS = ("ZET_CHAT_APPEND_URL", "ZETTLAB_AGENT_SHARE_ACTION_URL")
_ACTION_TOKEN_ENV = "ZETTLAB_AGENT_ACTION_TOKEN"

_ENSURE_PATH = "/api/v1/internal/snapshot/agent-protection/ensure"
_FINISH_PATH = "/api/v1/internal/snapshot/agent-protection/finish"

# 服务端 ensure 的同步上限是 30s，客户端留一点余量再放弃。
_ENSURE_TIMEOUT = 35.0
_FINISH_TIMEOUT = 10.0

# 响应体上限：正常载荷只有几个 ID 和状态字符串。
_MAX_RESPONSE_BYTES = 256 * 1024

# 同轮自建文件的追踪上限（PRD §7.3）。溢出后放弃豁免、回退为正常触发快照——
# 宁可多拍，也不让这个集合无界增长。
_MAX_CREATED_TRACKED = 512

# 同时追踪的轮状态上限。溢出逐出最老的轮：它的 finish 变成 no-op，pin 由服务端
# TTL + 周期自愈释放——退化方向是「晚一点解 pin」，不是丢保护。
_MAX_TRACKED_TURNS = 8

# task_id → turn_id 映射的上限（嵌套 dispatch 的轮继承用）。
_MAX_TASK_TURNS = 64


# 需要保护的工具。execute_code 沙箱里的 hermes_tools.write_file 也回流到
# handle_function_call 被前三个名字覆盖；project 模式的 execute_code 还能用
# Python open()/Path.write_text() 直接改 session cwd 里的用户文件而不经过任何
# 文件工具，所以 execute_code 本体也要在启动前保护实际 cwd（Codex review P1）。
_FILE_MUTATING_TOOLS = frozenset({"write_file", "patch"})
_GUARDED_TOOLS = _FILE_MUTATING_TOOLS | {"terminal", "execute_code"}

# V4A patch 的 header 提取，与 tools/patch_parser.py 的规则同源：`***` 后空格
# 可选（parser 用 \s*，接受 `***Update File:`），Move 有 src 与 dst 两个端点。
# 规则不一致会让抽不到的路径退回「只保护工作目录」，跨目录的 patch 目标失去
# 恢复点（Codex review P1）。
_V4A_FILE_RE = re.compile(r"^\*\*\*\s*(?:Update|Add|Delete)\s+File:\s*(.+)$", re.MULTILINE)
_V4A_MOVE_RE = re.compile(r"^\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)$", re.MULTILINE)

# terminal 的保护判定是**只读安全清单**而不是破坏性黑名单：`python -c`、
# `git apply`、`tar -xf`、`unzip -o` 这类写文件的命令数不胜数，黑名单永远列不
# 全（Codex review P1）。无法证明只读的命令一律按 cwd 保护——同轮同 target
# 幂等，多判的代价只是每轮多一张快照。
_WRITEISH_SHELL_RE = re.compile(r">|<\(|\$\(|`|\btee\b")
_SHELL_CHAIN_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_READONLY_FIRST_TOKENS = frozenset({
    "ls", "cat", "grep", "rg", "egrep", "fgrep", "find", "head", "tail",
    "less", "more", "wc", "pwd", "echo", "printf", "stat", "file", "which",
    "type", "env", "printenv", "ps", "df", "du", "date", "whoami", "id",
    "uname", "md5sum", "sha1sum", "sha256sum", "sort", "uniq", "cut", "tr",
    "diff", "cmp", "readlink", "basename", "dirname", "hostname", "uptime",
    "free", "tree", "realpath", "test", "true", "false", "sleep",
})
_READONLY_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "remote", "rev-parse",
    "describe", "shortlog", "blame", "ls-files",
})

# 命令文本里的绝对路径 token：cwd 之外的写入目标（rm /home/alice/... 或脚本里
# 的 Path("/home/...").write_text）也要尽力保护（Codex review P1）。这些路径走
# **附加** ensure：范围外（403）只跳过、不阻断——它们是 cwd 保护之外的加餐。
_ABS_PATH_TOKEN_RE = re.compile(r"(?<![\w.-])/(?:[\w.+@%-]+/)*[\w.+@%-]+")
_MAX_ANCILLARY_PATHS = 16


class _TurnState:
    """一轮任务的追踪状态。"""

    def __init__(self, turn_id: str) -> None:
        self.turn_id = turn_id
        self.created: set[str] = set()
        self.created_overflowed = False
        self.ensured = False


_lock = threading.Lock()
# 轮状态按 turn_id 键控（插入序），并发轮互不覆盖（Codex review P1）。
_states: dict[str, _TurnState] = {}
# task_id → turn_id：嵌套 dispatch（execute_code 沙箱 RPC / MCP bridge）不带
# turn_id，凭它们携带的 task_id 回落到外层轮（Codex review P1）。
_task_turns: dict[str, str] = {}
_degraded_logged = False
_nonloopback_logged = False


def _scoped_env(name: str, default: str = "") -> str:
    """读环境变量。multiplex 下必须走 profile scope，否则会串到别的 agent。"""
    try:
        from agent.secret_scope import get_secret

        value = get_secret(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    try:
        from agent.secret_scope import is_multiplex_active

        if is_multiplex_active():
            return default
    except Exception:
        pass
    return os.environ.get(name, default)


def _is_loopback_host(host: Optional[str]) -> bool:
    """只认字面量 loopback IP 与 localhost。

    不能按字符串前缀判断：`127.evil.example` / `127.0.0.1.attacker` 是普通
    DNS hostname，前缀匹配会把 token 发去外部主机（Codex review P1）。
    """
    if not host:
        return False
    h = host.strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _local_server_origin() -> str:
    """从注入的回调地址派生 local-server origin，只信任 loopback。

    action token 和待保护路径会随请求发往这个 origin：如果环境变量被注入 / 残留
    了非 loopback 地址，第一次写文件就会把 token 泄漏出去，所以这里必须显式校验
    host（Codex review P1）。校验不过按「非设备环境」处置。
    """
    global _nonloopback_logged
    for env_key in _ORIGIN_ANCHOR_ENVS:
        raw = _scoped_env(env_key, "").strip()
        if not raw:
            continue
        parts = urlsplit(raw)
        if not parts.scheme or not parts.netloc:
            logger.debug("zettlab snapshot guard: malformed %s: %r", env_key, raw)
            continue
        if parts.scheme not in ("http", "https") or not _is_loopback_host(parts.hostname):
            if not _nonloopback_logged:
                _nonloopback_logged = True
                logger.warning(
                    "zettlab snapshot guard: %s is not a loopback URL; "
                    "refusing to send the action token there",
                    env_key,
                )
            continue
        return urlunsplit((parts.scheme, parts.netloc, "", "", "")).rstrip("/")
    return ""


def _abs_path(path: Any) -> str:
    """把工具参数里的路径归一成绝对路径。服务端只接受绝对路径。"""
    raw = str(path or "").strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    if not os.path.isabs(expanded):
        base = os.getenv("TERMINAL_CWD") or os.getcwd()
        expanded = os.path.join(base, expanded)
    return os.path.normpath(expanded)


def _resolve_write_path(path: Any, task_id: str) -> str:
    """解析 write_file / patch 的目标路径，与文件工具自己的口径对齐。

    实际写入按 task_id 走 ``tools.file_tools._resolve_path_for_task``（会话注册
    的 cwd 优先于进程 env）；guard 若用进程级 cwd 解析相对路径，会给错误目录拍
    快照而真正被写的文件没有恢复点（Codex review P1）。没有 task_id 或解析器不
    可用时退回进程级解析。
    """
    raw = str(path or "").strip()
    if not raw:
        return ""
    if task_id:
        try:
            from tools.file_tools import _resolve_path_for_task

            return os.path.normpath(str(_resolve_path_for_task(raw, task_id)))
        except Exception:
            pass
    return _abs_path(raw)


def _terminal_workdir(arguments: dict[str, Any], task_id: str) -> str:
    """破坏性 shell 命令报工作目录而不是解析命令行里的路径。

    从任意 shell 命令里可靠地抽出被改动的文件是做不到的；而快照本来就是主文件
    夹级的，报 cwd 足以让服务端定位到正确的 snapshot_target 并拍下整个目录。

    cwd 的优先级与 terminal 的 _resolve_command_cwd 同源：显式 workdir 参数 >
    会话自己的 cwd 记录（get_session_cwd，即该会话的 `cd` 状态）> 进程 env。
    每一级都要求目录在 host 上真实存在：Docker backend 挂载 cwd 时会话记录是
    容器内的 `/workspace/...`，原样上报会让快照落在不存在的路径、真正被改的
    host 目录失去恢复点（Codex review P1）——不存在就退回下一级。
    """
    explicit = str(arguments.get("workdir") or "").strip()
    if explicit:
        p = _abs_path(explicit)
        if os.path.isdir(p):
            return p
    if task_id:
        try:
            from tools.terminal_tool import get_session_cwd

            recorded = get_session_cwd(task_id)
            if recorded:
                p = _abs_path(recorded)
                if os.path.isdir(p):
                    return p
        except Exception:
            pass
    return _abs_path(os.getenv("TERMINAL_CWD") or os.getcwd())


def _command_is_probably_readonly(command: str) -> bool:
    """报告一条 shell 命令是否**可证明**只读。证明不了就按需要保护处理。"""
    if not command.strip():
        return True
    if _WRITEISH_SHELL_RE.search(command):
        return False
    for segment in _SHELL_CHAIN_SPLIT_RE.split(command):
        tokens = [t for t in segment.strip().split() if not _ENV_ASSIGN_RE.match(t)]
        if not tokens:
            continue
        head = os.path.basename(tokens[0])
        if head == "git":
            if len(tokens) < 2 or tokens[1] not in _READONLY_GIT_SUBCOMMANDS:
                return False
            continue
        if head not in _READONLY_FIRST_TOKENS:
            return False
    return True


def _ancillary_abs_paths(text: str, primary: list[str]) -> list[str]:
    """从命令 / 脚本文本里抽出**已存在**的绝对路径，作为 cwd 之外的附加保护。"""
    out: list[str] = []
    seen = set(primary)
    for m in _ABS_PATH_TOKEN_RE.finditer(text or ""):
        p = os.path.normpath(m.group(0))
        if p in seen or not os.path.lexists(p):
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= _MAX_ANCILLARY_PATHS:
            break
    return out


def _execute_code_workdir(arguments: dict[str, Any], task_id: str) -> str:
    """project 模式 execute_code 的实际运行目录；非 project 模式返回空。

    strict 模式的脚本只能经沙箱 RPC 的 hermes_tools 写文件——那条路已被
    write_file / patch 覆盖；project 模式脚本能用 Python 直接改 session cwd
    里的用户文件，必须在启动前保护该目录（Codex review P1）。
    """
    try:
        from tools.code_execution_tool import _get_execution_mode, _resolve_child_cwd

        if _get_execution_mode() != "project":
            return ""
        cwd = str(_resolve_child_cwd("project", "", task_id or "") or "").strip()
        if cwd:
            return _abs_path(cwd)
    except Exception as exc:
        logger.debug("zettlab snapshot guard: execute_code cwd resolution failed: %s", exc)
        # 判定不了就按 project 处置，保护回退 cwd——宁可多拍。
    return _terminal_workdir(arguments, task_id)


def _extract_v4a_paths(patch_body: str) -> list[str]:
    """按 patch_parser 的等价规则抽取 V4A patch 触达的所有路径。"""
    paths: list[str] = []
    for m in _V4A_FILE_RE.finditer(patch_body):
        p = m.group(1).strip()
        if p:
            paths.append(p)
    for m in _V4A_MOVE_RE.finditer(patch_body):
        for p in (m.group(1).strip(), m.group(2).strip()):
            if p:
                paths.append(p)
    return paths


def _paths_for(tool_name: str, arguments: dict[str, Any], task_id: str) -> list[str]:
    """列出这次调用可能改动的绝对路径；返回空表示不需要保护。"""
    if tool_name == "write_file":
        return [p for p in (_resolve_write_path(arguments.get("path"), task_id),) if p]

    if tool_name == "patch":
        mode = str(arguments.get("mode") or "replace").strip()
        if mode == "replace":
            return [p for p in (_resolve_write_path(arguments.get("path"), task_id),) if p]
        # V4A patch 是唯一能触发删除 / 移动的模型路径，一次可能涉及多个文件。
        # 抽不出路径不等于安全：退回工作目录，让整个主文件夹进保护。
        raw_paths = _extract_v4a_paths(str(arguments.get("patch") or ""))
        resolved = [_resolve_write_path(p, task_id) for p in raw_paths]
        return [p for p in resolved if p] or [_terminal_workdir(arguments, task_id)]

    if tool_name == "terminal":
        command = str(arguments.get("command") or "")
        if _command_is_probably_readonly(command):
            return []
        return [p for p in (_terminal_workdir(arguments, task_id),) if p]

    if tool_name == "execute_code":
        return [p for p in (_execute_code_workdir(arguments, task_id),) if p]

    return []


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """拒绝跟随任何重定向。

    urllib 默认会带着非 Content-* 的请求头（包括 action token）跟到 3xx 指向
    的任意地址——loopback 端口被劫持或返回外部重定向时，token 会直接出设备
    （Codex review P1）。3xx 一律按 HTTPError 走 fail-closed。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


_OPENER = urllib.request.build_opener(_NoRedirectHandler())


def _post(path_suffix: str, payload: dict[str, Any], timeout: float) -> tuple[Optional[dict], str]:
    """向 local-server internal 面发一次请求。

    返回 ``(data, error_kind)``：``error_kind`` 为空表示成功；``unconfigured``
    表示这不是设备环境（CLI / 单测），``not_supported`` 表示老 local-server 上
    没有这个接口。两者都不构成阻断理由，其余都构成。
    """
    origin = _local_server_origin()
    token = _scoped_env(_ACTION_TOKEN_ENV, "").strip()
    if not origin or not token:
        return None, "unconfigured"

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        origin + path_suffix,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Zettlab-Agent-Action-Token": token,
        },
    )
    try:
        with _OPENER.open(req, timeout=timeout) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        payload_out = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            # 老 local-server 没有这个路由：设备不具备该能力。
            return None, "not_supported"
        detail = ""
        try:
            detail = exc.read(_MAX_RESPONSE_BYTES).decode("utf-8", "replace")
        except Exception:
            pass
        logger.warning("zettlab snapshot guard: HTTP %s from %s: %s", exc.code, path_suffix, detail[:512])
        return _error_payload(detail), "http_error"
    except Exception as exc:
        logger.warning("zettlab snapshot guard: request to %s failed: %s", path_suffix, exc)
        return None, "transport_error"

    if not isinstance(payload_out, dict):
        return None, "bad_response"
    data = payload_out.get("data")
    if not isinstance(data, dict):
        return None, "bad_response"
    return data, ""


def _error_payload(detail: str) -> Optional[dict]:
    try:
        parsed = json.loads(detail)
    except Exception:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        return {"_error": parsed["error"]}
    return None


def _blocked(message: str, *, outcome: str = "unknown", tool: str = "", started: float = 0.0) -> str:
    """阻断一次破坏性操作，并留下一条结构化日志。

    这条日志是这套 fail-closed 机制在板子上唯一的可观测出口：被挡住的写入，用户
    的体感只是「Agent 突然不肯改文件了」，基本不会有人提单。只记枚举、工具名和耗
    时——**不记路径**（路径只进 local-server 受权限控制的审计表）。
    """
    elapsed_ms = int((time.monotonic() - started) * 1000) if started else -1
    logger.warning(
        "zettlab snapshot guard blocked a write: outcome=%s tool=%s duration_ms=%d",
        outcome,
        tool or "unknown",
        elapsed_ms,
    )
    return json.dumps({"error": message}, ensure_ascii=False)


def _note_task_turn_locked(task_id: str, turn_id: str) -> None:
    """登记 task → turn 映射（调用方须持锁）。有上限，溢出逐出最老的。"""
    if _task_turns.get(task_id) == turn_id:
        return
    _task_turns.pop(task_id, None)
    while len(_task_turns) >= _MAX_TASK_TURNS:
        _task_turns.pop(next(iter(_task_turns)), None)
    _task_turns[task_id] = turn_id


def _state_for_locked(turn_id: str) -> _TurnState:
    """取（或建）该轮的状态（调用方须持锁）。溢出逐出最老的轮。"""
    state = _states.get(turn_id)
    if state is None:
        while len(_states) >= _MAX_TRACKED_TURNS:
            evicted = next(iter(_states))
            _states.pop(evicted, None)
            logger.info(
                "zettlab snapshot guard: evicted turn state over the cap; "
                "its pin will be released by the server-side TTL"
            )
        state = _TurnState(turn_id)
        _states[turn_id] = state
    return state


def maybe_require_snapshot(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    turn_id: str = "",
    task_id: str = "",
) -> Optional[str]:
    """破坏性文件操作前确保保护快照就绪。

    返回 ``None`` 表示放行；返回 JSON 错误字符串表示**不要执行这次操作**，该字符
    串会作为工具结果回给模型。设备环境里任何不确定的情况一律阻断（fail-closed）：
    没有恢复点就动用户文件，是这套机制唯一不能接受的失败方式。
    """
    global _degraded_logged

    turn = str(turn_id or "").strip()
    task = str(task_id or "").strip()
    # 每次带 turn 的 dispatch 都登记 task → turn（包括不设防的 execute_code）：
    # 它孵化的沙箱 RPC 二次进入时只带 task_id，凭这里的映射回落到外层轮。
    if turn and task:
        with _lock:
            _note_task_turn_locked(task, turn)

    if tool_name not in _GUARDED_TOOLS:
        return None

    paths = _paths_for(tool_name, arguments, task)
    if not paths:
        return None

    # 设备环境判定先于一切：非设备环境（CLI / 单测 / 未注入回调与 token 的部署）
    # 完全不介入，嵌套 dispatch 与 MCP bridge 不该在这里被 turn 契约挡住
    # （Codex review P1）。
    if not _local_server_origin() or not _scoped_env(_ACTION_TOKEN_ENV, "").strip():
        return None

    started = time.monotonic()

    if tool_name == "terminal" and bool(arguments.get("background")):
        # 后台破坏性命令会跑到 turn 结束、pin 释放之后，恢复点可能在写入完成前
        # 就被清理；保护窗口对不上就不放行，让模型改用前台执行
        # （Codex review P1）。
        return _blocked(
            "Background terminal commands that modify files are not covered by "
            "protection snapshots. Re-run the command in the foreground "
            "(background=false). The command was NOT executed.",
            outcome="background_write", tool=tool_name, started=started,
        )

    if not turn and task:
        with _lock:
            turn = _task_turns.get(task, "")
    if not turn:
        # 没有轮标识就无法做幂等，会把每次写入都变成一张新快照。这属于调度层
        # 契约被破坏，放行比拍一堆快照更糟，所以阻断。
        return _blocked(
            "File protection snapshot unavailable: missing turn id. The file was not modified.",
            outcome="missing_turn_id", tool=tool_name, started=started,
        )

    with _lock:
        state = _state_for_locked(turn)
        # 本轮自己创建的文件，后续修改不再触发快照。
        if tool_name in _FILE_MUTATING_TOOLS and not state.created_overflowed:
            remaining = [p for p in paths if p not in state.created]
            if not remaining:
                return None
            paths = remaining
        pending_new = [p for p in paths if not os.path.lexists(p)]

    data, err = _post(
        _ENSURE_PATH,
        {"turnId": turn, "paths": paths, "title": _title_for(paths)},
        _ENSURE_TIMEOUT,
    )

    if err == "unconfigured":
        return None  # 不是设备环境（CLI / 测试），本机制不适用
    if err == "not_supported":
        if not _degraded_logged:
            _degraded_logged = True
            logger.info(
                "zettlab snapshot guard: local-server has no agent file protection endpoint; "
                "proceeding without protection snapshots"
            )
        return None
    if err or data is None:
        detail = ""
        if isinstance(data, dict) and isinstance(data.get("_error"), dict):
            detail = str(data["_error"].get("message") or "")
        return _blocked(
            "Could not create a protection snapshot before modifying files"
            + (f" ({detail})" if detail else "")
            + ". The file was NOT modified. Tell the user the change did not happen; do not retry blindly.",
            outcome=f"ensure_{err or 'bad_response'}", tool=tool_name, started=started,
        )

    if not data.get("ready"):
        # 服务端只在真正失败时才会给 ready=false（无保护路径它自己就放行了）。
        return _blocked(
            "The protection snapshot is not ready. The file was NOT modified.",
            outcome="not_ready", tool=tool_name, started=started,
        )

    _log_unprotected(data, tool_name)

    with _lock:
        state = _state_for_locked(turn)
        state.ensured = True
        if tool_name in _FILE_MUTATING_TOOLS:
            for p in pending_new:
                if len(state.created) >= _MAX_CREATED_TRACKED:
                    state.created_overflowed = True
                    break
                state.created.add(p)

    # cwd 之外的绝对路径目标（`rm /home/...`、脚本里的 Path("/home/...")）走
    # **附加** ensure：尽力给它们也建恢复点，但任何失败（含范围外 403）只记
    # 日志不阻断——cwd 主保护已就绪，这是加餐；把 /tmp 一类范围外路径判成
    # 硬拒绝反而会把整条命令误杀（Codex review P1）。
    if tool_name in ("terminal", "execute_code"):
        extras = _ancillary_abs_paths(
            str(arguments.get("command") or arguments.get("code") or ""), paths)
        if extras:
            extra_data, extra_err = _post(
                _ENSURE_PATH,
                {"turnId": turn, "paths": extras, "title": _title_for(extras)},
                _ENSURE_TIMEOUT,
            )
            if extra_err:
                logger.info("zettlab snapshot guard: ancillary ensure skipped (%s)", extra_err)
            elif isinstance(extra_data, dict):
                _log_unprotected(extra_data, tool_name)
    return None


def _title_for(paths: list[str]) -> str:
    """快照标题：首个文件名，多文件时带上数量。服务端会按 32 code point 截断。"""
    if not paths:
        return ""
    first = os.path.basename(paths[0]) or paths[0]
    if len(paths) == 1:
        return first
    return f"{first} 等 {len(paths)} 个文件"


def _log_unprotected(data: dict, tool_name: str) -> None:
    """无保护放行：改动没有恢复点，用户当场不知情，这条日志是唯一的现场记录。

    服务端已经把路径写进受权限控制的审计表；这里只记原因枚举，不记路径。
    """
    reasons = {
        str(op.get("unprotectedReason"))
        for op in (data.get("operations") or [])
        if isinstance(op, dict) and op.get("unprotected") and op.get("unprotectedReason")
    }
    if not reasons:
        return
    logger.info(
        "zettlab snapshot guard: allowing a write with no recovery point (reason=%s tool=%s)",
        ",".join(sorted(reasons)),
        tool_name or "unknown",
    )


def finish_turn(
    state: str = "completed",
    *,
    turn_id: str = "",
    error_code: str = "",
    error_stage: str = "",
) -> None:
    """一轮任务收尾：上报终态并解除该轮保护快照的 pin。

    ``turn_id`` 指明收哪一轮（agent 的 ``_current_turn_id``），**只做精确匹配**：
    没有受保护写入的轮本来就没有状态条目，未命中时去收「唯一余轮」会把另一个
    还在写的轮的 pin 提前解掉（Codex review P1）。不带 ``turn_id`` 时只有恰好
    只剩一轮在跟踪才收它（cron 拿不到 agent 实例的兜底）；其余情况宁可不收，
    代价只是等服务端 TTL 自愈。本轮没发生过保护快照时是纯 no-op。上报失败只
    记日志。
    """
    turn = str(turn_id or "").strip()
    with _lock:
        current: Optional[_TurnState] = None
        if turn:
            current = _states.pop(turn, None)
        elif len(_states) == 1:
            _, current = _states.popitem()
        elif _states:
            logger.info(
                "zettlab snapshot guard: ambiguous finish for %d concurrent turn(s); "
                "leaving their pins to the server-side TTL",
                len(_states),
            )
    if current is None or not current.ensured:
        return

    _post(
        _FINISH_PATH,
        {
            "turnId": current.turn_id,
            "state": state,
            "errorCode": error_code,
            "errorStage": error_stage,
        },
        _FINISH_TIMEOUT,
    )


def reset_for_test() -> None:
    """测试钩子：丢弃进程内的轮状态。"""
    global _degraded_logged, _nonloopback_logged
    with _lock:
        _states.clear()
        _task_turns.clear()
        _degraded_logged = False
        _nonloopback_logged = False
