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

import glob
import ipaddress
import json
import logging
import os
import re
import shlex
import shutil
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
# text_to_speech 也是文件写入面：自定义 output_path 会先删再写任意路径
# （tts_tool），已存在的用户文件必须先有恢复点（Codex review P1）。
_FILE_MUTATING_TOOLS = frozenset({"write_file", "patch", "text_to_speech"})
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
# 单个 & 也是 control operator：`ls & rm x` 是两段命令，漏拆会让 & 后的写入段
# 藏进只读判定（Codex review P1）。&& 在前保证优先整体匹配。
_SHELL_CHAIN_SPLIT_RE = re.compile(r"&&|\|\||;|\||\n|&")
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# 注意不收这些「看似只读」的命令：find 有 -delete / -exec、sort 有 -o、tree 有
# -o、env 可以执行任意命令（Codex review P1）。宁可让它们多触发一次 cwd 保护。
# 只读安全清单：**只收行为不受配置 / 环境驱动的命令**。
#
# git、less/more、rg 已被整体移出（Codex review P1 ×N）：它们的行为由用户配置
# 与环境变量驱动，能挂上任意外部命令——git 有 diff.external、diff.<drv>.textconv、
# core.fsmonitor、core.pager、alias.*、hooks；less/more 有 LESSOPEN / LESSCLOSE
# 预处理器；rg 有 RIPGREP_CONFIG_PATH，配置文件里的 --pre 会对每个候选文件执行
# 任意命令（只有 --no-config 才忽略它）。逐个子命令 / flag 去堵是无穷尽的，而
# 它们本就不该出现在「可证明只读」的清单里。代价只是这些命令会按 cwd 拍一张幂
# 等快照（每轮每目录一张），不是阻断。
#
# grep / egrep / fgrep 保留：它们没有「配置文件里挂外部命令」的等价面
# （GREP_OPTIONS 早已移除，且从不执行命令）。
#
# diff 不收：`-l/--paginate` 会把输出交给 PATH 上的 `pr` 执行——参数即可挂
# 外部命令（Codex review P1）。cmp 无此面，保留。
#
# 同理不收**参数就能写文件**的命令：uniq 的第二个位置参数是 OUTPUT
# （`uniq in.txt out.txt` 直接覆盖 out.txt，Codex review P1）；file 的
# `-C -m` 会编译写出 .mgc。数「非 option 操作数」需要维护每个命令的带值
# flag 表，与逐 flag 堵配置驱动命令是同一条不归路——直接移出清单。
_READONLY_FIRST_TOKENS = frozenset({
    "ls", "cat", "grep", "egrep", "fgrep", "head", "tail",
    "wc", "pwd", "echo", "printf", "stat", "which",
    "type", "printenv", "ps", "df", "du", "date", "whoami", "id",
    "uname", "md5sum", "sha1sum", "sha256sum", "cut", "tr",
    "cmp", "readlink", "basename", "dirname", "hostname", "uptime",
    "free", "realpath", "test", "true", "false", "sleep",
})
# 命令文本里的绝对路径 token：cwd 之外的写入目标（rm /home/alice/... 或脚本里
# 的 Path("/home/...").write_text）也要尽力保护（Codex review P1）。这些路径走
# **附加** ensure：范围外（403）只跳过、不阻断——它们是 cwd 保护之外的加餐。
_ABS_PATH_TOKEN_RE = re.compile(r"(?<![\w.+@%*?{}\[\]-])/(?:[\w.+@%*?{},\[\]-]+/)*[\w.+@%*?{},\[\]-]+")
# 引号字面量里的路径可以含空格（`rm -f '/home/a/My Documents/x'`、
# `open("/home/a/My Documents/x","w")`），裸 token 正则会在空格处截断而漏掉
# 真实目标（Codex review P1）。shell 与 Python 文本统一按引号对提取。除绝对
# 路径外，home 前缀（~ / $HOME / ${HOME}）与 `../` 相对目标也要接住——它们
# 同样能指到 cwd 之外的用户文件（Codex review P1）。
_QUOTED_PATHISH_RES = (
    re.compile(r"'((?:/|~/|\$HOME/|\$\{HOME\}/|\.\./)[^'\n]+)'"),
    re.compile(r'"((?:/|~/|\$HOME/|\$\{HOME\}/|\.\./)[^"\n]+)"'),
)
# home / parent 裸 token 同样要带 glob 字符：`rm -rf ~/Doc*` 截成 `~/Doc` 后
# lexists 不到，整条附加保护会静默漏掉（Codex review P1）。
_BARE_HOME_TOKEN_RE = re.compile(
    r"(?<![\w.-])(?:~|\$HOME|\$\{HOME\})/(?:[\w.+@%*?{},\[\]-]+/)*[\w.+@%*?{},\[\]-]+")
_BARE_PARENT_TOKEN_RE = re.compile(
    r"(?<![\w.-])\.\.(?:/(?:\.\.|[\w.+@%*?{},\[\]-]+))+")
_GLOB_CHARS = ("*", "?", "[")
_MAX_ANCILLARY_PATHS = 16

# shell 会把引号变量与后缀拼成同一个路径词：`rm -f "$HOME"/Documents/a.txt`
# 删的是 HOME 下的真实文件，但正则要求 $HOME/ 出现在同一个匹配里就抽不到它
# （Codex review P1）。提取前先把引号包裹的 HOME 归一成裸 $HOME。
_QUOTED_HOME_RE = re.compile(r"""["']\$\{?HOME\}?["']""")
# brace expansion：`rm -f /home/a/Doc{1,2}.txt` 真会删两个文件，静态可确定的
# 形态要展开后再过滤（Codex review P1）。只处理不含嵌套的简单组，结果有上限。
_BRACE_GROUP_RE = re.compile(r"\{([^{}]*,[^{}]*)\}")


def _expand_braces(tok: str, limit: int = _MAX_ANCILLARY_PATHS) -> list[str]:
    """展开静态可确定的 brace 组（{a,b}），最多 limit 个结果。

    不支持嵌套与序列（{1..9}）：那些形态展开成本高、收益低，展不开时原样返回，
    由 lexists 过滤掉——退化方向是「少保护一个附加目标」，与其它无法界定的
    shell 形态一致（PRD §2 Phase 1 边界）。
    """
    out = [tok]
    while True:
        m = _BRACE_GROUP_RE.search(out[0])
        if m is None:
            return out[:limit]
        alts = m.group(1).split(",")
        expanded: list[str] = []
        for item in out:
            hit = _BRACE_GROUP_RE.search(item)
            if hit is None:
                expanded.append(item)
                continue
            for alt in alts:
                expanded.append(item[:hit.start()] + alt + item[hit.end():])
                if len(expanded) >= limit:
                    break
            if len(expanded) >= limit:
                break
        out = expanded or [tok]


def _subprocess_home() -> str:
    """工具子进程实际生效的 HOME。

    terminal / execute_code 子进程经 apply_subprocess_home_env 处理，
    home_mode=profile / 容器 / 缺 HOME fallback 时 HOME 会被换成
    {HERMES_HOME}/home——按进程 HOME 展开会保护错目标（Codex review P1）。
    与其同源的 get_subprocess_home 返回 None 表示沿用当前 HOME。
    """
    try:
        from hermes_constants import get_subprocess_home

        home = get_subprocess_home(dict(os.environ))
        if home:
            return home
    except Exception:
        pass
    return os.environ.get("HOME") or os.path.expanduser("~")


def _normalize_pathish(tok: str, base_dir: str) -> str:
    """把提取出的路径样 token 归一成绝对路径；归一不了返回空串。"""
    tok = tok.strip()
    if tok.startswith(("~", "$HOME", "${HOME}")):
        home = _subprocess_home()
        for prefix in ("${HOME}", "$HOME", "~"):
            if tok == prefix or tok.startswith(prefix + "/"):
                tok = home + tok[len(prefix):]
                break
    elif tok.startswith(".."):
        if not base_dir:
            return ""
        tok = os.path.join(base_dir, tok)
    if not tok.startswith("/"):
        return ""
    return os.path.normpath(tok)


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
# 折叠容器 key（共享容器把并发轮折叠到同一个 key，通常 "default"）单独建帐：
# key → 仍在进行的轮集合（有序）。恰好一轮时嵌套 RPC 回落到它；多轮并发时归
# 属不可判定，受保护写入 fail-closed——直接覆盖登记会把 ensure 归错轮、随对
# 方 finish 提前解 pin（Codex review P1）。finish 时把轮从集合里摘除。
_collapsed_task_turns: dict[str, dict[str, None]] = {}
_MAX_COLLAPSED_KEYS = 64
_degraded_logged = False
_nonloopback_logged = False


class _UnresolvableScope(Exception):
    """multiplex 下 profile scope 未绑定：读不到配置 ≠ 不是设备环境。"""


def _scoped_env(name: str, default: str = "") -> str:
    """读环境变量。multiplex 下必须走 profile scope，否则会串到别的 agent。

    scope 未绑定（UnscopedSecretError）时抛 _UnresolvableScope 而不是回落到
    默认值：那样 guard 会把「读不到 token」当成「不是设备环境」而放行写入，
    用户文件在没有恢复点的情况下被改（Codex review P1）。这是调度层契约被破
    坏，fail-closed。
    """
    try:
        from agent.secret_scope import get_secret

        value = get_secret(name, "")
        if value:
            return str(value)
    except Exception as exc:
        if type(exc).__name__ == "UnscopedSecretError":
            raise _UnresolvableScope(str(exc)) from exc
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


# 与 terminal_tool._get_env_config 同源的后端判定（都从环境变量派生）：只有
# 容器后端里 /workspace 才可能是 host 路径的 bind 视图。
_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona"})


def _bridge_terminal_env() -> None:
    """触发 terminal_tool 的懒桥接，把 config.yaml 的 terminal.* 灌进环境变量。

    幂等；桥接不可用（非 hermes 环境 / 单测桩缺失）时静默降级，读到什么算什么。
    """
    try:
        from tools.terminal_tool import _ensure_terminal_env_bridged

        _ensure_terminal_env_bridged()
    except Exception as exc:
        logger.debug("zettlab snapshot guard: terminal env bridge unavailable: %s", exc)


def _terminal_env_type() -> str:
    """terminal 的有效 backend 类型（先触发懒桥接）。"""
    _bridge_terminal_env()
    return (os.getenv("TERMINAL_ENV") or "local").strip().lower()


def _terminal_backend_is_remote() -> bool:
    """报告工具的有效 backend 是否在远端主机执行（ssh）。

    远端文件系统不在本机快照的覆盖面内，按本机路径 ensure 只会造出假恢复点
    （Codex review P1）。
    """
    return _terminal_env_type() == "ssh"


# ssh backend 下会把写入送去远端执行的工具面：terminal（SSHEnvironment 跑命
# 令）、write_file / patch（file_tools._get_file_ops 按 env_type 建
# SSHEnvironment）、execute_code（非 local 走 _execute_remote）。
# text_to_speech 的输出走主进程本地路径语义，不在此列。
_REMOTE_UNSAFE_TOOLS = frozenset({"terminal", "write_file", "patch", "execute_code"})


def _json_env_list(name: str) -> list:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _mount_to_bind_spec(val: str) -> str:
    """把 --mount 的 csv 形式转成 host:container；非 bind 返回空串。"""
    kv: dict[str, str] = {}
    for field in val.split(","):
        k, _, v = field.partition("=")
        kv[k.strip().lower()] = v.strip()
    if kv.get("type", "bind") != "bind":
        return ""
    src = kv.get("source") or kv.get("src") or ""
    dst = kv.get("target") or kv.get("destination") or kv.get("dst") or ""
    return f"{src}:{dst}" if src and dst else ""


def _bind_mount_specs() -> list[str]:
    """全部 bind 挂载的 host:container[:opts] 规格。

    两个来源：TERMINAL_DOCKER_VOLUMES，以及 TERMINAL_DOCKER_EXTRA_ARGS 里的
    -v / --volume / --mount type=bind——extra_args 会被 DockerEnvironment 原样
    追加进 docker run，漏掉它们会让这些挂载下的写入按容器字面路径建快照
    （Codex review P1）。
    """
    specs = [v for v in _json_env_list("TERMINAL_DOCKER_VOLUMES")
             if isinstance(v, str) and ":" in v]
    extra = [str(a) for a in _json_env_list("TERMINAL_DOCKER_EXTRA_ARGS")]
    i = 0
    while i < len(extra):
        arg = extra[i]
        nxt = extra[i + 1] if i + 1 < len(extra) else ""
        if arg in ("-v", "--volume") and ":" in nxt:
            specs.append(nxt)
            i += 2
            continue
        if arg.startswith(("-v=", "--volume=")):
            val = arg.split("=", 1)[1]
            if ":" in val:
                specs.append(val)
            i += 1
            continue
        if arg == "--mount" and nxt:
            spec = _mount_to_bind_spec(nxt)
            if spec:
                specs.append(spec)
            i += 2
            continue
        if arg.startswith("--mount="):
            spec = _mount_to_bind_spec(arg.split("=", 1)[1])
            if spec:
                specs.append(spec)
            i += 1
            continue
        i += 1
    return specs


def _container_path_maps() -> list[tuple[str, str]]:
    """容器路径前缀 → host 路径的映射表（最长前缀优先）。

    镜像 DockerEnvironment 的挂载语义（Codex review P1）：docker_volumes 里
    每一条 host 侧为绝对路径的 bind 都会把 host 目录暴露进容器（不止
    /workspace——`/home/a/Pictures:/mnt/pics` 下写 /mnt/pics 动的是 host 的
    Pictures）；显式挂到 /workspace 的 volume 优先于
    docker_mount_cwd_to_workspace 的 cwd bind（后者在显式挂载存在时被跳过）。
    named volume（host 侧非路径）不进表——那不是 host 用户文件，保留容器路径
    交给服务端 scope 校验。local 后端不做任何映射。
    """
    # TERMINAL_* 是懒桥接的：`hermes serve` / Desktop / ACP 等路径要等
    # terminal_tool._get_env_config() 才把 config.yaml 的 terminal.backend /
    # cwd / docker_volumes 灌进环境变量。不先触发桥接就读 env，会在 Docker 会
    # 话里按 local backend 判定、给错路径建快照（Codex review P1）。
    _bridge_terminal_env()

    env_type = (os.getenv("TERMINAL_ENV") or "local").strip().lower()
    if env_type not in _CONTAINER_BACKENDS:
        return []
    maps: list[tuple[str, str]] = []
    workspace_taken = False
    for spec in _bind_mount_specs():
        host, _, rest = spec.strip().partition(":")
        container = rest.split(":", 1)[0].strip()
        if not container.startswith("/"):
            continue
        container = os.path.normpath(container)
        if container == "/workspace" or container.startswith("/workspace/"):
            workspace_taken = True
        if host.startswith(("/", "~")):
            maps.append((container, os.path.abspath(os.path.expanduser(host))))
    flag = (os.getenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE") or "false").strip().lower()
    if not workspace_taken and flag in {"true", "1", "yes"}:
        host = os.path.abspath(os.path.expanduser(os.getenv("TERMINAL_CWD") or os.getcwd()))
        if os.path.isdir(host):
            maps.append(("/workspace", host))
    maps.sort(key=lambda m: len(m[0]), reverse=True)
    return maps


def _map_container_path(p: str) -> str:
    """把容器路径按 volume 映射表反解回 host 路径。

    挂载会话里 file_tools / 会话 cwd 记录落在容器口径上；原样上报会让快照落
    在 host 上不存在（或错误）的容器路径上，真正被改的 host 目录没有恢复点
    （Codex review P1）。映射按 terminal 的 volume / 挂载配置判定（最长前缀
    优先），不看 host 上是否恰好存在同名目录（Codex review P1）；映射不到的
    容器路径原样保留，由服务端 scope 校验 fail-closed。
    """
    if not p.startswith("/"):
        return p
    for container, host in _container_path_maps():
        if p == container or p.startswith(container + "/"):
            tail = p[len(container):].lstrip("/")
            return os.path.normpath(os.path.join(host, tail)) if tail else host
    return p


def _resolve_write_path(path: Any, task_id: str) -> str:
    """解析 write_file / patch 的目标路径，与文件工具自己的口径对齐。

    实际写入按 task_id 走 ``tools.file_tools._resolve_path_for_task``（会话注册
    的 cwd 优先于进程 env）；guard 若用进程级 cwd 解析相对路径，会给错误目录拍
    快照而真正被写的文件没有恢复点（Codex review P1）。解析器不可用时退回进程
    级解析。

    **空 task_id 也要走会话解析**：handler 侧签名是
    ``write_file_tool(..., task_id: str = "default")``，直连 registry.dispatch
    只带 turn_id 的调用最终按 default 会话的 cwd / profile HOME 落盘。这里若
    因为 task_id 为空就跳过、退回进程级 cwd，default 会话 `cd` 过之后 guard 就
    会给错目录建快照，真正被写的文件没有恢复点（Codex review P1）。
    """
    raw = str(path or "").strip()
    if not raw:
        return ""
    try:
        from tools.file_tools import _resolve_path_for_task

        return _map_container_path(
            os.path.normpath(str(_resolve_path_for_task(raw, task_id or "default"))))
    except Exception:
        pass
    return _map_container_path(_abs_path(raw))


def _managed_output_fallback() -> str:
    """受管网关下的平台 output 目录；不可用返回空串。仅用于措辞判断。"""
    try:
        from tools.environments.local import managed_fallback_cwd

        return str(managed_fallback_cwd(None) or "")
    except Exception:
        return ""


def _managed_effective_workdir(cwd: str) -> str:
    """把候选 cwd 过一遍执行侧同一个解析器，返回命令真正会跑的目录。

    必须调用 ``_managed_terminal_cwd`` 用的那个函数，而不是在兜底链里另插一
    层：终端的默认 cwd 是硬编码的 ``/root``（0700 root），受管身份穿不进去，
    执行侧本来就会回退到平台 output 目录。守卫若按 ``/root`` 请求快照，就会被
    scope 判定 403、把所有非只读命令整条拦死——根因是锚点无主，不是该放行。
    两边各自推导则会漂移成「快照拍在 A、命令跑在 B」，那正是本守卫要防的裂缝。
    非受管 / Windows / 模块缺失时原样返回，绝不抛。
    """
    try:
        from tools.environments.local import managed_effective_cwd

        return str(managed_effective_cwd(cwd) or cwd)
    except Exception:
        return cwd


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
    # TERMINAL_CWD 是懒桥接的：local backend 且没有 session_cwd / 显式 workdir
    # 时直接读它兜底，此前若没有任何调用触发过 _get_env_config()，config.yaml
    # 的 terminal.cwd 根本不在环境里——真实 terminal_tool 会先桥接再跑命令，
    # guard 不桥接就会给 Hermes 进程 cwd 建快照，命令实际跑在 terminal.cwd
    # （Codex review P1）。
    _bridge_terminal_env()

    session_cwd = ""
    try:
        from tools.terminal_tool import get_session_cwd

        # 空 task_id 不能跳过：get_session_cwd 明确把 None / 空 key 读作
        # "default" 记录，terminal_tool 内部也按 `task_id or "default"` 归一。
        # 跳过就退回进程级 cwd，而命令实际跑在 default 会话 `cd` 到的目录
        # （Codex review P1，与 _resolve_write_path 同源的口径缺口）。
        session_cwd = str(get_session_cwd(task_id) or "")
    except Exception:
        session_cwd = ""
    if session_cwd:
        session_cwd = _map_container_path(_abs_path(session_cwd))

    explicit = str(arguments.get("workdir") or "").strip()
    if explicit:
        try:
            from tools.runtime_workdir import resolve_runtime_workdir

            explicit = str(resolve_runtime_workdir(explicit) or "").strip()
        except ValueError:
            # Registry dispatch rejects an unavailable semantic alias before
            # this gate. Direct guard calls still fail safe by protecting the
            # fallback cwd; the terminal handler will reject the same alias.
            explicit = ""
    if explicit:
        # `~` 按工具子进程实际生效的 HOME 展开：workdir 的 `cd` 由 shell 按子
        # 进程 $HOME 解释，home_mode=profile / 缺 HOME fallback 时它是
        # {HERMES_HOME}/home，用 Hermes 进程的 expanduser 会给真实 OS HOME 建
        # 快照、实际被写的 profile home 没有恢复点（Codex review P1）。
        if explicit == "~" or explicit.startswith("~/"):
            expanded = _subprocess_home() + explicit[1:]
        else:
            expanded = os.path.expanduser(explicit)
        if os.path.isabs(expanded):
            p = _map_container_path(os.path.normpath(expanded))
        else:
            # 相对 workdir 是「在会话 cwd 下 cd」的语义，必须先锚到会话自己的
            # cwd，进程 env 里同名目录会指向错误位置（Codex review P1）。锚点
            # 若来自 TERMINAL_CWD（容器口径，如 /workspace/Project），join 出
            # 的结果也仍是容器口径，要反解后再做 host 存在性检查——否则
            # `workdir=\"../Documents\"` 会在 host 上 isdir 失败、错退回保护
            # cwd，真实被写的 host Documents 没有恢复点（Codex review P1）；
            # session_cwd 已是 host 口径，再过一次映射是 no-op。
            base = session_cwd or _abs_path(os.getenv("TERMINAL_CWD") or os.getcwd())
            p = _map_container_path(os.path.normpath(os.path.join(base, expanded)))
        if os.path.isdir(p):
            return _managed_effective_workdir(p)
    if session_cwd and os.path.isdir(session_cwd):
        return _managed_effective_workdir(session_cwd)
    # TERMINAL_CWD 本身可能就是容器口径（config 把 cwd 写成 /workspace）：显式
    # workdir 与 session_cwd 都做了反解，fallback 不反解会给本机字面 /workspace
    # 建快照，真实被写的是 bind 到它的 host 目录（Codex review P1）。
    return _managed_effective_workdir(
        _map_container_path(_abs_path(os.getenv("TERMINAL_CWD") or os.getcwd()))
    )


# 单个 `&`（非 `&&` / `2>&1` / `&>`）把命令甩到后台。
_SHELL_AMP_BACKGROUND_RE = re.compile(r"(?<![&>|])&(?![&>])")

# nohup / setsid 把子进程甩出保护窗口；env / command / exec / nice 一类包裹层
# 不改变「最终执行谁」，判定时逐层剥掉再看真正的命令头。值集合列出的 flag 会
# 吃掉后面一个参数词（nice -n 10、env -u VAR）。
# coproc 是 bash 关键字：把命令挂成后台协进程，与 nohup / setsid 同罪
# （Codex review P1）。
_DAEMONIZE_HEADS = frozenset({"nohup", "setsid", "coproc"})
_WRAPPER_VALUE_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"-u", "-C", "--unset", "--chdir"}),
    "command": frozenset(),
    "exec": frozenset({"-a"}),
    "nice": frozenset({"-n"}),
    "ionice": frozenset({"-c", "-n", "-p"}),
    "stdbuf": frozenset({"-i", "-o", "-e"}),
    "time": frozenset({"-f", "-o"}),
}
# env 短选项簇里的 S（-S / -vS / -Sxxx）：--split-string 会把值重新拆成真正的
# 命令词，不能当不透明参数跳过。
_ENV_SPLIT_STRING_RE = re.compile(r"^-[a-zA-Z]*S")
# sh/bash 的 -c（含 -lc / -ec 短选项簇）：后面的字面命令串会重新进入 shell 解析。
_SHELL_HEADS = frozenset({"sh", "bash", "zsh", "dash", "ksh"})
_SHELL_DASH_C_RE = re.compile(r"^-[a-zA-Z]*c$")
# shlex punctuation_chars 模式下会单独成 token 的 shell 操作符字符。
_SHELL_PUNCT_CHARS = frozenset("();<>|&")


def _env_split_string_value(flag: str, words: list[str], i: int) -> tuple[Optional[str], int]:
    """取出 env -S / --split-string 携带的字符串；不是该形态返回 (None, i)。"""
    if flag == "--split-string":
        if i < len(words):
            return words[i], i + 1
        return "", i
    if flag.startswith("--split-string="):
        return flag.split("=", 1)[1], i
    if _ENV_SPLIT_STRING_RE.match(flag):
        rest = flag.split("S", 1)[1]
        if rest:
            return rest, i
        if i < len(words):
            return words[i], i + 1
        return "", i
    return None, i


def _segment_daemonizes(words: list[str]) -> bool:
    """报告一个链段（已按 shell word 语义分好词）是否经 nohup / setsid 自后台化。"""
    i = 0
    while i < len(words):
        word = words[i]
        if _ENV_ASSIGN_RE.match(word):
            i += 1
            continue
        name = os.path.basename(word)
        if name in _DAEMONIZE_HEADS:
            return True
        if name in _SHELL_HEADS:
            # `sh -c '字面命令串'` 会重新进入 shell 解析，与 env -S 同类：递归
            # 跑同一套自后台化判定（Codex review P1）。变量间接
            # （sh -c \"$CMD\"）在 shlex 展开后只剩 $CMD 字面量、`sh script.sh`
            # 裸脚本执行，均属 PRD §2 Phase 1 静态判定边界。
            # bash 语义：-c 的命令串是**第一个非选项操作数**，`--` 终止选项
            # 解析——`bash -c -- 'cmd'` 执行的是 cmd 而不是 `--`，把 `--`
            # 当命令串递归会判出 False 放行（Codex review P1）。同理
            # `bash -c -l 'cmd'` 的命令串也在后续选项之后。扫过全部选项与
            # 至多一个 `--` 再取命令串递归。
            j = i + 1
            saw_dash_c = False
            while j < len(words):
                flag = words[j]
                if flag == "--":
                    j += 1
                    break
                if _SHELL_DASH_C_RE.match(flag):
                    saw_dash_c = True
                    j += 1
                    continue
                if flag.startswith("-"):
                    j += 1
                    continue
                break
            if saw_dash_c:
                return j < len(words) and _shell_self_backgrounds(words[j])
            return False
        if name in _WRAPPER_VALUE_FLAGS:
            # `command -v xxx` 只查名字不执行，不是包裹层。
            if name == "command" and i + 1 < len(words) and words[i + 1] in ("-v", "-V"):
                return False
            value_flags = _WRAPPER_VALUE_FLAGS[name]
            i += 1
            while i < len(words) and words[i].startswith("-"):
                flag = words[i]
                i += 1
                if name == "env":
                    # `env -S "setsid ..."` 会把字符串重新拆成命令词执行——按
                    # 不透明参数跳过就漏掉了里面的 daemonizer（Codex review
                    # P1）。递归按 shell word 拆开、拼回词流继续判定；拆不了
                    # fail-closed。env 自身的 ${VAR} 展开属于 Phase 1 的静态
                    # 判定边界（等价于 sh -c "$CMD" 间接层）。
                    value, ni = _env_split_string_value(flag, words, i)
                    if value is not None:
                        i = ni
                        try:
                            words[i:i] = shlex.split(value, posix=True)
                        except ValueError:
                            return True
                        break
                if flag in value_flags and i < len(words) and not words[i].startswith("-"):
                    i += 1
            continue
        return False
    return False


def _shell_self_backgrounds(command: str) -> bool:
    """报告一条 shell 命令是否会自行后台化（`cmd &`、nohup / setsid 包裹）。

    必须按 shell word 语义解析：`'setsid' -f ...` 的引号在执行时会被 shell 剥
    掉、跑的仍是 setsid，朴素空格 split 把引号留在 token 里就漏判（Codex
    review P1）；链接符也要引号感知——`LESSOPEN='|rm %s' less` 的 `|` 在引号
    里，不是管道。用 shlex 的 punctuation_chars 模式一次拿到词与操作符，按操
    作符重新分段判定。引号不配对等解析不了的形态 fail-closed 按自后台化处理
    ——识别不准就不放行。
    """
    if _SHELL_AMP_BACKGROUND_RE.search(command):
        return True
    try:
        lex = shlex.shlex(command, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        words = list(lex)
    except ValueError:
        return True
    segment: list[str] = []
    segments = [segment]
    for word in words:
        if word and all(ch in _SHELL_PUNCT_CHARS for ch in word):
            segment = []
            segments.append(segment)
            continue
        segment.append(word)
    return any(_segment_daemonizes(seg) for seg in segments)


def _shell_function_shadows(name: str) -> bool:
    """报告某个命令名是否被**导出的** bash function 取代。

    bash 用 ``BASH_FUNC_<name>%%`` 形式的环境变量传递导出函数，terminal 子进程
    继承同一份 env，所以这里看得见（Codex review P1）。**看不见**的是会话内用
    ``alias`` / ``function`` 定义、或 rc 文件里定义的同名符号——那是 shell 进程
    的内部状态，纯判定函数拿不到，要拿只能在会话里执行 `type -a`（每条命令一次
    额外往返 + 副作用面）。这属于 PRD §2 Phase 1 的静态判定边界；缓解在于：定
    义 alias 的命令本身（`alias`/`function`/`source`）不在只读清单里，会先触发
    一次 cwd 保护。
    """
    return any(
        key.startswith(f"BASH_FUNC_{name}") for key in os.environ
    )


def _shell_init_hooks_present() -> bool:
    """报告终端会话会加载的 shell init 文件是否存在且非空。

    LocalEnvironment 建会话时显式 source init 文件并把 alias 快照进会话：rc
    文件里的 `alias ls='rm ...'` / 同名 function 不需要模型在会话内定义就已生
    效，命令头证明不了任何事（Codex review P1）。**优先复用 terminal 自己的
    `_resolve_shell_init_files()`**：它覆盖 `terminal.shell_init_files` 自定义
    列表与 auto_source_bashrc 的登录链，和真实会话 source 的是同一份清单
    （Codex review P1）。解析器不可用时退回固定三件套 + 子进程 HOME。
    """
    _bridge_terminal_env()
    try:
        from tools.environments.local import _resolve_shell_init_files

        files = _resolve_shell_init_files()
    except Exception:
        files = None
    if files is not None:
        for path in files:
            try:
                if os.path.getsize(path) > 0:
                    return True
            except OSError:
                continue
        return False
    home = _subprocess_home()
    for name in (".profile", ".bash_profile", ".bashrc"):
        try:
            if os.path.getsize(os.path.join(home, name)) > 0:
                return True
        except OSError:
            continue
    return False


# 只读放行要求首词真实解析到这些系统目录里的二进制：terminal 子进程继承
# PATH，`/home/alice/bin/ls` 这类同名 helper 排在系统目录前时，按名放行等于
# 放行任意程序（Codex review P1）。usrmerge 的 /bin -> /usr/bin 由 realpath
# 归一，两种前缀都收。
_TRUSTED_BIN_PREFIXES = ("/usr/bin/", "/bin/", "/usr/sbin/", "/sbin/")


def _resolves_to_system_binary(name: str) -> bool:
    """报告命令名在当前 PATH 下是否解析到可信系统目录里的真实二进制。

    解析不到（纯 shell builtin 如 ``type``）或 realpath 落在用户可写路径 →
    False，调用方放弃只读快捷、退化成拍一张幂等 cwd 快照——多拍无害，放错
    才有害。PATH 取 Hermes 主进程环境：LocalEnvironment 子进程与主进程同源
    继承，config 注入的差异属于 Phase 1 静态判定边界。
    """
    try:
        resolved = shutil.which(name)
        if not resolved:
            return False
        return os.path.realpath(resolved).startswith(_TRUSTED_BIN_PREFIXES)
    except Exception:
        return False


def _command_is_probably_readonly(command: str) -> bool:
    """报告一条 shell 命令是否**可证明**只读。证明不了就按需要保护处理。"""
    if not command.strip():
        return True
    env_type = _terminal_env_type()
    if env_type in _CONTAINER_BACKENDS:
        # 容器 backend 的命令在容器内 shell 执行：terminal.docker_env /
        # TERMINAL_DOCKER_ENV 在容器创建时以 `-e` 注入（BASH_ENV=/workspace/
        # hook.sh 会先于任何 `docker exec ... bash -c` 命令被 source），镜像
        # 自带的 ENV / rc 文件更是主进程完全看不见——`_shell_init_hooks_
        # present()` 只查得到 Hermes 自己的环境（Codex review P1）。子进程
        # 环境不可验证就没有「可证明只读」。代价只是容器会话的命令按 cwd 拍
        # 幂等快照，不是阻断。
        return False
    if os.environ.get("BASH_ENV") or os.environ.get("ENV") or _shell_init_hooks_present():
        # 非交互 bash 启动时会先 source BASH_ENV（POSIX sh 用 ENV）指向的脚
        # 本；LocalEnvironment 的会话初始化还会 source 用户 rc 文件并快照
        # alias——命令头还没执行，环境钩子先跑任意代码 / 同名 alias 已生效。
        # 这种环境下没有「可证明只读」的命令，与 BASH_FUNC_ 导出函数遮蔽同款
        # 处置（Codex review P1 ×2）。代价只是这些环境里所有命令都按 cwd 拍
        # 幂等快照，不是阻断。
        return False
    if _WRITEISH_SHELL_RE.search(command):
        return False
    for segment in _SHELL_CHAIN_SPLIT_RE.split(command):
        raw_tokens = segment.strip().split()
        tokens = [t for t in raw_tokens if not _ENV_ASSIGN_RE.match(t)]
        if not tokens:
            continue
        if len(tokens) != len(raw_tokens):
            # env 赋值能改写命令行为——`GIT_EXTERNAL_DIFF=rm git diff` 会对每
            # 个 diff 路径执行 rm（Codex review P1）。带 env 前缀的命令一律不
            # 判只读，按需要保护处理（代价只是多拍一张快照）。
            return False
        head = tokens[0]
        if "/" in head or "\\" in head:
            # 带路径的可执行文件（./ls、/tmp/cat）是项目 / 临时目录里的任意程
            # 序，与同名系统命令毫无关系——只对无路径分隔符的命令名做只读放行
            # （Codex review P1）。
            return False
        if head not in _READONLY_FIRST_TOKENS:
            return False
        if _shell_function_shadows(head):
            # 导出的 bash function（BASH_FUNC_ls%%=...）会取代同名系统命令，
            # 且随 env 一路传进 terminal 子进程——它在这里是可见的，命中即不
            # 判只读（Codex review P1）。会话内 alias / function 见函数注释。
            return False
        if env_type == "local" and not _resolves_to_system_binary(head):
            # 命令名会被 PATH 遮蔽：`/home/alice/bin/ls` 排在系统目录前时，
            # `ls` 执行的是任意 helper（Codex review P1）。只有解析到可信系
            # 统前缀的真实二进制才走只读快捷；ssh backend 下本机解析对远端
            # PATH 无意义，保持既有口径（远端文件系统本就不在快照覆盖面）。
            return False
    return True


def _managed_readonly_python_sources(text: str) -> set[str]:
    """Return active-profile Python entrypoints mounted read-only for terminal.

    Only the first script operand is exempted. Every later absolute argument is
    still a possible write target and remains covered by ancillary snapshots.
    """

    if (
        os.environ.get("HERMES_MANAGED_GATEWAY") != "1"
        or _terminal_env_type() != "local"
    ):
        return set()
    # 豁免根必须与执行侧只读挂载同源：挂载脚本读的是 backend run_env 里的
    # HERMES_HOME，其注入顺序是 per-profile 的 context override 覆盖进程 env
    # （_inject_hermes_home_env）。这里按同一顺序取值；get_hermes_home() 只
    # 作末位兜底——它在两者都缺时会退回平台默认目录，而那棵树从未被挂成只
    # 读，按它豁免就是免检洞。相对路径在挂载侧同样不生效，一律不豁免。
    hermes_home = ""
    try:
        from hermes_constants import get_hermes_home_override

        hermes_home = str(get_hermes_home_override() or "").strip()
    except Exception:
        hermes_home = ""
    if not hermes_home:
        try:
            hermes_home = _scoped_env("HERMES_HOME", "").strip()
        except _UnresolvableScope:
            return set()
    if not hermes_home:
        try:
            from hermes_constants import get_hermes_home

            hermes_home = str(get_hermes_home())
        except Exception:
            return set()
    if not hermes_home or not os.path.isabs(hermes_home):
        return set()
    try:
        from pathlib import Path

        skills_root = (Path(hermes_home) / "skills").resolve(strict=True)
    except (OSError, RuntimeError):
        return set()

    try:
        from tools.environments.local import _managed_python_skill_sources

        return {
            str(resolved)
            for _lexical, resolved in _managed_python_skill_sources(
                text, skills_root.parent
            )
        }
    except Exception:
        return set()


def _ancillary_abs_paths(
    text: str,
    primary: list[str],
    base_dir: str = "",
    *,
    readonly_sources: Optional[set[str]] = None,
) -> list[str]:
    """从命令 / 脚本文本里抽出**已存在**的写入目标，作为 cwd 之外的附加保护。

    覆盖绝对路径、home 前缀（~ / $HOME / ${HOME}，按当前 profile 的 home 展
    开）与 `../` 相对目标（锚到 base_dir，即主保护用的工作目录）。引号字面量
    优先（能带空格、更精确），裸 token 正则兜底；lexists 过滤截断碎片。
    提取前先把 `"$HOME"/x` 这类引号拼接归一成 `$HOME/x`，候选再过 brace 展开
    （Codex review P1 ×2）。
    """
    text = _QUOTED_HOME_RE.sub("$HOME", text or "")
    raw: list[str] = []
    for rx in _QUOTED_PATHISH_RES:
        raw.extend(m.group(1) for m in rx.finditer(text))
    raw.extend(m.group(0) for m in _ABS_PATH_TOKEN_RE.finditer(text))
    raw.extend(m.group(0) for m in _BARE_HOME_TOKEN_RE.finditer(text))
    raw.extend(m.group(0) for m in _BARE_PARENT_TOKEN_RE.finditer(text))
    # shell 会把相邻的 quoted / unquoted 段拼成同一个 word：
    # `rm "$HOME"/"Documents"/a.txt` 与 `rm $HOME/Documents/a.txt` 同义，逐段
    # 正则在引号处断开、抽不到整体。再按 shell word 语义切一遍，凡是路径形态的
    # word 整词入候选（Codex review P1）。解析不了（引号不配对 / 非 shell 文
    # 本）就只靠上面的正则，提取是加餐、正则仍在。
    try:
        base = os.path.normpath(base_dir) if base_dir else ""
        rbase = os.path.realpath(base) if base else ""

        def _escapes(real: str) -> bool:
            return real != rbase and not real.startswith(rbase.rstrip("/") + "/")

        def _static_cd_target(target: str, anchor: str) -> str:
            """静态解析 cd / pushd 的目标目录；解析不了返回空串。"""
            if target.startswith(("~", "$HOME", "${HOME}")):
                normalized = _normalize_pathish(target, anchor)
                return os.path.realpath(normalized) if normalized else ""
            if any(ch in target for ch in "$`"):
                return ""
            if target.startswith("/"):
                return os.path.realpath(os.path.normpath(target))
            if not anchor:
                return ""
            return os.path.realpath(os.path.normpath(os.path.join(anchor, target)))

        # 静态 `cd` / `pushd` 会改变后续命令的实际目录：`cd docs && rm a.txt`
        # 在 docs（可能是指向 ~/Documents 的 symlink）里删文件，而 docs 与
        # a.txt 都不含 `/`，逐词判定接不住（Codex review P1）。顺序扫描词流
        # 维护「当前目录」：cd 进的目录逃逸出主保护目录就整目录入候选，后续
        # 相对词也改锚它。目标静态解析不了（变量、`cd -`、flag 形态）就停掉
        # 跟踪——锚回 base 会给错误目录背书，宁可少提取也不错提取；变量间接
        # 本身属于 PRD §2 Phase 1 静态判定边界。
        cur = rbase
        cur_known = bool(base)
        dir_stack: list[str] = []
        at_head = True
        skip_next = False
        words = shlex.split(text, posix=True)
        for k, word in enumerate(words):
            if skip_next:
                skip_next = False
                continue
            if word and all(ch in _SHELL_PUNCT_CHARS for ch in word):
                at_head = True
                continue
            head_pos = at_head
            at_head = False
            if head_pos and _ENV_ASSIGN_RE.match(word):
                at_head = True  # env 赋值前缀不消耗段首位置
                continue
            if cur_known and head_pos and word in ("cd", "pushd", "popd"):
                if word == "popd":
                    if dir_stack:
                        cur = dir_stack.pop()
                    else:
                        cur_known = False
                    continue
                nxt = words[k + 1] if k + 1 < len(words) else ""
                if nxt and all(ch in _SHELL_PUNCT_CHARS for ch in nxt):
                    nxt = ""
                if word == "pushd":
                    dir_stack.append(cur)
                if not nxt:
                    cur = os.path.realpath(_subprocess_home())
                elif nxt.startswith("-"):
                    cur_known = False  # cd - / cd -P dir：会话状态或 flag 形态
                    continue
                else:
                    resolved = _static_cd_target(nxt, cur)
                    if not resolved:
                        cur_known = False
                        continue
                    cur = resolved
                    skip_next = True
                if _escapes(cur):
                    raw.append(cur)
                continue
            if word.startswith(("/", "~", "$HOME", "${HOME}", "../")):
                raw.append(word)
            elif cur_known and base and "/" in word and not word.startswith("-"):
                # 相对路径可能经内部 `..` 段（sub/../../x）**或目录 symlink**
                # （docs/a.txt，docs → ~/Documents）跳出主保护目录，前缀正则
                # 与 `../` 开头判定都接不住（Codex review P1 ×2）。锚到当前
                # 目录（无 cd 时即 base_dir）后按 realpath 判逃逸：仍在主目
                # 录内的交给 cwd 快照，跳出去的按真实目标整词入候选。
                real = os.path.realpath(os.path.normpath(os.path.join(cur, word)))
                if _escapes(real):
                    raw.append(real)
    except ValueError:
        pass
    candidates: list[str] = []
    for tok in raw:
        if "{" in tok and "," in tok:
            candidates.extend(_expand_braces(tok))
        else:
            candidates.append(tok)
    out: list[str] = []
    seen = set(primary)
    readonly = {os.path.realpath(path) for path in (readonly_sources or set())}
    for cand in candidates:
        p = _normalize_pathish(cand, base_dir)
        if p:
            # 容器会话里命令 / 脚本引用的是容器口径路径：先按 volume 表反解
            # 再做存在性过滤，否则挂载目录下的目标在 host 上 lexists 不到、
            # 整条附加保护静默漏掉（Codex review P1）。
            p = _map_container_path(p)
        if not p:
            continue
        # glob 目标（rm -rf /home/a/Doc*）按 shell 语义展开后逐个保护——不展
        # 开的话字面量 lexists 不到、整条命令的写入目标静默漏掉
        # （Codex review P1）。glob 只读磁盘、无副作用；父目录同时纳保，覆盖
        # 「展开结果被删掉后目录本身也变了」的情形。
        expanded = [p]
        if any(ch in p for ch in _GLOB_CHARS):
            # 流式取前 N 个匹配：`rm -rf /home/a/Pictures/*` 可能命中几万个条
            # 目，glob.glob + sorted 会先把整份列表读进内存再截断，在 2GB 端侧
            # 预算下 guard 自己就可能卡住或 OOM（HR1 / Codex review P1）。
            matches: list[str] = []
            try:
                for m in glob.iglob(p):
                    matches.append(m)
                    if len(matches) >= _MAX_ANCILLARY_PATHS:
                        break
            except Exception:
                matches = []
            prefix = re.split(r"[*?\[]", p, 1)[0]
            parent = prefix if prefix.endswith("/") else os.path.dirname(prefix)
            parent = parent.rstrip("/")
            # 父目录放前面：截断时它最该保住——整个目录的恢复点覆盖面最大。
            expanded = ([parent] if parent else []) + matches
        for item in expanded:
            if not item or item in seen or not os.path.lexists(item):
                continue
            if os.path.realpath(item) in readonly:
                continue
            seen.add(item)
            out.append(item)
            if len(out) >= _MAX_ANCILLARY_PATHS:
                return out
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
            return _map_container_path(_abs_path(cwd))
    except Exception as exc:
        logger.debug("zettlab snapshot guard: execute_code cwd resolution failed: %s", exc)
        # 判定不了就按 project 处置，保护回退 cwd——宁可多拍。
    return _terminal_workdir(arguments, task_id)


_TRUSTED_VIDEO_EDIT_SCRIPT_NAMES = frozenset({
    "preference_resolver.py",
    "workflow_state.py",
    "cloud_render_business.py",
    "normalize.py",
})
_TRUSTED_VIDEO_EDIT_WRITE_OPTIONS = frozenset({
    "--output",
    "--path",
    "--state-file",
    "--workflow-state",
})


def _trusted_video_edit_write_paths(
    command: str,
    arguments: dict[str, Any],
    task_id: str,
) -> Optional[list[str]]:
    """Return explicit write paths for an integrity-pinned video helper.

    ``terminal_tool`` intercepts these commands before a shell is spawned and
    runs the pinned helper source in the trusted worker.  Treating that direct
    Python command like arbitrary shell makes the generic guard snapshot the
    gateway launch cwd (``/root`` on-device), which is outside every agent
    scope and blocks even read-only ``plan-migrate`` calls.  Reuse the runner's
    exact parser/trust decision, then protect only the helper's explicit state,
    output, or cleanup paths.  A new out-of-scope path is still sent to the
    server and rejected; untrusted/wrapped Python keeps the generic cwd guard.

    ``None`` means this is not a trusted direct helper command.  ``[]`` means
    the trusted helper has no explicit filesystem mutation for this call.
    """
    if not any(name in command for name in _TRUSTED_VIDEO_EDIT_SCRIPT_NAMES):
        return None
    try:
        from tools.terminal_tool import _parse_video_edit_runtime_command

        parsed = _parse_video_edit_runtime_command(command)
    except Exception as exc:
        logger.debug(
            "zettlab snapshot guard: trusted video-edit parse unavailable: %s",
            exc,
        )
        return None
    if parsed is None or len(parsed.argv) < 2:
        return None

    script_name = os.path.basename(str(parsed.argv[1]))
    if script_name not in _TRUSTED_VIDEO_EDIT_SCRIPT_NAMES:
        return None

    base_dir = _terminal_workdir(arguments, task_id)
    paths: list[str] = []
    seen: set[str] = set()
    argv = [str(value) for value in parsed.argv[2:]]
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, inline_value = token.partition("=")
        if option not in _TRUSTED_VIDEO_EDIT_WRITE_OPTIONS:
            index += 1
            continue
        if separator:
            raw_path = inline_value
            index += 1
        elif index + 1 < len(argv):
            raw_path = argv[index + 1]
            index += 2
        else:
            # The helper parser will reject the missing value before writing.
            index += 1
            continue
        path = _normalize_pathish(raw_path, base_dir)
        if not path:
            # Relative helper write paths are invalid by contract.  Report the
            # resolved cwd target anyway so a future helper regression cannot
            # turn them into an unguarded write.
            path = _abs_path(os.path.join(base_dir, raw_path))
        if path and path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


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
        trusted_video_paths = _trusted_video_edit_write_paths(
            command,
            arguments,
            task_id,
        )
        if trusted_video_paths is not None:
            return trusted_video_paths
        if _command_is_probably_readonly(command):
            return []
        return [p for p in (_terminal_workdir(arguments, task_id),) if p]

    if tool_name == "execute_code":
        return [p for p in (_execute_code_workdir(arguments, task_id),) if p]

    if tool_name == "text_to_speech":
        # 默认输出走工具自己的生成目录，不涉用户文件；只有自定义 output_path
        # 需要保护（tts 会先删已存在的目标，Codex review P1）。command provider
        # 会把后缀改写成配置的 output_format 再删 / 写（_configured_command_
        # tts_output_path），最终落点可能不是参数原样——原路径 + 四种合法格式
        # （COMMAND_TTS_OUTPUT_FORMATS）的同名变体一并纳保（Codex review P1）。
        out = str(arguments.get("output_path") or "").strip()
        if not out:
            return []
        # TTS 在 Hermes **主进程**里 Path(output_path).expanduser() 落盘：相对
        # 路径锚进程 cwd、不走 session cwd，也不做容器 volume 反解——这里必须
        # 用同一套语义，否则在 Docker 会话 / 有 session cwd 时会给错路径建快照
        # （Codex review P1）。
        resolved = os.path.abspath(os.path.expanduser(out))
        if not resolved:
            return []
        paths = [resolved]
        stem, _ = os.path.splitext(resolved)
        for fmt in ("mp3", "wav", "ogg", "flac"):
            variant = f"{stem}.{fmt}"
            if variant != resolved and os.path.lexists(variant):
                paths.append(variant)
        return paths

    return []


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """拒绝跟随任何重定向。

    urllib 默认会带着非 Content-* 的请求头（包括 action token）跟到 3xx 指向
    的任意地址——loopback 端口被劫持或返回外部重定向时，token 会直接出设备
    （Codex review P1）。3xx 一律按 HTTPError 走 fail-closed。
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


# ProxyHandler({}) 显式禁用代理：build_opener 默认会按 HTTP(S)_PROXY 环境变量
# 装代理，而 Python 不会自动豁免 loopback——没配 NO_PROXY 时带 token 的请求会
# 先发去外部代理（Codex review P1）。internal 面只走本机直连。
_OPENER = urllib.request.build_opener(
    urllib.request.ProxyHandler({}),
    _NoRedirectHandler(),
)


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
        if exc.code == 403:
            # 这个端点上的 403 就是「路径不在 Agent 可写范围内」，按状态码识别
            # ——错误体解析不出 code 时也不能退化成「未知失败」而阻断整条命令
            # （Codex review P1 的既有语义）。
            return _error_payload(detail), "out_of_scope"
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


def _note_collapsed_turn_locked(key: str, turn_id: str) -> None:
    """把轮记进折叠容器 key 的在册集合（调用方须持锁）。"""
    turns = _collapsed_task_turns.get(key)
    if turns is None:
        while len(_collapsed_task_turns) >= _MAX_COLLAPSED_KEYS:
            _collapsed_task_turns.pop(next(iter(_collapsed_task_turns)), None)
        turns = _collapsed_task_turns.setdefault(key, {})
    turns[turn_id] = None


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
    # Docker/SSH 后端的 RPC 带的是**折叠后**的容器 task key（共享容器把
    # delegate 子任务折叠回 "default"），折叠 key 单独按「在册轮集合」建帐——
    # 不能直接覆盖进 _task_turns，并发轮会把彼此的映射踩掉、ensure 归错轮
    # （Codex review P1）。
    if turn and task:
        collapsed = ""
        try:
            from tools.terminal_tool import _resolve_container_task_id

            collapsed = str(_resolve_container_task_id(task) or "")
        except Exception:
            collapsed = ""
        with _lock:
            _note_task_turn_locked(task, turn)
            if collapsed and collapsed != task:
                _note_collapsed_turn_locked(collapsed, turn)

    if tool_name not in _GUARDED_TOOLS:
        return None

    paths = _paths_for(tool_name, arguments, task)
    # strict 模式 execute_code 没有可界定的主保护目录，但它只是换了子进程 cwd、
    # 没有文件系统隔离，脚本仍能写任意绝对路径：走 ancillary-only，对代码里
    # **已存在**的绝对路径尽力建恢复点（Codex review P1）。
    ancillary_only = not paths and tool_name == "execute_code"
    if not paths and not ancillary_only:
        return None

    started = time.monotonic()

    # 设备环境判定先于一切：非设备环境（CLI / 单测 / 未注入回调与 token 的部署）
    # 完全不介入，嵌套 dispatch 与 MCP bridge 不该在这里被 turn 契约挡住
    # （Codex review P1）。但 multiplex 下 profile scope 未绑定属于**判定不
    # 了**，不是「不是设备环境」——放行会让该 profile 的用户文件在无恢复点的
    # 情况下被改，所以 fail-closed（Codex review P1）。
    try:
        if not _local_server_origin() or not _scoped_env(_ACTION_TOKEN_ENV, "").strip():
            return None
    except _UnresolvableScope as exc:
        logger.warning("zettlab snapshot guard: profile scope unbound: %s", exc)
        return _blocked(
            "File protection is unavailable: this tool call is not bound to an "
            "agent profile, so the device cannot create a recovery point. The "
            "file was NOT modified.",
            outcome="unbound_scope", tool=tool_name, started=started,
        )

    if tool_name in _REMOTE_UNSAFE_TOOLS and _terminal_backend_is_remote():
        # ssh backend 的写入在**远端主机**执行——不止 terminal：file_tools 的
        # _get_file_ops 按 env_type=ssh 建 SSHEnvironment，execute_code 在非
        # local 时走 _execute_remote（Codex review P1 ×2）。本机快照护不住远端
        # 文件，按本机路径 ensure 出来的是一个看似成功的假恢复点；远端还可能就
        # 是设备自己（ssh 到 loopback），那更是绕开保护直改用户文件。设备形态
        # 只用 local / docker，这里 fail-closed；只读命令不受影响（在
        # _paths_for 已放行）。
        return _blocked(
            "File modifications on the ssh backend run on a remote host; the "
            "device cannot create a recovery point for remote files. Use a "
            "local/docker backend for file modifications. The operation was "
            "NOT executed.",
            outcome="remote_backend", tool=tool_name, started=started,
        )
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
    if tool_name == "terminal" and _shell_self_backgrounds(str(arguments.get("command") or "")):
        # shell 自行后台化（结尾 `&`、nohup / setsid 包裹）与 background=true
        # 同罪：finish 解 pin 时子进程可能仍在写（Codex review P1）。只对非只
        # 读命令生效——只读命令在 _paths_for 就被放行了。
        return _blocked(
            "Commands that background themselves ('&', nohup, setsid) are not "
            "covered by protection snapshots. Re-run the command in the "
            "foreground. The command was NOT executed.",
            outcome="background_write", tool=tool_name, started=started,
        )

    ambiguous_turn = False
    if not turn and task:
        with _lock:
            turn = _task_turns.get(task, "")
            if not turn:
                live = _collapsed_task_turns.get(task) or {}
                if len(live) == 1:
                    turn = next(iter(live))
                elif len(live) > 1:
                    ambiguous_turn = True
    if not turn:
        if ancillary_only:
            return None  # 加餐保护做不了幂等就不做，不阻断
        if ambiguous_turn:
            # 共享容器里多轮并发：折叠 key 分不清这次写入属于哪一轮，归错轮
            # 会随对方 finish 提前解 pin。不确定就不放行（Codex review P1）。
            return _blocked(
                "File protection snapshot unavailable: multiple concurrent turns "
                "share this sandbox, so this write cannot be attributed to a turn. "
                "Re-run after the other turn finishes. The file was NOT modified.",
                outcome="ambiguous_turn", tool=tool_name, started=started,
            )
        # 没有轮标识就无法做幂等，会把每次写入都变成一张新快照。这属于调度层
        # 契约被破坏，放行比拍一堆快照更糟，所以阻断。
        return _blocked(
            "File protection snapshot unavailable: missing turn id. The file was not modified.",
            outcome="missing_turn_id", tool=tool_name, started=started,
        )

    if ancillary_only:
        # strict execute_code 的唯一保护就是这次 ancillary ensure：建不起恢复
        # 点必须阻断（required=True，fail-closed），不能保护失败还放行写入
        # （Codex review P1）。
        return _ensure_ancillary(
            tool_name, arguments, turn, [], task=task, required=True, started=started)

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
        # scope 越界的阻断本身是条死胡同：output 目录可用时给模型指一条能走通
        # 的路；不可用时不加——别教一个必然失败的姿势。只对 terminal 说，
        # write_file / patch 没有 workdir 参数，对它们提这句同样是死胡同。
        hint = ""
        if (
            tool_name == "terminal"
            and _is_out_of_scope(data, err)
            and _managed_output_fallback()
        ):
            hint = (
                " If the working directory is outside the agent-writable scope, "
                "retry with workdir='agent_output' to run in the agent's "
                "writable output directory."
            )
        return _blocked(
            "Could not create a protection snapshot before modifying files"
            + (f" ({detail})" if detail else "")
            + ". The file was NOT modified. Tell the user the change did not happen; do not retry blindly."
            + hint,
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

    if tool_name in ("terminal", "execute_code"):
        # 主 cwd 之外的目标同样 fail-closed（required=True）：范围外路径逐个跳
        # 过，范围内的建不出恢复点就阻断——它们是货真价实的用户文件，只记日志
        # 放行等于让写入无恢复点发生（Codex review P1）。
        return _ensure_ancillary(
            tool_name, arguments, turn, paths, task=task, required=True, started=started)
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


def _is_out_of_scope(data: Optional[dict], err: str = "") -> bool:
    """报告一次 ensure 失败是否为范围外路径。

    两种识别：_post 把该端点上的 403 归一成 err="out_of_scope"（错误体解析不出
    code 时的兜底），或错误体里带 SNAPSHOT_AGENT_PATH_OUT_OF_SCOPE。
    """
    if err == "out_of_scope":
        return True
    return (
        isinstance(data, dict)
        and isinstance(data.get("_error"), dict)
        and str(data["_error"].get("code") or "") == "SNAPSHOT_AGENT_PATH_OUT_OF_SCOPE"
    )


def _ensure_ancillary(
    tool_name: str,
    arguments: dict[str, Any],
    turn: str,
    exclude: list[str],
    *,
    task: str = "",
    required: bool = False,
    started: float = 0.0,
) -> Optional[str]:
    """给命令 / 脚本文本里 cwd 之外的绝对路径目标建恢复点。

    terminal 下这是主保护（cwd）之外的加餐：任何失败只记日志不阻断。strict
    execute_code 下（required=True）这是**唯一**的保护：传输失败 / ready=false
    时必须阻断——否则恢复点没建成脚本仍会覆盖用户文件，违背 fail-closed 底线
    （Codex review P1）。范围外路径（403）在两种模式下都只跳过：脚本引用
    /etc 一类范围外文件多是只读，硬拒绝会把整条命令误杀（Codex review P1）；
    范围外的**写入**本就不在保护范围承诺内。成功后标记本轮 ensured，finish
    才会释放这些 operation 的 pin。
    """
    text = str(arguments.get("command") or arguments.get("code") or "")
    readonly_sources = (
        _managed_readonly_python_sources(text) if tool_name == "terminal" else set()
    )
    extras = _ancillary_abs_paths(
        text,
        exclude,
        base_dir=_terminal_workdir(arguments, task),
        readonly_sources=readonly_sources,
    )
    if not extras:
        return None
    data, err = _post(
        _ENSURE_PATH,
        {"turnId": turn, "paths": extras, "title": _title_for(extras)},
        _ENSURE_TIMEOUT,
    )
    if not err and isinstance(data, dict) and data.get("ready"):
        _log_unprotected(data, tool_name)
        with _lock:
            _state_for_locked(turn).ensured = True
        return None
    if err in ("unconfigured", "not_supported"):
        return None  # 不是设备环境 / 老 local-server：本机制不适用
    if not required:
        logger.info("zettlab snapshot guard: ancillary ensure skipped (%s)", err or "not_ready")
        return None
    if _is_out_of_scope(data, err):
        if len(extras) == 1:
            return None  # 单路径批次：批量结果就是它自己的结果，无需重试
        # 批量里混了范围外路径会整批 403：逐路径重试，范围外跳过，其余必须建成。
        ensured_any = False
        for p in extras:
            d2, e2 = _post(
                _ENSURE_PATH,
                {"turnId": turn, "paths": [p], "title": _title_for([p])},
                _ENSURE_TIMEOUT,
            )
            if not e2 and isinstance(d2, dict) and d2.get("ready"):
                ensured_any = True
                _log_unprotected(d2, tool_name)
                continue
            if e2 in ("unconfigured", "not_supported") or _is_out_of_scope(d2, e2):
                continue
            return _blocked(
                "Could not create a protection snapshot for the file paths this "
                "script modifies. The script was NOT executed. Tell the user the "
                "change did not happen; do not retry blindly.",
                outcome=f"ancillary_{e2 or 'not_ready'}", tool=tool_name, started=started,
            )
        if ensured_any:
            with _lock:
                _state_for_locked(turn).ensured = True
        return None
    return _blocked(
        "Could not create a protection snapshot for the file paths this "
        "script modifies. The script was NOT executed. Tell the user the "
        "change did not happen; do not retry blindly.",
        outcome=f"ancillary_{err or 'not_ready'}", tool=tool_name, started=started,
    )


def finish_turn(
    state: str = "completed",
    *,
    turn_id: str = "",
    error_code: str = "",
    error_stage: str = "",
) -> None:
    """一轮任务收尾：上报终态并解除该轮保护快照的 pin。

    ``turn_id`` 指明收哪一轮（agent 的 ``_current_turn_id``），**只做精确匹配**；
    不带 ``turn_id`` 一律不收。曾经的「唯一余轮」兜底并不安全：一个无写入的轮
    （拿不到 agent 实例的调用方）收尾时，若进程里唯一的状态恰好属于另一个还在
    写的轮，会把对方的 pin 提前解掉（Codex review P1）。真实的受保护写入必然
    有 agent 实例、拿得到 ``_current_turn_id``；空 id 宁可不收，代价只是等服务
    端 TTL 自愈。本轮没发生过保护快照时是纯 no-op。上报失败只记日志。
    """
    turn = str(turn_id or "").strip()
    with _lock:
        current: Optional[_TurnState] = None
        if turn:
            current = _states.pop(turn, None)
            # 轮结束即从折叠容器 key 的在册集合摘除：剩下的那一轮重新变得
            # 可归属（Codex review P1）。
            for key in list(_collapsed_task_turns):
                turns = _collapsed_task_turns[key]
                turns.pop(turn, None)
                if not turns:
                    _collapsed_task_turns.pop(key, None)
        elif _states:
            logger.info(
                "zettlab snapshot guard: finish without a turn id while %d turn(s) "
                "tracked; leaving their pins to the server-side TTL",
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
        _collapsed_task_turns.clear()
        _degraded_logged = False
        _nonloopback_logged = False
