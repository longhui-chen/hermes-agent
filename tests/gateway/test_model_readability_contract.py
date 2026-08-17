"""模型文件可读性契约 —— Hermes 侧的门。

契约（`SPEC-MODEL-FILE-READABILITY-CONTRACT.md` §1.1）：

> 附件进入模型提示前，必须**在实际运行环境里** `open` + `regular-file` + `read`
> 成功；否则明确失败。⛔ 仅拼一句 `[file: path]` 不算可读契约。

⭐ 门的判据一律是**行为**（真的建文件、真的验、真的读），
⛔ 不按变量名 / 扩展名 / 目录前缀扫描 —— SPEC §6 明确禁止那种静态判据。
"""
from __future__ import annotations

import os
import stat

import pytest

from gateway.model_readability import (
    EVIDENCE_MOUNT_DECLARED,
    EVIDENCE_VERIFIED_READ,
    FAILURE_CODES,
    KNOWN_ENVIRONMENTS,
    USER_MESSAGES,
    verify_artifact_readable,
)


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """把 Hermes cache 根指到 tmp，返回 ``cache/documents`` 目录。"""
    import hermes_constants

    root = tmp_path / "hermes"
    (root / "cache" / "documents").mkdir(parents=True)
    (root / "cache" / "images").mkdir(parents=True)

    def _fake_dir(new_subpath, old_name=None):
        return root / new_subpath

    monkeypatch.setattr(hermes_constants, "get_hermes_dir", _fake_dir)
    # ⚠️ 模块内是 `from hermes_constants import get_hermes_dir`（函数内导入），
    # patch 源模块即可生效。
    monkeypatch.setenv("TERMINAL_ENV", "local")
    return root / "cache" / "documents"


def _write(d, name="report.pdf", data=b"%PDF-1.7 hello"):
    p = d / name
    p.write_bytes(data)
    return str(p)


# ───────────────────────── 正路径 ─────────────────────────

def test_readable_artifact_gets_a_receipt(cache_root):
    r = verify_artifact_readable(_write(cache_root))
    assert r.ok, f"正常文件没拿到回执:{r.failure_code} {r.failure_detail}"
    assert r.evidence == EVIDENCE_VERIFIED_READ
    # ⛔ ``sha256`` 必须是 None:我们只读头 4 KiB 证明可读,**没有**全文摘要。
    # 上一版断言 ``len(r.sha256) == 64`` —— 它钉的是「当时的实现顺手算了个
    # 摘要」,不是需求;而那次全文读正是把 gateway 事件循环堵死的那件事。
    # 把前缀摘要塞进叫 sha256 的字段冒充全文摘要,比没有这个字段更坏。
    assert r.sha256 is None, "只读了前缀却给出了一个自称 sha256 的值"
    assert r.size == len(b"%PDF-1.7 hello")
    assert r.user_message is None
    # 关键几步必须真的跑过，⛔ 不是"函数返回了 ok"
    for step in ("open_nofollow", "fd_matches_path", "regular_file", "read_ok"):
        assert step in r.checks, f"缺少 {step}:{r.checks}"


def test_local_backend_model_path_is_byte_identical(cache_root):
    """🔴 ⛔ 不许弄坏原来对的：local 后端下路径**逐字不变**。

    local 时 gateway 与模型同一个文件系统，本来就恒等；
    新增校验若改了这个字符串，所有现存的 local 部署（板端/云机）全会踩空。
    """
    src = _write(cache_root)
    r = verify_artifact_readable(src)
    assert r.model_path == src, f"local 后端下路径被改写了:{r.model_path!r} != {src!r}"


def test_unknown_file_type_is_not_rejected(cache_root):
    """⛔ 任意文件类型不退化（SPEC §7.4）。

    ⭐ 这条是**刻意不照抄** LS 的：它的 resolver 会 `Classify` 内容并可能
    `classify_failed`。Hermes 已有自己的 media_types，再分类一次会造出第二个
    真相源，而且会把无扩展名 / 未知 MIME 的合法文件挡掉。
    """
    src = _write(cache_root, name="weird.thing", data=b"\x01\x02\x03\xff\xfe")
    r = verify_artifact_readable(src)
    assert r.ok, f"未知类型被挡掉了:{r.failure_code}"


# ───────────────────────── LS 每一个分支的对照 ─────────────────────────

def test_empty_input_fails(cache_root):
    assert verify_artifact_readable("").ok is False


@pytest.mark.parametrize("bad", ["relative/path.pdf", "/tmp/has\x00nul"])
def test_malformed_path_fails(cache_root, bad):
    r = verify_artifact_readable(bad)
    assert r.ok is False and r.model_path is None


def test_missing_file_is_expired_not_generic(cache_root):
    r = verify_artifact_readable(str(cache_root / "gone.pdf"))
    assert r.failure_code == "attachment_expired", (
        f"文件不存在应当可区分,而不是笼统失败:{r.failure_code}")


def test_directory_is_not_a_regular_file(cache_root):
    d = cache_root / "subdir"
    d.mkdir()
    r = verify_artifact_readable(str(d))
    assert r.ok is False
    assert r.failure_code in FAILURE_CODES


def test_zero_length_file_is_expired(cache_root):
    r = verify_artifact_readable(_write(cache_root, name="empty.bin", data=b""))
    assert r.failure_code == "attachment_expired"


def test_symlink_is_refused_by_o_nofollow(cache_root):
    """LS 用 `openNoFollow`；这里必须同样挡住。"""
    real = cache_root / "real.pdf"
    real.write_bytes(b"data")
    link = cache_root / "link.pdf"
    os.symlink(real, link)
    r = verify_artifact_readable(str(link))
    assert r.ok is False, "符号链接被放行了 —— path-swap 可以指到 cache 外"


def test_file_outside_cache_root_is_still_accepted_when_readable(tmp_path, cache_root):
    """🔴 **刻意放宽**：cache 根之外但确实可读的 artifact 必须放行。

    我第一版照抄了 LS 的 ``outside_agent_uploads``（必须落在某个根内）。
    ⚠️ LS 那条成立是因为它的 artifact 只来自 App uploads 根 —— 封闭来源；
    Hermes 的 ``media_urls`` 由二十多个 adapter 产出，落点不止
    ``_CACHE_DIRS`` 那 8 个目录。照抄的后果实测是：cache 根尚未创建、
    或 artifact 落在别处时，**所有**附件被判不可读（两条既有门因此变红）。
    ⭐ **误杀合法附件比原缺陷严重得多** —— 原缺陷只是「模型拿到读不到的
    路径」，误杀是「所有附件都不给模型」。

    ⇒ 回到 SPEC §1.1 的不变量本身：它要求「``open`` + ``regular-file`` +
    ``read`` 成功」，**⛔ 没有「在某个根内」**。
    """
    outsider = tmp_path / "outside.pdf"
    outsider.write_bytes(b"real content")
    r = verify_artifact_readable(str(outsider))
    assert r.ok is True, (
        f"cache 外但确实可读的 artifact 被误杀 —— 会让所有非 cache 附件消失:"
        f"{r.failure_code}")
    assert r.model_path == str(outsider)


def test_oversized_file_is_refused(cache_root):
    src = _write(cache_root, name="big.bin", data=b"x" * 2048)
    r = verify_artifact_readable(src, max_bytes=1024)
    assert r.ok is False and r.failure_code == "attachment_transfer_failed"


def test_read_failure_is_caught_even_when_stat_looks_fine(cache_root, monkeypatch):
    """⭐ 这一条是**比先例更强**的那格：LS 只 `Seek`，这里真的 `read`。

    挂载还在、`stat` 正常，但 I/O 报错（远端 fs 掉线、稀疏洞）——
    只有真读才会红。
    """
    src = _write(cache_root)
    real_read = os.read

    def _boom(fd, n):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(os, "read", _boom)
    try:
        r = verify_artifact_readable(src)
    finally:
        monkeypatch.setattr(os, "read", real_read)
    assert r.ok is False, "read 报 EIO 却仍然发了回执"
    assert r.failure_code == "attachment_transfer_failed"


# ───────────────────────── 执行环境闭集 ─────────────────────────

def test_unknown_environment_is_refused(cache_root, monkeypatch):
    """⛔ 未知 backend 不许按「本地路径大概可用」放行（SPEC §3）。"""
    monkeypatch.setenv("TERMINAL_ENV", "some_new_sandbox")
    r = verify_artifact_readable(_write(cache_root))
    assert r.failure_code == "execution_environment_unsupported"


# ⚠️ modal 从这份「无回执」清单里移出去了:它**有**回执机制
# (``tools/environments/modal.py`` 已挂 cache 并持续同步),上一版把它
# 一并拒掉,等于让 Modal 会话升级后完全失去附件能力 —— 见下方专门那条。
@pytest.mark.parametrize("env", sorted(KNOWN_ENVIRONMENTS - {"local", "docker", "modal"}))
def test_backends_without_receipt_fail_instead_of_leaking_host_path(cache_root, monkeypatch, env):
    """🔴 族 A 本身：**没有回执就不许把主机路径塞给模型**。

    SPEC §3 逐条确认过 singularity / modal / daytona / vercel / ssh
    都没有把 artifact 的目标路径回写给提示 sink。
    ⇒ 明确失败，⛔ 不是"原样返回主机字符串"。
    """
    monkeypatch.setenv("TERMINAL_ENV", env)
    r = verify_artifact_readable(_write(cache_root))
    assert r.ok is False, f"{env} 下没有回执却发了通行证"
    assert r.model_path is None, f"{env} 下把主机路径泄漏给了模型:{r.model_path}"
    assert r.failure_code == "attachment_runtime_unavailable"


def test_modal_maps_the_cache_like_docker(cache_root, monkeypatch):
    """🔴 ⛔ 不许因为「backend 名字是 modal」就把附件一律判不可用。

    ``tools/environments/modal.py`` 已通过 ``iter_cache_files()`` 把 cache 挂进
    容器、并用 ``FileSyncManager`` 持续同步 ⇒ 目标路径与传输机制都存在。
    本模块(本批新增)此前按名字无条件拒 ⇒ Modal 会话升级后**完全失去图片与
    文档处理能力**。⭐ 证据等级仍如实标 ``mount_declared`` —— 我们没在容器内 open 过。
    """
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    import tools.credential_files as cf

    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": "/root/.hermes/cache/documents/report.pdf")

    r = verify_artifact_readable(_write(cache_root))
    assert r.ok, f"Modal 下可读附件被拒:{r.failure_code}"
    assert r.model_path == "/root/.hermes/cache/documents/report.pdf"
    assert r.evidence == EVIDENCE_MOUNT_DECLARED, "没在容器内 open 过却标成了 verified"


def test_modal_outside_the_cache_root_still_fails_explicitly(cache_root, monkeypatch, tmp_path):
    """🔴 必须保持不变:映射不出来时仍然**明确失败**,⛔ 不许退回主机路径。"""
    monkeypatch.setenv("TERMINAL_ENV", "modal")
    import tools.credential_files as cf

    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": None)

    outsider = tmp_path / "outside.pdf"
    outsider.write_bytes(b"data")
    r = verify_artifact_readable(str(outsider))
    assert r.ok is False
    assert r.model_path is None, f"把主机路径泄漏给了模型:{r.model_path}"
    assert r.failure_code == "attachment_runtime_unavailable"


def test_docker_keeps_working_via_mount_translation(cache_root, monkeypatch):
    """⛔ 不许弄坏原来对的：Docker 的 cache mount 转换必须继续可用。

    ⭐ 但证据等级要如实标成 `mount_declared` —— 我们**没有**在容器内
    open 过它。把它标成 `verified_read` 就是假装有证据。
    """
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    import tools.credential_files as cf

    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": "/root/.hermes/cache/documents/report.pdf")

    r = verify_artifact_readable(_write(cache_root))
    assert r.ok is True, f"Docker 下既有的可读路径被新校验拒了:{r.failure_code}"
    assert r.model_path == "/root/.hermes/cache/documents/report.pdf"
    assert r.evidence == EVIDENCE_MOUNT_DECLARED, (
        "把「只有挂载声明」标成了「已验证读取」—— 那是假装有证据")


def test_docker_without_a_mount_fails_closed(cache_root, monkeypatch):
    """Docker 下映射不出容器路径 ⇒ 挂载清单与 cache 闭集不一致 ⇒ 必须失败。"""
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    import tools.credential_files as cf

    monkeypatch.setattr(
        cf, "map_cache_path_to_container",
        lambda host, container_base="/root/.hermes": None)

    r = verify_artifact_readable(_write(cache_root))
    assert r.ok is False and r.failure_code == "attachment_runtime_unavailable"


# ───────────────────────── 失败必须可行动、且不泄漏 ─────────────────────────

def test_every_failure_code_has_an_actionable_message():
    """闭集自检：每一类失败都要有用户能照着做的下一步。"""
    for code in FAILURE_CODES:
        msg = USER_MESSAGES.get(code)
        assert msg, f"{code} 没有用户文案"
        assert any(k in msg for k in ("请", "重试", "重新")), (
            f"{code} 的文案没有可行动的下一步:{msg}")


def test_user_message_never_leaks_the_path_or_raw_error(cache_root):
    """⛔ 用户文案里不许出现绝对路径 / 原始 error / 内部字段。"""
    src = _write(cache_root, name="secret-report.pdf")
    os.chmod(src, 0o000)
    try:
        r = verify_artifact_readable(src)
    finally:
        os.chmod(src, 0o600)
    if r.ok:                      # root 跑测试时 chmod 挡不住
        pytest.skip("以 root 运行，权限用例不适用")
    msg = r.user_message or ""
    assert src not in msg and "secret-report" not in msg, f"路径泄漏进用户文案:{msg}"
    assert "Errno" not in msg and "Traceback" not in msg, f"原始错误泄漏:{msg}"
    # 但内部必须留得下来，供排查
    assert r.source_path == src, "内部连原始路径都没留 —— 出问题没法查"
    assert r.failure_detail, "内部没留原始错误摘要"


def test_failure_codes_are_a_closed_set():
    """⭐ 兜底码必须在集合里，否则未知失败会 assert 崩掉。"""
    assert "attachment_delivery_failed" in FAILURE_CODES
    assert set(USER_MESSAGES) == set(FAILURE_CODES), (
        "文案表与失败码闭集不一致 —— 会有失败拿不到文案")


# ---------------------------------------------------------------------------
# 有界探测(Codex PR #339 P1-2)
#
# 本函数的 7 个调用点**全部**在 async def 里(gateway/run.py 的 _handle_message /
# _handle_active_session_busy_message / _run_agent_inner ×5),而它是**同步**的
# ⇒ 读多久就把整个 gateway 事件循环堵多久,所有其它会话一起卡。
# 上一版读到 EOF、默认上限 512 MiB,只为算一个**没有任何生产消费者**的 sha256。
# 在 1C2G 无 swap 的板子上这属于整机级。
# ---------------------------------------------------------------------------


class TestReadIsBounded:
    """⛔ 不许读到 EOF —— 契约要的是「read 成功」,不是「读完」。"""

    def test_large_file_reads_at_most_one_probe_chunk(self, cache_root, monkeypatch):
        """拿一个**真的大文件**喂它,数它到底 read 了多少字节。

        ⭐ 判据是**实测读入总量**,⛔ 不是「源码里还有没有 while 循环」——
        后者是词法判据,换个写法就绕过去了。
        """
        from gateway import model_readability

        big = _write(cache_root, "big.bin", b"\xab" * (3 * 1024 * 1024))  # 3 MiB

        real_read = model_readability.os.read
        total = {"bytes": 0, "calls": 0}

        def counting_read(fd, n):
            chunk = real_read(fd, n)
            total["bytes"] += len(chunk)
            total["calls"] += 1
            return chunk

        monkeypatch.setattr(model_readability.os, "read", counting_read)
        r = model_readability.verify_artifact_readable(big)

        assert r.ok, f"大文件被误判不可读:{r.failure_code} {r.failure_detail}"
        assert "read_ok" in r.checks, "可读性这一步根本没跑 ⇒ 上面的上限断言是空的"
        assert total["calls"] == 1, f"read 被调了 {total['calls']} 次,应当只探测一次"
        assert total["bytes"] <= model_readability._PROBE_BYTES, (
            f"读入 {total['bytes']} 字节 > 上限 {model_readability._PROBE_BYTES} —— "
            f"事件循环会被堵住"
        )
        # 阳性对照:文件确实远大于上限,⛔ 否则上面的断言恒真
        assert r.size == 3 * 1024 * 1024 > model_readability._PROBE_BYTES

    def test_size_limit_still_rejects_oversized_files(self, cache_root):
        """🔴 必须保持不变:``max_bytes`` 判的是 ``st_size``,与读多少无关。"""
        from gateway.model_readability import verify_artifact_readable

        big = _write(cache_root, "huge.bin", b"x" * 8192)
        r = verify_artifact_readable(big, max_bytes=1024)

        assert not r.ok
        assert r.failure_code == "attachment_transfer_failed"
        assert "size_limit" not in r.checks

    def test_io_error_on_the_probe_is_still_reported(self, cache_root, monkeypatch):
        """🔴 必须保持不变:挂载在但 I/O 坏 ⇒ 仍然红(这是 read 检查的存在理由)。"""
        from gateway import model_readability

        src = _write(cache_root)

        def boom(fd, n):
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(model_readability.os, "read", boom)
        r = model_readability.verify_artifact_readable(src)

        assert not r.ok
        assert r.failure_code == "attachment_transfer_failed"

    def test_short_file_still_passes(self, cache_root):
        """🔴 必须保持不变:小文件(远小于一个探测块)判定逐字不变。"""
        from gateway.model_readability import verify_artifact_readable

        r = verify_artifact_readable(_write(cache_root, "tiny.txt", b"hi"))
        assert r.ok and r.size == 2 and "read_ok" in r.checks
