import errno
import json
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tools import process_security


def _status_value(name: str, status: str) -> int:
    prefix = name + ":"
    line = next(line for line in status.splitlines() if line.startswith(prefix))
    value = line.split(":", 1)[1].strip().split()[0]
    return int(value, 16) if name.startswith("Cap") else int(value)


@pytest.mark.parametrize("unsupported_errno", [errno.EINVAL, errno.ENOSYS])
def test_hardening_accepts_unsupported_optional_ptracer_prctl(
    monkeypatch, unsupported_errno
):
    calls = []
    capability_clear = []

    def fake_prctl(option, arg2=0):
        calls.append((option, arg2))
        if option == process_security._PR_SET_PTRACER:
            raise OSError(unsupported_errno, "unsupported Yama prctl")
        if option == process_security._PR_GET_DUMPABLE:
            return 0
        if option == process_security._PR_GET_NO_NEW_PRIVS:
            return 1
        return 0

    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(process_security, "_prctl", fake_prctl)
    monkeypatch.setattr(
        process_security,
        "_clear_sensitive_capabilities",
        lambda: capability_clear.append(True),
    )

    assert process_security.harden_sensitive_process(
        no_new_privs=True,
        drop_ptrace=True,
    ) is True
    assert (process_security._PR_SET_DUMPABLE, 0) in calls
    assert (process_security._PR_SET_NO_NEW_PRIVS, 1) in calls
    assert (process_security._PR_GET_DUMPABLE, 0) in calls
    assert (process_security._PR_GET_NO_NEW_PRIVS, 0) in calls
    assert capability_clear == [True]


def test_hardening_rejects_unexpected_ptracer_prctl_error(monkeypatch):
    def fake_prctl(option, arg2=0):
        if option == process_security._PR_SET_PTRACER:
            raise OSError(errno.EPERM, "unexpected ptracer failure")
        return 0

    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(process_security, "_prctl", fake_prctl)

    assert process_security.harden_sensitive_process() is False


def test_drop_ptrace_does_not_enable_no_new_privs_when_not_requested(
    monkeypatch,
):
    calls = []
    capability_clear = []

    def fake_prctl(option, arg2=0):
        calls.append((option, arg2))
        if option == process_security._PR_GET_DUMPABLE:
            return 0
        if option == process_security._PR_GET_NO_NEW_PRIVS:
            raise AssertionError("no_new_privs must not be read")
        return 0

    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(process_security, "_prctl", fake_prctl)
    monkeypatch.setattr(
        process_security,
        "_clear_sensitive_capabilities",
        lambda: capability_clear.append(True),
    )

    assert process_security.harden_sensitive_process(
        no_new_privs=False,
        drop_ptrace=True,
    ) is True
    assert (process_security._PR_SET_DUMPABLE, 0) in calls
    assert (process_security._PR_SET_PTRACER, 0) in calls
    assert (process_security._PR_SET_NO_NEW_PRIVS, 1) not in calls
    assert (process_security._PR_GET_NO_NEW_PRIVS, 0) not in calls
    assert capability_clear == [True]


def test_unprivileged_elevation_capability_boundary_fails_closed_flow(
    monkeypatch,
):
    class FakeLibC:
        @staticmethod
        def capset(*_args):
            return 0

    def empty_capability_sets():
        header = process_security._CapHeader(
            version=process_security._LINUX_CAPABILITY_VERSION_3,
            pid=0,
        )
        return header, (process_security._CapData * 2)()

    def fake_prctl(option, arg2=0):
        if option == process_security._PR_CAPBSET_READ:
            return 1
        if option == process_security._PR_CAPBSET_DROP:
            raise OSError(errno.EPERM, "CAP_SETPCAP is unavailable")
        if option == process_security._PR_GET_DUMPABLE:
            return 0
        if option == process_security._PR_GET_NO_NEW_PRIVS:
            raise AssertionError("approved elevation must retain the exec contract")
        return 0

    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(process_security, "_libc", lambda: FakeLibC())
    monkeypatch.setattr(process_security, "_capability_sets", empty_capability_sets)
    monkeypatch.setattr(process_security, "_prctl", fake_prctl)
    monkeypatch.setattr(process_security.os, "geteuid", lambda: 1000)

    assert process_security.harden_sensitive_process(
        no_new_privs=False,
        drop_ptrace=True,
    ) is False


def test_bind_process_to_parent_sets_death_signal_and_closes_parent_exit_race(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(
        process_security,
        "_prctl",
        lambda option, arg2=0: calls.append((option, arg2)) or 0,
    )
    monkeypatch.setattr(process_security.os, "getppid", lambda: 1234)

    assert process_security.bind_process_to_parent(1234) is True
    assert calls == [(process_security._PR_SET_PDEATHSIG, signal.SIGKILL)]

    assert process_security.bind_process_to_parent(
        1234,
        death_signal=signal.SIGTERM,
    ) is True
    assert calls[-1] == (process_security._PR_SET_PDEATHSIG, signal.SIGTERM)

    monkeypatch.setattr(process_security.os, "getppid", lambda: 4321)
    assert process_security.bind_process_to_parent(1234) is False


def test_enable_child_subreaper_sets_and_verifies_linux_boundary(monkeypatch):
    calls = []
    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(
        process_security,
        "_prctl",
        lambda option, arg2=0: calls.append((option, arg2)) or 0,
    )
    monkeypatch.setattr(process_security, "_get_child_subreaper", lambda: 1)

    assert process_security.enable_child_subreaper() is True
    assert calls == [(process_security._PR_SET_CHILD_SUBREAPER, 1)]

    monkeypatch.setattr(process_security, "_get_child_subreaper", lambda: 0)
    assert process_security.enable_child_subreaper() is False


@pytest.mark.parametrize(
    "mandatory_failure",
    [
        "set_dumpable",
        "set_no_new_privs",
        "clear_ptrace_capability",
        "verify_dumpable",
        "verify_no_new_privs",
    ],
)
def test_hardening_still_fails_closed_for_mandatory_boundaries(
    monkeypatch, mandatory_failure
):
    def fake_prctl(option, arg2=0):
        if option == process_security._PR_SET_PTRACER:
            raise OSError(errno.EINVAL, "unsupported Yama prctl")
        if mandatory_failure == "set_dumpable" and option == process_security._PR_SET_DUMPABLE:
            raise OSError(errno.EPERM, "dumpable boundary failed")
        if mandatory_failure == "set_no_new_privs" and option == process_security._PR_SET_NO_NEW_PRIVS:
            raise OSError(errno.EPERM, "no_new_privs boundary failed")
        if option == process_security._PR_GET_DUMPABLE:
            return 1 if mandatory_failure == "verify_dumpable" else 0
        if option == process_security._PR_GET_NO_NEW_PRIVS:
            return 0 if mandatory_failure == "verify_no_new_privs" else 1
        return 0

    def fake_clear_ptrace_capability():
        if mandatory_failure == "clear_ptrace_capability":
            raise OSError(errno.EPERM, "capability boundary failed")

    monkeypatch.setattr(process_security, "_IS_LINUX", True)
    monkeypatch.setattr(process_security, "_prctl", fake_prctl)
    monkeypatch.setattr(
        process_security,
        "_clear_sensitive_capabilities",
        fake_clear_ptrace_capability,
    )

    assert process_security.harden_sensitive_process(
        no_new_privs=True,
        drop_ptrace=True,
    ) is False


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux prctl/capability boundary",
)
def test_managed_process_boundary_is_inherited_without_preexec_flow():
    code = textwrap.dedent(
        r"""
        import ctypes
        import json
        import subprocess
        import sys
        from pathlib import Path

        from tools.process_security import harden_sensitive_process

        assert harden_sensitive_process(no_new_privs=True, drop_ptrace=True)
        libc = ctypes.CDLL(None, use_errno=True)
        parent_dumpable = libc.prctl(3, 0, 0, 0, 0)
        child = subprocess.run(
            [
                sys.executable,
                "-c",
                "from pathlib import Path; print(Path('/proc/self/status').read_text())",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        print(json.dumps({"parent_dumpable": parent_dumpable, "child": child.stdout}))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    status = result["child"]
    assert result["parent_dumpable"] == 0
    assert _status_value("NoNewPrivs", status) == 1
    for capability in (19, 24):
        capability_mask = 1 << capability
        for name in ("CapEff", "CapPrm", "CapInh", "CapAmb"):
            assert _status_value(name, status) & capability_mask == 0
    if _status_value("Uid", status) == 0:
        for capability in (19, 24):
            assert _status_value("CapBnd", status) & (1 << capability) == 0


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux prctl/process_vm_readv boundary",
)
def test_model_child_cannot_read_sensitive_parent_memory_flow():
    code = textwrap.dedent(
        r"""
        import ctypes
        import json
        import os
        import subprocess
        import sys

        from tools.process_security import harden_sensitive_process

        secret = ctypes.create_string_buffer(b"scoped-secret")
        assert harden_sensitive_process(no_new_privs=True, drop_ptrace=True)
        attack = r'''\
        import ctypes
        import json
        import os
        import sys

        class IOVec(ctypes.Structure):
            _fields_ = [("base", ctypes.c_void_p), ("length", ctypes.c_size_t)]

        destination = ctypes.create_string_buffer(13)
        local = IOVec(ctypes.cast(destination, ctypes.c_void_p), len(destination))
        remote = IOVec(ctypes.c_void_p(int(sys.argv[2])), len(destination))
        libc = ctypes.CDLL(None, use_errno=True)
        ctypes.set_errno(0)
        attached = libc.ptrace(16, int(sys.argv[1]), 0, 0)
        ptrace_errno = ctypes.get_errno()
        ptrace_blocked = attached == -1 and ptrace_errno in {1, 13}
        if attached == 0:
            os.waitpid(int(sys.argv[1]), os.WUNTRACED)
            libc.ptrace(17, int(sys.argv[1]), 0, 0)
        ctypes.set_errno(0)
        read = libc.process_vm_readv(
            int(sys.argv[1]),
            ctypes.byref(local),
            1,
            ctypes.byref(remote),
            1,
            0,
        )
        vm_blocked = read == -1 and ctypes.get_errno() in {1, 13}
        try:
            with open(f"/proc/{sys.argv[1]}/mem", "rb", buffering=0):
                proc_mem_blocked = False
        except OSError:
            proc_mem_blocked = True
        print(json.dumps({
            "ptrace_blocked": ptrace_blocked,
            "vm_blocked": vm_blocked,
            "proc_mem_blocked": proc_mem_blocked,
        }))
        '''
        completed = subprocess.run(
            [sys.executable, "-c", attack, str(os.getpid()), str(ctypes.addressof(secret))],
            text=True,
            capture_output=True,
            check=True,
        )
        print(completed.stdout)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=True,
    )

    result = json.loads(completed.stdout)
    assert result == {
        "ptrace_blocked": True,
        "vm_blocked": True,
        "proc_mem_blocked": True,
    }


def test_systemd_service_keeps_root_ptrace_for_hidden_proc_audit_flow():
    service = (
        Path(__file__).resolve().parents[2]
        / "zpk"
        / "init.d"
        / "zettlab-claw.service"
    ).read_text()

    assert "NoNewPrivileges=true" in service
    assert "ProtectProc=invisible" in service
    assert "CapabilityBoundingSet=~CAP_SYS_ADMIN" in service
    assert "CapabilityBoundingSet=~CAP_SYS_PTRACE" not in service
