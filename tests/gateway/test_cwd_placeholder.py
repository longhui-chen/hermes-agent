"""Unit tests for gateway.cwd_placeholder.resolve_placeholder_terminal_cwd."""

from gateway.cwd_placeholder import resolve_placeholder_terminal_cwd


class TestResolvePlaceholderTerminalCwd:
    def test_local_placeholder_uses_messaging_cwd(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="local",
            messaging_cwd="/home/user/project",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) == "/home/user/project"


    def test_docker_placeholder_mount_off_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="docker",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) is None


    def test_docker_placeholder_mount_on_without_messaging_cwd_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd=".",
            terminal_backend="docker",
            messaging_cwd=None,
            docker_mount_cwd_to_workspace=True,
            home_fallback="/home/user",
        ) is None

    def test_ssh_placeholder_unset(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd="cwd",
            terminal_backend="ssh",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) is None

    def test_explicit_configured_cwd_passthrough(self):
        assert resolve_placeholder_terminal_cwd(
            configured_cwd="/explicit/path",
            terminal_backend="docker",
            messaging_cwd="/home/user",
            docker_mount_cwd_to_workspace=False,
            home_fallback="/home/user",
        ) == "/explicit/path"


class TestManagedGatewayLocalPlaceholder:
    """受管网关下的 local + placeholder 不许落到 Path.home()。

    板上 multiplex 以 root 跑，home_fallback 就是 /root：模型 shell 进不去、也在
    所有快照目标之外。把它种进 TERMINAL_CWD 会让下游全都当成刻意指定的锚点——
    文件工具在 _authoritative_workspace_root() 就停下，够不到平台 output 兜底，
    每一次相对写入都变成 scope 越界阻断。
    """

    def _resolve(self, **kwargs):
        base = dict(
            configured_cwd="",
            terminal_backend="local",
            messaging_cwd=None,
            docker_mount_cwd_to_workspace=False,
            home_fallback="/root",
        )
        base.update(kwargs)
        return resolve_placeholder_terminal_cwd(**base)

    def test_managed_local_placeholder_leaves_cwd_unset(self):
        assert self._resolve(managed_gateway=True) is None
        assert self._resolve(configured_cwd=".", managed_gateway=True) is None

    def test_unmanaged_local_placeholder_still_uses_home(self):
        assert self._resolve() == "/root"
        assert self._resolve(managed_gateway=False) == "/root"

    def test_managed_local_still_honours_explicit_values(self):
        # 显式配置和 MESSAGING_CWD 是有人明确指定的锚点，不受影响。
        assert self._resolve(configured_cwd="/volume1/work", managed_gateway=True) == "/volume1/work"
        assert self._resolve(messaging_cwd="/volume1/work", managed_gateway=True) == "/volume1/work"
