from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "zpk" / "pypi-mirror.sh"


def _uv_index_config(primary: str, fallback: str) -> str:
    command = (
        f"source {SCRIPT}; "
        f"_uv_index_config {primary!r} {fallback!r}"
    )
    return subprocess.check_output(["bash", "-c", command], text=True)


def test_uv_mirror_config_uses_selected_mirror_before_default_fallback() -> None:
    config = _uv_index_config(
        "http://mirrors.cloud.aliyuncs.com/pypi/simple/",
        "https://pypi.org/simple/",
    )

    assert config == (
        '[[index]]\n'
        'url = "http://mirrors.cloud.aliyuncs.com/pypi/simple/"\n'
        "\n"
        '[[index]]\n'
        'url = "https://pypi.org/simple/"\n'
        "default = true\n"
    )


def test_uv_mirror_config_omits_fallback_when_primary_is_default() -> None:
    config = _uv_index_config("https://pypi.org/simple/", "https://pypi.org/simple/")

    assert config == (
        '[[index]]\n'
        'url = "https://pypi.org/simple/"\n'
        "default = true\n"
    )


def test_pip_mirror_config_does_not_use_extra_index_url() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    code_lines = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )

    assert "extra-index-url" not in code_lines
