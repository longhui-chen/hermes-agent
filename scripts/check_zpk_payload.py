#!/usr/bin/env python3
"""Smoke-test a hermes-agent ZPK build environment.

This is intentionally narrower than ``hermes doctor``.  It verifies the
packaging contract that a freshly staged ZPK must satisfy without probing user
configuration, provider credentials, system packages, or lazy backend deps.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELF_PACKAGE = "hermes-agent"
DEFAULT_INSTALL_SPEC = ".[all]"

CORE_IMPORTS = [
    "hermes_bootstrap",
    "hermes_cli.main",
    "tools.environments.local",
    "tools.process_registry",
]

# Only packages with a stable import module and expected eager runtime use
# should be listed here.  Lazy/provider/native/dev-only deps are skipped below.
PACKAGE_IMPORTS = {
    "agent-client-protocol": "acp",
    "aiohttp": "aiohttp",
    "anthropic": "anthropic",
    "croniter": "croniter",
    "fastapi": "fastapi",
    "fire": "fire",
    "google-api-python-client": "googleapiclient",
    "google-auth": "google.auth",
    "google-auth-httplib2": "google_auth_httplib2",
    "google-auth-oauthlib": "google_auth_oauthlib",
    "httpx": "httpx",
    "jinja2": "jinja2",
    "langfuse": "langfuse",
    "mcp": "mcp",
    "openai": "openai",
    "prompt-toolkit": "prompt_toolkit",
    "psutil": "psutil",
    "pyjwt": "jwt",
    "python-dotenv": "dotenv",
    "pyyaml": "yaml",
    "requests": "requests",
    "rich": "rich",
    "ruamel-yaml": "ruamel.yaml",
    "simple-term-menu": "simple_term_menu",
    "tenacity": "tenacity",
    "uvicorn": "uvicorn",
    "youtube-transcript-api": "youtube_transcript_api",
}

SKIP_PACKAGES = {
    # Quarantined / intentionally absent.
    "mistralai",
    # Lazy provider and backend deps.  NOTE: `anthropic` is NOT in this set —
    # it is baked into the ZPK via ZPK_INSTALL_SPEC (ZET-1399: the lazy-install
    # ladder is broken on devices), so when the install spec names its extra
    # the import must be verified, not skipped.
    "exa-py",
    "firecrawl-py",
    "parallel-web",
    "fal-client",
    "edge-tts",
    "modal",
    "daytona",
    "vercel",
    "hindsight-client",
    "python-telegram-bot",
    "discord-py",
    "brotlicffi",
    "slack-bolt",
    "slack-sdk",
    "qrcode",
    "mautrix",
    "markdown",
    "aiosqlite",
    "asyncpg",
    "aiohttp-socks",
    "elevenlabs",
    "faster-whisper",
    "sounddevice",
    "numpy",
    "honcho-ai",
    "boto3",
    "dingtalk-stream",
    "alibabacloud-dingtalk",
    "lark-oapi",
    # Dev/build tools do not prove runtime package health.
    "debugpy",
    "pytest",
    "pytest-asyncio",
    "pytest-xdist",
    "pytest-split",
    "ty",
    "ruff",
}


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _parse_requirement(req: str) -> tuple[str, list[str], str | None]:
    """Return (normalized package name, extras, marker) for a PEP 508-ish req."""
    raw_req, _, marker = req.partition(";")
    match = re.match(r"\s*([A-Za-z0-9_.-]+)\s*(?:\[([A-Za-z0-9_, .-]+)\])?", raw_req)
    if not match:
        return "", [], marker.strip() or None
    extras = []
    if match.group(2):
        extras = [_normalize_name(extra.strip()) for extra in match.group(2).split(",") if extra.strip()]
    return _normalize_name(match.group(1)), extras, marker.strip() or None


def _marker_applies(marker: str | None) -> bool:
    if not marker:
        return True

    # This packaging smoke test only needs the common platform markers used in
    # pyproject.toml.  Unknown markers are treated as applicable so dependency
    # drift is visible instead of silently ignored.
    platform = sys.platform
    for op, value in re.findall(r"sys_platform\s*(==|!=)\s*['\"]([^'\"]+)['\"]", marker):
        matched = platform == value
        if op == "==" and not matched:
            return False
        if op == "!=" and matched:
            return False
    return True


def _parse_install_spec(spec: str) -> list[str]:
    extras: list[str] = []
    for match in re.finditer(r"(?:\.|[A-Za-z0-9_.-]+)\[([A-Za-z0-9_, .-]+)\]", spec):
        extras.extend(_normalize_name(extra.strip()) for extra in match.group(1).split(",") if extra.strip())
    return extras


def _load_pyproject() -> dict:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _resolve_extra_deps(optional_deps: dict[str, list[str]], extras: list[str]) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    seen_extras: set[str] = set()

    def visit(extra: str) -> None:
        extra = _normalize_name(extra)
        if extra in seen_extras:
            return
        seen_extras.add(extra)
        for req in optional_deps.get(extra, []):
            package, nested_extras, marker = _parse_requirement(req)
            if not package or not _marker_applies(marker):
                continue
            if package == SELF_PACKAGE:
                for nested in nested_extras:
                    visit(nested)
                continue
            resolved.append((extra, package))

    for extra in extras:
        visit(extra)
    return resolved


def _hermes_command() -> Path | str:
    bin_dir = Path(sys.executable).resolve().parent
    candidates = [
        bin_dir / "hermes",
        bin_dir / "hermes.exe",
        PROJECT_ROOT / "venv" / "bin" / "hermes",
        PROJECT_ROOT / "venv" / "Scripts" / "hermes.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return "hermes"


def _check_command(command: list[str | Path]) -> str:
    proc = subprocess.run(
        [str(part) for part in command],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        output = proc.stdout.strip()
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, command))}\n{output}")
    return proc.stdout.strip()


def _check_imports(modules: list[str]) -> None:
    failures = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report smoke-test context.
            failures.append(f"{module}: {type(exc).__name__}: {exc}")
    if failures:
        details = "\n".join(f"  - {failure}" for failure in failures)
        raise RuntimeError(f"import smoke test failed:\n{details}")


def _modules_for_package(package: str) -> list[str]:
    mapped = PACKAGE_IMPORTS.get(package)
    if mapped:
        return [mapped]

    try:
        dist = importlib.metadata.distribution(package)
    except importlib.metadata.PackageNotFoundError:
        return []

    top_level = dist.read_text("top_level.txt") or ""
    modules = []
    for line in top_level.splitlines():
        module = line.strip()
        if module and re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$", module):
            modules.append(module)
    return modules


def _check_seed_policy() -> None:
    """Seed-policy guard: a packaged build that ships a seed policy must seed the
    curated bundled-skill set on a fresh profile.

    Resolves the policy / bundled skills exactly the way runtime does (via
    tools.skills_sync) and runs a real seed into a throwaway HERMES_HOME.
    Tolerant: an un-curated build (no policy) is skipped rather than failed, so
    this never breaks a payload that legitimately ships no seed policy.
    """
    code = (
        "import os, tempfile, sys\n"
        "os.environ['HERMES_HOME'] = tempfile.mkdtemp()\n"
        "from tools import skills_sync as s\n"
        "pol = s._read_seed_policy()\n"
        "if pol is None:\n"
        "    print('seed policy: none (un-curated build) — skipped'); sys.exit(0)\n"
        "bd = s._get_bundled_dir()\n"
        "if not bd.exists():\n"
        "    print('FAIL bundled skills dir missing: %s' % bd); sys.exit(3)\n"
        "r = s.sync_skills(quiet=True)\n"
        "want = len(pol['seed_set']); got = len(r.get('copied', []))\n"
        "if r.get('policy_error') or got != want:\n"
        "    print('FAIL seeded %d, policy wants %d (policy_error=%s)' % (got, want, r.get('policy_error'))); sys.exit(3)\n"
        "print('seed policy ok: %d curated skills seeded (policy + config resolved)' % got)\n"
    )
    out = _check_command([sys.executable, "-c", code])
    print(out.splitlines()[-1] if out else "seed policy check ran")


def main() -> int:
    install_spec = os.environ.get("ZPK_INSTALL_SPEC", DEFAULT_INSTALL_SPEC)
    extras = _parse_install_spec(install_spec)

    print(f"ZPK payload check: install spec {install_spec}")

    version_output = _check_command([_hermes_command(), "--version"])
    print(f"hermes --version ok: {version_output}")

    # ZET-1399: devices have no system uv and Debian ships ensurepip in the
    # absent python3.11-venv package, so the venv-seeded pip is the ONLY
    # working tier of the tools/lazy_deps.py install ladder on a ZPK device.
    # A payload without it silently bricks every lazy-installable backend.
    pip_output = _check_command([sys.executable, "-m", "pip", "--version"])
    print(f"pip seed ok: {pip_output}")

    _check_imports(CORE_IMPORTS)
    print(f"core imports ok: {', '.join(CORE_IMPORTS)}")

    _check_seed_policy()

    pyproject = _load_pyproject()
    optional_deps = pyproject.get("project", {}).get("optional-dependencies", {})
    deps = _resolve_extra_deps(optional_deps, extras)

    modules: list[str] = []
    skipped: list[str] = []
    seen_modules: set[str] = set()
    for extra, package in deps:
        if package in SKIP_PACKAGES:
            skipped.append(f"{package} ({extra}: skipped)")
            continue
        package_modules = _modules_for_package(package)
        if not package_modules:
            skipped.append(f"{package} ({extra}: no import mapping)")
            continue
        for module in package_modules:
            if module not in seen_modules:
                modules.append(module)
                seen_modules.add(module)

    if modules:
        _check_imports(modules)
        print(f"eager extra imports ok: {', '.join(modules)}")
    else:
        print("eager extra imports ok: none")

    if skipped:
        print(f"skipped extra packages: {', '.join(skipped)}")

    print("ZPK payload check ok")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - concise Makefile-facing error.
        print(f"ZPK payload check failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
