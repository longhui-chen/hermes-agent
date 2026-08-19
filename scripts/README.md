# Scripts

## `check_zpk_payload.py`

Smoke-tests the `hermes-agent` ZPK build environment after `make zpk-venv`
installs the fixed `ZPK_INSTALL_SPEC`
(`.[all,langfuse,anthropic,zpk-runtime]`).

The Makefile creates the venv with the overridable `UV` executable. It first
runs a locked, wheel-only dependency sync with `--no-install-project --no-build`,
then builds Hermes non-editably with the locked `setuptools` already in the venv
and `--no-build-isolation`. The `zpk-runtime` extra keeps both the device-side
`pip` fallback and build backend behind `uv.lock` artifact hashes; the build
never falls back to an unlocked install.

The script is called by `Makefile` during `make zpk-pack`. It verifies the
packaging contract only:

- `hermes --version` runs from the freshly created venv.
- Core runtime modules import successfully.
- `pyproject.toml` is parsed to resolve the current install extra.
- Eager runtime dependencies from the selected extra import successfully.
- Lazy provider dependencies, development tools, and intentionally excluded or
  quarantined packages are skipped.

Do not use this script as a replacement for `hermes doctor`. It must not probe
user config, credentials, external services, system packages, or optional
provider backends.

## `check_zpk_stage.py`

Checks the staged `hermes-agent` tree after the Makefile copies it into
`zpk/lib/hermes-agent`. It compares regular files under the source `plugins/`
and `venv/` runtime trees with the staged copies and fails when tar exclusions
remove nested runtime content. Deliberately excluded caches, bytecode,
development environments, egg metadata, ZPK stamps, and symlinks are ignored.
