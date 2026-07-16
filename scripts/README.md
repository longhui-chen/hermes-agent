# Scripts

## `check_zpk_payload.py`

Smoke-tests the `hermes-agent` ZPK build environment after `make zpk-venv`
installs `ZPK_INSTALL_SPEC` (default: `.[all]`).

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
development environments, egg metadata, ZPK stamps, symlinks, and the
non-product `plugins/hermes-achievements/` plugin are ignored.
