.PHONY: zpk-venv zpk-stage zpk-pack clean-zpk

UV ?= uv
ZPK_OUTPUT ?= build/zettlab-claw.zpk
ZPK_SRC_DIR := zpk/lib/hermes-agent
# ZET-1399: `anthropic` left `[all]` on 2026-05-12 in favour of lazy install,
# but on ZPK devices the lazy-install ladder (uv -> pip -> ensurepip) is fully
# broken: no system uv, uv-created venvs ship without pip, and Debian splits
# ensurepip into the (absent) python3.11-venv package. Anything a device needs
# at runtime must therefore be baked into the ZPK venv here.
override ZPK_INSTALL_SPEC := .[all,langfuse,anthropic,zpk-runtime]
override ZPK_UV_SYNC_EXTRAS := \
	--extra all \
	--extra langfuse \
	--extra anthropic \
	--extra zpk-runtime
ZPK_PACK_JOBS ?= 0
ZPK_VERBOSE ?= 0
ZPK_LOG_DIR ?= build
ZPK_UV_VENV_LOG ?= $(ZPK_LOG_DIR)/zpk-uv-venv.log
ZPK_UV_INSTALL_LOG ?= $(ZPK_LOG_DIR)/zpk-uv-install.log

# Do not let an inherited developer/CI uv configuration replace lockfile
# sources, constraints, project roots, Python selection, or hash verification.
# UV_CACHE_DIR is intentionally preserved so release builders can provide a
# reviewed cache without changing dependency resolution.
override ZPK_UV_ENV := env \
	-u UV_BUILD_CONSTRAINT \
	-u UV_COMPILE_BYTECODE \
	-u UV_CONFIG_FILE \
	-u UV_CONSTRAINT \
	-u UV_DEFAULT_INDEX \
	-u UV_DEV \
	-u UV_ENV_FILE \
	-u UV_EXCLUDE \
	-u UV_EXCLUDE_NEWER \
	-u UV_EXTRA_INDEX_URL \
	-u UV_FIND_LINKS \
	-u UV_FORK_STRATEGY \
	-u UV_FROZEN \
	-u UV_INDEX \
	-u UV_INDEX_STRATEGY \
	-u UV_INDEX_URL \
	-u UV_INSECURE_HOST \
	-u UV_INSECURE_NO_ZIP_VALIDATION \
	-u UV_ISOLATED \
	-u UV_LOCKED \
	-u UV_MANAGED_PYTHON \
	-u UV_NATIVE_TLS \
	-u UV_NO_BINARY \
	-u UV_NO_BINARY_PACKAGE \
	-u UV_NO_BUILD \
	-u UV_NO_BUILD_ISOLATION \
	-u UV_NO_BUILD_PACKAGE \
	-u UV_NO_CONFIG \
	-u UV_NO_DEFAULT_GROUPS \
	-u UV_NO_DEV \
	-u UV_NO_EDITABLE \
	-u UV_NO_ENV_FILE \
	-u UV_NO_GROUP \
	-u UV_NO_INDEX \
	-u UV_NO_INSTALL_LOCAL \
	-u UV_NO_INSTALL_PROJECT \
	-u UV_NO_INSTALL_WORKSPACE \
	-u UV_NO_MANAGED_PYTHON \
	-u UV_NO_PROJECT \
	-u UV_NO_SOURCES \
	-u UV_NO_SOURCES_PACKAGE \
	-u UV_NO_SYNC \
	-u UV_NO_VERIFY_HASHES \
	-u UV_OFFLINE \
	-u UV_OVERRIDE \
	-u UV_ONLY_INSTALL_LOCAL \
	-u UV_ONLY_INSTALL_PROJECT \
	-u UV_ONLY_INSTALL_WORKSPACE \
	-u UV_PRERELEASE \
	-u UV_PREVIEW \
	-u UV_PROJECT \
	-u UV_PROJECT_ENVIRONMENT \
	-u UV_PYTHON \
	-u UV_PYTHON_DOWNLOADS \
	-u UV_PYTHON_PREFERENCE \
	-u UV_PYTHON_SEARCH_PATH \
	-u UV_RESOLUTION \
	-u UV_REQUIRE_HASHES \
	-u UV_SHOW_RESOLUTION \
	-u UV_SKIP_WHEEL_FILENAME_CHECK \
	-u UV_SYSTEM_PYTHON \
	-u UV_SYSTEM_CERTS \
	-u UV_TORCH_BACKEND \
	-u UV_VENV_CLEAR \
	-u UV_VENV_RELOCATABLE \
	-u UV_VENV_SEED \
	-u UV_WORKING_DIR \
	UV_NO_CONFIG=1 \
	HERMES_ZPK_BUILD=1

ZPK_GLOBAL_EXCLUDES := \
	--exclude=.git \
	--exclude=.gk \
	--exclude=.worktrees \
	--exclude=__pycache__ \
	--exclude='*.pyc' \
	--exclude=.venv \
	--exclude=node_modules \
	--exclude=.pytest_cache \
	--exclude=.ruff_cache \
	--exclude=.mypy_cache \
	--exclude='*.egg-info'

# These are repository-root paths deliberately omitted from the product payload.
# Keep the ./ prefix:
# an unanchored tar exclude also removes same-named runtime directories inside
# plugins/ and venv/ (for example plugins/web and botocore/data).
ZPK_ROOT_EXCLUDES := \
	--exclude=./build \
	--exclude=./dist \
	--exclude=./data \
	--exclude=./logs \
	--exclude=./tmp \
	--exclude=./tests \
	--exclude=./docs \
	--exclude=./examples \
	--exclude=./website \
	--exclude=./web \
	--exclude=./ui-tui \
	--exclude=./nix \
	--exclude=./environments \
	--exclude=./packaging \
	--exclude=./wandb \
	--exclude=./testlogs \
	--exclude=./venv/.zpk-venv.stamp \
	--exclude=./venv/.zpk-install-spec \
	--exclude=./zpk

ZPK_EXCLUDES := $(ZPK_GLOBAL_EXCLUDES) $(ZPK_ROOT_EXCLUDES)

zpk-venv:
	@echo "Preparing zettlab-claw ZPK venv..."
	@rm -rf venv python-runtime
	@mkdir -p "$(ZPK_LOG_DIR)"
	@if [ "$(ZPK_VERBOSE)" = "1" ]; then \
		$(ZPK_UV_ENV) "$(UV)" --no-progress venv venv --python 3.11 --no-managed-python --no-python-downloads; \
	else \
		$(ZPK_UV_ENV) "$(UV)" --no-progress venv venv --python 3.11 --no-managed-python --no-python-downloads >"$(ZPK_UV_VENV_LOG)" 2>&1 || { \
			echo "uv venv failed; showing last 120 log lines from $(ZPK_UV_VENV_LOG)"; \
			tail -n 120 "$(ZPK_UV_VENV_LOG)" 2>/dev/null || true; \
			exit 1; \
		}; \
	fi
	@echo "Installing locked zettlab-claw dependencies ($(ZPK_INSTALL_SPEC))..."
	@if [ "$(ZPK_VERBOSE)" = "1" ]; then \
		$(ZPK_UV_ENV) UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$(CURDIR)/venv" \
			"$(UV)" --no-progress sync --locked --no-dev --no-editable --no-install-project --no-build $(ZPK_UV_SYNC_EXTRAS) && \
		$(ZPK_UV_ENV) UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$(CURDIR)/venv" \
			"$(UV)" --no-progress sync --locked --no-dev --no-editable --no-build-isolation \
				--reinstall-package hermes-agent $(ZPK_UV_SYNC_EXTRAS); \
	else \
		$(ZPK_UV_ENV) UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$(CURDIR)/venv" \
			"$(UV)" --no-progress sync --locked --no-dev --no-editable --no-install-project --no-build $(ZPK_UV_SYNC_EXTRAS) >"$(ZPK_UV_INSTALL_LOG)" 2>&1 || { \
			echo "uv locked dependency sync failed; showing last 160 log lines from $(ZPK_UV_INSTALL_LOG)"; \
			tail -n 160 "$(ZPK_UV_INSTALL_LOG)" 2>/dev/null || true; \
			exit 1; \
		}; \
		$(ZPK_UV_ENV) UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$(CURDIR)/venv" \
			"$(UV)" --no-progress sync --locked --no-dev --no-editable --no-build-isolation \
				--reinstall-package hermes-agent $(ZPK_UV_SYNC_EXTRAS) >>"$(ZPK_UV_INSTALL_LOG)" 2>&1 || { \
			echo "uv locked project sync failed; showing last 160 log lines from $(ZPK_UV_INSTALL_LOG)"; \
			tail -n 160 "$(ZPK_UV_INSTALL_LOG)" 2>/dev/null || true; \
			exit 1; \
		}; \
	fi
	@ZPK_INSTALL_SPEC="$(ZPK_INSTALL_SPEC)" venv/bin/python scripts/check_zpk_payload.py

zpk-stage: zpk-venv
	@echo "Staging zettlab-claw ZPK payload..."
	@test -x venv/bin/hermes
	@rm -rf "$(ZPK_SRC_DIR)"
	@mkdir -p "$(ZPK_SRC_DIR)"
	@tar $(ZPK_EXCLUDES) -cf - . | tar -xf - -C "$(ZPK_SRC_DIR)"
	@rm -f "$(ZPK_SRC_DIR)/venv/lib64"
	@python_bin=$$(readlink -f venv/bin/python); \
	rm -f "$(ZPK_SRC_DIR)/venv/bin/python" "$(ZPK_SRC_DIR)/venv/bin/python3" "$(ZPK_SRC_DIR)/venv/bin/python3.11"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python3"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python3.11"
	@chmod 0755 "$(ZPK_SRC_DIR)/venv/bin/python" "$(ZPK_SRC_DIR)/venv/bin/python3" "$(ZPK_SRC_DIR)/venv/bin/python3.11"
	@find "$(ZPK_SRC_DIR)" -type l -delete
	@python3 scripts/check_zpk_stage.py "$(ZPK_SRC_DIR)"
	@chmod 0755 zpk/install.sh zpk/update.sh zpk/uninstall.sh zpk/bin/hermes \
		zpk/libexec/hermes-secure-launcher.py zpk/zpk-systemd.sh \
		zpk/prepare-claw-service.sh zpk/init.d/start.sh zpk/init.d/stop.sh
	@python3 scripts/check_zpk_secrets.py zpk
	@echo "zettlab-claw ZPK payload staged at $(ZPK_SRC_DIR)"

zpk-pack: zpk-stage
	mkdir -p build
	python3 ../my-scripts/zpk-pack.py -p . -o "$(ZPK_OUTPUT)" --jobs "$(ZPK_PACK_JOBS)" --quiet
	@python3 scripts/check_zpk_secrets.py "$(ZPK_OUTPUT)" || { rm -f "$(ZPK_OUTPUT)"; exit 1; }

clean-zpk:
	rm -rf build zpk/.check-app "$(ZPK_SRC_DIR)"
