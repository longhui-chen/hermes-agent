.PHONY: zpk-venv zpk-stage zpk-pack clean-zpk

ZPK_OUTPUT ?= build/zettlab-claw.zpk
ZPK_SRC_DIR := zpk/lib/hermes-agent
PYPI_INDEX_URL ?= https://pypi.tuna.tsinghua.edu.cn/simple/
# ZET-1399: `anthropic` left `[all]` on 2026-05-12 in favour of lazy install,
# but on ZPK devices the lazy-install ladder (uv -> pip -> ensurepip) is fully
# broken: no system uv, uv-created venvs ship without pip, and Debian splits
# ensurepip into the (absent) python3.11-venv package. Anything a device needs
# at runtime must therefore be baked into the ZPK venv here.
ZPK_INSTALL_SPEC ?= .[all,langfuse,anthropic]
ZPK_PACK_JOBS ?= 0
ZPK_VERBOSE ?= 0
ZPK_LOG_DIR ?= build
ZPK_UV_VENV_LOG ?= $(ZPK_LOG_DIR)/zpk-uv-venv.log
ZPK_UV_INSTALL_LOG ?= $(ZPK_LOG_DIR)/zpk-uv-install.log
ZPK_UV_FLAGS ?= --no-progress

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
	--exclude=./plugins/hermes-achievements \
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
		uv $(ZPK_UV_FLAGS) venv venv --python 3.11; \
	else \
		uv $(ZPK_UV_FLAGS) venv venv --python 3.11 >"$(ZPK_UV_VENV_LOG)" 2>&1 || { \
			echo "uv venv failed; showing last 120 log lines from $(ZPK_UV_VENV_LOG)"; \
			tail -n 120 "$(ZPK_UV_VENV_LOG)" 2>/dev/null || true; \
			exit 1; \
		}; \
	fi
	@echo "Installing zettlab-claw dependencies ($(ZPK_INSTALL_SPEC))..."
	@if [ "$(ZPK_VERBOSE)" = "1" ]; then \
		UV_LINK_MODE=copy uv $(ZPK_UV_FLAGS) pip install --python venv/bin/python --index-url "$(PYPI_INDEX_URL)" "$(ZPK_INSTALL_SPEC)"; \
	else \
		UV_LINK_MODE=copy uv $(ZPK_UV_FLAGS) pip install --python venv/bin/python --index-url "$(PYPI_INDEX_URL)" "$(ZPK_INSTALL_SPEC)" >"$(ZPK_UV_INSTALL_LOG)" 2>&1 || { \
			echo "uv pip install failed; showing last 160 log lines from $(ZPK_UV_INSTALL_LOG)"; \
			tail -n 160 "$(ZPK_UV_INSTALL_LOG)" 2>/dev/null || true; \
			exit 1; \
		}; \
	fi
	@echo "Seeding pip into ZPK venv (device-side lazy-install fallback)..."
	@if [ "$(ZPK_VERBOSE)" = "1" ]; then \
		UV_LINK_MODE=copy uv $(ZPK_UV_FLAGS) pip install --python venv/bin/python --index-url "$(PYPI_INDEX_URL)" pip; \
	else \
		UV_LINK_MODE=copy uv $(ZPK_UV_FLAGS) pip install --python venv/bin/python --index-url "$(PYPI_INDEX_URL)" pip >>"$(ZPK_UV_INSTALL_LOG)" 2>&1 || { \
			echo "uv pip install (pip seed) failed; showing last 160 log lines from $(ZPK_UV_INSTALL_LOG)"; \
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
		zpk/zpk-systemd.sh zpk/prepare-claw-service.sh zpk/init.d/start.sh zpk/init.d/stop.sh
	@echo "zettlab-claw ZPK payload staged at $(ZPK_SRC_DIR)"

zpk-pack: zpk-stage
	mkdir -p build
	python3 ../my-scripts/zpk-pack.py -p . -o "$(ZPK_OUTPUT)" --jobs "$(ZPK_PACK_JOBS)" --quiet

clean-zpk:
	rm -rf build zpk/.check-app "$(ZPK_SRC_DIR)"
