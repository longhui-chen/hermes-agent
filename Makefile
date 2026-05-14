.PHONY: zpk-venv zpk-stage zpk-pack clean-zpk

ZPK_OUTPUT ?= build/zettlab-claws.zpk
ZPK_SRC_DIR := zpk/lib/hermes-agent
PYPI_INDEX_URL ?= https://pypi.tuna.tsinghua.edu.cn/simple/

ZPK_EXCLUDES := \
	--exclude=.git \
	--exclude=.gk \
	--exclude=.worktrees \
	--exclude=__pycache__ \
	--exclude='*.pyc' \
	--exclude=.venv \
	--exclude=node_modules \
	--exclude=.pytest_cache \
	--exclude=build \
	--exclude=dist \
	--exclude=data \
	--exclude=logs \
	--exclude=tmp \
	--exclude=zpk

zpk-venv:
	rm -rf venv python-runtime
	uv venv venv --python 3.11
	UV_LINK_MODE=copy uv pip install --python venv/bin/python --index-url "$(PYPI_INDEX_URL)" .
	test -x venv/bin/hermes
	venv/bin/hermes --version

zpk-stage:
	test -x venv/bin/hermes
	rm -rf "$(ZPK_SRC_DIR)"
	mkdir -p "$(ZPK_SRC_DIR)"
	tar $(ZPK_EXCLUDES) -cf - . | tar -xf - -C "$(ZPK_SRC_DIR)"
	rm -f "$(ZPK_SRC_DIR)/venv/lib64"
	python_bin=$$(readlink -f venv/bin/python); \
	rm -f "$(ZPK_SRC_DIR)/venv/bin/python" "$(ZPK_SRC_DIR)/venv/bin/python3" "$(ZPK_SRC_DIR)/venv/bin/python3.11"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python3"; \
	cp "$$python_bin" "$(ZPK_SRC_DIR)/venv/bin/python3.11"
	chmod 0755 "$(ZPK_SRC_DIR)/venv/bin/python" "$(ZPK_SRC_DIR)/venv/bin/python3" "$(ZPK_SRC_DIR)/venv/bin/python3.11"
	find "$(ZPK_SRC_DIR)" -type l -delete
	chmod 0755 zpk/install.sh zpk/update.sh zpk/uninstall.sh zpk/bin/hermes

zpk-pack: zpk-stage
	mkdir -p build
	python3 ../my-scripts/zpk-pack.py -p . -o "$(ZPK_OUTPUT)"

clean-zpk:
	rm -rf build zpk/.check-app "$(ZPK_SRC_DIR)"
