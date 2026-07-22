#!/bin/bash
set -euo pipefail

install_systemd_services() {
    local app_root="$1"
    local app_base
    app_base=$(dirname "$app_root")

    local service_dir="$app_root/init.d"
    local systemd_dir="${SYSTEMD_DIR:-/lib/systemd/system}"

    if ! command -v systemctl >/dev/null 2>&1; then
        echo "systemctl is required" >&2
        return 1
    fi

    if [ ! -d "$service_dir" ]; then
        echo "missing service directory: $service_dir" >&2
        return 1
    fi

    mkdir -p "$systemd_dir"

    local installed=0
    local template service_name target tmp
    for template in "$service_dir"/*.service; do
        [ -f "$template" ] || continue
        service_name=$(basename "$template")
        target="$systemd_dir/$service_name"
        tmp="$target.tmp.$$"

        echo "  installing $service_name -> $target"
        sed -e "s|__APP_BASE__|$app_base|g" "$template" > "$tmp"
        chmod 0644 "$tmp"

        if [ -f "$target" ] && cmp -s "$tmp" "$target"; then
            rm -f "$tmp"
        else
            mv "$tmp" "$target"
        fi
        installed=$((installed + 1))
    done

    if [ "$installed" -eq 0 ]; then
        echo "no service templates found in $service_dir" >&2
        return 1
    fi

    systemctl daemon-reload

    for template in "$service_dir"/*.service; do
        [ -f "$template" ] || continue
        service_name=$(basename "$template")
        echo "  enabling $service_name"
        systemctl enable "$service_name"
    done
}
