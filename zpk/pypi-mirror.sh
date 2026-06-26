#!/bin/bash
# PyPI 镜像源自动探测 + 写系统级 pip/uv 配置。
#
# 由 hermes-agent zpk install.sh / update.sh 在 OTA 安装/升级时调用：在
# 阿里云内网 / 阿里云公网 / 清华 / pypi.org 几个源里挑最先响应的写入
# /etc/pip.conf 和 /etc/uv/uv.toml，解决境内（尤其阿里云 ECS）从 pypi.org
# lazy-install hermes skill 依赖（uv pip install / pip install，见
# tools/lazy_deps.py）超时。
#
# 为什么写 /etc 全局：hermes 装依赖走 uv→pip 两级 ladder，uv 默认读不到 venv
# 私有配置，只认 /etc/uv/uv.toml（或 UV_CONFIG_FILE / cwd uv.toml），所以要让
# uv 那级也走镜像就得落到系统级。
#
# 安全（HR3/HR5）：公网源一律 https 强校验、绝不写 trusted-host。唯一例外是
# 阿里云 VPC 内网镜像 mirrors.cloud.aliyuncs.com —— 它实测最快（深圳 ~80ms vs
# 公网 https 172ms），但只有 http（https 证书 SAN 不匹配该域名，ECS 上 curl
# 实测 SSL err 60）。trusted-host / allow-insecure-host 只对 http 源写（pip 对
# http index 必须 trusted-host 才放行，https 不需要），而候选源里唯一的 http 就
# 是这个 VPC 内网镜像——它只在阿里云内网可解析、流量不出 VPC，信任边界等同于
# 信任 ECS 平台本身。这是 HR5 下为 ECS lazy-install 提速对该内网 http 源的显式 trade。
#
# fallback（HR2）：uv 靠默认 first-index 策略真 fallback 到 pypi.org（主源找不到
# 才查下一个、不跨源取最高版本）；pip 无 fallback 语义（extra-index-url 是双索引
# 取最高 = dependency-confusion，不用），只配单镜像 index-url。
# 写入：已配 index-url 则幂等跳过；写前 .bak 备份 + 原子替换（整文件替换非合并，
# 故 guard 保证仅在文件无 index-url 时才写，设备上这两文件通常本就不存在）。
setup_pypi_mirror() {
    local pip_conf="/etc/pip.conf"
    local uv_toml="/etc/uv/uv.toml"
    local pip_ok=false uv_ok=false

    # if 包裹避免 `grep && var=` 在 set -e 下 grep 未命中即退出
    if grep -qE '^[[:space:]]*index-url[[:space:]]*=' "$pip_conf" 2>/dev/null; then
        pip_ok=true
    fi
    if grep -q '\[\[index\]\]' "$uv_toml" 2>/dev/null; then
        uv_ok=true
    fi

    if $pip_ok && $uv_ok; then
        echo "  PyPI mirror already configured, skip"
        return 0
    fi

    # 候选源：阿里云内网(http,最快) + 阿里云公网/清华/pypi.org(https)
    local mirrors=(
        http://mirrors.cloud.aliyuncs.com/pypi/simple/pip/
        https://mirrors.aliyun.com/pypi/simple/pip/
        https://pypi.tuna.tsinghua.edu.cn/simple/pip/
        https://pypi.org/simple/pip/
    )

    local tmpdir
    tmpdir=$(mktemp -d)
    local lock_dir="$tmpdir/lock"
    # 并行探测，mkdir 原子锁选「最先响应者」（first responder wins）
    for url in "${mirrors[@]}"; do
        ( curl -sf --connect-timeout 3 --max-time 5 "$url" >/dev/null 2>&1 \
            && mkdir "$lock_dir" 2>/dev/null && echo "$url" > "$tmpdir/winner" ) &
    done
    wait

    local winner
    winner=$(cat "$tmpdir/winner" 2>/dev/null || true)
    rm -rf "$tmpdir"
    winner="${winner%pip/}"

    # pypi.org 最快、或全部探测失败 → 用默认源，不写任何配置
    if [ -z "$winner" ] || [ "$winner" = "https://pypi.org/simple/" ]; then
        echo "  pypi.org fastest or no mirror reachable, keep default"
        return 0
    fi

    echo "  fastest PyPI mirror: $winner"
    local fallback="https://pypi.org/simple/"  # 仅 uv 用作 first-index fallback

    # 只有 http 源才需要放开 TLS 校验（pip 对 http index 必须 trusted-host 才放行，
    # https 不需要、也绝不放开）。候选里唯一的 http 源就是阿里云 VPC 内网镜像。
    local pip_trust="" uv_insecure=""
    case "$winner" in
        http://*)
            local host=${winner#*://}; host=${host%%/*}
            pip_trust=$'\n'"trusted-host = $host"
            uv_insecure="allow-insecure-host = [\"$host\"]"$'\n\n'
            echo "  (http mirror $host: TLS check relaxed for this host only)"
            ;;
    esac

    if ! $pip_ok; then
        # pip 无 fallback 语义：extra-index-url 是「双索引取最高版本」(dependency
        # confusion)，不是 fallback。故只配单镜像 index-url——镜像是 PyPI 全量镜像，
        # 不需 pypi.org 兜底；真要兜底也只有 uv 那级能做(见下 first-index)。
        _atomic_write "$pip_conf" "[global]
index-url = $winner$pip_trust"
        echo "  wrote $pip_conf"
    fi

    if ! $uv_ok; then
        mkdir -p "$(dirname "$uv_toml")"
        # uv 默认 index-strategy=first-index：主源(default)命中就用、找不到才查下一个，
        # 按顺序、不跨源取最高版本 = 真 fallback(无 dependency confusion)，故保留
        # pypi.org 作 fallback。allow-insecure-host 是顶层 key，须在任何 [[index]] 之前。
        _atomic_write "$uv_toml" "${uv_insecure}[[index]]
url = \"$winner\"
default = true

[[index]]
url = \"$fallback\""
        echo "  wrote $uv_toml"
    fi
}

# 原子替换 + 备份：写前把已存在的 dest 备份到 .bak，再 tmp→mv 原子换上避免半写；
# 注意是整文件替换非合并，旧内容靠 .bak 留存可回滚。
_atomic_write() {
    local dest="$1" content="$2"
    local tmp
    tmp=$(mktemp "${dest}.XXXXXX") || return 1
    printf '%s\n' "$content" > "$tmp"
    chmod 0644 "$tmp"
    if [ -f "$dest" ]; then
        cp -a "$dest" "${dest}.bak" 2>/dev/null || true
    fi
    mv -f "$tmp" "$dest"
}
