#!/usr/bin/env bash
# Linux server inventory and a snapshot of resource usage.
# Usage: bash server_info.sh [output_file.txt]

set -u

HOST_NAME=$(hostname -s 2>/dev/null || hostname)
SAFE_HOST=$(printf '%s' "$HOST_NAME" | tr -c '[:alnum:]_.-' '_')
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
OUTPUT_FILE=${1:-"./server_info_${SAFE_HOST}_${TIMESTAMP}.txt"}

section() {
    printf '\n%s\n' '============================================================'
    printf '%s\n' "$1"
    printf '%s\n' '============================================================'
}

show_command() {
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 || printf '命令运行失败: %s\n' "$1"
    else
        printf '命令不可用: %s\n' "$1"
    fi
}

cpu_usage_snapshot() {
    if [[ ! -r /proc/stat ]]; then
        printf '无法读取 /proc/stat\n'
        return
    fi
    local cpu user nice system idle iowait irq softirq steal rest
    local total1 total2 idle1 idle2 diff_total diff_idle
    read -r cpu user nice system idle iowait irq softirq steal rest < /proc/stat
    total1=$((user + nice + system + idle + iowait + irq + softirq + steal))
    idle1=$((idle + iowait))
    sleep 1
    read -r cpu user nice system idle iowait irq softirq steal rest < /proc/stat
    total2=$((user + nice + system + idle + iowait + irq + softirq + steal))
    idle2=$((idle + iowait))
    diff_total=$((total2 - total1))
    diff_idle=$((idle2 - idle1))
    awk -v t="$diff_total" -v i="$diff_idle" 'BEGIN {
        if (t > 0) printf "CPU 使用率（1 秒采样）: %.2f%%\n", 100 * (t - i) / t
        else print "CPU 使用率: 无法计算"
    }'
}

{
    printf 'Linux 服务器基本信息报告\n'
    printf '生成时间: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z %z')"
    printf '主机名: %s\n' "$HOST_NAME"
    printf '执行用户: %s\n' "$(id -un)"

    section '1. 操作系统与内核'
    if [[ -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        printf '发行版: %s\n' "${PRETTY_NAME:-未知}"
        printf '系统 ID: %s\n' "${ID:-未知}"
        printf '版本 ID: %s\n' "${VERSION_ID:-未知}"
    else
        printf '未找到 /etc/os-release\n'
    fi
    printf '内核版本: %s\n' "$(uname -r)"
    printf '系统架构: %s\n' "$(uname -m)"
    printf '完整内核信息: %s\n' "$(uname -a)"
    if command -v systemd-detect-virt >/dev/null 2>&1; then
        printf '虚拟化环境: %s\n' "$(systemd-detect-virt 2>/dev/null || printf '未检测到')"
    fi

    section '2. CPU 信息'
    if command -v lscpu >/dev/null 2>&1; then
        LC_ALL=C lscpu
    elif [[ -r /proc/cpuinfo ]]; then
        grep -m 1 -E '^(model name|Hardware|Processor)' /proc/cpuinfo || true
        printf '逻辑处理器数: %s\n' "$(grep -c '^processor[[:space:]]*:' /proc/cpuinfo || true)"
    else
        printf 'CPU 信息不可读取\n'
    fi
    cpu_usage_snapshot

    section '3. 内存与交换空间'
    show_command free -h
    if [[ -r /proc/meminfo ]]; then
        awk '
            /^MemTotal:/ { total = $2 }
            /^MemAvailable:/ { available = $2; has_available = 1 }
            END {
                if (total > 0 && has_available)
                    printf "内存使用率（按 MemAvailable 计算）: %.2f%%\n", 100 * (total - available) / total
            }
        ' /proc/meminfo
    fi

    section '4. 磁盘、分区与文件系统'
    printf '%s\n' '[磁盘和分区]'
    show_command lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
    printf '\n%s\n' '[文件系统空间使用情况]'
    show_command df -hT
    printf '\n%s\n' '[inode 使用情况]'
    show_command df -ih

    section '5. 网络信息'
    printf '%s\n' '[网卡与 IP 地址]'
    if command -v ip >/dev/null 2>&1; then
        show_command ip -brief address
        printf '\n%s\n' '[路由表]'
        show_command ip route
    elif command -v hostname >/dev/null 2>&1; then
        show_command hostname -I
    else
        printf '无法获取网络信息\n'
    fi

    section '6. 运行时间和系统负载'
    show_command uptime
    if [[ -r /proc/loadavg ]]; then
        printf '1/5/15 分钟平均负载: '
        awk '{ print $1, $2, $3 }' /proc/loadavg
    fi
    printf '\n%s\n' '[登录用户]'
    show_command who

    section '7. 可选的 NVIDIA GPU 信息'
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv 2>&1 || true
    else
        printf '未检测到 nvidia-smi（可能未配置 NVIDIA GPU）\n'
    fi

    printf '\n报告结束\n'
} > "$OUTPUT_FILE" || {
    printf '错误：生成报告失败，请检查输出路径和写入权限：%s\n' "$OUTPUT_FILE" >&2
    exit 1
}

printf '服务器信息已保存到: %s\n' "$OUTPUT_FILE"
