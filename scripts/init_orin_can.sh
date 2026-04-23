#!/bin/bash
# CAN interface setup for Jetson Orin
# Onboard mttcan: c310000.mttcan -> can0, c320000.mttcan -> can1
# External USB-CAN: can2, can3

set -e

echo "=== Step 1: Bring down all CAN interfaces ==="
for iface in $(ip -br link show type can 2>/dev/null | awk '{print $1}'); do
    sudo ip link set "$iface" down 2>/dev/null || true
    echo "Down: $iface"
done

echo ""
echo "=== Step 2: Rename onboard CAN interfaces ==="

rename_mttcan() {
    local hw_addr=$1
    local target_name=$2
    local current_name=""

    for iface in /sys/class/net/*/device; do
        local dev_path=$(readlink -f "$iface" 2>/dev/null)
        if echo "$dev_path" | grep -q "$hw_addr"; then
            current_name=$(basename $(dirname "$iface"))
            break
        fi
    done

    if [ -z "$current_name" ]; then
        echo "[WARN] No onboard CAN interface found for $hw_addr"
        return 1
    fi

    if [ "$current_name" = "$target_name" ]; then
        echo "$current_name is already $target_name, no rename needed"
    else
        echo "Renaming $current_name -> $target_name"
        sudo ip link set "$current_name" down 2>/dev/null || true
        sudo ip link set "$current_name" name "$target_name"
    fi
    return 0
}

rename_mttcan "c310000.mttcan" "can0"
rename_mttcan "c320000.mttcan" "can1"

echo ""
echo "=== Step 3: Bring up external USB-CAN interfaces (can2, can3) ==="

FAIL=0
for iface in can2 can3; do
    if ! ip link show "$iface" &>/dev/null; then
        echo "[ERROR] $iface not detected — check USB-CAN adapter is plugged in and driver is loaded"
        FAIL=1
        continue
    fi
    echo "Configuring $iface ..."
    sudo ip link set "$iface" type can bitrate 1000000
    sudo ip link set "$iface" up
    echo "$iface up (bitrate=1000000)"
done

echo ""
echo "=== Current CAN interface status ==="
ip -br link show type can 2>/dev/null || true

echo ""
if [ "$FAIL" -ne 0 ]; then
    echo "=== Some external interfaces failed, check errors above ==="
    exit 1
else
    echo "=== Done: can0,can1 (onboard) can2,can3 (USB-CAN) ==="
fi