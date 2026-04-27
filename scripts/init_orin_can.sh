#!/bin/bash
# CAN interface setup for Jetson Orin
# Onboard mttcan: c310000.mttcan -> can0, c320000.mttcan -> can1
# External USB-CAN: can2, can3

set -e

echo "============================================================"
echo "🚀  CAN Interface Setup — Jetson Orin"
echo "============================================================"
echo ""
echo "📋 Interface plan:"
echo "   c310000.mttcan  →  can0  (onboard)"
echo "   c320000.mttcan  →  can1  (onboard)"
echo "   USB-CAN adapter →  can2  (external, bitrate=1000000)"
echo "   USB-CAN adapter →  can3  (external, bitrate=1000000)"
echo ""

# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "🔻 Step 1: Bring down all CAN interfaces"
echo "============================================================"

CAN_IFACES=$(ip -br link show type can 2>/dev/null | awk '{print $1}')

if [ -z "$CAN_IFACES" ]; then
    echo "   ℹ️  No active CAN interfaces found — nothing to bring down."
else
    for iface in $CAN_IFACES; do
        sudo ip link set "$iface" down 2>/dev/null || true
        echo "   ⬇️  Brought down: $iface"
    done
fi
echo ""

# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "🏷️  Step 2: Rename onboard mttcan interfaces"
echo "============================================================"

rename_mttcan() {
    local hw_addr=$1
    local target_name=$2
    local current_name=""

    echo ""
    echo "   🔍 Looking for interface with hardware address: $hw_addr ..."

    for iface in /sys/class/net/*/device; do
        local dev_path
        dev_path=$(readlink -f "$iface" 2>/dev/null)
        if echo "$dev_path" | grep -q "$hw_addr"; then
            current_name=$(basename "$(dirname "$iface")")
            break
        fi
    done

    if [ -z "$current_name" ]; then
        echo "   ⚠️  [WARN] No onboard CAN interface found for hardware address '$hw_addr'."
        echo "        → Check that the Jetson mttcan driver is loaded and the device is present."
        return 1
    fi

    echo "   ✅  Found interface: '$current_name' (matches $hw_addr)"

    if [ "$current_name" = "$target_name" ]; then
        echo "   ✔️  '$current_name' is already named '$target_name' — no rename needed."
    else
        echo "   ✏️  Renaming '$current_name' → '$target_name' ..."
        sudo ip link set "$current_name" down 2>/dev/null || true
        sudo ip link set "$current_name" name "$target_name"
        echo "   ✅  Successfully renamed '$current_name' → '$target_name'."
    fi
    return 0
}

rename_mttcan "c310000.mttcan" "can0"
rename_mttcan "c320000.mttcan" "can1"
echo ""

# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "🔌 Step 3: Configure and bring up external USB-CAN interfaces"
echo "============================================================"
echo ""

FAIL=0
for iface in can2 can3; do
    echo "   🔎 Checking for interface: $iface ..."
    if ! ip link show "$iface" &>/dev/null; then
        echo "   ❌ [ERROR] '$iface' not detected."
        echo "        → Make sure the USB-CAN adapter is plugged in and the driver is loaded."
        FAIL=1
        echo ""
        continue
    fi

    echo "   📡 Configuring '$iface' (type=can, bitrate=1000000) ..."
    sudo ip link set "$iface" type can bitrate 1000000
    sudo ip link set "$iface" up
    echo "   ✅ '$iface' is UP   (bitrate=1000000)"
    echo ""
done

# ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "📊 Current CAN interface status"
echo "============================================================"
echo ""
ip -br link show type can 2>/dev/null || echo "   ℹ️  No CAN interfaces found."
echo ""

# ─────────────────────────────────────────────────────────────
echo "============================================================"
if [ "$FAIL" -ne 0 ]; then
    echo "⚠️  Setup completed with errors — one or more external interfaces failed."
    echo "   → Review the [ERROR] messages above for details."
    echo "============================================================"
    exit 1
else
    echo "✅  Setup complete!"
    echo "   Onboard : can0 (c310000.mttcan)  |  can1 (c320000.mttcan)"
    echo "   External: can2 (USB-CAN)          |  can3 (USB-CAN)"
    echo "============================================================"
fi