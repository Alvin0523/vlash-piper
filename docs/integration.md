# Integration Notes

**Goal:** Run VLASH (MIT HAN Lab) with the Piper arm (WeGo Robotics `lerobot_piper`) on Jetson AGX Orin inside a single Pixi environment.

Four code changes were required. None touch VLASH's core logic or the robot driver — they are all compatibility shims.

---

## Why lerobot_piper

Three Piper-related forks exist; only one works with VLASH:

| Fork | Status |
|---|---|
| `lerobot_piper_community` (brucecai) | No Piper robot code despite the name |
| `lerobot-agilex` (official Agilex) | Uses ROS2 inference scripts — incompatible with VLASH's robot interface |
| `lerobot_piper` (WeGo Robotics) ✅ | Has `PiperFollower` robot class + `PiperMotorsBus` with CAN driver |

---

## 1. `lerobot_piper/pyproject.toml`

Three line changes:

| Field | Before | After | Why |
|---|---|---|---|
| `version` | `0.3.3` | `0.4.1` | VLASH pins `lerobot==0.4.1`. Without the bump, pixi resolves to official lerobot 0.4.1 from PyPI (no Piper support), silently overwriting lerobot_piper. |
| `rerun-sdk` | `>=0.21.0,<0.23.0` | `>=0.23.0` | Versions 0.21–0.22 have no `manylinux_2_28_aarch64` wheels — the package is simply absent for Jetson Orin's architecture. |
| `transformers` | `>=4.50.3,<4.52.0` | `>=4.50.3` | VLASH's git-pinned transformers is ≥4.52.0. The upper bound caused a hard dependency conflict. Removing it lets both coexist. |

---

## 2. `lerobot_piper/src/lerobot/utils/constants.py` (new file)

```python
from lerobot.constants import *  # noqa: F401, F403
```

VLASH imports `OBS_IMAGES`, `ACTION`, `OBS_STATE` from `lerobot.utils.constants`. In lerobot_piper these constants live at `lerobot.constants`, not `lerobot.utils.constants`. This one-liner shim bridges the import path without modifying either upstream package.

---

## 3. `lerobot_piper/src/lerobot/utils/visualization_utils.py`

Two changes:

**Added at the bottom of the file:**
```python
init_rerun = _init_rerun
```
VLASH imports `init_rerun` (public name) but lerobot_piper only defines `_init_rerun` (private name). This alias bridges the gap without renaming the private function.

**Renamed all 4 occurrences of `rr.Scalar(` → `rr.Scalars(`**  
rerun ≥ 0.23 renamed `rr.Scalar` to `rr.Scalars`. Without this fix, running with `display_data: true` crashes immediately with `AttributeError: module 'rerun' has no attribute 'Scalar'`.

---

## 4. `vlash/vlash/configs/run_config.py`

Two changes:

**Added import:**
```python
from lerobot.robots.piper_follower import PiperFollowerConfig  # noqa: F401
```
draccus (lerobot's config parser) populates its robot type registry from imports. Without this, `type: piper_follower` in any YAML raises `KeyError: 'piper_follower'` — the type is simply not registered.

**Wrapped reachy2 import in try/except:**
```python
try:
    from lerobot.robots.reachy2 import Reachy2RobotConfig  # noqa: F401
except ModuleNotFoundError:
    pass
```
lerobot_piper does not ship the reachy2 module. The bare import in VLASH crashed on startup even when reachy2 is never used, because the import runs at module load time before any config is parsed.

---

## 5. `pixi.toml` (new file)

A single Pixi environment that resolves all conflicts between lerobot_piper and VLASH:

- Pins Jetson-specific PyTorch wheels from `pypi.jetson-ai-lab.io/jp6/cu126`
- Installs lerobot_piper as editable as the `lerobot` package (satisfying VLASH's `lerobot==0.4.1` pin)
- Installs vlash as editable
- `[activation]` runs `scripts/activate.sh` which preloads the correct `libstdc++.so.6` to fix pyarrow `GLIBCXX` mismatch on aarch64
- Sets `platforms = ["linux-aarch64"]`

Key dependency pins:

```toml
[pypi-options]
extra-index-urls = ["https://pypi.jetson-ai-lab.io/jp6/cu126/+simple/"]
index-strategy = "unsafe-best-match"

[pypi-dependencies]
torch = "==2.11.0"
torchvision = "==0.26.0"
torchcodec = "==0.10.0"
triton = "==3.6.0"
lerobot = { path = "lerobot_piper", editable = true, extras = ["feetech", "smolvla", "intelrealsense"] }
vlash = { path = "vlash", editable = true }
piper-sdk = ">=0.4.1"
wego-piper = ">=0.0.2"
peft = "==0.18.0"
bitsandbytes = ">=0.48.2"
```
