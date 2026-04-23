# VLaSH + Piper Arm Integration Changes

## Goal
Use VLaSH (MIT HAN Lab) with the Piper arm via lerobot_piper (WeGo Robotics fork).

## Why lerobot_piper
- `lerobot_piper_community` (brucecai fork) — no piper code at all despite the name
- `lerobot-agilex` (official Agilex) — uses ROS2 inference scripts, incompatible with VLaSH's robot interface
- `lerobot_piper` (WeGo Robotics) — has `piper_follower` robot class + `PiperMotorsBus` with CAN driver

---

## Changes Made

### 1. `lerobot_piper/pyproject.toml`
**Lines changed:**
- `version = "0.3.3"` → `version = "0.4.1"`
- `rerun-sdk>=0.21.0,<0.23.0` → `rerun-sdk>=0.23.0`
- `transformers>=4.50.3,<4.52.0` → `transformers>=4.50.3` (removed upper bound)

**Why version bump:** VLaSH pins `lerobot==0.4.1`. Without this bump pixi pulls official
lerobot 0.4.1 from PyPI (no piper support), overwriting lerobot_piper.

**Why rerun-sdk:** Versions 0.21–0.22 have no `manylinux_2_28_aarch64` wheels for Jetson Orin.

**Why transformers:** smolvla pinned `<4.52.0` but VLaSH's git-pinned transformers is ≥4.52.0,
causing a conflict. Removing the upper bound lets both coexist.

---

### 2. `lerobot_piper/src/lerobot/utils/constants.py` (new file)
```python
from lerobot.constants import *  # noqa: F401, F403
```

**Why:** VLaSH imports `from lerobot.utils.constants import OBS_IMAGES, ACTION, OBS_STATE`.
In lerobot_piper these constants live at `lerobot.constants`, not `lerobot.utils.constants`.

---

### 3. `lerobot_piper/src/lerobot/utils/visualization_utils.py`
**Lines added/changed:**
- Added at bottom: `init_rerun = _init_rerun`
- Changed all `rr.Scalar(` → `rr.Scalars(` (4 occurrences)

**Why init_rerun alias:** VLaSH imports `init_rerun` (public) but lerobot_piper only defines
`_init_rerun` (private). Alias bridges the gap without modifying VLaSH.

**Why rr.Scalars:** rerun ≥0.23 renamed `rr.Scalar` to `rr.Scalars`. Without this fix
`--display_data=true` crashes immediately.

---

### 4. `vlash/vlash/configs/run_config.py`
**Lines added:**
```python
from lerobot.robots.piper_follower import PiperFollowerConfig  # noqa: F401
```
And wrapped the reachy2 import in try/except:
```python
try:
    from lerobot.robots.reachy2 import Reachy2RobotConfig  # noqa: F401
except ModuleNotFoundError:
    pass
```

**Why PiperFollowerConfig import:** draccus (lerobot's config parser) requires all robot
config subclasses to be imported before YAML parsing. Without this import, `type: piper_follower`
in any YAML raises `KeyError: 'piper_follower'`.

**Why reachy2 try/except:** lerobot_piper does not ship the reachy2 robot module.
The bare import in VLaSH crashes on startup even when not using reachy2.

---

### 5. `pixi.toml` (new file)
Sets up a single pixi environment that installs both repos and resolves all conflicts.

```toml
[workspace]
name = "vlash-piper"
version = "0.1.0"
channels = ["conda-forge"]
platforms = ["linux-64", "linux-aarch64"]

[dependencies]
python = "3.10.*"
ffmpeg = "==7.1.1"
cmake = ">=3.29"
wrapt = "<2"

[pypi-options]
extra-index-urls = ["https://download.pytorch.org/whl/cu128"]
index-strategy = "unsafe-best-match"

[pypi-dependencies]
torch = ">=2.6.0"
torchvision = ">=0.21.0"
torchcodec = ">=0.3.0"
lerobot = { path = "lerobot_piper", editable = true, extras = ["feetech", "smolvla", "intelrealsense"] }
vlash = { path = "vlash", editable = true }
piper-sdk = ">=0.4.2"
wego-piper = ">=0.0.2"
peft = "==0.18.0"
bitsandbytes = ">=0.48.2"
flask = ">=3.1.2"

[tasks]
can = "bash scripts/can.sh"
teleop = "bash scripts/teleop.sh"
record = "bash scripts/record.sh"
```

**Key decisions:**
- `cmake = ">=3.29"` in conda deps — PyTorch index only has cmake 3.25, conda has 3.29+
- `index-strategy = "unsafe-best-match"` — lets uv search all indexes for best version
- `wrapt = "<2"` — piper-sdk needs python-can which needs wrapt<2, but conda defaults to wrapt 2.x
- `piper-sdk = ">=0.4.2"` — 0.4.1 was yanked (buggy arm_status data)
- `intelrealsense` extra — needed for Intel RealSense D435I cameras on Jetson Orin

---

### 6. `scripts/can.sh` (new file)
CAN interface setup for Jetson Orin:
- Step 1: bring down all existing CAN interfaces
- Step 2: rename onboard mttcan (`c310000.mttcan` → `can0`, `c320000.mttcan` → `can1`)
- Step 3: bring up USB-CAN adapters (`can2`, `can3`) at 1 Mbps

**Why custom script:** WeGo's original script maps USB port IDs specific to their dev machine.
Orin's onboard CAN (mttcan) needs different handling from USB-CAN adapters.

---

### 7. `scripts/teleop.sh` (new file)
```bash
LEADER_PORT=can3
FOLLOWER_PORT=can2
lerobot-teleoperate \
  --robot.type=piper_follower --robot.port="$FOLLOWER_PORT" --robot.id=follower \
  --teleop.type=piper_leader  --teleop.port="$LEADER_PORT"  --teleop.id=leader
```
Both arms on USB-CAN (can2=follower, can3=leader). Swap ports if arms are on wrong side.

---

### 8. `scripts/record.sh` (new file)
Records demonstrations to a local dataset. Key flags:
- `--robot.cameras` uses `intelrealsense` type with `serial_number_or_name` field
- `--dataset.push_to_hub=false` saves locally to `./data`
- `--display_data=false` required when running headless over SSH (no X server)

Camera serials: `348122073292` (USB 2.1, wrist, 15fps max) and `048322071496` (USB 3.2, overhead, 30fps).

---

## Hardware Setup (Jetson Orin)

- **CAN interfaces:** can0/can1 = onboard mttcan, can2/can3 = USB-CAN adapters (Piper arms)
- **Cameras:** Two Intel RealSense D435I. USB 3.2 camera supports 30fps; USB 2.1 limited to 15fps
- **Rerun viewer:** aarch64 wheel does not bundle the viewer binary — install separately via
  `cargo install rerun-cli --version 0.31.3 --locked`

---

## How to Install
```bash
cd ~/vlash_piper   # or wherever the repo lives
pixi install
pixi shell
```

## How to Run
```bash
pixi run can       # bring up CAN interfaces
pixi run teleop    # teleoperate (test arms first)
pixi run record    # record a dataset
```

## How to Run Inference (after fine-tuning)
Edit `vlash/examples/inference/async.yaml`:
```yaml
robot:
  type: piper_follower
  port: can2
  id: follower
  cameras:
    wrist:
      type: intelrealsense
      serial_number_or_name: "348122073292"
      width: 640
      height: 480
      fps: 15
    overhead:
      type: intelrealsense
      serial_number_or_name: "048322071496"
      width: 640
      height: 480
      fps: 30

policy:
  path: <path to fine-tuned checkpoint>
  device: cuda

single_task: "your task description"
display_data: false
```

Then run:
```bash
vlash run your_inference_config.yaml
```