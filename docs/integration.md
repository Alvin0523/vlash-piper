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
**Line changed:** `version = "0.3.3"` → `version = "0.4.1"`

**Why:** VLaSH pins `lerobot==0.4.1` in its dependencies. Without this bump, pip/pixi
would ignore the local lerobot_piper and pull official lerobot 0.4.1 from PyPI (which
has no piper support), overwriting lerobot_piper.

---

### 2. `lerobot_piper/src/lerobot/utils/constants.py` (new file)
```python
from lerobot.constants import *  # noqa: F401, F403
```

**Why:** VLaSH imports `from lerobot.utils.constants import OBS_IMAGES, ACTION, OBS_STATE`.
In lerobot_piper (and all 0.3/0.4 forks), these constants live at `lerobot.constants`,
not `lerobot.utils.constants`. This file re-exports everything so the import path works.

---

### 3. `pixi.toml` (new file at `/home/wm/4901D/pixi.toml`)
Sets up a single pixi environment that installs:
- `lerobot_piper` as the `lerobot` package (editable, with feetech/smolvla/piper extras)
- `vlash` as editable
- `piper-sdk` and `wego-piper` (CAN hardware stack for piper arm)
- PyTorch with CUDA 12.8 (required for RTX 5080 / Blackwell)

**Why pixi at 4901D level:** Both lerobot_piper and vlash are separate local repos that
need to share one environment. Anchoring pixi here rather than inside either repo keeps
the dependency management neutral.

---

## How to Install
```bash
cd /home/wm/4901D
pixi install
pixi shell
```

## How to Run Inference
Copy `vlash/examples/inference/async.yaml` and change the robot section:
```yaml
robot:
  type: piper_follower
  port: can0          # your CAN interface (e.g. can0, can1)
  id: my_piper
  cameras:
    wrist:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
```

Then run:
```bash
vlash run your_inference_config.yaml
```