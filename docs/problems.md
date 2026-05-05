---
icon: lucide/bug
---

# Problems, Diagnostics & Fixes Log

**Project:** VLASH + Piper on Jetson AGX Orin  
**Repository:** [Alvin0523/vlash-piper](https://github.com/Alvin0523/vlash-piper)  
**Last Updated:** 2026-05-04

This document is the running log of every problem hit during integration, recording, training, and inference — with root cause analysis, evidence, and resolution for each.

---

## Table of Contents

1. [USB Camera Frame Drop — Episodes 020 & 041 (test4)](#1-usb-camera-frame-drop--episodes-020--041-test4)
2. [Config Mismatch: state_cond / Camera Names](#2-config-mismatch-state_cond--camera-names)
3. [Integration Issues Fixed](#3-integration-issues-fixed)
4. [Dependency Risks on aarch64](#4-dependency-risks-on-aarch64)

---

## 1. USB Camera Frame Drop — Episodes 020 & 041 (test4)

**Status:** Root cause confirmed — episodes 020 & 041 are invalid for training.  
**Report Date:** 2026-04-28  
**Dataset:** `data/test4/`

### Summary

Episodes 020 and 041 experienced severe video frame loss during recording. Action data is complete (911 and 935 frames respectively), but video frames were reduced to only ~18% of expected (165 and 169 frames vs. 911 and 935 expected). The files are structurally valid MP4 files — the loss happened at the camera capture layer, not during encoding or transfer.

Critically: **this is a systemic issue, not isolated to these two episodes.** Even "normal" episodes in the same dataset show only ~6 fps video capture vs. 30 fps requested, meaning all episodes are affected to varying degrees.

### Frame Count Analysis

**Episode 020:**

| Metric | Value |
|---|---|
| Expected frames | 911 @ 30 fps = 30.37 s |
| Action frames recorded | 911 ✓ (complete) |
| Video frames in mdat | 165 ✗ (18.1% of expected) |
| Actual capture rate | 5.44 fps |
| mdat — up camera | 770,361 bytes (165 frames @ 4,669 bytes/frame) |
| mdat — wrist camera | 3,059,170 bytes (165 frames @ 18,540 bytes/frame) |

**Episode 041:**

| Metric | Value |
|---|---|
| Expected frames | 935 @ 30 fps = 31.17 s |
| Action frames recorded | 935 ✓ (complete) |
| Video frames in mdat | 169 ✗ (18.1% of expected) |
| Actual capture rate | 5.43 fps |
| mdat — up camera | 1,021,986 bytes (169 frames @ 6,047 bytes/frame) |
| mdat — wrist camera | 4,831,820 bytes (169 frames @ 28,591 bytes/frame) |

**"Normal" episodes (systemic baseline):**

| Episode | Expected | Actual | Duration | Actual FPS |
|---|---|---|---|---|
| 000 | 594 | 120 | 19.77 s | 6.07 fps |
| 005 | 514 | 107 | 17.10 s | 6.26 fps |
| 010 | 563 | 115 | 18.73 s | 6.14 fps |
| **020** | **911** | **165** | **30.33 s** | **5.44 fps** |
| **041** | **935** | **169** | **31.13 s** | **5.43 fps** |

All episodes consistently capture ~6 fps instead of 30 fps. This is a systemic hardware/configuration issue, not an isolated software bug or transfer error.

### MP4 File Integrity

All four affected files were confirmed structurally valid:

| File | Status | Structure |
|---|---|---|
| ep020_up | ✅ Complete | mdat (770,361 bytes) + moov (2,429 bytes) — complete |
| ep020_wrist | ✅ Complete | mdat (3,059,170 bytes) + moov (6,261 bytes) — complete |
| ep041_up | ✅ Complete | mdat (1,021,986 bytes) + moov (4,837 bytes) — complete |
| ep041_wrist | ✅ Complete | mdat (4,831,820 bytes) + moov (6,401 bytes) — complete |

All files: ftyp box present, mdat complete, moov complete, no truncation, no orphaned data, no partial data sections.

### File Size Paradox

Episodes 020 and 041 are **larger** than average, not smaller:

| Episode | Total Size | % of Average |
|---|---|---|
| 020 | 3.66 MB | 144% |
| 041 | 5.59 MB | 220% (+4.36 SD above mean) |
| Average | 2.54 MB | 100% |

Explanation: larger files with fewer frames = higher bytes-per-frame, consistent with the encoder writing higher-quality keyframes when frame rate drops (fewer frames = more encoding budget per frame).

### Why Transfer Corruption Can Be Ruled Out

| Indicator | Expected if Transfer Corruption | Actual |
|---|---|---|
| File truncation | Yes — files cut off midway | ✅ No — all files complete |
| Missing moov box | Yes — metadata lost first | ✅ No — moov present and complete |
| Box size mismatches | Yes — declared > actual | ✅ No — all sizes match perfectly |
| Incomplete mdat | Yes — video data cut short | ✅ No — mdat properly closed |
| Partial files (.tmp) | Yes — visible transfer remnants | ✅ No — none found |
| Files smaller than average | Usually yes | ✅ No — files are larger than average |

Zero out of six transfer corruption indicators present. Transfer was clean.

### Root Cause: Camera Capture Layer Dropped Frames

```
Robot Control:   ████████████████████ (911/911 frames)  ✓ Complete
Video Capture:   ███░░░░░░░░░░░░░░░░░ (165/911 frames)  ✗ Dropped 82%
```

The robot control loop and action recording ran at the intended rate throughout the episode. The camera capture thread dropped 82% of frames in real-time. Both processes operated independently — the control loop does not gate on camera frames.

**The mdat boxes contain exactly the number of frames in the episode stats** — proving frames were dropped during capture, not during encoding or transfer.

### Likely Root Causes (Ranked by Probability)

1. **USB Bandwidth Saturation** — Dual RealSense cameras (USB 3.2 + USB 2.1) streaming 30 fps may exceed available USB bus bandwidth. Both cameras sharing bandwidth with other USB devices compounds this.
2. **AV1 Encoding Bottleneck** — AV1 is computationally expensive. The Jetson CPU likely cannot sustain dual 30 fps real-time AV1 encoding while also running the robot control loop.
3. **System Resource Contention** — Robot control + dual video encoding exceeds Jetson Orin real-time budget for this workload.
4. **librealsense Driver Buffer Overflow** — The librealsense driver's internal buffer overflows at sustained 30 fps, silently dropping frames at the driver level.

### Modification Timestamps (Confirming Normal Recording, Not Transfer Issue)

**Episode 020:**
- up camera: 2026-04-27 16:25:01
- wrist camera: 2026-04-27 16:24:55
- Spread: 6 seconds (normal for dual-camera recording start delay)

**Episode 041:**
- up camera: 2026-04-27 17:01:01
- wrist camera: 2026-04-27 17:00:48
- Spread: 13 seconds (acceptable for dual-camera recording)

### Immediate Actions

- ✅ **Do NOT re-transfer these episodes** — they transferred completely; re-transfer will produce identical files
- ❌ **Do NOT use episodes 020 or 041 for training** — insufficient video frame count
- Mark episodes 020 and 041 as invalid in dataset metadata

### Fix Options

| Option | Description | Trade-off |
|---|---|---|
| **A: Lower FPS** | Set cameras to `fps: 15` or resolution to 320×240 | Reduced visual quality; must retrain models at same fps |
| **B: Separate USB bandwidth** | Use PCIe USB expansion card; give each camera its own USB controller | Hardware cost |
| **C: Switch codec** | Use H.264 instead of AV1 | Lower CPU encoding cost; slightly lower quality |
| **D: Post-record encoding** | Record raw frames during teleoperation, encode after episode ends | Disk space; more complex pipeline |

**Recommended immediate fix:** Test `fps: 15` for the wrist camera (USB 2.1 port) and verify the frame count improves.

### Verification Command

```bash
# Check frame count for any episode
ffprobe -v quiet -show_streams \
  data/<dataset>/videos/chunk-000/observation.images.wrist_episode_000000.mp4 \
  | grep nb_frames

# Check USB bandwidth
lsusb -v | grep -A 5 "RealSense"

# Monitor during recording
nvidia-smi dmon -s pucvmet
```

### Code-Level Hardening (Recommended)

Add real-time frame drop detection to the recording script:

```python
import time
import logging

episode_start = time.time()
frame_count = 0
expected_fps = 30

# ... in capture loop:
frame_count += 1
elapsed = time.time() - episode_start
actual_fps = frame_count / elapsed if elapsed > 0 else 0
if elapsed > 2.0 and actual_fps < expected_fps * 0.8:
    logging.warning(
        f"Frame drop detected: {actual_fps:.1f} fps vs {expected_fps} fps expected"
    )
```

---

## 2. Config Mismatch: state_cond / Camera Names

**Status:** Ongoing risk — must check manually before each inference run.  
**Consequence:** Silent failure — arm trembles, freezes, or moves erratically with no error message.

### Problem

Different training runs used different model configurations. Attempting to run inference with a YAML config that doesn't match the model's `config.json` produces no error — the model runs with wrong inputs and produces garbage actions.

### Model Variants

| Model | `state_cond` | Camera names in config.json | Correct inference camera names |
|---|---|---|---|
| test3 | `true` | `side` | Must name camera `side` in inference YAML |
| test6 | `false` | `wrist`, `up` | Must name cameras `wrist` and `up`; no joint state fed |
| test8 | `true` | `wrist`, `up` | Must name cameras `wrist` and `up`; joint state required |

### Failure Modes

| Mismatch type | Symptom |
|---|---|
| Wrong camera name | Camera not found → crash, or wrong camera silently fed to wrong model input |
| `state_cond` mismatch | Model receives joint state it wasn't trained with → arm trembles and freezes |
| `n_action_steps` > `chunk_size` | Index out of bounds error at runtime |
| `n_action_steps` < `chunk_size` | Works, but only first N actions of each chunk executed |

### How to Check Before Running

```bash
# Inspect model config before running inference
python3 -c "
import json
with open('models/<model>/pretrained_model/config.json') as f:
    cfg = json.load(f)
print('state_cond:', cfg.get('state_cond'))
print('input_features:', list(cfg.get('input_features', {}).keys()))
print('chunk_size:', cfg.get('chunk_size'))
"
```

Then verify the inference YAML:
- Camera names in YAML match keys in `input_features`
- `n_action_steps` ≤ `chunk_size`
- `state_cond` in training config matches whether joint state is included in the inference robot config

---

## 3. Integration Issues Fixed

These were all resolved during initial integration. Documented here for reference.

### 3.1 `KeyError: 'piper_follower'` on VLASH startup

**Cause:** draccus (lerobot's config parser) uses a type registry that is populated by importing config subclasses. `PiperFollowerConfig` was never imported, so the registry had no entry for `piper_follower`.

**Fix:** Added `from lerobot.robots.piper_follower import PiperFollowerConfig` to `vlash/vlash/configs/run_config.py`.

### 3.2 `ModuleNotFoundError: lerobot.robots.reachy2` on VLASH startup

**Cause:** VLASH's `run_config.py` imported reachy2 unconditionally at module load. lerobot_piper does not ship the reachy2 module.

**Fix:** Wrapped the import in `try/except ModuleNotFoundError`.

### 3.3 `ImportError: cannot import name 'OBS_IMAGES' from 'lerobot.utils.constants'`

**Cause:** VLASH imports constants from `lerobot.utils.constants`. In lerobot_piper, these constants are at `lerobot.constants`.

**Fix:** Created `lerobot_piper/src/lerobot/utils/constants.py` containing `from lerobot.constants import *`.

### 3.4 `AttributeError: module 'rerun' has no attribute 'Scalar'`

**Cause:** rerun ≥ 0.23 renamed `rr.Scalar` to `rr.Scalars`. lerobot_piper's visualization code used the old name.

**Fix:** Renamed all 4 occurrences of `rr.Scalar(` → `rr.Scalars(` in `visualization_utils.py`.

### 3.5 `AttributeError: module 'lerobot.utils.visualization_utils' has no attribute 'init_rerun'`

**Cause:** VLASH imports `init_rerun` (public) but lerobot_piper only defines `_init_rerun` (private).

**Fix:** Added `init_rerun = _init_rerun` alias at the bottom of `visualization_utils.py`.

### 3.6 pixi pulls official lerobot 0.4.1 from PyPI, overwriting lerobot_piper

**Cause:** VLASH pins `lerobot==0.4.1`. lerobot_piper was versioned `0.3.3`, so pixi saw it as outdated and resolved to official lerobot 0.4.1 from PyPI (which has no Piper support).

**Fix:** Bumped `version` in `lerobot_piper/pyproject.toml` from `0.3.3` to `0.4.1`.

### 3.7 No `manylinux_2_28_aarch64` wheel for rerun-sdk 0.21–0.22

**Cause:** rerun-sdk versions 0.21 and 0.22 were not published with an aarch64 wheel. Pixi install fails on Jetson.

**Fix:** Changed rerun-sdk constraint from `>=0.21.0,<0.23.0` to `>=0.23.0` in `lerobot_piper/pyproject.toml`.

### 3.8 transformers version conflict between smolvla and VLASH

**Cause:** lerobot_piper's smolvla extra pinned `transformers<4.52.0`. VLASH's git-pinned transformers is ≥4.52.0. Hard conflict.

**Fix:** Removed the `<4.52.0` upper bound from `lerobot_piper/pyproject.toml`, leaving `>=4.50.3`.

### 3.9 `GLIBCXX_3.4.30 not found` pyarrow crash

**Cause:** Pixi ships its own `libstdc++.so.6` which may be an older version than what pyarrow requires. On aarch64 Jetson, this causes a GLIBCXX symbol version mismatch when pyarrow is imported.

**Fix:** `scripts/activate.sh` preloads the correct `libstdc++.so.6` from the system before any Python imports. This script runs automatically via `[activation]` in `pixi.toml`.

---

## 4. Dependency Risks on aarch64

Known risks when running on Jetson Orin (linux-aarch64) — tracked here for future JetPack/package updates.

| Package | Risk | Notes |
|---|---|---|
| `torch` | Requires JetPack-specific wheel | Must use `pypi.jetson-ai-lab.io/jp6/cu126`. Standard PyPI torch does not have CUDA support on aarch64. |
| `GradScaler(device.type, ...)` | New API requires PyTorch ≥ 2.4 | LeRobot uses this API. VLASH does not. If LeRobot is used directly on Jetson with an older PyTorch, it will fail. |
| `accelerate` | Must match JetPack PyTorch | If PyTorch version changes, accelerate must be updated in sync or AMP may behave incorrectly. |
| `bitsandbytes` | May need source compile | Binaries for aarch64 are not always available; may need to build from source for new JetPack versions. |
| `triton` | JetPack-specific wheel required | Standard PyPI triton does not work on nvgpu. Pinned to `3.6.0` from the Jetson AI Lab index. |
| `rerun-sdk` | aarch64 wheels missing for some versions | 0.21–0.22 have no aarch64 wheels. Pinned to ≥0.23.0 which has them. Check release notes before upgrading. |
