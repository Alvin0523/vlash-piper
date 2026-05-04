# VLASH on Piper Arm — Full Technical Reference

**Project:** VLASH (MIT HAN Lab) + Piper Arm (WeGo Robotics) on Jetson AGX Orin  
**Repository:** [Alvin0523/vlash-piper](https://github.com/Alvin0523/vlash-piper)  
**Last Updated:** 2026-05-04  
**Task:** White ball sorting (pick-and-place)

---

## Table of Contents

1. [System & Hardware](#1-system--hardware)
2. [Architecture Overview](#2-architecture-overview)
3. [Integration — What Was Changed](integration.md)
4. [Training Configuration](#4-training-configuration)
5. [Async vs Sync — Deep Dive](#5-async-vs-sync--deep-dive)
6. [VLASH vs LeRobot Training: Architecture Comparison](#6-vlash-vs-lerobot-training-architecture-comparison)
7. [Data Collection Guide](#7-data-collection-guide)
8. [Camera Setup](#8-camera-setup)
9. [Inference Configuration — Line-by-Line](#9-inference-configuration--line-by-line)
10. [Making Sync Inference Smoother](#10-making-sync-inference-smoother)
11. [Datasets & Models](#11-datasets--models)
12. [HuggingFace Repository Setup](#12-huggingface-repository-setup)
13. [Benchmarking & Evaluation](#13-benchmarking--evaluation)
14. [Pixi Task Reference](#14-pixi-task-reference)

---

## 1. System & Hardware

### Compute

| Component | Detail |
|---|---|
| **Board** | NVIDIA Jetson AGX Orin Developer Kit |
| **Unified RAM** | 64 GB LPDDR5 |
| **GPU** | Ampere integrated (nvgpu driver) |
| **JetPack** | 6.2.2 (L4T R36.5.0, build 2026-01-16) |
| **CUDA** | 12.6 |
| **OS / Arch** | Ubuntu, linux-aarch64 |
| **Python** | 3.10 |
| **PyTorch** | 2.11.0 (Jetson AI Lab wheel: `pypi.jetson-ai-lab.io/jp6/cu126`) |
| **Torchvision** | 0.26.0 |
| **Torchcodec** | 0.10.0 |
| **Triton** | 3.6.0 |

### Robot

| Component | Detail |
|---|---|
| **Robot arm** | Piper (WeGo Robotics), 6-DOF |
| **Communication** | CAN bus — `can2` (follower), `can3` (leader) |
| **SDK** | `piper-sdk >= 0.4.1`, `wego-piper >= 0.0.2` |

### Cameras

| Camera | Serial | Bus | Max FPS | Mount |
|---|---|---|---|---|
| Wrist | `348122073292` | USB 2.1 | 15 fps reliable | Back of forearm, angled 30–45° downward |
| Overhead | `048322071496` | USB 3.2 | 30 fps | ~50 cm above table, offset to non-arm side |

Both cameras: Intel RealSense, 640×480 resolution.

### Inference Backend

- **No TensorRT** conversion applied.
- **No `torch.compile`** — crashes on Jetson's nvgpu driver (see [problems.md](problems.md)).
- **No int8 quantization.** Model runs in **bfloat16** (natively supported on Orin's Ampere GPU).
- Kernel fusions enabled: `fuse_qkv: true`, `fuse_gate_up: true` (safe on all hardware).

---

## 2. Architecture Overview

### PI0.5

PI0.5 is a **flow-matching Vision-Language-Action (VLA) model** built on a PaliGemma-style vision-language backbone. At every inference step it takes:

- One or more camera images (wrist + overhead in this setup)
- Robot joint state (6 DOF)
- A natural-language task string (e.g. `"white ball sorting"`)

…and outputs a **chunk** of 50 future joint-position actions (`chunk_size = 50`, ~1.67 s at 30 Hz).

The model is fine-tuned using **bfloat16** mixed precision. The Jetson Orin AGX supports bfloat16 natively at hardware level; do not change this to float32 — it doubles VRAM usage and slows inference with no quality benefit.

### VLASH

VLASH (MIT HAN Lab) is the training and inference framework wrapping PI0.5. Its key contribution over standard LeRobot-style training is **Temporal Delay Augmentation (TDA)**: training the model to be robust to stale observations, enabling true asynchronous inference where the next chunk is pre-computed while the current chunk is executing.

The VLASH framework provides:
- A custom dataset class (`VLASHDataset`) that injects random observation delays during training
- An async inference loop that overlaps GPU computation with robot execution
- LoRA fine-tuning support with automatic checkpoint merging
- `Accelerator`-based training with gradient accumulation and multi-GPU support

### Piper + LeRobot

The Piper arm is controlled through WeGo Robotics' `lerobot_piper` fork. It provides:
- `PiperFollowerConfig` and `PiperFollower` robot class with CAN driver
- `PiperMotorsBus` for joint state read/write over CAN
- Integration with lerobot's teleop and data recording pipeline

This repo (`vlash-piper`) combines VLASH, lerobot_piper, and custom integration glue into a single Pixi-managed environment.

---

## 3. Integration — What Was Changed

Four code changes were required to make VLASH work with lerobot_piper on Jetson Orin. None of these touch VLASH's core logic or the robot driver — they are all compatibility shims.

### 3.1 `lerobot_piper/pyproject.toml`

Three line changes:

| Field | Before | After | Why |
|---|---|---|---|
| `version` | `0.3.3` | `0.4.1` | VLASH pins `lerobot==0.4.1`. Without the bump, pixi resolves to official lerobot 0.4.1 from PyPI (no Piper support), overwriting lerobot_piper. |
| `rerun-sdk` | `>=0.21.0,<0.23.0` | `>=0.23.0` | Versions 0.21–0.22 have no `manylinux_2_28_aarch64` wheels — the package is simply absent for Jetson Orin's architecture. |
| `transformers` | `>=4.50.3,<4.52.0` | `>=4.50.3` | VLASH's git-pinned transformers is ≥4.52.0. The upper bound caused a hard dependency conflict. Removing it lets both coexist. |

### 3.2 `lerobot_piper/src/lerobot/utils/constants.py` (new file)

```python
from lerobot.constants import *  # noqa: F401, F403
```

VLASH imports `OBS_IMAGES`, `ACTION`, `OBS_STATE` from `lerobot.utils.constants`. In lerobot_piper, these constants live at `lerobot.constants`, not `lerobot.utils.constants`. This one-liner shim bridges the import path without modifying either upstream package.

### 3.3 `lerobot_piper/src/lerobot/utils/visualization_utils.py`

Two changes:

**Added at the bottom of the file:**
```python
init_rerun = _init_rerun
```
VLASH imports `init_rerun` (public name) but lerobot_piper only defines `_init_rerun` (private name). Adding this alias means VLASH's import works without changing either package's internal naming.

**Renamed all 4 occurrences of `rr.Scalar(` → `rr.Scalars(`**  
rerun ≥ 0.23 renamed the `rr.Scalar` API to `rr.Scalars`. Without this fix, running with `display_data: true` crashes immediately with `AttributeError: module 'rerun' has no attribute 'Scalar'`.

### 3.4 `vlash/vlash/configs/run_config.py`

Two changes:

**Added import:**
```python
from lerobot.robots.piper_follower import PiperFollowerConfig  # noqa: F401
```
draccus (lerobot's config parser) requires all robot config subclasses to be imported before YAML parsing begins. Without this import, setting `type: piper_follower` in any YAML raises `KeyError: 'piper_follower'` immediately — the type registry is empty.

**Wrapped reachy2 import in try/except:**
```python
try:
    from lerobot.robots.reachy2 import Reachy2RobotConfig  # noqa: F401
except ModuleNotFoundError:
    pass
```
lerobot_piper does not ship the reachy2 robot module. The bare import in VLASH crashed on startup even when not using reachy2 — because the import happens at module load time, before any config is parsed.

### 3.5 `pixi.toml` (new file)

A single Pixi environment that resolves all conflicts between lerobot_piper and VLASH:

- Pins Jetson-specific PyTorch wheels from `pypi.jetson-ai-lab.io/jp6/cu126`
- Installs lerobot_piper as editable (as the `lerobot` package, satisfying VLASH's pin)
- Installs vlash as editable
- Activation script `scripts/activate.sh` preloads correct `libstdc++.so.6` to fix pyarrow `GLIBCXX` mismatch on aarch64
- Sets `platforms = ["linux-aarch64"]`

---

## 4. Training Configuration

### 4.1 Common Settings

These settings are the same across all training modes (async LoRA, sync, sync LoRA):

| Parameter | Value | Notes |
|---|---|---|
| **Steps** | 50,000 | Reasonable for pick-and-place. Underfitting < 20k makes arm uncertain; overfitting on small datasets > 60k makes it brittle. |
| **Batch size** | 16 | |
| **Optimizer** | AdamW | |
| **Learning rate** | 5.0e-5 | Proven value from VLASH paper. Do not increase. If loss diverges, halve it. |
| **LR betas** | [0.9, 0.95] | |
| **Weight decay** | 1.0e-10 | |
| **Scheduler** | Cosine decay with warmup | |
| **Warmup steps** | 1,000 | Do not reduce. PI0.5's LM backbone is highly LR-sensitive; skipping warmup causes divergence in the first few hundred steps. |
| **Peak LR** | 5.0e-5 | |
| **Decay LR** | 2.5e-6 | |
| **dtype** | bfloat16 | Natively supported on Orin Ampere. Do not change to float32. |
| **state_cond** | true | Joint state fed as extra conditioning input. Must match between training and inference. |
| **Checkpoint freq** | Every 5,000 steps | |
| **Video backend** | torchcodec | |
| **Workers** | 4 | |
| **Seed** | 1000 | |

### 4.2 Async + LoRA Training

```yaml
# vlash/examples/train/pi05/async_lora_piper.yaml
max_delay_steps: 8
shared_observation: true

lora:
  enable: true
  backend: peft
  r: 16
  alpha: 16
  dropout: 0
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  extra_trainable_modules:
    - action_in_proj
    - action_out_proj
    - time_mlp_in
    - time_mlp_out
    - state_proj
    - state_mlp_in
    - state_mlp_out
    - embeddings
    - input_layernorm
    - post_attention_layernorm
```

LoRA rank `r=16`, `alpha=16` means the effective learning rate scaling is `alpha/r = 1.0`. LoRA merges into full weights at checkpoint time — the saved model is a normal full-weight model, not a LoRA adapter.

Pixi task: `pixi run train-pi05-lora`

### 4.3 Sync Training (no LoRA)

```yaml
# vlash/examples/train/pi05/sync_piper.yaml
max_delay_steps: 0
# (no lora block)
```

Setting `max_delay_steps: 0` disables Temporal Delay Augmentation — the dataset returns standard `(obs_t → actions_{t..t+49})` samples. No `shared_observation` needed.

Pixi task: `pixi run train-pi05-sync`

### 4.4 Sync + LoRA Training

Same as sync but with the LoRA block added:

```yaml
# vlash/examples/train/pi05/sync_lora_piper.yaml
max_delay_steps: 0

lora:
  enable: true
  # ... same LoRA config as async_lora_piper.yaml
```

Pixi task: `pixi run train-pi05-sync-lora`

### 4.5 Key Training Invariants

- `chunk_size` is fixed at training time. At inference, `n_action_steps` must match `chunk_size` (both 50 here) or be set intentionally lower. Setting it higher than `chunk_size` will index out of bounds.
- `state_cond` is baked into the model weights. A model trained with `state_cond: true` that is run with a config that does not feed joint state will produce garbage output silently.
- Camera names in `input_features` of `config.json` must exactly match the camera names in the inference YAML. A name mismatch causes either a crash or silent wrong-input feeding.

---

## 5. Async vs Sync — Deep Dive

### 5.1 The Core Problem Async Solves

In sync inference:
1. Execute all 50 actions from the current chunk (~1.67 s)
2. **Pause** while GPU runs inference for the next chunk (~200–500 ms)
3. Execute the next 50 actions

The pause at step 2 is visible as the arm freezing between action chunks. The policy also uses an observation that is up to 50 steps old by the time it finishes executing the chunk — its view of the world is stale.

In async inference:
- At action step 42 of the current 50-step chunk, inference starts for the next chunk **in parallel**
- By the time action step 50 finishes, the next chunk is already computed
- The arm never pauses; motion is continuous

The `inference_overlap_steps` parameter controls when the pre-computation starts:
```
n_action_steps=50, inference_overlap_steps=8:
  Steps 0–41: execute actions normally
  Step 42: GPU starts computing next chunk (using observation at step 42)
  Steps 43–49: arm executes while GPU computes
  Step 50: next chunk ready; execute immediately
```

### 5.2 Why Async Requires Temporal Delay Augmentation

When the GPU computes the next chunk using the observation at step 42, the chunk will begin executing at step 50 — 8 steps in the future. The model must predict actions for a state it hasn't seen yet.

A sync-trained model cannot do this — it was trained on `(obs_t → actions_t..t+49)` where the observation is always fresh. Ask it to predict `actions_{t+8..t+57}` from `obs_t` and it will produce the wrong actions because it doesn't know the arm has already moved 8 steps.

VLASH's Temporal Delay Augmentation (TDA) solves this by training the model to handle delayed observations:

$$\text{sample} = (\text{obs}_{t},\ \text{delay}=d,\ \text{actions}_{t+d \ldots t+d+49})$$

where $d$ is sampled uniformly from $[0, \texttt{max\_delay\_steps}]$ for each training example.

The model learns: "I'm seeing an observation from $d$ steps ago — the arm has already moved. Predict actions that account for this." With `max_delay_steps: 8`, the model learns to handle up to 8 steps of staleness, which exactly matches `inference_overlap_steps: 8`.

### 5.3 Shared Observation Efficiency

When `shared_observation: true`, instead of running a separate forward pass for each delay value (0, 1, ..., 8) per batch, VLASH runs the vision-language backbone **once** and applies custom attention masks to produce all delay outputs in a single forward pass. This gives approximately `(max_delay_steps + 1)`× training throughput improvement.

### 5.4 The Jetson Constraint

Async inference requires `compile_model: true`:

```python
# From run_config.py — this is a hard runtime check
if self.inference_overlap_steps > 0 and not self.policy.compile_model:
    raise ValueError(
        "When inference_overlap_steps > 0, policy.compile_model must be True. "
        "Async inference requires compiled model for CPU overlapping."
    )
```

`torch.compile` crashes on Jetson Orin's nvgpu driver. Therefore, **async inference cannot run on Jetson Orin**.

**On Jetson Orin, you are locked to sync mode:**
```yaml
compile_model: false
inference_overlap_steps: 0
```

See [problems.md](problems.md) for the full crash analysis.

### 5.5 Comparison Table

| Aspect | Sync | Async |
|---|---|---|
| `inference_overlap_steps` | `0` | `> 0` (e.g. 8) |
| `compile_model` | `false` | `true` (required) |
| `max_delay_steps` (training) | `0` | `> 0` (e.g. 8) |
| `shared_observation` (training) | not set | `true` |
| Dataset type | Standard `LeRobotDataset` | `VLASHDataset` with TDA |
| What the model learns | Actions from current observation | Actions from stale observation (up to N steps old) |
| Arm behavior at chunk boundary | **Pauses** while GPU infers | **Continuous** — next chunk pre-computed |
| Works on Jetson Orin | **Yes** | **No** (compile crash) |
| Works on desktop GPU (RTX, etc.) | Yes | Yes |

---

## 6. VLASH vs LeRobot Training: Architecture Comparison

`lerobot_piper/src/lerobot/scripts/train.py` (301 lines) vs `vlash/vlash/train.py` (634 lines).

### 6.1 Top-Level Architecture Differences

| Dimension | LeRobot `train.py` | VLASH `train.py` |
|---|---|---|
| Lines | 301 | 634 |
| Distributed training | Single GPU only | `Accelerator` — supports multi-GPU |
| Mixed precision (AMP) | Manual `GradScaler` + `torch.autocast` | `accelerator.autocast()` — automatic |
| Gradient accumulation | Not supported | `grad_accum_steps` |
| Dataset | Standard `LeRobotDataset` | Custom `VLASHDataset` with temporal delay augmentation |
| LoRA | Not supported | Full LoRA support; auto-merge at checkpoint |
| Auto-resume | Manual `--resume` flag | `auto_resume()` — auto-detects latest checkpoint |
| In-loop sim eval | Yes (`eval_policy` inside training loop) | No (moved to external eval script) |

### 6.2 `update_policy` Function Signatures

**LeRobot:**
```python
def update_policy(
    train_metrics, policy, batch, optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,   # manual AMP management
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
)
```

**VLASH:**
```python
def update_policy(
    train_metrics, policy, batch, optimizer,
    grad_clip_norm,
    accelerator: Accelerator,  # replaces GradScaler — auto AMP management
    lr_scheduler=None,
    lock=None,
    *,
    loss_scale: float = 1.0,           # gradient accumulation scaling
    do_step: bool = True,              # whether to call optimizer.step
    use_shared_observation: bool = False,  # shared-observation forward pass
)
```

### 6.3 Forward Pass

**LeRobot:**
```python
with torch.autocast(device_type=device.type) if use_amp else nullcontext():
    loss, output_dict = policy.forward(batch)
```

**VLASH:**
```python
with accelerator.autocast():
    loss, output_dict = policy.forward(batch)
    raw_loss = loss.detach()   # saved for logging — unscaled true loss
    loss = loss * loss_scale   # gradient accumulation scaling
```

VLASH saves `raw_loss` separately so the logged loss value is always the unscaled real loss value, regardless of gradient accumulation factor.

### 6.4 Backward Pass and GradScaler

The most significant difference.

**LeRobot (explicit GradScaler):**
```python
# 1. Scale loss to prevent fp16 underflow
grad_scaler.scale(loss).backward()

# 2. Must unscale before clipping (restores true gradient magnitude)
grad_scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

# 3. step internally skips updates containing inf/NaN
grad_scaler.step(optimizer)

# 4. Dynamically adjust scale factor (up/down based on inf/NaN frequency)
grad_scaler.update()

optimizer.zero_grad()
```

**VLASH (Accelerator-wrapped, equivalent behavior):**
```python
# 1. accelerator.backward wraps GradScaler.scale().backward() internally
accelerator.backward(loss)

# 2. clip_grad_norm_ auto-unscales internally
if grad_clip_norm > 0:
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
else:
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))

# 3. step + zero_grad (unscale + inf/NaN skip already wrapped inside optimizer)
optimizer.step()
optimizer.zero_grad()
```

The behavior is functionally identical. VLASH's version is shorter because Accelerator wraps the boilerplate.

**Why GradScaler exists:** Mixed precision (AMP) uses fp16 for computation. Gradients can underflow to zero in fp16. GradScaler multiplies the loss by a large scale factor before backprop to amplify gradients above the fp16 floor, then divides them back before the optimizer step.

```
forward (fp16)  →  loss × scale_factor  →  backward()  →  unscale_()  →  clip  →  step
                                                                              ↓
                                                                     grad_scaler.update()
                                                               (raise/lower scale_factor)
```

VLASH's Accelerator encapsulates this entire flow; LeRobot exposes it explicitly.

### 6.5 Behavioral Differences Summary

| Behavior | LeRobot | VLASH |
|---|---|---|
| AMP activation | `use_amp` config flag | Accelerator config |
| `unscale_` timing | Explicit, before clip | Auto, inside `clip_grad_norm_` |
| inf/NaN skip | `grad_scaler.step` handles | Accelerator's optimizer handles |
| Loss logged | `loss.item()` (may be loss-scaled) | `raw_loss.item()` (always unscaled) |
| Gradient clipping | Unconditional | Only when `grad_clip_norm > 0` |
| Gradient accumulation | Not supported | `loss_scale = 1/grad_accum_steps` |

### 6.6 VLASHDataset vs LeRobotDataset

**LeRobot:**
```python
dataset = make_dataset(cfg)  # standard LeRobotDataset
```
Returns `(obs_t, actions_{t..t+49})` — always fresh observation.

**VLASH:**

`VLASHDataset` implements Temporal Delay Augmentation:
```
chunk_size = 50, max_delay_steps = 8

Standard query: [t, t+1, ..., t+49]  (50 actions, always aligned with obs_t)

VLASH query:
  d = random(0, 8)                   # sampled per training example
  obs: timestep t
  actions: [t+d, t+1+d, ..., t+49+d] # shifted by delay d
  delay_cond: d                       # passed to model as conditioning
```

When `shared_observation=True`, uses `SharedObservationVLASHDataset` — runs the vision-language backbone once and applies attention masks to compute all delay values simultaneously.

### 6.7 LoRA in VLASH

```python
# Applied before Accelerator wrapping
apply_lora(cfg.lora, policy, verbose=is_main_process)

# At checkpoint time: merge LoRA into full weights and save merged model
if cfg.lora.enable and is_lora_policy(unwrapped_policy):
    merged_policy = clone_and_merge_lora_policy(unwrapped_policy, cfg.lora, ...)
    save_checkpoint(..., policy=merged_policy, ...)
```

LeRobot has no LoRA support — every checkpoint saves the full model weights. VLASH LoRA checkpoints are ready for inference (merged weights) without any extra step.

### 6.8 Dependency Risk on aarch64 (Jetson)

| Package | LeRobot | VLASH | aarch64 Risk |
|---|---|---|---|
| `torch` (JetPack) | Shared | Shared | Requires JetPack-specific wheel |
| `GradScaler(device.type, ...)` | Uses new API | Not used | LeRobot at risk if PyTorch < 2.4 |
| `shutup` | Depends on | Not used | Extra install needed |
| `accelerate` | No dependency | Depends on | Must match JetPack PyTorch version |
| `bitsandbytes` | No | Indirect | May need source compile on aarch64 |

---

## 7. Data Collection Guide

### 7.1 Goal

Every episode is a demonstration. The model learns to copy the **distribution** of your movements. Bad demonstrations teach bad policies. Quality beats quantity — 20 clean episodes beat 50 messy ones.

### 7.2 DOs

**Plan the full motion before pressing record.**
Think: reach → grasp → lift → transport → place. Know where the object is, where the target is, and how you'll handle each sub-motion before starting.

**Be smooth and deliberate.**
Slow, controlled movements are better than fast jerky ones. The arm has joint velocity limits — if you move the leader too fast, the follower will lag and the recorded state will not match the intended trajectory.

**Keep the object in the same workspace region across episodes.**
The policy learns to pick from a distribution of positions. If you scatter the object randomly across the whole table, you need far more episodes to cover that distribution. Keep it in a consistent 10–15 cm zone. Only introduce variation once you have ≥ 50 clean episodes in the base zone.

**Pause ~0.5 s after closing the gripper before lifting.**
This gives the camera a clear keyframe of the grasp and teaches the policy to confirm the grasp before moving on.

**Complete the full motion including release.**
Guide the object all the way to the target and release. Do not stop halfway and call it done. The policy learns from the entire trajectory.

**Verify the episode visually before saving.**
Check `data/<dataset>/videos/` after each episode. Delete any episode where:
- The arm bumped the table or objects
- The object was dropped or slipped
- The camera view was blocked
- You made a large correction mid-motion (sudden direction reversal)

**Use consistent lighting.**
The VLA backbone is vision-heavy. Changes in lighting between recording sessions confuse the model. Record all episodes in identical lighting conditions to deployment.

**Record both sub-steps clearly:**
1. Approach and grasp the object
2. Transport and place at the target

**Aim for ≥ 50 high-quality episodes for a simple pick-place task. 100–200 is better.**

### 7.3 DON'Ts

**Don't make large trajectory corrections.**
If you reach for the object and miss, do not swing back for a second attempt. Stop the episode, discard it, reposition, and start again. The model will learn "failed approach + correction" as a valid pattern, creating jerky inference behavior.

**Don't vary the target position between episodes without a plan.**
If the target moves, the model must learn target localization too. Fix it until the base task works, then add variation.

**Don't record when the CAN connection is unstable.**
If you see joint feedback lag or the follower arm is sluggish, stop. A sluggish follower means the recorded state does not reflect where the arm actually was — this introduces action-state misalignment.

**Don't exceed ~60% of max joint velocity.**
Movements that are too fast will cause the follower to lag. The recorded actions will look like they "overshoot" because the leader moved faster than the follower could follow.

**Don't record with inconsistent grasp height.**
Picking at wildly different heights creates a bimodal grasp distribution. The policy will average between them and grasp at the middle — which works for neither.

**Don't mix different objects or targets without intent.**
A distribution mismatch is fine if intentional (for generalization), problematic if accidental.

### 7.4 Arm Start Position

The arm's starting position is part of the observation the model conditions on. If the arm starts from different positions across episodes, the model cannot learn a clean "beginning of task" state.

**Recommended home position:**
1. Joints 1–4 (shoulder/elbow): extended forward, slightly above table height, hovering above the pick zone — not resting on the table.
2. Joints 5–6 (wrist): neutral, gripper pointing straight down.
3. Gripper: fully open.
4. Clear gap (~5–10 cm) between gripper and the pick object.

To find and save your exact home joint angles:
```bash
pixi run teleop
# drive to home position; read joint state from terminal output
# note the values; reproduce them before each episode
```

**Do NOT rest the arm on the table as the home position.** A resting pose forces the model to learn an upward lift from rest as the very first move, wasting episode timesteps and making approach trajectories inconsistent.

### 7.5 Checking for Frame Drops After Each Session

```bash
ffprobe -v quiet -show_streams \
  data/<dataset>/videos/chunk-000/observation.images.wrist_episode_000000.mp4 \
  | grep nb_frames
```

If `nb_frames` is ~18% of the action frame count, the USB camera dropped frames. See [problems.md](problems.md) for the full USB frame drop analysis and fixes.

---

## 8. Camera Setup

### 8.1 How Many Cameras

Two cameras (wrist + overhead) are required for reliable pick-and-place:

- The wrist cam sees the grasp close-up but loses the target during transport
- The overhead cam sees the global scene but misses fine gripper detail

One camera alone is not enough to cover both sub-tasks.

### 8.2 Wrist Camera (`wrist`, serial `348122073292`, USB 2.1)

- Mount on the back of the forearm link, pointing forward-down
- Field of view: gripper fingers + ~15 cm in front of gripper
- Angle: ~30–45° downward from horizontal (not straight down — misses approach; not horizontal — misses grasp)
- Keep rigidly fixed relative to the wrist joint. Any wobble adds noise that looks like camera motion to the model.
- **USB 2.1 limits this camera to ~15 fps reliably.** Set `fps: 15` if recording at 15 fps, and match inference YAML fps to recording fps. Mismatched fps is a common silent failure — model trained at 15 fps but running inference at 30 fps sees a repeated frame every other step.

### 8.3 Overhead Camera (`up`, serial `048322071496`, USB 3.2)

- Mount ~40–60 cm above table, offset ~20 cm to the non-arm side, angled ~45° downward
- Field of view: must show the pick zone AND the target simultaneously throughout the entire task, including during transport
- Do not mount directly overhead (straight down) — the arm will occlude itself during approach
- USB 3.2 → 30 fps is achievable

### 8.4 Stability Requirements

Both cameras must be **rigidly mounted**. Even 1–2 mm vibration during arm motion introduces motion blur that the model has never seen in training. After each recording session, verify cameras haven't shifted by comparing frames from the first and last episodes.

### 8.5 Lighting

- Avoid direct sunlight (changes angle and intensity through the day)
- Use fixed artificial lighting positioned to minimize shadows under the gripper
- The target opening must be clearly visible to the overhead camera — avoid placing it where it is in the arm's shadow during the placement phase

---

## 9. Inference Configuration — Line-by-Line

Reference: `vlash/examples/inference/sync_piper.yaml` (the working Jetson config).

### 9.1 Robot Block

```yaml
robot:
  type: piper_follower
```
Tells draccus (lerobot's config parser) to instantiate `PiperFollowerConfig`. This type must be imported in `run_config.py` before YAML parsing — see [Section 3.4](#34-vlashvlashconfigsrun_configpy).

```yaml
  port: can2
```
Which CAN interface the follower arm is on. `can2` = follower, `can3` = leader. Swapping these sends inference commands to the leader arm.

```yaml
  id: follower
```
Logical name used internally by lerobot for multi-arm setups.

```yaml
  cameras:
    wrist:
      type: intelrealsense
      serial_number_or_name: "348122073292"
      width: 640
      height: 480
      fps: 30
```
Camera named `wrist`. The name must exactly match what the model was trained with. `serial_number_or_name` uniquely identifies the physical device — without it, lerobot picks whichever camera it finds first. PI0.5 internally resizes images before the vision encoder, so 640×480 is just the capture resolution. Set `fps: 15` if USB 2.1 is dropping frames.

```yaml
    up:
      type: intelrealsense
      serial_number_or_name: "048322071496"
      width: 640
      height: 480
      fps: 30
```
Camera named `up` (overhead). USB 3.2, genuine 30 fps achievable.

### 9.2 Policy Block

```yaml
policy:
  path: models/sync_test8/pretrained_model
```
Path to the checkpoint directory (containing `config.json` + `model.safetensors`). Resolved relative to the working directory where you run `vlash run`. From `~/vlash_piper` this resolves correctly.

```yaml
  n_action_steps: 50
```
**Critical.** How many actions from the model's 50-step chunk are actually executed. Must match the model's `chunk_size` in `config.json` (both are 50 here). Setting this lower than `chunk_size` is valid and can reduce chunk-boundary jerkiness (executes only the first N steps of each chunk). Setting it higher than `chunk_size` causes an index out-of-bounds error.

```yaml
  compile_model: false
```
**Must be `false` on Jetson Orin.** See [Section 5.4](#54-the-jetson-constraint) and [problems.md](problems.md).

```yaml
  device: cuda
```
Run on the Jetson's GPU. `cpu` is valid but ~10× slower.

```yaml
  fuse_qkv: true
  fuse_gate_up: true
```
Kernel fusions for attention (Q/K/V) and SwiGLU FFN (gate/up) layers. Reduces memory bandwidth and speeds up inference. Safe to keep `true` on all hardware; does not change model outputs.

### 9.3 Task String

```yaml
single_task: "white ball sorting"
```
The natural-language instruction passed to PI0.5's language backbone at every inference step. **Must match the phrasing used during data recording.** PI0.5 after fine-tuning is specialized to the phrasing it saw in training — it is not a general instruction-following model. Changing wording can degrade performance.

### 9.4 Control Parameters

```yaml
fps: 30
```
Target control loop frequency. If inference takes longer than 1/30 s ≈ 33 ms (expected in sync mode on Jetson), the loop runs slower than 30 Hz. This is normal.

```yaml
control_time_s: 600
```
Total wall-clock seconds to run before auto-stopping. Set to a shorter value (e.g. 30) when testing to avoid runaway long sessions.

### 9.5 Visualization and Feedback

```yaml
display_data: false
```
Opens a rerun viewer showing live camera feeds and joint state. If running over SSH without X forwarding, **must be `false`** — otherwise the process hangs waiting for a display.

```yaml
play_sounds: true
```
Audio beeps for episode start/end/error. Set to `false` when running headless.

### 9.6 Action Quantization

```yaml
action_quant_ratio: 1
```
How many control ticks each action step is held for:
- `1` = every action sent to arm at the control rate (normal)
- `2` = each action held for 2 ticks, arm moves at half speed
- Higher = slower arm, fewer inferences needed per unit time

`action_quant_ratio: 2` is the `infer-async-fast` mode and effectively halves the inference frequency required to sustain real-time control.

### 9.7 Overlap / Async Control

```yaml
inference_overlap_steps: 0
```
`0` = sync mode: play all 50 actions, pause, run inference, play next 50.  
`> 0` = async mode: requires `compile_model: true`, which crashes on Jetson.

### 9.8 Async Inference Config (for reference — not usable on Jetson)

```yaml
# vlash/examples/inference/async_piper.yaml
policy:
  compile_model: true   # required for async — crashes on Jetson nvgpu
inference_overlap_steps: 8
action_quant_ratio: 1   # or 2 for infer-async-fast
```

### 9.9 Pre-Run Checklist

- [ ] `single_task` matches training phrasing exactly
- [ ] Camera names in YAML match model's `input_features` in `config.json`
- [ ] `n_action_steps` matches model's `chunk_size` or is intentionally lower
- [ ] `compile_model: false` (Jetson Orin)
- [ ] `inference_overlap_steps: 0` (sync mode)
- [ ] Jetson in max performance mode: `sudo nvpmodel -m 0 && sudo jetson_clocks`
- [ ] CAN interfaces up: `pixi run can`
- [ ] Arm at home position before starting
- [ ] Object in pick zone, target in place zone

---

## 10. Making Sync Inference Smoother

The core problem in sync mode is the **chunk boundary freeze**: arm plays 50 actions (~1.67 s at 30 Hz), then freezes while GPU infers (~200–500 ms on Jetson Orin for PI0.5), then plays next 50.

### Strategy 1: Reduce `n_action_steps` (most impactful)

Execute fewer actions per chunk so inference happens more frequently with fresher observations:

```yaml
policy:
  n_action_steps: 25   # reduced from 50
```

With 25 steps at 30 Hz you infer every ~0.83 s instead of every ~1.67 s. The pause duration is the same absolute time, but happens more frequently with fresher observations and shorter committed trajectories, so errors are corrected faster.

**Trade-off:** More GPU utilization. Monitor thermal throttling with `sudo tegrastats`.  
**Tuning:** Try `n_action_steps: 25` first. If smoother, try `20`. Do not go below `10` — inference setup overhead becomes significant.

### Strategy 2: Maximize Jetson Performance Mode

```bash
sudo nvpmodel -m 0    # maximum power mode
sudo jetson_clocks    # lock GPU/CPU clocks at maximum
```

Without this, Jetson may be in a lower power mode causing slower, more variable inference — resulting in irregular chunk-boundary pauses.

### Strategy 3: Reduce `fps` Slightly

If inference latency is just above one control tick (33 ms at 30 Hz), the arm stutters every cycle. Dropping `fps` to 25 gives more slack per tick:

```yaml
fps: 25
```

### Strategy 4: Close Competing Processes

During inference, stop:
- rerun viewer (`display_data: false`) if running headless
- Any other GPU workloads
- Background processes with high CPU/memory usage

### Strategy 5: Use `action_quant_ratio` to Reduce Effective Arm Speed

```yaml
action_quant_ratio: 2
```

Holds each action for 2 control ticks, halving effective arm speed. The chunk plays over ~3.3 s instead of ~1.67 s, giving more time for observations to "catch up."

**Do not combine with very low `n_action_steps`** — the arm will move too slowly to complete the task.

### Strategy 6: Tune If Performance Is Still Poor

1. If success rate < 40%: the model needs more data or more training steps. Config tuning will not fix an undertrained model.
2. If SR is 40–70% but motion is jerky: reduce `n_action_steps` from 50 → 25.
3. If motion is too fast: set `action_quant_ratio: 2`.
4. If arm freezes visibly between chunks: run `sudo jetson_clocks` and retest.
5. If wrist camera drops frames: set `fps: 15` and verify USB port assignment.

---

## 11. Datasets & Models

### 11.1 Dataset Inventory

| Dataset | Location | Cameras | Notes |
|---|---|---|---|
| test3 | `data/test3/` | side | `state_cond: true`; first async LoRA + sync training run |
| test4 | `data/test4/` | wrist + up | Episodes 020 & 041 have severe frame drop — do not use for training |
| test5 | `data/test5/` | wrist + up | — |
| test6 | `data/test6/` | wrist + up | `state_cond: false` model variant |
| test8 | `data/test8/` | wrist + up | **Final dataset; both async and sync final models trained on this** |

### 11.2 Model Inventory

| Model | Path | Mode | Dataset |
|---|---|---|---|
| async_test8 | `models/async_test8/` | Async + LoRA | test8 |
| sync_test8 | `models/sync_test8/` | Sync | test8 |

Each model directory contains:
- `pretrained_model/` — `config.json` + `model.safetensors` (inference-ready)
- `training_state/` — optimizer state, scheduler, step count (for resuming training)

### 11.3 Config Compatibility Warning

Different datasets produced models with different configs. **Mixing them causes silent failure:**

| Model | `state_cond` | Camera names | Inference YAML to use |
|---|---|---|---|
| test3 | `true` | `side` | Must have camera named `side`; joint state required |
| test6 | `false` | `wrist`, `up` | Must have cameras named `wrist` and `up`; no joint state |
| test8 | `true` | `wrist`, `up` | Must have cameras named `wrist` and `up`; joint state required |

Always check `config.json` in the pretrained_model directory before running inference:
```bash
cat models/<model>/pretrained_model/config.json | python3 -m json.tool | grep -E "state_cond|input_features|camera"
```

---

## 12. HuggingFace Repository Setup

### 12.1 Repository Structure

Two separate repos following HuggingFace best practices:

| Repo | Type | URL | Purpose |
|---|---|---|---|
| `Frieddeli/vlash` | Dataset | https://huggingface.co/datasets/Frieddeli/vlash | Raw data, training datasets |
| `Frieddeli/vlash-models` | Model | https://huggingface.co/Frieddeli/vlash-models | Trained weights, checkpoints |

Separate repos because dataset and model have different update frequencies, access patterns, versioning needs, and permission requirements.

### 12.2 Upload Commands

**Uploading datasets:**
```bash
# Upload training data
hf upload Frieddeli/vlash /path/to/train_data train --repo-type dataset

# Upload with commit message
hf upload Frieddeli/vlash /path/to/data data \
  --repo-type dataset \
  --commit-message "Add test8 dataset v1" \
  --commit-description "Final ball sorting dataset, 50+ episodes"
```

**Uploading models:**
```bash
# Upload checkpoint
hf upload Frieddeli/vlash-models /path/to/model checkpoint_async_test8

# Upload specific file types only
hf upload Frieddeli/vlash-models /path/to/model model \
  --include "*.safetensors" "*.json"

# Large model (>1 GB)
hf upload-large-folder Frieddeli/vlash-models /path/to/large_model model
```

---

## 13. Benchmarking & Evaluation

### 13.1 Primary Metric: Task Success Rate

$$SR = \frac{\text{successful attempts}}{\text{total attempts}} \times 100\%$$

Run **at minimum 20 trials** per evaluation, from the same starting configuration used in training. Report as percentage.

**Definition of success for white ball sorting:**
- Ball is grasped (not just touched)
- Ball is transported (leaves the surface)
- Ball is placed in the target zone and released (not just touching the rim)
- Ball remains in target zone after gripper opens and retracts

### 13.2 Secondary Metrics

| Metric | What it tells you | How to measure |
|---|---|---|
| **Time to completion** | Policy efficiency | Stopwatch from episode start to task completion |
| **Grasp success rate** | Separates grasp failures from transport failures | Count episodes where grasp succeeded even if place failed |
| **Place success rate (given grasp)** | Isolates transport/placement quality | Count successful places among episodes with successful grasps |
| **Motion smoothness (jerk)** | Fluid vs. jerky motion | Log follower joint velocity; compute $\frac{d^3q}{dt^3}$ |
| **Reaction latency** | Time from observation to first motion | Measure from object placement to first arm velocity |
| **Robustness to position variation** | Generalization | Vary object position ±5 cm from training distribution; re-evaluate SR |

### 13.3 Evaluation Protocol

1. **Fixed setup:** Same object, same target, same lighting, same camera positions as training.
2. **Run 20 trials.** After each trial, manually reset: arm to home, object to pick zone, target in place zone.
3. **Record video** of all trials for later analysis.
4. **Same protocol for every model variant** you want to compare. Physical setup must be identical across model evaluations.

### 13.4 Missing Numbers (to fill in)

The following metrics could not be extracted from the workspace and must be measured:

- [ ] **Inference latency (ms/step)** — measured on Orin for sync mode
- [ ] **Success rate** — Sync vs Async *(Async not runnable on Orin — desktop comparison only)*
- [ ] **Task completion time** — average seconds per full sort cycle
- [ ] **Reaction latency** — time from ball placement to first arm movement
- [ ] **Chunk boundary freeze duration** — visible pause time in sync mode

---

## 14. Pixi Task Reference

| Task | Command | Description |
|---|---|---|
| `can` | `bash scripts/init_orin_can.sh` | Activate CAN interfaces (`can2`, `can3`) |
| `teleop` | `bash scripts/teleop.sh` | Teleoperation (leader → follower) |
| `cam_check` | `python3 -m lerobot.find_cameras realsense` | List and verify RealSense cameras |
| `record` | `bash scripts/record.sh` | Record episodes (edit dataset name, episode count, task text in script) |
| `train-pi05-lora` | `vlash train vlash/examples/train/pi05/async_lora_piper.yaml` | Fine-tune PI0.5 async + LoRA |
| `train-pi05-sync` | `vlash train vlash/examples/train/pi05/sync_piper.yaml` | Fine-tune PI0.5 sync (no LoRA) |
| `train-pi05-sync-lora` | `vlash train vlash/examples/train/pi05/sync_lora_piper.yaml` | Fine-tune PI0.5 sync + LoRA |
| `infer-sync` | `vlash run vlash/examples/inference/sync_piper.yaml` | Sync inference on Piper |
| `infer-async` | `vlash run vlash/examples/inference/async_piper.yaml` | Async inference *(not usable on Jetson)* |
| `infer-async-fast` | `vlash run vlash/examples/inference/async_piper.yaml --action_quant_ratio=2` | Async inference 2× speed *(not usable on Jetson)* |
