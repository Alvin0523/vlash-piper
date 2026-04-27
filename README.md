# VLASH Piper Workspace

This workspace combines two repos as git submodules and gives one Pixi workflow for data collection, fine-tuning, and deployment.

## 0) Clone This Repo Recursively (Required)

```bash
git clone --recursive https://github.com/Alvin0523/vlash_piper.git
cd vlash_piper
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Submodules used by this workspace:

- `lerobot_piper` (`https://github.com/Alvin0523/lerobot_piper.git`)
- `vlash` (`https://github.com/Alvin0523/vlash.git`)

## 1) Environment Setup

```bash
pixi install
```

Note: training tasks preload Pixi's `libstdc++.so.6` to avoid `pyarrow` `GLIBCXX` mismatch on Jetson/aarch64.

## 2) Hardware Checks

Verify camera:

```bash
pixi run cam_check
```

Setup CAN interfaces (`can2`, `can3`):

```bash
pixi run can
```

Optional check:

```bash
ip -br link show type can
```

## 3) Data Collection (Already Built In)

Data collection is already wired through Pixi:

```bash
pixi run record
```

This calls `scripts/record.sh` (edit dataset name, episode count, and task text there).

## 4) Fine-Tune (PI0.5 Async vs Sync Comparison)

Before training, edit these fields in the YAML you will run:

- `dataset.repo_id`
- `dataset.root`
- `job_name`
- `output_dir`

```bash
# async + LoRA
pixi run train-pi05-lora

# sync (non-LoRA)
pixi run train-pi05-sync

# sync + LoRA (optional)
pixi run train-pi05-sync-lora
```

## 5) Inference via Pixi Tasks

```bash
# async inference
pixi run infer-async

# async inference with action quantization speedup
pixi run infer-async-fast
```

## 6) What Is "Sync"?

`sync` means VLASH training/inference without asynchronous delay modeling (for training config, this is typically `max_delay_steps: 0`).

It is not `lerobot` training. It is still run by `vlash train ...` / `vlash run ...`, just using the sync-style VLASH config.

## 7) Pixi Task Mapping

Pixi tasks now run VLASH commands directly (no extra train/infer wrapper scripts):

- `pixi run train-pi05-lora` -> `vlash train vlash/examples/train/pi05/async_lora_piper.yaml`
- `pixi run train-pi05-sync` -> `vlash train vlash/examples/train/pi05/sync_piper.yaml`
- `pixi run train-pi05-sync-lora` -> `vlash train vlash/examples/train/pi05/sync_lora_piper.yaml`
- `pixi run infer-async` -> `vlash run vlash/examples/inference/async.yaml`
- `pixi run infer-async-fast` -> `vlash run vlash/examples/inference/async.yaml --action_quant_ratio=2`

Sync + LoRA note:

- Yes, sync can use LoRA. LoRA and async/sync are separate choices.
- Async/sync is controlled by `max_delay_steps` (`8` for async style, `0` for sync style).
