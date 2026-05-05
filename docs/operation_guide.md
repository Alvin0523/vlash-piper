---
icon: lucide/play-circle
---

# Operation Guide

Full step-by-step workflow for hardware setup, data collection, fine-tuning, and inference on the PIPER arm with VLASH.

---

## 1. Clone Recursively

```bash
git clone --recursive https://github.com/Alvin0523/vlash-piper.git
cd vlash-piper
```

Already cloned without `--recursive`?

```bash
git submodule update --init --recursive
```

## 2. Install Environment

```bash
pixi install
```

!!! note
    Training tasks preload Pixi's `libstdc++.so.6` to avoid a `pyarrow` `GLIBCXX` mismatch on Jetson / aarch64.

---

## 3. Hardware Checks

Verify the camera is detected:

```bash
pixi run cam_check
```

Set up CAN interfaces (`can2`, `can3`):

```bash
pixi run can
```

Confirm CAN interfaces are live:

```bash
ip -br link show type can
```

---

## 4. Data Collection

The recorded dataset for this project is published on Hugging Face:

> **Dataset:** [huggingface.co/datasets/Frieddeli/vlash](https://huggingface.co/datasets/Frieddeli/vlash)

To record your own episodes:

```bash
pixi run record
```

This calls `scripts/record.sh`. Edit that file to configure:

- `DATASET_NAME` — local folder name for the dataset
- `NUM_EPISODES` — how many episodes to record
- `TASK_DESC` — natural language task description (must match training)

Recorded datasets land in `data/`.

---

## 5. Fine-Tune (π0.5)

Before training, edit the following fields in the YAML you plan to run:

| Field | Description |
|---|---|
| `dataset.repo_id` | HuggingFace dataset repo ID |
| `dataset.root` | Local path to your dataset (under `data/`) |
| `job_name` | Experiment name for logging |
| `output_dir` | Checkpoint output path (under `models/`) |

```bash
# Async + LoRA — recommended for Jetson deployment
pixi run train-pi05-lora

# Sync — no async delay modeling, no LoRA
pixi run train-pi05-sync

# Sync + LoRA
pixi run train-pi05-sync-lora
```

!!! tip
    **Async vs Sync** is controlled by `max_delay_steps` in the YAML: `8` = async style, `0` = sync style.
    LoRA is a separate independent choice — sync configs can also use LoRA.

---

## 6. Download the Model

The fine-tuned π0.5 checkpoint for this project is published on Hugging Face:

> **Model:** [huggingface.co/Frieddeli/vlash](https://huggingface.co/Frieddeli/vlash)

Download it into `models/` before running inference:

```bash
huggingface-cli download Frieddeli/vlash --local-dir models/vlash-pi05
```

Then set the path in your inference YAML:

```yaml
policy:
  path: models/vlash-pi05
```

---

## 7. Inference

```bash
# Async inference — requires desktop GPU with torch.compile support
pixi run infer-async

# Async inference with 2× action quantization speedup
pixi run infer-async-fast
```

!!! warning "Jetson Orin"
    `torch.compile` is not supported on the Jetson nvgpu driver. Use sync inference configs on Jetson:
    ```yaml
    compile_model: false
    inference_overlap_steps: 0
    ```
    Use `vlash/examples/inference/sync_piper.yaml` which already has both set correctly.
    See [Problems & Diagnostics](problems.md) for the full root cause analysis.

---

## 8. Pixi Task Reference

| Task | Underlying Command |
|---|---|
| `pixi run train-pi05-lora` | `vlash train vlash/examples/train/pi05/async_lora_piper.yaml` |
| `pixi run train-pi05-sync` | `vlash train vlash/examples/train/pi05/sync_piper.yaml` |
| `pixi run train-pi05-sync-lora` | `vlash train vlash/examples/train/pi05/sync_lora_piper.yaml` |
| `pixi run infer-async` | `vlash run vlash/examples/inference/async.yaml` |
| `pixi run infer-async-fast` | `vlash run vlash/examples/inference/async.yaml --action_quant_ratio=2` |
| `pixi run record` | `scripts/record.sh` |
| `pixi run cam_check` | Camera detection check |
| `pixi run can` | CAN interface initialisation (`can2`, `can3`) |
| `pixi run docs-build` | Build the Zensical docs site into `site/` |
| `pixi run docs-serve` | Serve the docs locally with live reload at `localhost:8000` |
