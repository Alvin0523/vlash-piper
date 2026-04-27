# Training and Inference Workflow

This page documents the current PI0.5 workflow for this repository.

## What to Edit Before Training

Edit these fields in the YAML you plan to run:

- `dataset.repo_id`
- `dataset.root`
- `job_name`
- `output_dir`

YAML files:

- Async + LoRA: `vlash/examples/train/pi05/async_lora_piper.yaml`
- Sync (non-LoRA): `vlash/examples/train/pi05/sync_piper.yaml`
- Sync + LoRA: `vlash/examples/train/pi05/sync_lora_piper.yaml`

## Pixi Tasks

Training:

```bash
pixi run train-pi05-lora
pixi run train-pi05-sync
pixi run train-pi05-sync-lora
```

Inference:

```bash
pixi run infer-async
pixi run infer-async-fast
```

## Task to Command Mapping

- `train-pi05-lora` -> `vlash train vlash/examples/train/pi05/async_lora_piper.yaml`
- `train-pi05-sync` -> `vlash train vlash/examples/train/pi05/sync_piper.yaml`
- `train-pi05-sync-lora` -> `vlash train vlash/examples/train/pi05/sync_lora_piper.yaml`
- `infer-async` -> `vlash run vlash/examples/inference/async.yaml`
- `infer-async-fast` -> `vlash run vlash/examples/inference/async.yaml --action_quant_ratio=2`

## Async vs Sync

- Async mode: `max_delay_steps: 8`
- Sync mode: `max_delay_steps: 0`

LoRA is independent from async/sync. You can run LoRA in either mode.

## Local Dataset Location

This repo stores local data under:

- `data/`

The record script already writes local datasets to `./data`.
