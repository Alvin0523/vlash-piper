# 训练脚本对比：lerobot vs vlash

> 对比文件：
> - `lerobot_piper/src/lerobot/scripts/train.py`（301 行）
> - `vlash/vlash/train.py`（634 行）

---

## 一、总体架构对比

| 维度 | lerobot `train.py` | vlash `train.py` |
|---|---|---|
| 代码行数 | 301 行 | 634 行 |
| 分布式训练 | ❌ 单 GPU | ✅ `Accelerator`，支持多 GPU |
| 混合精度（AMP） | 手动 `GradScaler` + `torch.autocast` | `accelerator.autocast()` 自动管理 |
| 梯度累积 | ❌ 无 | ✅ `grad_accum_steps` |
| 数据集 | 标准 `LeRobotDataset` | 自定义 `VLASHDataset`（含时序延迟增强） |
| LoRA 支持 | ❌ 无 | ✅ 支持 LoRA，checkpoint 时自动 merge |
| 自动恢复 | 手动指定 `resume` | ✅ `auto_resume()` 自动检测 checkpoint |
| 仿真环境评估 | ✅ 训练中内嵌 `eval_policy` | ❌ 无（移到外部 eval 脚本） |

---

## 二、`update_policy` 核心训练步骤对比

### 2.1 函数签名

**lerobot**
```python
def update_policy(
    train_metrics, policy, batch, optimizer,
    grad_clip_norm,
    grad_scaler: GradScaler,   # 手动管理 AMP
    lr_scheduler=None,
    use_amp: bool = False,     # 是否开启 AMP
    lock=None,
)
```

**vlash**
```python
def update_policy(
    train_metrics, policy, batch, optimizer,
    grad_clip_norm,
    accelerator: Accelerator,  # 替代 GradScaler，自动管理 AMP
    lr_scheduler=None,
    lock=None,
    *,
    loss_scale: float = 1.0,          # 梯度累积缩放系数
    do_step: bool = True,             # 是否执行 optimizer.step
    use_shared_observation: bool = False,  # 共享观测前向
)
```

---

### 2.2 前向传播（Forward Pass）

**lerobot**
```python
with torch.autocast(device_type=device.type) if use_amp else nullcontext():
    loss, output_dict = policy.forward(batch)
```

**vlash**
```python
with accelerator.autocast():
    loss, output_dict = policy.forward(batch)
    raw_loss = loss.detach()   # 单独保存原始 loss 用于日志
    loss = loss * loss_scale   # 梯度累积缩放
```

> **差异**：vlash 额外保存 `raw_loss`，确保日志记录的是未缩放的真实 loss；
> lerobot 在无梯度累积时两者相同，但有梯度累积时 lerobot 无此机制。

---

### 2.3 反向传播（Backward Pass）与 GradScaler

这是两者最核心的差异。

**lerobot（显式 GradScaler）**

```python
# Step 1：GradScaler 将梯度放大（防止 fp16 下溢）
grad_scaler.scale(loss).backward()

# Step 2：裁剪前必须先 unscale（还原真实梯度量级）
grad_scaler.unscale_(optimizer)
grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

# Step 3：step 内部自动跳过含 inf/NaN 的更新
grad_scaler.step(optimizer)

# Step 4：动态调整放大系数（上调/下调）
grad_scaler.update()

optimizer.zero_grad()
```

**vlash（Accelerator 封装）**

```python
# Step 1：accelerator.backward 内部封装了 GradScaler.scale().backward()
accelerator.backward(loss)

# Step 2：clip_grad_norm_ 内部自动做 unscale
if grad_clip_norm > 0:
    grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
else:
    # 不裁剪但仍计算 norm 用于监控
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))

# Step 3：直接 step（unscale + skip inf/NaN 已封装进 optimizer）
optimizer.step()
optimizer.zero_grad()
```

#### GradScaler 原理说明

混合精度训练（AMP）中使用 fp16 计算，梯度值可能因过小而变为 0（下溢）。GradScaler 的作用：

```
前向：fp16 计算（速度快）
      ↓
loss × scale_factor → backward()   ← GradScaler 放大梯度
      ↓
unscale_(optimizer)                 ← 还原真实梯度量级
      ↓
clip_grad_norm_()                   ← 裁剪（基于真实量级）
      ↓
grad_scaler.step(optimizer)         ← 含 inf/NaN 则跳过此步
      ↓
grad_scaler.update()                ← 动态调整 scale_factor
```

vlash 的 Accelerator 把以上流程全部封装，行为完全等价，只是代码更简洁。

---

### 2.4 行为差异汇总

| 行为 | lerobot | vlash |
|---|---|---|
| AMP 是否开启 | 由 `use_amp` 配置决定 | 由 accelerator config 决定 |
| `unscale_` 时机 | 手动，裁剪前显式调用 | 自动，藏在 `clip_grad_norm_` 内 |
| inf/NaN 跳过更新 | `grad_scaler.step` 自动跳过 | accelerator 封装的 optimizer 跳过 |
| loss 日志值 | `loss.item()`（已含 loss_scale） | `raw_loss.item()`（未缩放的原始值） |
| 梯度裁剪条件 | 无条件裁剪 | `grad_clip_norm > 0` 才裁剪 |
| 梯度累积 | ❌ 不支持 | ✅ `loss_scale = 1/grad_accum_steps` |

---

## 三、数据集差异

### lerobot：标准数据集
```python
dataset = make_dataset(cfg)  # 标准 LeRobotDataset
```

### vlash：时序延迟增强数据集

vlash 的核心创新——`VLASHDataset` 实现**时序延迟增强（Temporal Delay Augmentation）**：

```
chunk_size = 50, max_delay_steps = 12

普通数据集查询：[idx, idx+1, ..., idx+49]（50 个动作）

VLASH 查询：
  offset = random(0, 12)
  query  = [idx+offset, idx+1+offset, ..., idx+49+offset]
```

**意义**：训练策略学会应对"过时观测"——观测可能比实际晚 0~12 步，策略需预测
机器人未来的位置而非当前位置，从而在异步推理时保持鲁棒性。

当 `shared_observation=True` 时，使用 `SharedObservationVLASHDataset`，
一次前向计算所有 offset 的结果，显著提升训练效率。

---

## 四、LoRA 支持（vlash 独有）

```python
# 应用 LoRA（在 Accelerator 封装前）
apply_lora(cfg.lora, policy, verbose=is_main_process)

# Checkpoint 时合并 LoRA 权重，输出推理就绪的完整模型
if cfg.lora.enable and is_lora_policy(unwrapped_policy):
    merged_policy = clone_and_merge_lora_policy(unwrapped_policy, cfg.lora, ...)
    save_checkpoint(..., policy=merged_policy, ...)
```

lerobot 不支持 LoRA，每次 checkpoint 保存完整模型权重。

---

## 五、单 GPU 下去除 Accelerator 的等价替换

若只有单 GPU，可将 vlash 的 Accelerator 调用替换为直接 PyTorch 调用：

| vlash（Accelerator） | 替换为（原生 PyTorch） |
|---|---|
| `accelerator.autocast()` | `torch.autocast('cuda') if use_amp else nullcontext()` |
| `accelerator.backward(loss)` | `grad_scaler.scale(loss).backward()` |
| `accelerator.clip_grad_norm_(...)` | `grad_scaler.unscale_(optimizer)` + `torch.nn.utils.clip_grad_norm_(...)` |
| `optimizer.step()` | `grad_scaler.step(optimizer)` + `grad_scaler.update()` |
| `accelerator.unwrap_model(policy)` | 直接使用 `policy` |
| `accelerator.prepare(...)` | 删除，手动 `.to(device)` |

替换后，vlash 的训练逻辑与 lerobot 本质完全相同，
仅保留 `loss_scale`（梯度累积）和 `raw_loss`（日志分离）两点小差异。

---

## 六、依赖风险（Jetson AGX / aarch64）

| 包 | lerobot | vlash | aarch64 风险 |
|---|---|---|---|
| `torch` (JetPack) | ✅ 共用 | ✅ 共用 | 需用 JetPack 专供 wheel |
| `GradScaler(device.type, ...)` | 🔴 新 API，需 PyTorch ≥ 2.4 | ✅ 无此调用 | lerobot 有风险 |
| `shutup` | 🔴 依赖 | ✅ 无 | 需额外安装 |
| `accelerate` | ✅ 无依赖 | 🟡 依赖 | 需与 JetPack PyTorch 版本匹配 |
| `bitsandbytes` | ❌ 无 | 🟡 间接依赖 | 需从源码编译 |
