# FastWAM LIBERO Rothko 训练与评测细节审查

> 审查日期：2026-08-07  
> 审查范围：LIBERO 数据采样、Rothko 表征、DiT/VAE 训练、验证、diffusion、语言条件和在线评测。  
> 本文记录当前代码事实、潜在影响及后续建议，不表示文中建议已经实现。

## 1. 总结与建议优先级

在继续投入新的长时间训练前，建议优先处理以下四项：

1. 将 DiT 数据集改成显式的完整窗口索引，避免 episode 尾部 padding 窗口参与训练。
2. 明确 suite、task、episode、window 之间的采样权重，并尽量统一 DiT、Rothko norm stats 和 VAE decoder 的数据分布口径。
3. 建立固定且有代表性的 validation manifest，而不是只验证训练集开头四个相邻窗口。
4. 修正梯度累计场景下的 train loss 日志；当前 W&B 只记录最后一个 microbatch，不是 effective global batch 的平均。

其次建议对 language padding mask 和 absolute EE pose condition 做受控消融。还应把固定 diffusion seed、OSC action clipping、无效的 CFG 配置等写入实验元数据，避免配置看起来已启用、实际却没有生效。

## 2. 数据采样与 padding

### 2.1 Dataset 长度按总帧数计算

当前 `BaseLerobotDataset.__len__()` 返回所有数据集的总帧数。每个原始帧都可能成为 17 帧 chunk 的起点，并未预先只建立完整窗口索引。

当前 LIBERO 配置还设置了：

```yaml
skip_padding_as_possible: false
```

因此 episode 最后 16 个起点也会被采样，越界的 observation/action 会被复制 episode 最后一行并标记为 padding。

四个 suite 中带有未来 padding 的起点比例为：

| Suite | Episodes | Frames | 完整 RGB 起点 | 尾部 padding 起点 | padding 比例 |
|---|---:|---:|---:|---:|---:|
| LIBERO-Spatial | 434 | 53,229 | 46,285 | 6,944 | 13.05% |
| LIBERO-Object | 457 | 67,309 | 59,997 | 7,312 | 10.86% |
| LIBERO-Goal | 433 | 52,895 | 45,967 | 6,928 | 13.10% |
| LIBERO-10 | 388 | 104,280 | 98,072 | 6,208 | 5.95% |

### 2.2 latent padding mask 以 4 个像素帧为一组

Wan VAE 的 temporal downsample factor 为 4。当前 loss mask 只有在一组 4 帧全部为 padding 时，才排除对应 latent：

```python
latent_future_is_pad = pixel_is_pad[:, 1:].reshape(
    batch_size, future_latent_frames, factor
).all(dim=2)
```

这意味着：

- `1 个真实帧 + 3 个 padding 复制帧` 仍被当成一个完整有效的 latent target。
- 完整 16 步窗口与只剩少量有效 future 的尾部窗口，在单个模态内部可能获得近似相同的 sample 权重。
- RGB future 从下一帧开始，Rothko future 从当前 action 开始，两种模态的 padding 边界并不完全相同。
- 被复制的 episode 尾帧会进入 VAE latent，并可能参与部分有效 latent group 的监督。

这是当前最值得优先修复的数据问题。建议 Dataset 直接枚举完整连续窗口，而不是在取样后再通过 padding mask 修补。

## 3. Suite、task、episode 的训练权重

### 3.1 四套件联合训练按帧数加权

当前 sampler 在整个拼接数据集的 frame index 上均匀采样。各 suite 实际训练权重约为：

| Suite | Frames | 训练权重 |
|---|---:|---:|
| LIBERO-10 | 104,280 | 37.5% |
| LIBERO-Object | 67,309 | 24.2% |
| LIBERO-Goal | 52,895 | 19.0% |
| LIBERO-Spatial | 53,229 | 19.1% |

因此 LIBERO-10 获得的训练权重接近 Goal 或 Spatial 的两倍。

### 3.2 suite 内部也不是 task-balanced

长 episode 会产生更多 frame/window 起点，因此对应 task 得到更多优化样本。当前没有：

- suite-balanced sampler；
- task-balanced sampler；
- episode-balanced sampler。

这不是必然错误，但必须明确目标究竟是按 frame、window、episode、task 还是 suite 加权。

### 3.3 三条训练链路使用了不同分布

当前实际存在三种数据权重口径：

1. DiT：按原始 frame 数采样，并包含 episode 尾部 padding 起点。
2. Rothko norm stats：每个 episode 最多均匀采样 32 个 action window，近似按 episode 加权。
3. VAE decoder：每个 task 留出固定 episode 后，枚举剩余数据的全部 action-valid window。

因此 norm stats、VAE decoder 与 DiT 看到的分布不一致。

## 4. Rothko norm stats 的具体设计

### 4.1 当前 center-frac=0.6 stats 仍由四套件共同计算

当前 Goal-only center-frac=0.6 实验使用的 stats 文件记录了四个 suite：

- 1,712 个 episode；
- 每个 episode 32 个窗口；
- 总计 54,784 个窗口；
- 876,544 个 future translation vector。

因此 Goal-only DiT 并没有使用 Goal-only translation stats，而是使用 all-four-suite stats。

### 4.2 每个 episode 最多只采 32 个窗口

当 episode 的 action-valid 窗口超过 32 个时，脚本通过 `linspace + round` 在 episode 内均匀选择最多 32 个起点，并不是遍历全部窗口。

这种口径近似让每个 episode 权重相等，但与按 frame/window 训练的 DiT 分布不同。

### 4.3 只统计 translation

经验统计只计算 chunk 起点坐标系下的 XYZ 相对位移：

- rotation/direction 使用解析范围 `[-dir_scale, dir_scale]`；
- gripper 使用解析范围 `[-1, 1]`；
- 只有 center translation 使用数据分位数。

### 4.4 使用绝对值的对称 Q99.95

每个 XYZ channel 分别统计 `abs(relative_position)` 的 99.95% 分位数，然后构造：

```text
[-bound_x, +bound_x]
[-bound_y, +bound_y]
[-bound_z, +bound_z]
```

这会舍弃正负方向分布的不对称性。约 0.05% 的绝对位移元素会超过对应 bound，并在 Rothko normalize 时被截断到 `[-1,1]`。

### 4.5 所有 horizon step 等权计入分位数

每个选中的窗口包含 16 个 future translation vector。第 1 步与第 16 步具有相同统计权重，但远期 displacement 往往更大，因此会更明显地影响高分位 bound。

## 5. Validation 的代表性

当前 data config 没有独立 `val`，因此：

```python
val_ds = train_ds
```

当前固定 validation 又只使用 index 0、1、2、3。这四个 index 通常来自：

- 同一个 suite；
- 同一个 task；
- 同一个 episode；
- 四个高度重叠的相邻窗口。

因此当前 `val_loss`：

- 不是 held-out generalization；
- 不代表全部 task/suite；
- 可能很早稳定；
- 不适合单独判断模型整体是否停止学习。

固定 diffusion timestep/noise 本身有利于横向对比，问题主要在固定样本组缺乏代表性。

建议建立显式 validation manifest，例如每个 task 固定 2～4 个 episode，再在每个 episode 内固定若干完整窗口。

## 6. 梯度累计与训练日志

### 6.1 W&B train loss 只记录最后一个 microbatch

梯度累计为 4 时，优化器确实累计四个 microbatch 的梯度；但只有触发 optimizer step 的最后一个 microbatch 被写入日志：

```python
loss, loss_dict = train_model.training_loss(sample)

if accelerator.sync_gradients:
    global_loss = gather(loss)
```

因此 W&B 上的曲线不是 effective global batch 的平均，而是最后一个 microbatch 的多卡均值。这会让曲线比实际累计 batch 更抖，也会影响对 plateau 和震荡的判断。

### 6.2 samples/sec 少乘了 gradient accumulation

当前吞吐量按：

```text
optimizer_steps_per_sec × batch_size × world_size
```

计算，但没有乘 `gradient_accumulation_steps`。例如 accumulation=4 时，日志里的 samples/sec 会比实际处理样本数小约 4 倍。

### 6.3 sampler 正常连续训练会逐 epoch reshuffle

虽然自定义 sampler 的 `epoch` 不在 trainer 循环中显式设置，但经过 `Accelerator.prepare()` 后，Accelerate 的 `DataLoaderShard` 会在每轮 iterator 开始时调用 sampler 的 `set_epoch()`。

因此“正常连续训练每个 epoch 完全同序”不是问题。该结论已经通过当前安装的 Accelerate 1.12.0 实现确认。

## 7. Language condition

### 7.1 padding token 全部被视为有效 token

训练和评测都执行：

```python
context[~context_mask] = 0.0
context_mask = torch.ones_like(context_mask)
```

真实语言可能只有十几个 token，但固定 128 个 token 全部参与 cross-attention。padding embedding 在进入带 bias 的 text projection 后不一定继续为零。

这是为保持 Wan2.2 既有行为而做的选择，但可能影响语言利用能力。由于改成真实 mask 会改变预训练分布，不建议无对照地直接修改，适合做受控消融。

### 7.2 `text_cfg_scale` 和 `negative_prompt` 对 visual-action 分支无效

虽然 eval config 暴露了：

```yaml
text_cfg_scale: 1.0
negative_prompt: ""
```

visual-action 分支没有执行 classifier-free guidance，相关参数不会改变当前模型输出。修改这些配置不能构成有效实验。

### 7.3 LIBERO 当前没有 proprio/absolute pose token

模型配置为：

```yaml
proprio_dim: null
```

因此 processor 虽然会生成 proprio，但 visual-action 模型不会使用它。

## 8. RAY0、绝对位姿与 modality identity

### 8.1 RAY0 不包含当前绝对 EE pose

LIBERO Rothko 以每个 chunk 的第一帧为基准：

- 第一帧相对位移为零；
- 第一帧相对旋转为 identity；
- RAY0 主要只保留当前 gripper 状态。

模型只能从 RGB 推断机械臂当前处于工作空间的哪个位置。最终 decode 时才用 simulator 的当前绝对 EE pose 将相对轨迹重新锚定。

该设计有利于相对运动泛化，但会丢失显式的工作空间边界、奇异位姿和绝对高度等信息，是需要实验验证的设计风险。

### 8.2 没有显式 modality embedding

horizon=16 时，每个模态 VAE encode 后有 5 个 latent frame，拼接布局为：

```text
RGB0 RGB1 RGB2 RGB3 RGB4 | RAY0 RAY1 RAY2 RAY3 RAY4
```

模型没有额外的 RGB/Rothko modality embedding，只通过以下信息区分两种模态：

- 像素/latent 内容；
- block 的 temporal RoPE 位置；
- attention mask；
- 在序列中的固定位置。

因此 RAY0 在 RoPE 中位于 latent time index 5，并不与 RGB0 共享同一时间位置。这符合“把动作图伪装成视频后半段”的预训练范式，但也是明确的结构假设。

### 8.3 单臂 Rothko 的左右两块完全重复

LIBERO 的 224×224 单臂 Rothko 被横向复制为 224×448，以匹配 `[agentview | wrist]` RGB canvas。两块并不对应两个相机或两只手，decode 时对两块 raw map 取平均。

## 9. Attention mask 的真实语义

当前 `rgb_then_raymap_block_causal` 的关系为：

- RGB0 与 RAY0 两个 clean condition query 只能看到彼此。
- Future RGB 可以看到完整 RGB block 和 RAY0，但不能看到 future Rothko。
- Future Rothko 可以看到完整 RGB block 和完整 Rothko block。
- Future RGB 内部不是严格逐时间 causal。
- Future Rothko 内部也不是严格逐时间 causal。

因此模型是在一次 diffusion 去噪中联合生成整个 RGB future 和整个 Rothko future，而不是自回归逐步预测。Rothko/action 可以利用同步生成的 future RGB trajectory。

## 10. Diffusion timestep、weight 和 loss 含义

### 10.1 shift=5 的训练 timestep 分布

训练先采样：

```text
u ~ Uniform(0,1)
sigma = 5u / (1 + 4u)
```

大致统计为：

- sigma 均值约 0.747；
- 中位数约 0.833；
- 75% 分位约 0.938；
- 90% 分位约 0.978。

因此采样明显偏向较高噪声。

但训练 loss weight 在 sigma≈0.5 时最高，在 0 和 1 附近接近零。因此实际目标是“高噪声样本更多，但中噪声 loss 权重更高”，不是简单只强调高噪声。

### 10.2 RGB 和 Rothko 共用 timestep

同一个 sample 内，RGB/Rothko 使用相同的 sigma/timestep，但 `randn_like` 为各个 latent 元素生成独立噪声。

### 10.3 `loss_*_raw` 不是普通未加权 MSE

当前：

```python
loss_rgb = (loss_rgb_per_sample * timestep_weight).mean()
loss_raymap = (loss_raymap_per_sample * timestep_weight).mean()
```

日志中的 `loss_rgb_raw` 和 `loss_raymap_raw` 已经乘过 diffusion timestep weight。`raw` 只表示还没乘 modality lambda。

因此：

- `loss_rgb_raw≈0.1` 不能当成普通 latent MSE；
- `loss_raymap_raw≈0.008` 不能直接换算成 EE pose 误差；
- 两者的绝对大小不能直接比较模态学习难度。

最终总 loss 使用：

```text
loss_total = 1 × loss_rgb + 5 × loss_raymap
```

## 11. VAE decoder 微调的独立设计

### 11.1 数据 split 与 DiT 不同

VAE decoder 微调会：

- 每个 task 固定留出 2 个 episode；
- 枚举剩余 episode 的全部 action-valid window；
- 训练时每个 epoch 重新 shuffle；
- validation 从留出 episode pool 中固定随机采样窗口。

这比 DiT 当前 validation 更合理，但它与 DiT 和 norm stats 的数据权重仍不同。

### 11.2 只训练 `conv2 + decoder`

VAE encoder 和 `conv1` 保持冻结，仅训练：

- `vae.model.conv2`；
- `vae.model.decoder`。

因此 latent encoder 空间不变，主要改变 Rothko latent 到像素 map 的恢复能力。

### 11.3 region loss 分别求均值

center、direction、gripper 三个区域分别求 masked L1，然后按照配置权重相加。当前 all-four-suite 配置中三项权重均为 1.0。

这意味着区域贡献由显式 loss weight 决定，而不是由像素数量决定。

### 11.4 存在不参与 loss 的过渡带

direction 区域排除了 center 外扩 `boundary_margin` 后的区域；center loss 只监督 center；outer border 单独监督 gripper。因此 center 外侧的一圈 transition band 不参与 direction/gripper loss，也不用于 rotation decode。

## 12. 在线评测与控制

### 12.1 每次 replan 使用相同 diffusion seed

当前 visual-action eval 每个 task、trial 和 replan 都使用全局 `cfg.seed`，默认 42。

因此每次规划的初始 latent noise 相同。观测和语言不同仍会使输出不同，但 noise-induced error pattern 可能具有较强相关性。

这样能让策略更确定、方便复现，但必须作为评测设计记录下来。

### 12.2 初始状态与环境 seed

每个任务使用 LIBERO 官方 initial state 列表，trial 按 index 依次取值；如果请求数量超过可用 initial state 数量，则通过取模重复。

环境本身还会使用 `cfg.seed`。代码注释指出，即使 initial state 固定，环境 seed 也可能影响物体位置。

### 12.3 评测先执行 30 个 dummy action

当前默认 `num_steps_wait=30`，环境 reset 并设置 initial state 后，先执行 30 步 dummy action 让场景稳定，再进行第一次模型规划。

训练数据来自 `no_noops` 数据集，不包含这段开头 idle action；模型看到的是等待结束后的实际 RGB/EE 状态。

### 12.4 absolute target 会被转换并裁剪为 OSC delta

模型 decode 得到的是 absolute EE target。执行每一步前，会相对当前真实 EE pose 转换成 LIBERO OSC command：

- 单步位置尺度为 0.05 m；
- 单步旋转尺度为 0.5 rad；
- 六个 motion dimension 分别 clamp 到 `[-1,1]`。

因此模型预测的目标如果相对当前实际 EE 太远，机器人不会一步到达该目标，只会执行裁剪后的最大 delta。后续 waypoint 又会基于新的实际状态重新转换。

训练专家 target 来自合法 OSC action，通常在控制范围内；模型 rollout 漂移后则可能频繁触发裁剪。只有打开 control trace 时才会保存 `clipped_dimensions`。

### 12.5 gripper 先连续 decode，再二值化

Rothko border 通过 median 解码成 `[0,1]` 连续 gripper 值。LIBERO eval 默认以 0.5 为阈值二值化，再转换成环境的 `-1=open, +1=close` 约定。

### 12.6 replan 的时间尺度

LIBERO 数据 fps 为 20：

- 16 步对应 0.8 秒；
- `replan16` 为预测 16 步并开环执行 16 步；
- `replan8` 为预测 16 步并执行前 8 步，即每 0.4 秒重规划。

默认关闭 action ensemble 时，不进行跨 chunk 平滑或加权平均。

## 13. 普通 dataset stats 在 visual-action 模型中的作用

当前 LIBERO visual-action 模型：

- 没有 action expert；
- `proprio_dim=null`；
- Rothko 使用 bypass processor 的 raw OSC pose/gripper side-channel。

因此普通 `dataset_stats.json` 虽然仍用于 processor 的 action/proprio normalization，但这些 normalized action/proprio 最终没有进入 visual-action DiT loss 或 inference。

真正决定动作表征尺度的是 Rothko norm stats，而不是普通 `dataset_stats.json`。

当前 checkpoint 仍校验 `dataset_stats.json` 指纹。从模型语义上看，这个校验较严格：它能保护实验配置一致性，但普通 dataset stats 的数值目前不会改变 LIBERO visual-action 的实际输出。

## 14. Dataset 错误处理并非完全 fail-fast

外层 `RobotVideoDataset` 的 `sample_error_mode=raise` 会让预处理异常直接报错。

但底层 `BaseLerobotDataset` 在加载视频或 parquet 失败时，会最多重试 5 次，并随机换一个 index：

```python
sample_idx = np.random.randint(len(self))
```

因此部分底层 I/O 错误可能被随机样本替换掩盖，外层无法感知。这与配置注释所表达的“corrupt/missing sample 直接失败”并不完全一致。

## 15. 优化器、冻结模块及运行细节

- Full fine-tuning 会训练整个 video DiT。
- VAE、text encoder 和 tokenizer 不参与 DiT full fine-tuning。
- 当前 LIBERO 没有 proprio encoder。
- AdamW 对所有可训练 DiT 参数统一使用 `weight_decay=0.01`，没有为 bias、LayerNorm 等建立 no-decay parameter group。
- AdamW betas 为 `(0.9, 0.95)`。
- gradient clipping 阈值为 1.0。
- 没有 EMA 模型，checkpoint/eval 直接使用当前训练权重。
- LR warmup 固定为总 optimizer steps 的 5%；训练 10 epoch 相当于前约 0.5 epoch warmup。
- cosine scheduler 最终 LR 为初始 LR 的 1%。
- `num_workers=8` 是每个 GPU process 8 个，四卡训练实际会创建最多 32 个 DataLoader worker。
- `pin_memory` 默认开启。
- 当最终 step 同时命中 `save_every` 时，当前代码可能在同一步执行一次周期保存和一次最终保存，产生重复 I/O。
- W&B 的显式 `_finish_wandb()` 当前未被训练主循环调用，通常依赖进程正常退出时自动 flush。

## 16. 已知但本轮不作为首要问题的事项

训练视频原始分辨率为 512×512，评测环境渲染为 256×256，二者最终都通过共享的 `build_libero_rgb_canvas()` resize 到每相机 224×224，并拼成 224×448。

该源分辨率差异仍然存在，但当前训练/评测已经统一：

- 输入类型处理；
- bilinear interpolation；
- `align_corners=False`；
- `antialias=True`；
- `[-1,1]` normalization；
- `[agentview | wrist]` 布局。

根据当前决策，暂不把 512×512 与 256×256 的源图差异作为首要修改项。

## 17. 建议的后续实施顺序

### 阶段 A：修复训练统计口径

1. 新建显式 `WindowIndex`，只包含完整 RGB+action 17 帧窗口。
2. 决定采样目标：优先建议 suite-balanced，再在 suite 内 task-balanced。
3. norm stats 支持显式 suite/task 选择和 `all_windows` 模式。
4. 在 stats metadata 中记录每个 suite/task/episode/window 的实际计数和权重。
5. 让 VAE、DiT、stats 尽可能共享同一 window manifest。

### 阶段 B：修复可观测性

1. 累计所有 microbatch 的 train loss 后再写 W&B。
2. 修正 samples/sec 的 gradient accumulation 计数。
3. 固定一份覆盖所有 task 的 validation manifest。
4. 增加 padding/clipping/saturation 统计：
   - `padded_sample_count`；
   - `partially_valid_latent_count`；
   - `rothko_saturation_rate_xyz`；
   - `eval_osc_clipped_dimension_count`。

### 阶段 C：模型设计消融

建议一次只改变一个变量：

1. 当前 Wan-style all-ones language mask vs 真实 language mask。
2. 无 proprio vs 添加当前 absolute EE pose token。
3. 无 modality embedding vs RGB/Rothko modality embedding。
4. 固定 diffusion seed vs 按 episode/replan 派生 seed。
5. all-four-suite stats vs suite-specific stats。

这些消融应使用相同数据 manifest、训练步数、VAE、eval initial states 和 replan 配置，避免多个变量同时变化。
