# FastWAM LIBERO-Plus 直接微调方案

> 状态：训练数据方案已确认，side-channel、2% source-group split 与固定 validation windows 已完成，尚未开始正式训练  
> 日期：2026-09-01  
> 已确认表示方式：两条主实验路线都使用 `center_frac=0.5`  
> 已确认初始化：Wan2.1 1.3B center=0.5 与 Wan2.2 5B center=0.5 并行进入后续 LIBERO-Plus 实验；center=0.6 仅保留为历史对照。  
> 已确认训练与选择目标：使用纯 LIBERO-Plus 数据直接微调；原始 LIBERO 不参与训练，也不作为 validation、早停、checkpoint 选择或最终评测判据。Plus 内部按 source group 留出约 2% validation。

## 1. 目标

从已经在原始 LIBERO 四套件上训练完成的两个 video-only Rothko checkpoint 出发，分别对 Wan2.1 1.3B 和 Wan2.2 5B 使用 LIBERO-Plus 扰动数据进行直接域适配。模型选择完全以 LIBERO-Plus 表现为准。

最终希望得到一个模型，同时满足：

- LIBERO-Plus 的相机、光照、背景、布局、噪声等扰动鲁棒性提高；
- 保持现有连续 17 帧输入、预测 16 步、`replan=8` 的控制范式；
- 不改变原始 LIBERO、LIBERO-Plus 和 RoboTwin 已有评测入口的默认行为；
- 所有训练、stats、VAE、checkpoint 和评测目录均可追溯且不会跨 benchmark 静默混用。

## 2. 当前事实

### 2.1 两条已确认的初始化路线

#### 路线 A：Wan2.1 1.3B，center_frac=0.5

```text
runs/libero_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-4/
  2026-08-28_17-06-22/checkpoints/weights/step_021700.pt
```

匹配 VAE：

```text
runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/
  Wan2.1_VAE_libero_rothko_step007498.safetensors
```

#### 路线 B：Wan2.2 5B，center_frac=0.5

```text
runs/libero_all4_rothko_2cam224_full_1e-4/
  2026-08-04_15-32-37/checkpoints/weights/step_021700.pt
```

匹配 VAE：

```text
runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2/
  Wan2.2_VAE_libero_rothko_step005600.safetensors
```

历史 Wan2.1 center=0.6 checkpoint 不进入主微调矩阵。`robust_joint` 等新解码方式属于独立解码消融，不参与本轮比较。两条主路线统一使用：

```text
legacy decoder
replan=8
ensemble=off
每任务 50 trials
```

### 2.2 LIBERO-Plus 数据规模

```text
dataset root: data/libero_plus/libero_plus_lerobot
tasks:        40
episodes:     14,347
frames:       2,238,036
fps:          20
cameras:      front + wrist
```

按 action trajectory SHA256 恢复出的真实独立专家轨迹只有 1,681 条。其余 episode 主要是同一专家动作轨迹在不同视觉扰动环境中的 replay，平均每条 source trajectory 对应约 8.5 个视觉 replay。

固定无泄漏切分：

```text
requested validation proportion: 2%
train: 1,641 source groups / 13,992 episodes / 2,179,641 frame starts
val:      40 source groups /    355 episodes /    58,395 frame starts
```

由于共有 40 个任务并要求每任务至少保留一个 validation source group，实际 group 比例为 `40 / 1681 = 2.38%`；每个任务恰好一个 validation group。同一 source trajectory 不得跨 train/val。

冻结文件：

```text
data/libero_plus/libero_plus_lerobot_source_manifest_val02_seed42.jsonl
SHA256: e166f06836dcb98b964468cd41c8ba68bf8b315b40ed0b8295d28258583bcc30
```

固定 offline validation window 清单：

```text
data/libero_plus/libero_plus_fixed_val_windows_val02_seed42_n40.json
SHA256: 69f79ae4f5a45eec83d188abe37a0b5a044b0042f975c0e1b1df55ba2c7d29fd
```

该清单从 355 个 held-out episodes 中固定选出：

```text
loss samples:   40（每个 base task 1 个完整 17 帧窗口）
visual samples:  4（每个 suite 1 个，为上述 40 个的子集）
diffusion timestep/noise: 由每个样本的 diffusion_seed 固定
```

清单已逐条与真实 `BaseLerobotDataset` 索引对照，40/40 个
`val_dataset_index` 均正确映射到指定的 `episode_index/frame_index`；
验证数据集长度为 58,395。该 validation 只用于监测训练异常和可视化，
不用于 early stopping、checkpoint 选择或正式成功率计分。

### 2.3 当前训练阻塞项

Plus parquet 已经完整写入并验证 video-only Rothko 训练所需的绝对 EE side-channel：

```text
observation.state.ee_pose_wxyz
action.osc_target_pose_wxyz
observation.state.gripper_open
action.gripper_open
```

14,347 episodes、2,238,036 帧已全部通过转换后的逐 episode 校验。
2% source-group split、fixed validation windows 和 Rothko stats 全窗口审计已完成。
已确认继续使用 source checkpoint 的 Original Rothko stats，并已接入 Plus
专用 train/val data config。普通 processor stats 也已确认沿用 clean LIBERO
`dataset_stats.json` 作为 checkpoint-compatible 身份文件。剩余主要数据
阻塞项是生成 Plus 语言 embedding cache。

## 3. 已确认的双 backbone 训练与评测流程

center fraction 已固定为 0.5。Wan2.1 1.3B 和 Wan2.2 5B 都直接训练完整 1 个 Plus epoch，不设置 280-task pilot，也不在 0.25 epoch 暂停决定是否继续。

两条路线保持相同：

```text
相同 Plus train/val split
相同的全部 train frame starts
相同 epoch 定义：无放回遍历全部训练样本一次
相同有效 global batch（显存允许时）
相同训练目标和 attention mask
最终评测均使用完整 10,030 项 LIBERO-Plus
replan=8
ensemble=off
legacy decoder
```

只允许以下内容按 backbone 匹配变化：

- 初始化 checkpoint；
- Wan 模型与 tokenizer/VAE 规格；
- checkpoint 对应的 VAE；
- checkpoint 对应的 Rothko stats fingerprint。两边几何都必须保持 center_frac=0.5。

最终分别报告两个 backbone 的 Plus 总成功率、四个 suite、40 个 base task 和七类扰动成功率。如果只能保留一个部署模型，再根据完整 Plus 成功率、各类别稳定性、速度和显存综合选择。

## 4. 正式训练协议

每个 backbone 只运行一组正式训练：

| 初始化 | 训练数据 | 训练长度 | 目的 |
|---|---|---|---|
| 对应的原始 LIBERO checkpoint | 100% Plus | 完整 1 epoch | 提高 LIBERO-Plus 扰动鲁棒性 |

因此主训练矩阵为 `2 backbones x 1 protocol`。不额外运行小规模零样本 pilot；已有零样本结果可以作为历史参考，但不阻塞训练。原始 LIBERO replay 不进入当前训练。

## 5. Plus-only 数据采样

第一轮直接沿用当前 FastWAM 的 epoch sampler，不新增 task-balanced 或 source-group-balanced sampler。

当前 `ResumableEpochSampler` 在每个 epoch 对 `len(dataset)` 生成一次无放回随机排列：

```text
indices = torch.randperm(len(dataset))
```

因此完整跑完 1 epoch 后，每个 Plus train frame start 都会被遍历一次。不同 source trajectory 的 replay 数和 episode 长度不同，确实会让它们贡献不同数量的训练窗口，但这些窗口本身就是当前 Plus 数据集定义的全部训练样本；对它们重新分层平衡会造成欠采样或过采样，并改变原始数据分布。

第一轮固定采用：

```text
数据范围：全部 Plus train frame starts
采样方式：每个 epoch 无放回 shuffle
epoch 数：1
不做 task/source/replay 重加权
```

只有未来决定训练少于一个完整 epoch，或者结果显示特定 base task 因数据量差异明显学习不足时，才把 balanced sampler 作为单独消融实验。

### 5.1 Plus 内部验证泄漏

Plus train/val 必须按 source trajectory hash 分组切分，避免同一专家动作轨迹的不同视觉 replay 同时进入训练集和验证集：

```text
Plus train 中的视觉 replay
    与 Plus validation 中的视觉 replay
    来自同一条专家 action trajectory
```

可以额外做 Original/Plus action hash 交叉审计，用来解释零样本能力和数据来源重合，但它不再决定当前 Plus-only 训练切分。最终结论以在线 LIBERO-Plus 评测为准。

### 5.2 固定 offline validation

Plus 专用 data config 显式使用同一份 source-group manifest：

```text
data.train.episode_split = train
data.val.episode_split   = val
val_set_proportion       = 0
```

因此 train/val 不再由 dataset 启动时的随机 episode split 决定。训练任务配置需显式设置：

```text
eval_sample_manifest: ./data/libero_plus/libero_plus_fixed_val_windows_val02_seed42_n40.json
```

每次 validation 对 40 个固定窗口计算 deterministic diffusion loss，并仅对
4 个固定子集运行完整去噪、解码、EE 误差与视频保存。这避免每次
validation 因为换了数据窗口、diffusion timestep 或 noise 而无法比较，同时把
完整去噪的额外开销限制在 4 个样本。未设置该字段的旧 LIBERO、RoboTwin 和历史任务仍走原验证逻辑。

## 6. Rothko 几何与 normalization stats

### 6.1 微调前后只能使用一套几何

当前两条路线均已固定 center_frac=0.5。源 checkpoint、Plus 训练、stats、VAE 和部署必须统一使用：

```text
相同 center_frac
相同 focal
相同 center_scale/dir_scale
相同 boundary_margin/outer_margin
相同 224x448 duplicate-horizontal 布局
```

禁止任一路线在 Plus 数据、stats、VAE 或部署阶段静默切换到 center_frac=0.6。

### 6.2 Plus 直接微调的 stats 策略

一次训练和对应部署始终只能使用一份 Rothko stats。虽然训练数据现在只有 Plus，但初始化 checkpoint 已经学习了原始 LIBERO stats 下的动作像素尺度，因此不能只因 domain 改变就无检查地切换 stats。

正确流程：

1. 为 Plus 添加 EE/OSC side-channel；
2. 在选定 center_frac 下统计 Plus 全部训练窗口的相对位移分布；
3. 计算 Original stats 应用于 Plus 时的每轴 clipping ratio；
4. 同时生成 Plus-only stats 作为候选；
5. 在正式训练前固定唯一 stats，训练、validation 和部署全程一致。

优先策略：

- 如果 source checkpoint 的 Original stats 在 Plus 上 clipping 仍接近目标 Q99.95 水平，优先继续使用 Original stats，以保持 checkpoint 已学到的动作到 Rothko 映射；
- 如果 Plus 明显超出 Original bound，再把 Plus-only stats 作为独立候选；
- 使用 Plus-only stats 会改变初始化 checkpoint 的动作像素尺度，因此必须先通过 codec roundtrip 和训练 smoke test，不能未经检查直接启动正式训练；
- 当前没有混合训练，不需要计算 union stats。

Plus 发布的 `norm_stats.json` 不用于 Rothko。审计已确认其 action mean 与 parquet action 表征不一致，而 video-only Rothko 路径本来也不需要普通 action/state processor stats。

### 6.3 2026-09-01 全窗口 stats 审计结果

两个 source checkpoint 实际共用以下 Original stats：

```text
data/libero_mujoco3.3.2/
  libero_rothko_region_symmetric_q99p95_h16_224x448.pt
file SHA256:       1e82a892d4c97f63e8198b354c33de1cd704ec15fffa8e902305631c1aeba4f6
codec fingerprint: fb75ea4665b40a822f5e19c4d54a4439384fe8865db7ca90aa6cc7350405a21c
bounds xyz (m):    [0.20725, 0.20486, 0.22343]
```

这份历史 stats 是每个 Original LIBERO episode 均匀采 32 个完整窗口统计的，
不是全窗口统计。本次审计没有沿用这种抽样方法，而是枚举 Plus train split 的：

```text
13,992 episodes
2,179,641 frame starts
34,874,256 relative translation vectors
episode 尾部使用 future_action_index=min(start+offset, episode_length-1)
```

Original stats 应用到 Plus 后的逐轴 clipping ratio：

| split | X | Y | Z |
|---|---:|---:|---:|
| train 全窗口 | 0.03613% | 0.04885% | 0.03359% |
| held-out val 全窗口 | 0.03157% | 0.04816% | 0.03318% |

三个轴在 train 和 held-out val 上都没有超过 Q99.95 对应的 0.05% 目标。

同时生成的 Plus-only 候选 stats：

```text
data/libero_plus/
  libero_plus_rothko_region_symmetric_q99p95_h16_centerfrac05_train_allwindows_224x448.pt
file SHA256:       adcde3694313d1cf5c638c7c934623671a32d21328a6ca6fe6102353148693a8
codec fingerprint: 802d6cd4b0bcf2362d58a5714e6634992f32239a4e13a150c2dd76245adbbdfc
bounds xyz (m):    [0.20413, 0.20473, 0.21831]
```

Plus-only bounds 相对 Original bounds 分别为 `98.49% / 99.94% / 97.71%`，即候选
stats 反而更窄。在 held-out val 上，Plus-only 候选的逐轴 clipping ratio 为
`0.04281% / 0.04923% / 0.04495%`，也没有产生更好的覆盖优势。

真实轨迹 codec roundtrip 与 metadata 严格加载测试已通过。实际轨迹中的非零
位置误差来自 Q99.95 超界后的预期 clipping，不是编解码公式错误；
因为 Plus-only bounds 更窄，其极端样本误差反而更大。

因此已确认的唯一正式 stats 是 **Original checkpoint-compatible stats**。
它已充分覆盖 Plus，同时不改变 source checkpoint 已学习的 Rothko 像素尺度。
Plus-only stats 仅保留为审计产物，不进入第一轮正式训练。

### 6.4 普通 processor `dataset_stats.json`

当前 LIBERO video-only 配置明确为：

```text
proprio_dim: null
video_dit_config.action_conditioned: false
```

因此 processor 归一化后的普通 `action` 和 `proprio` 不进入 DiT；Rothko
监督来自绕过 processor normalizer 的 raw EE/gripper side-channel。该 stats 在当前
路径中的作用是：

- 满足 processor 的结构要求；
- 作为 checkpoint 中的 dataset-stats identity；
- 在评测时做 fingerprint 一致性校验。

正式 Plus 配置因此沿用：

```text
data/libero_mujoco3.3.2/dataset_stats.json
file SHA256: 1e519a3ab1da802ba5731e0bd70728fa11d20e99d21378ebbf2671159cddb076
canonical fingerprint: b7089da10c0739d1226965844c89be932f9dac195e621f10ae16fa786578bf8f
```

该 canonical fingerprint 与 Wan2.1 source checkpoint 内嵌值完全一致。Wan2.2 source
checkpoint 较旧，未内嵌该字段，但其原训练 config 使用的也是同一文件。
本轮不计算 Plus processor stats。如果未来启用 proprio encoder、action expert 或
action-conditioned DiT，必须重新审计该决定。

## 7. Plus parquet side-channel

需要为每个 parquet 添加：

```text
observation.state.ee_pose_wxyz       [x,y,z,qw,qx,qy,qz]
action.osc_target_pose_wxyz          [x,y,z,qw,qx,qy,qz]
observation.state.gripper_open       [open]
action.gripper_open                  [open]
```

时间对齐固定为：

```text
窗口起点 t：
  Rothko frame 0     = observation EE pose at t
  Rothko frames 1:17 = action rows t:t+16 的 absolute OSC targets
  RGB frames 0:17    = observation rows t:t+17
```

保留此前已经确认的采样行为：

- 使用所有 frame start；
- episode 尾部继续 replication padding；
- 保留 image/action pad mask；
- 不改成只训练完整 17 帧窗口；
- stats 枚举必须复现相同尾部 padding 规则。

写入要求：

- 原字段逐值保持不变；
- 单 episode 临时文件后原子替换；
- 支持中断继续和内容校验；
- 更新 `meta/info.json`、episode stats 等 metadata；
- quaternion norm、action roundtrip 和时间对齐测试全部通过后才进入全量转换。

## 8. VAE 策略

### 8.1 DiT 训练

训练时直接加载对应架构的原始 Wan VAE，并冻结整个 VAE：

```text
Wan2.1 1.3B -> 原始 Wan2.1 VAE
Wan2.2 5B   -> 原始 Wan2.2 VAE
model.vae_safetensors_path=null
```

训练的 RGB/Rothko latent 监督只由 VAE `encoder + conv1` 产生。现有 Rothko VAE 微调只修改了 `conv2 + decoder`，不会改变 encoder latent，因此训练和 latent 预计算都不需要加载微调后的 VAE safetensors。

### 8.2 评测与部署

推理时需要把 DiT 预测的 Rothko latent 解码成 Rothko map，再恢复 EE action，因此评测/部署加载已有 center=0.5 decoder 权重：

```text
Wan2.1 1.3B -> Wan2.1 centerfrac05 VAE step7498
Wan2.2 5B   -> Wan2.2 centerfrac05 VAE step5600
```

这些完整 safetensors 中的 encoder 仍是原始权重，实际改动仅为 `conv2 + decoder`。训练 validation 的 latent loss 不依赖 decoder；如果训练期间需要观察解码后的 EE 指标，可以单独使用部署 decoder 运行 validation，但该指标不参与反向传播。

如果使用新的 Plus-only stats 后 VAE 对 Rothko map 的重建误差明显上升，再单独进行 Plus Rothko VAE decoder 微调。VAE 微调与 DiT 的 Plus 直接微调必须是两个独立实验阶段，并分别保存权重和评测结果。

## 9. 文本条件

Plus 训练数据 metadata 当前只有 40 条 canonical instruction。建立独立缓存：

```text
data/text_embeds_cache/libero_plus/
```

2026-09-01 已完成第一版缓存。当前决定是维持训练数据原状，不把
LIBERO-Plus `Language Instructions` 评测中的改写指令注入本轮训练；语言改写增强
留作后续独立实验。

缓存生成与兼容性检查结果：

```text
unique prompts: 40
context_len: 128
encoder id: wan22ti2v5b
每个 context: [128, 4096], bfloat16
每个 mask:    [128], bool
截断 prompt: 0/40
缺失/多余/非有限值: 0/0/0
总大小: 42,074,760 bytes
cache content fingerprint:
9c980acaab4643673a6b7098869bf76d40dcbe176a11d410c9bcb75f9d8f4616
```

Wan2.1 T2V 1.3B 和 Wan2.2 TI2V 5B 使用相同 UMT5 encoder/tokenizer，因此共用
上述缓存身份。直接按照 Plus `tasks.jsonl` 的混排顺序、每批 16 条重新编码时，
有 10 条 BF16 context 与 clean LIBERO 缓存不逐 bit 相同；prompt hash、token mask、
encoder 和 tokenizer 均一致。进一步按照原始 clean LIBERO 的四目录顺序重新编码后，
40/40 的 `context` 和 `mask` 均与旧缓存逐 tensor 相同，证明差异来自 batch 组合改变
后的 BF16 数值路径，而不是语言文本或模型权重不一致。为保证从 clean checkpoint
微调时的语言条件完全连续，最终 Plus 独立目录中的 40 个文件从
`data/text_embeds_cache/libero/` 原样复制，并已验证 40/40 逐字节一致。

多训练 epoch 不能自动解决 `Language Instructions` 扰动。如果最终完整 Plus 评测表明语言类明显较差，需要额外增加可验证的语言改写增强，而不是仅增加训练步数。

第一轮保持 text encoder 冻结，避免在只有 40 条 canonical instruction 的情况下破坏已有语言表示。

## 10. 建议的微调超参数

从原始 LIBERO 已训练 checkpoint 做 weight-only 初始化：

```text
resume=/path/to/checkpoints/weights/step_021700.pt
```

不能加载原始 LIBERO 的完整 state 目录。旧 optimizer、scheduler、epoch、sampler 和 dataloader 状态不属于 Plus 数据。

第一轮建议：

```text
finetune.method: full
learning_rate: 1e-5
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0
mixed_precision: bf16
VAE: frozen
text encoder: frozen
attention mask: 保持 checkpoint 对应配置
```

不沿用从原始 Wan 开始训练时的 `1e-4`。这次是已有策略的域适配，先用 `1e-5` 降低训练震荡和初始化能力被快速破坏的风险。

2026-09-01 已创建两份 Plus 专用正式配置：

```text
configs/task/libero_plus_all4_rothko_centerfrac05_2cam224_full_wan21_1_3b_1e-5.yaml
configs/task/libero_plus_all4_rothko_centerfrac05_2cam224_full_wan22_5b_1e-5.yaml
```

两份配置均为 4 GPU、effective global batch=128、1 Plus epoch，预计约
17,029 optimizer steps；Wan2.1 使用 `batch_size=4, grad_accum=8`，Wan2.2
使用 `batch_size=8, grad_accum=4`。两者都从对应 clean LIBERO
`step_021700.pt` 做 weight-only 初始化，显式使用固定 validation manifest、
原始冻结 VAE、clean Rothko stats 和 Plus 独立语言缓存。

原 trainer 曾将 warmup 固定为 5%。现已增加 `warmup_ratio` 配置项：缺省值仍为
0.05，保证旧任务行为不变；两份 Plus 配置显式设为 0.03，并将该字段加入完整状态
恢复兼容性指纹，避免使用不同 warmup 的 optimizer/scheduler state 被静默混用。

## 11. epoch 定义与训练长度

Plus-only 训练将一个 epoch 明确定义为覆盖一次 Plus train frame starts：

```text
1 Plus epoch = 期望采样完 2,179,641 个 Plus train frame starts
```

```text
Total Plus train samples: 2,179,641
```

对应 optimizer steps：

```text
effective global batch=256 -> 约 8,515 steps / Plus epoch
effective global batch=128 -> 约 17,029 steps / Plus epoch
```

第一轮连续训练完整 1 Plus epoch。中间 checkpoint 只用于故障恢复和事后分析，不触发在线 pilot 或提前筛选：

```text
0.25 epoch
0.50 epoch
0.75 epoch
1.00 epoch
```

每个阶段检查：

- 固定 Plus validation；
- RGB/Rothko train loss；
- deterministic validation loss；
- NaN、梯度异常或数据管线错误。

除非出现报错、NaN、持续梯度异常或确认的数据管线错误，否则不中途停止；validation plateau 不作为提前停止条件。1 epoch 完成后再根据完整 Plus 评测决定是否需要第二阶段。

## 12. checkpoint 与恢复策略

建议按照 1 Plus epoch 的总 steps 动态换算：

- weights：约每 0.25 epoch 保存；
- 完整 state：0.5 epoch 和最终各一份；
- validation：约每 0.1 epoch；
- W&B：独立 project/group/name，明确记录 backbone、center_frac、`training_domain=plus_only`、Plus manifest fingerprint 和 stats fingerprint。

完整 state 必须包含 optimizer、scheduler、global step、epoch、batch offset、sampler 和 RNG 状态。仅有 weights 只能作为新的 weight-only 微调起点，不能称为完整恢复。

## 13. 评测矩阵

### 13.1 训练前

- center_frac 固定为 0.5；
- 锁定两条路线各自的 checkpoint、VAE 与 stats fingerprint；
- 完成 dataset、codec、train/validation/save/reload smoke test。

### 13.2 训练中

定期运行固定的 Plus offline validation，用于发现训练异常；不运行在线仿真 pilot，也不据此提前停止正常训练。

### 13.3 最终

- LIBERO-Plus：完整 10,030 tasks x 1 trial；
- 按 suite、基础任务、七类扰动和难度分别汇总；
- 对齐比较两个 backbone 完成 1 epoch Plus 直接微调后的结果。

所有评测第一轮统一使用：

```text
legacy decoder
replan=8
ensemble=off
checkpoint 对应 VAE
checkpoint/Plus-training 对应的唯一 Rothko stats
```

新增加的 robust decoder 只能作为后续独立消融，不与训练收益同时改变。

## 14. 执行阶段

### Phase 0：冻结双 backbone 初始化

1. 固定两条路线各自的 checkpoint、VAE 和源 stats；
2. 固定 center_frac=0.5；
3. 不运行 280-task 在线 pilot。

### Phase 1：Plus 数据准备

1. 备份 metadata；
2. 单 episode side-channel dry-run；
3. 10 episode 写入与 simulator/controller 对齐验证；
4. 全量 side-channel 转换；
5. 按 Plus source trajectory hash 审计 replay 分组；
6. 固定无泄漏 train/val split；
7. 固定 deterministic validation windows；
8. 可选执行 Original/Plus action hash 交叉审计，仅用于数据来源分析。

### Phase 2：stats 与 dataset smoke test

1. 计算 Plus 全窗口分布；
2. 检查 source Original stats 对 Plus 全窗口的逐轴 clipping ratio；
3. 同时生成 Plus-only stats，通过覆盖率、codec roundtrip 和 smoke test 选定唯一 stats；
4. codec roundtrip；
5. 普通窗口和尾部 padding 窗口可视化；
6. [x] 生成并校验 Plus text embedding cache（40 条 canonical instruction，
   与 clean LIBERO 缓存逐字节一致）。

### Phase 3：训练 smoke test

```text
1 train step
1 deterministic validation
1 weights save
1 full-state save
checkpoint reload
同一样本再次 validation
```

### Phase 4：直接训练 1 Plus epoch

两个 backbone 分别连续训练完整 1 epoch。按 0.25/0.5/0.75/1.0 epoch 保存 weights，但不中途运行在线评测或决定是否继续；完整 state 只保留 0.5 和 1.0 epoch。

### Phase 5：最终完整评测

完成 LIBERO-Plus 全量评测，形成两个 backbone 微调结果的可复现对照表。

## 15. 启动正式训练前必须确认的决策

- [x] center_frac 固定为 0.5；
- [x] 主实验 backbone 为 Wan2.1 1.3B 与 Wan2.2 5B；
- [x] 两条初始化 checkpoint 路径；
- [x] 两条路线对应 VAE 路径；
- [x] 训练数据使用 100% LIBERO-Plus，原始 LIBERO 不参与训练；
- [x] source Original stats 可覆盖 Plus，正式训练保持 checkpoint-compatible stats；
- [ ] effective global batch 选择 128 或 256；
- [ ] 训练使用 4 卡还是 8 卡；
- [ ] 最大可用存储和 checkpoint retention；
- [ ] W&B project/group/name。

在以上事项确认前，不启动正式长训。
