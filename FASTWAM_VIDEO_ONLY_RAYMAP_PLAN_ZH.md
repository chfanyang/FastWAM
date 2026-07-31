# FastWAM Video-Only Raymap Action 方案与备选设计

最后更新：2026-07-28

## 1. 文档目的

本文记录将 FastWAM 改造成“仅使用 Wan Video Expert，通过生成
Rothko/Raxel Raymap 预测机器人动作”的最终第一版方案，以及讨论过但暂未选择的
备选方案。

本文用于：

- 固化已经确认的架构决策，避免后续实现时发生歧义；
- 给出数据、模型、训练、推理和 RoboTwin 部署的完整改造路径；
- 保留未选择的方案，便于后续消融和失败回退；
- 记录当前仍需通过实验决定的细节。

本文最初用于方案确认；截至 2026-07-24，第一版 Rothko 路径已经完成代码实现与
真实 Wan2.2 单 batch 验证。本文后半部分继续保留未选择的备选方案，便于后续消融。

当前第一版采用连续 Temporal RoPE：

```text
RGB latent:    0, 1, 2, 3, 4
Rothko latent: 5, 6, 7, 8, 9
```

不是分别重置成两组 `0...4`。

## 1.1 当前实现状态

已完成：

- 全部 27,500 个 episode 增加 `action.endpose` 和
  `observation.state.endpose`；
- 生成 horizon 16、上下复制布局对应的 Rothko normalization stats；
- 实现纯 PyTorch Rothko encode/decode 与 gripper 外边界编码；
- Dataset 同频返回 17 帧 RGB、17 帧 Rothko、raw EE pose 和 normalized qpos；
- 移除新路径中的 ActionDiT/MoT，只实例化 Wan Video Expert；
- RGB/Rothko 分别 VAE encode，再按 latent 时间 block 拼接；
- 实现条件 latent 索引 `[0,5]`、连续 RoPE `0...9` 和双条件 attention mask；
- 实现 RGB/Rothko 两段 latent flow-matching loss；
- 实现联合 RGB/Rothko 去噪、Rothko decode 和 16D EE action 输出；
- RoboTwin policy 自动识别新模型并使用 `action_type="ee"`，旧模型仍使用 qpos；
- 为第一批 8 个 RoboTwin 任务完成 17,748 条唯一语言指令的 embedding 缓存；
- 在物理 GPU 6、7 上完成 100-step、双卡 ZeRO-1 真实训练并成功保存 checkpoint；
- 增加原生 LoRA、adapter-only checkpoint 以及 full/LoRA 双加载路径；
- 增加 rank-64、attention + FFN 的单任务 `click_alarmclock` 配置；
- 当前单任务配置关闭 qpos/proprio token，只保留语言作为 cross-attention 条件；
- 将旧 `condition_frames_causal` 改为与 `[RGB | Rothko]` 布局匹配的
  `rgb_then_raymap_block_causal`；
- 将默认 Rothko loss 权重由 1 提高为 5；
- RoboTwin rollout 保持 16 步开环 action chunk，但每个环境步都获取新观测用于流畅录像；
- 支持为每次 replan 分别保存模型预测的 RGB 和 Rothko MP4；
- EE decode 的当前绝对位姿 anchor 改为 FP32，避免 BF16 量化误差；
- checkpoint 新增 attention-mask 语义校验，并可从旧 run 的 `config.yaml` 恢复缺失元信息。

真实模型验证结果：

```text
Wan2.2 + RoboTwin batch size = 1
forward loss = 0.2496
RGB loss = 0.1117
Rothko loss = 0.1379
DiT gradient tensors = 825
proprio encoder gradient = yes
forward/backward peak allocated = 20.23 GiB

1-step inference output:
RGB = [1,3,17,384,320]
Rothko = [1,3,17,384,320]
pose = [1,17,14]
EE action = [1,16,16]
inference peak allocated = 12.98 GiB

original Wan VAE roundtrip, 4 个分散 validation 窗口:
position mean / P95 / max = 0.60 / 1.97 / 2.98 mm
rotation mean / P95 / max = 0.314 / 0.787 / 1.417 degree
gripper MAE / max = 0.000061 / 0.001953
```

8 任务 100-step smoke training 结果：

```text
run = runs/robotwin_video_only_rothko_3cam_384_1e-4/2026-07-24_11-44-25
train samples = 1,268,383
val samples = 13,744
per-GPU batch size = 1
global batch size = 2
step 100 total loss = 0.1612
step 100 RGB loss = 0.1193
step 100 Rothko loss = 0.0418
step 100 learning rate = 1.0e-6
speed near the end = 0.92 step/s, 1.85 samples/s
```

这次运行没有出现 OOM、NaN 或 Inf。推理权重已经在 CPU 上完整反序列化验证，
ZeRO optimizer/model/scheduler/random state 也都已落盘。由于本次
`wandb.enabled=false` 且标准输出没有重定向到文件，不能在运行结束后恢复完整的
1--100 step loss 曲线；上面记录的是训练终端中的最终一步指标。

当前上下两份 Rothko 的第一版解码实现是“像素平均后再解码”。只使用上半份、上下
分别解码后融合仍保留为后续消融。

## 2. 目标

最终目标是利用预训练视频 diffusion model 的生成能力，把机器人动作表示为视频式的
Raymap，并通过预测未来 Raymap 得到机器人动作。

核心动机是：

- 不再为 action 使用独立的 Action Expert；
- 不再把连续动作向量当作与视频完全不同的模态；
- 把动作转换为三通道图像，使 RGB 和 action 都经过 Wan VAE；
- 让唯一的 Wan Video Expert 同时预测未来 RGB 和未来 Raymap；
- 尽可能保持预训练 video diffusion 的输入、latent、去噪和输出范式；
- 缩小图像模态与机器人动作模态之间的表示差距。

新模型不是“只预测 action、不建模未来世界”，而是必须同时生成：

```text
未来 RGB
未来 Rothko/Raxel Raymap
```

Raymap 再被解析成机器人 EE action。

## 3. 相关现有代码

### 3.1 FastWAM

重点文件：

```text
src/fastwam/models/wan22/fastwam.py
src/fastwam/models/wan22/wan_video_dit.py
src/fastwam/models/wan22/wan_video_vae.py
src/fastwam/models/wan22/mot.py
src/fastwam/models/wan22/action_dit.py
src/fastwam/datasets/lerobot/robot_video_dataset.py
src/fastwam/datasets/lerobot/base_lerobot_dataset.py
src/fastwam/datasets/lerobot/processors/fastwam_processor.py
src/fastwam/trainer.py
src/fastwam/runtime.py
experiments/robotwin/fastwam_policy/deploy_policy.py
configs/data/robotwin.yaml
configs/model/fastwam.yaml
configs/task/robotwin_uncond_3cam_384_1e-4.yaml
```

### 3.2 RoboTwin Raymap

参考实现：

```text
/mnt/hwdata/cfy/wam/RoboTwin/rothko_raymap_vae_sim_replay.py
/mnt/hwdata/cfy/wam/RoboTwin/raxel_vae_sim_replay.py
/mnt/hwdata/cfy/wam/RoboTwin/add_gripper_pose.py
```

Rothko normalization 与验证结果：

```text
/mnt/hwdata/cfy/wam/RoboTwin/ROTHKO_FINAL_VALIDATION_RESULTS.md
/mnt/hwdata/cfy/wam/RoboTwin/ROTHKO_50X5_ALL_RESULTS.md
/mnt/hwdata/cfy/wam/RoboTwin/ROTHKO_NORM_COMPARISON_RESULTS.md
```

当前 Rothko 正式基线参数：

```text
representation = rothko_raymap
focal = 0.2
center_scale = 1
dir_scale = 1
center_frac = 0.5
boundary_margin = 8
normalization = region-aware symmetric Q99.95
dtype = bfloat16
```

对应旧版 stats：

```text
/mnt/hwdata/cfy/wam/RoboTwin/experiments/rothko_norm/
region_symmetric_q99p95_cs1_ds1_f0.2_cf0.5.pt
```

注意：该 stats 对应旧的 16-frame chunk 和下半部分 padding 布局。新方案改变了
窗口定义与下半部分内容，正式训练前需要生成匹配新布局的 stats。

## 4. 当前 FastWAM 基线

### 4.1 数据时序

当前 RoboTwin 配置：

```text
observation frames = 33
action horizon = 32
action_video_freq_ratio = 4
RGB sample indices = 0, 4, 8, ..., 32
RGB frames = 9
```

三个相机被拼成一张 `384 x 320` RGB：

```text
┌────────────────────┐
│ head camera        │ 256 x 320
├──────────┬─────────┤
│ left     │ right   │ 128 x 160 each
└──────────┴─────────┘
```

### 4.2 当前模型

当前 FastWAM 包含：

```text
Wan Video Expert
ActionDiT Action Expert
MoT mixed attention
Wan VAE
text encoder/tokenizer
proprio encoder
video scheduler
action scheduler
```

训练同时优化：

```text
loss_video
loss_action
```

当前 `infer_action()` 为了加速，只预填充一次当前视频帧的 video K/V，之后主要运行
Action Expert 的去噪，不生成未来 RGB。

### 4.3 Released FastWAM Video Expert 的性质

Released RoboTwin checkpoint：

```text
checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt
```

checkpoint 元信息：

```text
step = 29355
video expert tensors = 825
action expert tensors = 824
has proprio_encoder = true
```

它最初来自 Wan2.2 TI2V 5B，但随后在 RoboTwin 数据上参与完整 FastWAM 训练。

它已经适配：

- RoboTwin 三相机布局；
- `384 x 320` RGB；
- 9 帧 RGB 时序；
- RoboTwin 语言与任务分布；
- 当前 qpos proprio；
- 机器人运动视频。

但它不是纯 RGB domain adaptation：

- Video query 不读取 Action token；
- Action query 可以读取第一帧 Video K/V；
- action loss 因此可以通过第一帧 Video K/V 路径间接更新 Video Expert；
- Video Expert 同时收到 RGB latent loss 的梯度。

第一版最终决定不使用该 checkpoint，而使用原始 Wan2.2 权重。Released Video
Expert 保留为后续初始化消融方案。

## 5. 已确认的第一版方案

### 5.1 模型初始化

已确认：

```text
init_source = original_wan22
```

具体含义：

- Wan Video Expert 从原始 Wan2.2 TI2V 5B 初始化；
- Wan VAE 使用原始 Wan2.2 VAE；
- 文本模型使用原始 Wan 文本组件；
- 不加载 released FastWAM checkpoint；
- 不实例化 ActionDiT；
- 不实例化 MoT；
- proprio encoder 随机初始化并训练。

### 5.2 唯一生成 Expert

新模型只保留：

```text
Wan Video Expert
```

训练器兼容接口：

```text
self.video_expert = video_expert
self.dit = self.video_expert
```

需要删除或停用：

```text
self.action_expert
self.mot
train_action_scheduler
infer_action_scheduler
ActionDiT checkpoint
MoT attention mask
MoT KV cache inference
```

### 5.3 保留当前 proprio token

已确认第一版保留 14D qpos proprio。

数据内容：

```text
left 6 joints
left gripper
right 6 joints
right gripper
```

训练时只使用当前窗口第一时刻：

```text
proprio = proprio[:, 0, :]
```

编码方式：

```text
14D normalized qpos
    -> Linear(14, 4096)
    -> one proprio token
    -> append after language tokens
```

完整条件为：

```text
language tokens
current qpos proprio token
current RGB latent
current Rothko latent
```

proprio token 只是 Video Expert 的 cross-attention 条件，不是 Action Expert，也不被
模型预测。

选择它的原因：

- 相对 Rothko 的第 0 帧通常是零平移与单位旋转；
- 第 0 帧 Rothko 不表达绝对 EE 位姿或完整关节构型；
- RGB 虽然能看到机械臂，但 qpos token 能明确提供当前构型；
- 解码时虽然可以在模型外使用当前 EE 基准，模型预测相对运动时仍受益于当前状态。

### 5.4 Horizon 与频率

第一版已确认：

```text
future action horizon = 16
current condition frame = 1
pixel frames per modality = 17
```

RGB、Rothko 和 action 改为同频，不再保持旧的 4:1 action/RGB 比例。

建议时序：

```text
RGB frame 0
    = current observation RGB

RGB frame 1...16
    = future observation RGB after action 0...15

Rothko frame 0
    = current observation.state.endpose

Rothko frame 1...16
    = future action.endpose 0...15
```

所有未来 Rothko 都以当前 `observation.state.endpose` 为同一个相对位姿基准，不在
16 帧内部再次切 chunk 或更换基准。

### 5.5 RGB 与 Rothko 分别经过 VAE

已确认不进行空间拼接，也不把 RGB/Rothko 的像素帧先串起来再统一经过 VAE。

采用：

```text
RGB video [B, 3, 17, 384, 320]
    -> Wan VAE
    -> z_rgb [B, 48, 5, 24, 20]

Rothko video [B, 3, 17, 384, 320]
    -> the same Wan VAE
    -> z_ray [B, 48, 5, 24, 20]
```

两者使用同一个冻结的 Wan VAE 实例与相同权重，但分别调用 encode/decode，避免
VAE temporal convolution 在 RGB/Rothko 模态边界处混合。

### 5.6 Latent 在时间维分段拼接

已确认采用 block，而不是交叉排列：

```text
z_all =
[
    RGB_0, RGB_1, RGB_2, RGB_3, RGB_4,
    RAY_0, RAY_1, RAY_2, RAY_3, RAY_4
]
```

形状：

```text
z_all [B, 48, 10, 24, 20]
```

Wan DiT patch size 是 `[1, 2, 2]`：

```text
tokens per latent frame = 12 * 10 = 120
total tokens = 10 * 120 = 1200
```

当前 FastWAM 约 360 tokens，因此：

```text
token 数约 3.3 倍
self-attention 理论计算量约 11 倍
```

这需要在单 batch smoke test 中实测显存。

### 5.7 Rothko 图像布局

原参考实现中，每个时刻的双臂有效 Rothko 位于上半部分：

```text
top:    [left arm | right arm] = 192 x 320
bottom: padding                = 192 x 320
```

第一版已确认把下半部分改成上半部分的复制：

```text
┌──────────────────────────────┐
│ left-arm       | right-arm   │ 192 x 320
├──────────────────────────────┤
│ left-arm copy  | right copy  │ 192 x 320
└──────────────────────────────┘
              384 x 320
```

每个单臂 tile 为：

```text
192 x 160
```

上下两份包含相同 Rothko 和相同 gripper 编码。

解码建议：

- 上下两份共同参与解码；
- 中心平移区域使用联合中位数；
- 方向区域合并后统一做 Kabsch/Wahba SVD；
- 或先分别解码上下两份，再对平移和旋转做稳健融合；
- 两种解码融合方式都应与单份解码做离线误差比较。

### 5.8 Gripper 视觉编码

Rothko/Raxel 原实现只编码：

```text
left xyz + quaternion
right xyz + quaternion
```

原 replay 中 gripper 直接来自 expert joints，并不是从 Raymap 解码得到。新模型需要
自行预测完整 action，因此必须把左右 gripper 编码进图像。

已确认利用 Rothko EE 解码忽略的外侧 8 像素边界。

建议每个 `192 x 160` 单臂 tile：

```text
outer margin = 8 pixels
border pixels = corresponding gripper value
interior pixels = Rothko origin/direction representation
```

上下副本写入相同值。

若原始 gripper 已在 `[0, 1]`：

```text
gripper_pixel = 2 * gripper - 1
```

解码：

```text
gripper_pixel_pred = median(all valid duplicated border pixels)
gripper_pred = clamp((gripper_pixel_pred + 1) / 2, 0, 1)
```

若原始 gripper 不是 `[0, 1]`，则使用数据集范围或固定机器人控制范围做线性映射。

需要验证 VAE 空间卷积导致的边界信息向内部泄漏。可采用：

```text
outer 4 pixels = gripper code
next 4 pixels  = guard band
decoder margin = 8
```

作为更保守的备选边界布局。

### 5.9 Diffusion 训练

两段 latent 拼接后使用同一个 video flow-matching scheduler。

第一版建议每个 batch sample 只采样一个共同 timestep：

```text
t_rgb = t_ray = t
```

构造：

```text
noise_all = randn_like(z_all)
z_noisy = add_noise(z_all, noise_all, t)
target = training_target(z_all, noise_all, t)
```

两个已知条件 latent 不加噪：

```text
RGB_0
RAY_0
```

每次训练前向与推理 scheduler step 后都恢复这两个 latent。

### 5.10 双条件 attention mask

当前 `first_frame_causal` 只支持一个条件 latent frame。新方案需要扩展成
`multi_condition`。

建议 mask：

```text
query \ key    RGB_0  RGB_future  RAY_0  RAY_future
RGB_0            yes       no       yes       no
RGB_future       yes       yes      yes       yes
RAY_0            yes       no       yes       no
RAY_future       yes       yes      yes       yes
```

含义：

- `RGB_0` 与 `RAY_0` 都是当前已知条件，可以互相读取；
- 两个条件 query 都不能读取未来；
- 所有未来 RGB/Rothko query 可以读取两个条件；
- 所有未来 RGB/Rothko token 可以联合双向注意，完成联合去噪；
- 不存在 action token 或跨 Expert attention。

### 5.11 Latent 监督

第一版已确认只做 RGB 与 Raymap latent 的 flow-matching 监督。

不做：

```text
VAE decode 后图像 loss
Rothko decode 后 position loss
rotation geodesic loss
direct gripper loss
joint qpos loss
```

loss 按两个 block 分开统计：

```text
loss_rgb = MSE(pred_rgb_future, target_rgb_future)
loss_ray = MSE(pred_ray_future, target_ray_future)

loss_total =
    lambda_rgb * loss_rgb
  + lambda_ray * loss_ray
```

两个条件 latent 不计入 loss。

padding mask 也必须按两个 block 分别构建，再映射到 VAE latent 时间步。

### 5.12 推理

推理流程：

1. 读取当前三个相机 RGB，并生成 `384 x 320` RGB；
2. 读取当前 14D qpos，并生成 proprio token；
3. 读取当前左右 EE 位姿；
4. 用当前 EE 位姿生成 Rothko frame 0；
5. 将当前 RGB 单帧通过 VAE 得到 `RGB_0` 条件 latent；
6. 将当前 Rothko 单帧通过 VAE 得到 `RAY_0` 条件 latent；
7. 分别为未来 RGB latent 与未来 Raymap latent初始化噪声；
8. 拼成 `[RGB block | RAY block]`；
9. 每个 denoising step 都运行完整 Wan Video Expert；
10. 每一步恢复 `RGB_0` 和 `RAY_0`；
11. 去噪结束后拆出 RGB/Raymap latent block；
12. 分别通过同一个 VAE decode；
13. 丢弃两个序列的 frame 0；
14. 将 Rothko frame 1...16 解码为 EE pose 与 gripper；
15. 返回未来 RGB 与 16 个 EE actions；
16. RoboTwin 执行前 `replan_steps` 个 action，然后重新规划。

与当前 FastWAM 不同，新推理不能只预填充一次 Video K/V。因为未来 RGB 和 Raymap
都在被生成，所以每一个 diffusion step 都必须运行完整 Video Expert。

### 5.13 RoboTwin 控制

已确认从当前：

```python
task_env.take_action(action, action_type="qpos")
```

切换为：

```python
task_env.take_action(action, action_type="ee")
```

每个解码后的 action：

```text
left xyz + left quaternion_wxyz + left gripper
right xyz + right quaternion_wxyz + right gripper
```

当前解码基准可直接从 RoboTwin 获取：

```python
task_env.robot.get_left_ee_pose()
task_env.robot.get_right_ee_pose()
```

不需要在部署 policy 内额外实例化 SAPIEN FK。

## 6. 已确认细节与仍需实验确认项

### 6.1 Temporal RoPE

第一版已确认保持 Wan 原生连续时间位置：

```text
RGB latent RoPE time = 0, 1, 2, 3, 4
RAY latent RoPE time = 5, 6, 7, 8, 9
```

优点：

- 对 Video Expert 的结构修改最少；
- 不需要新增 modality embedding；
- block 顺序可以直接使用当前 3D RoPE 生成逻辑；
- 便于先验证整个系统是否可训练。

该项已于 2026-07-24 确认。第一版直接使用连续 `0...9`，同时保留 reset RoPE
作为后续消融方案。

### 6.2 RGB/action 时间对齐

已确认：

```text
RGB frame 0 = state before action 0
RGB frame i+1 = observation after action i
RAY frame 0 = current state EE pose
RAY frame i+1 = target EE pose of action i
```

已用 20 个跨任务/随机 episode 检查：`action[t]` 与
`observation.state[t+1]` 完全一致，FK 得到的 `action.endpose[t]` 与
`observation.state.endpose[t+1]` 也完全一致。因此上述对齐是第一版的确定实现。

### 6.3 Gripper 控制范围

已统计全部 27,500 个 episode：

```text
action[:, 6]
action[:, 13]
```

左右 gripper 的真实范围均为 `[0,1]`；RoboTwin `set_gripper` 和 EE action
路径也明确使用 `[0,1]`。第一版确定使用 `pixel = 2 * gripper - 1`。

### 6.4 共享 timestep

第一版暂定 RGB/Rothko 共用一个 timestep。后续可尝试两种模态独立 timestep，但
这需要更复杂的 per-token timestep modulation。

### 6.5 上下 Rothko 融合方式

需要离线比较：

```text
只解码上半份
上下像素先融合再解码
上下分别解码再融合 pose
```

以 position、rotation、gripper 误差和 simulator replay 成功率决定最终实现。

## 7. 数据改造计划

### 7.1 当前数据状态

当前训练数据：

```text
/mnt/hwdata/cfy/FastWAM/data/robotwin2.0/robotwin2.0
```

parquet 原先只有：

```text
observation.state
action
```

两者都是 14D joint/qpos。2026-07-24 已全量增加：

```text
action.endpose
observation.state.endpose
```

新列是 14D 双臂 EE pose，27,500/27,500 个 episode 已通过 schema、元数据和
数值检查。

### 7.2 使用 add_gripper_pose.py

参考脚本：

```text
/mnt/hwdata/cfy/wam/RoboTwin/add_gripper_pose.py
```

它通过与 RoboTwin 相同的 SAPIEN FK 添加：

```text
action.endpose
observation.state.endpose
```

两列都是：

```text
[left xyz + quat_wxyz, right xyz + quat_wxyz]
```

需要同时更新：

```text
meta/info.json
meta/episodes_stats.jsonl
```

注意：

- 该脚本会原地重写 parquet；
- 正式执行前应先用少量 episode 验证；
- 应检查目标数据集是否需要备份或使用临时副本；
- 脚本名含 `gripper_pose`，但新增 endpose 列不含 gripper 开合；
- gripper 仍从原 `action` 和 `observation.state` 的第 6、13 维读取。

### 7.3 Loader 改造

不建议把 endpose 直接塞进当前 action normalizer/merger。

建议 loader 独立返回：

```text
action                  normalized joint action, [16, 14]
proprio                 normalized state qpos, [17, 14]
action_endpose           raw EE pose, [16, 14]
state_endpose            raw EE pose, [17, 14]
video                    RGB, [3, 17, 384, 320]
action_is_pad
image_is_pad
context
context_mask
```

Rothko target 使用：

```text
base = state_endpose[0]
future = action_endpose[0:16]
```

gripper target 使用：

```text
action[0:16, [6, 13]]
```

但必须取得 normalization 前的原始 gripper 值。可让 processor 额外保留
`raw_action`，或在 normalizer 前提取 gripper。

### 7.4 Raymap 生成位置

推荐不要将所有 Rothko 图预先保存成视频文件，避免增加大量磁盘占用。

建议：

- parquet 离线增加轻量 endpose 数值列；
- Dataset 返回 EE pose 与 raw gripper；
- 在 Dataset CPU worker 或 model input preparation 中按需生成 Rothko；
- 优先实现纯 Torch、向量化、缓存 canonical rays 的 encoder；
- 对比 CPU worker 生成和 GPU batch 生成的吞吐与显存；
- 若 CPU 成为瓶颈，再考虑小型按 episode cache，而不是直接预存完整视频。

## 8. 模型改造计划

建议新增模型，而不是直接破坏现有 FastWAM：

```text
src/fastwam/models/wan22/fastwam_visual_action.py
```

建议类名：

```text
FastWAMVisualAction
```

或更明确：

```text
FastWAMVideoOnlyRaymap
```

建议新增配置：

```text
configs/model/fastwam_video_only_raymap.yaml
configs/task/robotwin_video_only_rothko_3cam_384_1e-4.yaml
```

保留现有 FastWAM 的原因：

- 可以继续运行 released checkpoint；
- 可以做严格 baseline 对照；
- 避免一次改造破坏当前可工作的训练和评测；
- 方便逐步迁移 trainer 与 deployment。

模型配置建议包含：

```yaml
representation: rothko
init_source: original_wan22
action_horizon: 16
keep_proprio: true
latent_layout: rgb_then_ray
temporal_rope_mode: continuous
num_condition_latents_per_modality: 1
gripper_encoding: rothko_outer_margin
rothko_duplicate_vertical: true
lambda_rgb: 1.0
lambda_ray: 1.0
```

## 9. Checkpoint 设计

新 checkpoint 不再保存 MoT 或 Action Expert。

建议格式：

```python
{
    "dit": video_expert.state_dict(),
    "proprio_encoder": proprio_encoder.state_dict(),
    "step": step,
    "torch_dtype": str(dtype),
    "visual_action_config": {
        "representation": "rothko",
        "action_horizon": 16,
        "latent_layout": "rgb_then_ray",
        "temporal_rope_mode": "continuous",
        "gripper_encoding": "outer_margin",
        "rothko_duplicate_vertical": True,
        "norm_stats_path": "...",
        "focal": 0.2,
        "center_scale": 1.0,
        "dir_scale": 1.0,
        "center_frac": 0.5,
        "boundary_margin": 8,
    },
}
```

加载时必须检查 checkpoint metadata 与 Raymap codec、normalization stats 是否一致。

## 10. 实施阶段

### 阶段 A：Golden codec

1. 从 RoboTwin replay 脚本提取 Rothko/Raxel 核心逻辑；
2. 建立统一 codec 接口；
3. 实现上下复制；
4. 实现 gripper border 编解码；
5. 与参考脚本做逐数值测试；
6. 验证纯 encode/decode 近零误差；
7. 验证原始 Wan VAE roundtrip 误差；
8. 重新计算新窗口、新布局 normalization stats。

建议接口：

```python
encode(
    future_endpose,
    current_endpose,
    gripper,
) -> raymap_video

decode(
    raymap_video,
    current_endpose,
) -> ee_actions
```

### 阶段 B：数据

1. 复制/适配 `add_gripper_pose.py`；
2. 先处理 1 个 episode；
3. 校验 quaternion 顺序必须为 `wxyz`；
4. 校验 FK 与 RoboTwin `get_left/right_ee_pose()` 一致；
5. 全量补充 endpose 列；
6. 扩展 loader；
7. 检查 17 RGB、1 current state、16 action target 对齐；
8. 检查 episode 尾部 padding。

### 阶段 C：Video-only model

1. 加载 original Wan2.2 Video Expert；
2. 移除 ActionDiT/MoT 依赖；
3. 实现两个 VAE encode；
4. 实现 latent block concat/split；
5. 实现双条件 timestep；
6. 实现双条件 attention mask；
7. 实现分段 RGB/Ray latent loss；
8. 更新 checkpoint save/load。

### 阶段 D：推理

1. 初始化两个条件 latent；
2. 初始化两个未来噪声 block；
3. 运行完整 video denoising；
4. 拆分并分别 VAE decode；
5. Rothko + gripper decode；
6. 返回 RGB rollout 与 EE action；
7. 离线对齐 GT 并输出误差与可视化。

### 阶段 E：RoboTwin policy

1. policy 获取当前 EE pose；
2. policy 构造当前 Rothko；
3. 调用新 `infer_action`；
4. 切换到 `action_type="ee"`；
5. 保持 action queue/replan；
6. 保存预测 RGB、预测 Rothko、解码 action；
7. 单任务闭环测试。

### 阶段 F：训练验证

建议验证顺序：

1. codec unit test；
2. VAE roundtrip；
3. 单 sample 前向；
4. 单 batch 前向/反向；
5. 显存与速度 profiling；
6. 单 batch 过拟合；
7. 单 episode 过拟合；
8. `click_alarmclock` 小数据训练；
9. 离线 Raymap 与 EE 误差；
10. RoboTwin 单任务闭环；
11. 多任务扩展；
12. 备选方案消融。

## 11. 未选择的备选方案

本节保留所有讨论过的主要替代方案。

### 11.1 初始化：Released FastWAM Video Expert

方案：

```text
从 released checkpoint 中提取 mot.mixtures.video.*
去掉前缀后加载到独立 Wan Video Expert
同时加载 released proprio_encoder
```

优点：

- 已适配 RoboTwin RGB 与三相机布局；
- 已适配机器人运动；
- 已适配任务语言与 proprio；
- 小数据初期可能收敛更快。

缺点：

- 不是纯 RGB domain adaptation；
- action loss 曾间接更新 Video Expert；
- 只见过 9 帧 RGB；
- 没见过 Raymap block；
- 不利于严格判断原始 video diffusion prior 的贡献。

状态：

```text
未选择；保留为初始化消融。
```

### 11.2 Proprio：当前 EE pose token

用当前：

```text
left xyz + quaternion
right xyz + quaternion
```

替代 14D qpos proprio。

优点：

- 与 Rothko 表示语义一致；
- 不需要模型处理关节冗余；
- 更接近最终 EE control。

缺点：

- quaternion 与 position 需要专门 normalization；
- 丢失关节构型与冗余自由度；
- 相同 EE pose 可能对应不同关节状态；
- 需要训练新的 state encoder。

状态：

```text
未选择；第一版保留 qpos proprio。
```

### 11.3 Proprio：完全移除

条件只剩：

```text
language + RGB + Rothko
```

优点：

- 最严格的视觉生成范式；
- 不存在额外机器人状态 token。

缺点：

- 当前相对 Rothko frame 0 基本是 canonical map；
- 模型只能从 RGB 推断机器人状态；
- 可能增加动作歧义和训练难度。

状态：

```text
未选择；可做严格视觉-only 消融。
```

### 11.4 RGB/Rothko 空间拼接

每个时刻：

```text
[RGB | Rothko] -> 384 x 640
```

优点：

- 时间一一对应；
- 一次 VAE encode；
- RGB/Rothko 同一 frame 内空间邻接。

缺点：

- 空间 token 翻倍；
- VAE 空间卷积在模态边界混合；
- horizon 16 时约 1200 tokens，horizon 32 时约 2160 tokens；
- 改变预训练分辨率分布；
- 用户已明确暂不采用一张图空间合并。

状态：

```text
已排除第一版。
```

### 11.5 像素时间序列先拼接，再统一 VAE encode

方案：

```text
[RGB pixel sequence | Rothko pixel sequence] -> one VAE encode
```

优点：

- 实现直观；
- Video Expert 输入仍是时间 block。

缺点：

- Wan VAE 有 causal temporal convolution；
- 模态边界会被同一 VAE temporal receptive field 混合；
- 总像素帧数量还需 padding 到 `T % 4 == 1`；
- decode 时难以干净拆分两种模态。

状态：

```text
未选择；使用分别 VAE encode 后 latent 拼接。
```

### 11.6 Latent 交叉排列

方案：

```text
RGB_0, RAY_0, RGB_1, RAY_1, ...
```

优点：

- 相同时刻的跨模态 latent 在序列中相邻；
- 可能方便局部跨模态交互。

缺点：

- 时间 RoPE 把模态切换当成时间运动；
- 同一模态不再时间连续；
- 更偏离预训练视频的 temporal topology；
- split/mask 更复杂。

状态：

```text
未选择；第一版使用 RGB block + Ray block。
```

### 11.7 保持旧的 RGB/action 1:4 频率

方案：

```text
32 actions
9 RGB frames
33 或 32 Rothko frames
```

问题：

- RGB 与 Raymap 无法一一对应；
- 两段 latent 时间尺度不同；
- action 高频信息与视频低频信息的对应关系需要额外设计；
- 可能需要插值、下采样或 temporal alignment module。

状态：

```text
未选择；第一版 RGB/Rothko/action 同频。
```

### 11.8 Horizon 32

方案：

```text
33 RGB frames -> 9 RGB latent frames
33 Ray frames -> 9 Ray latent frames
total latent frames = 18
total DiT tokens = 2160
```

优点：

- 与当前 FastWAM 32-action horizon 一致；
- 一次预测覆盖更长控制时间；
- 减少 replanning 频率。

缺点：

- 相比当前 360 tokens 增长到 2160；
- attention 理论计算量约为当前 36 倍；
- 更容易 OOM；
- 更长相对运动需要重新统计更宽 normalization 范围；
- 长 horizon Raymap 更容易出现 clipping 与累计误差。

状态：

```text
暂缓；第一版 horizon 16。
```

### 11.9 原始下半部分随机 padding

方案：

```text
上半 Rothko
下半 continuous random padding
```

优点：

- 与现有 VAE replay 验证一致；
- 已有正式 stats 和重建结果；
- 避免大面积固定值。

缺点：

- 下半部分不承载 action 信息；
- 浪费一半图像；
- gripper 仍需其他编码方式。

状态：

```text
未选择；第一版上下复制 Rothko。
```

### 11.10 Gripper 独立图像序列

新增第三段 latent：

```text
[RGB block | Ray block | Gripper-image block]
```

优点：

- 不污染 Rothko；
- gripper 监督区域独立；
- 保持全视觉生成。

缺点：

- 增加序列长度和 attention 成本；
- 新增第三模态与第三条件；
- gripper 两个标量被扩展成整段视频，表示效率低。

状态：

```text
未选择；保留为边界编码失败时的回退方案。
```

### 11.11 Gripper 独立小 Head

由 Video Expert hidden states 额外预测 gripper。

优点：

- 计算量小；
- 监督直接；
- 实现简单。

缺点：

- 不再是完整 action 全部通过视觉表示生成；
- 引入额外 action head；
- 与本项目缩小模态差距的目标不完全一致。

状态：

```text
未选择；仅作为工程回退。
```

### 11.12 暂时复用 GT gripper

像原 VAE replay 一样，EE 来自 Raymap，gripper 来自 expert。

优点：

- 可以快速验证 EE trajectory；
- 与现有 replay 逻辑一致。

缺点：

- 真正 policy 推理时没有 GT gripper；
- 无法完成独立闭环；
- 不能视为完整 action prediction。

状态：

```text
不作为最终模型；仅允许用于 codec/replay 离线诊断。
```

### 11.13 直接 action-space supervision

训练时：

```text
predicted flow
    -> estimated clean latent
    -> VAE decode
    -> Rothko decode
    -> EE pose/gripper loss
```

可包含：

```text
position L1/L2
rotation geodesic loss
gripper L1/BCE
```

优点：

- 直接优化最终控制误差；
- 可以降低“像素误差小但 action 误差敏感”的问题。

缺点：

- 每个训练 step 需要 VAE decode；
- 显存和计算成本高；
- Rothko median/SVD 的梯度性质复杂；
- 混合精度下 SVD 需要额外处理；
- 可能破坏简单的预训练 diffusion 训练范式。

状态：

```text
第一版不做；基础 latent 模型跑通后再尝试。
```

### 11.14 Qpos 控制

继续输出并执行 qpos：

```text
action_type="qpos"
```

问题：

- Rothko/Raxel 解码得到 EE pose，不是 joint qpos；
- EE pose 到 qpos 需要 IK；
- IK 可能多解；
- 需要处理关节限位、连续性与失败；
- 会重新引入 action-space 特定模块。

状态：

```text
未选择；第一版使用 EE control。
```

### 11.15 Raxel

Raxel 把相机中心与方向相加到整个像素场中：

```text
raxel = direction * dir_scale + origin * center_scale
```

优点：

- 表示形式统一；
- 无显式中心/方向区域切分；
- 已有 VAE replay 和 global stats 代码。

缺点：

- 平移与旋转在每个像素中耦合；
- 解码需要联合 Umeyama/Kabsch；
- 对 VAE reconstruction bias 更敏感；
- Rothko 已有更完整的正式消融和验证结果。

状态：

```text
第一版优先 Rothko；codec 接口应保留 Raxel 可配置能力。
```

### 11.16 Temporal RoPE 重置

方案：

```text
RGB temporal positions = 0,1,2,3,4
RAY temporal positions = 0,1,2,3,4
```

优点：

- 对应 RGB/Ray 时刻共享时间坐标；
- 更符合两个同步模态的语义。

缺点：

- 同一序列中出现重复 temporal RoPE；
- Video Expert 无法只靠位置区分模态；
- 与预训练连续时间位置不同；
- 可能需要 segment/modality embedding。

状态：

```text
保留为 RoPE 消融。
```

### 11.17 Modality embedding

给 RGB/Ray token 添加：

```text
e_rgb
e_ray
```

优点：

- 明确告诉 Video Expert token 属于哪个模态；
- 对 RoPE 重置方案尤其有帮助；
- 参数量很小。

缺点：

- 原始 Wan 没有该 embedding；
- 新参数随机初始化；
- 增加预训练结构改动；
- latent 分布本身可能已经足够区分模态。

状态：

```text
第一版暂不添加；后续与 RoPE reset 联合消融。
```

### 11.18 RGB/Ray 独立 diffusion timestep

方案：

```text
t_rgb != t_ray
```

优点：

- 可以让模型处理不同噪声程度的两个模态；
- 可能改善跨模态条件生成；
- 可以进行类似 multimodal diffusion 的异步去噪。

缺点：

- 需要 per-token timestep；
- scheduler 和 loss 更复杂；
- 偏离单视频 diffusion 预训练；
- 推理 schedule 配合更困难。

状态：

```text
第一版使用共享 timestep；独立 timestep 作为高级消融。
```

### 11.19 使用 finetuned VAE decoder

RoboTwin 中存在 Raxel/Rothko VAE decoder finetune 代码。

优点：

- 可能降低 Raymap reconstruction error；
- 提高最终 EE 位姿精度。

缺点：

- RGB 与 Raymap 共用 VAE 时，finetune decoder 可能损害 RGB；
- 两种模态可能需要不同 decoder；
- 不再严格使用原始预训练 VAE；
- 增加 checkpoint 与部署复杂度。

状态：

```text
第一版使用冻结的原始 Wan VAE；后续再考虑双 decoder 或 adapter。
```

## 12. 建议的第一批消融

在主链路跑通后，建议按以下优先级尝试：

1. `original_wan22` vs `released_fastwam_video` 初始化；
2. continuous RoPE vs per-modality reset RoPE；
3. no modality embedding vs modality embedding；
4. horizon 8 vs 16 vs 32；
5. 上下复制融合方式；
6. gripper 8-pixel full border vs 4-pixel code + 4-pixel guard；
7. shared timestep vs independent timestep；
8. qpos proprio vs EE proprio vs no proprio；
9. Rothko vs Raxel；
10. latent-only loss vs additional decoded action loss；
11. 原始 VAE vs Raymap decoder adaptation。

## 13. 第一版完成标准

第一版不能只以训练 loss 正常下降作为完成。

最低完成标准：

- 当前数据成功补充 EE pose；
- codec 与参考实现数值一致；
- 新 Rothko 布局可经原始 Wan VAE roundtrip；
- gripper 可从 VAE 重建图稳定解码；
- Video-only 模型单 batch 前向/反向通过；
- 两个条件 latent 在训练和推理中保持不变；
- RGB/Ray latent loss 分别正确计算；
- 单 batch 或单 episode 可过拟合；
- 推理确实生成未来 RGB，而不是只生成 action；
- Rothko 能解码出 16 个 EE actions；
- RoboTwin 使用 `action_type="ee"` 完成单任务闭环；
- 保存足够的 RGB、Raymap、EE pose 和 gripper 中间结果以便定位问题；
- 原有 FastWAM baseline 仍可运行，没有被新实现破坏。

## 14. 当前决策摘要

```text
Representation:
    Rothko first, keep Raxel interface as alternative

Initialization:
    original_wan22

Prediction expert:
    Wan Video Expert only

Action Expert:
    removed

MoT:
    removed

Proprio:
    keep current normalized 14D qpos token

Horizon:
    16 future actions

Pixel frames:
    17 RGB + 17 Rothko

VAE:
    same frozen original Wan VAE
    RGB/Rothko encoded and decoded separately

Latent layout:
    [RGB block | Rothko block]

Latent frames:
    5 RGB + 5 Rothko = 10

Raymap layout:
    upper Rothko + duplicated lower Rothko

Gripper:
    encode in decoder-ignored outer 8-pixel margin

Condition latent frames:
    RGB_0 + RAY_0

Diffusion timestep:
    shared RGB/Ray timestep for first version

Loss:
    RGB latent flow loss + Raymap latent flow loss

Direct action loss:
    disabled for first version

Inference:
    jointly generate future RGB and future Raymap

Control:
    RoboTwin EE control

Temporal RoPE:
    confirmed continuous 0...9
    reset/modality embedding retained as alternatives
```

## 15. 当前 8 任务实验与完整运行命令

本节记录 2026-07-24 实际使用的数据范围、缓存位置、运行命令和 100-step 验证
结果。命令默认从仓库根目录执行：

```text
/mnt/hwdata/cfy/FastWAM
```

当前机器只允许使用物理 GPU 4、5、6、7，不应使用 GPU 0--3。实际语言缓存使用了
4、5、6、7；100-step 训练启动时 4、5 被其他用户占用，因此训练改用 6、7。

不要为这些命令额外创建或设置临时的 `HF_HOME`、`HF_DATASETS_CACHE` 或
`XDG_CACHE_HOME`。当前环境已经有统一缓存配置：

```text
XDG_CACHE_HOME=/mnt/hwdata/cfy/XDG_CACHE
HF_HOME=/mnt/hwdata/cfy/XDG_CACHE/huggingface
```

模型下载缓存沿用上述环境默认值；文本 embedding 的最终输出位置由数据配置决定，
与 Hugging Face 下载缓存不是一回事。

### 15.1 当前选择的 8 个任务

训练和 validation 使用完全相同的任务集合，配置文件为
`configs/task/robotwin_video_only_rothko_3cam_384_1e-4.yaml`。

| RoboTwin 任务名 | 配置中的标识 |
|---|---|
| Stack Bowls Three | `stack_bowls_three` |
| Stack Blocks Three | `stack_blocks_three` |
| Place Shoe | `place_shoe` |
| Lift Pot | `lift_pot` |
| Blocks Ranking Size | `blocks_ranking_size` |
| Click Alarmclock | `click_alarmclock` |
| Blocks Ranking RGB | `blocks_ranking_rgb` |
| Turn Switch | `turn_switch` |

每个任务在 released RoboTwin 数据中对应 550 个连续 episode，合计 4,400 个唯一
episode。滑窗后本次训练实际构建出：

```text
train dataset = 1,268,383 samples
val dataset = 13,744 samples
```

关键数据路径：

```text
RoboTwin dataset:
    /mnt/hwdata/cfy/FastWAM/data/robotwin2.0/robotwin2.0

qpos/action normalization stats:
    /mnt/hwdata/cfy/FastWAM/data/robotwin2.0/dataset_stats.json

Rothko normalization stats:
    /mnt/hwdata/cfy/FastWAM/data/robotwin2.0/rothko_region_symmetric_q99p95_h16_384x320.pt

text embedding cache:
    /mnt/hwdata/cfy/FastWAM/data/text_embeds_cache/robotwin
```

Rothko stats 是根据完整的 27,500 个 episode、50 个任务计算的，不只来自上述
8 个任务。其主要参数为：

```text
horizon = 16
pixel frames = 17
image = 384 x 320
focal = 0.2
center_scale = 1.0
dir_scale = 1.0
center_frac = 0.5
boundary_margin = 8
outer_margin = 8
duplicate_vertical = true
translation quantile = 0.9995
translation absolute bounds XYZ = [0.27947, 0.25924, 0.25548] m
```

### 15.2 为数据补充 EE endpose

该步骤使用 `RoboTwin` conda 环境，因为脚本依赖 SAPIEN 和 RoboTwin 的
aloha-agilex URDF。脚本会原地修改 parquet 和 metadata；当前 27,500 个 episode
已经全部完成，不需要为现有数据重复执行。

先验证 episode 0--9，同时不更新全量 metadata：

```bash
conda activate RoboTwin
python add_gripper_pose.py \
  --limit 10 \
  --num-workers 1 \
  --no-meta-update
```

确认结果后，全量或断点式重跑：

```bash
conda activate RoboTwin
python add_gripper_pose.py \
  --num-workers 8 \
  --skip-existing
```

输出列为：

```text
action.endpose
observation.state.endpose
```

每列均为 14 维：

```text
[left xyz + quaternion(wxyz), right xyz + quaternion(wxyz)]
```

最初直接加载含 visual/collision 的 URDF 时，8 个 SAPIEN worker 会被底层进程
直接终止并表现为 `BrokenProcessPool`。当前脚本会为 FK 构造移除了
visual/collision 的临时 URDF，并使用 CPU PhysX-only scene，因此不需要渲染 GPU。

### 15.3 计算 Rothko normalization stats

当前 stats 可用下面的显式命令复现：

```bash
conda activate fastwam
python scripts/compute_rothko_norm_stats.py \
  --dataset-root /mnt/hwdata/cfy/FastWAM/data/robotwin2.0/robotwin2.0 \
  --output /mnt/hwdata/cfy/FastWAM/data/robotwin2.0/rothko_region_symmetric_q99p95_h16_384x320.pt \
  --horizon 16 \
  --windows-per-episode 32 \
  --quantile 0.9995 \
  --bins 100000 \
  --histogram-max 1.0 \
  --workers 16 \
  --focal 0.2 \
  --center-scale 1.0 \
  --dir-scale 1.0 \
  --center-frac 0.5 \
  --boundary-margin 8 \
  --outer-margin 8
```

这次统计使用了：

```text
27,500 episodes
880,000 windows
28,160,000 relative XYZ vectors
```

### 15.4 为 8 个任务计算语言 embedding

使用物理 GPU 4、5、6、7：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --standalone --nproc_per_node=4 \
  scripts/precompute_text_embeds.py \
  task=robotwin_video_only_rothko_3cam_384_1e-4 \
  +overwrite=false
```

任务范围直接从 task config 中读取，因此无需再在命令行重复 8 个任务名。结果：

```text
8 tasks
4,400 episodes
17,748 unique prompts required
15,626 newly encoded
2,122 existing cache entries reused
0 overlength prompts
```

最终缓存目录约 21 GB，共有 21,098 个 `.pt` 文件；目录还包含此前其他 RoboTwin
任务留下的 embedding，因此文件总数大于这次 8 任务所需的 17,748。

### 15.5 已执行的双卡 100-step smoke training

实际运行命令如下：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 2 \
  task=robotwin_video_only_rothko_3cam_384_1e-4 \
  max_steps=100 \
  eval_every=0 \
  save_every=100 \
  log_every=1
```

运行配置：

```text
ZeRO stage = 1
mixed precision = BF16
physical GPUs = 6, 7
per-GPU batch size = 1
global batch size = 2
gradient accumulation = 1
learning rate = 1.0e-4
LR scheduler = cosine
weight decay = 1.0e-2
```

`accelerate` 启动时发现默认端口 29500 已占用并自动选择了其他可用端口；本次双卡
训练仍正常完成。后续若必须固定进程端口，应在 launcher 中显式向
`accelerate launch` 传递 `--main_process_port`，不能只根据这条提示推断最终端口。

运行目录：

```text
/mnt/hwdata/cfy/FastWAM/runs/robotwin_video_only_rothko_3cam_384_1e-4/2026-07-24_11-44-25
```

第 100 step：

```text
total loss = 0.1612
loss_rgb = 0.1193
loss_raymap = 0.0418
learning rate = 1.0e-6
speed = approximately 0.92 step/s, 1.85 samples/s
```

`lambda_rgb=1.0`、`lambda_raymap=1.0`，所以 total loss 是两个分项之和，只有显示
小数位带来的舍入差异。训练中 loss 会因 global batch 只有 2、任务/样本和
diffusion timestep 随机采样而逐步波动；没有出现 OOM、NaN、Inf 或 loss 爆炸。
100 step 只用于验证数据、模型、反向传播和保存链路，不能用于判断模型已经收敛，
也不能代表最终 RoboTwin 成功率。

本次 `eval_every=0`，所以没有 validation loss、PSNR/SSIM 或闭环成功率。本次
`wandb.enabled=false` 且 stdout 没有保存到日志文件，因此无法在运行结束后重建
完整 loss 曲线。后续正式实验应启用 W&B，或者保存标准输出，例如：

```bash
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 2 \
  task=robotwin_video_only_rothko_3cam_384_1e-4 \
  2>&1 | tee /absolute/path/to/train.log
```

注意：上面的日志示例没有覆盖 `max_steps`、`eval_every`、`save_every`，会使用 task
config 中的正式训练默认值。开始正式长程训练前应先确认可用 GPU、评测开销和保存
间隔。

### 15.6 Checkpoint 与完整性检查

推理权重：

```text
/mnt/hwdata/cfy/FastWAM/runs/robotwin_video_only_rothko_3cam_384_1e-4/2026-07-24_11-44-25/checkpoints/weights/step_000100.pt
```

恢复训练状态：

```text
/mnt/hwdata/cfy/FastWAM/runs/robotwin_video_only_rothko_3cam_384_1e-4/2026-07-24_11-44-25/checkpoints/state/step_000100
```

检查结果：

```text
inference checkpoint size = 9,999,832,761 bytes
root keys = [dit, step, torch_dtype, visual_action_config, proprio_encoder]
step = 100
tensor count = 827
total parameters = 4,999,849,152
all checkpoint tensors = BF16
CPU torch.load = passed

optimizer rank-0 shard = 29,999,132,081 bytes
optimizer rank-1 shard = 29,999,132,337 bytes
ZeRO model state = 11,409,529,417 bytes
trainer_state global_step = 100
entire run directory = approximately 76 GB
```

保存的 `visual_action_config` 已确认包含：

```text
representation = rothko
action_horizon = 16
latent_layout = rgb_then_raymap
temporal_rope_mode = continuous_0_9
condition_latent_indices = [0, 5]
```

`max_steps=100` 与 `save_every=100` 都会触发结束保存，因此 trainer 在 step 100
对同一路径保存了两次，但只是覆盖同一个 `step_000100`，没有生成两份目录。退出时
出现 `.nfs*` multiprocessing 临时目录无法立即删除和未显式
`destroy_process_group()` 的 warning；主训练进程返回码为 0，两个保存流程均完成，
推理权重也已完整加载验证，因此这些 teardown warning 没有损坏 checkpoint。

### 15.7 后续正式训练

使用两张空闲卡继续正式训练时，可运行：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 2 \
  task=robotwin_video_only_rothko_3cam_384_1e-4
```

如果物理 GPU 4、5、6、7 均空闲，可改为四卡：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=robotwin_video_only_rothko_3cam_384_1e-4
```

正式训练 checkpoint 必须包含 `visual_action_config`，部署端会检查 horizon、latent
布局、条件索引和 `temporal_rope_mode=continuous_0_9`，避免误加载旧 FastWAM
checkpoint。

训练后可先对单个任务做 RoboTwin 闭环评测；以下以物理 GPU 4 和
`click_alarmclock` 为例：

```bash
conda activate fastwam_robotwin
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_video_only_rothko_3cam_384_1e-4 \
  ckpt=/absolute/path/to/step_xxxxxx.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/mnt/hwdata/cfy/FastWAM/data/robotwin2.0/dataset_stats.json \
  gpu_id=4
```

其余 7 个任务替换 `EVALUATION.task_name` 后分别运行。评测时新 checkpoint 会走
Rothko decode 和 RoboTwin `action_type="ee"`，旧 FastWAM checkpoint 仍走 qpos。

当前仍需通过训练后实验完成的项目：

- 单样本/单 episode overfit；
- 多步采样的 RGB、Rothko 和 EE 误差曲线；
- 8 个任务分别进行 RoboTwin 闭环成功率评测；
- 上下 Rothko 融合方式消融；
- 后续 Raxel 路径。

## 16. LoRA 微调方案

### 16.1 动机

当前方法已经把机器人动作转换为 Rothko 三通道图像，并与 RGB 一样经过冻结的 Wan
VAE，最终在同一个 Video DiT latent 空间中建模。相比原始 FastWAM 的独立
ActionDiT，新的动作表示与预训练视频模态更接近，因此不一定需要更新约 50 亿个
Video DiT 参数。

LoRA 第一版用于回答：

```text
原始 Wan2.2 Video DiT 的视觉生成能力是否已经足够，
只需用少量低秩参数学习 RGB/Rothko 时序布局和机器人任务条件？
```

预期收益：

- 大幅减少 trainable parameters、gradient 和 optimizer state；
- 减少过拟合 8 个 RoboTwin 任务的风险；
- adapter checkpoint 从约 10 GB 降到约 50 MB；
- 可以保留同一个原始 Wan2.2 base，为不同任务集合分别保存 adapter；
- 便于比较 full fine-tuning 与 parameter-efficient fine-tuning。

需要明确的限制：

- Wan2.2 的完整冻结参数仍要驻留 GPU，LoRA 不等于量化；
- VAE encode、10-frame DiT attention 和 activation 显存基本不变；
- LoRA 主要节省 gradient、optimizer state 和推理 checkpoint；
- 如果主要 OOM 来自长视频 attention activation，LoRA 不能单独解决；
- adapter 容量可能不足以同时学习 Rothko 新模态、双条件 attention 和机器人控制。

### 16.2 第一版已选择方案

第一版不引入 Hugging Face PEFT 依赖，而是在仓库中实现独立的 `LoRALinear`：

```text
y = frozen_linear(x) + scaling * B(A(dropout(x)))
scaling = alpha / rank
```

初始化：

```text
A = Kaiming uniform
B = zero
```

因此刚注入尚未训练的 LoRA 输出与原始 Wan2.2 完全一致，不会在 step 0 改变基座
行为。

默认参数：

```text
rank = 16
alpha = 16
scaling = 1
dropout = 0
```

注入范围为 30 个 `DiTBlock` 中 self-attention 和 cross-attention 的全部投影：

```text
self_attn.q
self_attn.k
self_attn.v
self_attn.o
cross_attn.q
cross_attn.k
cross_attn.v
cross_attn.o
```

每层 8 个 Linear，共：

```text
30 layers x 8 modules = 240 LoRA modules
```

Wan2.2 当前 attention Linear 都是 `3072 -> 3072`。rank 16 时：

```text
per module:
    16 x 3072 + 3072 x 16 = 98,304 parameters

all attention adapters:
    98,304 x 240 = 23,592,960 parameters
```

相对于 4,999,849,152 个 DiT 参数，attention LoRA 约占：

```text
0.472%
```

此外继续全量训练：

```text
proprio_encoder: Linear(14, 4096)
parameters: 14 x 4096 + 4096 = 61,440
```

保留 proprio encoder 训练的原因是它是当前模型新加入且随机初始化的 qpos token
映射。如果将其冻结，模型会一直收到未经学习的随机 proprio token。VAE、text
encoder、原始 Video DiT 参数全部冻结。

### 16.3 为什么第一版只选择 attention

当前新范式相对原始视频预训练的主要变化是：

- latent 时间轴变为 `[RGB block | Rothko block]`；
- condition latent 从一个变为 `[0,5]` 两个；
- attention mask 需要同时处理两个条件 block；
- text/proprio 条件需要影响 Rothko 生成；
- RGB 与 Rothko future token 需要相互建模。

这些变化首先直接作用在 self-attention 和 cross-attention，因此第一版只对
`q/k/v/o` 做 LoRA，保持 patch embedding、FFN、time embedding 和 output head
冻结。这样参数量较小，也最容易判断“只调整 token/condition 交互”是否足够。

### 16.4 Trainer 冻结与 optimizer 规则

全局训练配置新增：

```yaml
finetune:
  method: full # full or lora
  train_proprio_encoder: true
  lora:
    rank: 16
    alpha: 16.0
    dropout: 0.0
    target_modules:
      - self_attn.q
      - self_attn.k
      - self_attn.v
      - self_attn.o
      - cross_attn.q
      - cross_attn.k
      - cross_attn.v
      - cross_attn.o
```

`method=full` 是默认值，保持原来的全量 DiT 微调行为。`method=lora` 时：

1. 从原始 Wan2.2 加载完整 Video DiT；
2. 在 optimizer/DeepSpeed 初始化前注入 LoRA；
3. 冻结整个模型；
4. 只重新打开 `lora_A`、`lora_B`；
5. 根据 `train_proprio_encoder` 打开 proprio encoder；
6. optimizer 只接收 `requires_grad=True` 的参数；
7. eval 后恢复 train mode 时再次检查冻结状态，避免意外解冻 base。

Trainer 会输出：

```text
fine-tuning method
trainable parameter count
total parameter count
trainable ratio
whether proprio encoder is trainable
```

### 16.5 LoRA checkpoint 设计

全量微调保持原来的 checkpoint：

```text
checkpoint_type = full
dit = complete Video DiT state_dict
proprio_encoder = complete proprio state_dict
visual_action_config
```

LoRA 推理权重改为 adapter-only：

```text
checkpoint_type = lora_adapter
lora = only all lora_A/lora_B tensors
proprio_encoder = complete proprio state_dict
fine_tuning:
    method = lora
    base_model_id = Wan-AI/Wan2.2-TI2V-5B
    rank / alpha / dropout / target_modules
    module_count / parameter_count
    train_proprio_encoder
visual_action_config:
    Rothko codec parameters
    action_horizon
    latent layout
    temporal RoPE mode
    condition latent indices
```

加载 LoRA checkpoint 时：

1. 正常从 `base_model_id` 对应的原始 Wan2.2 权重构建 Video DiT；
2. 校验当前 base model ID 与 checkpoint；
3. 根据 checkpoint 中的 LoRA config 自动注入 adapter；
4. 严格检查全部 A/B key 和 tensor shape；
5. 加载 proprio encoder；
6. 校验 Rothko/horizon/latent/RoPE 元信息；
7. 不要求评测配置预先显式注入 LoRA。

推理 adapter 的预计大小：

```text
23,592,960 BF16 LoRA parameters x 2 bytes
approximately 47.2 MB
plus proprio encoder and torch serialization metadata
```

真实 DeepSpeed 保存结果为：

```text
47,460,187 bytes
approximately 46 MiB as reported by ls -lh
```

注意：`checkpoints/weights/step_xxxxxx.pt` 会显著缩小，但
`checkpoints/state/step_xxxxxx` 是 Accelerate/DeepSpeed 的完整训练恢复状态，仍
可能包含冻结 base 的模型 state，因此不会缩小到只有几十 MB。训练恢复目录与可移植
推理 adapter 是两个不同用途的产物。

### 16.6 Base checkpoint 边界

第一版 adapter-only LoRA 明确绑定：

```text
Wan-AI/Wan2.2-TI2V-5B original base
```

不能直接把第 15 节的 100-step full-finetuned checkpoint 当作 LoRA base，然后只
保存 adapter。否则 adapter 会隐式依赖那份约 10 GB 的 full checkpoint，单独复制
adapter 到其他机器时无法复现。

因此第一版行为是：

- LoRA 从配置中的原始 Wan2.2 base 开始；
- `.pt` 权重恢复只接受 LoRA adapter checkpoint；
- 完整训练恢复只接受相同 LoRA 架构保存的 DeepSpeed state directory；
- 尝试把 full-finetuned `.pt` 加载进 LoRA run 时直接报错。

后续若确实希望在 full checkpoint 上继续 LoRA，有三种备选实现：

1. adapter checkpoint 显式记录并依赖 full base checkpoint 的内容哈希和路径；
2. 发布时把 full base 与 LoRA merge，导出新的约 10 GB 完整 DiT；
3. 计算 full checkpoint 相对原始 Wan 的 delta，与 LoRA 一起保存。

第一版选择原始 Wan base，以保证 adapter 独立、可移植和实验定义清晰。

### 16.7 8 任务 LoRA 配置

新增任务配置：

```text
configs/task/robotwin_video_only_rothko_3cam_384_lora_1e-4.yaml
```

它使用与第 15 节 full fine-tuning 完全相同的 8 个任务、数据、Rothko stats、语言
embedding 和 scheduler。正式 LoRA 配置利用较小的 optimizer/gradient 显存，将
per-GPU batch 从 full smoke test 的 1 提高到 32，并改变：

```text
finetune.method = lora
LoRA rank/alpha/dropout/targets
per-GPU batch size = 32
num_workers = 2
max_steps = 30,000
eval_every = 500
save_every = 2,000
wandb.enabled = true
wandb.project = fast-wam
wandb.group = robotwin_video_only_rothko_8task_lora_r32
```

使用两张 GPU 时：

```text
micro global batch = 32 x 2 = 64
effective global batch = 64
```

使用四张 GPU 时：

```text
micro global batch = 32 x 4 = 128
effective global batch = 128
```

当前 `gradient_accumulation_steps=1`。实际双卡 rank-32、per-GPU batch 4 运行时，
每张 80 GB GPU 仅占用约 15.3 GB，因此正式配置进一步提高到 per-GPU batch 32。
第一次使用 `batch_size=32, num_workers=8` 时在首批数据的 pin-memory 阶段触发
`CUDA error: invalid argument`，尚未进入模型 forward；因此保留 batch 32，将
`num_workers` 降到 2 以减少双进程预取和共享/pinned memory 并发。LoRA 会显著减少
gradient 和 optimizer state，但不会消除 10-frame Video DiT activation；首次成功
进入 forward 后仍需观察峰值显存，若 OOM 再降低 batch。

双卡短程 smoke training：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 2 \
  task=robotwin_video_only_rothko_3cam_384_lora_1e-4 \
  max_steps=10 \
  batch_size=1 \
  eval_every=0 \
  save_every=10 \
  log_every=1
```

上面的 smoke 命令显式覆盖 `batch_size=1`，用于排查功能；正式配置默认
`batch_size=32`、`max_steps=30000`。双卡训练时共访问约 192 万个样本，相当于
当前 1,268,383 个训练窗口约 1.51 epoch。

正式双卡训练：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 2 \
  task=robotwin_video_only_rothko_3cam_384_lora_1e-4
```

四卡训练：

```bash
conda activate fastwam
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=robotwin_video_only_rothko_3cam_384_lora_1e-4
```

评测 adapter checkpoint：

```bash
conda activate fastwam_robotwin
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_video_only_rothko_3cam_384_lora_1e-4 \
  ckpt=/absolute/path/to/weights/step_xxxxxx.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=/mnt/hwdata/cfy/FastWAM/data/robotwin2.0/dataset_stats.json \
  gpu_id=4
```

single 会把 task config 中的 `finetune.method` 显式传给 RoboTwin policy。LoRA
checkpoint 会先加载原始 `Wan-AI/Wan2.2-TI2V-5B`，再加载 adapter；full
checkpoint 仍可跳过原始 DiT 加载。Rothko normalization stats 会相对 FastWAM
项目根目录解析为绝对路径，不再受 RoboTwin 子进程工作目录影响。

manager 现在支持任务列表与非连续物理 GPU ID。四任务、四卡完整评测（每个任务依次
执行 `demo_clean` 和 `demo_randomized`）：

```bash
conda activate fastwam_robotwin
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_video_only_rothko_3cam_384_lora_1e-4 \
  ckpt=/absolute/path/to/weights/step_xxxxxx.pt \
  'EVALUATION.task_names=[stack_bowls_three,place_shoe,click_alarmclock,blocks_ranking_rgb]' \
  EVALUATION.dataset_stats_path=/mnt/hwdata/cfy/FastWAM/data/robotwin2.0/dataset_stats.json \
  'MULTIRUN.gpu_ids=[2,3,6,7]' \
  MULTIRUN.max_tasks_per_gpu=1
```

设置了 `MULTIRUN.gpu_ids` 时会直接使用这些物理 GPU；未设置时，如果外部存在
`CUDA_VISIBLE_DEVICES`，manager 会从其中选择前 `MULTIRUN.num_gpus` 个设备，
否则保持原有的 `0..num_gpus-1` 行为。

### 16.8 第一轮对比指标

LoRA 不能只比较训练 loss。与 full fine-tuning 应至少比较：

```text
trainable parameter count
peak allocated/reserved GPU memory
optimizer checkpoint size
portable weights checkpoint size
steps/s and samples/s
RGB latent loss
Rothko latent loss
validation loss
RGB PSNR/SSIM
Rothko PSNR/SSIM
decoded EE position/rotation/gripper error
8-task RoboTwin closed-loop success rate
```

公平对比要求：

- 相同 8 个任务和 episode；
- 相同 seed；
- 相同 global batch；
- 相同训练 step；
- 相同 LR scheduler；
- 相同 diffusion timestep sampling；
- 相同 checkpoint/evaluation step；
- 分别报告 RGB 和 Rothko，不只报告 total loss。

第一轮建议先做：

1. 1--10 step smoke test，验证梯度、保存和加载；
2. 单 sample 或单 episode overfit，判断 rank 16 是否有足够容量；
3. 与 full fine-tuning 相同的 100-step run；
4. 比较两者 loss、显存、速度和 checkpoint；
5. 只有短程行为正常后再跑正式 8 任务训练。

### 16.9 保留的 LoRA 备选方案

如果 attention-only rank 16 容量不足，按以下顺序增加能力：

1. `rank=32`，保持 attention-only；
2. 在 attention LoRA 之外训练每层 `modulation`；
3. 将 `ffn.0` 和 `ffn.2` 加入 LoRA；
4. 解冻 output head；
5. 解冻 patch embedding；
6. attention + FFN 全部 Linear LoRA；
7. 使用不同 LR：LoRA 较高、proprio encoder 较低；
8. 为 RGB/Rothko 增加 modality embedding；
9. 使用 DoRA 或其他 adapter；
10. 改用 PEFT，以换取标准生态的 merge/export 功能。

不建议第一轮直接对所有 Linear 做 LoRA，因为这样无法判断收益究竟来自 attention
布局适配还是 FFN/输入输出映射，同时会增加约 1,671 万个 FFN LoRA 参数。

如果 attention-only 的 RGB loss 正常而 Rothko loss 明显落后，优先尝试：

```text
output head / FFN / patch embedding adaptation
```

因为这更可能说明问题出在新视觉分布的输入输出映射，而不是 token attention。

如果两个 loss 都难以下降，优先尝试：

```text
rank 32
attention + FFN LoRA
full fine-tuning baseline
```

如果训练 loss 很低但闭环控制失败，应优先检查 codec、VAE reconstruction、时序误差
累积和 EE decode，而不是继续增加 LoRA rank。

### 16.10 当前实现与验证状态

已完成：

- 内置 `LoRALinear`，无新增第三方依赖；
- suffix-based target module 注入与严格 target 检查；
- base 参数冻结与 adapter-only trainable 标记；
- trainer 的 `full/lora` 双模式；
- proprio encoder 独立训练开关；
- adapter-only save/load；
- base model ID、LoRA config 和 Rothko config 校验；
- 旧 full checkpoint 保存/加载保持兼容；
- 8 任务 LoRA Hydra 配置；
- 小模型初始输出一致性测试；
- adapter state 保存/加载一致性测试；
- 30 层结构 target 匹配测试；
- 真实 Wan2.2 5B 模型构建；
- 双卡真实 batch forward/backward；
- 双卡 1-step 和 2-step LoRA smoke training；
- adapter checkpoint CPU load；
- 干净 5B base 自动注入并加载 adapter。

30 层结构测试结果：

```text
matched LoRA modules = 240
expected Wan 5B adapter parameters = 23,592,960
first module = blocks.0.self_attn.q
last module = blocks.29.cross_attn.o
status = PASS
```

真实双卡 LoRA 验证使用物理 GPU 6、7，GPU 4、5 当时被其他用户占用。2-step
运行目录：

```text
runs/robotwin_video_only_rothko_3cam_384_lora_1e-4/2026-07-24_12-41-47
```

关键初始化结果：

```text
LoRA modules = 240
LoRA parameters = 23,592,960
LoRA + proprio trainable parameters = 23,654,400
all model parameters including frozen VAE etc. = 5,728,130,780
trainable ratio over complete model = 0.412951%

ZeRO-1 after optimizer initialization:
memory allocated = approximately 10.73 GB
memory cached = approximately 10.83 GB
```

2-step loss：

```text
step 1:
    total = 0.7288
    RGB = 0.5333
    Rothko = 0.1955

step 2:
    total = 0.6178
    RGB = 0.4349
    Rothko = 0.1829
```

这两步只能证明 forward/backward 和 optimizer 正常，不能用来判断收敛趋势。随机
sample 和 diffusion timestep 会让极短程 loss 与第 15 节 full run 的 step 100
不可直接比较。

第一次真实 DeepSpeed adapter 保存暴露了一个 storage 问题：DeepSpeed 参数是 flat
buffer 的 view，直接 `.cpu()` 会把较大的 backing storage 一起保存，使文件达到：

```text
94,645,977 bytes
```

修复方式是在提取每个 A/B 和 proprio tensor 时强制 `.cpu().clone()`，得到紧凑且
独立的 CPU storage。修复后重新执行 1-step 验证：

```text
run:
    runs/robotwin_video_only_rothko_3cam_384_lora_1e-4/2026-07-24_12-47-52

portable adapter:
    checkpoints/weights/step_000001.pt
    47,460,187 bytes
    480 A/B tensors
    23,592,960 BF16 adapter parameters
    CPU torch.load = PASS
```

LoRA DeepSpeed 恢复状态：

```text
rank-0 optimizer shard = 141,949,701 bytes
rank-1 optimizer shard = 141,949,893 bytes
model state = 11,457,064,189 bytes
entire 1-step run = approximately 11 GB
```

相比第 15 节 full fine-tuning：

```text
full optimizer shard per rank = approximately 30 GB
LoRA optimizer shard per rank = approximately 142 MB

full portable weights = approximately 10 GB
LoRA portable adapter = approximately 47.5 MB
```

DeepSpeed model state 仍包含冻结 base，所以 LoRA 的恢复目录仍约 11 GB；这是正常
现象。

最后在新进程中只使用 GPU 6：

1. 从原始 Wan2.2 权重重新构建干净的 5B base；
2. 读取 47,460,187-byte adapter；
3. 根据 checkpoint 元信息自动注入 240 个 LoRA module；
4. 严格加载 480 个 A/B tensor；
5. 逐值比较首个 `blocks.0.self_attn.q` 的 A/B；
6. 检查 module/parameter count。

结果：

```text
checkpoint_type = lora_adapter
step = 1
modules = 240
adapter_parameters = 23,592,960
device = cuda:0 (physical GPU 6 under CUDA_VISIBLE_DEVICES)
status = LOAD_PASS
```

并检查了 1-step checkpoint 中从全零初始化的 `lora_B`：

```text
B tensors with nonzero updates = 240 / 240
nonzero B elements = 11,789,560 / 11,796,480
global B L2 norm = 0.339443
status = UPDATE_PASS
```

这说明所有 240 个目标模块都实际收到了梯度并完成了 optimizer update，不只是完成了
冻结 base 的前向。

当前没有 OOM、NaN 或 Inf。`num_workers=2` 的 2-step run 退出时仍出现 NFS
multiprocessing 临时目录 warning；`num_workers=0` 的 1-step run 没有该 NFS
warning，只剩未显式 `destroy_process_group()` 的退出 warning，两个主命令返回码
均为 0。

仍需完成：

- 双卡 10-step LoRA smoke training；
- full 与 LoRA 的同 seed、同 sample、同 100-step 对比；
- adapter checkpoint 通过完整 RoboTwin policy 初始化；
- 单 episode overfit；
- 8 任务正式训练；
- RoboTwin 闭环评测。

## 17. 2026-07-26 至 2026-07-28 的当前改造

本节记录第 16 节第一版 LoRA 之后的实际代码演进。第 16 节保留为历史设计和对比
依据；当前单任务实验应以本节及下面的配置文件为准：

```text
configs/task/robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4.yaml
configs/model/fastwam_video_only_raymap.yaml
configs/sim_robotwin.yaml
```

### 17.1 为什么先改成单任务高容量 LoRA

早期 4/8 任务 rank-32 attention-only LoRA 在约 1k step 后 RGB loss 基本进入平台，
继续训练到约 7k step 仍主要上下波动。闭环评测也没有得到可用表现。当前判断是：

- 需要同时学习 RoboTwin RGB、Rothko 新视觉分布和新的 block 时序关系；
- attention-only adapter 只能调整 token 交互，不能充分调整每层 FFN 的特征变换；
- 多任务训练会让“容量不足、任务冲突、控制链路问题”混在一起，难以定位。

因此新增单任务 `click_alarmclock` 配置，先判断模型是否能够把一个任务真正学会。当前
配置不是从 released FastWAM checkpoint 继续训练，而是：

```text
original Wan-AI/Wan2.2-TI2V-5B
    + newly initialized LoRA adapters
```

LoRA 当前设置：

```text
rank = 64
alpha = 64
dropout = 0
targets:
    self_attn.q/k/v/o
    cross_attn.q/k/v/o
    ffn.0
    ffn.2
```

30 层总计：

```text
attention LoRA modules = 240
FFN LoRA modules = 60
all LoRA modules = 300
trainable LoRA parameters = 161,218,560
approximately 3.22% of the 4,999,849,152-parameter Video DiT
```

这比 rank-32 attention-only 的 47,185,920 个参数大约增加到 3.42 倍。当前配置的
训练参数：

```text
task = click_alarmclock
per-GPU batch size = 8
gradient accumulation = 2
learning rate = 1e-4
max steps = 10,000
save every = 1,000
eval every = 500
```

四卡时 effective global batch 为：

```text
8 x 4 x 2 = 64
```

### 17.2 当前不再把 qpos 拼到语言 token 后

单任务配置显式设置：

```yaml
model:
  proprio_dim: null

finetune:
  train_proprio_encoder: false
```

因此当前模型的 cross-attention context 只有语言 embedding，不再把当前 normalized
qpos 经 `proprio_encoder` 映射后追加到语言 token。这样做是为了避免随机初始化的
qpos token 污染语言条件，并让实验更直接地检验：

```text
语言 + 当前 RGB + 当前 Rothko
    -> 未来 RGB + 未来 Rothko
```

当前模型实际上没有显式接收绝对 EE pose。Rothko codec 以每个 action chunk 的第一帧
为坐标基准，所以 `RAY_0` 中：

```text
relative position = 0
relative rotation = identity
```

`RAY_0` 只保留固定的 canonical ray pattern 和当前 gripper 编码，不包含当前绝对
位置或绝对朝向。当前 RGB 可能让模型间接推断机器人状态，但这不等价于精确的绝对
EE pose。真实的 `current_endpose` 只在模型生成 Rothko 后提供给几何 decoder，用于
把相对轨迹恢复成世界坐标，并没有参与 DiT 预测。这是当前无 qpos 方案的明确设计
风险。后续备选包括：

1. 重新引入独立的 state token，但不与语言语义混为同一类 token；
2. 为语言、状态、RGB、Rothko 增加显式 modality/type embedding；
3. 在 Rothko 图中加入更直接的绝对位姿锚点；
4. 保持当前无 qpos 版本作为消融基线。

### 17.3 RGB 与 Rothko 的当前 attention mask

VAE latent 的时间布局保持：

```text
[RGB_0, RGB_1, RGB_2, RGB_3, RGB_4,
 RAY_0, RAY_1, RAY_2, RAY_3, RAY_4]
```

其中 `RGB_0` 和 `RAY_0` 是 clean condition，其余 8 个 latent frame 被加噪并参与
flow-matching。旧 `condition_frames_causal` 只限制两个 condition query 不能读取
future token，但所有 future RGB/Rothko query 仍能读取完整的两段 noisy future。
这不符合“先生成未来 RGB，再让动作图利用未来世界轨迹”的预期依赖关系。

当前改为：

```text
video_attention_mask_mode = rgb_then_raymap_block_causal
```

其 frame-level 可见关系为：

```text
query RGB_0 / RAY_0:
    only keys RGB_0 and RAY_0

query future RGB:
    all RGB keys + RAY_0
    cannot see future noisy Rothko keys

query future Rothko:
    all RGB keys + all Rothko keys
```

因此未来 RGB 不会利用待预测的未来 Rothko，而未来 Rothko 可以利用联合去噪得到的
完整未来 RGB 轨迹。这里的 “block causal” 是模态 block 方向上的因果约束，不是
RGB block 内逐帧的下三角时间因果；同一个 diffusion forward 中，未来 RGB frame
之间仍是双向 attention。

模型会检查：

- condition frame 必须恰好是两个；
- RGB condition 必须是 frame 0；
- Rothko condition 必须位于第二个等长 block 的起点；
- `[RGB | Rothko]` 两段 latent frame 数必须相同。

### 17.4 Rothko loss 权重提高到 5

当前模型配置：

```yaml
loss:
  lambda_rgb: 1.0
  lambda_raymap: 5.0
```

总 loss 为：

```text
loss_total = loss_rgb_raw + 5 * loss_raymap_raw
```

需要特别注意：Trainer/W&B 中记录的 `loss_raymap` 已经乘过
`lambda_raymap=5`，不是 raw Rothko MSE。因此读取曲线时：

```text
raw Rothko loss = logged loss_raymap / 5
```

提高权重的目的不是改变 diffusion target，而是避免数值明显更小的 Rothko latent
loss 在联合优化中被 RGB 主导。后续比较不同权重时必须同时保存 raw 和 weighted
loss；当前实现的日志只直接给出 weighted value。

### 17.5 Action chunk 与录像连续性修正

训练数据保持 17 个连续 timestep：

```text
num_frames = 17
global_sample_stride = 1
action_video_freq_ratio = 1
```

因此一个训练窗口是：

```text
当前帧 + 连续未来 16 帧
```

窗口起点由 dataset sampler 在所有合法窗口中随机抽样。Rothko future pose 是相对
当前 action chunk 的第一帧/当前 EE anchor 编码，不是统一相对整个 episode 的第一
帧。

评测控制策略仍是：

```text
一次 forward 预测 16 步
连续执行这 16 步
队列耗尽后再用新观测 replan
```

之前 `skip_get_obs_within_replan=true` 会让 RoboTwin 在执行 action queue 时跳过中间
观测获取，导致保存的 rollout MP4 看起来像两帧之间隔了很久。这不是 dataset 抽样
间隔，也不是模型只预测稀疏帧，而是评测录像没有逐环境步刷新。

当前配置改为：

```yaml
EVALUATION:
  skip_get_obs_within_replan: false
```

现在每个 simulator step 都获取并记录最新观测，所以 rollout 视频连续；但 policy
的 `should_request_observation()` 仍只在 action queue 为空时要求把观测送入模型。
也就是说控制仍是 16 步 open-loop chunk，并没有变成每步重新推理。

### 17.6 EE decode 精度修正

Rothko future pose 是相对当前 EE pose 的表示，decode 时需要用当前绝对
`endpose` 恢复世界坐标。此前当前 pose anchor 可能被转换成 BF16，毫米级位置和小角度
旋转会受到不必要的量化影响。

当前在 policy 和模型内部都将 decode anchor 显式保持为：

```text
torch.float32
```

Raymap 网络本身仍按训练 mixed precision 运行；这里只提高几何恢复阶段的 anchor
精度，不增加 DiT 的显存开销。

### 17.7 Checkpoint 的 attention 语义与 LoRA 加载

新 checkpoint 的 `visual_action_config` 会额外保存：

```text
video_attention_mask_mode
loss_weights.rgb
loss_weights.raymap
```

加载时会严格检查当前模型和 checkpoint 的 attention mask 是否一致，防止把用旧
`condition_frames_causal` 训练的权重静默放进
`rgb_then_raymap_block_causal` 模型。

部分已经启动的长程 run 使用的是新 mask，但其进程在 checkpoint metadata 代码更新
之前启动，所以 `.pt` 内没有 `video_attention_mask_mode`。对这类 checkpoint，loader
会向上查找相邻 run 的 `config.yaml`，读取：

```text
model.video_dit_config.video_attention_mask_mode
```

如果既没有 checkpoint 字段，也找不到有效 run config，才按真正的旧 checkpoint
处理为：

```text
condition_frames_causal
```

LoRA 评测时必须给出具体 adapter 文件，例如：

```text
.../checkpoints/weights/step_010000.pt
```

不能把 `.../checkpoints/` 目录传给 `ckpt`。adapter-only checkpoint 加载流程是先从
原始 Wan2.2 构建 base，再根据 checkpoint 元信息注入并加载 LoRA；full checkpoint
则加载完整 DiT state。

### 17.8 预测 RGB/Rothko 视频保存

RoboTwin 评测配置新增：

```yaml
EVALUATION:
  save_prediction_videos: true
```

每次模型重新规划后，都会把本次 forward 生成的两种模态分别保存：

```text
<evaluation output>/
  predictions/
    <task_config>/
      episode_000/
        replan_000_env_step_0000_rgb.mp4
        replan_000_env_step_0000_raymap.mp4
        replan_001_env_step_0016_rgb.mp4
        replan_001_env_step_0016_raymap.mp4
```

两个视频均为 10 FPS。RGB 视频使用模型实际看到/生成的 `384 x 320` 三相机布局：

```text
top 256 x 320: head camera
bottom-left 128 x 160: left wrist
bottom-right 128 x 160: right wrist
```

Rothko 视频是独立的 `384 x 320` 图，而不是与 RGB 合并在同一张像素图内。两者先
分别通过同一个 Wan VAE，再在 latent 时间维按 `[RGB block | Rothko block]` 拼接。

RoboTwin 自带的 episode rollout MP4 使用另一套可视化画布：上方 raw head camera
较窄，右上可能出现黑色 padding，下方放两个 wrist camera。该黑块只属于 simulator
录像布局，不是模型 RGB 输入的一部分。

### 17.9 Manager 与 single evaluation 改造

`run_robotwin_manager.py` 当前支持：

- `EVALUATION.task_name=<one task>`；
- `EVALUATION.task_names=[task_a,task_b,...]`；
- `MULTIRUN.gpu_ids=[4,5,6,7]` 形式的非连续物理 GPU；
- `MULTIRUN.max_tasks_per_gpu`；
- worker 失败时终止同批其余 worker 并保存 summary；
- 预先校验未知 task；
- 禁止同时设置 `task_name` 和 `task_names`。

Hydra list override 需要整体作为一个 shell argument。正确写法：

```bash
'EVALUATION.task_names=[stack_bowls_three,place_shoe,click_alarmclock,blocks_ranking_rgb]'
'MULTIRUN.gpu_ids=[4,5,6,7]'
```

引号由 shell 消费，不会传入 Python 字符串；其作用是避免 shell 对方括号等字符做
展开。开引号后不能多一个前导空格，例如下面是错误的：

```text
' EVALUATION.task_names=[...]'
```

`eval_robotwin_single.py` 还会把 `finetune.method`、预测视频开关、dataset stats 和
Rothko stats 的绝对路径传入 RoboTwin policy。LoRA 模式强制加载原始 base；
Rothko stats 相对路径会先按 FastWAM 项目根目录解析，避免子进程切换到 RoboTwin
目录后找错文件。

### 17.10 当前单任务训练与评测命令

四卡训练：

```bash
conda activate fastwam
cd /mnt/hwdata/cfy/FastWAM

CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29543 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4
```

使用 10k adapter、GPU 7 做单任务闭环评测：

```bash
conda activate fastwam_robotwin
cd /mnt/hwdata/cfy/FastWAM

TOKENIZERS_PARALLELISM=false \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4 \
  ckpt=/mnt/hwdata/cfy/FastWAM/runs/robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4/2026-07-26_12-27-09/checkpoints/weights/step_010000.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.eval_num_episodes=15 \
  EVALUATION.replan_steps=16 \
  EVALUATION.save_prediction_videos=true \
  EVALUATION.dataset_stats_path=/mnt/hwdata/cfy/FastWAM/runs/robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4/2026-07-26_12-27-09/dataset_stats.json \
  'MULTIRUN.gpu_ids=[7]' \
  MULTIRUN.max_tasks_per_gpu=1
```

`TOKENIZERS_PARALLELISM=false` 只关闭 Hugging Face tokenizer 内部 CPU thread pool，
用于避免 tokenizer 初始化后再 `fork` 时的 deadlock warning；它不关闭多 GPU
评测，也不改变模型输出。

### 17.11 当前仍需验证的事项

下一阶段应优先比较：

1. 单任务 10k checkpoint 的完整 15-episode `demo_clean` 和
   `demo_randomized` 成功率；
2. 保存的 predicted RGB/Rothko MP4 是否具有正确时序、布局和动作方向；
3. 每次 replan 的第 1 步 EE jump，以及 16 步 chunk 末端累计误差；
4. 新 block mask 与旧 `condition_frames_causal` 的同配置消融；
5. `lambda_raymap=1/2/5` 的 raw loss、decode error 和闭环成功率；
6. rank-64 attention-only、rank-64 attention+FFN 和 full fine-tuning；
7. 无 qpos、独立 state token、带 modality embedding 三种条件方式；
8. 单任务验证稳定后，再逐步扩展到 4 任务和完整 8 任务。
