# FastWAM Video-Only Raymap Action 方案与备选设计

最后更新：2026-07-24

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
- 在物理 GPU 6、7 上完成 100-step、双卡 ZeRO-1 真实训练并成功保存 checkpoint。

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
