# FastWAM LIBERO 单臂 Rothko 训练与评测改造方案

> 文档状态：第一版代码与数据改造已完成，待正式训练/评测
> 更新时间：2026-07-31
> 目标：在不改变现有 RoboTwin 训练和评测行为的前提下，让 FastWAM 在
> LIBERO 上只使用 Wan video expert，通过联合预测未来 RGB 和单臂 Rothko
> raymap 来输出机器人动作。

## 1. 已确认的选择

以下决定已经由用户确认，后续实现以此为准。

### 1.1 数据范围

直接使用现有 LIBERO LeRobot 数据：

```text
/mnt/hwdata/cfy/FastWAM/data/libero_mujoco3.3.2
```

同时训练以下四个套件：

```text
libero_spatial_no_noops_lerobot
libero_object_no_noops_lerobot
libero_goal_no_noops_lerobot
libero_10_no_noops_lerobot
```

当前总量约为：

```text
40 个任务
1712 个 episodes
277713 帧
```

四套数据继续按照当前 FastWAM 的方式合并为一个
`MultiLeRobotDataset`，统一 shuffle 和采样。第一版不增加：

- 单任务筛选；
- suite/task 均衡采样；
- 单任务正式训练配置；
- 对 LIBERO 数据的重新采集。

### 1.2 允许修改 parquet

允许在现有 LIBERO parquet 中增加 Rothko 所需的绝对 EE 信息。第一版使用：

```text
专家增量 action
    + 当前绝对 EE pose
    -> 专家 action 对应的绝对 OSC controller target
```

不使用“执行一个仿真步后实际到达的下一帧 EE pose”作为正式 Rothko future
target。后者只保留为后续消融方案。

### 1.3 单臂 Rothko 布局

一个 `224 x 224` 单臂 Rothko 横向复制为：

```text
[single-arm Rothko | identical single-arm Rothko]
                  224 x 448
```

它与现有 LIBERO 两相机 RGB 画布保持完全相同的空间尺寸。

### 1.4 时间长度

第一版使用连续 17 帧：

```text
frame 0       = 当前条件
frames 1..16  = 未来预测目标
action horizon = 16
```

数据是 20 Hz，因此一个 chunk 覆盖约 0.8 秒。第一版不沿用标准 FastWAM 的
32 action / 9 RGB frames 的 4:1 时间比例。

### 1.5 模型与 VAE 初始化

第一版从原始 Wan2.2 初始化：

```text
Wan-AI/Wan2.2-TI2V-5B video DiT
原始 Wan2.2 VAE
```

明确不使用：

- released LIBERO FastWAM checkpoint 的 video expert；
- RoboTwin released FastWAM checkpoint；
- `Wan2.2_VAE_rothko_baseline_step005000.safetensors`；
- 其他微调 VAE。

对应配置应保持：

```yaml
model:
  skip_dit_load_from_pretrain: false
  vae_safetensors_path: null
```

## 2. 当前 LIBERO FastWAM 基线

当前配置位于：

```text
configs/data/libero_2cam.yaml
configs/task/libero_uncond_2cam224_1e-4.yaml
configs/model/fastwam.yaml
```

四套 LIBERO 数据被一起加载。一个标准样本包含：

```text
33 个连续 observation
32 个连续 7D action
每隔 4 帧采一张 RGB
最终得到 9 帧、224 x 448 的双相机 RGB
```

模型结构为：

```text
language + RGB + proprio
    -> video expert + action expert + MoT
    -> future RGB + 7D delta action
```

目标改造后的结构为：

```text
language + current RGB + current single-arm Rothko
    -> Wan video expert only
    -> future RGB + future single-arm Rothko
    -> Rothko decode
    -> absolute EE controller targets
    -> LIBERO normalized delta OSC actions
```

## 3. 专家 action 对应的绝对 OSC target

## 3.1 已有原始字段

现有 parquet 已经包含：

```text
observation.states.ee_state
    [x, y, z, axis_angle_x, axis_angle_y, axis_angle_z]

observation.states.gripper_state
    [left_finger_qpos, right_finger_qpos]

action
    [dx, dy, dz, dRx, dRy, dRz, gripper]
```

其中 `observation.states.ee_state` 是世界坐标下的绝对 EE position 和
absolute axis-angle rotation；`action[:6]` 是送入 robosuite
`OSC_POSE` controller 的归一化增量命令。

## 3.2 Controller 参数

当前 LIBERO/robosuite `OSC_POSE` 配置为：

```text
position output range = [-0.05, 0.05] m
rotation output range = [-0.5, 0.5] rad
control_delta = true
```

因此：

```text
delta_position_m
    = 0.05 * action_xyz

delta_rotation_axis_angle
    = 0.5 * action_rotation
```

## 3.3 绝对 target 计算

设当前绝对位姿为：

```text
p_current
R_current
```

专家 action 对应的绝对 controller target 为：

```text
p_target
    = p_current + 0.05 * action_xyz

R_delta
    = axis_angle_to_matrix(0.5 * action_rotation)

R_target
    = R_delta @ R_current
```

这里的旋转左乘顺序必须与 robosuite
`set_goal_orientation()` 保持一致，不允许根据经验更换顺序。

最终 parquet 中保存：

```text
position xyz + quaternion wxyz
```

四元数内部统一为 `wxyz`。从 LIBERO observation 的 axis-angle 转换时，要在接近
180 度的区域进行数值稳定测试；LIBERO 数据中确实存在接近 `pi` 的 absolute
rotation。

## 3.4 时间对齐

对于窗口起点 `t`：

```text
Rothko frame 0
    = observation EE pose at row t

Rothko frame 1
    = action row t 对应的 absolute OSC target

Rothko frame 2
    = action row t+1 对应的 absolute OSC target

...

Rothko frame 16
    = action row t+15 对应的 absolute OSC target
```

对应 RGB：

```text
RGB frames
    = observation rows t ... t+16
```

所以：

```text
action target row t+k
    对应 observation transition t+k -> t+k+1
```

## 4. Parquet 扩展方案

## 4.1 新增脚本

新增：

```text
scripts/add_libero_osc_target_pose.py
```

脚本遍历四个 dataset 的全部 parquet，默认支持：

```text
--dry-run
--workers
--skip-existing
--verify
--limit
```

## 4.2 推荐新增字段

每一行增加：

```text
observation.state.ee_pose_wxyz
    shape [7]
    当前绝对 EE xyz + quaternion wxyz

action.osc_target_pose_wxyz
    shape [7]
    当前行 expert action 对应的绝对 OSC target

observation.state.gripper_open
    shape [1]
    当前物理夹爪状态映射到 [0,1]
```

使用这些名字是为了兼容 `BaseLerobotDataset` 现有 side-channel 规则：

```text
raw_state_meta key=ee_pose_wxyz
    -> observation.state.ee_pose_wxyz

raw_action_meta key=osc_target_pose_wxyz
    -> action.osc_target_pose_wxyz
```

未来 action gripper 不需要重复新增字段，直接读取原始 `action[..., -1]`。

## 4.3 Gripper 当前状态

`observation.states.gripper_state` 是两个 Panda gripper joint qpos。转换脚本根据
官方 Panda gripper joint range 将它映射为：

```text
0 = closed
1 = open
```

实现前先在四套数据上核对：

- 两个关节的符号；
- fully-open joint range；
- 与原始 action gripper 的时序关系；
- 是否存在超出理论 joint range 的数值。

最终使用的 joint range 和转换公式必须写入新增字段的 metadata 和 Rothko stats
metadata，不能只作为代码中的隐式常量。

## 4.4 元数据更新

脚本同步更新：

```text
meta/info.json
meta/episodes_stats.jsonl
```

新 feature 的 dtype 为 `float32`，并写明 shape 和 names。

在批量执行前：

1. 备份小型 metadata 文件；
2. 先在一个 episode 的临时副本上验证；
3. 检查新增列和旧列逐行长度一致；
4. 检查原始 action/state 列内容没有变化；
5. 再运行全量转换。

每个 parquet 使用：

```text
同目录临时文件
    -> 完整写入和读取验证
    -> os.replace 原子替换
```

单个 worker 异常时不能留下半写入 parquet。

## 4.5 Idempotency

重复执行脚本时：

- 字段存在且 metadata/数值验证通过：跳过；
- 字段存在但 metadata 不匹配：报错；
- 只有部分字段存在：报错，不静默覆盖；
- 只有显式 `--force` 才允许重新计算和覆盖。

## 5. 单臂 LIBERO Rothko codec

## 5.1 与 RoboTwin 隔离

保留现有：

```text
RothkoCodec
    environment = robotwin
    num_arms = 2
    pose_dim = 14
    gripper_dim = 2
    image = 384 x 320
```

新增：

```text
LiberoRothkoCodec
    environment = libero
    num_arms = 1
    pose_dim = 7
    gripper_dim = 1
    image = 224 x 448
```

推荐文件：

```text
src/fastwam/representations/libero_rothko.py
```

可以复用 `rothko.py` 内部的单机械臂几何、四元数和 SVD 解码函数，但不得改变
现有 `RothkoCodec` 的公开输入输出和 checkpoint metadata。

## 5.2 画布

单个基础 tile：

```text
height = 224
width = 224
```

完整画布：

```text
left tile  = single-arm Rothko
right tile = identical copy
image      = 224 x 448
```

第一版参数沿用 RoboTwin Rothko 几何默认值，除图像/tile 尺寸和单臂布局外：

```text
focal = 0.2
center_scale = 1.0
dir_scale = 1.0
center_frac = 0.5
boundary_margin = 8
outer_margin = 8
```

如果 stats 或 VAE roundtrip 表明这些参数不适合 LIBERO，再作为显式消融调整。

## 5.3 编码

输入：

```text
pose       [B,T,7] or [T,7]
gripper    [B,T,1] or [T,1]
```

流程：

1. 以 chunk frame 0 为共同绝对 pose anchor；
2. future translation 转到 frame-0 EE coordinate frame；
3. future rotation 表示为相对 frame-0 rotation；
4. center region 写 relative translation；
5. peripheral region 写 rotated canonical ray directions；
6. normalization 到 `[-1,1]`；
7. gripper 写入 decoder 不使用的外侧 8 像素；
8. 横向复制完整 tile；
9. 返回 `[B,3,T,224,448]`。

## 5.4 解码

1. 横向两份 tile 分别读取；
2. 几何 raw values 取平均；
3. gripper border 取中位数；
4. median 解码 relative origin；
5. SVD 解码 relative rotation；
6. 使用当前真实 absolute EE pose 恢复世界坐标 target；
7. 返回：

```text
pose       [B,T,7]
gripper    [B,T,1]
```

## 6. LIBERO Rothko normalization stats

新增：

```text
scripts/compute_libero_rothko_norm_stats.py
```

输入为四个已经添加 absolute target 字段的 LIBERO dataset。

统计窗口与正式训练完全一致：

```text
current
    = observation.state.ee_pose_wxyz[t]

future
    = action.osc_target_pose_wxyz[t:t+16]

horizon
    = 16
```

统计所有 future target 相对 current anchor 的 EE-frame translation，生成对称
Q99.95 bounds。

输出：

```text
data/libero_mujoco3.3.2/
    libero_rothko_region_symmetric_q99p95_h16_224x448.pt
    libero_rothko_region_symmetric_q99p95_h16_224x448.json
```

metadata 至少包含：

```text
environment = libero
representation = rothko
encoding = current_ee_plus_future_absolute_osc_targets
layout = single_arm_duplicated_horizontal
image_size = [224,448]
tile_size = [224,224]
pose_dim = 7
gripper_dim = 1
quaternion_order = wxyz
action_horizon = 16
pixel_frames = 17
controller_position_scale = 0.05
controller_rotation_scale = 0.5
translation_quantile
translation_abs_bounds_xyz_m
observed_max_abs_xyz_m
focal
center_scale
dir_scale
center_frac
boundary_margin
outer_margin
gripper_conversion
dataset_roots
num_episodes
num_windows
```

## 7. Dataset 适配

新增：

```text
configs/data/libero_rothko_2cam224.yaml
```

核心配置：

```yaml
train:
  dataset_dirs:
    - ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot

  num_frames: 17
  global_sample_stride: 1
  action_video_freq_ratio: 1
  video_size: [224,448]
  concat_multi_camera: horizontal
  raymap_representation: libero_rothko
  val_set_proportion: 0.0

  raw_action_meta:
    - key: default
      raw_shape: 7
    - key: osc_target_pose_wxyz
      raw_shape: 7

  raw_state_meta:
    - key: ee_pose_wxyz
      raw_shape: 7
    - key: gripper_open
      raw_shape: 1
```

第一版维持当前 FastWAM 的四套件联合采样方式，不增加
`libero_task_names` 或 suite filter。

Dataset 返回：

```text
video
    [3,17,224,448]

raymap
    [3,17,224,448]

current_endpose
    [7]

future_endpose
    [16,7]

future_gripper
    [16,1]

image_is_pad
    [17]

raymap_is_pad
    [17]
```

配置分支必须明确：

```text
raymap_representation = null
    -> 标准 FastWAM，不生成 raymap

raymap_representation = rothko
    -> 原 RoboTwin 双臂 Rothko

raymap_representation = libero_rothko
    -> 新 LIBERO 单臂 Rothko
```

不能通过 pose shape 或 dataset path 自动猜测环境。

## 8. RGB 预处理

LIBERO 继续使用两相机：

```text
agentview
eye-in-hand
```

每个相机生成 `224 x 224`，水平拼为：

```text
[agentview | wrist]
       224 x 448
```

训练和部署应抽取共享的 LIBERO canvas builder，明确统一：

- 输入颜色顺序；
- LIBERO observation 的 180 度旋转；
- resize/crop 顺序；
- interpolation；
- antialias；
- uint8/float 转换；
- `[-1,1]` normalization；
- 相机左右顺序。

现有标准 LIBERO 配置的视觉行为不能被新 Rothko 配置隐式改变；共享 builder 上线前要
对现有训练路径做数值对比。

## 9. 模型适配

## 9.1 复用的核心

继续复用：

```text
FastWAMVideoOnlyRaymap
Wan2.2 video DiT
Wan2.2 VAE
RGB/Rothko 分别 VAE encode
[RGB latent block | Rothko latent block]
flow-matching scheduler
rgb_then_raymap_block_causal attention mask
RGB/Rothko 分项 loss
LoRA 注入、保存和续训
```

17 pixel frames 经 Wan VAE 后，每个 modality 得到：

```text
5 latent frames
```

联合 latent 时间布局：

```text
RGB latent frames     0..4
Rothko latent frames  5..9
```

条件 latent：

```text
RGB_0
Rothko_0
```

预测：

```text
RGB_1..4 latent groups
Rothko_1..4 latent groups
```

## 9.2 去除双臂硬编码

当前 visual-action model 和 Trainer 中存在 RoboTwin 专用假设：

- pose 必须是 14 维；
- gripper 必须是 2 维；
- 固定切分左右臂；
- 固定拼成 16D RoboTwin EE action；
- 验证指标固定比较左右臂字段。

改为由 codec 明确提供：

```text
environment
pose_dim
gripper_dim
num_arms
layout
encode/decode
pack deployment action（如适用）
```

通用模型输出：

```text
prediction["video"]
prediction["video_tensor"]
prediction["raymap"]
prediction["pose"]
prediction["gripper"]
```

RoboTwin 为兼容现有 deployment，继续保留：

```text
prediction["action"]
    -> 原 16D 双臂 EE action
```

LIBERO eval 不使用该 RoboTwin action packing，而是读取 `pose/gripper` 后在
LIBERO adapter 中转换成 7D OSC action。

## 9.3 模型初始化

新增 LIBERO 模型配置：

```text
configs/model/fastwam_video_only_libero_rothko.yaml
```

它明确从原始 Wan 初始化：

```yaml
_target_: fastwam.runtime.create_fastwam_video_only_raymap
skip_dit_load_from_pretrain: false
vae_safetensors_path: null
action_horizon: 16
```

第一版不增加 regular FastWAM checkpoint 到 video-only model 的 video expert
迁移逻辑。

## 9.4 Proprio

第一版建议保持与当前已验证的 RoboTwin Rothko LoRA 配置一致：

```yaml
model:
  proprio_dim: null
```

即不把 qpos/proprio token 追加到语言 context 后。当前绝对 EE pose只用于：

- 构造 Rothko frame 0；
- 将模型预测的 relative Rothko 恢复为世界坐标 target；
- 部署时做反馈控制。

如果后续需要研究显式 proprio，这是单独消融，不与第一版混合。

## 9.5 Loss

初始配置：

```yaml
loss:
  lambda_rgb: 1.0
  lambda_raymap: 5.0
```

W&B/Trainer 中同时记录：

```text
raw loss_rgb
weighted loss_rgb
raw loss_raymap
weighted loss_raymap
total loss
```

避免把乘过 `lambda_raymap` 的值误认为 raw Rothko loss。

## 10. LIBERO 推理与控制

只在 LIBERO 评测链路中增加 visual-action 分支：

```text
experiments/libero/eval_libero_single.py
experiments/libero/libero_utils.py
```

不修改 RoboTwin policy 的控制方式。

## 10.1 每次 replan

1. 读取当前 agentview 和 wrist RGB；
2. 构造训练一致的 `224 x 448` RGB；
3. 读取当前 `robot0_eef_pos`；
4. 读取当前 `robot0_eef_quat` 并从 `xyzw` 转成 `wxyz`；
5. 读取当前 gripper qpos 并映射到 `[0,1]`；
6. 构造单帧当前 Rothko；
7. 调用 video-only model 联合预测未来 RGB 和 Rothko；
8. Rothko decode 得到 16 个 absolute EE targets 和 gripper；
9. 保存 target 队列；
10. 执行前 `replan_steps` 个 target，然后重新规划。

## 10.2 Absolute target 转增量 action

执行 target `k` 前重新读取真实当前位姿：

```text
p_actual
R_actual
```

位置 action：

```text
action_xyz
    = (p_target - p_actual) / 0.05
```

旋转 action：

```text
R_delta
    = R_target @ R_actual.T

action_rotation
    = matrix_to_axis_angle(R_delta) / 0.5
```

随后：

```text
clip motion action to [-1,1]
gripper threshold at 0.5
convert gripper to LIBERO env convention
env.step(action)
```

这里使用每步真实反馈，而不是在 replan 时一次性把整个 absolute target chunk 转成
固定 delta action chunk。

## 10.3 推荐评测参数

```yaml
EVALUATION:
  action_horizon: 16
  replan_steps: 10
  binarize_gripper: true
```

模型预测 16 个 target，但执行 10 步后重新规划。

## 10.4 诊断输出

增加可选：

```yaml
EVALUATION:
  save_prediction_videos: true
  save_control_trace: true
```

每次 replan 可保存：

```text
predicted_rgb.mp4
predicted_raymap.mp4
control_trace.jsonl
```

`control_trace.jsonl` 每个 env step 记录：

```text
episode
env_step
replan_index
target_index
predicted_target_xyz
predicted_target_rotation
actual_xyz_before_action
actual_rotation_before_action
sent_delta_action
predicted_gripper
clipped_dimensions
```

这些记录只属于 LIBERO visual-action eval，不加入 RoboTwin 的默认评测输出。

## 11. 训练配置

新增：

```text
configs/task/libero_rothko_2cam224_lora_r64_attn_ffn_1e-4.yaml
```

第一版建议使用已有 LoRA 机制：

```yaml
finetune:
  method: lora
  train_proprio_encoder: false
  lora:
    rank: 64
    alpha: 64.0
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
      - ffn.0
      - ffn.2
```

基础训练设置先对齐当前 LIBERO FastWAM：

```yaml
learning_rate: 1.0e-4
num_epochs: 10
max_steps: null
lr_scheduler_type: cosine
weight_decay: 1.0e-2
```

允许使用的物理 GPU 仅为：

```text
4,5,6,7
```

显存 smoke test 前不锁死 per-GPU batch。目标有效 global batch 暂定对齐标准 FastWAM
的 128，例如：

```text
per-GPU batch 8
x 4 GPUs
x gradient accumulation 4
= effective global batch 128
```

最终 batch/accumulation 根据单 batch 实测调整，但不通过更换 GPU 0..3 解决。

## 12. 验证与测试门槛

## 12.1 Parquet 转换验证

正式全量执行前，先验证一个 episode：

1. 新增字段 shape 正确；
2. 所有值 finite；
3. quaternion norm 接近 1；
4. 原始列逐元素未变化；
5. action target 的 position/rotation 符合 controller 公式；
6. metadata 能被 LeRobot loader 读取；
7. 重复运行具有 idempotency。

全量执行后报告：

```text
episodes ok/failed/skipped
rows processed
position target range
rotation delta range
gripper-open range
invalid quaternion count
```

## 12.2 OSC roundtrip

必须实现独立 CPU 测试：

```text
expert normalized action
    -> absolute OSC target
    -> 使用同一 current pose 反算 normalized action
```

要求：

```text
position action error <= 数值浮点容差
rotation action error <= 数值浮点容差
gripper exact match
```

这个测试验证格式转换，不是重新生成或恢复专家数据。

## 12.3 Codec 几何 roundtrip

不经过 VAE：

```text
absolute targets
    -> LiberoRothkoCodec.encode
    -> LiberoRothkoCodec.decode
```

报告：

```text
position error in mm
rotation geodesic error in degree
gripper error
```

同时测试：

- quaternion 正负号等价；
- 接近 180 度的 rotation；
- 零位移/零旋转；
- 最大统计范围附近；
- 两份横向 tile 的融合；
- gripper border。

## 12.4 Original Wan VAE roundtrip

使用原始 Wan VAE：

```text
Rothko
    -> VAE encode
    -> VAE decode
    -> Rothko geometry decode
```

报告：

```text
Rothko pixel/latent reconstruction
position error
rotation error
gripper accuracy
```

如果原始 VAE roundtrip 已经无法保留可用动作精度，应先分析 layout/stats，而不是立即
切换到 RoboTwin 微调 VAE。

## 12.5 Dataset smoke test

从四个 suite 分别随机抽样，检查：

```text
video shape = [3,17,224,448]
raymap shape = [3,17,224,448]
current pose shape = [7]
future pose shape = [16,7]
future gripper shape = [16,1]
RGB/Rothko range = [-1,1]
帧连续
动作与 RGB 时间对齐
padding mask 对齐
语言缓存可命中
```

## 12.6 GPU smoke test

只使用 GPU 4/5/6/7：

1. 单 batch dataset + collate；
2. 原始 Wan model load；
3. forward；
4. backward；
5. LoRA trainable parameter 检查；
6. optimizer step；
7. 完整 20-step inference；
8. RGB/Rothko decode；
9. 显存和耗时记录。

## 12.7 短训练验证

在完整四套件 dataset 上跑短训练，不做任务筛选：

```text
100 step
```

检查：

- loss finite；
- RGB/Rothko loss 都能正常反传；
- checkpoint 可以保存和加载；
- checkpoint metadata 正确；
- 续训恢复 optimizer/scheduler/step；
- 验证生成视频和 action decode 可运行。

如需纯过拟合检查，可以固定极少数 batch 做临时诊断，但不把它做成正式单任务训练配置。

## 12.8 完整训练和评测

短测试通过后：

```text
四套件联合训练
10 epochs
```

使用现有 LIBERO manager 评测全部 40 个任务：

```text
experiments/libero/run_libero_manager.py
```

对比：

```text
现有 FastWAM action-expert baseline
新 video-only LIBERO Rothko
```

报告：

```text
overall success rate
per-suite success rate
per-task success rate
RGB loss/PSNR
Rothko raw/weighted loss
EE position decode error
EE rotation decode error
gripper accuracy
OSC action clipping rate
```

## 13. RoboTwin 无回归要求

## 13.1 配置隔离

现有配置不改语义：

```text
configs/data/robotwin_rothko.yaml
configs/model/fastwam_video_only_raymap.yaml
configs/task/robotwin_*rothko*.yaml
configs/sim_robotwin.yaml
```

LIBERO 使用全新的：

```text
configs/data/libero_rothko_2cam224.yaml
configs/model/fastwam_video_only_libero_rothko.yaml
configs/task/libero_rothko_2cam224_lora_r64_attn_ffn_1e-4.yaml
```

## 13.2 Codec 隔离

```text
raymap_representation = rothko
    -> 永远表示 RoboTwin dual-arm codec

raymap_representation = libero_rothko
    -> 永远表示 LIBERO single-arm codec
```

旧名字不能自动重定向到新 codec。

## 13.3 Checkpoint 隔离

visual-action checkpoint metadata 增加并严格校验：

```text
environment
codec_type
pose_dim
gripper_dim
num_arms
image_size
tile_size
layout
action_horizon
controller scales（LIBERO）
quaternion_order
loss weights
attention mask mode
```

LIBERO checkpoint 传给 RoboTwin deployment、或 RoboTwin checkpoint 传给 LIBERO
deployment 时必须立即报清晰错误，不能以 `strict=False` 静默加载。

## 13.4 RoboTwin 回归测试

改造前后用固定输入验证：

```text
RoboTwin Rothko encode 数值一致
RoboTwin Rothko decode 数值一致
RoboTwin gripper border 一致
RoboTwin stats 仍可加载
RoboTwin checkpoint 仍可严格加载
RoboTwin 16D EE action packing 一致
RoboTwin dataset sample shape 一致
RoboTwin Hydra resolved config 一致
```

RoboTwin 预期 shape：

```text
RGB        [3,17,384,320]
Rothko     [3,17,384,320]
pose       14D dual-arm
gripper    2D dual-arm
action     16D dual-arm EE control
```

第一版不修改：

```text
experiments/robotwin/fastwam_policy/deploy_policy.py
third_party/RoboTwin
RoboTwin action_type=ee 执行方式
RoboTwin replan/execute 语义
```

如果实现中发现必须修改上述文件，应先说明原因和影响，再由用户决定。

## 14. 实施顺序

严格按以下顺序执行，每一阶段通过后再进入下一阶段。

### Phase 0：冻结基线

1. 保存当前 git status/commit；
2. 记录 RoboTwin codec 固定输入输出；
3. 记录现有 LIBERO dataset sample；
4. 添加 RoboTwin regression tests。

### Phase 1：OSC 数学与 parquet 单文件转换

1. 实现 axis-angle/quaternion/matrix 转换；
2. 实现 expert action -> absolute target；
3. 实现 absolute target -> normalized action；
4. 通过 OSC roundtrip；
5. 在一个 parquet 临时副本上添加字段；
6. 验证 metadata 和 loader。

### Phase 2：全量 parquet 扩展

1. 备份四套 dataset 的 metadata；
2. 全量原子写入新增字段；
3. 更新 info/stats；
4. 全量 verify；
5. 输出转换报告。

### Phase 3：单臂 codec 与 stats

1. 实现 `LiberoRothkoCodec`；
2. 通过 raw geometry roundtrip；
3. 计算四套件 normalization stats；
4. 检查 clipping rate；
5. 通过 original Wan VAE roundtrip。

### Phase 4：Dataset

1. 新增 LIBERO Rothko data config；
2. 接入新 raw columns；
3. 在线生成 17 帧 Rothko；
4. 检查全部 shape、padding 和时间对齐；
5. 确认没有任务筛选。

### Phase 5：Model/Trainer 泛化

1. codec factory 显式区分环境；
2. 去掉 Trainer/visual model 的双臂维度硬编码；
3. 保留 RoboTwin action 输出兼容；
4. 增加严格 checkpoint metadata；
5. 跑 RoboTwin regression tests。

### Phase 6：LIBERO eval

1. 构造 current Rothko；
2. 联合预测 RGB/Rothko；
3. decode absolute targets；
4. 每步 feedback 转换为 OSC delta action；
5. 保存 prediction video 和 control trace；
6. 单 episode smoke test。

### Phase 7：训练

1. GPU 4/5/6/7 单 batch 测试；
2. 确定 batch/gradient accumulation；
3. 四套件 100-step test；
4. checkpoint save/load/resume；
5. 四套件完整 10-epoch 训练。

### Phase 8：完整评测与文档

1. 评测四个 suite、40 个任务；
2. 与现有 FastWAM baseline 对比；
3. 汇总 loss、decode error、clipping 和 success rate；
4. 更新 README/运行命令；
5. 记录后续消融。

## 15. 预计文件改动

### 新增

```text
FASTWAM_LIBERO_ROTHKO_PLAN_ZH.md
scripts/add_libero_osc_target_pose.py
scripts/compute_libero_rothko_norm_stats.py
src/fastwam/representations/libero_rothko.py
src/fastwam/control/libero_osc.py
configs/data/libero_rothko_2cam224.yaml
configs/model/fastwam_video_only_libero_rothko.yaml
configs/task/libero_rothko_2cam224_lora_r64_attn_ffn_1e-4.yaml
tests/test_libero_osc.py
tests/test_libero_rothko.py
tests/test_robotwin_rothko_regression.py
```

### 谨慎修改

```text
src/fastwam/datasets/lerobot/robot_video_dataset.py
src/fastwam/models/wan22/fastwam_visual_action.py
src/fastwam/runtime.py
src/fastwam/trainer.py
experiments/libero/eval_libero_single.py
experiments/libero/libero_utils.py
```

### 第一版不修改

```text
configs/data/robotwin_rothko.yaml
configs/sim_robotwin.yaml
experiments/robotwin/fastwam_policy/deploy_policy.py
third_party/RoboTwin
```

## 16. 后续备选方案

以下不属于第一版，只在基线完成后做消融：

1. 用未来实际 EE pose 代替 absolute OSC target；
2. 用一个非复制的 `224 x 448` 单臂 Rothko；
3. 使用连续 33 帧、预测 32 步；
4. 从 released LIBERO FastWAM video expert 初始化；
5. 使用微调过的 Rothko VAE；
6. 给模型增加 proprio token；
7. full fine-tuning 对比 LoRA；
8. LoRA rank/target modules 消融；
9. `lambda_raymap` 消融；
10. `replan_steps` 消融；
11. 当前绝对 EE pose 的显式 modality embedding；
12. 单步 feedback controller gain/saturation 消融。

## 17. 第一版完成标准

同时满足以下条件才算 LIBERO Rothko 第一版完成：

1. 四套 parquet 新字段全部生成并验证通过；
2. expert action 与 absolute OSC target roundtrip 通过；
3. 单臂 Rothko raw codec roundtrip 通过；
4. original Wan VAE roundtrip 误差完成量化；
5. 四套件 dataset 可返回连续 17 帧 RGB/Rothko；
6. 原始 Wan2.2 video-only 模型可 forward/backward/infer；
7. LIBERO eval 能将 predicted Rothko 转成合法 7D OSC action；
8. 可以保存预测 RGB、Rothko 和 control trace；
9. 四套件联合训练可以保存、加载和断点续训；
10. 全 40 任务评测可以由 manager 运行；
11. RoboTwin codec、checkpoint、dataset 和 eval 回归测试全部通过；
12. 不存在 LIBERO/RoboTwin checkpoint 静默交叉加载。

## 18. 2026-07-31 第一版实施记录

### 18.1 已完成的数据改造

四套 LIBERO parquet 已增加：

```text
observation.state.ee_pose_wxyz
action.osc_target_pose_wxyz
observation.state.gripper_open
```

执行结果：

```text
episodes = 1712/1712
rows = 277713
failed = 0
position action roundtrip max error = 1.776e-15
rotation action roundtrip max error = 3.653e-10
```

`info.json` 和 `episodes_stats.jsonl` 已同步更新；原 metadata 的备份文件为：

```text
meta/info.json.before_libero_osc_fields
meta/episodes_stats.jsonl.before_libero_osc_fields
```

全量转换后又运行了一次 `--verify`，1712 个 episode 全部通过。

使用的命令：

```bash
python scripts/add_libero_osc_target_pose.py \
  --workers 8 \
  --skip-existing

python scripts/add_libero_osc_target_pose.py \
  --workers 8 \
  --verify
```

### 18.2 已生成的 stats

文件：

```text
data/libero_mujoco3.3.2/
  dataset_stats.json
  libero_rothko_region_symmetric_q99p95_h16_224x448.pt
  libero_rothko_region_symmetric_q99p95_h16_224x448.json
```

Rothko translation Q99.95 对称范围：

```text
x = +/-0.20725 m
y = +/-0.20486 m
z = +/-0.22343 m
```

实际观测到的最大绝对 relative translation：

```text
x = 0.2431555 m
y = 0.2578584 m
z = 0.2646427 m
```

生成命令：

```bash
python scripts/compute_libero_rothko_norm_stats.py --workers 8
```

### 18.3 已完成的代码

新增：

```text
src/fastwam/representations/libero_osc.py
src/fastwam/representations/libero_rothko.py
src/fastwam/datasets/libero_rgb.py
scripts/add_libero_osc_target_pose.py
scripts/compute_libero_rothko_norm_stats.py
configs/data/libero_rothko_2cam224.yaml
configs/model/fastwam_video_only_libero_rothko.yaml
configs/task/libero_rothko_2cam224_lora_r64_attn_ffn_1e-4.yaml
tests/test_libero_osc.py
tests/test_libero_rothko.py
tests/test_libero_rgb.py
tests/test_visual_action_representation_metadata.py
```

已实现：

1. expert delta action 与 absolute OSC target 的双向转换；
2. 接近 pi 的 absolute rotation 数值稳定处理；
3. Panda 双 finger qpos 到 `[0,1]` gripper-open 的转换；
4. `224 x 224` 单臂 Rothko 横向复制到 `224 x 448`；
5. 训练/评测共用 LIBERO RGB canvas builder；
6. dataset 显式区分 `rothko` 和 `libero_rothko`；
7. video-only model 根据 representation 实例化相应 codec；
8. checkpoint 保存 representation metadata，并拒绝 LIBERO/RoboTwin 交叉加载；
9. Trainer 去除固定 14D pose/2D gripper 的验证硬编码；
10. loss 同时记录 raw、weighted 和 total；
11. LIBERO 推理从 predicted absolute target 做逐步真实 EE feedback 控制；
12. 可选保存 predicted RGB、predicted Rothko 和 control trace；
13. 原标准 LIBERO action-expert 推理分支保持原行为；
14. RoboTwin dataset/config/eval 文件没有为本功能修改。

### 18.4 已通过的验证

CPU 单元测试：

```text
8/8 passed
```

覆盖：

```text
OSC roundtrip
接近 pi rotation
gripper mapping
single-arm Rothko geometry/gripper roundtrip
quaternion sign equivalence
NumPy/Tensor RGB preprocessing equivalence
checkpoint representation isolation
legacy RoboTwin checkpoint compatibility
```

真实 dataset smoke sample：

```text
dataset length       = 277713
video                = [3,17,224,448]
raymap               = [3,17,224,448]
action               = [16,7]
current_endpose      = [7]
future_endpose       = [16,7]
future_gripper       = [16,1]
```

GPU 4 单 batch/单 step smoke：

```text
original Wan2.2 video DiT loaded
original Wan2.2 VAE loaded
LoRA modules = 300
LoRA parameters = 161218560
total parameters = 5865694940
trainable ratio = 2.748499%
forward/backward/optimizer/checkpoint passed
initial single-GPU allocated memory about 12.5 GiB
```

测试产生的 `/tmp/fastwam_libero_rothko_smoke`（约 14 GiB）已经删除。

### 18.5 正式训练命令

仅使用允许的物理 GPU 4、5、6、7：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
MASTER_PORT=29617 \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
bash scripts/train_zero1.sh 4 \
  task=libero_rothko_2cam224_lora_r64_attn_ffn_1e-4
```

默认有效 global batch：

```text
8 per GPU x 4 GPUs x gradient accumulation 4 = 128
```

### 18.6 待正式运行后确认

以下项目需要训练 checkpoint 后完成，不应在当前阶段伪报为已验证：

1. original Wan VAE 对真实 Rothko 的单独 roundtrip 量化报告；
2. 完整 `model.infer()` rollout smoke；
3. LoRA checkpoint 的中断恢复实跑；
4. LIBERO 仿真中 predicted target feedback controller 的 episode smoke；
5. manager 对四个 suite、40 个任务的完整调度；
6. 正式 success rate、clipping count 和 control trace 分析；
7. 与标准 FastWAM LIBERO baseline 的对比。

## 19. 2026-08-01 LIBERO 代码审计与修复

保留训练 `512 -> 224`、仿真 `256 -> 224` 的 RGB 源分辨率差异，不对此做改动。

本轮完成：

1. 为 h32 生成独立的
   `libero_rothko_region_symmetric_q99p95_h32_224x448.pt/.json`，配置不再复用 h16 stats；
2. codec 校验 stats 的 environment、representation、layout、图像/单 tile 尺寸、
   horizon、pixel frames、Rothko 几何和边界参数，防止错误 stats 静默加载；
3. LIBERO dataset 使用 `sample_error_mode: raise`，损坏或缺失样本直接暴露；
   共享 dataset 默认仍为 `fallback`，因此不改变 RoboTwin 现有行为；
4. 评测严格检查 `num_trials`、`action_horizon`、`replan_steps` 和视频采样比例；
5. 支持 `num_trials` 超过 LIBERO 原始 init-state 数量时循环复用 tensor state；
6. rollout 视频补存最后一次 action 后的环境画面，predicted Rothko 视频只保存实际执行长度；
7. manager 使用每次运行独立的 tmux session，不再删除其他评测；
8. Ctrl-C、worker 失败和正常退出都会清理本次 session；
9. 每个 worker 默认设置 21600 秒 watchdog，可通过
   `MULTIRUN.worker_timeout_seconds` 调整；
10. 后续调度每轮可填满全部空闲 GPU slot；
11. worker 命令路径和参数做 shell quoting，并固定使用 manager 当前 Python 解释器；
12. validation 新增米制 position MAE/RMSE、quaternion sign-invariant rotation
    geodesic degree、gripper MAE/accuracy，不再只依赖混合单位的 aggregate L1/L2；
13. 清除 checkpoint loader 中 return 之后永远不可达的 legacy 分支；
14. `eval_libero_single.py` 同时支持直接执行与 Python module import。

h32 Q99.95 translation bounds：

```text
x = 0.31388 m
y = 0.35864 m
z = 0.34010 m
```

本轮验证：12 个单元测试全部通过，h16/h32 Hydra 配置与对应 stats 严格校验通过，
Python compile、shell syntax 和 `git diff --check` 通过。

## 20. 16 步 full fine-tuning 对照配置

新增：

```text
configs/task/libero_rothko_2cam224_full_1e-4.yaml
```

该配置从 original Wan2.2 开始全量微调 video DiT，VAE、text encoder 继续冻结；
不启用 LoRA。为与 h16 LoRA 实验保持相同的有效 global batch，四卡配置为：

```text
per-GPU batch = 4
gradient accumulation = 4
effective global batch = 4 x 4 x 4 = 64
epochs = 10
learning rate = 1e-4
```

为与原始 FastWAM 训练范式保持一致，默认使用 ZeRO-1。若全量 DiT 梯度配合
`batch_size=4` 出现 OOM，再将启动脚本切换为 ZeRO-2，以额外分片 gradients；
不应通过改变 action/RGB/Rothko 时序来省显存。
