# FastWAM DiT 预测 Latent 的 VAE Decoder 适配记录

更新日期：2026-09-06

## 1. 动机

已有 Rothko VAE decoder 只使用干净的 VAE encoder latent 训练：

```text
GT Rothko -> frozen VAE encoder -> clean latent -> trainable conv2 + decoder -> GT Rothko
```

实际 rollout 中，decoder 接收的是 FastWAM DiT 采样得到的 predicted latent。即使 predicted latent 表达的动作与专家动作基本一致，它仍可能和 clean latent 存在分布差异。此次实验的目标是让 decoder 适应这种真实推理输入，而不是让 decoder 修正 DiT 的动作决策。

最终确定的监督方式是：

```text
DiT predicted latent -> expert GT Rothko
clean latent         -> expert GT Rothko（较小权重的保持项）
```

候选样本在进入训练集前必须满足严格的动作一致性筛选。因此，这里的 expert GT Rothko 是与 predicted latent 表达的动作足够接近的监督信号。此前讨论过的“把 predicted latent 解码出的动作重新绘制成 canonical Rothko”没有作为最终训练目标。

## 2. 固定的模型、数据与几何配置

### 2.1 生成 predicted latent 的 FastWAM

- 模型：Wan2.2 5B、LIBERO 四套件、center_frac=0.5。
- 训练运行：`runs/libero_all4_rothko_2cam224_full_1e-4/2026-08-04_15-32-37`
- checkpoint：`checkpoints/weights/step_021700.pt`
- checkpoint SHA256：`6aa1b182d2c8db41c9d529c7c55218ebf317243f205a22c19d64fc7523fe16e7`
- diffusion inference steps：20。

### 2.2 VAE 初始化

- 初始化权重：`runs/libero_rothko_vae_decoder_all4_h16_bs2_ga8_lr1e-5_ep2_rerun_fixed/Wan2.2_VAE_libero_rothko_step007498.safetensors`
- 训练部分：VAE `conv2 + decoder`。
- VAE encoder 保持冻结。

### 2.3 Rothko 配置

- stats：`data/libero_mujoco3.3.2/libero_rothko_region_symmetric_q99p95_h16_224x448.pt`
- 表示：单臂 Rothko 横向复制，输出 224x448。
- horizon：16 个未来动作。
- `center_frac=0.5`
- `focal=0.2`
- `center_scale=1.0`
- `dir_scale=1.0`
- `boundary_margin=8`
- `outer_margin=8`
- quaternion 顺序：wxyz。
- gripper：外侧边界编码为 `2 * g - 1`。

## 3. 严格筛选的 predicted-latent 缓存

缓存位置：

```text
data/libero_predicted_ray_latent_adaptive/
  wan22_5b_step021700_strict50_seed20260908/merged
```

缓存覆盖四个 LIBERO suites、共 40 个任务，每个任务严格保留 50 条不同窗口：

- 总记录数：2,000。
- 任务数：40。
- 每任务：50。
- 唯一窗口：2,000。
- 唯一 diffusion seed：2,000。
- 生成候选总数：5,181（首次 5,145，补齐 36）。
- 总体接受率：约 38.6%。
- 首次生成得到 1,997 条；`libero_spatial task0` 缺 3 条，随后用不重复窗口补齐到 50 条。

严格筛选条件针对完整 16 帧动作序列：

```text
position mean <= 0.010 m
position max  <= 0.020 m
rotation mean <= 1.5 degrees
rotation max  <= 3.0 degrees
all gripper frames correct
```

这些条件的含义是：只让“DiT predicted latent 解码后的动作”和专家动作已经高度一致的样本参与 decoder 适配，避免 decoder 被要求把本质上错误的 DiT 动作强行变成专家动作。

相关脚本：

- `scripts/cache_dit_predicted_ray_latents.py`
- `scripts/cache_dit_predicted_ray_latents_adaptive.py`
- `scripts/analyze_dit_predicted_ray_latents.py`
- `scripts/finetune_rothko_vae_from_dit_latents.py`
- `scripts/run_dit_latent_vae_adaptation_4gpu.sh`

`cache_dit_predicted_ray_latents_adaptive.py` 已支持非破坏性的 `--resume-from` 补齐：保留已有合格记录，跳过已经尝试过的窗口，并校验 checkpoint、数据和筛选阈值身份。

## 4. 8-GPU smoke/pilot 实验

该运行已完成记录，但其权重目录在记录完成后清理，不作为正式产物保留。

### 4.1 数据划分与训练参数

- 总数据：2,000。
- 固定验证：每任务 5 条，共 200 条。
- 训练：每任务 45 条，共 1,800 条。
- world size：8。
- 每卡 batch：1。
- 有效全局 batch：8。
- epoch：1。
- optimizer steps：225。
- 学习率：1e-6。
- weight decay：0。
- gradient clipping：1.0。
- predicted-latent loss weight：1.0。
- clean-latent loss weight：0.25。
- trainable parameters：555,051,580（约 555M）。
- 单卡显存：约 9.2GB。
- 实际训练约 8.5 分钟，包含加载和验证的完整运行约 12 分钟。

### 4.2 固定验证集的图像空间结果

| 指标 | 训练前 | 训练后 | 变化 |
|---|---:|---:|---:|
| predicted total loss | 0.0182873 | 0.0182534 | -0.185% |
| predicted center MAE | 0.0124583 | 0.0124214 | -0.296% |
| predicted direction MAE | 0.00578305 | 0.00578376 | 基本不变 |
| clean total loss | 0.0024020 | 0.0029721 | +23.7% |

clean 指标的相对退化明显，但其绝对误差仍小；clean reconstruction 只应作为保持项和回归监控，不是最终优化判据。最终目标是降低 rollout 时 predicted latent 的动作解码误差。

### 4.3 同一固定 200 条验证集的动作空间结果

使用 legacy Rothko decoder，对训练前后的 VAE 做成对比较：

| predicted-latent 指标 | 训练前 | 训练后 | 变化 |
|---|---:|---:|---:|
| position mean | 5.640354 mm | 5.621815 mm | -0.01854 mm（-0.329%） |
| position max | 9.779564 mm | 9.734778 mm | -0.04479 mm（-0.458%） |
| rotation mean | 0.872255 deg | 0.872432 deg | +0.000177 deg（+0.020%） |
| rotation max | 1.509629 deg | 1.502841 deg | -0.006788 deg（-0.450%） |
| gripper accuracy | 100% | 100% | 不变 |

成对样本中：

- position mean 改善：125/200。
- rotation mean 改善：95/200。

bootstrap 95% 置信区间：

- position mean delta：-0.01854 mm，CI `[-0.04307, +0.00628]`，穿过 0。
- position max delta：-0.04479 mm，CI `[-0.08656, -0.00441]`。
- rotation mean delta：+0.000177 deg，CI `[-0.003245, +0.003738]`，穿过 0。
- rotation max delta：-0.006788 deg，CI `[-0.01236, -0.001125]`。

clean-latent 的绝对动作误差由约 0.409 mm / 0.063 deg 增加到约 0.599 mm / 0.095 deg，仍处于很小的绝对量级。

## 5. 当前结论

1. 数据生成、严格筛选、固定 train/val 划分、8-GPU decoder 微调和动作空间对比链路已经跑通。
2. 225 optimizer steps、1e-6 学习率不足以让 555M 可训练参数充分适配；该结果是 smoke test，不是方法性能上限。
3. predicted position max 和 rotation max 有小幅、统计上较稳定的改善；position mean 和 rotation mean 的变化还不足以排除随机波动。
4. 当前没有运行新 VAE 的完整 LIBERO rollout，因此不能仅根据离线误差断言成功率提升。
5. 当前最大的限制是状态覆盖量。训练集只有 1,800 条，即每个任务 45 条窗口。

## 6. 与原始完整 VAE decoder 微调的数据量对比

原始 LIBERO 四套件 VAE decoder 微调使用：

- 每 epoch：239,929 个训练窗口。
- 共 2 epochs：累计 479,858 次样本访问。
- 有效 batch：64。
- 每 epoch：3,749 optimizer steps。
- 总计：7,498 optimizer steps。

当前 pilot 的 1,800 条训练样本仅相当于原始单 epoch 数据量的约 0.75%。即使反复训练更多 epoch，也无法替代 predicted-latent 状态多样性。

## 7. 下一阶段候选规模

当前尚未决定正式规模。已讨论两个候选：

### 方案 A：每任务 200 条

- 总数据：8,000。
- 固定验证：每任务 20 条，共 800。
- 训练：每任务 180 条，共 7,200。
- 3 epochs、全局 batch 8 时：每 epoch 900 steps，总计 2,700 steps。
- 相当于原始 VAE decoder 单 epoch 数据量的约 3.0%（按训练部分计）。

### 方案 B：每任务 500 条

- 总数据：20,000。
- 若每任务保留 50 条验证：验证 2,000，训练 18,000。
- 数据覆盖更合理，但严格样本的生成耗时和推理成本显著增加。

正式实验前需要共同确定：每任务目标数、固定验证比例、训练 epoch、学习率，以及是否将 action-space validation 直接集成进每个 epoch 的验证与 checkpoint 选择。

## 8. 不应删除或覆盖的资产

- 2,000 条严格筛选 merged cache：它是后续扩容和代码验证的基础。
- 原始正式 VAE：`...rerun_fixed/Wan2.2_VAE_libero_rothko_step007498.safetensors`。
- FastWAM step 21700 checkpoint。
- 上述四个数据生成/分析/训练脚本及运行辅助脚本。

已经清理的仅是 smoke/pilot 权重与临时评测产物；所有关键数值已记录在本文档中。

## 9. 本次曾实现的代码改动与回退记录

以下改动均属于本次“使用 DiT predicted latent 微调 VAE decoder”方案，曾在提交
`c5f8e51` 之后的工作区中实现，但没有提交到 Git：

### 9.1 Dataset 输出当前夹爪状态

文件：`src/fastwam/datasets/lerobot/robot_video_dataset.py`

- 在 raymap 数据路径中增加 `current_gripper`。
- RoboTwin 从当前 qpos 中读取双臂夹爪值。
- LIBERO 从当前 `gripper_open` 中读取单臂夹爪值。
- 将该字段随 sample 返回，用于把当前状态和 16 步未来专家动作重新绘制为完整的 17 帧 Rothko target。

### 9.2 推理直接导出 DiT predicted raymap latent

文件：`src/fastwam/models/wan22/fastwam_visual_action.py`

在 `FastWAMVideoOnlyRaymap.infer()` 中曾增加：

```text
decode_raymap: bool = True
return_raymap_latents: bool = False
```

功能包括：

- `return_raymap_latents=true` 时直接返回 DiT 采样得到的 raymap latent。
- `decode_raymap=false` 时跳过无人使用的 VAE pixel decode。
- 如果传入 `current_endpose` 并要求控制 action，则禁止关闭 raymap decode，因为控制输出仍依赖 Rothko 像素解码。

这样生成适配数据时可以直接保存真实推理 latent，避免
`predicted latent -> pixels -> re-encode latent` 带来的额外误差。

### 9.3 单元测试

文件：`tests/test_fastwam_train_only_rgb_aux.py`

曾新增测试以验证：

- 不解码未来 RGB；
- 不解码 Rothko 像素；
- 直接取得形状为 `[B, 16, T_latent, H_latent, W_latent]` 的 raymap latent；
- 上述模式不会调用 VAE decoder。

该测试在 `fastwam_libero` 环境中通过；当时目标文件共 `7 passed`。

### 9.4 新增实验脚本

曾新增但未提交以下脚本：

- `scripts/cache_dit_predicted_ray_latents.py`
  - 任务均衡地抽取 LIBERO 窗口；
  - 运行 FastWAM diffusion sampling；
  - 保存 predicted raymap latent、clean latent、专家 pose/gripper 和数据身份信息。
- `scripts/cache_dit_predicted_ray_latents_adaptive.py`
  - 对 decoded predicted latent 计算平移、旋转与夹爪误差；
  - 使用严格阈值筛选；
  - 按任务自适应补齐合格记录；
  - 支持 `--resume-from` 非破坏性补齐；
  - 合并并验证多 worker cache 的任务数、窗口唯一性、diffusion seed 唯一性与配置身份。
- `scripts/analyze_dit_predicted_ray_latents.py`
  - 对指定 VAE 计算 predicted/clean latent 的动作空间解码误差；
  - 支持固定的 task-stratified validation split。
- `scripts/finetune_rothko_vae_from_dit_latents.py`
  - DDP 微调 VAE `conv2 + decoder`；
  - 主监督为 `predicted latent -> expert GT Rothko`；
  - 辅助保持项为 `clean latent -> expert GT Rothko`；
  - 对缓存、checkpoint、stats、筛选报告和数据形状进行一致性检查。
- `scripts/run_dit_latent_vae_adaptation_4gpu.sh`
  - 早期四卡 cache 生成、merge 和单 epoch pilot 的启动脚本。

### 9.5 回退决定

2026-09-06 决定暂停该方案的代码开发，先讨论其他方向。因此：

- 上述 3 个已跟踪源码/测试文件恢复至提交 `c5f8e51` 的状态；
- 上述 5 个未跟踪实验脚本从工作区删除；
- smoke/pilot 运行目录及权重已删除；
- 本文档保留，作为方案、实验数值与未来可能重新实现时的设计记录；
- 已生成的 2,000 条严格筛选 merged cache 暂时保留，不属于 Git 工作区源码改动。
