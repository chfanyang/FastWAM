# FastWAM RoboTwin EE 预测—执行偏差与自碰撞问题排查记录

## 1. 文档目的

本文记录 `click_alarmclock` 单任务模型在 RoboTwin 评测中出现右臂异常翘起问题的完整排查过程，重点回答以下问题：

1. 模型预测的 EE 轨迹是否在第 30 步发生了突变；
2. 模型目标 EE 与机器人实际到达 EE 为什么开始明显不一致；
3. 问题来自模型预测、IK、轨迹规划还是物理仿真；
4. 专家轨迹是否会经过相同区域；
5. 当前 CuRobo 碰撞配置修正到了什么程度；
6. 还有哪些问题尚未解决。

本文对应的主要失败样例为：

- 任务：`click_alarmclock`
- 环境配置：`demo_clean`
- seed：`4300008`
- checkpoint：
  `robotwin_click_alarmclock_rothko_3cam_384_lora_r64_attn_ffn_1e-4`
  的 `step_010000.pt`
- 推理方式：每次预测 16 步，连续执行 16 步，`replan_steps=16`

排查期间没有改变 FastWAM 的“预测 16 步、执行 16 步”运行机制。

## 2. 最初观察到的现象

在失败视频中，右臂在约第 30～31 个环境动作附近突然向一个异常方向翘起。此后：

- 当前 RGB 观测发生明显变化；
- 后续模型预测开始持续偏离；
- 后续 replan 的第一帧与上一个 replan 预测视频的最后一帧看起来不一致；
- 最终 episode 失败。

一开始存在三种主要猜测：

1. 模型在 action 30 附近直接预测了一个不连续的 EE 目标；
2. 模型目标是合理的，但 CuRobo IK 求出了错误的关节解；
3. IK 和关节目标本身合理，但机器人执行时受到碰撞或动力学影响，没有到达规划位置。

仅凭评测 MP4 无法区分这三种情况，因此增加了逐动作诊断记录。

## 3. 建立逐步 EE 执行诊断

评测诊断会在同一份 CSV 中保存每个环境动作的以下信息：

- 模型给出的左右臂目标 EE：
  - xyz；
  - quaternion，顺序为 wxyz；
  - gripper；
- 执行前实际 EE；
- 执行后实际 EE；
- 执行后实际 EE 相对模型目标的：
  - 位置误差；
  - 旋转误差；
- CuRobo 规划状态；
- 执行前关节位置；
- CuRobo 规划轨迹的终点关节位置；
- 执行后的实际关节位置；
- drive target；
- 关节速度；
- SAPIEN 接触对、接触冲量和接触发生的物理子步；
- 对应帧图像路径。

主要原始记录：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_seed4300008_velocity_trace_step100_20260730/click_alarmclock/predictions/demo_clean/episode_000/ee_execution.csv`

带接触冲量的记录：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_seed4300008_contact_trace_step100_20260730/click_alarmclock/predictions/demo_clean/episode_000/ee_execution.csv`

## 4. 模型目标 EE 与实际执行 EE 的对比

### 4.1 第 28～29 步仍然正常

原始故障轨迹中，右臂执行后相对模型目标的误差为：

| 环境动作 | CuRobo 状态 | 位置误差 | 旋转误差 |
|---:|---|---:|---:|
| 28 | Success | 约 1.7 mm | 约 0.25° |
| 29 | Success | 约 1.8 mm | 约 0.27° |

这说明在故障发生前：

- 模型目标可以被 CuRobo 正常规划；
- SAPIEN 中的机器人基本可以准确到达规划目标；
- EE 预测到执行的转换链路整体正常。

### 4.2 第 30 步开始出现明显偏差

原始故障轨迹中：

| 环境动作 | CuRobo 状态 | 位置误差 | 旋转误差 |
|---:|---|---:|---:|
| 30 | Success | 约 7～11 mm | 约 2～3.3° |
| 31 | Success | 约 28～33 mm | 约 9～11° |
| 32 | Success | 约 43～44 mm | 约 18～20° |
| 33 | IK Fail | 约 14 mm | 约 3° |

不同诊断回放中的绝对数值会因物理求解的细微差异略有变化，但转折点始终是第 30 步。

关键点是：

- 第 30～32 步 CuRobo 返回的仍然是 `Success`；
- 规划终点关节位置是平滑的；
- drive target 也正确设置成了 CuRobo 的规划终点；
- 但实际关节位置开始明显偏离 drive target；
- 实际 EE 因而开始明显偏离模型目标。

所以这不是“控制代码没有发送关节目标”，也不是模型 EE 在第 30 步突然不连续。

### 4.3 模型在第二个 action chunk 内的目标是连续的

当前 `replan_steps=16`：

- action 1～16：第一个预测 chunk；
- action 17～32：第二个预测 chunk；
- action 33 开始：第三次预测。

故障点对应：

- action 30：第二个 chunk 的第 14 个动作；
- action 31：第二个 chunk 的第 15 个动作；
- action 32：第二个 chunk 的第 16 个动作；
- action 33：发生异常观测后的第一次新预测。

第二个 chunk 内，模型给出的右臂 EE 位置和旋转是连续变化的。真正显著的模型目标变化发生在 action 33，即机器人已经被物理碰撞推离原轨迹、模型拿到了异常新观测之后。

因此不能用 action 33 之后的错误反推 action 30 是模型输出跳变。

## 5. 关节目标、实际关节与接触冲量

### 5.1 第 30 步规划目标仍然平滑

一条原始轨迹中，第 29、30 步右臂规划终点关节位置约为：

- action 29：
  `[0.619, 1.994, 2.271, -1.798, -0.009, 0.641]`
- action 30：
  `[0.633, 2.044, 2.334, -1.842, -0.012, 0.652]`

它们是连续变化的，不存在某个关节突然跨越很大的角度。

### 5.2 实际关节从第 30 步开始被推离 drive target

action 30 之后，实际关节 2、3 相对规划目标开始快速增大；到 action 31～32，实际关节与 drive target 的偏差已经非常明显。

这说明异常运动不是 CuRobo 直接命令出来的，而是物理仿真中的外力或碰撞约束改变了实际机器人状态。

### 5.3 接触冲量明确定位到物理自碰撞

接触诊断发现异常接触对为：

`fr_link5 <-> fr_link3`

这里的 `fr_*` 是 Aloha-AgileX 右臂 link 命名，不是 Franka 机器人。

接触冲量在 action 29 前为 0，从 action 30 开始显著出现：

| 环境动作 | 最大接触冲量 | 累计接触冲量 |
|---:|---:|---:|
| 28 | 0 | 0 |
| 29 | 0 | 0 |
| 30 | 约 3.47 | 约 120.27 |
| 31 | 约 6.08 | 约 378.15 |
| 32 | 约 5.44 | 约 760.86 |

因此，右臂“突然翘起”的直接原因已经确定：

> 右腕附近的 `fr_link5` 与上臂/前臂结构 `fr_link3` 发生物理自碰撞，SAPIEN 的碰撞响应把实际关节推离了 CuRobo 规划轨迹。

这解释了为什么：

- 模型目标平滑；
- CuRobo 规划关节平滑；
- 实际机械臂仍然发生大幅异常运动。

## 6. 为什么 CuRobo 原先没有阻止碰撞

CuRobo 不直接使用 SAPIEN 的完整碰撞 mesh 做实时规划，而是使用
`collision_aloha_right.yml` 中配置的碰撞球近似机械臂几何。

原始右臂碰撞球存在明显漏建模：

- `fr_link3` 只有一排较稀疏的球；
- `fr_link5` 只有一个球；
- 对 mesh 表面采样后发现，大量几何表面没有被碰撞球覆盖；
- 尤其缺少此次发生接触的腕部和 link3 之间的关键区域。

所以原来的 CuRobo 判断该关节解无碰撞，而 SAPIEN 使用更完整的碰撞几何执行时发生了真实接触。

这不是 IK 的正向运动学计算错误，而是：

> IK 解能满足模型目标 EE，但 CuRobo 使用的简化碰撞模型没有识别该解会在 SAPIEN 中自碰撞。

## 7. 与专家轨迹的对比

专家轨迹文件：

`click_alarmclock_seed4300008.npz`

其中：

- `joints`：252 × 14；
- `endpose`：252 × 14；
- 左臂全程基本静止；
- 右臂完成接近、按压和返回。

### 7.1 模型不是从一开始就完全偏离专家

将模型每个右臂目标 EE 与专家轨迹中的右臂 EE 做最近邻比较：

- action 1 与专家初始阶段非常接近；
- action 16～29 整体沿着与专家相似的接近方向运动；
- action 30 的模型目标与专家轨迹中最近的 EE：
  - 位置只差约 4.5 mm；
  - 旋转只差约 3.9°。

所以 action 30 不是一个完全随机或远离任务目标的模型输出。

### 7.2 小的姿态差异在工作空间边界会放大为自碰撞差异

action 30 的模型目标约为：

`[0.0815, -0.1299, 1.1449, 0.4964, -0.4929, 0.5040, 0.5066]`

专家轨迹中一个邻近的安全 EE 约为：

`[0.0821, -0.1342, 1.1440, 0.5039, -0.4697, 0.4960, 0.5286]`

对应关节也存在差异：

- 模型/CuRobo action 30 规划解约为：
  `[0.633, 2.044, 2.334, -1.842, -0.012, 0.652]`
- 邻近专家姿态的关节约为：
  `[0.641, 1.981, 2.233, -1.738, 0.002, 0.660]`

虽然 EE 的位置和旋转差异不大，但模型目标要求右臂进一步内折，尤其使中间关节进入更危险的组合。

这表明当前问题不是简单的二选一：

- 不能说完全是“模型预测错了”，因为模型轨迹连续且接近专家轨迹；
- 也不能说完全是“IK 随机解错了”，因为模型的精确目标确实比专家安全姿态更靠近自碰撞边界；
- 原 CuRobo 碰撞球漏检，使这个边缘目标被错误地当成了安全目标。

更准确的结论是：

> 模型预测进入了 Aloha-AgileX 的狭窄自碰撞边界；原 CuRobo 碰撞近似又没有覆盖该区域，最终允许了一个在 SAPIEN 中会发生物理自碰撞的 IK 解。

## 8. 碰撞球修正

当前只修改 FastWAM 内复制的 RoboTwin assets，不修改原始
`/mnt/hwdata/cfy/RoboTwin/assets`。

主要文件：

`third_party/RoboTwin/assets/embodiments/aloha-agilex/collision_aloha_right.yml`

修正内容：

- 为 `fr_link3` 主体增加三排碰撞球；
- 为 `fr_link3` 近端凸起增加覆盖；
- 将 `fr_link5` 从一个碰撞球扩展为覆盖腕部主体和凸起的多球模型；
- 删除实验中被证明过度保守的 `fr_link3` 远端 cap 球；
- `fr_link3` 和 `fr_link5` 的额外 self-collision buffer 最终保持为 0。

规划配置：

`third_party/RoboTwin/assets/embodiments/aloha-agilex/curobo_right.yml`

当前测试值：

- `num_ik_seeds: 32`
- `num_trajopt_seeds: 4`
- `num_graph_seeds: 4`
- `max_attempts: 10`

## 9. 修正后的固定轨迹回放结果

为避免正常评测中的专家检查自动跳过 seed，使用：

- `setup_demo(seed=4300008)`；
- 原失败记录中保存的完全相同 EE action；
- 同一个场景；
- 同样逐动作执行；
- 不重新调用模型。

这样可以隔离验证碰撞配置本身。

主要结果：

| 环境动作 | 修正后规划状态 | 位置误差 | 旋转误差 | 物理碰撞冲量 |
|---:|---|---:|---:|---:|
| 28 | Success | 约 1.7 mm | 约 0.26° | 0 |
| 29 | Success | 约 1.8 mm | 约 0.27° | 0 |
| 30 | IK Fail | 约 10.7 mm | 约 2.0° | 0 |
| 31 | IK Fail | 约 16.9 mm | 约 3.3° | 0 |
| 32 | IK Fail | 约 20.8 mm | 约 4.3° | 0 |

修正后：

- action 1～29 仍可正常规划和执行；
- action 30 的危险目标在 IK 阶段被拒绝；
- 不再出现 `fr_link5 <-> fr_link3` 的非零碰撞冲量；
- 不再出现右臂被物理碰撞突然弹开的现象。

对应记录：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_saved_right_trace_collision_spheres_v3_20260730/ee_trace_replay.csv`

## 10. 碰撞余量分析

用修正后的碰撞球对原失败轨迹的规划终点关节进行离线分析：

| 环境动作 | `fr_link3`—`fr_link5` 近似余量 |
|---:|---:|
| 20 | +6.79 mm |
| 21 | +6.24 mm |
| 22 | +5.84 mm |
| 23 | +5.35 mm |
| 24 | +4.99 mm |
| 25 | +4.43 mm |
| 26 | +4.03 mm |
| 27 | +3.63 mm |
| 28 | +3.21 mm |
| 29 | +0.41 mm |
| 30 | -1.62 mm |
| 31 | -3.22 mm |
| 32 | -4.67 mm |

正值表示碰撞球仍有间隙，负值表示发生穿透。

这个结果与 SAPIEN 接触冲量从 action 30 开始出现完全一致，说明当前碰撞球边界没有在很早的位置误杀正常轨迹。

### 10.1 当前碰撞球对专家轨迹的验证

将 `click_alarmclock_seed4300008.npz` 中全部 252 帧专家右臂关节轨迹送入当前 CuRobo 碰撞模型，结果为：

- 总帧数：252；
- 任意 self-collision pair 出现负余量的帧数：0；
- `fr_link3–fr_link5` 出现负余量的帧数：0；
- 全轨迹最小余量：约 `+0.382 mm`；
- 最危险帧：专家 frame 248；
- 最危险接触对：`fr_link3 <-> fr_link5`。

专家轨迹后段的 `fr_link3–fr_link5` 余量连续下降：

- frame 241：约 `+1.597 mm`；
- frame 244：约 `+0.796 mm`；
- frame 246：约 `+0.482 mm`；
- frame 248～251：约 `+0.382 mm`。

因此当前碰撞球没有误杀 seed 4300008 的专家轨迹，同时能够拒绝原模型 action 30 的约 `-1.62 mm` 穿透。这个结果支持保留当前碰撞边界，并将后续重点转向模型目标 EE 与专家安全 EE 之间的少量位置/旋转偏差。

逐帧结果：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/expert_seed4300008_collision_clearance_20260730/right_arm_clearance.csv`

### 10.2 action 30 的位置/旋转交叉消融

在同一 seed 4300008 场景中使用原模型动作回放至 action 29，然后保持完全相同的当前关节状态，对 action 30 分别规划四种目标：

| action 30 目标 | 规划结果 | 最小碰撞余量 |
|---|---|---:|
| 模型 xyz + 模型旋转 | IK Fail | 无解 |
| 模型 xyz + 专家旋转 | Success | `+2.481 mm` |
| 专家 xyz + 模型旋转 | IK Fail | 无解 |
| 专家 xyz + 专家旋转 | Success | `+3.183 mm` |

这里使用的专家目标是专家 frame 237，它是专家轨迹中与模型 action 30 目标接近的安全 EE：

- 模型与专家 xyz 差异：约 `4.478 mm`；
- 模型与专家旋转差异：约 `3.872°`。

消融结果表明：

- 保持模型 xyz 不变，只把旋转换成专家旋转，IK 立即在第一次尝试成功；
- 保持模型旋转不变，只把 xyz 换成专家 xyz，10 次尝试后仍然 `IK Fail`；
- 成功的“模型 xyz + 专家旋转”解仍有 `+2.481 mm` 自碰撞余量。

因此 action 30 越过自碰撞边界的主要因素已经定位为模型的末端旋转误差，而不是约 4.5 mm 的位置误差。后续训练修正应优先提高 Rothko 对旋转的表达和监督强度。

完整结果：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_pose_ablation_20260730/results.json`

### 10.3 action 30 的旋转可行性阈值

固定模型 action 30 的 xyz，只将四元数从模型旋转沿最短路径逐步插值到专家旋转：

| 向专家方向修正的角度 | 相对专家剩余旋转误差 | 规划结果 |
|---:|---:|---|
| 0.000° | 3.872° | IK Fail |
| 0.387° | 3.485° | IK Fail |
| 0.774° | 3.098° | IK Fail |
| 1.162° | 2.710° | IK Fail |
| 1.549° | 2.323° | Success |
| 1.936° | 1.936° | Success |
| 2.323° | 1.549° | Success |
| 2.710° | 1.162° | Success |
| 3.098° | 0.774° | Success |
| 3.485° | 0.387° | Success |
| 3.872° | 0.000° | Success |

当前离散扫描将该场景的可行性边界定位在约 `1.162°～1.549°` 的旋转修正之间。也就是说，模型不需要完全复制专家旋转；只需要将当前旋转预测向安全方向改善约 1.2～1.6°，就可以在相同 xyz 下找到无碰撞 IK。

完整扫描结果：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_pose_ablation_20260730/results_with_rotation_sweep.json`

## 11. 增加规划 seed 是否能解决

已经测试：

- `num_trajopt_seeds=4`；
- `num_graph_seeds=4`；
- 将 `num_ik_seeds` 临时增加到 128。

action 30 仍然为：

- `status=IK Fail`
- `valid_query=True`
- `attempts=10`
- `trajopt_time=0`

这说明失败发生在 IK 阶段，还没有进入 TrajOpt：

- 增加 TrajOpt seed 无法解决；
- 增加 Graph seed 无法解决；
- 128 个 IK seed 仍未找到满足该精确 EE 目标的无碰撞解。

当前没有证据表明只靠更多随机初始化就能为 action 30 找到安全 IK 分支。

## 12. 当前已证实结论

1. action 30 前模型目标、规划目标和实际执行基本一致。
2. 第二个 action chunk 内模型 EE 预测连续，没有在 action 30 突然跳变。
3. 原 CuRobo 规划终点关节连续，控制代码也正确设置了 drive target。
4. action 30 开始，实际关节被物理自碰撞推离 drive target。
5. 直接碰撞对是 Aloha-AgileX 右臂的 `fr_link5 <-> fr_link3`。
6. 原因之一是原 CuRobo 碰撞球没有覆盖实际发生接触的 mesh 区域。
7. 模型 action 30 的目标接近专家轨迹，但精确位置和方向使右臂比专家姿态更内折。
8. 修正碰撞球后，危险动作在 action 30 被 CuRobo 拒绝，物理弹开消失。
9. 增加到 128 个 IK seed 仍找不到该精确目标的无碰撞解。
10. 当前碰撞球允许 seed 4300008 的全部 252 帧专家轨迹通过，最小余量约为 `+0.382 mm`。
11. action 30 的位置/旋转交叉消融确认，约 `3.872°` 的旋转误差是触发 IK 自碰撞失败的主要因素。
12. 当前修正提供的是“正确识别危险目标”，还没有让任务在该目标下继续成功。

## 13. 尚未解决的问题

### 13.1 任务生成范围是否过于靠近操作边界

`click_alarmclock.py` 当前采样范围为：

- x：`[-0.25, 0.25]`
- y：`[-0.2, 0.0]`
- 排除：`abs(x) < 0.05`
- 绕 y 轴随机旋转：`[0, 3.14]`

seed 4300008 的场景确实使右臂在按压时接近自碰撞边界，但专家轨迹能够完成该场景。因此不能仅凭这一个样例立即认定任务生成范围无效。

后续需要统计：

- 失败 seed 的闹钟位置和姿态；
- 成功 seed 的闹钟位置和姿态；
- 专家轨迹的最小自碰撞余量；
- 模型目标相对专家姿态的旋转偏差；
- 失败是否集中在某个 x/y/朝向区域。

如果失败明显集中在极端区域，再缩小生成范围会比放松碰撞检测更合理。

### 13.2 模型是否需要学习更安全的 EE 姿态

模型目标与专家目标只有几毫米、几度差异，却足以越过自碰撞边界。这说明后续训练或动作表示可能需要更加关注：

- 末端旋转误差；
- 接近工作空间边界时的安全姿态；
- EE 目标映射到关节空间后的可行性；
- 专家姿态附近的局部误差是否被 Rothko loss 充分区分。

### 13.3 规划失败后如何处理

目前保持原 FastWAM 行为，不改变 action chunk 执行逻辑：

- 不会在规划失败后清空剩余动作；
- 不会提前重新观测和 replan；
- 仍保持每次预测 16 步并执行 16 步。

因此当前碰撞修正会让危险动作安全失败，但不保证 episode 恢复成功。

## 14. action 30 旋转误差在 Rothko 与 VAE latent 中有多显著

为了判断约 `3.872°` 的右 EE 旋转误差是否会被当前训练目标充分惩罚，额外做了两层定量检查。整个检查固定：

- 第二个 action chunk 的 frame 0 为执行完全局 action 16 后的实际双臂 EE；
- action 30 对应该 chunk 中 zero-based pixel index 14；
- 左臂、右臂 xyz、夹爪和其他帧完全相同；
- 唯一变量是 action 30 的右 EE quaternion：
  - 一份使用模型预测；
  - 一份使用最近的专家 frame 237 旋转。

### 14.1 Rothko 像素空间

模型和专家旋转相差：

- `3.872050°`

右臂方向区域中的差异为：

- MSE：`0.00090750`
- RMSE：`0.030125`
- MAE：`0.025571`
- `98.31%` 的方向区域数值变化超过 `1e-3`

右臂 center/translation 区域差异严格为零，符合“固定 xyz，只改变 rotation”的测试设计。

归一化没有削弱或裁剪这次旋转差异：

- 当前方向区域 stats 是 `lo=-1, hi=1`，span 恒为 `2`；
- 模型版本和专家旋转版本的 clipping fraction 都是 `0`；
- 因此该误差不是因为 norm stats 截断而丢失。

但是，差异经过空间和时间平均后被明显稀释：

- 单帧完整 `384 x 320` 双臂 Rothko MSE：`0.00034692`
- 再平均到完整 17 帧：约 `0.00002041`

也就是说 Rothko 表示本身能够看见旋转误差，但当前 loss 不知道“右臂末端、接近自碰撞边界、action 30”是高风险区域；它只把这一误差当成整段双臂视频中的局部误差。

### 14.2 原始 Wan2.2 VAE latent 空间

使用当前训练实际采用的原始 VAE：

`checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors`

将上述两段 17 帧 Rothko 分别 VAE encode 后：

- latent shape：`[1, 48, 5, 24, 20]`
- 差异只出现在 latent frame 4；
- latent frame 0、1、2、3 的差异均为零；
- latent frame 4 MSE：`0.00048002`
- 按训练中的 4 个 future latent frame 等权平均：`0.00012000`
- 再乘当前 `lambda_raymap=5`：`0.00060002`

flow matching 的 target 是 `noise - clean_latent`，因此在使用同一 noise 时，两种旋转对应的 target 差异就是 clean latent 差异的相反数。上面的 latent MSE 可以直接反映当前目标函数区分这两种旋转的信号量级。

需要注意：这是“其他 16 帧固定，只改变一个未来帧”的受控局部敏感度测试，不是实际训练 batch 的 loss 分解。VAE 是时空非线性编码器，不能把 `0.00060002` 直接与训练曲线中的 `0.008` 作严格比例解释。它只能证明该旋转误差在 latent 中可见、且集中于最后一个 latent frame。把真实预测 chunk 中单独一帧替换为专家旋转会人为制造时间不连续，因此该替换实验的数值没有作为结论采用。

### 14.3 VAE 自身的 round-trip 精度

为了排除“DiT 已经预测正确，但原始 Wan VAE 把 Rothko 解坏”的可能，使用 seed 4300008 的连续专家 frame 223–239 做了：

`Rothko encode -> Wan VAE encode -> Wan VAE decode -> Rothko decode`

结果：

- 未来 16 帧右 EE 位置误差均值：`0.761 mm`
- 未来 16 帧右 EE 位置误差最大值：`1.531 mm`
- 未来 16 帧右 EE 旋转误差均值：`0.259°`
- 未来 16 帧右 EE 旋转误差最大值：`0.373°`
- 专家 frame 237 的右 EE 旋转误差：`0.317°`

VAE round-trip 的旋转误差明显小于：

- 模型与专家的 `3.872°` 差异；
- 约 `1.2°–1.55°` 的规划可行性跨界修正量。

因此原始 Wan VAE 不是这次失败的主要误差来源。主要问题仍是 DiT 对末端旋转的预测精度，以及当前全图、全时域平均的 latent MSE 没有强调接近运动学/碰撞边界的关键局部误差。

### 14.4 对后续训练修改的含义

当前证据不支持优先做以下改动：

- 继续增加 IK/TrajOpt seed；
- 放松新碰撞球；
- 单纯更换 Rothko norm stats；
- 把主要问题归因于 VAE round-trip。

训练侧更值得优先比较的是：

1. 给 raymap 的后期 latent frame 或关键动作阶段更高权重；
2. 添加能够直接约束解码后 EE rotation 的辅助目标，而不是只依赖全 latent MSE；
3. 对接近自碰撞边界的专家样本进行困难样本加权或过采样；
4. 在不改变 `predict 16 / execute 16` 的前提下，统计更多失败 seed 的“旋转误差—碰撞余量—规划成败”关系，再决定具体阈值。

仅继续增大统一的 `lambda_raymap` 会同时放大平移、旋转、夹爪、双臂和所有时刻的误差，针对性较弱；它可以作为低成本对照实验，但不是当前最有解释力的修复。

## 15. 碰撞参数放宽扫描

为了验证是否能通过继续调整碰撞球，为模型 action 30 的精确 EE 目标找到另一个安全 IK 解，对 `fr_link3 ↔ fr_link5` 的 self-collision 阈值做了运行时扫描。该扫描只修改 CuRobo 内存中的 collision offset，结束后自动恢复，没有永久覆盖资产配置。

首先确认：

- `collision_sphere_buffer: 0.004` 会增加机器人对外界碰撞球的半径；
- CuRobo 会在 self-collision offset 中抵消这部分全局 buffer；
- 因此调整全局 `collision_sphere_buffer` 不会改变 `fr_link3 ↔ fr_link5` 的实际 self-collision 边界；
- 真正有效的是 link-specific self-collision offset 或原始球半径。

将 link3/link5 的总等效阈值从 `0` 逐步放宽到 `6 mm`，结果为：

- `0–1.50 mm`：`IK Fail`
- `1.75 mm`：首次 `Success`
- `1.75–6.00 mm`：均为 `Success`

但所有 Success 都收敛到几乎同一组关节解：

`[0.6334, 2.0442, 2.3335, -1.8423, -0.0115, 0.6520]`

没有出现新的安全 IK 分支。这与该机械臂用 6 个关节约束 6D EE pose 的结构一致：碰撞参数只能允许或拒绝已有运动学解，不能凭空创造新的精确 pose 解。

### 15.1 SAPIEN 真实 mesh 验证

按 articulation 的正确关节索引，将三组关节姿态直接放入相同 seed 的 SAPIEN 场景：

| 姿态 | `fr_link3 ↔ fr_link5` 最小 separation | 最大单步冲量 | 结论 |
|---|---:|---:|---|
| `1.75 mm` 放宽后的候选 | `-0.836 mm` | `25.074` | 真实穿透，不安全 |
| 旧 contact-trace action30 | `-1.240 mm` | `36.856` | 真实穿透，不安全 |
| 专家 frame237 | `+4.690 mm` | `0` | 安全 |

负 separation 表示 mesh penetration。因此 `1.75 mm` 不是找到了新安全解，而是把一个真实穿模解重新标成 CuRobo Success。

结论：

- 不能通过继续缩小球或放宽 self-collision buffer 安全执行当前精确模型目标；
- 当前碰撞配置拒绝 action30 是正确行为；
- 不应永久固化任何能让该 action30 通过的放宽参数；
- 若要继续执行，需要改变目标 pose，例如在模型 quaternion 附近做 collision-aware rotation projection，而不是改变碰撞事实。

扫描结果：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_collision_margin_sweep_20260731/results.json`

正确关节索引的 SAPIEN 静态接触验证：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_collision_margin_sweep_20260731/static_sapien_contacts_correct_indices.json`

### 15.2 排除“新增碰撞球误杀了另一条安全 IK 分支”

仅扫描碰撞阈值还不能完全排除一种可能：后来补充的 `fr_link3`、`fr_link5` 碰撞球也许拒绝了某条没有被默认 seed 找到的安全 IK 分支。

因此又做了一次不依赖新增碰撞球最终判定的枚举：

1. 只在运行时关闭 CuRobo self-collision 检查，保留关节限制和场景碰撞；
2. 对 action 30 的同一个精确 6D EE target 使用 `4096` 个 Halton IK 初值；
3. 将求得的关节角按 `2π` 周期归一化并聚类，去掉同一物理解的周期重复；
4. 只保留位置误差小于 `0.1 mm`、旋转 metric 小于 `1e-4` 的高精度分支；
5. 将每条分支直接设置到同 seed 的 SAPIEN Aloha-AgileX articulation 中，用真实 collision mesh 检查自碰撞。

统计结果：

- `4096` 个 IK 初值中有 `2953` 个在关闭 self-collision 后收敛；
- 原始关节角因 `2π` 周期形成 `1621` 个表面 cluster；
- 周期归一化后得到 `9` 个 cluster，其中 `3` 个是没有精确到达 target 的单次离群解；
- 最终得到 `6` 条不同的高精度物理 IK 分支；
- SAPIEN mesh 判定安全的高精度分支为 `0` 条。

| 分支 | 与 action30 起始关节姿态的 L2 距离 | 最深真实 mesh 穿透 | 最深碰撞对 |
|---:|---:|---:|---|
| 1（最近分支） | `0.098 rad` | `0.827 mm` | `fr_link5 ↔ fr_link3` |
| 2 | `1.487 rad` | `35.573 mm` | `fr_link5 ↔ fr_link3` |
| 3 | `4.805 rad` | `64.068 mm` | `fr_link6 ↔ fr_link3` |
| 4 | `5.053 rad` | `64.074 mm` | `fr_link6 ↔ fr_link3` |
| 5 | `5.141 rad` | `63.424 mm` | `fr_link6 ↔ fr_link3` |
| 6 | `5.493 rad` | `63.149 mm` | `fr_link6 ↔ fr_link3` |

这不是解析意义上的全局无解证明，但已经是对该 6-DOF 精确 pose 的高密度数值枚举。最重要的是，最终安全性由 SAPIEN 真实 mesh 判定，而不是由后来手工添加的碰撞球判定。因此对于这一个 action 30：

- 不是新增碰撞球误杀了某条已经存在的安全精确解；
- 当前精确 EE target 在枚举到的全部高精度物理解中均发生真实自碰撞；
- 重新拟合碰撞球可以改善整个机器人碰撞模型的误报/漏报，但不能把这个精确 target 变成安全 target；
- 要安全执行，需要改变 target pose（当前证据指向小幅修正 quaternion）或提高模型旋转预测精度。

测试过程中对 self-collision 的关闭只存在于进程内存，退出前已经恢复，没有永久修改 CuRobo、RoboTwin 或 FastWAM 配置。

周期归一化后的精简结果：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_ik_branch_enumeration_20260731/canonical_summary.json`

完整原始枚举：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_ik_branch_enumeration_20260731/results.json`

## 16. 可视化与诊断文件

本轮产生的 18 个临时诊断目录已经统一移动到：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731`

目录说明：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/README.md`

原失败评测视频：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_seed4300008_velocity_trace_step100_20260730/click_alarmclock/episode0_randomized-false_success-false.mp4`

独立 RoboTwin observer/third-view 视频：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_saved_right_trace_third_view_20260730/seed4300008_observer_camera.mp4`

原始逐帧图像和三相机 canvas：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_seed4300008_velocity_trace_step100_20260730/click_alarmclock/predictions/demo_clean/episode_000/frames`

修正碰撞球后的固定动作回放：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_saved_right_trace_collision_spheres_v3_20260730/ee_trace_replay.csv`

128 IK seed 测试：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_saved_right_trace_collision_spheres_v5_ik128_20260730/ee_trace_replay.csv`

CuRobo 详细状态测试：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/replay_saved_right_trace_collision_debug_20260730/ee_trace_replay.csv`

action 30 Rothko 像素信号量化：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_pose_ablation_20260730/rothko_rotation_signal.json`

action 30 原始 Wan VAE latent 信号量化：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_pose_ablation_20260730/rothko_rotation_vae_latent_signal.json`

连续专家轨迹原始 Wan VAE round-trip：

`evaluate_results/robotwin_ee_execution_diagnosis_seed4300008_20260730_20260731/action30_pose_ablation_20260730/expert_rothko_original_vae_roundtrip.json`

## 17. 一句话总结

seed 4300008 的失败不是模型 EE 在第 30 步突然跳变，也不是控制目标没有发送；模型的连续目标把 Aloha-AgileX 右臂推入了一个非常狭窄的自碰撞边界，原 CuRobo 碰撞球漏检该区域，导致一个数学上满足 EE 目标但会在 SAPIEN 中发生 `fr_link5 <-> fr_link3` 自碰撞的关节解被执行，物理碰撞随后把实际机械臂推离规划轨迹并污染了后续观测与预测。进一步绕开新增碰撞球枚举到的 6 条高精度 IK 分支全部存在真实 mesh 自碰撞，因此这次 action 30 不能靠换 IK seed 或放宽/重拟合碰撞球获得安全的同 pose 解。

## 18. 2026-07-31 恢复前的碰撞配置快照

本节记录问题排查期间最后使用的右臂碰撞配置。记录完成后，FastWAM 内置
RoboTwin 的碰撞球会恢复到 `/mnt/hwdata/cfy/wam/RoboTwin` 中原始
RoboTwin 版本；本节仅用于复现实验结果，不代表推荐配置。

恢复前文件：

`third_party/RoboTwin/assets/embodiments/aloha-agilex/collision_aloha_right.yml`

SHA256：

`f4a8bf1ded41c5dddb15ecaf7df9e6959f78dc40b9a968140089ba0ff3d323a2`

完整碰撞球参数：

```yaml
collision_spheres:
  fr_base_link:
    - center: [0.0, 0.0, 0.0]
      radius: 0.02
  fr_link1:
    - center: [0.0, 0.0, 0.0]
      radius: 0.02
  fr_link2:
    - center: [0.0, 0.0, 0.0]
      radius: 0.03
    - center: [-0.03, 0.0, 0.0]
      radius: 0.03
    - center: [-0.06, 0.0, 0.0]
      radius: 0.03
    - center: [-0.09, 0.0, 0.0]
      radius: 0.03
    - center: [-0.12, 0.0, 0.0]
      radius: 0.03
    - center: [-0.16, 0.0, 0.0]
      radius: 0.03
    - center: [-0.19, 0.0, 0.0]
      radius: 0.03
    - center: [-0.21, 0.0, 0.0]
      radius: 0.03
  fr_link3:
    - center: [0.0, -0.022, -0.06]
      radius: 0.033
    - center: [0.0, 0.0, -0.06]
      radius: 0.033
    - center: [0.0, 0.022, -0.06]
      radius: 0.033
    - center: [0.04, -0.022, -0.06]
      radius: 0.033
    - center: [0.04, 0.0, -0.06]
      radius: 0.033
    - center: [0.04, 0.022, -0.06]
      radius: 0.033
    - center: [0.08, -0.022, -0.06]
      radius: 0.033
    - center: [0.08, 0.0, -0.06]
      radius: 0.033
    - center: [0.08, 0.022, -0.06]
      radius: 0.033
    - center: [0.12, -0.022, -0.06]
      radius: 0.033
    - center: [0.12, 0.0, -0.06]
      radius: 0.033
    - center: [0.12, 0.022, -0.06]
      radius: 0.033
    - center: [0.16, -0.022, -0.06]
      radius: 0.033
    - center: [0.16, 0.0, -0.06]
      radius: 0.033
    - center: [0.16, 0.022, -0.06]
      radius: 0.033
    - center: [0.20, -0.022, -0.06]
      radius: 0.033
    - center: [0.20, 0.0, -0.06]
      radius: 0.033
    - center: [0.20, 0.022, -0.06]
      radius: 0.033
    - center: [0.24, -0.022, -0.06]
      radius: 0.033
    - center: [0.24, 0.0, -0.06]
      radius: 0.033
    - center: [0.24, 0.022, -0.06]
      radius: 0.033
    - center: [0.0, -0.02, -0.015]
      radius: 0.033
    - center: [0.0, 0.02, -0.015]
      radius: 0.033
    - center: [0.0, 0.0, 0.015]
      radius: 0.028
  fr_link4:
    - center: [0.065, 0.001, -0.062]
      radius: 0.027
  fr_link5:
    - center: [-0.002, -0.012, 0.052]
      radius: 0.03
    - center: [-0.002, 0.012, 0.052]
      radius: 0.03
    - center: [-0.002, -0.012, 0.08]
      radius: 0.03
    - center: [-0.002, 0.012, 0.08]
      radius: 0.03
    - center: [-0.002, -0.012, 0.108]
      radius: 0.027
    - center: [-0.002, 0.012, 0.108]
      radius: 0.027
    - center: [0.03, 0.0, 0.085]
      radius: 0.024
  fr_link6:
    - center: [0.022, 0.0, 0.0]
      radius: 0.03
    - center: [0.05, 0.0, 0.0]
      radius: 0.01
    - center: [0.07, 0.0, 0.0]
      radius: 0.01
    - center: [0.07, 0.03, 0.0]
      radius: 0.01
    - center: [0.07, 0.06, 0.0]
      radius: 0.01
    - center: [0.07, -0.03, 0.0]
      radius: 0.01
    - center: [0.07, -0.06, 0.0]
      radius: 0.01
  fr_link7:
    - center: [0.009, 0.001, 0.0]
      radius: 0.02
    - center: [0.057, -0.009, 0.0]
      radius: 0.01
    - center: [0.037, -0.007, -0.002]
      radius: 0.01
  fr_link8:
    - center: [0.063, 0.015, 0.0]
      radius: 0.01
    - center: [0.044, 0.013, 0.0]
      radius: 0.01
    - center: [0.005, 0.0, 0.0]
      radius: 0.02
  right_camera:
    - center: [0.0, 0.0, 0.0]
      radius: 0.02
```

对应的 CuRobo 碰撞相关参数：

```yaml
collision_sphere_buffer: 0.004
self_collision_buffer:
  fr_base_link: 0.00
  fr_link1: 0.00
  fr_link2: 0.00
  fr_link3: 0.00
  fr_link4: 0.00
  fr_link5: 0.00
  fr_link6: 0.00
  fr_link7: 0.00
  fr_link8: 0.00
  right_camera: 0.00
self_collision_ignore:
  fr_base_link: [fr_link1]
  fr_link1: [fr_link2]
  fr_link2: [fr_link3]
  fr_link3: [fr_link4]
  fr_link4: [fr_link5, fr_link7, right_camera, fr_link8]
  fr_link5: [fr_link6, fr_link7, right_camera, fr_link8]
  fr_link6: [fr_link7, right_camera, fr_link8]
  fr_link7: [fr_link8, right_camera]
```

排查期间还在 `curobo_right.yml` 中临时显式配置过：

```yaml
planner:
  frame_bias: [-0.2315, 0.3063, -0.781]
  num_ik_seeds: 32
  num_trajopt_seeds: 4
  num_graph_seeds: 4
  max_attempts: 10
```

恢复后保留本机可用的 URDF 和 collision YAML 绝对路径，只删除上述诊断期
planner seed 配置，并将 planner 代码恢复为 RoboTwin 原始默认值。
