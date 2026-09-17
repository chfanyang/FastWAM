# FastWAM Rothko 迁移至 VLABench：方案与数据核查清单

## 2026-09-15 独立评测环境进度

- 已从 fastwam 克隆 `fastwam_vlabench`，仿真依赖只安装在新环境。Torch 2.7.1+cu128、NumPy 1.26.4 保持不变；清单与约束见 `experiments/vlabench/requirements-eval.txt`、`constraints-eval.txt`。
- 官方 Evaluator 导入检查通过。补齐 OpenCV 4.10.0.84、colorama、peft 0.14.0；peft 是官方 evaluation 自动导入 OpenVLA 所需，并非改变本实验微调方法。
- 官方 obj.zip（4,987,960,728 bytes）与 scene.zip（1,015,309,482 bytes）已完整下载到 benchmark 自身 assets 目录。ZIP成员路径/符号链接安全检查通过，共38,474个成员，展开约12.02GB；本条记录时仍在解压。
- `pip check` 的 Rtree 1.2.0 平台告警仍存在，但实际 import rtree 成功；其 WHEEL 在 Tag 前存在空行，pip 使用的邮件头解析不能获取 Tags。没有修改包元数据，不声称 pip check 全通过。
- 新增 `experiments/vlabench/smoke_environment.py`：只加载 Track1 select_book 第0场景，保存 camera2/3 图像，经官方IK执行一次保持当前EE位姿的指令，不加载 DiT。
- assets 解压已完成；无模型检查已打印 `ENVIRONMENT_SMOKE_PASS`，输出 `runs/vlabench_environment_smoke`，日志 `logs/vlabench_environment_smoke.log`。camera2/3 均已保存 PNG，官方 IK 后单步控制完成；机器人原点 `[0,-0.4,0.78]`。此结果不是策略成功率，正式评测未启动。
- 独立入口补齐官方脚本的 robots/tasks 注册导入，未改 benchmark 控制循环。rrt 上游普通 wheel 遗漏 rrt 子目录，改为同一提交 `14e47963780affe3ff3bb9fc33a4b14e1a733e85` 的源码可编辑安装，源码位于忽略的 `third_party/rrt-algorithms`；安装脚本显式 `--src third_party`。
- 官方任务模块还间接需要 openai SDK（只导入，不调用远端服务）。固定 openai1.65.5、anyio4.8.0、typing-extensions4.15.0，避免升级偏离 FastWAM 的固定依赖。最终环境列表保存在 `runs/vlabench_eval_environment_freeze.txt`。

日期：2026-09-12。

状态：方案讨论阶段。用户已同意下述初始设定；本次仅整理文档，不授权自动安装环境、修改数据、预计算或启动训练。五项核查细节留待下一轮讨论。

## 1. 目标与隔离原则

将现有 LIBERO 的 video-only Rothko 方法迁移到 VLABench Primitive FT，先在 Track 1 测试。

- Benchmark：`third_party/VLABench`。
- 原始数据：`data/vlabench_primitive_ft_lerobot_video`。
- 原始下载文件保留不动。派生索引、manifest、统计、缓存使用独立命名。
- 新增 VLABench 专用数据适配、配置与评测入口；不改变既有 LIBERO、LIBERO-Plus、RoboTwin 实验默认行为。
- 建议使用独立 `fastwam_vlabench` 环境；不把 VLABench 的 MuJoCo/dm_control 依赖安装进现有评测环境。
- 若需要共享模型代码的新功能，必须默认关闭，并有旧配置回归检查。

## 2. 已确认的第一版实验设置

| 项目 | 设定 |
|---|---|
| Backbone | 原始 Wan2.2 5B，不从 LIBERO 已微调权重初始化 |
| 模型 | 仅 video expert，全微调 DiT |
| VAE | 原始 Wan2.2 VAE，encoder/decoder 均冻结 |
| 图像 | 前视与腕部两相机；各 224×224，横向拼成 224×448；具体原始相机映射待核查 |
| Rothko | 单臂 224×224，横向复制成 224×448 |
| 时序 | 连续 17 帧、预测 16 步；具体 action 起点待时间对齐核查 |
| 拼接 | RGB/Rothko 分别 VAE encode，再按 latent 时间维分两段拼接，不交叉 |
| 几何 | center_frac=0.5，沿用原 LIBERO 相对 chunk 表示；RAY0 为相对表示，不启用 absolute RAY0 |
| 条件 | 语言与当前图像/RAY0，不添加 proprio token |
| Attention | 既有 `rgb_then_raymap_block_causal`，不启用 ray-only 推理实验 |
| 监督 | RGB latent loss 权重 1，Raymap latent loss 权重 5；不新增直接动作监督 |
| 训练范围 | Primitive FT 的 10 类任务联合训练 |
| 训练预算 | 10 epoch，有效 batch128 |
| 优化 | LR 1e-4，5% warmup + cosine；weight decay 0.01，BF16 |
| 验证划分 | 1%；以完整 episode 为单位，若发现同源 replay 则按 source group 隔离 |
| 评测 | Track 1，10 个任务各 50 次，共 500 次 |
| 推理 | replan8，ensemble关闭，legacy Rothko 解码 |
| weights | 每 3 epoch + 最终，即第 3/6/9/10 epoch |
| 完整 state | 中间一次 + 最终，按第 5/10 epoch 规划 |

训练的 batch/GA 组合按卡数和显存测试决定，保持有效 batch128。epoch 对应步数根据最终 train manifest 计算，不按未经校验的元数据硬编码。

验证固定样本、timestep/noise 和 rollout seed；固定子集规模与任务覆盖下一轮确定。验证是离线指标，不能代替 Track 1 rollout。

推理仍联合预测未来 RGB 和 Raymap；不保存预测视频时可跳过无人使用的 RGB VAE decode，但不能因此删掉模型中的未来 RGB 预测部分。

## 3. 当前数据检查结果

### 2026-09-13 更新：改用 image v2.0 数据

当前选定 `data/vlabench_primitive_ft_lerobot`（image v2.0），不再使用下述
video v3 的错误 episode 区间元数据。全量扫描 5,000 episodes / 575,101 frames，
episode/frame/global index、时间戳、任务映射、7 维 state/actions 的有限性均通过；
抽查 459 张三路图片均可解码。没有逐张解码全部图片。

新增独立原始窗口适配器 `src/fastwam/datasets/vlabench_image.py`：

- 在内存中映射字段和 padding key，不重写或复制原始 Parquet。
- 保留导出相机名 `image` / `wrist_image`（也可显式选 `second_image`）；不宣称视角已核实。
- 输出 FastWAM 风格 `images/state/action/raw_state/raw_action` 字典。
- 默认读取 RGB/state[t:t+17]、actions[t:t+16]，尾部重复末帧并保留 padding mask。
  这是取样索引约定，不代表已经确认物理观测与动作的时序关系。
- 原始 state/actions 不进行坐标变换或夹爪翻转；另输出已确认极性的
  `raw_state.gripper_open` / `raw_action.gripper_open`。不归一化、不切分或编码 Rothko；
  缺失 stats 保持 None。
- 当前只是原始窗口读取层，尚未接入 `RobotVideoDataset` 或正式训练配置。
- 不修改公共加载器及 LIBERO/Plus/RoboTwin 配置。单 episode 加载测试的 Arrow
  缓存放在 `runs/vlabench_image_loadcheck/hf_cache`，不改全局缓存设置。

下一步仍需原始 HDF5 对齐控制语义，再决定 VLABench 专用 processor/Rothko 配置。

### 2026-09-13 原始 HDF5 对齐实测（单轨迹，不代表全数据逐条核实）

从已有原始下载的 gzip 前缀中提取了完整的 104,001,579 字节 HDF5 member，
没有续下整个原始包。审计脚本：`scripts/audit_vlabench_raw_image.py`，使用
`fastwam_libero` 环境（`fastwam` 环境没有 h5py，未为此安装或修改依赖）。
证据保存在 `data/vlabench_image_video_audit/raw_sample/source.json` 与 `comparison.json`。

- 原始 `select_poker/episode_472.hdf5` → image `episode_001194.parquet`。
- 指令 `Please pick the poker 3 of clubs`，70 帧。
- raw EE 的世界坐标减 robot position `[0, -0.4, 0.78]`；wxyz 四元数转
  SciPy `xyz` Euler 弧度；夹爪观测值不变：与全部 70 帧 state 的 max error 为 0。
- raw trajectory 前 6 维保留，最后一指命令按 `>0.03` 编成 0/1：与全部 actions
  的 max error 为 0。trajectory 在采集保存时已减 robot position，不应再减一次。
- 前、中、末三帧逐像素相等：`image=raw rgb[:,2]`（代码中的前视），
  `second_image=raw rgb[:,0]`，`wrist_image=raw rgb[:,3]`。
- `state[-1]` 来自 `get_ee_open_state()`，尽管名称有 open，代码实际在两指
  qpos 都小于 0.035 时返回 1；不能把这个 1 直接解释成打开。
- `actions[-1]=1` 对应原始命令 0.04，即打开；0 对应较小的夹爪命令。
  本例第 55 帧命令变成 0，而观测直到第 56 帧才变成 1。

时序诊断（位置为 L2 平均误差，旋转为 SO(3) 测地角）：

| 对比 | 位置误差 | 旋转误差 |
|---|---:|---:|
| state[t-1] vs action[t] | 18.381 mm | 3.116° |
| state[t] vs action[t] | 10.031 mm | 1.789° |
| state[t+1] vs action[t] | 3.807 mm | 0.582° |

这不是单凭相关性认定时序：`SkillLib.step_trajectory()` 内部执行后取观测，
但 `moveto()/pick()` 等外层会补初始观测、删除最后观测。必须结合外层看齐次序。
目前样本支持保留 RGB/state[t] → actions[t]，不整体移动 action 一帧；
target 与实际到达位姿仍有执行误差，不应要求二者完全相等。
尚未对其余任务/技能的全部轨迹做相同 HDF5 核实。

VLABench 专用处理建议（夹爪映射已获用户同意并实施，其余尚未接入训练/部署）：

1. Rothko 当前帧使用实际 state 的 EE；未来帧使用 actions 的目标 EE。
2. 当前夹爪编码为 `1-state_gripper`，未来保留 `action_gripper`，统一为高值表示打开；
   这是阈值化观测，不是连续开合宽度。
3. 两者位置均使用同一个机器人平移坐标系，Euler 转旋转矩阵后再做相对旋转。
4. 部署时将 robot-frame 目标位置加回 robot position，再交给世界坐标 IK；
   夹爪命令最终映射为两指 `[0.04,0.04]` / `[0,0]`。
5. 仅使用前视 `image` 与腕部 `wrist_image`。不修改官方全局夹爪函数或旧实验代码。

夹爪映射实现：`vlabench_gripper_open_fields()`，由 VLABench 专用窗口适配器调用。
不覆盖原始 7 维字段；新增 `raw_state['gripper_open']`（17×1）和
`raw_action['gripper_open']`（16×1），分别复用 state/action 的 padding mask。
后续 Rothko 编码必须显式读取这两个字段，不能再次直接取原始 state 的最后一维。

### 2026-09-13 固定 1% episode 切分

生成脚本：`scripts/build_vlabench_image_split.py`。
冻结文件：`data/vlabench_primitive_ft_lerobot/manifests/episode_split_val01_seed42.json`。
seed=42，10 类任务各 500 episodes，每类固定 495 train / 5 val。
合计 train=4,950 episodes / 569,392 frame starts；val=50 episodes / 5,709 frame starts。
frame starts 包含尾部 padding 起点，未通过丢弃尾部缩小数据。

扫描全部 5,000 条完整 float32 action 序列及 state+action 序列的 SHA256，
均未发现完全重复；train/val episode ID 与 action hash 集合无交集。
任务分配来自逐条检查的官方指令模板，并在 manifest 保存具体正则和 task_index。
数据没有原始 source trajectory/scene ID，因此这只是**按 episode 分层、精确重复检查**，
不能宣称已排除近似 replay 或同场景泄漏，也没有扫描全部图片哈希。

manifest 保存元数据 hash、每条轨迹 hash、切分名单和逐任务统计；脚本拒绝覆盖不一致的
既有 manifest。文件 SHA256：`496ad81dfb65c57bfdcf06c9f289044929c68b93b939b9528806c8890f48e19d`。
切分尚未接入正式训练配置；下述 norm_stats 已使用这份清单。

### 2026-09-13 全窗口 train-only Rothko norm_stats

脚本：`scripts/compute_vlabench_rothko_norm_stats.py`（运行环境 `fastwam_libero`，
CPU 计算、线程数 1；未安装依赖或使用 GPU）。输出：

`data/vlabench_primitive_ft_lerobot/norm_stats/vlabench_rothko_q99p95_h16_224x448_centerfrac05_train99.pt`

同名 `.json` 保存完整几何、manifest SHA256、统计口径及逐任务 train/val clipping。

- 仅拟合 4,950 个训练 episodes 的全部 569,392 起点 × 16 = 9,110,272 个未来位移。
- 位移公式：`R(state[t])^T @ (action[min(t+j,N-1)].xyz - state[t].xyz)`，j=0..15。
- 尾部重复末动作，594,000 个 padding 目标也计入分布；没有每 episode 抽 32 窗口。
- 相对 RAY0 恒为零，不额外计入未来位移分位数，沿用 LIBERO 的未来目标统计口径。
- 精确 `numpy.quantile(abs(relative_xyz), 0.9995, method='linear')`，逐轴对称 ±bound；
  不是分别取 signed Q0.05/Q99.95。无直方图近似、无 histogram_max 截断。
- 几何 224×448（224×224 横向复制），center_frac=0.5、focal=0.2、两种 scale=1、
  boundary_margin=outer_margin=8。方向分量固定 ±1，夹爪边界固定 ±1，不拟合旋转/夹爪分位数。
- float32 实际 bound：xyz = `[0.2380204499, 0.2332057655, 0.4162270129]` 米。

| Split | 起点数 | 未来向量数 | 任一轴超界的向量比例 | 含至少一次超界的窗口比例 |
|---|---:|---:|---:|---:|
| train | 569,392 | 9,110,272 | 0.11431% | 0.71603% |
| val | 5,709 | 91,344 | 0.27588% | 1.01594% |

验证集只用于上述报告，不影响 bound。训练逐轴超界约 0.05001%；验证 xyz 分别为
0.06131%、0.19596%、0.05364%。这些是**位置分量/向量/窗口**裁剪比例，不是全图像素比例。

新增独立 `VLABenchRothkoCodec` 只继承既有单臂复制几何、使用 `vlabench` 环境标识；
旧 LIBERO codec/config 均未修改。验证了错把此 stats 给 LIBERO codec 会明确拒绝。
真实 episode1194 的 start=0/40/69，统计位移与 codec 未归一化中心像素误差 ≤2.1e-17
（float64 几何校验）；同时测试了 90°参考旋转和尾部重复索引。
此时仍未启动 VAE 缓存、Rothko 训练或部署评测。

### 最初 video v3 数据的检查记录（保留供对照）

根据最初 video 版本的本地 `meta/info.json`、parquet 抽样和 README：

- 5,000 episodes，575,101 frames，标注 fps=10。
- LeRobot v3 分片格式，而非默认的一 episode 一 parquet。
- 三路视频：`image`、`second_image`、`wrist_image`，均为 480×480、AV1。
- `state` 与 `actions` 均为 7 维。
- `meta/tasks.parquet` 有 128 条语言索引，不代表 128 个基础任务类别。
- README 中 10 类任务与 Track 1 的任务名集合完全相同：

```text
select_painting
select_book
select_drink
select_chemistry_tube
select_poker
select_mahjong
select_toy
select_fruit
add_condiment
insert_flower
```

每个 Track 1 任务有 50 个配置。任务类别一致不等于场景、轨迹不重合。

本地转换脚本较旧：只声明两路图像，而下载数据有三路视频。因此其语义是重要线索，但不能直接认定它就是这份下载数据的精确生产版本。

## 4. 待讨论核查一：v3 episode 索引

### 已有证据

`data/chunk-000/file-000.parquet` 含 167 行，至少包含两个 episode：

| Episode | 实际行数 | 实际 index 范围 |
|---|---:|---|
| 0 | 93 | 0–92 |
| 1 | 74 | 93–166 |

但 `meta/episodes/chunk-000/file-000.parquet` 给 episode1 写的是 `dataset_from_index=74`、`dataset_to_index=148`，与实际数据不符。

5,000 条元数据的相邻边界中，4,945 处不满足前一个 to 等于后一个 from。这不是“4,945 条 episode 内容损坏”的结论，而是不能依赖这些边界字段直接切片的证据。

### 拟核查方法

1. 扫描所有数据分片的 episode_index、frame_index、index、timestamp、task_index。
2. 检查 episode 完整性、重复/缺失帧、排序和跨文件边界。
3. 根据实际记录重建每个 episode 的文件及行号映射，不假设每个文件只有一个 episode。
4. 独立验证视频文件、时间偏移和解码帧数；不能因数据行索引修正就认定视频自动正确。
5. 对首尾帧、随机中间帧及跨文件 episode 做图像/数值对应检查。

### 拟产物与验收

- 独立 episode manifest、差异报告、任务/语言映射。
- 实际唯一 episode 数和总帧数与报告一致，或者明确列出异常原因。
- 重建索引与随机访问逐行对应；不得静默跳过异常或随机换样本。
- 不直接覆盖原始 metadata。最终采用独立 reader 还是派生兼容数据目录，待核查后讨论。

## 5. 待讨论核查二：RGB/state/action 时间关系

### 已有证据

本地 `scripts/convert_to_lerobot.py` 将 ee_state 转成 position + Euler + gripper；actions 来自 trajectory。

`VLABench/utils/skill_lib.py` 的 `step_trajectory()` 有以下顺序：

```text
给定 EE waypoint → IK → env.step → get_observation → 保存 observation 与 waypoint
```

这提示部分记录可能是执行后观测，但不同 skill 的拼接、初始帧和下载数据转换过程尚未核实。不能直接断言所有 actions[t] 都需要移动一帧。

### 拟核查方法

- 追踪 trajectory_generation、skill、序列拼接和转换的索引规则。
- 抽取多个任务/episode，对比 state[t] 与 action[t-1/t/t+1]，并检查夹爪转换和图像事件。
- 数值接近只能辅助判断，不能用相关性代替数据生成顺序证据。
- 确认位置是世界坐标还是基座相对坐标、Euler 顺序/单位、内部 wxyz 四元数约定。
- 核实 fps=10 是否与实际执行/采样周期一致，而非只看视频标注。

### 验收与待决定点

冻结明确的数据契约：给定 RGB[t]/实际 state[t]，监督动作应从哪个索引开始。

RAY0 的参考使用当前实际 EE，而非误用控制目标的 FK；未来使用目标 EE。若需要偏移，只在 VLABench 适配层实施，并定义末尾 padding。

若下载文件无法提供充分来源证据，应报告不确定性，再讨论是否需要补充原始轨迹/生产版本；不自动下载整套其他数据。

## 6. 待讨论核查三：夹爪与相机

### 夹爪证据

本地 Franka `get_ee_open_state()`：两指位置均小于 0.035 时返回 True，源码注明这个 open 命名存在问题。

转换脚本把目标张开量大于 0.03 映射为动作 1，否则为 0。评测 OpenPi policy 对动作大于等于 0.1 返回两指各 0.04，否则返回 0。

因此 state 夹爪与 action 夹爪可能语义相反；不能原样拼到 Rothko 边界而不验证。

### 拟核查与决策

- 统计实际 state/actions 最后一维的取值，结合张开/闭合图像确认方向。
- 给当前夹爪状态和未来夹爪命令分别定义到统一 Rothko 编码的映射。
- 推理再映射到 benchmark 两指控制值；二值阈值需要明确记录，不静默选定。
- 不修改 benchmark 全局 `get_ee_open_state()`，避免影响专家策略和其他模型。

### 相机证据与核查

本地旧转换脚本使用 rgb[2] 为 front、rgb[3] 为 wrist，OpenPi policy 同样如此；下载数据另有 second_image，其来源需确认。

抽样三路图像，核对视角、方向、是否翻转、色彩通道；评测端再对同一 camera id 检查。第一版只用确认后的前视和腕部，相机映射写入 manifest/config。

dataset/deploy 共用图像处理函数，统一尺寸、插值、antialias、值域和归一化，不复制 LIBERO 已有的两端处理差异。

## 7. 待讨论核查四：Track 1 与训练场景重合

### 当前可确认

任务名集合一致。评测文件为：

`third_party/VLABench/VLABench/configs/evaluation/tracks/track_1_in_distribution.json`

仅凭名称、语言相同或者不同 seed，均不能证明场景无重合。

### 拟核查方法

- 检查训练数据是否含原始 scene/episode config、seed、source id 或其他来源映射。
- 核对 Track 1 的场景配置，包括物体实例、位姿、纹理、指令和随机化设置。
- 若两侧都有配置，规范化后生成 scene fingerprint 进行比对。
- 若训练数据缺少场景来源，只能报告“无法严格确认”，不能以图像看着不同代替证明。

不擅自修改官方 Track 1 来消除疑似重合；是否补充来源信息或增加额外诊断测试，需要用户决定。

## 8. 待讨论核查五：全部窗口统计、padding 与裁剪

### 已确定方向

- 使用独立 VLABench Rothko stats，不沿用 LIBERO stats。
- 使用最终 train split，不用 held-out validation 计算范围。
- 遍历全部训练窗口，不采用每 episode 均匀抽 32 个窗口的近似。
- 保留尾部不完整训练窗口，不能为统计方便只训练完整窗口。
- 采用 symmetric Q99.95；具体实现、输入量和量纲写入报告，不把名称当作充分说明。

### 需要下一轮讨论的规则

“遍历全部窗口”和“把人为 padding 值纳入分位数”是两件不同的事。

建议候选规则：遍历每个合法起点、纳入其中所有真实有效未来目标；尾部复制的 padding 不重复计入分位数。训练仍使用原有 padding 机制。重叠窗口中同一个真实目标对应不同参考起点，不能不加区分地去重。

最终规则尚未批准，不能在实现中自行定案。还需明确：是否保留 RAY0、旋转如何统计、横向重复图块是否重复计数。

### 统计报告要求

- 训练 episode/window 数、完整窗口数、尾部窗口数、有效/填充目标数。
- 位置与旋转编码的范围、原始极值、采用的分位数及样本权重规则。
- 裁剪前数值的超范围比例：按维度、按任一维超界的动作比例、按窗口、按任务分别报告。
- train 与 held-out 分开报告；held-out 超界情况只作诊断，不反向用于放宽 stats。
- 区分训练编码时的数值裁剪与控制执行时的安全裁剪，不能混为一谈。
- 保存数据、split、几何、padding、代码版本指纹，checkpoint/cache 引用同一身份。

## 9. 评测实现与可复现性

- 独立 `experiments/vlabench/` 入口，复用官方环境、IK和成功判据。
- EE 目标先正确还原环境坐标，再调用官方 IK；不直接套 LIBERO OSC 控制。
- 默认不额外做动作平滑、碰撞参数调整或改变最大步数。
- 记录实际语言、seed/episode config、目标/执行后 EE、IK 标志、步数及耗时。
- SR、IS、PS 分别报告；错误、正常失败和未完成分开统计。
- 官方 evaluator 会捕获异常后跳过该 episode；我们必须报告覆盖率和错误数，不能只对幸存 episode 的均值声称完整成功率。
- 支持逐 episode 保存及续测，身份包含模型、VAE、stats、控制、解码、场景配置。
- 视频使用简短 task/trial 名称；预测视频可选，不默认大量写盘。

论文《Learning to Use Imagination: Progress-Conditioned Future Utilization for World Action Models》明确报告 VLABench 10 任务×50 episodes 和 SR/IS/PS，但未在所查相关文字中指定 Track JSON；因此不能声称暂定 Track 1 已严格复现该文全部评测协议。

## 10. 执行阶段（等待讨论与审核）

1. 讨论并确认本文件五项核查的方法与边界。
2. 做只读数据核查，输出证据和需要决策的歧义。
3. 审核最终数据契约，再实现独立数据/控制适配。
4. 生成 1% 无泄漏 split、stats、语言缓存；保存完整参数来源。
5. 环境与 GT 编解码/控制检查；短训练覆盖验证、保存、恢复。
6. 冻结定义后决定 latent 预计算和显存/吞吐配置，再正式训练。
7. Track 1 完整评测。其他 tracks 暂不纳入第一版范围。

在核查结论未冻结前，不先计算大量 latent 缓存、不修改 benchmark 控制语义、不自动开始训练。

### select_book 在线评测准备（尚未联调通过）

- 新增独立 `experiments/vlabench/policy.py` 与 `eval_select_book.py`，不修改官方 benchmark 或 LIBERO/Plus 入口。
- 策略使用 front camera2/wrist camera3、实际 EE 减 robot base translation、state gripper取反。预测帧1..16为动作，跳过RAY0；还原 world XYZ 后交给官方 evaluator 的 IK/控制循环，无额外轨迹平滑。
- 入口读取 checkpoint 所在 run config；原始VAE、legacy decoder，动态编码实际环境语言。replan 和夹爪阈值必须显式传参；建议 replan8/threshold0.5，待用户确认。
- 可只选 Track1 select_book 50个配置，保存视频、每episode结果及实际语言；异常记录后抛出，不吞掉异常缩小成功率分母。不覆盖非空输出目录；续测尚未实现。
- 目前仅语法检查通过，未环境 rollout。发现缺少 dm_control/open3d/mediapy；assets目录当前只有base/robots，任务物体资源仍需核查下载。
- 用户要求独立环境：已启动从 fastwam 克隆 `fastwam_vlabench`（不修改 fastwam_libero/fastwam_libero_plus）；后续仅在新环境安装仿真依赖、检查渲染/单步IK后才可声称可以直接评测。

### 2026-09-15 Wan2.1 latent 预计算启动

- 用户授权八卡预计算，入口 `scripts/precompute_vlabench_wan21_8gpu.sh`，tmux `vlabench_wan21_latents`，日志 `logs/vlabench_wan21_latents.log`。
- 首先在 `runs/vlabench_latent_probe/<timestamp>` 检查32窗口，复用原预计算脚本的 latent 逐位比较及固定 RNG 下 training_loss 等价检查；通过后才自动全量计算。
- 全量目录：`data/vlabench_train99_rothko_centerfrac05_wan21_bf16_h16_latents`。训练窗口569392，每窗口RGB/Rothko两份[16,5,28,56] BF16，共约285.7GB，不包含验证集。
- batch8、每卡4个CPU workers、1024窗口/分片，原始Wan2.1 VAE，RGB与Rothko分别 encode，tiled=false。沿用已有脚本：会加载模型用于结束时的loss等价校验，不是只构造VAE的精简进程。
- Cache contract 包含 split SHA、数据元信息、Rothko stats fingerprint、预处理/位姿/夹爪约定；VLABench opt-in 缓存读取支持已有完整样本路径，pixel-free读取尚待训练加速接入，未静默开启训练缓存。
- 完成分片可续算，只有全部校验通过才写 `_SUCCESS`。本记录是启动记录，不表示缓存已经完成。

### 2026-09-14 固定验证样本组

- `scripts/build_vlabench_validation_windows.py` 从已有 val split 的每个任务固定抽两个不同 episode，各均匀选一个 frame start；seed42，全起点含 padding，未修改 train/val 切分。
- 冻结清单：`data/vlabench_primitive_ft_lerobot/manifests/validation_windows_10task_20samples_seed42.json`。共20个 loss样本、10个预测视频样本（每任务一个），其中2个窗口包含图像尾部 padding。
- 记录 episode、frame、language task_index、val_dataset_index、diffusion_seed 和 split SHA256；重新生成若内容改变则拒绝覆盖。
- 两种 VLABench backbone 配置使用同一清单；每500 step 验证，20步去噪。复用 trainer 已有 fixed-manifest 验证与 RNG 隔离逻辑，不修改公共 trainer。
- Dataset 新增 episode_split_metadata 供既有验证入口核验 split 指纹。20个记录均已通过实际读取的 episode/frame/padding 索引检查。
- 这是固定的小规模、task-balanced 监测集，不是全部 held-out 数据的无偏平均，也不能替代 Track 1 在线评测。
- 三步 smoke 脚本显式 `eval_sample_manifest=null`，仍只测一个验证样本，避免后续短测意外运行完整监测组。此次未启动正式训练或新的 GPU 测试。

### 2026-09-13：模型输入数据集接入（尚未 GPU 训练）

- 新增独立 `VLABenchVideoDataset`，只读 1% split manifest；stats 必须与 manifest SHA256 一致。
- 前视 `image` 与腕视 `wrist_image` 分别 bilinear + antialias resize 至 224×224，横拼、归一化至 [-1,1]；复用现有纯 RGB 布局函数，不修改 LIBERO 数据集。
- 连续 RGB/state 17 帧，action 16 帧；当前 state 与未来 action 的 xyz Euler 转 wxyz，使用各自已审计的 gripper_open 极性编码 Rothko。末尾重复且独立记录 RGB/action padding，不跨 episode。
- 默认懒加载 Parquet，仅缓存两个 episode 的压缩图像列，不生成全量 Arrow 磁盘副本。原窗口读取器仍保留旧默认行为，新视频数据集显式开启 lazy。
- 新旧读取器在 episodes 0/1 的窗口 0、40、92、93、166 上 RGB、state/action、夹爪与 padding 逐值一致。训练集长度 569392；首尾样本 RGB/Rothko 均为 [3,17,224,448]。
- Euler 转换与 SciPy 的 xyz 定义在 100 个随机姿态上匹配（float64 1e-12）；不是改变原始坐标系。
- 语言条件必须读取真实缓存（metadata 校验），不使用零向量替代；`include_text_context=False` 只用于 CPU 数据诊断。128 条语言缓存、模型工厂的 VLABench 分支、训练配置和 GPU 训练/验证/保存测试仍待后续完成。

### VLABench 语言缓存启动记录

- 配置：`configs/precompute_vlabench_text.yaml`，复用 `scripts/precompute_text_embeds.py`，UMT5 BF16、长度 128、原 DEFAULT_PROMPT、batch 16，四个 rank 分摊指令。
- 输出：`data/text_embeds_cache/vlabench`；不覆盖已有文件，不修改 HF_HOME 或其他数据集缓存。
- GPU：0–3；tmux：`vlabench_text_cache`；日志：`logs/vlabench_text_cache.log`。本记录表示已启动，完成情况以日志和缓存完整性校验为准。
- 启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
TOKENIZERS_PARALLELISM=false PYTHONPATH=src \
DIFFSYNTH_MODEL_BASE_PATH=/mnt/hwdata/cfy/FastWAM/checkpoints \
/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/torchrun \
  --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py \
  --config-name precompute_vlabench_text
```

### 专用配置接入结果（未启动 GPU 训练）

- 语言缓存 128/128 完成，全部通过 shape、BF16、有限值、mask、prompt hash 和编码器 metadata 校验；总计 134680192 bytes，无超长指令。
- 新配置：`configs/data/vlabench_rothko_2cam224.yaml`、`configs/model/fastwam_video_only_vlabench_rothko.yaml`、`configs/task/vlabench_rothko_2cam224_full_wan22_5b_1e-4.yaml`。
- 模型复用 Wan2.2 5B 结构，显式选择 `vlabench_rothko` 与 `future_rgb_mode=joint`；RGB/raymap latent 时间块拼接，原始冻结 VAE、无 proprio token，RGB/raymap 权重 1/5。
- 数据集显式暴露 `raymap_codec`，与公共训练入口核对 geometry 和 stats fingerprint。`pretrained_norm_stats` 指向同一 Rothko stats 的 JSON 元数据，并与 .pt metadata 严格比较，仅供 checkpoint 身份记录，不额外归一化 action。
- 正式预算配置暂定四卡 batch8/GA4、有效 batch128、10 epoch、LR1e-4、5% warmup、cosine、WD0.01；weights 每3 epoch＋最终，state 每5 epoch＋最终。尚未测显存/速度，不代表已通过正式训练验证。
- 当前沿用固定 seed42 的4样本验证，每500 step、20步去噪。这4个样本按现有 trainer 的连续索引选取，不是全10任务验证指标；后续若需要代表性监测应另定覆盖任务的固定样本组。
- CPU 检查：Hydra resolve、真实 train/val 构建（569392/5709）、双方完整样本含语言，以及 dataset/model codec contract 全通过；checkpoint metadata 11项测试通过，包括 VLABench/LIBERO 双向拒绝误加载。
- 没有修改旧 LIBERO/Plus/RoboTwin 配置。下一步需单独运行小 batch 的训练、验证、保存与恢复测试；使用 max_steps 时须覆盖 `save_every_epochs=null`、`state_save_every_epochs=null`，改为短测的 step 保存间隔。
