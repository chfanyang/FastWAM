# FastWAM 对 LIBERO-Plus 的独立训练与评测适配方案

> 文档状态：方案已确认并完成第二轮细节审查，代码尚未开始实施  
> 创建日期：2026-08-13  
> 当前开发分支：`feat/libero-plus-support`  
> LIBERO-Plus 源码：`third_party/LIBERO-plus`  
> 目标：在不影响原始 LIBERO 和 RoboTwin 的前提下，为 FastWAM 的
> video-only Rothko 路线增加独立的 LIBERO-Plus 训练与评测支持。

## 1. 已确认的核心决定

### 1.1 LIBERO 与 LIBERO-Plus 是两个 benchmark

两者不进行混合训练。Plus 增强训练不共用以下实验产物：

- 训练数据目录；
- action/dataset normalization stats；
- Rothko normalization stats；
- 文本 embedding cache；
- Hydra data/task/eval 配置；
- checkpoint 和运行目录；
- 评测结果目录与汇总文件。

约定的目录边界为：

```text
原始 LIBERO
  data/libero_mujoco3.3.2/
  data/text_embeds_cache/libero/
  runs/libero_.../
  evaluate_results/libero/

LIBERO-Plus
  data/libero_plus/libero_plus_lerobot/
  data/text_embeds_cache/libero_plus/
  runs/libero_plus_.../
  evaluate_results/libero_plus/
```

video-only Rothko 模型不使用 action/state 的 processor dataset stats。两种协议
分别使用各自训练时匹配的 Rothko stats；原始 LIBERO 与 Plus 的 Rothko stats
不得混用。

### 1.2 使用不分 suite 的完整 LeRobot 数据集

训练数据选择：

```text
Hugging Face: Sylvest/libero_plus_lerobot
本地目标：data/libero_plus/libero_plus_lerobot
```

这是 LIBERO-Plus 将四个基础 suite 合并、统一重编号后的 LeRobot v2.1 数据集，
训练时不再按 suite 拆成四个 dataset root。任务仍源于原始 LIBERO 的 40 个基础
技能，但 observation 是在 Plus 扰动环境中 replay 后重新记录的，因此不是原始
LIBERO 图像的简单复制。官方 `meta/info.json` 显示：

```text
episodes: 14,347
frames:   2,238,036
tasks:    40
fps:      20
cameras:  observation.images.front
          observation.images.wrist
state:    observation.state, 8D
action:   action, 7D
```

正式训练只使用该单一 dataset root，不使用 `libero_plus_data_4suite` 中的四个
suite ZIP。论文声称成功过滤后有超过 20,000 条轨迹，但当前完整合并 LeRobot/RLDS
公开版实际只有 14,347 条；官方没有解释剩余差异。因此 epoch、训练时长和数据
覆盖率全部按 14,347 条实际 metadata 计算，不能按论文数字估算。

### 1.3 Plus 内部四个 suite 一起训练

这里的“一起训练”仅指 LIBERO-Plus 内部：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

联合训练集共有 40 个基础任务。LIBERO-Plus 的 10,030 个任务是基于这些任务构造
的扰动评测任务，不是 10,030 条独立训练技能。

第一版加载一个完整 LeRobot dataset root，并采用数据集的完整有效窗口分布采样。
因此 episode 更长、有效窗口更多的任务会
获得更高采样概率，并不保证四个 suite 或 40 个任务等权。若后续需要等权训练，
应先恢复 episode 到 suite 的映射，再增加显式的 suite/task balanced sampler。

### 1.4 必须支持的两种实验协议

#### 协议 A：零样本鲁棒性评测

```text
原始 LIBERO 数据训练的 checkpoint
    + 原始 LIBERO dataset/Rothko stats
    + checkpoint 对应的 VAE
    -> LIBERO-Plus 10,030 个扰动任务
```

此协议不读取四个 Plus 训练数据目录，也不继续训练。它回答的是：仅在原始 LIBERO
上训练的 FastWAM，面对 Plus 分布偏移时还能保留多少成功率。

#### 协议 B：Plus 增强训练与评测

```text
四个 Plus suite LeRobot 数据联合微调
    + Plus dataset/Rothko stats
    + 独立 Plus text cache
    -> LIBERO-Plus 10,030 个扰动任务
```

此协议回答的是：只使用 Plus 扰动训练数据、从原始 Wan2.2 初始化训练后，相比只用
原始 LIBERO 数据训练的协议 A 改善了多少。两边的 video DiT 和 VAE 都从同一个原始
Wan2.2 权重开始，但分别使用各自的数据、Rothko 几何和 Rothko stats。

这里不能再表述为“只改变训练数据”：协议 A 的现有基线使用 `center_frac=0.5`，协议 B
已确定使用 `center_frac=0.6`，两者至少同时改变了训练域和动作图像表示。第一版用于
比较两套完整方案的实际鲁棒性；若要严格隔离 Plus 数据本身的增益，必须额外训练一套
“原始 LIBERO + center_frac=0.6 + 原始 VAE”的控制组，并匹配 optimizer updates、
effective batch size 和训练 epoch。

两种协议共用同一个 LIBERO-Plus 环境、任务枚举和成功判定代码，但不得共用运行
目录。最终汇总必须能把 A/B 按 suite、扰动类别、难度和基础任务逐项对齐比较。

## 2. 第一版模型基线

第一版复用当前 LIBERO Rothko 的总体范式，但配置和数据完全独立：

```text
language + current RGB + current Rothko
    -> Wan2.2 video expert
    -> future RGB latents + future Rothko latents
    -> VAE decode Rothko
    -> absolute EE target + gripper
    -> LIBERO OSC action
```

建议的第一版固定项：

```text
RGB cameras: front + wrist
RGB layout:  [front 224x224 | wrist 224x224] = 224x448
Rothko:      单臂 224x224 横向复制为 224x448
observations: 连续 17 帧
prediction:   连续未来 16 步
action/video frequency ratio: 1
video DiT initialization: 原始 Wan2.2
VAE initialization: 原始 Wan2.2 VAE
attention mask: rgb_then_raymap_block_causal
proprio token: 关闭
```

Rothko 几何参数第一轮建议沿用当前 LIBERO 基线：

```text
focal: 0.2
center_scale: 1.0
dir_scale: 1.0
center_frac: 0.6
boundary_margin: 8
outer_margin: 8
duplicate_horizontal: true
```

第一版直接使用 `center_frac=0.6`，并针对 LIBERO-Plus 重新计算 Rothko stats。
训练、评测及后续可能微调的 VAE 必须始终使用该几何配置及其匹配的 stats，不能
复用原始 LIBERO `center_frac=0.5` 实验的 Rothko stats 或针对 0.5 Rothko 微调过的
VAE。未经微调的原始 Wan2.2 VAE 本身不绑定 center fraction，可以作为两边相同的冻结
初始化权重。

## 3. 环境与资产隔离

LIBERO-Plus 使用与原始 LIBERO 相同的 Python 顶层包名 `libero`，官方安装方式会
替换原来的 LIBERO。因此不能在 `fastwam_libero` 环境里直接覆盖安装。

新增独立环境：

```text
fastwam_libero       -> 原始 LIBERO
fastwam_libero_plus  -> LIBERO-Plus
```

推荐流程：

1. 从已经可用的 `fastwam_libero` 克隆出 `fastwam_libero_plus`；
2. 只在新环境中卸载或覆盖原始 `libero`；
3. 在新环境中执行 `pip install -e third_party/LIBERO-plus`；
4. 安装 `third_party/LIBERO-plus/extra_requirements.txt`；
5. 下载官方 `assets.zip` 并解压到：

```text
third_party/LIBERO-plus/libero/libero/assets/
```

6. 使用独立的 LIBERO 配置路径，确保 `benchmark_root`、`bddl_files`、
   `init_states`、`assets` 都指向 `third_party/LIBERO-plus`；
7. 在两个 conda 环境里分别打印 `libero.__file__`，验证没有串环境。

上游代码已经支持 `LIBERO_CONFIG_PATH`，因此启动训练/评测时必须显式设置两个不同的
目录，例如：

```text
原始 LIBERO: LIBERO_CONFIG_PATH=/mnt/hwdata/cfy/FastWAM/.libero_original
LIBERO-Plus:  LIBERO_CONFIG_PATH=/mnt/hwdata/cfy/FastWAM/.libero_plus
```

不能让两套环境共用默认的 `~/.libero/config.yaml`。仅分 conda 环境不足以隔离这个
用户级配置文件，混用会让一个环境悄悄读取另一套 BDDL、init state 或 assets。

`third_party/LIBERO-plus/` 已整体加入 `.gitignore`，因此后续不能把关键修复只改在该目录
里，否则提交和换机器后会全部丢失。优先在受版本控制的 `experiments/libero_plus/` 中做
wrapper；若确实必须修改上游 Plus，需保存可重放 patch、记录上游 commit，并在环境安装
脚本中显式应用和校验该 patch。两个本地 `config.yaml` 只保存机器路径，不提交；仓库中
提交不含绝对路径的模板。

安装完成后的最低检查：

```text
fastwam_libero:
  libero.__file__ -> third_party/LIBERO/...

fastwam_libero_plus:
  libero.__file__ -> third_party/LIBERO-plus/...
```

LIBERO-Plus 四个评测 suite 的任务数应为：

```text
libero_spatial: 2402
libero_object:  2518
libero_goal:    2591
libero_10:      2519
total:         10030
```

## 4. 下载并验证训练数据

计划使用 Hugging Face CLI 下载完整合并 LeRobot 数据集：

```bash
mkdir -p data/libero_plus/libero_plus_lerobot

hf download Sylvest/libero_plus_lerobot \
  --repo-type dataset \
  --local-dir data/libero_plus/libero_plus_lerobot
```

下载后先执行只读审计，不立即修改 parquet：

1. 检查 `meta/info.json`、`meta/tasks.jsonl` 和实际 episode/stats metadata 布局；
2. 检查 episode、frame、task、video 数量；
3. 每个基础任务至少抽一个 episode，覆盖四个 suite/40 个基础任务；
4. 验证每个 episode 的 frame index、timestamp 和视频长度；
5. 验证相机 key 为 `front/wrist`，分辨率为 256×256；
6. 检查 `observation.state` 和 `action` 不存在 NaN/Inf；
7. 验证 episode 边界处不会跨 episode 取 17 帧窗口；
8. 验证 timestamp/FPS 与“一帧对应一个 env action step”的时间尺度一致；
9. 统计 action norm、连续重复帧、开头 settling 和结尾成功后静止段，量化 no-op 比例。

原始 FastWAM LIBERO 使用的是 `*_no_noops_lerobot`，而合并 Plus 发布版的名称没有表明
已经去除 no-op。第一版不能默认两者一致。若 Plus 含大量 no-op/成功后静止尾段，应先报告
其按 suite/category 的占比，再决定是否生成一个保持 action-RGB-side-channel 同步的
no-noop 派生版本；不能只删 action row 而不重建对应视频和 timestamp，也不能在不记录的
情况下静默过滤。

不能仅依据“LeRobot v2.1”假定 metadata 一定是
`meta/episodes.jsonl`/`meta/episodes_stats.jsonl`。下载后的 Hub revision 可能使用
JSONL，也可能使用分片 parquet metadata，而当前仓库内置 LeRobot reader 和
`scripts/add_libero_osc_target_pose.py` 只直接支持前一种。实现前必须加一个 schema
gate：识别实际格式；若 reader 不支持就显式转换 metadata 或增加适配器，不能只改
数据 parquet 后假装 metadata 已同步。

当前对样本 parquet 的初步观察是：

```text
observation.state[:6] 约为 EE xyz + axis-angle
observation.state[6:8] 约为 Panda 两指关节位置
action[:6]             约为 normalized delta OSC pose
action[6]              为 gripper command
```

但在批量写入前必须进一步确认，不能只依据一个 episode：

- 对多个任务比较 state 与仿真 observation 的 EE pose；
- 用当前 LIBERO OSC 公式做 action -> absolute target -> action 数值回环；
- 检查旋转组合顺序仍是 `R_target = R_delta @ R_current`；
- 检查 position scale 仍为 0.05 m，rotation scale 仍为 0.5 rad；
- 检查两指 qpos 到 `gripper_open` 的公式和符号；
- 统计 `action[6]` 的 unique values/分布，并确认其环境语义。

夹爪是这里的硬阻塞项。当前 `RobotVideoDataset` 对 LIBERO Rothko future gripper 使用
`raw_action[..., -1:].clamp(0,1)`；如果 Plus 的 raw action 仍是 LIBERO 环境约定的
`-1=open, +1=close`，这个写法会得到相反语义。转换必须在 schema 中显式声明，例如：

```text
若 raw action 为 -1=open, +1=close：gripper_open = (1 - action[6]) / 2
若发布数据已经是 0=close, 1=open：gripper_open = action[6]
```

不允许通过数值范围猜测后直接 clamp。转换后要同时检查 observation 的实际开合状态、
下一帧 gripper qpos 和 eval 端 `1 - 2 * gripper_open` 的方向一致。

另外，action -> absolute target -> action 的数学回环只能证明公式内部自洽，不能证明
发布数据的 state/action 语义正确。还需要比较 target 与下一帧实际 EE pose 的误差分布；
二者不必完全相等（controller target 不等于一步后的实际到达位置），但不同 suite、任务、
扰动类别之间不应出现异常跳变或整体符号反转。

### 4.1 训练发布版的信息缺失必须先量化

当前合并 LeRobot metadata 只有 40 个基础 task 字符串，不能据此假定语言扰动后的
instruction 仍被保留。正式训练前必须回答：

- 全部 episode 是否真的只有 40 个 canonical instruction；
- language perturbation episode 是否仍保存改写后的语言；
- 每个 episode 是否包含 suite、perturbation category、difficulty 和 source trajectory ID；
- 公开合并版实际覆盖哪些扰动类别。

目前公开数据可恢复出的类别标签看起来只有 `env`、`lighting`、`language`、
`camera_pose`、`sensor_noise` 五类，而官方评测有七类。这个结论必须用下载后的 metadata
再次核实。若确实缺少两类，协议 B 应明确称为“公开 Plus 训练发布版增强”，不能声称
训练覆盖了全部七类。若 language episode 也只保留 canonical instruction，则它提供了
视觉/环境增强，但没有提供 language rewrite 监督；不得人工编造改写文本。

suite/category/source trajectory 映射即使不参与第一版自然采样，也应保存成只读 sidecar
manifest，供数据审计、分组验证和结果解释使用。无法从官方数据可靠恢复的字段应写为
unknown，而不是根据 episode 编号猜测。

如果这些语义与原始 LIBERO 不一致，应停止转换并先修正 Plus 专用转换器。

## 5. 为 Rothko 增加 parquet side-channel

### 5.1 目标字段

每个 Plus parquet 增加：

```text
observation.state.ee_pose_wxyz       [x,y,z,qw,qx,qy,qz]
action.osc_target_pose_wxyz          [x,y,z,qw,qx,qy,qz]
observation.state.gripper_open       [open]
action.gripper_open                   [open]
```

时间对齐保持为：

```text
窗口起点 t：
  Rothko frame 0     = observation EE pose at t
  Rothko frames 1:17 = action rows t:t+16 对应的 absolute OSC targets
                       与显式 action.gripper_open
  RGB frames 0:17    = observation rows t:t+17
```

需要用模拟器 replay 明确验证 action row `t` 对应从 observation `t` 发出的 controller
command，而不是 `t+1` 或上一个 action。否则整个 Rothko future block 会产生一帧错位，
而仅检查 tensor shape 和轨迹平滑性无法发现。

### 5.2 实现方式

不直接改变原始 LIBERO 转换脚本的默认行为。建议采用以下一种实现：

```text
首选：给 scripts/add_libero_osc_target_pose.py 增加明确的 source schema 参数，
      但保留原始 LIBERO 默认 schema；Plus 配置显式选择 libero_plus_v21。

备选：新增 scripts/add_libero_plus_osc_target_pose.py，仅复用纯数学转换函数。
```

无论采用哪种方式，都必须满足：

- 原字段逐值不变；
- 单 episode 先写临时文件，再原子替换；
- 写入前保存 metadata 备份；
- 支持 `--dry-run`、`--verify`、`--skip-existing`；
- 只在所有 episode 成功后更新 metadata；
- 每条 quaternion 的 norm 接近 1；
- action 数值回环误差处于既定 tolerance 内；
- 中断后能够安全继续，而不是产生半写入字段。

metadata 更新范围不只包括 `info.json` 的 features；实际数据版本中若存在 episode stats、
dataset-level stats、parquet shard index 或 chunk/file size 信息，也必须同步更新并验证。
即使 video-only 训练不使用 processor stats，也不能留下一个对通用 LeRobot 工具而言自相
矛盾的数据集。

建议把 source 数据视为不可变输入。若空间允许，派生目录只复制 parquet/metadata，视频用
只读硬链接或符号链接复用；若必须原地写入，则至少记录 Hub repo revision、修改前 metadata
备份、逐 episode 转换状态和 parquet checksum。`--skip-existing` 必须校验字段内容和版本，
不能只看列名存在就跳过。

建议执行顺序：

```text
1 episode dry-run
10 episodes dry-run
10 episodes actual write
10 episodes verify
全数据写入
全数据 verify
```

## 6. Normalization：只保留 Rothko stats

本方案只需要 Rothko region stats。下面明确说明为什么不再保留 processor dataset
stats 依赖。

### 6.1 Processor/dataset stats 在本方案中不需要

旧 action-expert 路径需要 `dataset_stats.json` 来归一化 state/action，并对模型
预测的 normalized action 做反归一化。但当前 video-only Rothko 路径中：

```text
训练：loss 只使用 RGB latent 和 Rothko latent
      action/proprio 的数值不进入模型（proprio_dim=null）

Rothko target：直接读取 raw absolute EE pose/OSC target side-channel

部署：Rothko 直接解码成 absolute EE target
      再通过固定的 LIBERO controller scale 转为 normalized OSC action
```

因此 processor 的 action/state min-max stats 不参与训练目标，也不参与部署动作转换。
当前代码仍强制加载它，只是通用 `FastWAMProcessor`、旧 action 路径和 checkpoint
fingerprint 逻辑遗留下来的技术债，不应继续成为 LIBERO-Plus 的真实依赖。

实现时应：

- 为 video-only Rothko dataset 增加“不归一化 action/state”的显式模式；
- processor 仍负责图像变换、字段合并和 padding mask，但不构造 normalizer；
- `proprio_dim=null` 时不生成或不传入 normalized proprio；
- video-only Rothko 训练不计算、不加载、不复制 `dataset_stats.json`；
- video-only Rothko checkpoint 不保存或校验 dataset-stats fingerprint；
- LIBERO-Plus eval 的 Rothko 分支不再要求 `EVALUATION.dataset_stats_path`；
- 保持原始 action-expert/联合模型路径的 dataset stats 行为不变。

这不是只在 YAML 里写 `normalize_action_state: false` 就能完成：当前
`FastWAMProcessor` 构造函数没有该参数，preprocess 会无条件调用 normalizer；
`RobotVideoDataset` 对 validation 也会在缺少 stats 时直接报错，eval 还会无条件构造
processor 和 normalized proprio。实现时必须同时打通 processor、train/val dataset builder、
checkpoint metadata 和 Rothko eval 四条路径。关闭后 action/state 可以作为保持 shape 的 raw
pass-through 或不进入 model 的 placeholder，但 image transform、instruction、padding mask 和
raw side-channel 仍必须正常输出。

### 6.2 Rothko region stats（必须）

作用：决定相对 EE 位移在 Rothko 中心区域的像素编码范围。

建议路径：

```text
data/libero_plus/
  libero_plus_rothko_region_symmetric_q99p95_h16_centerfrac06_224x448.pt
  libero_plus_rothko_region_symmetric_q99p95_h16_centerfrac06_224x448.json
```

最终 stats 不再采用“每个 episode 均匀抽 32 个窗口”的近似值。需要先修改
`scripts/compute_libero_rothko_norm_stats.py`，使其支持：

- 一个完整合并 LeRobot dataset root；
- `--all-windows`，枚举每个 episode 的全部有效 17 帧窗口；
- 输出 dataset metadata/hash、episode 数、窗口数和实际 Rothko 参数；
- 输出每个 xyz channel 的 quantile bound、observed max 和 clipping ratio；
- 将 `environment`/`benchmark` 标记为 `libero_plus`；
- 文件名包含 horizon 和 center fraction，避免误用。

正式计算采用：

```text
quantile: 0.9995，即取 |relative_xyz| 的 Q99.95 作为对称边界
horizon: 16
sampling: all valid windows
center_frac: 0.6（第一版固定值）
```

这里的 `0.9995` 表示总计约 0.05% 的 absolute-value 尾部可能被裁剪；对于不对称分布，
不能等同为原始有符号数据严格的 `Q0.05~Q99.95`。JSON 中要同时保存有符号 quantiles、
absolute quantile、最终对称 bound 和实测 saturation/clipping ratio，避免名字产生歧义。

计算完成后，随机抽样进行 encode -> decode 数值回环，并报告：

- EE position MAE/max error；
- rotation geodesic error；
- gripper accuracy；
- 每个 xyz channel 的 clipping 数量和比例。

## 7. 文本 embedding cache

单独建立：

```text
data/text_embeds_cache/libero_plus/
```

即使 Plus 的 40 条基础语言中有一部分与原始 LIBERO 完全相同，也不复用原始
LIBERO cache 目录。这样可以避免后续加入语言扰动或改变 prompt 时产生不可见的
跨 benchmark 依赖。

需要检查：

- dataset 实际使用的每个完整 prompt（含 `DEFAULT_PROMPT` 模板）都有 cache；
- cache 的 embedding dimension、dtype 和 tokenizer/model ID 一致；
- 随机读取 dataset sample 时不触发现场文本编码；
- 不在训练 worker 内重复加载 UMT5。

如果第 4.1 节确认存在超过 40 条语言改写，则 cache 数量必须按实际唯一 prompt 计算，
不能硬编码为 40。零样本评测中的动态 Plus language instruction 不依赖训练 cache；评测
进程应由已加载的 text encoder 编码，并记录原始 instruction，不能替换成 canonical 文本。

## 8. 数据配置与代码改造

### 8.1 新增配置

建议新增：

```text
configs/data/libero_plus_rothko_2cam224.yaml
configs/task/libero_plus_rothko_2cam224_full_1e-4.yaml
configs/sim_libero_plus.yaml
```

Plus data config 的关键字段应为：

```yaml
dataset_dirs:
  - ./data/libero_plus/libero_plus_lerobot

shape_meta:
  images:
    - key: front
      raw_shape: [3, 256, 256]
      shape: [3, 224, 224]
    - key: wrist
      raw_shape: [3, 256, 256]
      shape: [3, 224, 224]

num_frames: 17
action_video_freq_ratio: 1
video_size: [224, 448]
concat_multi_camera: horizontal
raymap_representation: libero_rothko
pretrained_norm_stats: null
text_embedding_cache_dir: ./data/text_embeds_cache/libero_plus

processor:
  normalize_action_state: false
```

`raw_action_meta` 和 `raw_state_meta` 使用第 5 节新增的四个 side-channel 字段。future
gripper 必须读取显式的 `action.gripper_open`，不能继续对 raw action 末维直接 clamp。

### 8.2 数据层兼容

需要检查或修改：

```text
src/fastwam/datasets/lerobot/base_lerobot_dataset.py
src/fastwam/datasets/lerobot/robot_video_dataset.py
```

目标不是添加 Plus 特判，而是保证标准 LeRobot v2.1 单根目录、`front/wrist`
camera keys 和新增 raw side-channel 能通过现有通用接口读取。

必须保持原始 LIBERO 的 `image/wrist_image` 配置原样，不做全局 camera alias，防止
部署输入悄悄改变。

### 8.3 采样策略

第一版保持 valid-window-weighted 自然采样。实现时额外记录：

- 每个 task 的 episode 数；
- 每个 task 的有效 17 帧窗口数；
- 每个 suite 的窗口占比；
- 一个 epoch 的总 sample 数。

这能明确回答训练中的任务分布。后续如果启用 balanced sampler，应作为新配置，
不能静默改变第一版结果。

当前 `BaseLerobotDataset` 的长度接近 frame 数，episode 尾部会通过 padding 形成不完整
17 帧窗口；`skip_padding_as_possible` 只是随机重试，并不能定义确定的 valid-window
数据集。第一版应在构建索引时只保留 `start <= episode_length - 17` 的窗口，使训练样本、
stats 的 `all valid windows` 和 epoch 定义完全一致。训练中不应随机把尾部 padding sample
替换成另一个 task 的样本。若后续确实要训练 padding chunk，必须作为单独实验显式开启。

### 8.4 固定验证集与防止 replay 泄漏

验证不能直接令 `val_ds=train_ds`，也不能每次随机抽不同样本。需要固定：

- 一小组确定的 validation windows；
- diffusion timestep 和 noise seed；
- validation sample 顺序；
- 每次验证的样本数。

更关键的是，Plus 中多个扰动 replay 可能来自同一条原始专家轨迹。正式 train/val split
必须按 `source trajectory ID` 分组，而不是随机按 episode 切分，否则同一动作轨迹的不同
视觉扰动可能同时出现在 train 和 val 中，导致验证指标虚高。若合并发布版无法恢复这个 ID，
固定验证只能称为训练健康度监控，不能解释为独立泛化验证；最终结论以 10,030-task 在线
评测为准。

## 9. 训练前 smoke test

正式训练前按以下顺序验证：

### 9.1 Dataset-only test

固定若干 sample index，检查：

```text
video:          [B,3,17,224,448]
action:         [B,16,7]
context:        [B,L,D]
context_mask:   [B,L]
Rothko canvas:  [B,3,17,224,448]
```

保存一组可视化，确认：

- 左侧是 front，右侧是 wrist；
- RGB 时间连续；
- Rothko frame 0 是当前 EE；
- Rothko frames 1..16 对应连续 16 个 expert target；
- 横向两个 Rothko tile 完全一致；
- padding mask 与 episode 尾部一致。

因为第一版索引只包含完整窗口，正常训练 sample 的三种 padding mask 应全为 false；这里
保留 padding 检查，是为了验证 reader 在显式边界测试时不会跨 episode，而不是允许 padding
样本混入正式训练。

### 9.2 Codec roundtrip test

对多个 suite/task 做：

```text
absolute EE/action
  -> Rothko encode
  -> Rothko decode
  -> absolute target
  -> normalized OSC action
```

测试需要覆盖普通旋转、接近 pi 的旋转、gripper open/close 和接近 stats 边界的
样本。

除数学 roundtrip 外，还要增加语义测试：随机抽取 action，在同一模拟器/controller 配置中
执行一步，确认 xyz、rotation 和 gripper 的运动方向与转换后的 target 一致。否则一个内部
完全可逆、但符号或坐标系整体错误的转换仍可能通过 roundtrip。

### 9.3 一步训练测试

执行：

```text
1 train step
1 deterministic validation step
1 checkpoint save
checkpoint reload
同一 validation sample 再跑一次
```

检查 loss 中至少包含：

```text
loss_total
loss_rgb
loss_rgb_raw
loss_raymap
loss_raymap_raw
```

并确认 checkpoint 内记录：

- `raymap_representation=libero_rothko`；
- Rothko config；
- Rothko stats fingerprint；
- full finetune 配置；
- attention mask mode；
- VAE 路径或原始 VAE 标识。

还应记录数据集 repo revision/manifest hash、side-channel schema version、有效窗口数、
validation split fingerprint 和代码 commit。checkpoint reload 后必须拒绝几何配置、Rothko
stats 或 side-channel schema 不匹配，而不是只打印 warning 后继续。

## 10. 正式训练

第一轮采用全参数微调：

```text
原始 Wan2.2 video DiT
原始 Wan2.2 VAE（冻结）
完整 `libero_plus_lerobot` 数据训练
W&B 独立 group: libero_plus_rothko
```

批量大小、梯度累积、epoch 数和保存间隔应在一步 smoke test 测出真实显存与
steps/epoch 后确定，不能直接照搬原始 LIBERO 的绝对 step 数。配置中应同时记录：

```text
per_device_batch_size
world_size
gradient_accumulation_steps
effective_global_batch_size
samples_per_epoch
steps_per_epoch
num_epochs
```

全参数 checkpoint 和完整 optimizer state 体积很大。启动前应根据单份 `weights`/`state`
实测大小计算全程峰值存储，并分别配置轻量 weights 保存与可恢复 state 保存；设置 retention
策略时只能删除已经有后继完整 state 的旧 checkpoint，不能把“有权重”误认为“可 resume”。

正式运行目录必须以 `libero_plus_` 开头，防止与原始 LIBERO run 混淆。

第一版应先按实际 valid windows 计算 `steps_per_epoch`，再按 epoch 设置 `max_steps`；不要
沿用原始 LIBERO 的绝对 8,270/43,400 steps。学习率 warmup 也应按总 optimizer steps 的
比例计算。训练恢复必须同时恢复 optimizer、scheduler、GradScaler/AMP、global step、epoch、
dataloader/sampler state 和 RNG state；仅加载 `weights/step_xxx.pt` 不算完整 resume。

## 11. LIBERO-Plus 评测适配

### 11.1 协议选择必须显式化

评测入口增加显式字段，例如：

```yaml
EVALUATION:
  protocol: zero_shot_original_libero  # 或 plus_finetuned
```

它至少控制输出命名和配置校验，不能靠 checkpoint 路径字符串猜测协议。

`zero_shot_original_libero` 要求：

```text
checkpoint: 原始 LIBERO 训练 checkpoint
Rothko config/stats: checkpoint 训练时的原始 LIBERO 版本
VAE: checkpoint 对应版本
environment/task registry: LIBERO-Plus
```

`plus_finetuned` 要求：

```text
checkpoint: 完整合并 Plus 数据训练 checkpoint
Rothko config/stats: Plus 训练版本
VAE: checkpoint 对应版本
environment/task registry: LIBERO-Plus
```

加载时继续执行 Rothko config/stats fingerprint 校验。协议 A 必须使用原始 LIBERO
checkpoint 对应的 Rothko stats；协议 B 必须使用 Plus `center_frac=0.6` 的 Rothko
stats。两种协议都不读取 `dataset_stats.json`。

### 11.2 独立入口

建议新增独立目录，而不是让原始入口依赖当前环境里恰好安装了哪个 `libero`：

```text
experiments/libero_plus/eval_libero_plus_single.py
experiments/libero_plus/run_libero_plus_manager.py
experiments/libero_plus/run_libero_plus_parallel_test.sh
experiments/libero_plus/summarize_results.py
```

可以复用通用 rollout/policy/视频工具，但输出路径、默认 suite、trial 数和汇总逻辑
必须独立。

不能照搬当前“每个 task 启动一个 Python 进程”的 manager。Wan2.2 约 5B 参数，如果对
10,030 个 task 每次都重新加载模型，启动时间和存储 I/O 会压倒真实 rollout。第一版必须
采用常驻 GPU worker：

```text
manager 创建固定 task manifest
  -> 每张 GPU 启动一个 worker
  -> worker 只加载一次 checkpoint/VAE/text encoder
  -> 顺序消费多个 Plus task，逐 task 重建/关闭 env
  -> 每个 task 完成后原子写 result
```

worker 需要支持 resume/skip-completed、单 task timeout、异常隔离和显存清理。单个环境创建
失败不能让已完成结果丢失，也不能被静默计为 policy failure。

manager 自身不要初始化 CUDA、tokenizer 或 MuJoCo，再用 `spawn`/独立 Python subprocess
启动 GPU worker；worker 内单进程运行一个 env，不照搬 Plus lifelong evaluator 的
`SubprocVectorEnv(env_num=20)`。这样既符合每 task 1 trial，也避免 CUDA/tokenizer 初始化后
再 fork 引起死锁或 `tokenizers` warning。调度不依赖 tmux，终端 Ctrl-C 时 manager 应向所有
worker 转发 SIGINT/SIGTERM，并保留已原子写完的结果。

### 11.3 官方评测规模

LIBERO-Plus 包含：

```text
spatial 2402
object  2518
goal    2591
10      2519
total  10030
```

官方建议每个扰动 task 运行 1 个 trial，因此默认值为：

```yaml
EVALUATION:
  num_trials: 1
```

不能沿用原始 LIBERO 的每 task 50 trials，否则会把 10,030 个扰动任务扩展成
501,500 次 rollout。

任务清单要保存完整 task name/BDDL 标识，不只保存 suite 内会随版本变化的整数 task_id；
同时记录 LIBERO-Plus git commit、assets/config fingerprint 和
`task_classification.json` hash。A/B 必须复用同一份 frozen manifest。

正式全量运行前先用计划中的 GPU 数完成一个包含七类扰动的小型计时 pilot，测得 env
创建、一次 model load、单次 rollout 和每 task 平均耗时，再估算完整 wall time。完整评测
应可按 manifest shard 分批运行；不同 shard 合并时必须检查没有 task 重复或遗漏。

### 11.4 虚拟任务和扰动

Plus 的一部分任务名带有 `_view_..._initstate_...` 等后缀，并不存在同名的物理
BDDL 文件。LIBERO-Plus 的 benchmark/env wrapper 会映射到基础 BDDL，再应用相机、
初始状态、noise 等扰动。

评测适配必须保留完整的 `task.bddl_file` 传入 `OffScreenRenderEnv`，不能提前自行
去掉后缀；是否映射、如何施加扰动由 Plus 环境处理。

初始化必须走 Plus benchmark 自己的 `get_task_init_states`/virtual-task 逻辑。不能沿用原始
LIBERO 的 base task 50 个 init state 后自行按 trial index 取模，否则 Robot Initial States
类别会被错误地还原成原始分布。

至少对七类扰动各选一个任务做单元 smoke test：

```text
Objects Layout
Camera Viewpoints
Robot Initial States
Language Instructions
Light Conditions
Background Textures
Sensor Noise
```

### 11.5 评测输入与控制

部署侧继续使用：

```text
agentview_image
robot0_eye_in_hand_image
```

并经过与训练一致的 224×224 resize、antialias、归一化后拼成 224×448。需要用
固定图像比较确认 Plus 环境的相机方向是否仍需当前的 180 度翻转。

动作链路为：

```text
predicted Rothko
  -> absolute EE target
  -> current absolute EE pose
  -> normalized delta OSC action
  -> optional gripper binarization
  -> env.step(action)
```

`absolute_target_to_normalized_action(..., clip=True)` 会把超过 controller 范围的预测裁剪到
`[-1,1]`。Plus 的 OOD 扰动更可能触发该路径，因此每个 task 需要记录 xyz/rotation 各维的
pre-clip value、clip count 和最大超界幅度。否则模型明显预测越界仍可能被安全层掩盖，且
无法区分表示问题与控制问题。

两种协议的第一轮评测均使用各自 checkpoint 对应的 VAE、`replan_steps=16`、
ensemble 关闭，作为最小变量基线。A/B 对比时 inference steps、replan、ensemble、
trial 分配和 simulator 版本必须一致。后续再单独比较 `replan_steps=8`、VAE decoder
微调或 action ensemble。

为了让 A/B 逐 task 配对，环境 seed、init-state index、diffusion seed 和 task 顺序都应由
稳定的 task identity 派生并写入 manifest；不能依赖 worker/GPU 调度顺序。评测必须记录
环境实际返回的 instruction，尤其不能在 Language Instructions 类别中退化成 base canonical
instruction。

### 11.6 评测步数协议：沿用当前 FastWAM

第一版已确定完全沿用当前 FastWAM 原始 LIBERO evaluator 的控制步数和等待逻辑：

```text
libero_spatial: 400 个模型控制步
libero_object:  400 个模型控制步
libero_goal:    400 个模型控制步
libero_10:      700 个模型控制步
num_steps_wait: 沿用当前 FastWAM 配置
num_trials:     每个 LIBERO-Plus task 1 次
```

LIBERO-Plus 仓库自带 lifelong evaluator 的 `5 settling no-op + 600 control steps` 不作为
第一版主协议。这样协议 A/B 与已有 FastWAM LIBERO 评测保持同一控制口径，只把每个 task
的 trial 数从 50 改为 Plus 要求的 1。

dummy/wait steps 不计入上述模型控制步，`replan_steps` 也不能让最后一个 action chunk 超过
对应 suite 的控制步上限。每个结果文件仍需记录实际 `num_steps_wait`、
`max_control_steps`、`replan_steps` 和 `num_trials`，防止后续结果混用。

成功判定调用 Plus 环境的 `check_success()`/`_check_success()`；不能把通用 Gym `done`、
rollout 达到 horizon 或 worker 正常退出当作成功。每个 control step 后检查一次，达到成功
立即结束；达到 max control steps 后再做最终检查并记录失败。

## 12. 结果保存与汇总

建议目录：

```text
evaluate_results/libero_plus/
  zero_shot_original_libero/<task-config>/
    <timestamp-or-exp-name>/...
  plus_finetuned/<task-config>/
    <timestamp-or-exp-name>/
      manager_config.yaml
      tasks.txt
      task_logs/
      failed_tasks.txt
      libero_spatial/
      libero_object/
      libero_goal/
      libero_10/
      summary_by_suite.json
      summary_by_category.json
      summary_by_difficulty.json
      summary_all.json
```

每个 task 至少记录：

```text
suite
task_id
task_name
language
base task
perturbation category
difficulty（若 metadata 提供）
seed/init-state identifier
success
rollout steps
checkpoint
VAE
Rothko stats fingerprint
controller clip count/max overflow
status: success / policy_failure / infra_failure / timeout
```

最终不仅输出总成功率，还必须按以下维度汇总：

- 四个 suite；
- 七类 perturbation；
- difficulty；
- 40 个基础任务；
- 全部 10,030 个任务。

在 A/B 都完成后，另存一份 paired comparison，逐完整 task identity 报告：

```text
zero-shot success
Plus-finetuned success
absolute success-rate delta
由失败变成功的任务数
由成功变失败的任务数
```

只有 checkpoint 之外的评测条件全部相同时，才允许生成这份 A/B 对比。

`task_classification.json` 是 category/difficulty 汇总的权威来源。

成功率的分母只包含完成有效 rollout 的 task；infra failure/timeout 必须单独报告并补跑，
不能按失败混入模型成功率，也不能从分母静默丢弃。paired comparison 使用稳定的完整 task
identity，而不是仅用可能在不同 suite/version 中重复的 `task_id`。

10,030 个 task 若全部保存 rollout 和 prediction video 会占用大量空间。默认关闭视频，只
保存所有 infra failure、可配置比例的 policy failure 和固定少量 success 样本；完整逐步数值
结果使用 JSONL/parquet。`EVALUATION.output_root` 必须可配置，空间不足时可直接写
`/mnt/nodestor/cfy/evaluate_results/libero_plus`，不能先写满 hwdata 再迁移。

## 13. 回归测试：确保不影响原始 benchmark

每次 Plus 改造后都要运行以下回归检查：

1. 在 `fastwam_libero` 环境导入的仍是原始 LIBERO；
2. 原始 `configs/data/libero_rothko_2cam224.yaml` 内容和解析结果不变；
3. 原始 LIBERO 固定 dataset sample 的 tensor/hash 不变；
4. 原始 LIBERO checkpoint 能加载并完成一个固定 rollout；
5. Plus checkpoint 无法意外引用原始 LIBERO Rothko stats；
6. 原始 LIBERO checkpoint 无法意外引用 Plus Rothko stats；
7. RoboTwin dataset、train 和 eval 的 smoke test 不受影响。

如果共享模块必须修改，应优先增加显式参数并保持旧默认值，而不是根据文件名或
字段是否存在自动猜测 benchmark。

## 14. 推荐实施顺序

按以下阶段执行，每一阶段通过后再进入下一阶段：

1. 建立 `fastwam_libero_plus` 环境、独立 `LIBERO_CONFIG_PATH` 并安装 assets；
2. 验证 Plus benchmark 的四个任务数、七类扰动和 init-state 逻辑；
3. 生成带版本 fingerprint 的 frozen 10,030-task manifest；
4. 完成常驻 GPU worker 的 LIBERO-Plus 独立评测入口；
5. 用原始 LIBERO checkpoint 对七类扰动做协议 A smoke/计时 pilot；
6. 运行或分片运行协议 A 的 10,030-task 零样本基线；
7. 下载完整合并的 `libero_plus_lerobot` 数据集并固定 Hub revision；
8. 完成 metadata schema、语言、类别、state/action/gripper 和时间对齐审计；
9. 恢复并保存 episode 的 suite/category/source-trajectory sidecar manifest；
10. 实现 Plus side-channel 转换并完成 dry-run；
11. 批量写入并 verify 全部 parquet/metadata；
12. 建立确定的 valid-window index 和按 source trajectory 分组的验证 split；
13. 用全部有效窗口计算 Plus Rothko stats；
14. 计算实际全部唯一 prompt 的独立文本 embedding cache；
15. 新增 Plus data/model/task 配置；
16. 去掉 video-only 路径对 processor dataset stats 的依赖；
17. dataset-only、语义方向和 codec roundtrip 测试；
18. 一步训练、固定验证、完整 state 保存、reload/resume 测试；
19. 执行协议 B 的 Plus 增强训练；
20. 用相同 frozen manifest 运行协议 B 的 10,030 tasks；
21. 补跑 infra failure/timeout 后生成协议 A/B paired comparison；
22. 做原始 LIBERO 和 RoboTwin 回归测试；
23. 根据结果决定是否补做 center_frac=0.6 控制组、微调 VAE 或调整训练参数。

## 15. 第一版完成标准

只有同时满足以下条件，才认为 LIBERO-Plus 支持完成：

- Plus 环境和原始 LIBERO 环境互不覆盖；
- 两套环境使用不同的 `LIBERO_CONFIG_PATH`，配置和 assets fingerprint 已记录；
- 完整合并数据集的 14,347 个 episode 可读取，且 frame/task 数匹配 metadata；
- 实际 LeRobot metadata schema 已被 reader/转换器支持并通过一致性检查；
- state/action/gripper 语义、符号和 action/observation 时间对齐通过 simulator 验证；
- side-channel 全量写入和 verify 无错误；
- video-only Rothko train/eval 不再读取 `dataset_stats.json`；
- Plus Rothko stats 只来自 Plus 数据；
- 全部有效窗口参与正式 Rothko stats 计算；
- 正式训练只采完整 17 帧窗口，不随机替换 padded sample；
- 数据集实际使用的全部唯一 prompt 均有独立 cache；
- 固定验证 split/noise/timestep 可复现，且已说明是否能按 source trajectory 防泄漏；
- 一步 train/val/save/reload 数值正常；
- checkpoint 保存数据、side-channel、validation、Rothko stats/config 和代码 fingerprint；
- 完整 state checkpoint 可真正恢复 optimizer/scheduler/RNG/dataloader 状态；
- 七类扰动均可正确创建环境并 rollout；
- manager 可枚举 10,030 个任务、默认每 task 1 trial，且每个 GPU worker 只加载一次模型；
- task 结果可原子保存、resume、分片合并，infra failure 不混入 policy failure；
- 汇总能输出 suite/category/difficulty/base-task/overall 成功率；
- 原始 LIBERO checkpoint 可在协议 A 下评测且继续使用原训练 stats；
- Plus checkpoint 可在协议 B 下评测且拒绝错误的原始 LIBERO Rothko stats；
- A/B 能在相同 10,030-task 清单上生成逐任务对齐结果；
- 原始 LIBERO 与 RoboTwin smoke test 无回归。

## 16. 后续可选实验

第一版完成后再按独立配置尝试：

1. 原始 Wan VAE 对比 Plus Rothko decoder 微调 VAE；
2. natural sampling 对比 suite-balanced/task-balanced sampling；
3. `replan_steps=16` 对比 8/4；
4. action ensemble 开关与 decay；
5. 各 perturbation category 定向微调或数据增强；
6. 四-suite 发布版与 14,347-episode 合并发布版的差异审计。

这些实验都应使用新的 task/config 名称，不覆盖第一版基线结果。
