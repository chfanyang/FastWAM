# 待补实验

用于记录待完成的实验、前置校验和结果。参数草案不代表已批准启动。

## 1. LIBERO 四个 suite 全绝对位姿全量微调

- **状态**：2026-09-20 正式全绝对缓存已由远端2卡切到GPU0/3/4/5/6/7六卡续算；训练和闭环评测尚未启动。
- **目标**：在 LIBERO spatial、object、goal、10 四个 suite 上，验证全绝对位姿 Rothko 表示的效果，并与已有方案在同等评测条件下比较。
- **表示**：RAY0 使用当前 EE 的绝对位姿，未来 16 步也使用绝对目标位姿；位置和朝向都采用全绝对表示。
- **模型**：Wan2.1 T2V 1.3B，全量微调 DiT；使用原始 VAE，冻结 encoder/decoder。
- **输入**：连续 17 帧，预测 16 步；双视角，每块 224×224，总画布 224×448。
- **归一化**：四个 suite 的全部训练数据，按当前 EE 与构造动作目标的 XYZ 极值、每侧加 1 cm 余量统计；center_frac=0.5。
- **动作语义**：沿用现有“当前 EE＋原始 action → 绝对目标”的构造；eval 时预测目标仍转换回 OSC 增量 action。零旋转命令下，构造目标可能不同于控制器内部目标，但未证明转换有误，不因此修改标签。

### 已有准备

- [训练配置](configs/task/libero_all4_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4.yaml)
- [评测配置](configs/sim_libero_all_absolute.yaml)
- [全绝对 codec](src/fastwam/representations/libero_rothko_all_absolute.py)
- [norm_stats 统计脚本](scripts/compute_libero_all_absolute_stats.py)
- [本机缓存／训练／评测统一入口](scripts/libero_all_absolute_workflow.py)（默认只打印命令，`--execute` 才运行）
- [本机操作说明](experiments/libero/ALL_ABSOLUTE_RUNBOOK.md)；[原始 VAE 小审计](scripts/audit_libero_all_absolute.py)
- stats：`data/libero_mujoco3.3.2/libero_all4_rothko_all_absolute_minmax_margin01_h16_224x448_centerfrac05.pt`
- 已完成 CPU 侧窗口时刻对应、位姿转换、直接 Rothko 编解码及相关回归检查。
- 本地审查记录：`evaluate_results/libero/all4_absolute_review_20260917/REVIEW_ZH.md`（实验产物被 Git 忽略）。

### 正式训练前待完成

- **必须先算新的全绝对缓存，再启动微调。** 顺序为：小规模重建与一致性校验 → 全量缓存 → 检查缓存完整性与身份 → 正式训练。
- **缓存脚本现状**：专用本机入口已准备，复用 [precompute_visual_action_latents.py](scripts/precompute_visual_action_latents.py)，通过新任务配置分别编码 RGB、Rothko；支持独立 256 窗口测速和 8 卡全量缓存。默认每 GPU 1 个进程、VAE batch8、每进程 DataLoader workers4；实际吞吐需先测速。原程序包含抽样在线／缓存逐位一致和固定噪声 loss 校验；2026-09-19 已在远端 A800 完成 256 窗口 GPU 校验，RGB/Rothko latent 与固定噪声 training_loss 逐位一致。
- **独立缓存目录**：`data/libero_all4_rothko_all_absolute_2cam224_wan21_bf16_h16_latents`；不覆盖旧缓存。正式全量缓存已在远端 A800 GPU0/3 启动：tmux `libero_all4_absolute_cache_2gpu`，每卡1进程、batch8/workers4、每shard1024窗口。日志 `evaluate_results/libero/all4_cache_2gpu_launch_20260919_164305/cache.log`。中断后可按相同契约与shard大小显式切8卡续算，不同时运行两组写进程。

- [x] 原始 VAE 重建审计：四-suite 共 74 个普通、极值、大旋转及尾部窗口；报告 `evaluate_results/libero/preflight_all4_20260919/REPORT.md`。
- [x] 256 窗口小缓存：抽查完整 RGB/Rothko latent（含 RAY0）与在线编码逐位一致，固定噪声 training_loss 一致；这是抽样检查。
- [x] 文本：发现旧 BF16 批量缓存与单条在线编码存在差异；已新建 `data/text_embeds_cache/libero_wan21_bf16_batch1`，40/40 在实际 `fastwam_libero` 环境逐位一致，最大绝对差为 0。实验 1 与继承该配置的实验 2 共用新缓存；原缓存和其他实验不变。报告 `evaluate_results/libero/text_batch1_20260919/verify.json`。
- 文本准备脚本：[prepare_libero_single_text_cache.py](scripts/prepare_libero_single_text_cache.py)，固定 batch=1，记录编码器/tokenizer SHA256，拒绝覆盖已有目录；后续换硬件、环境或权重后运行 verify。无需因此重算视觉 latent。
- [x] 用户已确认训练参数与 checkpoint 选择：沿用之前 Wan2.1 四-suite 训练，改为 8 卡 batch4/GA4；每 3000 步保存权重和完整 state，结束也保存；评测使用最终 checkpoint。
- [ ] 锁定模型、VAE、stats、数据、代码和配置指纹，计算独立的全绝对 latent 缓存；不直接复用相对／混合表示的 Rothko latent。
- [ ] 核对基线的模型、VAE、样本数、horizon、padding、decoder、seed 和 replan 设置，确定可比口径。

### 已确认参数与执行约束

- 训练：8 卡，每卡 batch 4，GA4，有效 batch 128；10 epoch，预计 21,700 个 optimizer steps（启动时按实际数据长度核对）。
- 学习率：1e-4；线性 warmup 占总步数 5%，预计 1,085 步；随后 cosine 降至 1e-6。
- 其他：BF16，关闭 gradient checkpointing；weight decay=0.01，梯度裁剪=1.0；RGB/Rothko loss 权重为 1/5。
- 训练内验证：每 1000 步一次，沿用 4 个窗口、20 次去噪。
- 保存：每 3000 个 optimizer steps 保存权重和完整 state，训练结束再保存；不额外启用每次验证前保存。预计保存 3000、6000、9000、12000、15000、18000、21000、21700 步。
- 正式评测使用最终 checkpoint，预计 step 21700；不根据闭环结果反复挑选 checkpoint。legacy decoder，anchor=0，20 次去噪，replan8，关闭 action ensemble，seed=42，每任务 50 个场景。
- 新全绝对方案继续使用原始 VAE。之前相对方案已有评测使用微调 VAE step7498，因此比较时须标明 VAE 差异，不能称为完全同条件。
- 实验 1 只进行一次正式 DiT 训练和一次 original VAE 闭环评测；先完成小规模链路校验。实验 3 是新增的微调 decoder 对照，复用同一 DiT，并额外评测一次；original VAE 结果直接复用实验 1。
- 当前配置没有独立 held-out 切分，训练内验证不能作为泛化结果。
- 不影响正在运行的 VLABench 训练及已有 LIBERO、LIBERO-Plus 实验。

### 本机代码准备与操作步骤（2026-09-18）

已按 `/mnt/hwdata/cfy/FastWAM` 的现有环境准备，尚未启动缓存、训练或评测；暂不做迁移包。

| 代码入口 | 用途 |
|---|---|
| `scripts/libero_all_absolute_workflow.py` | CPU 前置检查、VAE 小审计、8 卡缓存测速、全量缓存、缓存完成检查、正式训练、正式评测 |
| `scripts/audit_libero_all_absolute.py` | 四-suite GT → 原始 VAE → 动作重建审计，可传入指定窗口清单 |
| `tests/test_libero_all_absolute_workflow.py` | 检查默认不启动任务、测速与正式缓存分离、评测参数显式传递 |
| `experiments/libero/ALL_ABSOLUTE_RUNBOOK.md` | 更详细的运行说明；本文件保留实验决策和主要操作顺序 |

**默认只准备命令，不会启动 GPU 任务。** 以下 `audit / benchmark / cache / train / eval` 命令确认后加 `--execute` 才真正执行；`check / check-cache` 直接执行 CPU 检查。

```bash
cd /mnt/hwdata/cfy/FastWAM
PY=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python

# 1. 检查已确认的参数、四-suite 数据 metadata、stats 和原始 VAE 身份。
"$PY" scripts/libero_all_absolute_workflow.py check

# 2. 原始 VAE 小审计；默认每个 suite 首／中／末窗口。
"$PY" scripts/libero_all_absolute_workflow.py audit --gpus 0

# 3. 8 卡、256 个窗口的小缓存测速，写入独立诊断目录。
"$PY" scripts/libero_all_absolute_workflow.py benchmark

# 4. 全量缓存，必须在正式训练之前完成。
"$PY" scripts/libero_all_absolute_workflow.py cache

# 5. 检查完整缓存的身份、覆盖、shard 大小及一致性验证记录。
"$PY" scripts/libero_all_absolute_workflow.py check-cache

# 6. 正式训练：8 卡 batch4/GA4，10 epoch，每3000步及结束保存。
"$PY" scripts/libero_all_absolute_workflow.py train

# 7. 替换成实际完成的 run 路径，使用最终 step21700 做四-suite 评测。
"$PY" scripts/libero_all_absolute_workflow.py eval --run /absolute/path/to/completed_run
```

- 缓存默认每 GPU 1 个进程，VAE batch8、每进程 DataLoader workers4，可用 `--batch-size`、`--num-workers` 测速调整；这与训练 batch/GA 不同。
- 正式缓存约需 130 GiB 原始 shard 空间，另预留临时文件和 checkpoint 空间。测速缓存只覆盖开头 256 个窗口，不能用于正式训练。
- 默认 GPU 为 0–7；真正执行前检查 GPU 占用，不停止其他任务。新建输出目录，不覆盖旧结果，不隐式续跑。
- 执行记录与日志：`evaluate_results/libero/all4_absolute/launch_*`；训练使用独立的新 run 目录。正式运行可放在独立 tmux 会话中。
- 本轮已通过：CPU 配置／数据／stats／VAE 身份检查、8 项相关测试、评测 manager 与 worker 的实际配置解析。
- 更新（2026-09-19）：上述74窗口 VAE 审计、256窗口小缓存以及40条文本一致性检查已完成，见本节前置检查记录；正式全量缓存和8卡训练仍未启动。
- 本轮验证记录：`evaluate_results/libero/all4_absolute_local_prepare_20260918/`。

### 结果待填写

- Run / checkpoint：待填写。
- 四个 suite 分别成功率、总成功率及统计分母：待填写。
- 与同条件基线的差异：待填写。
- 失败样例、训练耗时与评测耗时：待填写。

## 2. LIBERO Goal：动作表示与 attention mask 消融

- **状态**：正式计划保留 Latent-Repeat、两种 attention mask 和完整基线。Flatten 与 Shuffled 已移除，不再为其生成训练缓存、训练 DiT 或闭环评测；已完成的离线 VAE 诊断仅作为记录保留。
- **范围**：所有组均只使用 LIBERO Goal suite 训练，并只在 Goal suite 测试。
- **数量**：三个消融加完整方案基线，共四组 DiT 训练；另加一组复用基线 DiT 的微调 VAE decoder 对照，合计五组闭环评测。
- **目的**：分别检验 Rothko 空间表示与未来 RGB/action token 之间 attention 连接方式的作用。

### 五组设置

| 编号 | 设置 | 相对完整方案的改动 |
|---|---|---|
| 2.1 | **Latent-Repeat** | 将归一化后的数值动作向量重复铺成 latent-frame tensor，直接作为动作 latent，绕过动作侧 visual VAE；RGB 路径保持一致。 |
| 2.2 | **Decoupled Mask** | 保留未来 RGB 与动作的联合监督，但在去噪期间禁止未来 RGB 与未来动作 token 双向直接互相 attention。 |
| 2.3 | **Bidirectional Mask** | 保留 Rothko 表示，允许未来 RGB 与 Rothko latent 之间双向、无额外限制的相互 attention。 |
| 2.4 | **Rothko + 当前 attention mask（完整方案基线）** | 使用当前 Rothko 表示与 `rgb_then_raymap_block_causal` mask，作为以上三组的共同参照。 |
| 2.5 | **2.4 + 微调 VAE decoder** | 复用 2.4 训练得到的最终 DiT checkpoint，仅在测试时换成微调后的 VAE decoder；与 2.4 对比 decoder 微调的作用。 |

Latent-Repeat 已对照本地 `third_party/cosmos-policy` 代码：采用相同的归一化数值 chunk 展开、repeat 填充 latent、预测后对重复 chunk 取均值的核心方法；动作时序布局适配 Wan，具体见下方。属于沿用该编码思想，并非完整复现 Cosmos Policy 的 backbone、序列布局和训练流程。

### 已确认的共同设置

- **四组均采用全绝对位姿、Wan2.1 T2V 1.3B，全量微调 DiT。**
- **训练参数沿用实验 1 的四-suite 设置，仅将训练数据换成 LIBERO Goal**：10 epoch，8 卡，每卡 batch4、GA4，有效 batch128；LR=1e-4，线性 warmup 5%，随后 cosine 到 1e-6。
- BF16，关闭 gradient checkpointing；weight decay=0.01，梯度裁剪=1.0，训练 seed=42；RGB/action loss 权重：**Latent-Repeat（2.1）按用户确认使用 1/1，不额外上调动作 loss；2.2–2.4 沿用 1/5**。
- 每 1000 个 optimizer steps 验证，每 **2000 步**保存权重和完整 state，结束也保存；闭环评测使用最终 checkpoint。
- 总步数按 Goal 数据量重新计算，不沿用四-suite 的 21,700 步。按当前 52,895 个窗口、有效 batch128 估算，每 epoch 414 步，10 epoch 共 **4,140 步**，warmup **207 步**；预计保存 step2000、step4000 与最终 step4140，启动时核对实际步数。
- **2.4 启动准备（2026-09-20，ainode06）**：用户已授权使用 GPU0–7，W&B 已登录。四-suite 完整缓存通过 `check-cache`；Goal 缓存由 `scripts/extract_libero_goal_absolute_cache.py` 精确复制全量缓存中的 `[120538,173433)`，不重新全量编码。复制后仍须运行原缓存脚本，检查首／中／尾 latent 与固定噪声 training_loss 逐位一致，再生成 `_SUCCESS`，之后才启动训练。参数及源码指纹记录在远端 `evaluate_results/libero/goal24_launch_20260920/`。此次仅训练 2.4，不启动其他消融或闭环评测。
- **2.4 启动记录（2026-09-20 02:50 UTC）**：Goal 缓存 52 个 shard、52,895 窗口、24.72 GiB 已完成；索引 `[0,26447,52894]` 的 RGB／Rothko latent 与在线编码逐位一致，固定 seed123456 的 training_loss 逐位一致（0.9586604237556458）。远端 tmux `libero_goal24_absolute_train`；run `runs/libero_goal_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4/2026-09-20_02-48-42`；启动与训练日志 `evaluate_results/libero/goal24_launch_20260920/train.log`。参数沿用上述已确认配置，8卡、10epoch、每2000步及最终保存权重和完整state。
- W&B 已确认在线同步：[bgkr0dm0](https://wandb.ai/chfanyang-xi-an-jiaotong-university-/fast-wam/runs/bgkr0dm0)。实际日志确认 world_size=8、GA4、full DiT、`max_steps=4140`。训练／验证底层 dataset 均为 Goal 52,895 窗口，定期验证仍是既定固定4窗口，不是独立 held-out 泛化评测。
- **2.4 正式评测（2026-09-20 11:13 UTC，ainode06）**：训练已到 step4140 正常结束，最终权重与完整state均保存。使用 `step_004140.pt` + 原始冻结 VAE，legacy/anchor0、seed42、replan8、20步去噪、不使用action ensemble，Goal 10任务×50场景，共500场景；8卡、每卡最多2任务，10任务同时运行。EGL双相机256×256渲染检查通过，manager/worker配置一致。启动目录 `evaluate_results/libero/goal24_eval_launch_20260920_111156`，结果目录 `evaluate_results/libero/goal24_step4140_original_replan8_ep50_20260920_111156`，外层tmux `libero_goal24_eval_manager`。仅启动评测，结果待完成后填写。
- **2.1–2.4 均不微调 VAE，统一使用 original Wan2.1 VAE 做测试。** 需要 VAE 的路径在 DiT 训练／缓存阶段也使用同一原始冻结权重；不使用 VAE7498 或其他微调 VAE。**2.5 是单独新增的 decoder 微调对照**，定义见下方，不改变 2.1–2.4 的设置。
- Latent-Repeat 已确认采用下述 Wan 适配方案，绕过动作侧 VAE；RGB 路径仍使用原始冻结 VAE。
- **闭环测试协议已确认：沿用此前 Goal suite 的既有协议，不另设新协议。** replan8、20 次去噪，每任务 50 个场景、seed42，关闭 action ensemble；各组使用相同场景与其他既有控制设置。2.1–2.4 使用 original VAE，2.5 使用微调 decoder；各表示的动作读取方式随消融定义确定。

### 2.5：复用 2.4 DiT + 微调 VAE decoder（新增）

- **2026-09-20 启动（ainode06，13:06 UTC）**：用户要求停止2.4闭环评测并启动8卡Goal-only VAE微调。停止时保留477个已完成场景、420次成功，另23个未完成，不能作为完整500场景成绩；快照 `evaluate_results/libero/goal25_vae_launch_20260920_130306/eval_at_stop.json`。Goal缓存契约通过，60个训练/验证、首/末完整/末尾padding probe均与原始BF16 encoder逐位一致。tmux `libero_goal25_vae_2ep`；启动记录 `evaluate_results/libero/goal25_vae_launch_20260920_130306/`；输出 `runs/libero_goal_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2/`。沿用本节既定参数和原四-suite min/max stats，不修改范围或数据。
- 启动确认：world_size=8，413训练/20验证episode，50,445训练窗口，1,578总步数，warmup78，conv2＋decoder共73,295,603个可训练参数；step10 loss=0.039871，初始约7.43秒/step。W&B新run：[i9lduff0](https://wandb.ai/chfanyang-xi-an-jiaotong-university-/fast-wam/runs/i9lduff0)。

- 固定使用 **2.4 的最终 DiT checkpoint**，不新增一次 DiT 训练。
- 对同一 Goal 全绝对 Rothko 表示微调 VAE decoder，encoder 保持原始冻结权重；测试时搭配该 VAE 与 2.4 的 DiT。
- 仅用 Goal 训练数据进行 decoder 微调，不用闭环测试结果或测试场景数据拟合。
- 闭环协议与 2.4 完全相同，包括测试场景／seed、replan8、20 次去噪和动作处理；对比 original VAE 与微调 decoder 的差异。
- **用户已确认 decoder 微调沿用之前 LIBERO Wan2.1 的做法与参数。** 依据为 `runs/libero_rothko_vae_decoder_wan21_all4_centerfrac05_h16_bs2_ga8_lr1e-5_ep2/experiment_config.json` 和 `training_config.json`；本组仅换成 Goal 数据、全绝对 Rothko codec 和对应 stats，不直接复用旧相对表示或 VLABench 的微调 VAE。
- 从 original Wan2.1 VAE 初始化，冻结 encoder，训练 **conv2＋decoder**；输入为 **GT Rothko → 原始冻结 encoder 得到的 latent**，重建 GT Rothko，不改用 DiT 生成 latent。
- 训练参数沿用：**2 epoch、LR=1e-5、cosine、warmup 5%、最低 LR 比例 0.01、weight decay=0、梯度裁剪=1**；FP32 master 参数／梯度／AdamW 状态，BF16 autocast，分片优化器。
- **本组按用户确认使用 8 卡微调 decoder，每卡 batch2、GA4，有效 batch64**；历史配置为 4 卡、每卡 batch2、GA8，因此通过减半 GA 保持有效 batch 与其余训练参数一致，不套用 DiT 的 batch4/GA4。
- Loss 沿用中心／方向／夹爪区域 masked L1，权重 **1/1/1**；使用本次全绝对表示的 GT 图与匹配区域。
- 数据划分沿用每任务留出 2 个 episode、split_seed=20260801；验证 `uniform_per_episode`、每 episode 10 窗口、eval_batch1、eval_seed12345；训练 seed42。
- 保存与验证沿用 decoder 原规则：每400步保存 latest，每800步保存 step checkpoint、导出 VAE并验证，epoch 结束验证，训练结束保存并导出；测试用最终导出的 VAE。这里不套用 2.1–2.4 的 DiT 每2000步保存规则。
- 用户已确认包含尾部 edge-repeat padding；补齐帧参与重建 loss，验证动作误差排除补齐步。Goal 总50,445个训练窗口，789步/epoch，2epoch共1,578步，warmup78步。
- 状态：已列入计划，尚未启动 decoder 微调或评测。

### Goal 全绝对 decoder 代码准备（实验 2.5，2026-09-19）

- 已准备独立配置：`configs/vae/libero_goal_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json`；入口：`scripts/prepare_goal_absolute_decoder.py`；操作说明：[GOAL_ABSOLUTE_DECODER_RUNBOOK.md](experiments/libero/GOAL_ABSOLUTE_DECODER_RUNBOOK.md)。仅准备和CPU校验，未启动GPU审计、缓存、训练或评测。
- 按历史split、用户新增的尾部edge-repeat规则，固定训练413个episode、50,445窗口；验证20个episode、200窗口。8卡有效batch64，789步/epoch，2epoch总1,578步，warmup78步。
- 固定训练/验证episode、验证starts、训练缓存索引及身份：`evaluate_results/libero/goal_absolute_decoder_preparation_padding_20260919/manifest.json`。
- 复用待生成的Goal全绝对DiT缓存时，先检查数据契约、原始VAE哈希、stats、BF16精度、latent尺寸、完整性和缓存验证标记，再做60个覆盖每任务训练/验证首个、最后完整和最终padding窗口的GPU逐位一致性检查。当前缓存不存在，正式训练入口要求匹配的审计报告。
- 保存规则补充：历史trainer在epoch结束也保存latest，因此预计latest恢复点为400、789、800、1200、1578；独立权重checkpoint与VAE导出为800、最终1578；验证为789、800、1578。独立历史checkpoint不含优化器，续训使用latest。
- 验证使用原始冻结encoder缓存和BF16部署精度decoder、全绝对legacy/anchor0读取。旧相对目标/解码不参与本入口。公共训练文件与正在运行的VLABench任务未修改。
- 已新增尾部重复、当前RAY0及验证padding过滤检查，共6项CPU检查；实际缓存逐位一致、GPU训练/验证/恢复仍待缓存生成后测试。GPU阶段默认只打印命令，不自动开跑。

### 控制变量与实现前需锁定的细节

- 表示消融（2.1）沿用基线 mask；mask 消融（2.2–2.3）沿用基线 Rothko 表示。
- 四组使用同一 Goal 数据范围、窗口采样、horizon、位姿目标语义、归一化物理范围、RGB 输入与条件信息。
- 四组统一模型初始化、训练预算、有效 batch、优化器、学习率计划、训练 seed，以及最终 checkpoint 选择规则，按上面的已确认设置执行。2026-09-22 用户更新：Latent-Repeat 同样使用 action loss 权重5，RGB/action=1:5。
- 四组闭环测试场景、seed、每任务 episode 数、replan、去噪步数和夹爪处理均沿用上述已确认的既有 Goal 协议；报告 Goal 各任务及整体成功率与分母。
- Latent-Repeat 的位姿语义、数值铺排和读取规则按下述定义执行。
- Latent-Repeat 的向量、布局和读取方案已确认，见下方；实现时核对训练目标、当前条件和推理读取使用同一规则。
- 两种 mask 的连接定义已确认，见下方矩阵；实现时需检查每层都保持条件 token 不读取未来，避免 Decoupled 经条件 token 间接交换信息。
- RGB 与动作均保留监督；Latent-Repeat 使用 RGB/action 权重 1/5，保留 Wan 基线扩散目标与分模态求均值方式，不额外做 latent 方差匹配。其动作 latent 尺度与 VAE latent 不同，不能直接用不同表示间的原始 loss 大小评价优劣。

### Latent-Repeat：Cosmos 思想的 Wan 适配（已确认）

- **Loss 已更新（2026-09-22）**：RGB/action 权重为 **1/5**，即 `loss_rgb + 5 * loss_action`，与本轮基线一致；保留 Wan 的扩散训练目标、噪声／时间权重和分模态求均值方式，不额外做 latent 方差匹配。参考本地 Cosmos Policy 默认 `action_loss_multiplier=1`，但不声称完整复现其 EDM loss 或整体聚合方式。

- 保留基线 `[rgb0–4 | ray0–4]` latent 布局和 attention mask，不缩减未来动作 latent 帧数；每个动作 latent 帧形状为 `[16,28,56]`，共 25,088 个标量。
- 2026-09-22 本轮改以 absolute-RAY0＋未来相对目标 Wan2.1 为基线。使用8维向量 `[x,y,z,qw,qx,qy,qz,gripper]`：RAY0 XYZ按四-suite绝对 observation min/max映射；未来XYZ先变换为 `R0^T(p_target-p0)`，再按旧共享Q99.95对称 bounds映射到 `[-1,1]`；未来旋转为 `R0^T R_target`；四元数归一化为单位四元数；gripper 从 `[0,1]` 映射到 `[-1,1]`。不再套用 VAE latent 均值／标准差变换。
- 四元数逐帧统一符号：令 `qw >= 0`；若 `qw == 0`，令 `(qx,qy,qz)` 中第一个非零分量为正，不采用随窗口变化的符号选择。该约定不保证跨180°时分量连续。读取后重新归一化，范数接近零时须有数值保护并记录退化样本。
- `ray0`：当前 EE 位姿和当前 gripper，8 维向量 repeat 3,136 次，按 CHW reshape 为 `[16,28,56]`；作为干净条件，不包含未来动作。
- `ray1`：未来动作第 1–4 步；`ray2`：第 5–8 步；`ray3`：第 9–12 步；`ray4`：第 13–16 步。这里是显式动作分组，不声称等同于 VAE 的时间感受野。
- 每个未来 latent 帧先将 `[4,8]` 动作按时间顺序展开成 32 个标量，repeat 784 次，再按 CHW reshape 为 `[16,28,56]`。latent 帧直接整体填充，不执行图像 HWC 排列或双 tile 复制。
- 动作侧不经过 VAE encoder/decoder；RGB 仍沿用基线 VAE 编码。DiT 训练、噪声与采样保持 Wan 原有流程，不移植 Cosmos 的扩散参数化。
- 读取：分别展开生成的 ray1–4，每帧 reshape 为 `[784,4,8]`，沿 784 份重复取均值，再按时间拼回 `[16,8]`；按相同规则反归一化 XYZ、归一化四元数、恢复 gripper，最后沿用现有绝对目标到 OSC action 的转换。
- 源码依据：`cosmos_policy/models/policy_text2world_model.py::replace_latent_with_action_chunk` 和 `cosmos_policy/experiments/robot/cosmos_utils.py::extract_action_chunk_from_latent_sequence`。Cosmos 原实现把整个动作 chunk 铺入一个 latent 帧，必要时编码截断尾部、读取丢弃不完整重复；本实验 8 维／4 步尺寸正好整除，无需截断。

### 已归档：Action Representation 原始 VAE round trip 小诊断（2026-09-19）

- 已在本机完成，未启动消融训练；独立脚本：`scripts/audit_goal_representation_roundtrip.py`。
- Goal 10 个任务各随机抽 2 个 episode，每个随机取 1 个窗口，共 20 窗口，seed42；三组共用样本。原始 Wan2.1 VAE、BF16、17 帧、224×448、batch1；排除 padding，前8步156个有效目标、全部16步302个有效目标。
- Flatten-Resize 按已确认的 HWC 数值重复；Shuffled 在单 tile 内固定置换 RGB 像素后复制两份，解码后分别逆置换。置换可逐位还原。
- 前8步 GT→VAE→动作误差如下（均值 / P95）：

| 表示 | 平移 L2（mm） | 旋转（°） | 夹爪二值错误率 |
|---|---:|---:|---:|
| Rothko | 9.453 / 18.279 | 0.896 / 1.715 | 0% |
| Flatten-Resize HWC | 256.837 / 454.092 | 113.359 / 147.894 | 12.82% |
| Shuffled Rothko | 281.774 / 406.751 | 49.855 / 102.542 | 0.64% |

- 全16步平移／旋转均值：Rothko 9.117 mm / 0.852°；Flatten 267.337 mm / 111.513°；Shuffled 283.210 mm / 44.703°。
- 三组直接编解码误差接近零；仅 BF16 像素量化时平移均值均约0.580 mm，旋转均值 Rothko/Shuffled约0.00062°、Flatten约0.0364°。这批样本的大幅误差出现在 VAE round trip，不能归因于 DiT 未学好。
- 这是小样本 GT 重建诊断，并非闭环成功率；表示消融同时受到原始冻结 VAE 对空间组织的适配性影响，不能单独据此推断 DiT 学习难度。现已移除这两组正式训练和闭环评测，仅保留此离线诊断。
- 明细、采样窗口及固定置换：`evaluate_results/libero/representation_roundtrip_20260919/`（report.json、rows.json、selection.json、permutation.pt）。诊断 CUDA 峰值 allocated 1.235 GiB；现有 VLABench VAE 训练持续推进。

### 已归档：Flatten-Resize CHW 补充诊断（2026-09-19）

- 历史加测 CHW；HWC 和 CHW 均不再列入正式训练计划。与上述20窗口完全相同，仅把单tile的 repeat 后 reshape 改为 `[3,224,224]`，读取同步按CHW展开；仍复制双tile、使用原始BF16 VAE。
- 独立逐元素布局及直接往返校验通过；实际有效16步直接解码最大平移0.000258 mm、旋转0.0000248°，夹爪误差0。
- 前8步 VAE round trip：平移均值/P95 **215.920 / 396.150 mm**，旋转 **92.829 / 144.611°**，夹爪二值错误率 **16.03%**。
- 全16步：平移均值/P95 **215.773 / 382.248 mm**，旋转 **90.494 / 123.257°**，夹爪二值错误率 **17.55%**。
- CHW 的平移/旋转误差低于这批样本的HWC，但仍严重失真；夹爪二值误差更高。改变铺排顺序没有解决原始VAE的重建问题。
- 报告：`evaluate_results/libero/representation_roundtrip_chw_20260919/report.json`；运行命令在原诊断脚本上加 `--chw-only`。当时暂停的100 episode方案后续已由50 episode诊断替代。

### 已归档：50 个完整 episode 三表示 round trip（2026-09-19，已完成）

- 用户确认：Goal 每任务随机选5个完整episode，10任务共50个，seed42；遍历所选episode的所有frame starts，共 **6,472窗口**。本次三组为 **Rothko、Flatten-Resize CHW、Shuffled Rothko**，不是闭环policy评测。
- 原始Wan2.1 VAE、BF16、batch1；每组均测直接编解码、BF16像素量化、VAE round trip。前8/16步排除padding，汇总全体及每任务均值/P95/max、夹爪误差，记录四元数退化次数。
- 本机8卡各一个诊断进程，按完整episode分片，每卡784–869窗口，PyTorch显存分配上限15%；与VLABench VAE训练共享GPU，可能降低训练吞吐，不修改或停止训练。
- 脚本：`scripts/audit_goal_representation_episodes.py`；tmux：`goal_repr_roundtrip_50ep`。
- 输出：`evaluate_results/libero/goal_repr_50episodes_chw_20260919/`，采样在manifest.json，逐窗口9条记录保存在各shard的episode JSONL，progress.json记录进度，结束自动验证全部窗口/三表示/三链路覆盖后生成report.json及逐任务汇总。
- 此处比例以重叠窗口中的有效预测步为分母，不是独立物理时刻比例或闭环成功率。此前100-episode方案已由本次50-episode方案替代。

- **完成结果**：50 episode / 6,472窗口全部通过覆盖校验，8卡分片最长运行37.92分钟；前8步50,376、全16步97,552个有效目标，无四元数退化。
- 前8步（平移均值/P95 mm；旋转均值/P95 °）：Rothko **9.784/19.487；0.840/1.646**；Flatten CHW **246.962/518.505；86.621/135.263**；Shuffled **287.994/508.096；49.030/99.043**。
- 全16步均值：Rothko **9.636 mm / 0.799°**；Flatten CHW **244.668 mm / 86.641°**；Shuffled **286.869 mm / 47.079°**。前8步夹爪二值错误率分别 **0%、17.09%、0.35%**。
- 三组直接编解码最大误差均小于 **0.000366 mm / 0.000044°**，夹爪误差0。严重VAE重建误差在10个任务上均存在。
- 完整总体/逐任务结果：`evaluate_results/libero/goal_repr_50episodes_chw_20260919/SUMMARY.md` 与 `report.json`。未启动任何消融DiT训练或闭环评测。

### 两种 attention mask（已确认）

令 C 为当前 RGB0 与 RAY0 条件，V 为未来 RGB tokens，A 为未来 Rothko/action tokens。箭头表示信息从来源传向读取方；V、A 各组内部均全连接，C 内部保持当前实现的相互可见。文本 cross-attention 保持不变。

**Decoupled Mask：** V 和 A 都读取 C，但不能互相读取；C 不读取任何未来 tokens，避免跨层通过 C 间接交换信息。保留两种模态的预测与监督。

```mermaid
flowchart LR
    C["C：当前 RGB0 + RAY0"] --> V["V：未来 RGB"]
    C --> A["A：未来动作"]
    V -->|组内可见| V
    A -->|组内可见| A
```

**Bidirectional Mask：** 在相同条件连接上开放 V 与 A 的双向连接；两组未来 tokens 合起来全连接。C 仍不读取未来 tokens。

```mermaid
flowchart LR
    C["C：当前 RGB0 + RAY0"] --> V["V：未来 RGB"]
    C --> A["A：未来动作"]
    V --> A
    A --> V
    V -->|组内可见| V
    A -->|组内可见| A
```

当前完整基线为 V → A 的单向信息流（动作可读取未来 RGB，未来 RGB 不读取动作）。因此 Decoupled 去掉这一方向，Bidirectional 在基线上再加入 A → V。这里改变的是联合 token 序列的 self-attention mask，不是另加一个 cross-attention 模块；每层、每次去噪保持该连接约束。

用户已确认以下精确矩阵。行是 Query（读取方），列是 Key（被读取方）；1=允许，0=禁止。rgb0、ray0 为当前条件，1–4 为未来 latent 帧。每个帧级矩阵元素展开成对应帧的全部空间 token 之间的连接；文本 cross-attention 保持原样。

**Decoupled Mask**

```text
       rgb0 rgb1 rgb2 rgb3 rgb4 | ray0 ray1 ray2 ray3 ray4
rgb0     1    0    0    0    0 |   1    0    0    0    0
rgb1     1    1    1    1    1 |   1    0    0    0    0
rgb2     1    1    1    1    1 |   1    0    0    0    0
rgb3     1    1    1    1    1 |   1    0    0    0    0
rgb4     1    1    1    1    1 |   1    0    0    0    0
--------------------------------------------------------
ray0     1    0    0    0    0 |   1    0    0    0    0
ray1     1    0    0    0    0 |   1    1    1    1    1
ray2     1    0    0    0    0 |   1    1    1    1    1
ray3     1    0    0    0    0 |   1    1    1    1    1
ray4     1    0    0    0    0 |   1    1    1    1    1
```

**Bidirectional Mask**

```text
       rgb0 rgb1 rgb2 rgb3 rgb4 | ray0 ray1 ray2 ray3 ray4
rgb0     1    0    0    0    0 |   1    0    0    0    0
rgb1     1    1    1    1    1 |   1    1    1    1    1
rgb2     1    1    1    1    1 |   1    1    1    1    1
rgb3     1    1    1    1    1 |   1    1    1    1    1
rgb4     1    1    1    1    1 |   1    1    1    1    1
--------------------------------------------------------
ray0     1    0    0    0    0 |   1    0    0    0    0
ray1     1    1    1    1    1 |   1    1    1    1    1
ray2     1    1    1    1    1 |   1    1    1    1    1
ray3     1    1    1    1    1 |   1    1    1    1    1
ray4     1    1    1    1    1 |   1    1    1    1    1
```

### 两种 mask 的代码准备（2026-09-19）

- 已实现独立模式：Decoupled 为 `rgb_raymap_decoupled`，Bidirectional 为 `rgb_raymap_future_bidirectional`。当前 `rgb_then_raymap_block_causal` 分支及所有旧模式保持原样；不修改默认模型配置。
- 新模式都要求成对的 `[RGB0–4 | RAY0–4]` 布局、条件索引 `(0,5)`，条件只读取条件。Bidirectional 仅开放未来两组之间的连接，不使用原有允许条件读未来的 `bidirectional` 模式。
- 三组 Goal 专用配置：
  - 基线（2.4）：`configs/task/libero_goal_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4.yaml`
  - Decoupled（2.2）：`configs/task/libero_goal_all_absolute_mask_decoupled_wan21.yaml`
  - Bidirectional（2.3）：`configs/task/libero_goal_all_absolute_mask_future_bidirectional_wan21.yaml`
- 两个消融配置继承 Goal 基线，仅覆盖 mask、输出目录和运行名称。Goal 基线继承四-suite 全绝对配置，限定 Goal 数据并改为每2000步保存；全绝对 stats、原始 VAE、RGB/action 1:5、10epoch、batch4/GA4 和每1000步验证均保持已确认设置。
- 三组指向同一份待生成的 Goal 全绝对缓存：`data/libero_goal_rothko_all_absolute_2cam224_wan21_bf16_h16_latents`。没有生成缓存、启动训练或执行闭环评测；不能用不同表示的旧缓存替代。
- 两种模式均通过现有公共 forward 将 mask 传入每一层，训练与去噪复用同一路径；不增加模型参数。现有 checkpoint 元数据保存 mask 模式，加载时已有的不一致检查继续生效。
- CPU检查：精确10×10矩阵、空间token展开、非法布局拒绝、三层真实DiT输入扰动验证条件/分支隔离及允许的信息流、反向传播有限梯度、Hydra完整配置对比。测试：`tests/test_goal_attention_mask_ablation.py`；记录：`evaluate_results/libero/goal_mask_implementation_20260919/`。
- 2026-09-19 已在远端 A800 GPU0/3 完成两种 mask 的真实 Goal 窗口 GPU 检查（index26447、batch1、BF16、无 optimizer 更新）：缓存/在线 training_loss 逐位一致，完整 DiT 反向传播梯度有限、VAE 冻结；20步去噪返回有限的 `[1,17,7]` 位姿与 `[1,17,1]` 夹爪。每进程反向峰值 allocated 约21.83 GiB。报告：`evaluate_results/libero/goal_mask_gpu_20260919/{decoupled_v2,bidirectional_v2}/report.json`。这不替代正式 batch4/GA4 的8卡 optimizer/保存恢复验证，也不代表任务成功率。

### 执行顺序

- [x] 确认全绝对位姿、Wan2.1 1.3B、共同训练参数、原始冻结 VAE 和最终 checkpoint 规则。
- [x] 确认闭环测试沿用此前 Goal 协议，replan8、20 次去噪。
- [x] 确认 Decoupled / Bidirectional 的上述精确连接矩阵。
- [x] 确认 Latent-Repeat 使用 Cosmos 的数值重复／均值读取思想，保留 Wan ray0–4 布局，四个未来帧各承载 4 步动作。
- [x] 实现两种独立 mask 配置与 CPU 矩阵、跨层信息流和反向传播检查。
- [ ] 实现 Latent-Repeat 独立配置及往返、尺寸／尺度检查。
- [x] 两种 mask 的 GPU 小规模前向/反向、单窗口缓存读写一致性与20步推理链路检查（batch1；全量缓存与正式8卡训练尚未执行）。
- [ ] **先准备各表示所需的新缓存，再训练。** RGB 缓存仅在身份和数据契约一致时复用；动作表示不同不能冒用同一动作缓存。Mask 消融与基线可在契约一致时共享表示缓存。
- [ ] 分别执行 2.1–2.4 的四组 Goal-only DiT 训练与闭环测试，不影响已有实验。
- [x] 确认 2.5 的 conv2＋decoder 微调沿用之前 LIBERO Wan2.1 decoder 参数与 GT 重建方式。
- [ ] 完成 2.5 的 Goal 全绝对 decoder 微调，复用 2.4 的最终 DiT checkpoint，按同一 Goal 协议追加测试。
- [ ] 汇总成功率、训练／推理成本和代表失败案例；离线动作误差作为辅助指标，不以不同表示的训练 loss 代替任务成功率。

### 结果待填写

| 设置 | Run / checkpoint | Goal 成功数／总数 | 成功率 | 训练／测试耗时 |
|---|---|---|---|---|
| Latent-Repeat | 待填写 | 待填写 | 待填写 | 待填写 |
| Decoupled Mask | 待填写 | 待填写 | 待填写 | 待填写 |
| Bidirectional Mask | 待填写 | 待填写 | 待填写 | 待填写 |
| Rothko + 当前 attention mask | 待填写 | 待填写 | 待填写 | 待填写 |
| 2.5：2.4 DiT + 微调 VAE decoder | 复用 2.4 DiT；VAE checkpoint 待填写 | 待填写 | 待填写 | 待填写 |


## 3. 适配实验 1 的 LIBERO 四个 suite decoder 微调

- **状态**：专用配置、固定切分/缓存索引、审计和训练入口已准备，CPU检查通过；缓存尚未生成，GPU审计、训练及评测尚未启动。
- **目标**：为实验 1 的四-suite 全绝对位姿方案微调 VAE，比较同一 DiT 搭配 original VAE 与微调 decoder 的效果。
- **数据与表示**：使用 LIBERO spatial、object、goal、10 四个 suite；沿用实验 1 的全绝对 RAY0 和未来目标、17 帧／16 步、224×448 Rothko，以及同一份四-suite 全绝对 norm_stats。与 2.5 的 Goal-only decoder 分别训练和保存。
- **训练方式**：从 original Wan2.1 VAE 初始化，冻结 encoder，仅训练 **conv2＋decoder**；GT Rothko 经原始冻结 encoder 得到 latent，再重建 GT Rothko。其余做法沿用此前 decoder 微调。
- **训练参数**：**8 卡、每卡 batch2、GA4，有效 batch64；2 epoch、LR=1e-5、cosine、warmup 5%、最低 LR 比例 0.01、weight decay=0、梯度裁剪=1**。FP32 master 参数／梯度／AdamW 状态，BF16 autocast，分片优化器。
- **Loss**：中心／方向／夹爪区域 masked L1，权重 1/1/1，使用本次全绝对 GT 图和匹配区域。
- **划分与验证**：每任务留出 2 个 episode，split_seed=20260801；训练 seed42；验证 uniform_per_episode、每 episode 10 窗口、eval_batch1、eval_seed12345。仅用训练部分拟合 decoder。
- **保存与验证频率**：每400步保存 latest，每800步保存 step checkpoint、导出 VAE并验证，epoch 结束验证，结束保存并导出；测试使用最终导出的 VAE。总 steps 与 warmup 按实际训练窗口数重新计算。
- **尾部 padding（已确认）**：遍历训练episode全部frame starts，未来目标不足16步时edge-repeat最后动作；补齐帧参与重建loss，验证动作误差排除padding。按固定split共1,632个训练episode、264,409个窗口；8卡有效batch64，4,132步/epoch，2epoch共8,264步，warmup413步。四-suite专用入口已按此规则实现，不改历史公共trainer的默认无padding枚举。
- **缓存**：先核对 decoder 的窗口采样、padding、精度和 encoder 输入契约；仅在完全一致时复用实验 1 的原始 VAE 缓存，否则生成独立缓存，不覆盖已有缓存。
- **闭环评测**：复用实验 1 的最终 DiT checkpoint，仅换成此处微调的 VAE，encoder 保持原始冻结权重。四个 suite 均沿用实验 1 的场景、seed、每任务 50 次测试、replan8、20 次去噪及动作解码／控制设置。

### 已准备的代码与检查（2026-09-19）

- 配置：[libero_all4_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json](configs/vae/libero_all4_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json)。
- 独立入口：[prepare_all4_absolute_decoder.py](scripts/prepare_all4_absolute_decoder.py)，提供 `prepare / check-cache / audit / train`；GPU阶段默认只打印命令，必须显式 `--execute`。
- 操作说明：[ALL4_ABSOLUTE_DECODER_RUNBOOK.md](experiments/libero/ALL4_ABSOLUTE_DECODER_RUNBOOK.md)。
- 固定清单：`evaluate_results/libero/all4_absolute_decoder_preparation_padding_20260919/manifest.json`，包含训练/验证episode、800个验证窗口、264,409个训练缓存索引，以及配置/VAE/stats/数据契约身份。
- 缓存索引按实验1的spatial/object/goal/10顺序拼接，以dataset_root与episode ID共同标识样本；CPU已核对与实验1缓存契约一致。实际缓存目前不存在。
- 独立入口包含尾部edge-repeat、缓存训练、BF16部署解码验证及总体/逐suite前8/16步指标；验证动作误差排除padding，补齐帧参与训练重建loss。恢复要求完整优化器状态与匹配的数据/配置/缓存身份。
- 保存latest：每400步、epoch结束4132、最终8264；独立权重checkpoint/VAE导出：每800步及最终8264；验证：每800步和epoch结束。历史独立权重checkpoint不含优化器，完整续训用latest。
- **12项CPU测试通过**：6项四-suite新增检查，加6项Goal回归；覆盖40任务首尾窗口编解码、固定split、尾部padding、suite间同号episode防混淆、实验1缓存契约、配置与默认不启动GPU。记录：`evaluate_results/libero/all4_absolute_decoder_preparation_padding_20260919/cpu_tests.log`。
- 公共trainer、Goal专用入口和当前VLABench训练未修改。GPU训练/验证/恢复尚未实际测试，不以CPU检查替代。

### 待完成

- [x] 准备四-suite 全绝对 decoder 专用配置、固定数据划分与缓存契约检查入口。
- [ ] 实验1缓存完成后执行实际缓存检查与240个GPU逐位一致性probe。
- [x] GPU 单卡组件检查：encoder/conv1 冻结、conv2＋decoder 反向与 FP32 AdamW 更新、尾部padding参与loss、BF16部署解码有限；确定性设置下保存恢复后下一步 loss/权重逐位一致。仅单个末尾窗口、batch1/GA1，不替代完整验证集重建误差审计或8卡分片优化器续训测试。
- [ ] 完成 decoder 微调，独立保存 run、配置与最终 VAE 权重。
- [ ] 固定实验 1 最终 DiT，比较原始 VAE 与微调 decoder，报告各 suite 和整体成功数／总数、成功率及耗时。

| 设置 | DiT checkpoint | VAE checkpoint | spatial | object | goal | 10 | 整体成功率 |
|---|---|---|---|---|---|---|---|
| 实验 1：original VAE | 实验 1 最终 checkpoint | original Wan2.1 VAE | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| 实验 3：微调 conv2＋decoder | 同上 | 本实验最终 VAE，待训练 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 2026-09-19：远端两张 A800 的补充 GPU 检查

- 已按用户授权只用远端 GPU0/3，诊断目录 `evaluate_results/libero/goal_mask_gpu_20260919`；本机同步报告和脚本，未同步临时 checkpoint/latent 文件。完整报告见该目录 `REPORT.md`。未启动正式训练或全量缓存。
- 两种 mask：真实 Goal index26447、batch1、完整1.3B DiT、BF16；完整梯度检查、缓存/在线 loss 逐位比对和20步推理通过。诊断没有 optimizer 更新。
- Goal decoder（2.5）与四-suite decoder（3）：分别用末帧 index52894/277712，各含15个padding未来步，做2次单卡 FP32 AdamW 更新和一次恢复重放。保存权重/optimizer/scheduler/RNG；encoder/conv1不变，梯度有限、BF16部署解码有限。确定性计算设置下恢复后下一步 loss 和参数逐位一致。
- 默认非确定性计算下，恢复前后 loss 相同，参数最大差为 Goal 3.33e-8、四-suite 8.38e-9；确定性复核消除差异。正式训练配置未启用或修改确定性设置。
- 这些是 batch1/GA1、单卡非分片优化器组件检查，不能代替8卡 batch2/GA4 的 sharded optimizer、DataLoader/完整训练入口续训验证。缓存完成后还需各自60/240个正式逐位probe以及全验证集审计。

### 2026-09-20：四-suite缓存切六卡续算

- 用户授权使用远端GPU0/3/4/5/6/7，不使用GPU1/2。先停止原两卡全部进程，再清理2个未完成partial文件；227个完整shard（232,448窗口）保留，剩余45个shard。
- 同一输出目录、原始VAE、BF16、每进程batch8/workers4、每shard1024保持不变。底层缓存脚本检查原metadata后将剩余shard分配给6个rank。
- tmux：`libero_all4_absolute_cache_6gpu`。启动/清理记录、原metadata快照和日志：`evaluate_results/libero/all4_cache_6gpu_resume_20260920_013001/`。最终仍需通过latent逐位与training_loss一致性校验才算完成。

## Goal 原始 OSC 增量对照（2026-09-20）

用户取消“减桌高＋统一stats”，其代码已撤回到`4d2c60e`并归档，本机8卡该缓存作业已停止。历史数据/结果保留。当前改为独立`libero_rothko_osc_delta`：RAY0使用原始世界state，未来直接编码原始OSC平移/旋转命令；不减桌高、不分suite归一化、不重建绝对target。state bounds按全4suite当前state极值+每侧1cm，未来平移固定±5cm、旋转向量命令scale0.5rad。eval直接恢复OSC命令，保持控制器原逻辑，禁用绝对位姿ensemble。

Task `libero_goal_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4`，Eval `sim_libero_goal_osc_delta`。Wan2.1原始冻结VAE，预算保持原Goal10epoch、8卡bs4/GA4、LR1e-4、warmup5%、验证1000、保存2000+最终、baseline mask和RGB/action1:5。先做编解码与小缓存验证，再正式8卡缓存；训练启动前确认参数。
详见 [Goal OSC runbook](experiments/libero/GOAL_OSC_DELTA_RUNBOOK.md)。

原始OSC方案预检通过：30项回归、实际eval helper、20窗口原始VAE重建与16窗口真实training_loss一致性。已启动本机tmux `libero_goal_osc_cache`，8卡计算52,895窗口新缓存，batch2/每rank两个数据worker；训练未启动。


### OSC 两机分工（2026-09-20）

- 本机 ainode07：计算 Goal 缓存，之后只训练 Goal。
- 远端 ainode06（172.16.12.6）：计算 spatial/object/10 三个 suite 缓存，再接收本机已完成且验证通过的 Goal 缓存，合并后训练四个 suite。
- 两端原始 Wan2.1 VAE SHA256、19个源文件、stats 指纹、数据元信息及 PyTorch 版本已核对一致；沿用同一原始 OSC 表示。预检记录：`video_a800/osc_delta_sync_20260920/preflight.json`。
- 已停止远端旧 Goal 全绝对 VAE 训练；`checkpoint_latest.pt` 的实际 step=1200，含 optimizer/scheduler，未删除历史结果。
- 远端缓存 tmux：`libero_other3_osc_cache`；8卡，每rank batch2、2个数据worker，线程数2；224,818窗口。日志：`evaluate_results/libero/other3_osc_delta_cache_launch_20260920/cache.log`。
- 远端三-suite task：`libero_other3_rothko_osc_delta_2cam224_wan21_cache`；训练四-suite task：`libero_all4_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4`。
- Goal 缓存传输尚未完成。合并使用 `scripts/merge_libero_osc_caches.py`，按 spatial/object/goal/10 索引顺序逐字节重排分片，不重新编码；合并后必须运行标准缓存入口完成在线 latent/training_loss 验证才能用于训练。不能直接拼 metadata 或重命名分片。
- 此次只启动缓存；两台正式 DiT 训练均未启动，启动前确认训练参数。


### 2026-09-20 OSC 流程复核：补零语义已确认

用户明确：未来预测的是逐步 OSC 增量，尾部 padding 的平移／旋转应补零，夹爪保持最后值；RGB/state 仍重复末帧。当前 processor 符合该语义。此前把它误判成必须 edge-repeat 动作并要求重算的结论撤回；不应用 delta_action_dim_mask=null，不因此停止／重算缓存。raw/action tensor 共享确实存在，后续修改 processor 时需要留意，但当前结果符合要求。

本机 Goal 完整缓存已通过：indices 0/26447/52894 在线 latent 逐位一致，固定 seed123456 的 training_loss 逐位一致（1.1243473291397095），complete=true、_SUCCESS。远端缓存及四-suite合并后的实际校验、八卡训练／保存恢复仍待完成。训练内验证使用训练集前4窗口，四-suite中仅覆盖spatial，不是独立held-out验证。详见 `evaluate_results/libero/osc_pipeline_review_20260920/REPORT.md`。


### 2026-09-20：扩大新 OSC 训练内验证覆盖

- Goal：全部10任务，每任务3个不同episode，各取开头／中段／尾部一个窗口，共30窗口。
- 四-suite：全部40任务，每任务同样3窗口，共120窗口；suite顺序spatial/object/goal/10。
- 尾部窗口保留8个有效动作＋8个padding动作；OSC padding平移旋转补零、夹爪保持。其余两类窗口16步有效。
- 每1000 optimizer steps验证，20步去噪；固定窗口与seed(42+清单序号)，8卡轮流分摊（Goal每卡3–4窗口，四-suite每卡15窗口），每个窗口均计算loss和推理指标。总耗时会增加，实际耗时待第一次运行测量。
- 新选项eval_sample_indices仅两份OSC训练配置启用；旧配置未设置时保持原连续索引逻辑。原held-out manifest分支不变；训练样本、缓存、loss、保存间隔不变。
- 这是训练集固定probe，不是独立held-out验证。清单：configs/validation/libero_goal_osc_probes.json、libero_all4_osc_probes.json；可复现生成脚本scripts/prepare_libero_osc_validation.py。

### 2026-09-22：Latent-Repeat 隔离试验

- 用户要求不影响现有代码：仅新增 `libero_latent_repeat` 表示、数据适配器、模型子类、独立factory与task配置。原模型／数据／评测入口不修改，只有选择新task才启用。
- 配置 `configs/task/libero_goal_latent_repeat_absolute_ray0_wan21.yaml`：8卡batch4/GA4，有效128，10epoch/4140步，原始冻结Wan2.1 VAE仅用于RGB；mask与扩散loss继承当前基线，RGB/action=1:5。
- 直接复用20260921完成的Goal缓存的RGB部分；动作侧读取原始数据中的当前state和重建OSC目标，生成数值repeat tensor，不使用缓存中的Rothko latent。尾部目标继续沿用基线edge-repeat，保留原latent分组padding loss口径。
- 诊断脚本 `scripts/audit_libero_latent_repeat.py`，输出 `evaluate_results/libero/latent_repeat_smoke_20260922/`。正式训练未启动；诊断结果以该目录报告为准。

- 隔离诊断已通过：9项新测试＋5项原absolute-RAY0回归；5个真实窗口数值恢复与基线直接编解码一致；1次完整DiT参数更新＋20步去噪通过。缓存训练VAE调用0次，推理仅当前RGB encode1次、decode0次，峰值19.263GiB。详见 `evaluate_results/libero/latent_repeat_smoke_20260922/REPORT.md`；仍未启动正式训练。

- **2026-09-22 04:57 UTC 已启动正式Latent-Repeat训练**：本机8卡batch4/GA4，有效128，10epoch/4140步，RGB/action=1:5；tmux `libero_goal_latent_repeat_train`；run `runs/libero_goal_latent_repeat_absolute_ray0_wan21/2026-09-22_04-57-27`；启动日志 `evaluate_results/libero/goal_latent_repeat_train_launch_2026-09-22_04-57-27/`。W&B run `1w6rqd11`。每500步验证4个固定窗口，每2000步及最终保存权重和完整state。

- **Latent-Repeat 首次正式训练故障（2026-09-22 05:59 UTC）**：step500首次验证时，公共trainer的 `_to_batched_eval_sample` 白名单丢弃 `repeat_pose/repeat_gripper`，新模型报KeyError后退出；未达step2000保存点，无可续训checkpoint。原smoke未覆盖公共trainer验证适配层。修复仅在新增模型子类内：验证时从保留的当前/未来数值pose和未来gripper还原目标，当前gripper读取无VAE的GT RAY0；不修改公共trainer。补测脚本 `scripts/audit_libero_latent_repeat_validation.py`，结果目录 `evaluate_results/libero/latent_repeat_validation_fix_20260922/`。正式训练尚未重启。

- **2026-09-22 八卡完整短流程通过后重启**：诊断 `evaluate_results/libero/latent_repeat_e2e_20260922_062604/REPORT.md`，两步训练＋每步真实4窗口验证＋权重/完整state保存，然后从step1恢复并完成step2验证保存；两段exit0。关闭期NFS临时文件清理Errno16不影响已验证的state恢复。正式run `runs/libero_goal_latent_repeat_absolute_ray0_wan21/2026-09-22_06-34-36`，保持8卡batch4/GA4、10epoch/4140步、LR1e-4、RGB/action1:5、每500步验证、2000步及最终保存完整state/权重；从预训练初始化重新开始，不使用短测权重。

- **Latent-Repeat Goal 3090评测准备（2026-09-22）**：选`ainode16 /172.16.12.17`，独立目录`/mnt/data/cfy/FastWAM/eval_preparations/latent_repeat_goal_20260922`；短rollout实际加载step2000、执行两次replan并保存视频，exit0，峰值reserved14.37GiB；8卡每卡1worker，Goal10×50、replan8、denoise20、原始RGB VAE，等待最终step4140。启动流程与检查记录：`evaluate_results/libero/latent_repeat_3090_prepare_20260922_135110/README.md`。正式评测尚未启动。

- **2026-09-22 14:58 UTC Latent-Repeat最终Goal评测已启动**：本机训练正常结束（exit0），step4140完整state含8份优化器分片与8份RNG状态；最终权重传至ainode16并校验SHA256。tmux `goal_latent_repeat_eval`，8卡每卡1worker，Goal10×50，原始RGB VAE、replan8、20步去噪、seed42。远端输出`/mnt/data/cfy/FastWAM/eval_preparations/latent_repeat_goal_20260922/evaluate_results/goal_step4140_original_replan8_ep50_20260922_145809`；首批8任务运行、2任务排队。

- **15:00 UTC评测启动修正**：初次多worker子tmux未继承LIBERO_CONFIG_PATH，停在路径交互提示，尚未推理。仅修改3090独立快照scheduler，每个worker显式source environment.sh；停止本次两个会话后重新启动。实际输出目录改为`/mnt/data/cfy/FastWAM/eval_preparations/latent_repeat_goal_20260922/evaluate_results/goal_step4140_original_replan8_ep50_20260922_150003`。已确认8个worker均加载step4140，正式430步上限rollout运行，GPU约15.5GB；旧输出保留。

### 2026-09-23：四-suite absolute RAY0 / relative action VAE准备

新增独立入口`scripts/prepare_all4_absolute_ray0_decoder.py`和配置`configs/vae/libero_all4_absolute_ray0_decoder_wan21_bs2_ga4_lr1e-5_ep2.json`，复用原始Wan2.1 Rothko缓存，只训conv2+decoder。包含尾部padding，264409训练窗口、1632训练episodes；80验证episodes/400固定完整窗口（每episode5个，排除尾部padding窗口），按8卡分摊验证。默认8卡bs2/GA4、LR1e-5、2epoch/8264步、warmup413，每500步及最终完整保存、导出和验证。未来相对位姿解码必须传当前EE观测。用户随后确认不监督padding：保持全部frame-start窗口与edge-repeat输入，但重建loss仅监督RAY0和有效未来帧，DDP按跨rank有效帧数归一化。CPU24窗口检查与padding梯度检查已通过；GPU完整验证状态见`evaluate_results/libero/all4_absolute_ray0_decoder_preparation_val400_20260923/README.md`及日志。未启动正式训练。

- 最终val协议：80个独立验证episodes，每episode均匀取5个完整horizon窗口，共400个，无尾部padding窗口。八卡实测首轮val16.77秒（含验证VAE加载与汇总），训练首步8.3秒；保存完整state＋导出＋val约50余秒。预计8264步含周期保存验证约20–23小时。240个覆盖40任务/训练验证/首尾窗口的缓存审计全部BF16逐值一致。新入口修复八卡输出目录创建竞态（初始化进程组后在创建日志前同步）；诊断旧失败日志保留，正式长训练尚未启动。

- **2026-09-23 01:30 UTC 已授权启动正式四-suite absolute RAY0 / relative-action VAE**：tmux `all4_absolute_ray0_vae_train`，run `runs/libero_all4_absolute_ray0_decoder_wan21_bs2_ga4_lr1e-5_ep2`，启动日志`/mnt/hwdata/cfy/FastWAM/evaluate_results/libero/all4_absolute_ray0_vae_launch_20260923_013046`。8卡bs2/GA4有效64，2epoch/8264步，LR1e-5，只训conv2+decoder；padding不计重建loss；每500步及最终完整保存、导出、400完整窗口验证。新建训练，不加载smoke权重。

### 2026-09-23：四-suite训练完成后的三机 LIBERO-Plus 评测准备

由用户通知启动，不自动等待或启动。DiT采用ainode06的absolute-RAY0四-suite run最终step21700，VAE采用本机同表示decoder微调最终step8264；最终产物和训练完成状态届时复核。保持相对future-action、共享四-suite stats、replan8、20步去噪、seed42、legacy/anchor0、不ensemble。

- ainode16（3090）：Object2518 + Spatial偶数task_id1201，共3719，每卡1worker。
- ainode18（3090）：Goal2591 + Spatial奇数task_id1201，共3792，每卡1worker。
- ainode06（A800）：完整LIBERO-10的2519场景，每卡4worker。
- 每台8卡；Plus全部10030场景各1次，固定任务清单无交集且无遗漏。新增独立wrapper，不改现有manager/worker。最终使用隔离快照、SHA256检查、按结果身份续跑。
- 代码/清单/传输和启动步骤：`experiments/libero_plus/absolute_ray0_three_host/README.md`。当前仅准备及CPU检查；远端依赖、EGL和最终权重组合短rollout尚待传输后确认。没有启动评测。

### 2026-09-23：当前 absolute RAY0＋relative-action 三表示100窗口复核

已完成独立GT round-trip诊断，未改训练/评测代码。严格复用20260921 OSC诊断的Goal100个窗口（每任务10个episode，各1窗口），改为当前四-suite配置的绝对RAY0和相对当前EE的未来位姿；原始Wan2.1 VAE、BF16、batch1、17帧224×448、legacy/anchor0。Flatten CHW用相同逐帧物理bounds（RAY0绝对min/max、未来相对Q99.95）及裁剪，编码8维数值重复后reshape；解码未来局部位姿均通过同一当前EE观测还原。Shuffled使用固定seed42像素置换。

100窗口×3表示×3链路=900记录覆盖检查通过。尾部edge-repeat输入、统计排除padding，全16步1459个有效目标；其中4个超范围目标发生相同平移裁剪，均在后8步，直接编解码全16步平移均值0.03066mm，不应表述为所有目标完全无损。没有四元数退化。

| 表示 | VAE前8步平移均值/P95 mm | 旋转均值/P95 ° | 夹爪二值错误率 |
|---|---:|---:|---:|
| Rothko | 4.572/10.149 | 1.191/2.442 | 0% |
| Flatten CHW | 70.764/113.751 | 122.595/171.700 | 12.583% |
| Shuffled | 60.317/103.808 | 25.711/63.315 | 0.662% |

全16步平移/旋转均值依次为3.776mm/1.018°、79.404mm/117.940°、66.667mm/23.622°。结论仅针对原始冻结VAE的GT重建适配性，不是DiT学习或闭环SR。完整脚本、逐窗口及直接/BF16/VAE对照见`evaluate_results/libero/absolute_ray0_relative_repr100_20260923/`，汇总`results/SUMMARY.md`。

### 2026-09-23：Goal absolute-RAY0 DiT＋最终四-suite VAE评测

本机VAE已正常完成8264步，完整恢复点及完整BF16 VAE导出保存；最终400窗口验证位置0.745mm、旋转0.109°。用户授权后启动Goal 10任务×50次评测：Goal absolute-RAY0/relative-action DiT run `2026-09-21_18-42-23` step4140，配本次四-suite同表示VAE step8264；8卡、max_tasks_per_gpu=2、replan8、20去噪、seed42、legacy/anchor0、不ensemble。与20260922原始VAE评测共用DiT、dataset_stats及协议，仅替换VAE并允许明确的VAE身份差异。

tmux `goal_absolute_ray0_vae8264_eval`；启动记录`evaluate_results/libero/goal_absolute_ray0_wan21_vae8264_eval_launch_20260923_202057`；输出`evaluate_results/libero/goal_absolute_ray0_wan21_step4140_vae8264_replan8_ep50_20260923_202057`。这是普通LIBERO Goal500，不是LIBERO-Plus；另一台A800四-suite DiT训练未改动。

### 2026-09-24：Goal mask 消融改用绝对 RAY0＋相对 action

用户确认本轮两组都采用 `libero_rothko`、`frame0_pose_mode=absolute`：RAY0 为当前世界坐标 EE 位姿，未来16步为相对当前 EE 的目标位姿。此记录覆盖本轮 mask 训练的旧“全绝对”设定；旧配置和历史实验保留。

- 本机 ainode07：`libero_goal_absolute_ray0_relative_mask_decoupled_wan21`，8卡，`rgb_raymap_decoupled`。
- 另一台 ainode06：`libero_goal_absolute_ray0_relative_mask_future_bidirectional_wan21`，8卡，`rgb_raymap_future_bidirectional`。
- 两组均 Wan2.1 1.3B 全量 DiT、原始冻结 VAE；共用 Goal 52895窗口缓存 `data/libero_goal_rothko_absolute_ray0_wan21_bf16_h16_latents_20260921` 和 BF16 batch1 文本缓存。
- 10epoch，batch4/GPU、GA4、有效batch128，预计4140步；LR1e-4、warmup5%、cosine，RGB/action loss=1/5，BF16，无gradient checkpointing。
- 每1000步验证既定4固定窗口；每2000步和最终保存权重及完整state。独立配置不修改旧基线。
- 6项mask测试通过；缓存dataset contract、52895窗口、latent shape `[16,5,28,56]` 和原始VAE哈希通过预检。
- 启动脚本 `scripts/train_goal_relative_mask_8gpu.sh`；检查及训练日志 `evaluate_results/libero/goal_relative_mask_launch_20260924/`。
- **正式启动**：ainode07 Decoupled run `runs/libero_goal_absolute_ray0_relative_mask_decoupled_wan21/2026-09-24_15-26-29`，W&B `z0qs2lu1`；ainode06 Bidirectional run `runs/libero_goal_absolute_ray0_relative_mask_future_bidirectional_wan21/2026-09-24_15-27-53`，W&B `vte1ht5j`。两边日志确认 `world_size=8`、GA4、`max_steps=4140`、trainable DiT参数1,418,996,800。展开配置差异仅mask、run名与输出目录，见启动目录 `config_differences.json`。
- **2026-09-25 Goal闭环评测启动**：两台各8卡，分别评测本机Decoupled和ainode06 Bidirectional的最终step4140；原始Wan2.1 VAE、legacy/anchor0、seed42、replan8、20步去噪、无action ensemble，10任务×50场=500场/组，10任务同时运行（每卡最多2worker，时限12h）。两组均保留各自训练mask。启动日志 `evaluate_results/libero/goal_relative_mask_eval_20260925/`；结果目录 `evaluate_results/libero/goal_relative_mask_{decoupled,future_bidirectional}_step4140_original_replan8_ep50_20260925`（对应机器）。


### 2026-09-25：新增 Shuffled Action Raymap Goal 训练

用户重新授权训练 Shuffled 表示，补充此前仅做离线重建的计划。独立配置 `configs/task/libero_goal_shuffled_raymap_wan21.yaml`；绝对 RAY0、相对当前 EE 的未来目标，Wan2.1 1.3B 全量 DiT、原始冻结 VAE、基线 attention mask。每个224×224 tile 内以 seed42 固定置换 UV，三个颜色通道一起移动，两份 tile 使用同一置换；解码先逆置换。置换 SHA256：`7d2c094e41ab74856c4a3aeb01491853ecdc06df8a3aa266241c2f54ea74cfae`。

- 训练：10epoch，8卡 batch4/GPU、GA4，有效batch128，预计4140步；LR1e-4、warmup5%、cosine、RGB/action loss=1:5。每1000步验证4个固定窗口，每2000步及最终保存权重与完整state。
- 缓存：52895窗口，RGB逐位复用 `data/libero_goal_rothko_absolute_ray0_wan21_bf16_h16_latents_20260921`，动作重新编码到 `data/libero_goal_shuffled_raymap_wan21_bf16_h16_seed42_20260925`，不覆盖基线缓存。
- 隔离代码快照和启动脚本位于 ainode06 的 `video_a800/libero_goal_shuffled_20260925/`；tmux `goal_shuffled_raymap_train`。队列依次运行8卡每卡4个缓存worker、缓存/在线一致性检查、8卡训练；任一步失败即停止。日志在该目录 `logs/`，检查结果在 `smoke/`。
- 启动前已通过5窗口（含尾部）与基线逐像素对应、逆置换直接解码、padding/RGB/text不变检查；GPU单步训练、20步推理及checkpoint身份隔离通过，单步loss0.08588、梯度有限。此为流程检查，不代表模型效果。


### 2026-11-07：Goal Wan2.1 Video Model＋Action Expert 公平对照

参照 `evaluate_results/libero/closed_loop/goal_absolute_ray0_wan21_step4140_vae8264_replan8_ep50_20260923_202057`。新增独立工厂、数据适配、训练/评测入口；旧FastWAM及Action Raymap实现与配置均不修改。Wan2.1 Video DiT＋从同源权重插值初始化的Action Expert，当前绝对EE位姿/夹爪数值proprio，未来16步相对目标位姿8维数值动作；17帧RGB、同一Goal52895窗口、原始冻结VAE及BF16 batch1文本。复用已验证RGB缓存，不消费动作VAE latent。4卡每卡batch4/GA8，10epoch/4140步、有效batch128、LR1e-4、warmup5%、loss1:5，可评测weight每epoch保存一次（当前414步/epoch），完整state每2000步+最终；闭环Goal500、replan8、20步去噪、同一目标转OSC控制代码。

详见 [MATCHED_FASTWAM_GOAL.md](experiments/libero/MATCHED_FASTWAM_GOAL.md)。CPU配置/五窗口数据对齐/缩小MoT前后向及动作推理检查记录在 `evaluate_results/libero/preparation/matched_fastwam_goal_20261107/`。正式Action Expert初始化文件已于2026-11-07生成，耗时58.82秒、峰值RSS13.83GiB，来源/形状/数值检查通过；正式训练尚未启动，完整尺寸GPU与4卡恢复检查待做。98.2%参照模型使用额外四-suite decoder微调，故此为完整系统对照，需同时说明其与原始VAE93.0%结果的区别；不能声称只改变架构且总训练数据/计算完全一致。

### 2026-11-08：Goal 旋转区域常量轴角表示消融

保留当前绝对 RAY0＋未来相对当前 EE 的16步目标、中心平移和外框夹爪；只把旋转方向场改为整片常量三维最短轴角向量，除以 pi 映射到图像范围。RAY0 编码绝对旋转，未来帧编码 `conj(q0)*q_target`。两个224×224 tile 横向复制；读取时先均值合并对应像素，再在安全旋转区域逐分量取中位数，乘 pi、转四元数，结合当前观测 EE 恢复目标。没有方向场SVD和anchor校正。独立codec/dataset/model/factory/config/eval入口，不修改既有共享流程。详细命令及限制见 [AXIS_ANGLE_GOAL.md](experiments/libero/AXIS_ANGLE_GOAL.md)。

同一组Goal100窗口的原始冻结Wan2.1 VAE重建诊断，前8步755个有效目标：常量轴角平移均值/P95 **4.466/9.405 mm**、旋转 **2.019/4.594°**、夹爪错误0%；原Action Raymap分别 **4.572/10.149 mm**、**1.191/2.442°**、0%。这些是GT→VAE→动作误差，不是DiT预测或闭环成功率。最终修正审计目录 `evaluate_results/libero/offline_audits/axis_angle_vae100_20261108_quaternion/`；包含精确180°姿态的局部四元数实现修正，未修改公共旋转转换函数或其他表示。

准备检查目录 `evaluate_results/libero/preparation/axis_angle_pipeline_20261108/`：5项codec测试＋3项pipeline测试通过；256窗口缓存逐位RGB复用、动作在线一致、batch1/2一致及hash检查通过；5个真实窗口（含15步padding尾窗）对齐原始parquet；完整Wan2.1 batch1前后向、2步推理与公共trainer的20步验证入口通过。17帧训练编码的首个latent与单帧推理编码逐位一致。19.39–19.52 GiB是无优化器的batch1诊断显存，不代表正式batch4＋AdamW/ZeRO显存。

完整新动作缓存正在独立tmux `goal_axis_angle_cache_20261108` 生成：8卡每卡4worker、BF16 batch2，共52895窗口、26.54 GB；RGB逐位复用原Goal缓存，文本和norm_stats保持一致，新目录 `data/libero_goal_axis_angle_constant_wan21_bf16_h16_224x448_20261108`。分片可恢复，最终需hash＋在线复核成功。正式DiT训练尚未启动。准备配置为8卡bs4/GA4、有效128、10epoch/4140步、LR1e-4/warmup5%、原始冻结VAE、原mask及RGB/action1:5；每epoch保存可eval权重，完整state每2000步＋最终，验证每500步。闭环按Goal500场、replan8、20步去噪、原始VAE、同一pose→OSC流程，预设最终step4140用于对比。

**缓存完成更新（2026-11-08）**：52895窗口／207分片全部完成，所有分片hash通过、全量RGB逐位复用、32个worker编码探针通过；最终索引 `[0,122,137,26447,52894]` 在线动作编码与缓存逐位一致，`_SUCCESS`、`metadata.json.complete` 和 `check_online.json` 均确认完成。另有2项真实评测辅助接口测试通过，确认新模型工厂参数及现有pose→OSC控制保持一致。已具备启动条件，但用户明确要求**先不启动训练**；缓存工作流不会自动接着训练。

**用户随后授权正式启动（2026-11-08 17:31:45 UTC）**：8卡训练已单独启动，tmux `goal_axis_angle_train_20261108`；全新run `runs/libero_goal_axis_angle_constant_wan21/2026-11-08_17-31-45`，未覆盖旧实验。DeepSpeed ZeRO1、world_size8、bf16、bs4/GA4、10epoch/4140步；每epoch权重、每2000步＋最终完整state、每500步验证。W&B run `tjtcojxo` 已连通。启动记录/源码快照/日志在 `evaluate_results/libero/launch_records/goal_axis_angle_8gpu_2026-11-08_17-31-45/`。

### 2026-11-09：VLABench all10 step18000 续训准备，暂不启动

用户暂定完整可续训state改为每1000 optimizer steps＋结束保存。独立配置 `configs/vlabench_all10_resume18000_state1000.yaml` 以最近一次实际续训配置为快照，恢复同一原run的 `checkpoints/state/step_018000`；`state_save_every=1000`、`state_save_every_epochs=null`、`save_at_end=true`，避免epoch间隔覆盖步数间隔。下一完整恢复点step19000。其余模型、数据、原始冻结VAE、scheduler、有效batch128、总44490步及验证/权重规则保持原样；继续原W&B run `7ypn8sop`。

已将归档的2410字节cache_index.json逐字节复制回原配置引用位置，仅恢复这个小索引，缓存数据和归档结果不移动、不覆盖；这使原state的严格resume语义指纹保持一致，SHA256为 `f23f163ef4b53f4039ea9cd4a6b4429050545630e41f231af373f33fbc8cc34b`，无需放宽resume校验。检查记录 `evaluate_results/vlabench/preparation/resume18000_state1000_20261109/`。启动脚本 `scripts/resume_vlabench_all10_18000_state1000_8gpu.sh` 已准备，**未执行；等待用户后续启动指令**。

### 2026-11-09：比例缩小 Action Expert 的 FastWAM Goal 8卡训练

用户先提出后4卡、随后改为8卡。4卡准备未执行；现使用独立 task `libero_goal_matched_fastwam_ratio20_wan21_8gpu` 和启动脚本 `scripts/train_matched_fastwam_ratio20_8gpu.sh`。Wan2.1 Video DiT 1,418,996,800参数；Action Expert hidden560/FFN2240、30层/12头/head_dim128，共287,376,248参数，占Video20.25%。初始化为此前已核验的 `checkpoints/ActionDiT_matched_Wan21_560hdim_pose8.pt`；旧1024宽度模型及结果保持不变。

8卡bs4/GA4，有效128；Goal52895窗口、原RGB与文本缓存、绝对当前state＋16步相对目标8维动作；原始冻结RGB VAE，RGB/action loss1:5，LR1e-4/warmup5%、10epoch/4140步。Video、Action Expert及MoT混合注意力的梯度检查点全部关闭。用户确认**可eval权重每epoch（414步）保存，完整可续训state每2epoch（828步）及最终保存**，验证每500步。2epoch墙钟先按约1小时估计，待正式速度校准。新模型短测完成首批迭代后实测每卡约45.8GiB；该数值不是旧1024宽度模型的历史峰值。

启动队列tmux `goal_fastwam_ratio20_8gpu_20261109_023951`，记录 `evaluate_results/libero/launch_records/matched_fastwam_ratio20_8gpu_20261109_023951/`。先在独立smoke目录做2步训练/验证/完整state保存，再从step1恢复到step2，检查通过后自动启动正式新run `runs/libero_goal_matched_fastwam_ratio20_wan21_8gpu/20261109_023951`。两阶段配置/日志/状态分别保存，W&B仅正式阶段开启；VLABench续训不在此队列中。

恢复短测已通过，队列已进入正式阶段：8份优化器分片齐全，scheduler、8rank CUDA RNG及trainer进度一致；BF16参数非逐位一致，MoT参数有0.0807%不同，最大绝对差0.00024414，差异RMS约为单步更新RMS的2.22%；proprio为3.69%。满足预先设定的数值恢复容差，不声称GPU逐位确定性。完整记录 `resume_check.json`。正式训练从原始初始化重新开始，不继承smoke更新。

正式run已进入step10/4140，W&B `8lr59cu6` 已连接。起始实测约0.25 step/s（每步约4秒），总训练先估4.5小时左右，828步约55分钟＋验证/保存开销，与每约1小时完整恢复点相符；后续以稳态日志为准。
