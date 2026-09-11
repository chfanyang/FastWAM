# RoboTwin C2R 实施记录

## 已确认选择

**2026-09-07 更新：新 C2R 统一采用官方实际 EE 序列。** 当前原点使用官方 endpose[t]，
未来16步使用 endpose[t+1:t+17]，不再以目标关节 FK 作为新实验的 EE 监督。
夹爪仍从独立夹爪字段读取，并在接入前核对其语义。下面既有 FK 统计仅保留为审计记录，
不能直接作为切换实际 EE 后的正式 stats；对齐并新增字段后需要重新统计。
旧字段、旧 RoboTwin、LIBERO 和 Plus 实验行为均保持不变。

**后续用户明确授权变更：删除旧 FK 字段。** 已从全部27,500条 parquet 删除
action.endpose 和 observation.state.endpose，并清理 meta/info.json、
meta/episodes_stats.jsonl 中对应条目和 parquet 内嵌 pandas 列说明。
6,075,103行保留字段逐文件回读比较一致；原 joint、索引、时间戳、RGB和语言未修改。
执行脚本 scripts/remove_robotwin_fk_endpose.py，报告位于
data/robotwin2.0/c2r_clean50/removed_fk_endpose_report.json。
此授权覆盖上文“旧字段保留”的决定：旧 RoboTwin FK-based 训练配置现在不能直接读取
已删除字段；历史权重、旧 stats 仍保留，LIBERO/Plus 数据未修改。
新实际 EE 字段尚未添加，不能在此中间状态启动 Rothko 训练。

- 50 任务联合训练，每任务仅使用 FastWAM 发布数据中的 50 条 clean，合计 2,500 episodes。
- 不留独立验证集；固定训练样本用于诊断，不能解释为泛化指标。
- 先训练原始 Wan2.2 5B，之后训练原始 Wan2.1 1.3B；均为 video DiT 全微调，原始 VAE 冻结。
- 当前帧加未来 16 步，RGB/Rothko 分别编码，latent 时间分段拼接；沿用 rgb_then_raymap_block_causal。
- center_frac=0.5；legacy 解码；推理预测 RGB 和 Rothko；replan=8，不做 ensemble。
- 10 epochs，有效 batch=128，AdamW lr=1e-4、betas=(0.9,0.95)、weight_decay=0.01；cosine、warmup_ratio=0.05，bf16。
- weights：3、6、9、10 epoch；完整 state：5、10 epoch。间隔按 trainer 的 optimizer-step epoch 定义推导。
- 正式测试仅 demo_randomized，每任务 100 trials。

## 新配置

- configs/task/robotwin_c2r_rothko_centerfrac05_full_wan22_5b_1e-4.yaml
- configs/task/robotwin_c2r_rothko_centerfrac05_full_wan21_1_3b_1e-4.yaml
- configs/sim_robotwin_c2r.yaml

初始显存测试设置为四卡 batch8/GA4（用户2026-09-07确认），有效batch128，保留 gradient checkpointing，尚未实测吞吐或显存。诊断每 1000 steps 一次，W&B 的 eval 字段在本配置下改名为 train_diagnostic。内部指标字典仍沿用 val_loss 名称。

## 数据和统计

新参数 robotwin_data_variant 默认 all；clean 选择每个任务 550 条中的前 50 条，randomized 选择后 500 条。dataset 和语言缓存预计算共用筛选函数。

已对 click_alarmclock 官方 clean50 和 FastWAM 前 50 条做完整关节轨迹匹配：50/50 匹配。官方文件按字典序重排；state=q[:-1]，action=q[1:]；每 episode 少一行。其他任务边界尚需全量抽查，不能把这一验证当作 50 任务逐轨迹验证。

Rothko stats 的新命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/compute_rothko_norm_stats.py \
  --robotwin-data-variant clean --windows-per-episode 0 --workers 4 \
  --output data/robotwin2.0/c2r_clean50/rothko_q99p95_all_windows_h16.pt
```

windows-per-episode=0 显式选择全部起点，尾部按 dataset 重复末帧。以当前 EE 坐标系表示未来位移，统计两臂三轴绝对位移 Q99.95，用对称上下界。方向采用既有固定范围。使用 100000 bins、[0,1m] 直方图，属于近似分位数，分辨率 0.01mm；记录 overflow 和最大值，发布统计前检查是否溢出。PT/JSON 保存几何、窗口数及选择的 episode IDs。旧的 32 窗口默认行为不变。

通用 processor stats 的 pretrained_norm_stats=null，仅在所选 clean 上计算。旧混合数据 stats 不作为新实验输入。

本次统计已完成：2,500 episodes、549,787 窗口、17,593,184 个相对位移向量。Q99.95 三轴对称界限为 ±[0.26525, 0.25850, 0.25555] m；观测最大绝对位移为 [0.59573, 0.80503, 0.72976] m，均未超出直方图 1m 范围。CPU 统计阶段耗时 13.9 秒（不含 Python 导入和启动）。已验证总窗口数、episode 唯一性及尾部重复行为。

## 兼容性

- LIBERO、Plus 配置不修改。
- 已对比 LIBERO all4/Wan2.1 与 Plus all4/Wan2.2 配置：语言缓存选择、目录及默认开关与 HEAD 一致。
- 旧参数未启用 save_every_epochs、state_save_every_epochs 时沿用 step 保存。
- RoboTwin manager 默认仍先 clean 后 random；新 sim_robotwin_c2r 配置只跑 random。
- 尚未做完整 GPU 训练/保存/续训回归测试，不能以配置解析代替运行测试。

## 正式运行前未完成项

1. 检查全 50 任务的 clean 边界并冻结来源清单。
2. 量化训练 drive-target FK 状态与官方实测 EE、部署 EE 的差异。目前不修改已有 endpose 字段。
3. 验证 stats 完整性、溢出和 codec 编解码；检查 clean 语言缓存覆盖及 prompt 一致性。
4. 核对标准任务步数限制：当前 click_alarmclock=100 是历史诊断设置，不能直接用于正式基准；修改必须限定新评测入口。
5. 跑短训练，覆盖固定训练样本诊断、weights、完整 state 以及恢复；再确定显存优化与 latent 预计算。
6. 正式评测前补齐逐 trial 恢复、seed/语言记录和线程设置，并检查默认任务列表覆盖 50 项。

本阶段没有启动正式 GPU 训练，也没有修改 LIBERO/Plus 评测流程。

## EE 语义核对（2026-09-07）

核对范围仅为官方 click_alarmclock clean50，按已验证的字典序映射至 LeRobot episodes 2200–2249。
再次确认 observation.state 与官方 joint_action/vector[:-1] 一致（atol=1e-6）。
比较的是同一记录时刻 observation.state.endpose（控制目标 FK）与官方
endpose/left_endpose、endpose/right_endpose 的实际 EE，**不是模型预测误差，也不是 FK 算法误差的直接证明**。
两臂分别计数，共 8,504 个 arm-frame。位置使用欧氏距离；旋转使用归一化 wxyz 四元数的
2*acos(abs(dot(q1,q2)))，转换为度；四元数计算使用 float64。

| 指标 | 位置 mm | 旋转 ° |
| --- | ---: | ---: |
| 平均 | 1.3072 | 0.2640 |
| P95 | 3.2525 | 0.4960 |
| P99 | 27.4156 | 4.5171 |
| 最大 | 29.5877 | 11.6211 |

327 个 arm-frame 位置误差超过 10mm；66 个旋转误差超过 5°，两项不一定发生在相同帧。
最大旋转位于官方 episode16 / LeRobot2208 的左臂第48帧；第46–50帧均约11.6°，不是孤立数值尖峰。
最大位置位于官方 episode49 / LeRobot2244 的右臂第67–68帧。

现有训练以目标 FK 作为 chunk 当前坐标原点，部署则以实际 EE 为原点，存在已量化的语义差异。
这不意味着应把未来 action 改成实际 EE：动作控制目标和观测状态应分别讨论。
本轮仅更正 add_gripper_pose.py 对 state 的误导性注释，未改 parquet、FK、stats 或部署语义。
是否补充实际当前 EE 作为新 C2R 专用观测字段，需要用户决定；不能静默改变旧实验。

步数限制核对：两份本地 RoboTwin（/mnt/hwdata/cfy/RoboTwin 与 /mnt/hwdata/cfy/wam/RoboTwin）
均为 click_alarmclock=400。当前 third_party 版本为100；与前一份本地文件相比，整个 YAML
仅这一项不同。当前尚未修改共享文件，也不将本地副本冒充已在线核验的官方版本。

短程 GPU 训练/保存/恢复测试尚未执行；先说明上述语义差异，再决定是否按现有字段进行冒烟测试。

### 两份发布数据的直接对齐验证

使用 scripts/audit_robotwin_hdf5_alignment.py 只读比较已下载的 click_alarmclock clean50
HDF5 与 FastWAM LeRobot episodes2200–2249；报告保存为
data/robotwin2.0/c2r_clean50/click_alarmclock_hdf5_alignment_audit.json。
检查全部50条的关节数据及三个相机视频；像素比较统一缩至160×120，使用共同内部帧，
比较原始 HDF5 第 t-2、t-1、t、t+1、t+2 帧。不要求不同压缩格式逐像素相等。

- state[t] 与官方 joint_action[t] 最大误差2.73e-8；action[t] 与官方 joint_action[t+1] 相同精度。
- action[t] 与 state[t+1] 完全相等；对应 FK endpose 也完全相等（不含最后一行的跨界比较）。
- 50 episodes × 3 cameras，全部150个视频的平均像素误差都在 offset=0 时最小。

| 相机 | 对官方 t-1 的 MAE | 对官方 t 的 MAE | 对官方 t+1 的 MAE |
| --- | ---: | ---: | ---: |
| head | 4.500 | 3.154 | 4.515 |
| left wrist | 6.698 | 4.894 | 6.682 |
| right wrist | 7.896 | 5.281 | 7.899 |

MAE 单位为0–255像素值。这个实验支持 FastWAM RGB[t] 与官方同条记录RGB[t] 对齐，
没有整体前后错一帧的证据；结合官方同记录采集 EE 的实现，可将官方 endpose[t]
对到 FastWAM 第 t 行作为实际状态。不能将这一结论泛化为另外49任务已经逐条验证。
HDF5 实际 EE 与 FK(state[t]) 的差异不能通过整体平移一帧消除。

### 官方 clean50 下载

用户已授权保留全部原始 ZIP；只下载50个任务的 aloha-agilex_clean_50.zip，不下载 randomized。
2026-09-07 已在 tmux 会话 robotwin_clean50_download_20260907 启动：

```bash
source /home/cfy/clashctl/scripts/cmd/clashctl.sh
clashon
python -u scripts/download_robotwin_official_clean50.py
```

根目录 data/robotwin2.0_official_aloha_clean50；ZIP 在 raw_zips/<task>/，
解压数据在 extracted/<task>/。manifests/ 下保存固定 HF revision 的下载清单、
逐任务状态和 download.log。远端50个 ZIP 总计22.15 GiB，另需解压空间。
单连接顺序下载（上限20 MiB/s），有限重试、断点续传，按远端 SHA256 校验 ZIP、按 CRC 校验解压。
已有 click_alarmclock 经校验后复用。下载启动不代表全量对齐或新字段接入已经完成。

## 当前实施结果：独立实际 EE clean 数据集（2026-09-07）

本节覆盖前面尚未完成的状态说明。官方50个ZIP均已下载、SHA256校验、解压完成，原始ZIP保留。
全量核对50任务×50条共2,500条：FastWAM state/action 分别匹配官方 joint_action[:-1]/[1:]，
夹爪也匹配对应官方 endpose gripper，最大差2.9653e-8；无长度、索引或四元数有效性异常。
这是全量关节/EE字段核对；RGB逐视频偏移审计目前仍只覆盖 click_alarmclock，不能混为全量RGB审计。

已生成 data/robotwin2.0_c2r_clean50，约200MiB（不跟随视频软链接）：

- 2,500条 parquet / 549,787行；episode、global index、language index 重编号为连续索引。
- observation.state.ee_pose_wxyz = 官方实际 endpose[:-1]。
- action.ee_pose_wxyz = 官方实际 endpose[1:]。
- 两臂14维 xyz+wxyz，只转float32，不做FK或额外四元数符号变换。
- 原关节/夹爪数值、帧索引、时间戳、语言文本不变；7,500条视频软链接复用源视频。
- meta/c2r_source_manifest.json 保存源episode映射、官方HDF5路径和固定下载revision等来源。
- 所有写入均在新目录；原27,500条数据未因本次补字段发生变化。

构建命令（输出已存在时拒绝覆盖）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/hwdata/cfy/miniconda3/envs/fastwam_robotwin/bin/python \
  scripts/build_robotwin_c2r_actual_ee.py --workers 4
```

两种 C2R backbone 配置均已通过继承切到新目录；robotwin_data_variant=all 是指取独立clean目录全部数据，
**不是重新混入randomized**。robotwin_task_names=null，避免对2,500条紧凑编号套用旧550条/task映射。
dataset 新增 robotwin_ee_pose_key=ee_pose_wxyz 显式读取开关；旧默认仍为endpose，LIBERO/Plus逻辑不变。
新EE来源加入本配置的latent cache contract，不修改旧配置的contract。

新 Rothko stats：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python scripts/compute_rothko_norm_stats.py \
  --dataset-root data/robotwin2.0_c2r_clean50 --ee-pose-key ee_pose_wxyz \
  --windows-per-episode 0 --workers 4 --center-frac 0.5 \
  --output data/robotwin2.0_c2r_clean50/rothko_actual_ee_q99p95_all_windows_h16_centerfrac05.pt
```

全部窗口含末帧重复padding，H16、Q99.95、center_frac=.5；对称三轴界限
±[0.26561,0.25847,0.25293]m。旧目标FK stats 的 ±[0.26525,0.25850,0.25555]m 保留但不再用于C2R。
原始N帧对应N-1行的关系不变；未来EE最后一行可使用官方最终EE，RGB尾部仍沿用既有视频padding。

验证：3项单元测试通过（EE时间对齐与源字段保留、错误源轨迹拒绝、双backbone配置）。
实际CPU构建全量训练dataset，长度549787；样本0/10001/20002均成功读取真实语言缓存、
RGB和Rothko [3,17,384,320]、current EE [14]、future EE [16,14]，Raymap数值有限。
测试及通用processor stats位于 runs/robotwin_c2r_actual_ee_dataset_check；未启动模型训练。

语言表含137,077条clean指令，原目录 data/text_embeds_cache/robotwin 中覆盖11,305条，
尚缺125,772条。这里检查的是文件覆盖，不等同于逐文件内容身份校验；部分已有文件为无metadata旧格式。
正式训练前需要另行补齐语言缓存，不能忽略这个缺口。新数据字段接入不包含这项预计算。

2026-09-07 用户授权在GPU4,5,6,7补齐上述语言缓存。已启动tmux
robotwin_c2r_text_cache_4567，启动脚本及日志分别为
runs/robotwin_c2r_text_cache_20260907/launch.sh 和 precompute.log。
使用torchrun四进程、每卡batch16、BF16、context_len128，显式+overwrite=false；
缓存仍写入data/text_embeds_cache/robotwin，不覆盖已有文件、不改HF_HOME等缓存根路径。
待补125,772条，预计新增约123GiB。正式模型训练尚未启动；本条为启动记录，不表示预计算已完成。

### 语言缓存完成与短程验收安排（2026-09-07）

上述缓存任务已经完成：new=125772、skip=11305、overwrite=0，137077条clean指令全部有缓存。
用户同意按短程训练、诊断、保存、恢复的顺序验收，并将初始配置改为四卡batch8/GA4。
短程测试使用独立runs目录及命令覆盖，不修改正式10epoch预算及epoch保存频率。
计划20步，第18步执行训练样本诊断并保存weights和完整state，继续到20步；随后从第18步state
在独立输出目录恢复至20步，保持相同总步数/调度预算，检查恢复状态和后续loss。
本安排不是已经通过测试的结论；正式长训练仍待验收。

短程任务已在GPU4–7、tmux `robotwin_c2r_smoke_4567` 启动。脚本/日志位于
`runs/robotwin_c2r_smoke_20260907/launch.sh`、`initial.log`，输出位于 `initial/`。
3项单元测试通过；11:45 UTC 已完成前2步，未见OOM，nvidia-smi观测显存约57–61GiB/卡。
目前只验证了训练前两步，尚不能宣布诊断、保存和恢复成功。20步短测的5%warmup只有1步，
loss曲线不能用来判断正式10epoch训练的收敛质量。正式训练的warmup预算未改。

11:53 UTC initial已完成20步：第18步四卡诊断通过并输出4个视频，保存weights以及完整state文件集合
（模型、四份ZeRO优化器分片、scheduler、四份RNG、trainer_state.json）。最后两步loss为0.6090/0.5493。
退出时出现NFS multiprocessing临时目录清理EBUSY；该错误出现在训练完成后，不是OOM。
随后在GPU4–7的 `robotwin_c2r_resume_4567` 启动18→20步恢复测试，日志 `resume.log`，
输出 `resume/`；本次单独设TMPDIR=/tmp避免NFS临时文件清理问题，不改全局环境。
恢复测试尚在运行，不能仅根据文件齐全宣称完整恢复通过。

### 短测验收与全量latent缓存（2026-09-07）

恢复测试已完成18→20步：模型、四卡优化器、scheduler、RNG和数据位置均加载成功。
第19步loss连续/恢复均0.6090，第20步0.5493/0.5494；学习率一致，未宣称逐bit续训复现。

随后四卡对同一4096窗口做两轮缓存测速，使用原始Wan2.2 VAE、BF16、每卡batch8：

| workers/卡 | 分片窗口数 | 编码及校验耗时（不含加载） | 全量外推 |
| --- | --- | --- | --- |
| 4 | 256 | 380.893秒 | 14.2小时 |
| 8 | 1024 | 274.554秒 | 10.2小时 |

两轮均通过indices=[0,2048,4095]在线/缓存latent逐bit检查和固定种子的训练loss完全一致检查。
机器负载与文件缓存可能影响速度，不能把全部差异归因于参数。

用户授权4567四卡全量计算，并把分片再增至4096窗口。全量549787窗口，预计约236GiB；
135个分片，每片最多约1.76GiB。每卡batch8、workers8不变；分片是磁盘组织单位，不是GPU batch。
输出 `data/robotwin_c2r_clean50_wan22_centerfrac05_bf16_h16_latents`，
脚本/日志 `runs/robotwin_c2r_latents_full_20260907/{launch.sh,precompute.log}`，
tmux `robotwin_c2r_latents_full_4567`。不使用benchmark样本上限，不覆盖两轮测速目录。
Rothko stats仍为新实际EE all-windows centerfrac05；通用processor stats复用本次clean短测生成的文件。
保持默认编码实现与完整性检查；可重启复用已完成分片，未完成分片重算。
预计10–12小时，待全量实测校准。完成并校验前不接入正式训练。

用户随后授权在GPU0–3并行启动Wan2.1全量预计算。任务配置为
`robotwin_c2r_rothko_centerfrac05_full_wan21_1_3b_1e-4`，使用原始Wan2.1 VAE，
同一549787窗口、实际EE stats、centerfrac05；batch8/workers8/shard4096。
独立输出 `data/robotwin_c2r_clean50_wan21_centerfrac05_bf16_h16_latents`，
脚本/日志 `runs/robotwin_c2r_latents_wan21_full_20260907/{launch.sh,precompute.log}`，
tmux `robotwin_c2r_latents_wan21_0123`。不修改运行中的Wan2.2任务。
Wan2.1每模态latent预期[16,5,48,40]，两模态BF16全量约314.6GiB，不能混用Wan2.2缓存。

### 内存上限与3+5卡续算

两组同时计算时，64个DataLoader worker实际占用很大；发现原先针对LIBERO小词表设计的
`_text_context_memory_cache`没有淘汰机制，不适合RoboTwin的137077条语言。
用户授权停止、修复、换卡续算。停止后系统used降至约35GiB，GPU全部释放。
新增可选 `text_context_cache_max_entries`：默认None保留旧行为，0不缓存，正数采用LRU上限。
仅新C2R配置显式256，Wan2.1继承，语言数值/磁盘缓存不变；测试覆盖None/0/2三种模式及淘汰后重读数值。
这修复的是明确的无上限缓存问题，不代表其他内存来源已经被完全排除，续算仍需观察。

预计算现在由rank0冻结所有未完成分片清单后广播，再按world_size均分，避免换卡后负载不均或重复分配。
保留Wan2.2完成的67片、Wan2.1完成的32片；删除两组各4个不可续用的partial，完整bin未动。
Wan2.2使用GPU4,5,6，tmux `robotwin_c2r_latents_wan22_3gpu`，脚本/日志原run下 `resume_3gpu.sh/.log`。
Wan2.1使用GPU0,1,2,3,7，tmux `robotwin_c2r_latents_wan21_5gpu`，脚本/日志原run下 `resume_5gpu.sh/.log`。
每卡batch8、workers8、分片4096保持不变；缓存身份检查不变，完成后仍执行latent/loss一致性检查。

2026-09-08检查：上述两组在收尾barrier触发NCCL默认600秒超时（SIGABRT），不是日志显示的OOM。
Wan2.2保留134/135片，缺index94；Wan2.1保留133/135片，缺index56、113；已有文件长度核对通过。
新增仅预计算脚本参数 `--distributed-timeout-seconds` 默认7200，并在初始化process group前设定CUDA设备。
用户授权继续。因GPU0–3已有其他任务，使用GPU4补Wan2.2、GPU5/6补Wan2.1，分别是
tmux `robotwin_c2r_wan22_finish` / `robotwin_c2r_wan21_finish`；启动脚本和新日志为原run下
`finish_gpu4.sh/.log`、`finish_gpu56.sh/.log`。完成补片后仍需最终数值校验和_SUCCESS，当前不宣称完成。

### 2026-09-08 后续核验与7短任务正式实验

两套缓存已完成：Wan2.2与Wan2.1均135/135分片、549787窗口，具有_SUCCESS及complete=true；
抽样在线latent与缓存、固定噪声loss一致性检查通过。体积约235.94GiB / 314.59GiB。

8卡有效batch128的100步测速与最终训练样本诊断：

| 配置 | 稳定秒/step | nvidia-smi采样峰值 |
| --- | ---: | ---: |
| batch4/GA4，检查点关闭 | 6.13 | 76.8GiB |
| batch8/GA2，30层检查点开启 | 7.26 | 47.3GiB |
| batch8/GA2，20层开启 | 6.72 | 61.4GiB |
| batch8/GA2，10层开启 | 6.36 | 78.7GiB |

batch8/GA2全关在第一步前向OOM。选择batch4/GA4全关。
此方案另通过12步训练/4样本诊断/weights及完整state保存，采样峰值76.82GiB。
中途state恢复对照：16步总预算，14步保存再恢复到16；模型、优化器、scheduler、RNG及
epoch0/batch56/sample_offset1792恢复。15步loss均0.6649；16步连续0.5963、恢复0.5962，
LR一致，不声称逐bit相同。记录位于runs/robotwin_c2r_cache_train_bench8_20260908。

用户最终选择7个短任务（替代此前3任务候选，不启动3任务实验）：

| task | clean episodes | windows |
| --- | ---: | ---: |
| click_bell | 50 | 3855 |
| click_alarmclock | 50 | 4252 |
| grab_roller | 50 | 4728 |
| turn_switch | 50 | 4863 |
| lift_pot | 50 | 5554 |
| beat_block_hammer | 50 | 5682 |
| move_playingcard_away | 50 | 5884 |
| 总计 | 350 | 34818 |

配置：configs/task/robotwin_c2r_7task_rothko_centerfrac05_full_wan22_5b_1e-4.yaml。
数据源仍为data/robotwin2.0_c2r_clean50，使用其已完成的Wan2.2缓存、原始Wan2.2 5B及原始VAE，
从预训练模型重新开始，不加载任何测速权重。语言缓存、实际EE、全clean统计出的Rothko stats不变。
新专用RobotWinC2RTaskSubset先验证完整数据/缓存contract，再将子集窗口映射到原完整缓存索引；
不使用旧每任务550episode的task映射，不复制缓存，不改变原Dataset默认行为。
子集内自然窗口采样，尾部padding保留，错误替换限制在所选任务内。
预检报告：runs/robotwin_c2r_7task_preflight/report.json，已核对7任务首尾窗口及train/diagnostic同索引。

8卡0–7，batch4/GA4，有效batch128，GC关闭，10epochs=2730step；每epoch273step。
lr1e-4/cosine/warmup5%/wd0.01，bf16，ZeRO1，workers4，W&B online。
新可选eval_every_epochs=1沿用现有epoch步数计算；旧配置不设置时保持原eval_every。
不设held-out：每273step诊断原逻辑的固定4个训练样本，不能将该指标当作7任务rollout成功率。
weights:819/1638/2457/2730；state:1365/2730。预计纯训练4.65小时，含保存诊断约5小时。
tmux：robotwin_c2r_7task_train8。
run：runs/robotwin_c2r_7task_rothko_centerfrac05_full_wan22_5b_1e-4/2026-09-08_06-20-00。
启动命令保存在该run的launch.sh，日志train.log；后续评测仅对应7任务demo_randomized，尚未启动。
