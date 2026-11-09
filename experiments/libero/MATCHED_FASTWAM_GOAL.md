# Goal: Wan2.1 Video Model + Action Expert 对照

独立于既有 FastWAM / Action Raymap 工厂、数据集和评测入口。目标参照结果：
`evaluate_results/libero/closed_loop/goal_absolute_ray0_wan21_step4140_vae8264_replan8_ep50_20260923_202057`。
该模型为 DiT4140 + 四-suite decoder8264，Goal491/500；同一DiT原始VAE的已有结果为465/500。后者可用于区分额外decoder微调的影响。

## 固定设置

- Wan2.1 T2V 1.3B Video DiT，原始冻结 Wan2.1 RGB VAE；额外Action Expert hidden1024、FFN4096、30层、12头、head_dim128。Video/Action专家和proprio投影均训练，保留原始FastWAM MoT、loss计算与action-only推理实现。
- Goal433个episode、52,895个窗口；当前 + 16帧未来RGB，224×448双视角；连续16步动作。RGB每步采样，不使用4帧稀疏监督。
- 动作8维 `[relative_xyz, relative_quaternion_wxyz, gripper]`，相对chunk当前实测EE；当前状态8维为绝对EE位姿和夹爪，仅当前状态输入proprio投影，无未来信息。
- 位置bounds和clip沿用参照实验；四元数归一化并规范符号；夹爪0到1映射至负1到1。该数值归一化复用已验证Latent-Repeat的vectors函数，但不执行repeat、不生成动作latent、不调用动作VAE。
- 10epoch、预计4140 optimizer steps；4卡每卡batch4、GA8，有效batch128；AdamW betas(0.9,0.95)，LR1e-4、warmup5%、cosine、WD0.01、grad norm1、BF16、seed42；沿用基线scheduler shifts5。
- RGB/action loss权重1:5，对齐本次对照预算；原始FastWAM默认通常为1:1，因此本实验是目标与预算对齐的适配，不能声称逐项复现原论文默认设置。
- 每500步验证同一训练数据集前4个固定窗口（不是held-out验证），仅报告固定噪声/时间步的loss；每个epoch保存可评测权重（当前每414步，共10份）；每2000步及最终保存完整续训state。验证不生成未来RGB或闭环rollout，验证耗时不与参照实验直接比较。
- 原版FastWAM仅让动作读取当前视频，不读取未来视频，沿用其原生MoT mask和action-only推理；参照Raymap仍联合去噪未来RGB与动作latent。这是方法差异，不声称两边计算量、参数量或墙钟预算相同。
- 测试Goal10任务×50场、seed42、replan8、预测16步、20步去噪、无ensemble、30步初始化等待、相同任务时限及夹爪阈值。数值目标先还原绝对目标位姿，再复用同一absolute-target-to-OSC helper。
- 参照98.2%的decoder使用四-suite额外数据和训练；本对照不消费动作图像decoder，不能称为相同额外训练数据预算下的纯结构消融。

## 缓存和隔离

训练逐位复用 `data/libero_goal_rothko_absolute_ray0_wan21_bf16_h16_latents_20260921` 中RGB部分。该缓存具有相同原始VAE、17帧及224×448布局。动作latent被丢弃，数值目标从原始parquet构造；文本复用 `data/text_embeds_cache/libero_wan21_bf16_batch1`。不覆盖旧缓存，不修改已有stats。

数值目标来自同一窗口、同一OSC重建目标，尾部edge-repeat与原mask一致，不监督padding动作。checkpoint保存独立身份、模型维度和stats/VAE指纹，拒绝不匹配权重。新实验使用独立run目录。

## 准备和启动

先运行CPU小规模数据核对（不加载大模型）：

```bash
PYTHONPATH=src:. python scripts/audit_matched_fastwam.py --output evaluate_results/libero/preparation/matched_fastwam_goal_check.json
```

Action Expert初始化不能用旧Wan2.2文件，需从原始Wan2.1重新生成。准备脚本默认只打印计划，显式execute才读取大权重并写入新文件；不会覆盖已有文件：

```bash
python scripts/prepare_matched_fastwam_backbone.py
python scripts/prepare_matched_fastwam_backbone.py --execute
```

按原版方式对Action Expert backbone做逐维线性插值与alpha scaling，action encoder/head保持随机初始化；来源模型写入文件，训练工厂明确拒绝Wan2.2来源。已于2026-11-07生成，文件约1.20GiB；耗时58.82秒，峰值RSS13.83GiB，未使用GPU。参数形状、来源、820个backbone张量的BF16与有限性检查通过。记录：`evaluate_results/libero/preparation/matched_fastwam_init_20261107_153148/validation.json`。

正式训练使用专属入口，不能用共享train_zero1.sh替代（后者没有本实验的数值目标验证器）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_matched_fastwam_4gpu.sh
```

测试单任务（请使用本实验新权重）：

```bash
python experiments/libero/eval_matched_fastwam.py ckpt=/absolute/path/to/new/step_004140.pt EVALUATION.task_id=8
```

全Goal500场：管理器默认仅打印计划，execute才启动，每卡一worker、完成后补下一个任务，12h单任务上限；失败会退出并清理自己的子进程。

```bash
python experiments/libero/run_matched_fastwam_manager.py --checkpoint /absolute/path/to/new/step_004140.pt --output /absolute/path/to/new/results --gpu-ids 0 1 2 3 4 5 6 7 --execute
```

不要使用旧run_libero_manager.py启动本实验，旧worker不知道这里的8维数值目标恢复规则。

## 当前校验范围

诊断在 `evaluate_results/libero/preparation/matched_fastwam_goal_20261107/`。

- 配置可解析；原始FastWAM及Raymap配置未修改。
- 五个真实窗口覆盖开头、中段、15步padding的尾部；RGB/text/padding一致，数值恢复位姿与直接Raymap解码一致。
- CPU缩小模型使用真实MoT/VideoDiT/ActionDiT代码：cached/online输入构造一致；不调用VAE encoder的缓存loss、前向反向、两步动作推理和checkpoint隔离检查。
- 使用真实LIBERO评测helper、模拟数值推理输出验证：当前状态输入一致，恢复的16步OSC命令与原Raymap控制路径一致；不创建仿真环境。`cpu_tests_recheck.log` 4项及 `eval_helper.log` 1项通过。
- 正式Action Expert初始化权重已生成；完整尺寸GPU前后向、4卡ZeRO完整state恢复和真实环境rollout尚未验证。完成这些检查后再启动正式训练。

## 第二轮代码复查

复查记录：`evaluate_results/libero/preparation/matched_fastwam_goal_review_20261107/`。

- 修复专属评测入口的完整启动链：共享LIBERO入口会按Raymap数据标记应用预训练加载分支，删去Action Expert初始化参数。现在仅在专属入口的局部factory wrapper中恢复必要参数，并明确从完整checkpoint加载两个Expert，不再依赖初始化文件；finally恢复所有局部hooks。
- 评测显式传递inference seed和sigma shift，拒绝当前action-only实现不支持的text CFG/negative prompt，避免静默忽略配置。
- 新管理器收到SIGTERM时也进入自己的子进程清理流程。
- 6项模型CPU测试与2项评测接口CPU测试通过，包括完整评测启动到factory的参数处理。完整规模结构和所有backbone key的插值形状在meta device验证（不分配大模型权重、不替代真实GPU测试）。
- 与指定已保存训练run的batch/GA/epoch/LR/调度、数据窗口、图像尺寸、stats、RGB及文本缓存设置逐项核对通过。

2026-11-07 用户确认改为4卡训练：每卡batch4、GA8，有效batch仍128，10epoch预计4140步。GPU具体编号在启动时通过CUDA_VISIBLE_DEVICES指定；训练尚未启动。原8卡Raymap参照实验配置不变，墙钟时间与峰值显存需按4卡重新实测。

保存策略更新：用户确认每epoch保留一份可评测weight，save_every_epochs=1；state_save_every=2000与最终state保存保持不变。当前预计weight步数为414、828、1242、1656、2070、2484、2898、3312、3726、4140。

## 按原版参数比例缩小的 Action Expert 初始化（2026-11-08）

独立模型配置：`configs/model/fastwam_matched_wan21_ratio20.yaml`。Action Expert hidden560、FFN2240，保留30层、12头及head_dim128，完整参数量287,376,248，占Wan2.1 Video DiT的20.2521%；原版Wan2.2配置对应比例为20.4186%。此前1024宽度版本为642,939,912参数，占45.3095%，其配置、初始化及训练结果均保留。

新初始化：`checkpoints/ActionDiT_matched_Wan21_560hdim_pose8.pt`。直接从原始Wan2.1 backbone逐维插值并应用相同alpha scaling生成，BF16，820个backbone张量；与原流程一致，动作输入层及输出head在模型加载时随机初始化。该文件是未训练的backbone初始化，不是已微调checkpoint。

```bash
python scripts/prepare_matched_fastwam_backbone.py \
  --model-config configs/model/fastwam_matched_wan21_ratio20.yaml --execute
```

准备脚本原默认配置仍是1024宽度版本，已有输出会被拒绝覆盖。新文件全部张量形状、BF16与有限性检查、真实pretrained loader加载、加载后backbone逐位核对、完整30层ActionDiT CPU前后向通过；未执行optimizer更新，未使用GPU。旧1024初始化及旧模型配置SHA256均保持不变。记录在 `evaluate_results/libero/preparation/matched_fastwam_ratio20_init_20261108/validation.json`。本次没有启动287M模型训练；完整MoT训练与续训恢复短测需在训练前另外执行。

2026-11-09 用户授权8卡训练比例版。独立task
`libero_goal_matched_fastwam_ratio20_wan21_8gpu`，bs4/GA4有效128，10epoch/4140步，
每epoch保存可评测权重，完整state每2epoch（828步）＋最终保存，验证每500步。
所有梯度检查点关闭。tmux `goal_fastwam_ratio20_8gpu_20261109_023951` 先做
独立2步训练/验证/完整state保存与step1→2恢复，检查通过后自动进入正式训练。
启动记录 `evaluate_results/libero/launch_records/matched_fastwam_ratio20_8gpu_20261109_023951/`；
全新正式目录 `runs/libero_goal_matched_fastwam_ratio20_wan21_8gpu/20261109_023951`。
原4卡方案没有启动；旧1024宽度训练和其他实验未改。

8卡完整训练/验证/保存及step1→2恢复检查通过，已进入正式加载阶段。
恢复后的scheduler、CUDA RNG和trainer进度一致；BF16参数存在极小非逐位差异，
MoT最大绝对差0.00024414、差异RMS占更新RMS约2.22%，未声称逐位确定性。
正式阶段使用全新原始初始化，未沿用smoke的参数更新。

正式训练已进入step10/4140，W&B `8lr59cu6` 已连接；起始约0.25 step/s，
总预算先估4.5小时、每2epoch约55分钟＋保存/验证时间。
