# 原版 FastWAM 5B 真机对照

此实验使用仓库 `FastWAM` 的视频专家 + 独立 `ActionDiT` + `MoT`，与改造版 Rothko 实验分开。
训练损失、attention mask 和推理公式直接继承原版；新增代码仅负责真机数据、统计绑定和验证。
正式实验改为每个任务独立训练，顺序为 pick_place / insert_pen / stack_bowls_three / white_board，epoch 分别为 8 / 8 / 8 / 5。

## 固定配置

| 项目 | 原版真机对照 |
|---|---|
| 视频模型 | 原始预训练 Wan2.2 TI2V 5B |
| 动作模型 | `ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` 初始化 backbone，机器人动作输入/输出层按原版逻辑新建 |
| 训练参数 | 全量视频/动作专家及 proprio encoder；冻结 VAE 和文本编码器 |
| Loss | 原版 `video=1, action=1` |
| 输入相机 | 左侧、腕部，各 224×224，横向拼接 224×448 |
| 数据划分 | 与成功的改造版同一份 q99.95 实验 split，再按任务过滤；每个任务固定两条验证窗口 |
| 窗口 | pick 19,042/494；pen 30,266/879；bowl 26,940/501；whiteboard 20,354/536（训练/验证） |
| 时间范围 | 当前 + 未来 16 帧，10 Hz，不跨轨迹、不 padding |
| Epoch / LR | 8 / 8 / 8 / 5；`1e-4`，cosine，warmup 5% |
| Batch | 每卡 4，累积 4；8 卡时有效 128；更新数 1,192 / 1,896 / 1,688 / 800 |
| 梯度检查点 | video expert、action expert、MoT mixed attention 全部开启 |
| 保存 / 验证 | 每 1,000 步保存权重与恢复状态；每 500 步验证，结束保存 |

输入时间范围特意与真机改造版保持一致，采用 17 视频帧、16 动作；不是原 LIBERO 配置里的 9 视频帧、32 动作。
两组实验的骨干分别是 5B 与 1.3B，因此这是原版系统对照，不是只改变动作表示的单变量消融。

## 状态和动作

- 当前状态通过原版 proprio encoder 输入：`[基坐标 xyz, qw, qx, qy, qz, gripper]`，8 维。
- ActionDiT 预测 `[相对 xyz, 相对旋转向量 xyz, gripper]`，7 维。没有 RAY0/Rothko 图和 Rothko loss。
- 所有未来目标相对于窗口第 0 帧的末端坐标系，`p_rel=R0^T(p_future-p0)`、`R_rel=R0^T R_future`，不是相邻帧差分。
- 平移采用与成功的改造版相同的四任务训练窗口 q99.95 边界；主值旋转向量按 π 缩放，夹爪从 0/1 映射到 -1/1。
- 绝对位置按训练集边界归一化，四元数使用统一符号。夹爪实测宽度阈值仍为 0.08 m。
- 监督目标是未来实测位姿。旧数据没有遥操作命令，不能还原原始命令标签。

独立统计在 `data/realworld_artifacts/original_fastwam_wan22_5b_left_wrist_val2_g08_q9995_v1/action_stats.json`。旧的 extrema 统计文件保留，不被本实验覆盖。
权重 checkpoint 保存动作约定和统计指纹，加载时严格检查，拒绝误用 Rothko checkpoint 或不同划分。

## 缓存与训练入口

文本编码器和 prompt 相同，可复用改造版的文本缓存。
**原版当前沿用在线冻结 VAE 编码**，不读取改造版的 Wan2.1 RGB/Rothko latent 缓存。
它使用自己的原始 Wan2.2 VAE；本次未为原版增加视觉缓存通路。原版训练的耗时不能套用改造版缓存训练估算。

CPU 准备与测试：

```bash
bash scripts/realworld/workflow_baseline.sh prepare
bash scripts/realworld/workflow_baseline.sh verify
```

每个任务使用独立配置和输出目录。依次运行：

```bash
# 若共享文本缓存已经生成，跳过此步。
CUDA_VISIBLE_DEVICES=0 bash scripts/realworld/workflow_baseline.sh text

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 \
  REALWORLD_TASK=realworld_pick_place_original_fastwam_wan22_5b_ep8 \
  bash scripts/realworld/workflow_baseline.sh train

# 完成后依次换成：
# realworld_insert_pen_original_fastwam_wan22_5b_ep8
# realworld_stack_bowls_three_original_fastwam_wan22_5b_ep8
# realworld_white_board_original_fastwam_wan22_5b_ep5
```

输出目录为 `runs/<任务配置名>/<时间>/`。
入口使用独立 `train_baseline.py` 和 `OriginalRealWorldTrainer`，复用原有优化器/ZeRO/恢复循环；不修改公共模型、公共 trainer 或改造版训练入口。
恢复时传入相应 `resume=...` 和原 `output_dir=...`，遵循原 trainer 的恢复检查。

## 验证与当前边界

2026-09-25 使用 GPU 2--7 做了 6 卡 smoke，配置保持每卡 batch 4、累积 4、在线 Wan2.2 VAE、全量 5B/ActionDiT/proprio 训练和三层梯度检查点，只覆盖为 `max_steps=2` 且关闭验证与保存。模型共 6.73B 参数、训练 6.02B；ZeRO-1 初始化报告约 22 GB CUDA reserved，两个真实前反向更新完成且无 OOM。

同日正式启动 pick_place 8 卡训练，训练器确认 19,042 个训练窗口、494 个验证窗口、1,192 次更新。step 10 速度约 0.13 step/s，ETA 约 2 小时 35 分；8 卡显存约 57--58.5 GB/卡。其余任务将在前一任务完成后按既定顺序单独启动。

smoke 日志在 `evaluate_results/realworld/original_fastwam_5b_smoke6_20260925/train.log`，隔离输出配置在 `runs/realworld_all4_original_fastwam_wan22_5b_ep8/smoke6_20260925/`，没有产生 checkpoint。

每次使用与改造版相同的四个固定窗口，每任务一个，按 rank 分配；没有每次遍历全部验证轨迹。
验证计算原版联合训练 loss，并用原版 `infer_action` 生成 16 步动作，解码为绝对位姿。
记录位置误差（米）、旋转误差（度）、夹爪误差/准确率，以及归一化动作误差；每个窗口保存预测/真值 NPZ。
该原版验证入口不生成 RGB rollout 视频。

CPU 测试包含真实数据与改造版的窗口/图片一致性、动作往返、四元数符号与 π 旋转、数据集序列化、Hydra/factory 配置、checkpoint 身份拒绝。
另使用缩小维度的真实 WanVideoDiT + ActionDiT + MoT 和模拟 VAE，检查原版 loss 前向/反向、开启与关闭梯度检查点的梯度一致性、动作推理，以及四任务验证路径。
这些测试不加载 5B 权重，不代表已验证 8 卡显存、吞吐或真实策略性能。
现有 `serve.py` 仍仅服务改造版。原版使用下面的独立服务入口。

## 原版 FastWAM 真机部署

服务端 `scripts/realworld/serve_baseline.py` 加载训练目录的 config.yaml 和模型权重 checkpoint（不是 DeepSpeed 恢复状态目录），严格核对动作统计和 split。默认 20 步去噪，调用原版 `infer_action` 的视频 KV 缓存路径，预测 16 个相对于当前末端的目标，再用训练时的 ActionCodec 解码为基坐标绝对 fr3_link8 位姿和夹爪开度。

在有空闲 GPU 且已生成 checkpoint 后，从仓库目录运行（替换实际 checkpoint 路径）：

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/realworld/serve_baseline.sh \
  --config runs/realworld_pick_place_original_fastwam_wan22_5b_ep8/20260925_024100/config.yaml \
  --checkpoint /absolute/path/to/checkpoint.pt \
  --port 8010 --steps 20
```

如需预测 RGB，追加 `--save-rgb-dir evaluate_results/realworld/original_fastwam_predicted_rgb`。
这会调用联合 `infer_joint`，耗时会增加；关闭其额外 action-only 对照推理。
按客户端的 session / request / action index 保存前 8 步左相机和双相机 RGB。
原版没有 Rothko 输出，不生成 Rothko 图。保存目录被删除后会自动重建；图片写盘 OSError 不阻断有效动作响应。

工作站复制 `third_party/FastAPI/client_fastwam_ee.py`，在原先的 FastAPI 目录和 roby 环境运行 `python client_fastwam_ee.py`。
SSH 隧道连接远端 8010，校验服务 model_family 为 original_fastwam_wan22_5b。
控制流程继承已经跑通的 v3：左侧/腕部输入、16 步预测执行前 8 步、300ms 最短移动时间、逐动作实际 RGB 保存。
工作站录像与真实帧保存到 `inference_video_fastwam/`；你的方法继续使用 `client_rothko_ee.py` 和端口 8009。
原版客户端保留 10 秒响应超时；实际 5B 延迟需 checkpoint 就绪后的 GPU 推理测量，不能套用 Wan2.1 的速度。

CPU 部署测试已覆盖训练归一化/相对动作解码、真实客户端 metadata/轨迹校验函数、MessagePack NumPy 往返、未来 RGB 文件对齐与错误输入拒绝。
当前尚未完成真实 5B checkpoint 服务加载及工作站联调。
