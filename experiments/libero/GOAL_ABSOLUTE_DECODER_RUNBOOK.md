# Goal 全绝对 decoder 微调：实验 2.5

仅准备，尚未生成缓存或运行 GPU 阶段。原始 Wan2.1 VAE，冻结 encoder/conv1，训练 conv2+decoder；8卡 batch2/GA4，有效 batch64，2epoch，LR1e-5、warmup5%（78步）、cosine至1e-7，FP32 master+BF16 autocast、分片 AdamW。

## 数据与验证

沿用历史 LIBERO decoder 的 split_seed=20260801，每任务留出2个episode。Goal训练413个episode、50,445个窗口（全部frame starts，尾部edge-repeat）；验证20个episode，每episode均匀取10个窗口，共200个，包含首/末frame start。789步/epoch，共1,578步。已有四-suite全绝对stats固定使用，不重新拟合。

固定清单：`evaluate_results/libero/goal_absolute_decoder_preparation_padding_20260919/manifest.json`。同时记录训练窗口对应的DiT缓存索引、验证窗口、配置/VAE/stats/数据契约身份。该实验只允许读取训练清单中的窗口进行反向传播。

复用同表示Goal DiT缓存 `data/libero_goal_rothko_all_absolute_2cam224_wan21_bf16_h16_latents`，含52,895个frame starts；decoder从中筛选50,445个训练窗口（含尾部padding）和200个验证窗口。缓存含验证episode不等于参与decoder拟合。旧相对、混合表示缓存禁止复用。缓存当前尚未生成。

补齐方式：未来动作索引使用 `min(start+j,N-1)`，RAY0仍取真实当前state。补齐帧参与图像重建loss；验证动作误差只统计真实未来步，前8/16步各自排除padding。旧无padding准备清单保留在原目录，不覆盖。

## 命令

以下在仓库根目录执行，使用fastwam环境。`prepare`/`check-cache`只用CPU；`audit`/`train`默认仅打印命令，必须显式加`--execute`。训练前另行确认GPU可用，不停止现有任务。

```bash
export PYTHONPATH=$PWD/src:$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python
"$PY" scripts/prepare_goal_absolute_decoder.py prepare

# 先完成Goal基线新缓存的小规模GPU测速与一致性检查，再生成完整缓存。
# 下面是完整缓存命令示例，不会由prepare自动执行：
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
#   scripts/precompute_visual_action_latents.py \
#   --task libero_goal_rothko_all_absolute_2cam224_full_wan21_1_3b_1e-4 \
#   --output-dir data/libero_goal_rothko_all_absolute_2cam224_wan21_bf16_h16_latents \
#   --batch-size 2 --num-workers 2 --samples-per-shard 1024 --log-every 20

"$PY" scripts/prepare_goal_absolute_decoder.py check-cache
"$PY" scripts/prepare_goal_absolute_decoder.py audit
"$PY" scripts/prepare_goal_absolute_decoder.py train
```

缓存存在后，GPU审计命令加`--execute`：每任务分别从训练/验证池选一个episode，核对首个start、最后一个无padding start和最后一个frame start，共60个probe，要求原始BF16 encoder输出与缓存逐位相等。审计写入`cache_audit.json`，拒绝覆盖；缓存或配置身份变化后旧审计不能授权训练。

正式训练在独立tmux中执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/prepare_goal_absolute_decoder.py train --execute
```

这不是已运行命令。实际启动前仍需GPU小规模验证。新run目录非空会拒绝启动，不隐式续训。需要恢复时，在同一命令末尾追加 `--resume /absolute/path/checkpoint_latest.pt`；要求优化器状态、配置/数据身份和缓存metadata身份一致。

## 保存、验证和输出

- latest完整优化器恢复点：step400、789（epoch1结束）、800、1200、1578（最终）。rolling latest会按历史规则更新。
- 独立历史权重checkpoint：step800、1578。它们按旧规则不含优化器，不能作为完整续训点。
- 导出完整VAE：step800和最终1578；验证在789、800、1578。
- 验证使用冻结缓存latent + BF16部署精度decoder；全绝对legacy解码、anchor0。报告前8/16步位置、四元数测地角和夹爪误差，不使用旧相对decoder。
- 输出：`runs/libero_goal_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2/`；最终导出文件沿用公共trainer命名 `Wan2.1_VAE_libero_rothko_step001578.safetensors`，绝对表示身份记录在metadata和独立run目录中。
- 后续配对实验2.4的最终DiT做闭环评测；本次不启动闭环。

## 隔离与验证

`prepare_goal_absolute_decoder.py`只在自己的进程中适配已有decoder训练循环的目标/缓存/验证/metadata入口，不改公共trainer文件、LIBERO默认codec、已有实验配置或VLABench训练。它不运行在线训练encoder，训练目标使用全绝对codec，遇到意外在线编码直接报错。

已通过6项CPU测试：固定split/全frame-start窗口计数、全episode缓存索引边界、每任务首末窗口直接全绝对编解码、末帧重复及验证padding过滤、GPU阶段默认只打印、历史配置schema与保存策略。记录在 `evaluate_results/libero/goal_absolute_decoder_preparation_padding_20260919/cpu_tests.log`。

CPU检查尚未证明缓存实际内容一致、8卡训练显存/速度或GPU续训正确；这些待缓存生成后验证。
