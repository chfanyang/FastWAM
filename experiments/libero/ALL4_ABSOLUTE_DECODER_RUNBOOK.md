# 实验 3：四-suite 全绝对 VAE decoder 微调

已完成代码与CPU准备，尚未生成缓存、执行GPU审计、训练或闭环评测。独立入口 `scripts/prepare_all4_absolute_decoder.py`；配置 `configs/vae/libero_all4_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2.json`。不会修改公共trainer或Goal/VLABench入口。

## 固定设置

- 原始Wan2.1 VAE，冻结encoder/conv1，仅训练conv2+decoder；全绝对RAY0和未来16步，双tile 224×448，实验1的四-suite全绝对stats。
- 8卡batch2/GA4，有效batch64；2epoch，LR1e-5，warmup413步（5%），cosine降至1e-7；WD0，梯度裁剪1，FP32 master/BF16 autocast，分片AdamW。
- 中心/方向/夹爪masked L1权重1:1:1。遍历所有frame starts；尾部使用最后一个动作edge-repeat，补齐帧参与重建loss。验证动作误差排除补齐步。
- 每任务留出2个episode，split_seed=20260801；train seed42。验证每episode均匀取10个starts，包含首尾，共800个窗口。

| Suite | 训练窗口（含padding） |
|---|---:|
| spatial | 50,824 |
| object | 64,331 |
| goal | 50,445 |
| 10 | 98,809 |
| 合计 | 264,409 |

共1,632个训练episode、80个验证episode；4,132步/epoch，2epoch共8,264步。分布式batch补齐每epoch重复39个窗口，与episode尾部补帧不同。

固定清单与输入身份：`evaluate_results/libero/all4_absolute_decoder_preparation_padding_20260919/manifest.json`。同号episode在不同suite中使用数据集路径区分。验证清单和训练缓存索引均固定保存。

## 缓存与校验

复用实验1待生成的 `data/libero_all4_rothko_all_absolute_2cam224_wan21_bf16_h16_latents`，完整缓存应有277,713个starts。缓存顺序固定为spatial、object、goal、10，每个suite内按episode ID排列。

检查数据契约、原始VAE哈希、stats指纹、BF16精度、latent形状 `[16,5,28,56]`、完整样本数及原缓存验证标记。不能复用相对或混合表示缓存。

GPU审计在每个suite/task的训练、验证池各取一个episode，检查首个start、最后一个完整16步start、最后一个padding start，共240个probe；要求在线原始BF16 encoder输出与缓存逐位相等。审计身份与缓存metadata哈希写入 `cache_audit.json`，训练必须通过该检查。

## 操作入口

在仓库根目录、fastwam环境执行：

```bash
export PYTHONPATH=$PWD/src:$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
PY=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python

# CPU准备；重复运行仅接受完全相同的固定清单。
"$PY" scripts/prepare_all4_absolute_decoder.py prepare

# 实验1完整缓存生成后执行。
"$PY" scripts/prepare_all4_absolute_decoder.py check-cache

# 默认只打印GPU命令，不执行。
"$PY" scripts/prepare_all4_absolute_decoder.py audit
"$PY" scripts/prepare_all4_absolute_decoder.py train
```

GPU可用并完成前置检查后，单卡审计命令加 `--execute`。正式训练另行启动，在独立tmux中执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/prepare_all4_absolute_decoder.py train --execute
```

这些GPU命令本轮没有运行。输出目录非空会拒绝新训；恢复需显式追加 `--resume /absolute/path/checkpoint_latest.pt`，要求优化器状态、数据/配置身份、缓存metadata一致。独立历史权重checkpoint不可作为完整续训点。

## 保存与验证

- 每400步更新 `checkpoint_latest.pt`，包含优化器；epoch结束4132和最终8264也保存latest。
- 每800步保留独立权重checkpoint并导出完整VAE；最终8264也保存、导出。历史权重checkpoint按旧规则不含优化器。
- 验证：800的倍数直到8000，以及epoch结束4132、最终8264。使用同一原始encoder缓存、BF16部署decoder、全绝对legacy/anchor0解码；报告前8/16步、全体及每suite的平移/旋转/夹爪误差。
- Run：`runs/libero_all4_all_absolute_decoder_wan21_bs2_ga4_lr1e-5_ep2/`。
- 最终VAE：run内 `Wan2.1_VAE_libero_rothko_step008264.safetensors`。文件名沿用公共trainer规则，全绝对身份写入metadata。
- 后续复用实验1最终DiT，只更换最终VAE，沿用相同闭环场景、seed、replan8、去噪20次、每任务50场景；闭环本轮未启动。

CPU测试记录：`evaluate_results/libero/all4_absolute_decoder_preparation_padding_20260919/cpu_tests.log`。缓存实际内容、GPU显存/速度、训练及恢复仍需后续验证。
