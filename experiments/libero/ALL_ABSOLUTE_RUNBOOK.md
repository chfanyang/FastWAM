# LIBERO 四-suite 全绝对：本机操作记录

目前只准备代码，不启动 GPU 任务。仓库根目录为 `/mnt/hwdata/cfy/FastWAM`；缓存与训练使用本机 `fastwam` 环境，闭环评测使用 `fastwam_libero`。不涉及迁移打包或环境重装。

## 已锁定的实验

- Wan2.1 T2V 1.3B，全量 DiT；原始 VAE，冻结 encoder/decoder。
- RAY0 与未来 16 步均为绝对位姿，双视角 224×448；四-suite 共 277,713 个窗口。
- 8 卡、每卡 batch4、GA4，有效 batch128；10 epoch，预计 21,700 步。
- LR 1e-4，warmup 5%（预计 1,085 步），cosine 到 1e-6；BF16、gradient checkpointing 关闭。
- 每 1000 步验证，每 3000 步保存权重和完整 state，结束保存；不额外开启验证前保存。
- 最终 step21700 做一次正式闭环评测；四 suite，每任务 50 次，seed42、replan8、去噪20步、legacy、anchor0、不使用 action ensemble。
- 旧相对方案的评测用了微调 VAE7498；本实验使用原始 VAE，报告时说明这一差异。

## 入口及默认行为

统一入口：`scripts/libero_all_absolute_workflow.py`。

`check` / `check-cache` 是 CPU 检查；其他阶段默认只打印命令，**只有加 `--execute` 才会运行**。真正执行前检查指定 GPU 是否空闲，遇到占用直接退出，不停止其他任务。缓存、训练、评测是独立阶段，不会自动串联。

```bash
cd /mnt/hwdata/cfy/FastWAM
PY=/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python

# 1. 检查配置、数据 metadata、全绝对 stats、原始 VAE 身份。
"$PY" scripts/libero_all_absolute_workflow.py check

# 2. 查看原始 VAE 小审计命令；GPU 空闲后再加 --execute。
"$PY" scripts/libero_all_absolute_workflow.py audit --gpus 0

# 3. 查看 8 卡小缓存测速命令；确认后再加 --execute。
"$PY" scripts/libero_all_absolute_workflow.py benchmark

# 4. 查看 8 卡完整缓存命令；确认后再加 --execute。
"$PY" scripts/libero_all_absolute_workflow.py cache

# 5. 完整缓存生成后，检查覆盖、shard 字节数、身份及一致性验证记录。
"$PY" scripts/libero_all_absolute_workflow.py check-cache

# 6. 查看正式训练命令；会生成独立时间戳 run 目录。
"$PY" scripts/libero_all_absolute_workflow.py train

# 7. 将下面的路径替换为这次实际完成的 run，再查看评测命令。
"$PY" scripts/libero_all_absolute_workflow.py eval --run /absolute/path/to/completed_run
```

默认 GPU 为 0–7，可通过 `--gpus` 改为另外八个空闲 GPU；audit 只使用列表中的第一张卡。`--output` 可指定新的审计、测速、训练或评测目录。所有已存在的正式输出目录均拒绝覆盖；完整缓存目录固定为下面的独立路径。不会隐式恢复中断缓存或训练。

正式训练建议在独立 tmux 会话中运行，例如 GPU 可用且全部前置校验完成后：

```bash
tmux new-session -s libero_all4_absolute_train
cd /mnt/hwdata/cfy/FastWAM
/mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python scripts/libero_all_absolute_workflow.py train --execute
# Ctrl-b d 离开会话，训练继续。
```

## 缓存与诊断

- 使用原有 `precompute_visual_action_latents.py`，分别编码 RGB/Rothko，不改训练数据生成逻辑。
- 完整缓存目录：`data/libero_all4_rothko_all_absolute_2cam224_wan21_bf16_h16_latents`。
- 默认 8 个进程，每 GPU 1 个进程；VAE batch8、每进程 DataLoader workers4。这里的 batch/worker 与训练 batch/GA 不同；可通过 `--batch-size`、`--num-workers` 先测速再定。
- benchmark 只取开头 256 个窗口，分成 32 窗口的小 shard，并写入独立目录；不代表四-suite 质量审计，不能作为完整训练缓存。
- 完整缓存形状 `[16,5,28,56]`，RGB/action 两种模态，BF16；原始 shard 总量约 130 GiB，另留临时空间与 checkpoint 空间。
- 原缓存程序检查首／中／末窗口的逐位一致性及固定噪声 training loss；这些是抽样校验，不是逐窗口全量重编码验证。
- `check-cache` 拒绝旧表示、不同 stats/VAE、不同精度／尺寸、覆盖不全及 benchmark 缓存。它检查 shard 大小，不宣称检查了所有 shard 内容哈希。
- `audit_libero_all_absolute.py` 默认四-suite 各取首／中／末窗口，比较 GT 直接解码和原始 VAE 重建后动作；排除 padding，输出前8／16步位置 L2、四元数测地角误差和 gripper 误差。动作解码在 CPU 完成。
- 大旋转／极值覆盖需另行选样：`audit --indices-json /absolute/path/windows.json`，JSON 为 `{suite数据目录名: [suite内dataset索引, ...]}`，必须覆盖四个 suite。默认 12 窗口不代表已完成极值审计。
- 每次实际执行记录在 `evaluate_results/libero/all4_absolute/launch_*`：命令、输入身份、源码哈希、状态、`output.log`。日志可用 `tail -f` 查看。临时文件也放在该仓库的 `evaluate_results/libero` 下。

## 2026-09-19 GPU 前置检查

- 四-suite 共 74 个普通、位置极值、大旋转与尾部窗口完成原始 VAE 审计；结果：`evaluate_results/libero/preflight_all4_20260919/REPORT.md`。
- 两卡 256 窗口小缓存完成，抽查 latent 与实际 training_loss 逐位一致；正式全量缓存仍未启动。
- 新实验文本缓存改用 `data/text_embeds_cache/libero_wan21_bf16_batch1`，40 条文本已在实际评测环境逐位验证，旧缓存不覆盖。batch=1 消除本次发现的 BF16 batch 形状差异。
- 同名目录和新脚本已同步至远端 A800。换环境/硬件/编码器后重新 verify。

```bash
# 仅在新目录生成；使用获准的空闲 GPU，先设置 CUDA_VISIBLE_DEVICES。
PYTHONPATH=src /mnt/hwdata/cfy/miniconda3/envs/fastwam/bin/python scripts/prepare_libero_single_text_cache.py build \
  --cache-dir data/text_embeds_cache/libero_wan21_bf16_batch1 \
  --report evaluate_results/libero/text_batch1_20260919/build.json
PYTHONPATH=src /mnt/hwdata/cfy/miniconda3/envs/fastwam_libero/bin/python scripts/prepare_libero_single_text_cache.py verify \
  --cache-dir data/text_embeds_cache/libero_wan21_bf16_batch1 \
  --report evaluate_results/libero/text_batch1_20260919/verify.json
```

这些目录/报告当前已存在；重跑校验需指定新的 report 路径。脚本拒绝覆盖旧缓存或报告。正式 latent 缓存仍需完整性校验后才能训练。

## 本轮准备验证（2026-09-18）

- CPU 检查通过：四-suite 帧数分别为 53,229 / 67,309 / 52,895 / 104,280；合计 277,713。
- 全绝对 stats 内容指纹：`999f9e9031ac55bb40a0b89c60dbd1389852e33e6f173d9a58ba48e0f1c216f4`。这是 stats 内容指纹，不是 `.pt` 文件字节哈希，两者分别记录。
- 原始 VAE 文件 SHA256：`38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981`。
- 8 项 codec／入口测试通过；实际 Hydra manager 和 worker 配置分别解析通过，均保持全绝对、原始 VAE、legacy、anchor0、replan8、50 trials、20 denoising steps。
- 本地检查记录：`evaluate_results/libero/all4_absolute_local_prepare_20260918/`。
