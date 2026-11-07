# 真机：绝对 RAY0 + 相对未来动作

本文对应改造版模型。另有独立的[原版 FastWAM 5B 真机对照](BASELINE_FASTWAM.md)，使用数值动作 ActionDiT，不使用 Rothko 图。

2026-09-24：已新增[全量训练窗口 q99.95 norm 与仅重算 Rothko 的缓存流程](QUANTILE_CACHE.md)。下文 v2 路径和 max+5% 统计保留为旧方案，新方案使用独立 q9995 目录及配置。

这套接入只新增 `realworld` 文件，不修改已有 LIBERO / VLABench / RoboTwin 实验。
2026-09-23 已按用户授权生成文本缓存并启动 Wan2.1 八卡 latent 缓存流程；未启动训练或真机运动。
本次日志目录：`evaluate_results/realworld/wan21_cache_8gpu_20260923_202350/`，后台会话：`realworld_wan21_cache_8gpu_20260923_202350`。
只有 `latents_wan21_bf16/_SUCCESS` 出现才表示全部缓存与在线一致性检查完成；后台流程结束后不会自动训练。

## 固定实验约定

- 数据：`data/realworld_data` 下的四个任务，共 379 条轨迹。
- 每任务固定随机抽取 **2 条完整轨迹验证**（seed=42），共 8 条；371 条训练。
- 相机顺序：**左侧 `left_camera` → 腕部 `wrist_camera`**，RGB 画布为 `224×448`。
- 图像处理完全对应 `third_party/FastAPI/resize_stack_blocks.py`：PIL RGB、LANCZOS 等比缩放、居中黑边。在线 RealSense BGR 转 RGB 一次。
- 原始位姿：机器人模型基坐标系中的 `fr3_link8`，米，`xyz + xyzw`。
- 模型位姿：`xyz + wxyz`。RAY0 位置和朝向为绝对值；未来位姿以当前帧末端坐标系表达，解码后返回基坐标系绝对位姿。
- 每个窗口 17 帧：条件 `t`，未来目标 `t+1..t+16`，数据频率 10 Hz。不跨 episode，不用末端 padding。
- 原数据未保存控制指令，监督目标为未来实测位姿；夹爪标签为未来实测宽度 `>=0.08m` 时张开，否则闭合，`0=闭合/1=张开`。
  完全张开宽度为 0.085m。旧客户端的 0.06m 阈值会把 `pick_place/episode_00000` 和 `episode_00023` 全标成张开；这两条数据在夹住较宽物体时最小宽度分别约为 0.0607m 和 0.0652m，抽查图片也显示正在夹持物体。因此真机实验独立使用 0.08m，并由服务端 metadata 同步客户端。它仍是宽度推导标签，不是真实命令。
- 模型：原始预训练 Wan2.1 T2V 1.3B + 冻结原始 Wan2.1 VAE，full finetune，**8 epoch**，LR `1e-4`。不加载旧 LIBERO 策略或 decoder checkpoint。
- 默认每 GPU batch=4、梯度累积=4，开启梯度检查点；8 GPU 时有效 batch=128，8 epoch 共 6,040 次参数更新。GPU 数量在启动时指定。

## 文件与产物

| 用途 | 位置 |
|---|---|
| 数据与推理共用约定 | `src/fastwam/realworld/common.py` |
| 原生 PNG/NPZ Dataset | `src/fastwam/realworld/dataset.py` |
| 真机专用模型工厂 | `src/fastwam/realworld/runtime.py` |
| 推理适配与协议 | `src/fastwam/realworld/{policy,protocol}.py` |
| YARP 时序与动作检查 | `src/fastwam/realworld/execution.py` |
| 8 epoch 任务配置 | `configs/task/realworld_all4_ray0_absolute_wan21_ep8.yaml` |
| CPU 审计与归一化 | `scripts/realworld/prepare_data.py` |
| GPU 文本缓存 | `scripts/realworld/prepare_text.py` |
| 服务端 / 客户端 | `scripts/realworld/{serve,client_yarp}.py` |
| 独立测试 | `tests/realworld/test_realworld.py` |

所有派生产物在 `data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_v2/`，不写入源数据目录。

- `split.json`：逐 episode 划分、有效窗口、指令、状态 hash、相机文件清单指纹。
- `rothko_stats.pt/json`：训练集归一化和完整 codec 配置。
- `audit.json`：数量、时间间隔、夹爪宽度分布和范围审计。
- `validation_windows.json`：每任务一个固定验证窗口。
- 后续 GPU 阶段生成 `text/` 和 `latents_wan21_bf16/`。

归一化仅使用训练集：RAY0 位置取 train min/max 加 1 cm 余量；未来相对位置取所有有效训练窗口最大绝对值加 5% 余量。朝向范围为 `[-1,1]`。
底层直接复用未修改的 `LiberoRothkoCodec`，因此其内部表示名字仍为 `libero_rothko`；数据源、统计、缓存指纹均独立。模型工厂从真机统计文件注入 absolute RAY0 边界。

审计发现 `insert_pen/episode_00031` 的 70 帧夹爪始终张开，末端各轴运动范围约 0.01 mm，是一条近乎静止的轨迹。已按用户要求从源数据及指令索引中删除。v2 重新计算训练统计，保留 v1 原有的 8 条验证轨迹；现有 96,602 个训练窗口和 2,410 个验证窗口。旧 v1 产物仅供历史追溯，不再用于训练。

## CPU 准备与检查

在 FastWAM 根目录执行：

```bash
bash scripts/realworld/workflow.sh prepare
bash scripts/realworld/workflow.sh verify
```

这两个命令强制 `CUDA_VISIBLE_DEVICES=''`。准备过程是幂等的；若数据或划分改变，不会覆盖不同内容的旧统计，请改用新的产物目录和配置。
图片身份通过文件名、大小和修改时间审计；更改原始图片后必须重新准备。状态文件通过 SHA256 校验。
`text/cache/train` 启动前会先执行独立的 CPU preflight，逐轨迹复核源文件，并按阶段校验文本缓存和 latent 缓存。源数据发生变化时会拒绝继续使用旧缓存。

## GPU 空闲后手动运行

以下命令需要显式运行，不会被 CPU 准备命令自动触发。文本/缓存阶段已启动，训练阶段尚未启动。

```bash
# 只在指定 GPU 上预计算真实文本 embedding。
CUDA_VISIBLE_DEVICES=0 bash scripts/realworld/workflow.sh text

# 示意使用一张卡；GPU 列表与 NPROC_PER_NODE 按实际空闲资源指定。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/realworld/workflow.sh cache

# 已完成 text/cache 后，启动 8 epoch 策略训练。
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/realworld/workflow.sh train
```

8 卡示例为 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8`；显存需求尚未做 GPU 实测。缓存预计算复用仓库已有工具，但传入独立的真机 task 和输出路径。
训练输出在 `runs/realworld_all4_ray0_absolute_wan21_ep8/`。恢复训练可在 `train` 后传入仓库既有的 Hydra resume 参数。

## 训练后部署

服务端需要现有 FastWAM 环境，另需 `scripts/realworld/requirements-server.in` 中的 WebSocket/MessagePack 依赖（`pip install -r scripts/realworld/requirements-server.in`）。
当前 FastWAM 环境缺少 `websockets`，本次未安装或修改现有环境；部署前在相应环境中安装。

从 FastWAM 根目录启动（替换实际 run 路径）：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
python scripts/realworld/serve.py \
  --config runs/<run>/config.yaml \
  --checkpoint runs/<run>/<checkpoint>.pt \
  --device cuda:0 --port 8009
```

服务端默认仅监听 `127.0.0.1`，可以使用已有 SSH 隧道转发。启动时发送 metadata，此后每次接收 RGB 图像、当前绝对位姿、夹爪状态和任务指令，返回 16 步绝对位姿及夹爪状态。只接受已准备文本缓存中的指令。

真机端复用 `launch_fr3.sh` 和 `launch_robotiq2f.sh` 启动底层服务，但不同时运行键盘遥操作控制器。把 `scripts/realworld` 和 `src/fastwam/realworld` 放到真机可访问的 FastWAM 目录，设置 `PYTHONPATH`；使用已有 YARP 环境和原有 FR3 模型配置。

```bash
python scripts/realworld/client_yarp.py \
  --yarp-root /path/to/yarp \
  --server ws://127.0.0.1:8009 \
  --prompt 'Pick up the pen and insert it into the holder.' \
  --dry-run --max-cycles 1
```

`--dry-run` 读取真实相机和状态，但不启动机械臂控制策略、不发机械臂或夹爪动作。去掉该参数后才进入实际执行。
正常执行以 100 Hz YARP 笛卡尔阻抗控制器运行，默认每次执行前 4 个目标再重新观测；不会自动移动到初始化关节姿态，也不会连接时自动打开夹爪。

客户端检查响应 ID、动作坐标系、工作空间和单步跳变；超过 `--max-latency`（默认 5 秒）的响应不执行。速度默认 0.05 m/s、0.2 rad/s，必要时延长单步时间。
每段只加一次 30 ms 提前量，段内未触发速度限制的目标保持 100 ms 间隔。夹爪在对应 waypoint 的计划时间发送状态切换，避免把未来夹爪命令立即发出；计划时间不等于实测已经到位。
Ctrl-C 结束等待并退出控制器，YARP 的 `stop()` 会清空尚未执行的机械臂命令队列。

## 已知验证边界

CPU 测试覆盖图像与原 resize 脚本一致性、相机顺序、四元数/夹爪约定、真实数据窗口与 codec 往返、Hydra/训练器数据契约、协议与 OpenPI 编解码兼容性、模拟推理适配以及模拟 YARP 调度。
本地 WebSocket 集成测试使用真实服务端处理函数和模拟策略，覆盖 metadata 握手、NumPy 数组传输、错误返回及重新连接。调度测试使用 YARP 原始插值器，检查正常速度和触发速度限制时目标位姿与夹爪计划时间的一致性。
多进程加载另外使用真实 `spawn` DataLoader 验证，避免只检查主进程读取。临时 BF16 缓存测试覆盖文本缓存完整性、latent 分片读取、缓存模式跳过 PNG、位姿对齐，以及已打开 mmap 的数据集序列化恢复。服务端加载 checkpoint 后会校验训练时数据统计身份，拒绝混用不同统计和划分。
当前已有 `fastwam_robotwin` 环境包含集成测试所需的 `websockets==16.1` 和 `scipy==1.15.3`，以下命令已通过全部 15 项 CPU 测试；缺少这两个依赖时，对应集成测试会跳过：

```bash
REALWORLD_PYTHON=/mnt/hwdata/cfy/miniconda3/envs/fastwam_robotwin/bin/python \
  bash scripts/realworld/workflow.sh verify
```

工作流将临时目录设为本机 `/tmp`，避免 NFS 临时 socket 的清理问题；可通过 `REALWORLD_TMPDIR` 指定其他本地目录。
尚未执行 GPU VAE 重建评估、模型训练/推理、加载真实策略的服务端或真机闭环；训练完成前没有可部署的真机策略 checkpoint。
采集时的相机曝光时间戳未落盘，同编号数据只能视为近似同步；实测位姿与真实遥操作目标之间的偏差无法从旧数据还原。
