# 真机 q99.95 norm 与 Rothko 缓存重算

2026-09-24 用户授权：在四个任务全部训练数据上重新计算 norm，八卡每卡 4 个数据 worker 重算缓存。训练仍暂停。

- 统计：371 条训练轨迹，96,602 个窗口，1,545,632 个未来位移向量。
- 各轴取绝对位移的 99.95% 分位数，采用与 LIBERO 相同的 100,000-bin、[0,1 m] 直方图上边界算法；使用全部有效窗口，不采用每轨迹 32 窗口采样。
- 相对位置边界：`±[0.09253, 0.09474, 0.08849] m`。
- RAY0 绝对位置 min/max（含原来的 1 cm 余量）、相对参考系、方向编码、夹爪、图片预处理、轨迹划分与窗口顺序保持不变。
- 原 max+5% 统计和缓存保留。新目录：`data/realworld_artifacts/ray0_absolute_left_wrist_val2_g08_q9995_v1/`。
- 文本通过只读使用原文件的目录链接复用。RGB latent 从已校验旧缓存逐位复制到新缓存；全量循环不读 PNG、不重新编码 RGB。仅重新编码 Rothko。
- 最终新缓存仍是标准 RGB/Rothko BF16 双模态分片，可直接由现有训练读取器加载。
- 完成前不写 `_SUCCESS`；完成时沿用首/中/尾在线编码逐位一致及固定随机种子 training loss 一致性检查。少量最终校验会读取 PNG 并编码 RGB。

独立文件：

- `scripts/realworld/prepare_quantile_norm.py`：CPU 统计，保留源 split 字节身份。
- `src/fastwam/realworld/ray_cache.py`：验证源与目标数据契约，只允许 norm 指纹不同；按相同全局窗口索引提供旧 RGB 和新 Rothko。
- `scripts/realworld/recompute_ray_latents.py`：从仓库原缓存脚本独立派生，新增 RGB 复用，未修改原脚本。
- `configs/task/realworld_all4_ray0_absolute_wan21_q9995_ep8.yaml`：指向新 norm 的独立配置。旧任务仍指向原 norm；以后训练新 norm 应使用此配置或其单任务派生配置。

已做 CPU 检查：Hydra 解析、新旧窗口一致、首/中/尾 RGB 位级复用、无 PNG 读取、RAY0 不变、未来 raymap 改变、DataLoader 数据对象序列化。

本次启动参数：GPU0–7，8 个进程，每卡 batch=2 / workers=4，每分片 1,024 个窗口，共 95 片。
后台会话：`realworld_wan21_q9995_cache_20260924_010007`。
日志：`evaluate_results/realworld/wan21_q9995_ray_cache_8gpu_20260924_010007/cache.log`。
入口：同目录 `launch.sh`，只计算缓存，结束不会自动启动训练。
