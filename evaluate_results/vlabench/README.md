# VLABench 实验结果导航

2026-11-07 整理。首先查看 [INDEX.csv](INDEX.csv)，按 `path` 列进入新目录；不保留旧顶层路径软链接。

| 目录 | 内容 |
|---|---|
| `closed_loop/` | select_book 各表示及 all10 各 checkpoint/VAE 的闭环评测；续跑和重试分开保留 |
| `offline_audits/` | VAE/动作重建审计、heldout382 离线预测、专家回放、相机布局预览、分位数分析 |
| `preparation/` | norm bounds、数据/缓存准备、训练预检查、显存诊断及实现检查 |
| `launch_records/` | 启动、续训、缓存记录，旧顶层指针文件及单独 JSON |
| `archive/temporary/` | 原运行时临时目录，内容保留 |
| `archive/repository_records/` | 原 git 分批整理记录 |

## 索引口径

- 模型族单独一行，内部各评测另有索引行；辅助材料不计算闭环 SR。
- checkpoint、VAE、decoder、replan 和 seed 来自 `plan.json` 的 options 或 `manager_config.json`。旧配置未记录的字段保留 `unknown`，不从名称推断原始 VAE 或默认 decoder。
- 每次评测优先读取其 `summary.json`，没有时读取 `status.json`，从其中 episodes/records 获取逐场结果，再核对本目录已存在的 episode JSON。按照 `(task, episode)` 去重，不重复累计同一场。
- 继承的逐场结果计入该轮完整统计；`inherited_records` 与 `completed_from` 记录来源。不同轮次之间存在重叠，因此禁止直接对 INDEX.csv 的各行求和。
- `sr_percent` 是本轮可读取场次的成功比例；`expected` 是原记录/配置的目标场数。`partial` 不能当作完整成绩。`complete` 仅表示记录数量与目标相等，没有通过实际重跑验证。
- `needs_review` 表示原 summary 与可读取记录不一致，或同一场景存在冲突；该行不输出 SR，需要先复核。不静默覆盖或改写历史结果。
- `is_raw`、`ps_raw` 是已记录数值的原始均值（0到1），没有施加“不低于 SR”的显示修正。缺少分数的场次不补零；`metric_records` 是同时具有 IS 和 PS 数值的场次数。
- 此索引是本机结果的静态盘点，不保证包含远端尚未同步的结果。

## 移动和历史路径

原有94个顶层条目已经分类；另把 `vlabench_select_book_rothko_3cam192_wan21/heldout382_4ckpt_vae562` 从模型族提取到 `offline_audits/` 下的同名模型目录。

没有删除 JSON、视频、日志或临时数据，没有修改训练/评测代码，也没有改写原始记录里的绝对路径。旧路径不再有效；旧启动脚本、completed_from 和 inherited output 等历史引用需先按 [_organization_20261107/moves.json](_organization_20261107/moves.json) 映射到新路径后使用。匹配时优先最长旧路径前缀，避免把单独迁出的 heldout 审计误映射到 closed_loop。

整理记录在 [_organization_20261107/](_organization_20261107/)，包含移动清单、逐条完成日志、整理前索引和 inode/统计一致性验证。需要撤销时，先确认没有写入进程及目标冲突，按移动清单的反序将实际目录重命名回旧路径。

整理工具：`scripts/organize_vlabench_results.py`；默认预检查，`--apply` 才移动，并拒绝重复整理已有记录。
