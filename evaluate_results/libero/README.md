# LIBERO 实验结果导航

2026-11-07 整理。首先查看 [INDEX.csv](INDEX.csv)，再从 `path` 列进入规范目录。

| 目录 | 内容 |
|---|---|
| `closed_loop/` | 闭环评测；历史模型仍保留原有模型/评测两层结构，包含其既有分析子目录 |
| `offline_audits/` | 表征 round-trip、VAE 重建、离线动作预测误差分析 |
| `preparation/` | norm_stats、数据/缓存检查、实现校验、smoke/e2e、实验准备 |
| `launch_records/` | 训练、缓存、评测启动记录，以及旧顶层日志和 latest 指针 |
| `archive/failed_or_withdrawn/` | 启动失败、旧启动版本、已撤回方案；保留供追溯 |
| `archive/temporary/` | 原顶层临时目录；未删除内容，不代表正式结果 |

## 索引口径

- 一行可以是实验族、单次评测或辅助材料；`kind`、`status` 区分其性质。实验族行不重复汇总子实验。
- checkpoint、VAE、decoder、anchor、replan 和 denoising steps 来自保存的 `manager_config.yaml`；缺少字段时保留空值或 `unknown`，不从目录名猜测。
- `successes`、`completed_episodes`、`sr_percent` 来自每任务 `gpu*_task*_results.json`，按 `(task_suite, task_id)` 去重。SR 是已记录场次的加权成功率，不是 suite 等权平均；partial 的 SR 不代表完整评测成绩。
- `expected_tasks` 来自 `tasks.txt`，`expected_episodes` 为清单任务数乘配置中的 `num_trials`。`complete` 表示任务清单覆盖完整且每任务场数匹配；缺少验证依据则标记 `results_present_completion_unverified`。
- 有冲突的重复结果或清单之外的任务标记 `needs_review`，不输出总体 SR。没有直接每任务结果的目录标记 `no_task_results`，不等于实验失败，可能需要查看内部子目录。
- `per_suite` 提供每个 suite 的成功数、场数和 SR。此索引只覆盖本机普通 LIBERO，不包含 LIBERO-Plus 或尚未同步的远端结果。
- 索引是此次整理时的静态快照；后续增加或更新评测结果时需要重新汇总。

## 路径与保留策略

旧顶层的128个条目均已移动到分类目录。按用户要求，随后移除了本次整理创建的128个兼容软链接；实际数据全部保留。旧路径不再可用，请使用 INDEX.csv 的 path 列。legacy_path 列仅供历史映射查询；旧启动脚本或日志中记录的旧路径需要按 moves.json 映射到新位置后使用。

没有删除原始 JSON、视频、日志或临时文件，也没有合并不同 checkpoint、VAE、decoder、seed 或重跑版本。没有修改训练或评测代码。

移动清单和验证记录位于 [_organization_20261107/](./_organization_20261107/)。`moves.json` 记录每条旧/新路径；`completed.jsonl` 记录已执行移动；`index_before.json` 保留整理前索引；`verification.json` 核查旧路径 inode 一致及评测统计不变。

如需撤销，应先确保没有进程正在写入结果且旧路径没有被新文件占用，再依照清单将对应规范路径重命名回旧路径。不要删除规范路径中的实际数据。整理工具 `scripts/organize_libero_results.py` 默认只预览，并拒绝重复执行已有整理。
