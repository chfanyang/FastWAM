# RoboTwin episode 级续测

默认 `EVALUATION.resume=false`、`resume_tracking=false`，旧命令行为保持不变。
不改 LIBERO / LIBERO-Plus，不修改模型预测、seed 公式或动作执行。

## 新实验

新目录启动时加 `EVALUATION.resume_tracking=true`。
每完成一个 episode 原子更新 `<task>/progress_clean.json` 或 `progress_random.json`。
停止后使用同一实验目录、同一配置，加 `EVALUATION.resume=true`。
GPU 列表及每卡并发数可以调整。

恢复记录包含累计成功/已测次数、下一个环境 seed、关键策略设置以及
解析后的 model/processor 配置摘要。模型采样 seed 不变。
不允许把 replan、VAE、步数上限等不同条件混合统计。
任务已完成时单任务入口直接重建结果文件并跳过模型加载和 rollout；
manager 仍会启动该入口确认结果，再纳入最终汇总。

中断的 episode 不恢复半条轨迹，重新执行。该编号及之后的未提交
视频和预测目录会移到任务目录的 `interrupted/<phase>_<timestamp>/`，
已完成视频不会覆盖。预测视频编号与环境 episode 编号继续一致。
不保证仿真/规划器数值逐 bit 一致；这是 episode 边界恢复，不是进程快照。

## 旧实验导入

`scripts/import_robotwin_eval_progress.py` 从旧任务日志末次完整的
`Success rate ... current seed ...` 行恢复；逐个核对已经完成的视频及其
success 后缀。需要明确提供原始策略 options JSON。遇到多份日志、
冲突视频或日志/视频计数不符时不会猜测或覆盖已有进度文件。

## 2026-09-08 clean 停止记录

实验目录：
`evaluate_results/robotwin/robotwin_c2r_7task_rothko_centerfrac05_full_wan22_5b_1e-4_2026-09-08_06-20-00/ckpt002730_vaeOriginal_replan8_ensembleOff_clean100_20260908`

| 任务 | 成功/已测 | 下一个环境 seed |
|---|---:|---:|
| click_bell | 95/100 | 4300100 |
| click_alarmclock | 9/20 | 4300021 |
| grab_roller | 48/49 | 4300052 |
| turn_switch | 12/24 | 4300028 |
| lift_pot | 27/28 | 4300070 |
| beat_block_hammer | 20/25 | 4300028 |
| move_playingcard_away | 25/40 | 4300040 |

共236/286；目标每任务100次。步数上限仍为400，不是220。
原始 `launch.sh` 未改，恢复请用目录下 `resume.sh`，不要重跑 `launch.sh`。
`resume.sh` 默认 GPU1–7 每卡一个 worker，日志追加到 `console_resume.log`。
示例：

```bash
tmux new-session -d -s robotwin_c2r_7task_clean_resume \
  'bash /mnt/hwdata/cfy/FastWAM/evaluate_results/robotwin/robotwin_c2r_7task_rothko_centerfrac05_full_wan22_5b_1e-4_2026-09-08_06-20-00/ckpt002730_vaeOriginal_replan8_ensembleOff_clean100_20260908/resume.sh'
```

验证：3项 CPU 测试通过，覆盖实际 eval_policy 循环的恢复计数/seed/
模型 episode 编号、配置不符拒绝、日志导入与未提交视频归档；实际 Hydra
单任务入口的 subprocess 参数拦截测试通过。尚未做真实仿真的中断恢复对照。
