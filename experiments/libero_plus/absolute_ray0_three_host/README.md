# 四-suite absolute RAY0 / relative-action：三机 LIBERO-Plus 评测

仅准备，由用户通知后启动；不监测训练结束、不自动传输或启动、不停止任何现有作业。

| role | 主机 | 场景 | 总数 | 8卡并发 |
|---|---|---|---:|---:|
| 3090_object | 172.16.12.17 | Object全部2518 + Spatial偶数task_id 1201 | 3719 | 每卡1worker |
| 3090_goal | 172.16.12.18 | Goal全部2591 + Spatial奇数task_id 1201 | 3792 | 每卡1worker |
| a800_libero10 | 172.16.12.6 | LIBERO-10全部 | 2519 | 每卡4worker |

这里是 LIBERO-Plus 全部10030个扰动场景，每场景1次；不是原版LIBERO的每任务50次。Spatial按完整Plus task_id奇偶拆分，两个集合无重复、合并无遗漏。固定JSON保存语言、BDDL及init-state文件名；启动时与实际安装的benchmark核对。

协议：Wan2.1 1.3B、RAY0世界坐标绝对位姿、未来16步相对当前EE、replan8、20步去噪、seed42、legacy decoder/anchor0、不ensemble。当前EE仍用于相对位姿解码；不是OSC增量或全绝对实验。使用当前四-suite共享Rothko stats与原训练配置的绝对RAY0 bounds。批量关闭视频。每个worker只加载一次模型，依次执行分配到的多个场景。线程限制为2。

## 等用户通知后准备最终快照

DiT来源：ainode06上的
`runs/libero_all4_rothko_absolute_ray0_2cam224_full_wan21_1_3b_1e-4/2026-09-22_06-15-04`，最终step21700。
VAE来源：本机
`runs/libero_all4_absolute_ray0_decoder_wan21_bs2_ga4_lr1e-5_ep2`，最终step8264。

先确认两次训练正常结束及实际最终step，核对checkpoint元数据和对应run配置，再传输。若最终step变化，必须明确更新plan/launcher/文件身份，不能偷偷替换成中间checkpoint。SHA256只能证明文件一致，不能证明训练完成。

在`video_a800`下创建独立新快照，保留repo相对布局：

- `src/`、`configs/`、`experiments/libero/`、`experiments/libero_plus/`；保留实际当前代码，不只打包已提交的文件。
- `third_party/LIBERO-plus/`：源代码、BDDL、init_states、assets全部；排除.git、__pycache__。不可用普通LIBERO替代。
- `data/libero_mujoco3.3.2/libero_rothko_region_symmetric_q99p95_h16_224x448.pt`及同名json。
- `artifacts/dit_step021700.pt`：上述DiT run的最终weights；`dit_training_config.yaml`：同一run的config.yaml。
- `artifacts/vae_step008264.safetensors`：上述VAE run的完整VAE导出（含冻结encoder）；`vae_training_config.json`：对应training_config.json。
- `artifacts/dataset_stats.json`：同一DiT run保存的dataset_stats.json，不能拿其他run的替代。

无需传训练latent缓存、原始训练数据或optimizer state。
运行`python experiments/libero_plus/absolute_ray0_three_host/seal_artifacts.py`生成SHA256清单；再把同一个快照用`rsync -aP`（不用--delete）传到三台机器各自repo下新的`eval_preparations/plus_absolute_ray0_final`目录。不可覆盖已有快照或历史结果。

每台机器单独配置：

1. 快照下`checkpoints`链接到该机器原repo的`checkpoints`，检查Wan2.1 tokenizer、text encoder及原始模型文件存在。
2. 建立快照`data/libero_plus/.libero_config/config.yaml`，其中benchmark_root/bddl_files/init_states/assets指向**本快照**`third_party/LIBERO-plus/libero/libero`对应目录，datasets指向本快照`data/libero_plus`。不要改账户全局LIBERO配置。
3. 用对应role的Python运行`seal_artifacts.py --verify`核对传输。该工具只核对，不启动。
4. `bash experiments/libero_plus/absolute_ray0_three_host/run.sh ROLE check`做CPU配置/清单检查，不加载模型，不占GPU。
5. 特别注意ainode06此前只有普通LIBERO环境：launcher优先导入快照Plus源码，但仍需要实测依赖、MuJoCo/EGL与一个短rollout；不能仅凭CPU清单检查就宣称闭环可运行。3090同样先核对可用卡并短测最终DiT+VAE组合。

## 显式启动

在对应机器的快照目录内，用户通知启动且上述检查通过后：

```bash
# ROLE依次为3090_object、3090_goal、a800_libero10
# 在tmux新会话里运行；不复用已有会话名。
bash experiments/libero_plus/absolute_ray0_three_host/run.sh ROLE run
```

launcher默认使用plan.json记录的环境路径，必要时用`PYTHON=/正确路径/python`覆盖。正式输出位于快照`evaluate_results/plus_dit21700_vae8264/ROLE`。启动先验证权重/代码SHA256与评测身份；同目录显式续跑只跳过完整且身份一致的场景结果，旧manager日志会备份，错误文件不当作完成。保持同一代码快照、配置、权重和任务清单。

当前只完成本机CPU清单/配置检查；最终模型传输、三机环境和最终组合短rollout留到后续执行。没有启动任何新评测。
