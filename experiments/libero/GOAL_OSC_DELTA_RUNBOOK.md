# Goal 原始 OSC 增量实验

用户在2026-09-20取消减桌高方案，要求撤回该轮代码，在本机A800跑原始OSC方案。
撤回前代码/patch归档于 `evaluate_results/libero/withdraw_table_absolute_20260920_152042`；基线`4d2c60e`。旧缓存、原始数据、已完成结果保留，不再使用减桌高缓存。

## 表示和执行

独立表示 `libero_rothko_osc_delta`。
- 原始 `observation.state` = 世界XYZ、绝对轴角、两夹爪关节(8维)，构造RAY0的世界位姿和当前开合程度。
- 原始 `action` = 归一化OSC平移、旋转向量、open夹爪(7维)。未来16帧直接编码该命令，不读取额外`osc_target_pose_wxyz`或`ee_pose_wxyz`字段。
- 未来位置图写入 `action[:3]*0.05`米；旋转图使用`axis_angle_to_rotation(action[3:6]*0.5)`；gripper保持数据0关1开。
- RAY0位置bounds来自全4suite原始state极值，每侧加1cm；所有未来位置bounds固定±0.05m。旋转方向固定±1；不同时间段norm由codec明确区分。
- 不减桌高、不减窗口起点、不做suite独立归一化、不累积预测增量。
- decode返回第0帧世界位姿，后16帧增量pose7。`future_endpose`/`pred['pose']`在这个独立模式下的未来部分表示增量，重建误差是命令误差，不是绝对轨迹误差。
- eval直接把增量旋转转回旋转向量、除以控制器scale，裁剪命令到[-1,1]；环境夹爪=1-2*open；不调用绝对目标到OSC转换。控制器自身零旋转逻辑不修改；不新增预测死区/取整。VAE/DiT输出近零不保证严格等于零。
- eval禁用absolute-pose action ensemble。旧所有表示仍走原分支。

## 文件和预算

Task `libero_goal_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4`
Eval `sim_libero_goal_osc_delta`
Stats `data/libero_mujoco3.3.2/libero_all4_raw_state_osc_delta_h16_224x448.pt`
Fingerprint `cf144be988b35a5c527a67a1d194c45ecba899778966bf10faf71e3b960274e6`
Cache `data/libero_goal_osc_delta_2cam224_wan21_bf16_h16_latents`
Run `runs/libero_goal_rothko_osc_delta_2cam224_wan21/时间戳`

保持原Goal预算：Wan2.1 T2V1.3B全DiT、原始冻结VAE、8GPU bs4/GA4有效128、10epoch(4140steps)、LR1e-4、warmup5%、RGB/action1:5、原baseline mask、验证1000、weights/state每2000+最终。尚未启动训练，启动前确认参数。

## 命令

新机器先生成stats（拒绝覆盖已有文件）：
```bash
PYTHONPATH=src:. python scripts/compute_libero_osc_delta_stats.py --output-dir evaluate_results/libero/NEW_STATS_DIR
```

小规模审计：
```bash
PYTHONPATH=src:. python scripts/audit_libero_goal_osc_delta.py --output evaluate_results/libero/NEW_AUDIT_DIR --indices-json MANIFEST
```

正式缓存使用短TMPDIR，避免AF_UNIX socket路径过长：
```bash
mkdir -p evaluate_results/libero/osctmp
export TMPDIR=$PWD/evaluate_results/libero/osctmp
export PYTHONPATH=$PWD/src:$PWD
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m torch.distributed.run --standalone --nproc_per_node=8 scripts/precompute_visual_action_latents.py \
 --task libero_goal_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4 \
 --output-dir data/libero_goal_osc_delta_2cam224_wan21_bf16_h16_latents \
 --batch-size 2 --num-workers 2 --samples-per-shard 1024
```

训练（完整缓存验证通过后）：
```bash
bash scripts/train_zero1.sh 8 task=libero_goal_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4
```
Eval（新checkpoint；不要用旧absolute checkpoint）：
```bash
python experiments/libero/eval_libero_single.py --config-name=sim_libero_goal_osc_delta \
 ckpt=/ABS/NEW_RUN/checkpoints/weights/FINAL.pt EVALUATION.task_id=0 EVALUATION.num_trials=50
```

RAY0 state范围(m)：X[-0.49278069496,0.22031379700]，Y[-0.34093360066,0.40042642713]，Z[-0.00187181890,1.48277808189]。该范围只拟合state，因此不同于旧state+绝对target统计。

## 本机预检与缓存启动

- 30项新旧表示回归通过；真实eval环境helper验证原始命令、零命令、夹爪方向及禁用绝对pose ensemble通过。
- 20窗口覆盖10任务及尾部，h8有效90步；直接codec平移最大0.0000058mm、旋转最大0.000003°。原始VAE增量平移均值1.5372mm/P95 3.7813mm，旋转增量均值1.2361°/P95 2.4781°。这是命令重建误差，不能同绝对目标误差直接比较。
- 16窗口smoke完成；RGB/action indices0/8/15在线bit-exact；training_loss固定seed123456为1.1140246391，缓存与在线相等；complete=true和_SUCCESS。退出阶段NFS临时文件清理告警不影响已完成的验证。
- 正式缓存：tmux `libero_goal_osc_cache`，8GPU、每rank batch2、2个数据worker；52,895窗口，52分片。启动路径见 `evaluate_results/libero/goal_osc_delta_cache_latest_launch.txt`。
- 正式训练未启动，启动前确认参数。


### OSC 两机分工（2026-09-20）

- 本机 ainode07：计算 Goal 缓存，之后只训练 Goal。
- 远端 ainode06（172.16.12.6）：计算 spatial/object/10 三个 suite 缓存，再接收本机已完成且验证通过的 Goal 缓存，合并后训练四个 suite。
- 两端原始 Wan2.1 VAE SHA256、19个源文件、stats 指纹、数据元信息及 PyTorch 版本已核对一致；沿用同一原始 OSC 表示。预检记录：`video_a800/osc_delta_sync_20260920/preflight.json`。
- 已停止远端旧 Goal 全绝对 VAE 训练；`checkpoint_latest.pt` 的实际 step=1200，含 optimizer/scheduler，未删除历史结果。
- 远端缓存 tmux：`libero_other3_osc_cache`；8卡，每rank batch2、2个数据worker，线程数2；224,818窗口。日志：`evaluate_results/libero/other3_osc_delta_cache_launch_20260920/cache.log`。
- 远端三-suite task：`libero_other3_rothko_osc_delta_2cam224_wan21_cache`；训练四-suite task：`libero_all4_rothko_osc_delta_2cam224_full_wan21_1_3b_1e-4`。
- Goal 缓存传输尚未完成。合并使用 `scripts/merge_libero_osc_caches.py`，按 spatial/object/goal/10 索引顺序逐字节重排分片，不重新编码；合并后必须运行标准缓存入口完成在线 latent/training_loss 验证才能用于训练。不能直接拼 metadata 或重命名分片。
- 此次只启动缓存；两台正式 DiT 训练均未启动，启动前确认训练参数。


### 2026-09-20 OSC 流程复核：补零语义已确认

用户明确：未来预测的是逐步 OSC 增量，尾部 padding 的平移／旋转应补零，夹爪保持最后值；RGB/state 仍重复末帧。当前 processor 符合该语义。此前把它误判成必须 edge-repeat 动作并要求重算的结论撤回；不应用 delta_action_dim_mask=null，不因此停止／重算缓存。raw/action tensor 共享确实存在，后续修改 processor 时需要留意，但当前结果符合要求。

本机 Goal 完整缓存已通过：indices 0/26447/52894 在线 latent 逐位一致，固定 seed123456 的 training_loss 逐位一致（1.1243473291397095），complete=true、_SUCCESS。远端缓存及四-suite合并后的实际校验、八卡训练／保存恢复仍待完成。训练内验证使用训练集前4窗口，四-suite中仅覆盖spatial，不是独立held-out验证。详见 `evaluate_results/libero/osc_pipeline_review_20260920/REPORT.md`。


### 2026-09-20：扩大新 OSC 训练内验证覆盖

- Goal：全部10任务，每任务3个不同episode，各取开头／中段／尾部一个窗口，共30窗口。
- 四-suite：全部40任务，每任务同样3窗口，共120窗口；suite顺序spatial/object/goal/10。
- 尾部窗口保留8个有效动作＋8个padding动作；OSC padding平移旋转补零、夹爪保持。其余两类窗口16步有效。
- 每1000 optimizer steps验证，20步去噪；固定窗口与seed(42+清单序号)，8卡轮流分摊（Goal每卡3–4窗口，四-suite每卡15窗口），每个窗口均计算loss和推理指标。总耗时会增加，实际耗时待第一次运行测量。
- 新选项eval_sample_indices仅两份OSC训练配置启用；旧配置未设置时保持原连续索引逻辑。原held-out manifest分支不变；训练样本、缓存、loss、保存间隔不变。
- 这是训练集固定probe，不是独立held-out验证。清单：configs/validation/libero_goal_osc_probes.json、libero_all4_osc_probes.json；可复现生成脚本scripts/prepare_libero_osc_validation.py。
