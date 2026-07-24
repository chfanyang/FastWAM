# FastWAM MoT 实现详解

本文结合 FastWAM 当前仓库中的实际代码，解释 MoT 的模型组织方式、逐层计算过程、attention mask、训练数据流，以及动作推理时的 Video KV Cache。

主要代码：

- `src/fastwam/models/wan22/mot.py`
- `src/fastwam/models/wan22/fastwam.py`
- `src/fastwam/models/wan22/wan_video_dit.py`
- `src/fastwam/models/wan22/action_dit.py`

## 1. MoT 是什么

在这个仓库中，MoT 可以理解为：

```text
Mixture of Transformers
= 视频 token 使用视频专家
+ 动作 token 使用动作专家
+ 两类 token 在同一次 self-attention 中交换信息
```

它不是传统的 MoE：

```text
传统 MoE：
token → router → 选择若干 FFN expert

FastWAM MoT：
video token → 固定走 video expert
action token → 固定走 action expert
两者的 Q/K/V → 拼起来做一次 mixed attention
```

初始化代码如下：

```python
mot = MoT(
    mixtures={
        "video": video_expert,
        "action": action_expert,
    },
    mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
)
```

视频和动作模态不会通过 router 动态选择专家。视频 token 始终由视频专家处理，动作 token 始终由动作专家处理。

## 2. 两个专家的结构

### 2.1 Video Expert

Video Expert 来自 Wan 2.2 Video DiT，当前配置为：

```text
hidden_dim     = 3072
ffn_dim        = 14336
num_heads      = 24
attn_head_dim  = 128
num_layers     = 30
text_dim       = 4096
```

它主要负责处理：

```text
视频 latent token
视频 timestep
语言 context
视频位置编码
```

### 2.2 Action Expert

Action Expert 是 ActionDiT，当前配置为：

```text
hidden_dim     = 1024
ffn_dim        = 4096
num_heads      = 24
attn_head_dim  = 128
num_layers     = 30
text_dim       = 4096
action_dim     = 14
```

它主要负责处理：

```text
动作 token
动作 timestep
语言 context
动作位置编码
```

### 2.3 专家之间是否共享参数

两个专家各自拥有独立的：

```text
Q/K/V projection
attention output projection
LayerNorm
cross-attention
FFN
timestep modulation
```

它们不共享 Transformer block 参数。真正发生模态融合的位置是 mixed self-attention。

## 3. 为什么两个专家的 hidden_dim 可以不同

两个专家的 hidden dimension 分别是：

```text
video hidden_dim  = 3072
action hidden_dim = 1024
```

但它们的注意力维度相同：

```text
attention_dim = num_heads * attn_head_dim
              = 24 * 128
              = 3072
```

视频专家的 Q/K/V projection：

```text
[B, Sv, 3072] → [B, Sv, 3072]
```

动作专家的 Q/K/V projection：

```text
[B, Sa, 1024] → [B, Sa, 3072]
```

因此，虽然原始 token 的 hidden dimension 不同，Q/K/V 的最后一维都是 3072，可以沿序列维拼接。

mixed attention 完成后，两个专家分别使用自己的 output projection：

```text
video:
[B, Sv, 3072]
→ video attention output projection
→ [B, Sv, 3072]

action:
[B, Sa, 3072]
→ action attention output projection
→ [B, Sa, 1024]
```

所以 MoT 要求以下结构一致：

```text
num_heads
attn_head_dim
num_layers
```

但不要求 `hidden_dim` 和 `ffn_dim` 相同。

## 4. MoT 的输入

FastWAM 会先分别执行两个专家的 `pre_dit()`。

视频分支：

```python
video_pre = video_expert.pre_dit(
    x=latents,
    timestep=timestep_video,
    context=context,
    context_mask=context_mask,
)
```

得到：

```text
video tokens
video RoPE
video timestep modulation
video text context
video context mask
```

动作分支：

```python
action_pre = action_expert.pre_dit(
    action_tokens=noisy_action,
    timestep=timestep_action,
    context=context,
    context_mask=context_mask,
)
```

得到：

```text
action tokens
action RoPE
action timestep modulation
action text context
action context mask
```

然后传入 MoT：

```python
tokens_out = mot(
    embeds_all={
        "video": video_pre["tokens"],
        "action": action_pre["tokens"],
    },
    freqs_all={
        "video": video_pre["freqs"],
        "action": action_pre["freqs"],
    },
    context_all={
        "video": {
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        "action": {
            "context": action_pre["context"],
            "mask": action_pre["context_mask"],
        },
    },
    t_mod_all={
        "video": video_pre["t_mod"],
        "action": action_pre["t_mod"],
    },
    attention_mask=attention_mask,
)
```

视频和动作拥有独立的 timestep。例如：

```text
video timestep = 820
action timestep = 430
```

所以同一个 MoT layer 中，视频和动作可以处于不同的噪声强度。

## 5. 单层 MoT 的完整计算过程

MoT 当前有 30 层。下面以第 `i` 层为例。

### 5.1 视频专家生成 Q/K/V

视频 token 依次经过：

```text
LayerNorm
→ video timestep modulation
→ video Q/K/V projection
→ Q/K RMSNorm
→ RoPE
```

得到：

```text
q_video: [B, Sv, 3072]
k_video: [B, Sv, 3072]
v_video: [B, Sv, 3072]
```

### 5.2 动作专家生成 Q/K/V

动作 token 独立经过：

```text
LayerNorm
→ action timestep modulation
→ action Q/K/V projection
→ Q/K RMSNorm
→ RoPE
```

得到：

```text
q_action: [B, Sa, 3072]
k_action: [B, Sa, 3072]
v_action: [B, Sa, 3072]
```

这里使用的是动作专家自己的参数。

### 5.3 拼接 Q/K/V

MoT 沿序列维拼接：

```python
q_cat = torch.cat([q_video, q_action], dim=1)
k_cat = torch.cat([k_video, k_action], dim=1)
v_cat = torch.cat([v_video, v_action], dim=1)
```

得到：

```text
q_cat: [B, Sv + Sa, 3072]
k_cat: [B, Sv + Sa, 3072]
v_cat: [B, Sv + Sa, 3072]
```

序列顺序固定是：

```text
[全部 video tokens][全部 action tokens]
```

### 5.4 Mixed Attention

拼接后的 Q/K/V 进入一次 Flash Attention：

```python
mixed = flash_attention(
    q=q_cat,
    k=k_cat,
    v=v_cat,
    num_heads=24,
    ctx_mask=attention_mask,
)
```

它在形式上是一次联合 attention，但哪些 query 能读取哪些 key，由 attention mask 决定。

### 5.5 拆分 Attention 输出

联合 attention 输出再按原来的序列长度拆开：

```python
mixed_video = mixed[:, :Sv]
mixed_action = mixed[:, Sv:Sv + Sa]
```

随后：

```text
mixed_video
→ video attention output projection
→ video hidden space

mixed_action
→ action attention output projection
→ action hidden space
```

### 5.6 各自执行 Post Block

拆分以后，视频和动作分别执行自己的：

```text
attention residual + gate
→ text cross-attention
→ MLP modulation
→ FFN
→ FFN residual + gate
```

得到：

```text
updated_video_tokens
updated_action_tokens
```

然后进入下一层。

单层总体结构：

```text
video tokens ─→ video Q/K/V ─┐
                             ├→ mixed attention ─┬→ video post-block
action tokens ─→ action Q/K/V┘                  └→ action post-block
                                                        │
                                                   下一层 MoT
```

## 6. Timestep Modulation

每个 DiT block 都会根据 timestep 产生六组控制量：

```text
shift_msa
scale_msa
gate_msa
shift_mlp
scale_mlp
gate_mlp
```

self-attention 前的 modulation：

```text
attn_input =
    norm1(x) * (1 + scale_msa)
    + shift_msa
```

attention residual：

```text
x =
    residual_x
    + gate_msa * attention_output
```

FFN 前的 modulation：

```text
mlp_input =
    norm2(x) * (1 + scale_mlp)
    + shift_mlp
```

FFN residual：

```text
x =
    x
    + gate_mlp * ffn_output
```

视频和动作使用不同的 timestep modulation：

```text
video t_mod 由 video timestep 产生
action t_mod 由 action timestep 产生
```

因此，在同一层 mixed attention 中：

```text
视频专家知道视频当前的噪声强度
动作专家知道动作当前的噪声强度
```

## 7. Attention Mask

Attention mask 是理解 FastWAM MoT 的关键。

mask 的行表示 query，列表示允许读取的 key：

| Query / Key | 首帧视频 | 未来视频 | 动作 |
|---|---:|---:|---:|
| 首帧视频 | 允许 | 禁止 | 禁止 |
| 未来视频 | 允许 | 允许 | 禁止 |
| 动作 | 允许 | 禁止 | 允许 |

### 7.1 首帧视频只能看首帧

```text
first-frame video query
→ first-frame video keys
```

首帧视频 token 不允许读取未来视频 token，避免未来信息泄漏到当前观测表示中。

### 7.2 未来视频可以看所有视频

```text
future video query
→ first-frame video keys
→ future video keys
```

未来视频 token 之间可以交换信息。

### 7.3 视频不能看动作

```text
video query
→ action keys: 禁止
```

因此，视频表示不会依赖动作 token。

### 7.4 动作可以看首帧视频

```text
action query
→ first-frame video keys: 允许
```

动作 token 能够直接读取当前机器人视觉观测。

### 7.5 动作不能看未来视频

```text
action query
→ future video keys: 禁止
```

动作预测不依赖模型生成的未来视频。这与 FastWAM 的核心目标一致：

```text
动作直接根据当前视觉观测预测
不要求测试时先生成未来视频再决定动作
```

### 7.6 动作 token 之间双向可见

```text
action query
→ all action keys
```

整个 action horizon 是并行 diffusion 生成的，不是 autoregressive 地逐个动作生成。

例如 32 个动作 token：

```text
action_0 可以读取 action_1 ... action_31
action_31 也可以读取 action_0 ... action_30
```

## 8. 为什么仍称为 Mixed Attention

在当前 mask 下，动作 attention 使用：

```text
query = action Q

key =
    first-frame video K
    + action K

value =
    first-frame video V
    + action V
```

即：

```text
action_output =
    attention(
        query=action_Q,
        key=[first_frame_video_K, action_K],
        value=[first_frame_video_V, action_V]
    )
```

动作 attention 的输出同时包含：

```text
当前视觉信息
+
动作序列内部信息
```

而视频侧是：

```text
video_output =
    attention(
        query=video_Q,
        key=video_K,
        value=video_V
    )
```

因此当前结构中的主要跨模态信息流是：

```text
video observation → action
```

而不是完全对称的：

```text
video ↔ action
```

## 9. 文本与 Proprio 如何进入模型

语言 context 不会直接拼入 mixed self-attention 的 video/action token 序列。

每个专家完成 mixed self-attention 后，分别执行自己的 cross-attention：

```python
x = x + block.cross_attn(
    block.norm3(x),
    context,
)
```

因此：

```text
video tokens
→ video cross-attention
→ language/proprio context

action tokens
→ action cross-attention
→ language/proprio context
```

两个专家读取同一条语言指令，但使用各自独立的 cross-attention 参数。

RoboTwin 的 14 维 proprio 先经过：

```text
Linear(14, 4096)
```

再作为额外 context token 追加到语言 embedding 后面：

```text
[language token 0]
[language token 1]
...
[language token 127]
[proprio token]
```

视频专家和动作专家都可以通过各自的 cross-attention 读取这些 context。

## 10. 训练时的数据流

训练时，视频与动作分别加噪：

```text
clean video latent
→ video scheduler
→ noisy video latent

clean action
→ action scheduler
→ noisy action
```

视频与动作 timestep 独立采样：

```text
timestep_video = train_video_scheduler.sample_training_t(...)
timestep_action = train_action_scheduler.sample_training_t(...)
```

经过 MoT 后分别输出：

```text
pred_video
pred_action
```

并分别计算：

```text
video loss
action loss
```

最后：

```text
total loss =
    lambda_video * video_loss
    + lambda_action * action_loss
```

虽然视频 token 在 forward 中不能读取动作 token，但 Action Loss 可以通过以下路径把梯度传回视频表示：

```text
action output
→ mixed attention
→ first-frame video K/V
→ video expert
```

这些视频参数最终是否更新，还取决于 Trainer 的参数冻结设置。

## 11. 动作推理时的 Video KV Cache

动作 diffusion 推理通常包含多个去噪 step：

```text
random action noise
→ denoise step 1
→ denoise step 2
→ ...
→ clean action
```

在所有动作去噪 step 中，以下内容保持不变：

```text
当前输入图像
语言指令
proprio
video timestep
```

如果每一步都重新计算完整视频专家，会造成大量重复计算。

因此 MoT 提供：

```python
prefill_video_cache()
```

它只执行一次视频分支，并保存每一层的 Video K/V：

```text
layer 0: video K/V
layer 1: video K/V
...
layer 29: video K/V
```

注意缓存的是每一层进入 attention 时的 K/V，不只是最后一层的视频 token。

动作去噪的每一步只重新计算：

```text
action Q/K/V
```

然后使用：

```text
query = current action Q

key =
    cached video K
    + current action K

value =
    cached video V
    + current action V
```

总体流程：

```text
当前图像
  ↓
video expert prefill，一次
  ↓
30 层 video K/V cache
  │
  ├→ action denoise step 1
  ├→ action denoise step 2
  ├→ action denoise step 3
  └→ ...
```

之所以可以缓存，是因为：

```text
视频 token 不读取动作 token
视频输入在动作去噪过程中不变
video timestep 在动作去噪过程中不变
```

动作 timestep 每一步都会改变，所以 Action Q/K/V 仍然必须重新计算。

## 12. Gradient Checkpointing

MoT 中存在两部分 gradient checkpointing。

### 12.1 Mixed Attention Checkpoint

由下面的参数控制：

```text
mot_checkpoint_mixed_attn
```

开启后，训练前向不会保存 mixed attention 的全部中间激活，反向传播时重新执行这一部分。

### 12.2 Expert Post Block Checkpoint

每个专家还拥有：

```text
use_gradient_checkpointing
```

它控制以下部分是否 checkpoint：

```text
attention output projection
cross-attention
FFN
```

当前配置将两者都关联到：

```yaml
model.mot_checkpoint_mixed_attn: true
```

效果：

```text
显存占用下降
反向传播重计算增加
训练时间上升
数学计算结果不变
```

## 13. MoT 与普通共享 Transformer 的区别

如果使用一个普通共享 Transformer，视频和动作 token 必须使用相同的 hidden dimension、相同的 FFN，以及相同的一套 block 参数。

FastWAM MoT 则允许：

```text
video hidden_dim  = 3072
action hidden_dim = 1024

video ffn_dim     = 14336
action ffn_dim    = 4096
```

这样可以：

1. 保留 Wan Video DiT 的预训练视觉能力。
2. 使用更轻量的 ActionDiT 处理动作。
3. 只在统一的 attention space 中进行信息交互。
4. 为视频和动作使用独立的 timestep conditioning。
5. 在动作推理时缓存不变的 Video K/V。

## 14. 推荐调试位置

### Q/K/V 构造

```text
MoT._build_expert_attention_io()
```

建议查看：

```text
x.shape
q.shape
k.shape
v.shape
t_mod.shape
```

### 联合注意力

```text
MoT._mixed_attention()
```

建议查看：

```text
q_cat.shape
k_cat.shape
v_cat.shape
attention_mask.shape
```

### 完整逐层循环

```text
MoT.forward()
```

建议查看：

```text
layer_idx
seq_lens
mixed.shape
mixed_slice.shape
tokens_all["video"].shape
tokens_all["action"].shape
```

### Attention Mask

```text
FastWAM._build_mot_attention_mask()
```

建议直接查看三个区域：

```text
mask[:Sv, :Sv]   video → video
mask[:Sv, Sv:]   video → action
mask[Sv:, :Sv]   action → video
mask[Sv:, Sv:]   action → action
```

### 推理缓存

```text
MoT.prefill_video_cache()
MoT.forward_action_with_video_cache()
FastWAM.infer_action()
```

## 15. 总结

FastWAM 的 MoT 不是让 token 在视频专家和动作专家之间动态选择，而是：

```text
视频和动作使用各自独立的 Transformer 参数，
把各自生成的 Q/K/V 投影到统一的注意力空间，
通过有方向限制的 attention mask，
让动作读取当前视觉信息，
然后各自回到自己的专家中继续处理。
```

当前实现最核心的结构特征是：

```text
video expert 与 action expert 参数独立
Q/K/V attention space 维度一致
action 可以读取 first-frame video
action 不读取 future video
video 不读取 action
action horizon 内部双向 attention
文本和 proprio 通过各自的 cross-attention 注入
动作推理时复用逐层 Video K/V Cache
```
