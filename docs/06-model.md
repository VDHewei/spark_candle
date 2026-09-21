# 06 · `src/model.rs`

Spark-X2.5（`Spark2_5ForCausalLM`）的 Candle 实现，外加 Hugging Face 权重下载逻辑。
分成 8 个编号小节，下面按小节说明。

---

## 0. 默认值函数

| 函数 | 值 | 用于 |
| --- | --- | --- |
| `default_rope_theta()` | `10000.0` | `RopeParam::rope_theta` |
| `one()` | `1.0` | `RopeParam::partial_rotary_factor` |
| `yes()` | `true` | `SparkConfig::tie_word_embeddings` |
| `default_gate_act()` | `"sigmoid"` | `SparkConfig::gate_attn_act_mode` |

都是 `#[serde(default = "...")]` 的回调函数，保证旧版 config 也能加载。

---

## 1. 配置结构体

### `struct RopeParam`

一类注意力层各自的 RoPE 参数：`partial_rotary_factor`（参与旋转的维度比例）、`rope_theta`（基频）。

### `struct RopeParameters`

`full_attention` 与 `sliding_attention` 两套参数。混合注意力模型两者可以不同 θ。

### `struct SparkConfig`

`config.json` 的完整映射。

| 字段 | 说明 |
| --- | --- |
| `vocab_size` / `hidden_size` / `intermediate_size` | 词表、隐藏层、FFN 中间层维度 |
| `num_hidden_layers` / `num_attention_heads` / `num_key_value_heads` | 层数、查询头数、KV 头数 |
| `rms_norm_eps` | RMSNorm 的 eps |
| `head_dim: Option<usize>` | Spark-X2.5 显式给出（2048/8=256），**信任配置**，不要自己算 |
| `tie_word_embeddings` | 输出头是否复用嵌入矩阵 |
| `sliding_window: Option<usize>` | 滑动窗口大小 |
| `layer_types: Vec<String>` | 逐层类型（见下） |
| `rope_parameters` | 可选的两套 RoPE 参数 |
| `headwise_attn_output_gate` / `gate_attn_act_mode` | 注意力输出门控开关与激活函数 |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 结束与填充符 |

方法：

- `head_dim()` —— 优先用配置值，否则 `hidden_size / num_attention_heads`
- `layer_type(idx)` —— 读 `layer_types`；缺省时按 **3:1**（`idx % 4 == 3` 为 full，其余 sliding）兜底
- `is_sliding(idx)` —— `layer_type(idx) == "sliding_attention"`
- `rope_params(layer_type)` —— 返回 `(rope_theta, partial_rotary_factor)`

---

## 2. `struct RmsNorm`

只对最后一维归一化的 RMSNorm。

- `load(dim, eps, vb)` —— 从当前 VarBuilder 前缀读 `weight`
- `forward(xs)` —— 内部强制转 FP32 计算（与 PyTorch 行为一致），再转回输入精度

---

## 3. RoPE 相关

### `fn apply_rotary_emb(x, cos, sin)`

支持 partial RoPE：只对前 `rope_dim` 维做旋转，其余维度原样透传。

```
x_rot = x[..rope_dim]
rotate_half(x_rot) = [-x2, x1]
out_rot = x_rot * cos + rotate_half(x_rot) * sin
out = concat(out_rot, x[rope_dim..])       // d > rope_dim 时
```

### `fn create_rope_cache(head_dim, max_seq_len, theta, partial_rotary_factor, device)`

预计算 `(cos, sin)`，形状均为 `(max_seq_len, rope_dim)`。
`freq_i = 1 / theta^(i/rope_dim)`（i 步长 2），随后拼一份凑满 rope_dim。
缓存长度 = `--max-context`，推理时按 `pos_offset` 切片。

### `fn repeat_kv(x, n_rep)`

GQA 的 KV 广播：`(b, kv_heads, s, d) → (b, kv_heads*n_rep, s, d)`，
顺序与 torch 官方 `repeat_kv` 一致（逐头重复）。`n_rep == 1` 时直接返回。

### `fn build_attn_mask(seq_len, kv_len, pos_offset, sliding_window, device)`

构造加性掩码（被屏蔽位置为 `-inf`）。关键点：

> 滑动窗口层在增量解码时会把 cache 裁剪到窗口大小，此时 **cache 下标 ≠ 绝对位置**，
> 因此先用 `kv_start = (pos_offset + seq_len) - kv_len` 把下标换算回绝对位置，再做窗口判断。

---

## 4. `struct SparkAttention`

融合 QKV 投影 + GQA + 滑动窗口 + 逐头输出门控。`forward` 共 8 步：

| 步 | 内容 |
| --- | --- |
| 1 | `q_k_v_proj` 一次投影后切成 Q / K / V |
| 2 | `g_proj` 由**原始 hidden states** 算门控分数（在 RoPE 之前） |
| 3 | 对 Q、K 施加 RoPE |
| 4 | 与历史 KV-Cache 拼接（沿 seq 维） |
| 5 | 滑动窗口层在**增量解码**（`seq_len == 1`）时把 cache 裁到窗口大小 |
| 6 | `repeat_kv` 广播到查询头数 |
| 7 | SDPA；`seq_len > 1`（prefill）时显式加因果 / 滑动窗口掩码 |
| 8 | 逐头输出门控（sigmoid 或 silu，FP32）后经 `out_proj` 投影 |

`enum GateAct`：`Sigmoid` / `Silu`，由 `gate_attn_act_mode` 决定（非 `"silu"` 一律 sigmoid）。

---

## 5. `struct SparkMlp`

门控 MLP：`down_proj(gelu(gate_proj(x)) * up_proj(x))`。
激活分支强制 FP32 计算，乘法前转回输入精度。

---

## 6. `struct SparkDecoderLayer`

```
x → input_layernorm → self_attn → +residual
  → post_attention_layernorm → mlp → +residual
```

`is_sliding` 决定用哪套 RoPE 缓存。

---

## 7. `struct SparkModel`

主干：嵌入 + N 层解码器 + 最终 norm + lm_head + 两套 RoPE 缓存。

### `load(vb, cfg, max_seq_len)`

自动兼容三种 checkpoint 差异：

| 差异 | 处理 |
| --- | --- |
| 权重是否带 `model.` 前缀 | 探测 `model.embedding.weight` / `model.embed_tokens.weight` |
| 嵌入叫 `embedding` 还是 `embed_tokens` | 探测 `base_vb.pp("embedding").contains_tensor("weight")` |
| `tie_word_embeddings` | 为真时 `lm_head` 直接复用嵌入矩阵，否则读 `lm_head` |

### `forward(input_ids, pos_offset, kv_caches)`

只返回**最后一个位置**的 logits（自回归解码只需要下一步）。

---

## 8. 权重与配置文件准备

### `struct ModelFiles`

`config` / `tokenizer` / `weights`（有序去重）/ `chat_template` 四类本地路径。

### `pub fn prepare_model_files(repo_id, shard_count) -> ModelFiles`

```
设置 HF_ENDPOINT = https://hf-mirror.com（国内镜像，海外可注释）
  → ApiBuilder::new().with_progress(true)
  → 下载 config.json / tokenizer.json / chat_template.jinja
  → 尝试 model.safetensors.index.json
       ├─ 有：按 weight_map 的文件名清单下载（排序 + 去重）
       └─ 无：按 shard_count 推断 model-00001-of-000NN.safetensors
```

已缓存的文件由 `hf-hub` 保证不重复下载。

---

## 常见改动

| 需求 | 改哪里 |
| --- | --- |
| 换镜像 / 关闭镜像 | `prepare_model_files` 里的 `std::env::set_var("HF_ENDPOINT", ...)` |
| 支持新的层类型 | `SparkConfig::layer_type` 的兜底规则 + `SparkAttention::forward` |
| 换激活函数 | `SparkMlp::forward`（`gelu_erf`）与 `GateAct` |
| 新增配置项 | `SparkConfig` 加 `#[serde(default)]` 字段，避免旧 checkpoint 加载失败 |
