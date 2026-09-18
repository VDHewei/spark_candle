//! Spark-X2.5 (Spark2_5ForCausalLM) 的 Candle 实现与权重下载逻辑

use anyhow::{Context, Result as AnyhowResult};
use candle_core::{DType, Device, IndexOp, Module, Result as CandleResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use std::path::PathBuf;

// ==========================================
// 0. 常用默认值
// ==========================================
/// `rope_theta` 的默认值（config.json 未给出时使用）
fn default_rope_theta() -> f32 {
    10000.0
}
/// `partial_rotary_factor` 的默认值：对全部 head_dim 做旋转
fn one() -> f32 {
    1.0
}
/// `tie_word_embeddings` 的默认值：嵌入与输出头共享权重
fn yes() -> bool {
    true
}
/// `gate_attn_act_mode` 的默认值：注意力输出门控用 sigmoid
fn default_gate_act() -> String {
    "sigmoid".to_string()
}

// ==========================================
// 1. 模型配置结构体
// ==========================================
/// 一类注意力层（full / sliding）各自的 RoPE 参数
#[derive(Deserialize, Debug, Clone)]
pub struct RopeParam {
    /// 参与旋转的维度比例：1.0 表示整个 head_dim 都旋转
    #[serde(default = "one")]
    pub partial_rotary_factor: f32,
    /// RoPE 的基频 θ
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
}

/// 默认 RoPE 参数：全部维度参与旋转、θ = 10000
impl Default for RopeParam {
    fn default() -> Self {
        Self {
            partial_rotary_factor: 1.0,
            rope_theta: 10000.0,
        }
    }
}

/// 混合注意力模型下两类层各自独立的 RoPE 参数集合
#[derive(Deserialize, Debug, Clone, Default)]
pub struct RopeParameters {
    /// 全注意力层使用的参数
    #[serde(default)]
    pub full_attention: RopeParam,
    /// 滑动窗口层使用的参数
    #[serde(default)]
    pub sliding_attention: RopeParam,
}

/// `config.json` 的完整结构映射
///
/// 字段缺省时尽量给出与官方 `modeling_spark.py` 一致的兜底值，
/// 以便兼容不同版本 / 不同体量的 checkpoint。
#[derive(Deserialize, Debug, Clone)]
pub struct SparkConfig {
    /// 词表大小
    pub vocab_size: usize,
    /// 隐藏层维度
    pub hidden_size: usize,
    /// FFN 中间层维度
    pub intermediate_size: usize,
    /// 解码层数
    pub num_hidden_layers: usize,
    /// 查询头数
    pub num_attention_heads: usize,
    /// KV 头数（GQA，通常小于查询头数）
    pub num_key_value_heads: usize,
    /// RMSNorm 的 eps
    pub rms_norm_eps: f64,

    // Spark-X2.5 显式给出 head_dim（2048/8=256，但不要自己算，直接信任配置）
    #[serde(default)]
    pub head_dim: Option<usize>,

    #[serde(default = "yes")]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub sliding_window: Option<usize>,

    // 逐层类型：sliding_attention / full_attention
    #[serde(default)]
    pub layer_types: Vec<String>,

    #[serde(default)]
    pub rope_parameters: Option<RopeParameters>,

    // 注意力输出门控（headwise_attn_output_gate）
    #[serde(default)]
    pub headwise_attn_output_gate: bool,

    #[serde(default = "default_gate_act")]
    pub gate_attn_act_mode: String,

    #[allow(dead_code)]
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
}

impl SparkConfig {
    /// 每个注意力头的维度
    ///
    /// Spark-X2.5 显式给出了 `head_dim`（2048 / 8 = 256），
    /// 直接信任配置；没有时才按 `hidden_size / num_attention_heads` 推算。
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// 返回该层的类型；config 里没有 layer_types 时按 3:1（sliding:full）兜底
    pub fn layer_type(&self, layer_idx: usize) -> &'static str {
        match self.layer_types.get(layer_idx).map(|s| s.as_str()) {
            Some("full_attention") => "full_attention",
            Some("sliding_attention") => "sliding_attention",
            _ => {
                if layer_idx % 4 == 3 {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            }
        }
    }

    /// 该层是否为滑动窗口注意力层
    pub fn is_sliding(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == "sliding_attention"
    }

    /// 取指定层类型的 RoPE 参数
    ///
    /// # 参数
    /// - `layer_type`：`"full_attention"` 或 `"sliding_attention"`
    ///
    /// # 返回
    /// `(rope_theta, partial_rotary_factor)`；config 未给出时用默认值
    pub fn rope_params(&self, layer_type: &str) -> (f32, f32) {
        if let Some(rp) = &self.rope_parameters {
            let p = if layer_type == "sliding_attention" {
                &rp.sliding_attention
            } else {
                &rp.full_attention
            };
            (p.rope_theta, p.partial_rotary_factor)
        } else {
            (default_rope_theta(), 1.0)
        }
    }
}

// ==========================================
// 2. RMSNorm 层实现
// ==========================================
/// RMSNorm：只对最后一维做归一化，带可学习缩放
pub struct RmsNorm {
    /// 缩放权重，形状 `[hidden_size]`
    weight: Tensor,
    /// 数值稳定的 eps
    eps: f64,
}

impl RmsNorm {
    /// 从 VarBuilder 的当前前缀下读取 `weight`
    ///
    /// # 参数
    /// - `dim`：权重长度（= hidden_size）
    /// - `eps`：归一化 eps
    /// - `vb`：定位到该 norm 层的 VarBuilder
    pub fn load(dim: usize, eps: f64, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    /// 前向：内部强制用 FP32 计算再转回原精度，与 PyTorch 实现保持一致
    ///
    /// # 参数
    /// - `xs`：输入张量，形状 `[b, seq, hidden]`
    ///
    /// # 返回
    /// 与输入同形状、同精度的归一化结果
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let dtype = xs.dtype();
        // 与 PyTorch 实现保持一致：内部强制用 FP32 计算，再转回原精度
        let xs = xs.to_dtype(DType::F32)?;
        let variance = xs.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        xs.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }
}

// ==========================================
// 3. Rotary Embedding (RoPE)，支持 partial_rotary_factor
// ==========================================

/// 施加旋转位置编码：只对前 `rope_dim` 维做旋转，其余维度原样透传
///
/// 与官方 `modeling_spark.py` 完全一致，支持 `partial_rotary_factor < 1`。
///
/// # 参数
/// - `x`：Q 或 K，形状 `[b, heads, seq, head_dim]`
/// - `cos` / `sin`：RoPE 缓存切片，形状 `[seq, rope_dim]`
///
/// # 返回
/// 旋转后的张量，形状与 `x` 相同
fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> CandleResult<Tensor> {
    let (_b_sz, _h, _seq_len, d) = x.dims4()?;
    let rope_dim = cos.dim(D::Minus1)?;

    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;

    // cos/sin: (seq_len, rope_dim) -> (1, 1, seq_len, rope_dim)
    let cos = cos.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(0)?;

    let x_rot = x.narrow(D::Minus1, 0, rope_dim)?;
    let half = rope_dim / 2;
    let x1 = x_rot.narrow(D::Minus1, 0, half)?;
    let x2 = x_rot.narrow(D::Minus1, half, half)?;
    // rotate_half(x) = [-x2, x1]
    let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    let out_rot = (x_rot.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?;

    let out = if d > rope_dim {
        let x_pass = x.narrow(D::Minus1, rope_dim, d - rope_dim)?;
        Tensor::cat(&[&out_rot, &x_pass], D::Minus1)?
    } else {
        out_rot
    };
    out.to_dtype(dtype)
}

/// 预计算 RoPE 的 cos / sin 缓存
///
/// `freq_i = 1 / theta^(i / rope_dim)`（i 步长 2），随后复制一份拼成完整 rope_dim。
/// 缓存长度即上下文上限，推理时按 `pos_offset` 切片使用。
///
/// # 参数
/// - `head_dim`：注意力头维度
/// - `max_seq_len`：缓存覆盖的最大位置数（= `--max-context`）
/// - `theta`：RoPE 基频
/// - `partial_rotary_factor`：参与旋转的维度比例
/// - `device`：计算设备
///
/// # 返回
/// `(cos, sin)`，形状均为 `(max_seq_len, rope_dim)`
fn create_rope_cache(
    head_dim: usize,
    max_seq_len: usize,
    theta: f32,
    partial_rotary_factor: f32,
    device: &Device,
) -> CandleResult<(Tensor, Tensor)> {
    let rope_dim = ((head_dim as f32) * partial_rotary_factor).round() as usize;
    let rope_dim = rope_dim.max(2);
    let inv_freq: Vec<f32> = (0..rope_dim)
        .step_by(2)
        .map(|i| (1.0f64 / (theta as f64).powf(i as f64 / rope_dim as f64)) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq, device)?;
    let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(DType::F32)?;
    let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// GQA 的 KV 广播：把 `(b, kv_heads, s, d)` 扩展成 `(b, kv_heads * n_rep, s, d)`
///
/// 扩展顺序与 torch 官方 `repeat_kv` 保持一致（逐头重复而非整块复制）。
///
/// # 参数
/// - `x`：K 或 V，形状 `[b, kv_heads, s, d]`
/// - `n_rep`：每个 KV 头对应的查询头数；为 1 时直接原样返回
fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
}

/// 构造因果掩码（可选滑动窗口），被屏蔽位置为 `-inf`
///
/// 滑动窗口层在增量解码时会把 cache 裁剪到窗口大小，此时 cache 下标不再等于
/// 绝对位置，因此这里先把下标换算回绝对位置再做窗口判断。
///
/// # 参数
/// - `seq_len`：本次前向的 Query 长度
/// - `kv_len`：Key/Value 总长度（含历史 cache）
/// - `pos_offset`：本次 Query 的起始绝对位置
/// - `sliding_window`：`Some(w)` 时只可见最近 w 个位置
/// - `device`：计算设备
///
/// # 返回
/// 形状 `(seq_len, kv_len)` 的加性掩码，需与注意力分数相加
fn build_attn_mask(
    seq_len: usize,
    kv_len: usize,
    pos_offset: usize,
    sliding_window: Option<usize>,
    device: &Device,
) -> CandleResult<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; seq_len * kv_len];
    // 滑动窗口层在增量解码时会把 cache 裁剪到窗口大小，cache 下标不再等于绝对位置，
    // 这里把下标换算回绝对位置，窗口判断才不会错位
    let kv_start = (pos_offset + seq_len).saturating_sub(kv_len);
    for i in 0..seq_len {
        let q_pos = pos_offset + i;
        for j in 0..kv_len {
            let key_pos = kv_start + j;
            let visible = key_pos <= q_pos && sliding_window.map_or(true, |w| q_pos - key_pos < w);
            if visible {
                data[i * kv_len + j] = 0.0;
            }
        }
    }
    Tensor::from_vec(data, (seq_len, kv_len), device)
}

// ==========================================
// 4. Spark 混合注意力层 (Hybrid Attention + Output Gate)
// ==========================================
/// 注意力输出门控的激活函数种类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GateAct {
    /// sigmoid(x)
    Sigmoid,
    /// silu(x)
    Silu,
}

/// Spark 混合注意力层：融合 QKV 投影 + GQA + 滑动窗口 + 逐头输出门控
pub struct SparkAttention {
    /// 融合的 QKV 投影：`hidden -> q_dim + 2 * kv_dim`
    q_k_v_proj: Linear,
    /// 注意力输出门控：`hidden -> num_heads`；配置关闭时为 `None`
    g_proj: Option<Linear>,
    /// 输出投影：`q_dim -> hidden`
    out_proj: Linear,
    /// 门控激活函数
    gate_act: GateAct,
    /// 查询头数
    num_heads: usize,
    /// KV 头数
    num_kv_heads: usize,
    /// 每头维度
    head_dim: usize,
    /// `num_heads * head_dim`
    q_dim: usize,
    /// `num_kv_heads * head_dim`
    kv_dim: usize,
    /// 滑动窗口大小；`None` 表示全注意力层
    sliding_window: Option<usize>,
}

impl SparkAttention {
    /// 从 VarBuilder（已定位到 `self_attn`）加载权重并初始化
    ///
    /// # 参数
    /// - `vb`：定位到本层 `self_attn` 的 VarBuilder
    /// - `cfg`：模型配置
    /// - `layer_idx`：层序号，用于判断是否为滑动窗口层
    pub fn load(vb: VarBuilder, cfg: &SparkConfig, layer_idx: usize) -> CandleResult<Self> {
        let head_dim = cfg.head_dim();
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;

        let q_k_v_proj = linear_no_bias(cfg.hidden_size, q_dim + 2 * kv_dim, vb.pp("q_k_v_proj"))?;
        let g_proj = if cfg.headwise_attn_output_gate {
            Some(linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads,
                vb.pp("g_proj"),
            )?)
        } else {
            None
        };
        let out_proj = linear_no_bias(q_dim, cfg.hidden_size, vb.pp("out_proj"))?;

        let sliding_window = if cfg.is_sliding(layer_idx) {
            cfg.sliding_window
        } else {
            None
        };

        let gate_act = match cfg.gate_attn_act_mode.as_str() {
            "silu" => GateAct::Silu,
            _ => GateAct::Sigmoid,
        };

        Ok(Self {
            q_k_v_proj,
            g_proj,
            out_proj,
            gate_act,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            q_dim,
            kv_dim,
            sliding_window,
        })
    }

    /// 注意力前向，共 8 步：QKV 切分 → 门控分数 → RoPE → 拼 cache → 窗口裁剪
    /// → GQA 广播 → SDPA（含掩码）→ 输出门控与投影
    ///
    /// # 参数
    /// - `xs`：形状 `[b, seq, hidden]`
    /// - `cos` / `sin`：本段对应的 RoPE 切片，形状 `[seq, rope_dim]`
    /// - `kv_cache`：本层的 KV 缓存，函数内部就地更新
    /// - `pos_offset`：本段的起始绝对位置
    ///
    /// # 返回
    /// 注意力输出，形状 `[b, seq, hidden]`
    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        pos_offset: usize,
    ) -> CandleResult<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        // 1. 融合 QKV 投影后切分
        let qkv = self.q_k_v_proj.forward(xs)?;
        let q = qkv.narrow(D::Minus1, 0, self.q_dim)?;
        let k = qkv.narrow(D::Minus1, self.q_dim, self.kv_dim)?;
        let v = qkv.narrow(D::Minus1, self.q_dim + self.kv_dim, self.kv_dim)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 2. 门控分数（在 RoPE 之前由原始 hidden states 计算）
        let gate_score = match &self.g_proj {
            None => None,
            Some(g_proj) => Some(
                g_proj
                    .forward(xs)?
                    .reshape((b_sz, seq_len, self.num_heads, 1))?
                    .transpose(1, 2)?,
            ),
        };

        // 3. 位置编码
        let q = apply_rotary_emb(&q, cos, sin)?;
        let k = apply_rotary_emb(&k, cos, sin)?;

        // 4. 拼接历史 KV Cache
        let (mut k, mut v) = match kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k as &Tensor, &k], 2)?;
                let v = Tensor::cat(&[prev_v as &Tensor, &v], 2)?;
                (k, v)
            }
        };

        // 5. 滑动窗口层：增量解码阶段把 cache 裁剪到窗口大小
        //    （prefill 阶段保留完整 cache，靠 mask 实现窗口注意力）
        if let Some(window) = self.sliding_window {
            let kv_len = k.dim(2)?;
            if seq_len == 1 && kv_len > window {
                k = k.narrow(2, kv_len - window, window)?;
                v = v.narrow(2, kv_len - window, window)?;
            }
        }
        *kv_cache = Some((k.clone(), v.clone()));
        let kv_len = k.dim(2)?;

        // 6. GQA 广播
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(k, n_rep)?;
        let v = repeat_kv(v, n_rep)?;

        // 7. Scaled Dot-Product Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // prefill（seq_len > 1）时必须显式加因果 / 滑动窗口掩码
        let att = if seq_len > 1 {
            let mask =
                build_attn_mask(seq_len, kv_len, pos_offset, self.sliding_window, xs.device())?;
            att.broadcast_add(&mask.to_dtype(att.dtype())?)?
        } else {
            att
        };

        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        let mut context = att.matmul(&v)?; // (b, heads, seq, head_dim)

        // 8. 逐头输出门控
        if let Some(gate_score) = gate_score {
            let gate = match self.gate_act {
                GateAct::Sigmoid => candle_nn::ops::sigmoid(&gate_score.to_dtype(DType::F32)?)?,
                GateAct::Silu => gate_score.to_dtype(DType::F32)?.silu()?,
            };
            context = context
                .to_dtype(DType::F32)?
                .broadcast_mul(&gate)?
                .to_dtype(context.dtype())?;
        }

        let context = context
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.num_heads * self.head_dim))?;

        self.out_proj.forward(&context)
    }
}

// ==========================================
// 5. MLP 层实现（激活函数为 GELU）
// ==========================================
/// 门控 MLP（SwiGLU 变体，激活函数为 GELU）
pub struct SparkMlp {
    /// 门控分支投影：`hidden -> intermediate`
    gate_proj: Linear,
    /// 上投影：`hidden -> intermediate`
    up_proj: Linear,
    /// 下投影：`intermediate -> hidden`
    down_proj: Linear,
}

impl SparkMlp {
    /// 从 VarBuilder（已定位到 `mlp`）加载三个投影矩阵
    ///
    /// # 参数
    /// - `vb`：定位到本层 `mlp` 的 VarBuilder
    /// - `cfg`：模型配置（取 `hidden_size` / `intermediate_size`）
    pub fn load(vb: VarBuilder, cfg: &SparkConfig) -> CandleResult<Self> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// 前向：`down_proj(gelu(gate_proj(x)) * up_proj(x))`
    ///
    /// 激活分支强制用 FP32 计算（与 PyTorch 行为一致），乘法前再转回输入精度。
    ///
    /// # 参数
    /// - `xs`：形状 `[b, seq, hidden]`
    ///
    /// # 返回
    /// 同形状的 MLP 输出
    pub fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        // PyTorch: down_proj(act_fn(gate_proj(x)) * up_proj(x))，act_fn = gelu
        let lhs = self
            .gate_proj
            .forward(xs)?
            .to_dtype(DType::F32)?
            .gelu_erf()?;
        let lhs = lhs.to_dtype(xs.dtype())?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

// ==========================================
// 6. Spark 解码层组合
// ==========================================
/// 单个解码层：RMSNorm → 注意力（残差）→ RMSNorm → MLP（残差）
pub struct SparkDecoderLayer {
    /// 注意力前的 RMSNorm
    input_layernorm: RmsNorm,
    /// 混合注意力层
    self_attn: SparkAttention,
    /// MLP 前的 RMSNorm
    post_attention_layernorm: RmsNorm,
    /// 前馈网络
    mlp: SparkMlp,
    /// 是否滑动窗口层（决定用哪套 RoPE 缓存）
    is_sliding: bool,
}

impl SparkDecoderLayer {
    /// 从 VarBuilder（已定位到 `layers.<idx>`）加载整层权重
    ///
    /// # 参数
    /// - `vb`：定位到本层的 VarBuilder
    /// - `cfg`：模型配置
    /// - `layer_idx`：层序号
    pub fn load(vb: VarBuilder, cfg: &SparkConfig, layer_idx: usize) -> CandleResult<Self> {
        let input_layernorm =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let self_attn = SparkAttention::load(vb.pp("self_attn"), cfg, layer_idx)?;
        let post_attention_layernorm = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let mlp = SparkMlp::load(vb.pp("mlp"), cfg)?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
            is_sliding: cfg.is_sliding(layer_idx),
        })
    }

    /// 前向：两轮「归一化 → 子层 → 加残差」
    ///
    /// # 参数
    /// - `xs`：形状 `[b, seq, hidden]`
    /// - `cos` / `sin`：本段 RoPE 切片
    /// - `kv_cache`：本层 KV 缓存（就地更新）
    /// - `pos_offset`：本段起始绝对位置
    ///
    /// # 返回
    /// 同形状的层输出
    pub fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        pos_offset: usize,
    ) -> CandleResult<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, kv_cache, pos_offset)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

// ==========================================
// 7. SparkModel 核心大模型结构
// ==========================================
/// `Spark2_5ForCausalLM` 的主体：嵌入 + N 层解码器 + 输出头 + 两套 RoPE 缓存
pub struct SparkModel {
    /// 词嵌入，对应权重名 `model.embedding.weight`
    embed_tokens: Embedding,
    /// 全部解码层
    layers: Vec<SparkDecoderLayer>,
    /// 输出前的最终 RMSNorm
    norm: RmsNorm,
    /// 输出投影；`tie_word_embeddings` 时与嵌入共享权重
    lm_head: Linear,
    // 两种层类型各自独立的位置编码缓存
    /// 全注意力层的 cos 缓存
    cos_full: Tensor,
    /// 全注意力层的 sin 缓存
    sin_full: Tensor,
    /// 滑动窗口层的 cos 缓存
    cos_sliding: Tensor,
    /// 滑动窗口层的 sin 缓存
    sin_sliding: Tensor,
}

impl SparkModel {
    /// 构建模型结构并绑定权重
    ///
    /// 自动兼容三种常见差异：权重是否带 `model.` 前缀、嵌入叫 `embedding`
    /// 还是 `embed_tokens`、输出头是否与嵌入共享（`tie_word_embeddings`）。
    ///
    /// # 参数
    /// - `vb`：根 VarBuilder（已内存映射 Safetensors）
    /// - `cfg`：模型配置
    /// - `max_seq_len`：RoPE 缓存长度（= `--max-context`）
    ///
    /// # 返回
    /// 就绪的 `SparkModel`；权重缺失或形状不符时返回错误
    pub fn load(vb: VarBuilder, cfg: &SparkConfig, max_seq_len: usize) -> CandleResult<Self> {
        // 权重带 "model." 前缀（Spark-X2.5 的 base_model_prefix = "model"）
        let has_model_prefix = vb.contains_tensor("model.embedding.weight")
            || vb.contains_tensor("model.embed_tokens.weight");
        let base_vb = if has_model_prefix {
            vb.pp("model")
        } else {
            vb.clone()
        };

        // 嵌入层：Spark 用的是 "embedding"，其它 Qwen 系可能是 "embed_tokens"
        let emb_name = if base_vb.pp("embedding").contains_tensor("weight") {
            "embedding"
        } else {
            "embed_tokens"
        };
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, base_vb.pp(emb_name))?;
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, base_vb.pp("norm"))?;

        let mut layers = Vec::new();
        let vb_layers = base_vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = SparkDecoderLayer::load(vb_layers.pp(layer_idx), cfg, layer_idx)?;
            layers.push(layer);
        }

        // 输出头：tie_word_embeddings=true 时权重不单独存储，直接复用嵌入矩阵
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        let head_dim = cfg.head_dim();
        let (full_theta, full_prf) = cfg.rope_params("full_attention");
        let (sliding_theta, sliding_prf) = cfg.rope_params("sliding_attention");
        let (cos_full, sin_full) =
            create_rope_cache(head_dim, max_seq_len, full_theta, full_prf, vb.device())?;
        let (cos_sliding, sin_sliding) = create_rope_cache(
            head_dim,
            max_seq_len,
            sliding_theta,
            sliding_prf,
            vb.device(),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_full,
            sin_full,
            cos_sliding,
            sin_sliding,
        })
    }

    /// 前向推理：只返回**最后一个位置**的 logits（自回归解码只需下一步）
    ///
    /// # 参数
    /// - `input_ids`：形状 `[b, seq]` 的 Token id
    /// - `pos_offset`：本段在整段序列中的起始绝对位置（用于 RoPE 切片）
    /// - `kv_caches`：每层一份 KV 缓存，长度需等于层数（就地更新）
    ///
    /// # 返回
    /// 形状 `[b, vocab]` 的 logits
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pos_offset: usize,
        kv_caches: &mut [Option<(Tensor, Tensor)>],
    ) -> CandleResult<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;

        for (idx, layer) in self.layers.iter().enumerate() {
            let (cos_all, sin_all) = if layer.is_sliding {
                (&self.cos_sliding, &self.sin_sliding)
            } else {
                (&self.cos_full, &self.sin_full)
            };
            let cos = cos_all.narrow(0, pos_offset, seq_len)?;
            let sin = sin_all.narrow(0, pos_offset, seq_len)?;
            xs = layer.forward(&xs, &cos, &sin, &mut kv_caches[idx], pos_offset)?;
        }

        let xs = self.norm.forward(&xs)?;
        let last_token_logits = xs.i((.., seq_len - 1, ..))?;
        self.lm_head.forward(&last_token_logits)
    }
}

// ==========================================
// 8. 权重与配置文件准备
// ==========================================
/// 模型加载所需的全部本地文件路径（已确保存在于 HF 缓存中）
pub struct ModelFiles {
    /// `config.json`
    pub config: PathBuf,
    /// `tokenizer.json`
    pub tokenizer: PathBuf,
    /// Safetensors 分片（按文件名排序，已去重）
    pub weights: Vec<PathBuf>,
    /// `chat_template.jinja`
    pub chat_template: PathBuf,
}

/// 检查本地缓存并按需下载 Hugging Face 仓库中的配置与权重文件
///
/// 流程：设置镜像 → 建 API 客户端 → 取 config/tokenizer/模板 →
/// 有 `model.safetensors.index.json` 时按其清单取分片，
/// 否则按 `shard_count` 推断 `model-00001-of-000NN.safetensors`。
/// 已缓存的文件不会重复下载（由 `hf-hub` 保证）。
///
/// # 参数
/// - `repo_id`：仓库 ID，如 `XHToken/Spark-X2.5-1.7B`
/// - `shard_count`：缺少 index.json 时的分片数（对应 `--shards`）
///
/// # 返回
/// 各文件的本地路径；下载失败或 index.json 缺少 `weight_map` 时返回错误
pub fn prepare_model_files(repo_id: &str, shard_count: usize) -> AnyhowResult<ModelFiles> {
    println!("【系统提示】正在检查本地缓存与 HF 远程存储库：{}", repo_id);

    // 自动配置国内高速 HF 镜像源（海外环境可注释此行）
    std::env::set_var("HF_ENDPOINT", "https://hf-mirror.com");

    let api = ApiBuilder::new()
        .with_progress(true)
        .build()
        .context("初始化 Hugging Face API 客户端失败")?;
    let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

    println!("【系统提示】正在同步配置文件 config.json...");
    let config_path = repo.get("config.json").context("下载 config.json 失败")?;

    println!("【系统提示】正在同步分词器 tokenizer.json...");
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("下载 tokenizer.json 失败")?;

    println!("【系统提示】正在同步对话模板 chat_template.jinja...");
    let chat_template_path = repo
        .get("chat_template.jinja")
        .context("下载 chat_template.jinja 失败")?;

    // 优先用 index.json 精确推导分片列表，避免手工填错分片数
    let weight_paths = match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            println!("【系统提示】已发现分片索引，按其清单下载权重...");
            let index_str = std::fs::read_to_string(&index_path)
                .context("读取 model.safetensors.index.json 失败")?;
            let index: JsonValue = serde_json::from_str(&index_str)?;
            let weight_map = index
                .get("weight_map")
                .and_then(|v| v.as_object())
                .context("index.json 中缺少 weight_map 字段")?;
            let mut files: Vec<String> = weight_map
                .values()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            files.sort();
            files.dedup();

            let mut paths = Vec::new();
            for file_name in files {
                println!("【系统提示】正在同步权重分片文件 {}...", file_name);
                let path = repo
                    .get(&file_name)
                    .with_context(|| format!("下载权重分片 {} 失败", file_name))?;
                paths.push(path);
            }
            paths
        }
        Err(_) => {
            println!(
                "【系统提示】仓库未提供索引文件，按分片数 {} 推断文件名...",
                shard_count
            );
            let mut paths = Vec::new();
            for i in 1..=shard_count {
                let file_name = format!("model-{:05}-of-{:05}.safetensors", i, shard_count);
                println!("【系统提示】正在同步权重分片文件 {}...", file_name);
                let path = repo
                    .get(&file_name)
                    .with_context(|| format!("下载权重分片 {} 失败", file_name))?;
                paths.push(path);
            }
            paths
        }
    };

    println!("【系统提示】本地模型文件状态完好，全部校验成功！\n");
    Ok(ModelFiles {
        config: config_path,
        tokenizer: tokenizer_path,
        weights: weight_paths,
        chat_template: chat_template_path,
    })
}
