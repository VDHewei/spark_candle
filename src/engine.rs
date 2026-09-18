//! 推理引擎：模型加载、KV-Cache 会话管理、流式生成与采样

use crate::chat::{apply_chat_template, build_prompt_fallback, Message};
use crate::model::{prepare_model_files, SparkConfig, SparkModel};
use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::UNIX_EPOCH;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Copy)]
pub struct GenOptions {
    pub max_tokens: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    pub enable_thinking: bool,
}
pub const DEFAULT_MAX_TOKEN: usize = 8192; // 512
impl Default for GenOptions {
    fn default() -> Self {
        Self {
            max_tokens: DEFAULT_MAX_TOKEN,
            temperature: 0.7,
            top_p: 0.95,
            seed: None,
            enable_thinking: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct GenResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub finish_reason: &'static str,
    pub elapsed_ms: u128,
}

/// 单条会话的 KV-Cache 状态（TUI / HTTP 共用一份，多轮对话靠前缀复用避免重复 prefill）
struct Session {
    tokens: Vec<u32>,
    caches: Vec<Option<(Tensor, Tensor)>>,
}

/// 生成槽：同一时刻只允许一个生成任务占用共享的 KV-Cache 会话
///
/// HTTP 并发请求、TUI 取消后仍在收尾的旧任务都必须排队，
/// 否则多个任务会交错改写同一份会话，导致上下文错乱、反复重新 prefill。
struct GenSlot {
    owner: Mutex<Option<u64>>,
    cvar: Condvar,
}

impl GenSlot {
    fn acquire(&self, id: u64) {
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        while owner.is_some() {
            owner = self.cvar.wait(owner).unwrap_or_else(|e| e.into_inner());
        }
        *owner = Some(id);
    }

    fn release(&self, id: u64) {
        if let Ok(mut owner) = self.owner.lock() {
            if *owner == Some(id) {
                *owner = None;
                self.cvar.notify_all();
            }
        }
    }
}

/// 模型元信息：供 HTTP 接口描述当前加载的模型（Ollama /api/tags、/api/show 等）
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    /// 权重文件总大小（字节）
    pub size_bytes: u64,
    /// 权重文件最后修改时间（Unix 秒）
    pub modified_secs: u64,
    /// 计算精度，如 F16 / BF16 / F32
    pub dtype: String,
    pub max_context: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
}

/// 只读的模型资源，可在线程间安全共享
struct EngineInner {
    model: SparkModel,
    tokenizer: Tokenizer,
    cfg: SparkConfig,
    device: Device,
    template: PathBuf,
    system: Option<String>,
    max_context: usize,
    model_id: String,
    size_bytes: u64,
    modified_secs: u64,
    dtype: String,
}

/// 统计权重文件的总大小与最后修改时间
fn weight_stats(paths: &[PathBuf]) -> (u64, u64) {
    let mut size = 0u64;
    let mut mtime = 0u64;
    for p in paths {
        let Ok(meta) = std::fs::metadata(p) else { continue };
        size += meta.len();
        if let Ok(modified) = meta.modified() {
            if let Ok(d) = modified.duration_since(UNIX_EPOCH) {
                mtime = mtime.max(d.as_secs());
            }
        }
    }
    (size, mtime)
}

#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
    session: Arc<Mutex<Session>>,
    slot: Arc<GenSlot>,
    next_id: Arc<AtomicU64>,
}

impl Engine {
    pub fn load(
        repo_id: &str,
        shard_count: usize,
        dtype: Option<DType>,
        max_context: usize,
        system: Option<String>,
    ) -> Result<Arc<Engine>> {
        let started = Instant::now();
        info!(target: "app", repo = %repo_id, "开始加载模型");
        let files = prepare_model_files(repo_id, shard_count)?;

        let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
        println!("【系统提示】正在使用的硬件后端: {:?}", device);
        info!(target: "model", ?device, "硬件后端就绪");

        let config_str = std::fs::read_to_string(&files.config)?;
        let cfg: SparkConfig = serde_json::from_str(&config_str)?;
        info!(
            target: "model",
            layers = cfg.num_hidden_layers,
            heads = cfg.num_attention_heads,
            kv_heads = cfg.num_key_value_heads,
            head_dim = cfg.head_dim(),
            vocab = cfg.vocab_size,
            sliding_window = ?cfg.sliding_window,
            "模型配置解析完成"
        );
        println!(
            "【系统提示】模型配置: {} 层 / {} 注意力头 / {} KV 头 / head_dim {} / 词表 {} / 滑动窗口 {:?}",
            cfg.num_hidden_layers,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
            cfg.vocab_size,
            cfg.sliding_window
        );

        let tokenizer =
            Tokenizer::from_file(&files.tokenizer).map_err(|e| anyhow::anyhow!("{e}"))?;

        // 权重原生为 bfloat16：CUDA 用 BF16，CPU 用 F16（省一半内存且更快）
        let dtype = dtype.unwrap_or(if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        });
        println!(
            "【系统提示】正在从内存映射读取 Safetensors 权重（{:?}）...",
            dtype
        );
        info!(target: "model", ?dtype, shards = files.weights.len(), "开始读取 Safetensors 权重");
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&files.weights, dtype, &device)?
        };

        println!("【系统提示】正在初始化 Spark 模型结构并绑定权重...");
        let model = SparkModel::load(vb, &cfg, max_context)?;

        let (size_bytes, modified_secs) = weight_stats(&files.weights);
        let model_id = repo_id.rsplit('/').next().unwrap_or(repo_id).to_string();
        let engine = Engine {
            inner: Arc::new(EngineInner {
                model,
                tokenizer,
                cfg,
                device,
                template: files.chat_template,
                system,
                max_context,
                model_id,
                size_bytes,
                modified_secs,
                dtype: format!("{:?}", dtype),
            }),
            session: Arc::new(Mutex::new(Session {
                tokens: Vec::new(),
                caches: Vec::new(),
            })),
            slot: Arc::new(GenSlot {
                owner: Mutex::new(None),
                cvar: Condvar::new(),
            }),
            next_id: Arc::new(AtomicU64::new(0)),
        };
        println!(
            "【系统提示】模型加载完成，上下文上限 {} Token。",
            max_context
        );
        info!(
            target: "app",
            model_id = %engine.model_id(),
            max_context,
            elapsed_ms = started.elapsed().as_millis(),
            "模型加载完成"
        );
        Ok(Arc::new(engine))
    }

    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    pub fn max_context(&self) -> usize {
        self.inner.max_context
    }

    /// 当前加载模型的元信息（大小、精度、结构参数等）
    pub fn model_info(&self) -> ModelInfo {
        let cfg = &self.inner.cfg;
        ModelInfo {
            id: self.inner.model_id.clone(),
            size_bytes: self.inner.size_bytes,
            modified_secs: self.inner.modified_secs,
            dtype: self.inner.dtype.clone(),
            max_context: self.inner.max_context,
            layers: cfg.num_hidden_layers,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim(),
            vocab: cfg.vocab_size,
        }
    }

    /// 清空会话（丢弃 KV-Cache）
    pub fn reset(&self) {
        let mut session = self.session.lock().unwrap();
        let dropped = session.tokens.len();
        session.tokens.clear();
        session.caches.clear();
        debug!(target: "session", dropped_tokens = dropped, "已重置 KV-Cache");
    }

    /// 阻塞式生成，通过 on_delta 回调推送增量文本；cancel 置位后会尽快结束
    pub fn generate_blocking(
        &self,
        history: &[Message],
        opts: GenOptions,
        cancel: Arc<AtomicBool>,
        mut on_delta: impl FnMut(&str),
    ) -> Result<GenResult> {
        let start = Instant::now();
        let mut gen = self.start_generation_with_cancel(history, opts, cancel)?;
        loop {
            match gen.step()? {
                Some(delta) => {
                    if !delta.is_empty() {
                        on_delta(&delta)
                    }
                }
                None => break,
            }
        }
        let result = GenResult {
            text: gen.emitted.clone(),
            prompt_tokens: gen.prompt_len,
            completion_tokens: gen.produced,
            finish_reason: gen.finish_reason,
            elapsed_ms: start.elapsed().as_millis(),
        };
        info!(
            target: "gen",
            prompt_tokens = result.prompt_tokens,
            completion_tokens = result.completion_tokens,
            finish_reason = result.finish_reason,
            elapsed_ms = result.elapsed_ms,
            "生成统计"
        );
        Ok(result)
    }

    /// 创建一个可逐步推进的生成任务（供 SSE 流式接口使用）
    pub fn start_generation(&self, history: &[Message], opts: GenOptions) -> Result<Generation> {
        self.start_generation_with_cancel(history, opts, Arc::new(AtomicBool::new(false)))
    }

    /// 同 start_generation，但可通过 cancel 提前中断（TUI 的 Esc / Ctrl+C）
    ///
    /// 会阻塞等待生成槽空出：调用方需保证在 block_in_place 或独立线程中执行
    pub fn start_generation_with_cancel(
        &self,
        history: &[Message],
        opts: GenOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<Generation> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed) + 1;
        self.slot.acquire(id);
        match self.build_generation(id, history, opts, cancel) {
            Ok(gen) => Ok(gen),
            Err(e) => {
                self.slot.release(id);
                Err(e)
            }
        }
    }

    fn build_generation(
        &self,
        id: u64,
        history: &[Message],
        opts: GenOptions,
        cancel: Arc<AtomicBool>,
    ) -> Result<Generation> {
        let inner = self.inner.clone();
        let mut messages: Vec<Message> = Vec::with_capacity(history.len() + 1);
        if let Some(sys) = &inner.system {
            messages.push(Message {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }
        messages.extend_from_slice(history);

        let prompt =
            match apply_chat_template(&inner.template, &messages, true, opts.enable_thinking) {
                Ok(p) => p,
                Err(e) => {
                    warn!(target: "gen", error = %e, "Jinja 渲染失败，改用内置对话格式兜底");
                    build_prompt_fallback(&messages)
                }
            };
        debug!(target: "gen", prompt = %prompt, "渲染后的提示词");

        let mut ids = inner
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .get_ids()
            .to_vec();

        // 生成预算受上下文上限约束：prompt + max_tokens 必须能放进 max_context，
        // 否则前向推理时 RoPE 缓存会被越界切片，直接导致推理失败
        let max_ctx = inner.max_context.max(2);
        let mut max_tokens = opts.max_tokens.max(1).min(max_ctx - 1);
        // 提示词至少保留一半上下文，避免过大的 max_tokens 把提示词整个挤掉
        if ids.len() > max_ctx.saturating_sub(max_tokens) {
            max_tokens = max_tokens.min(max_ctx.saturating_sub(max_ctx / 2).max(1));
            warn!(
                target: "gen",
                requested = opts.max_tokens,
                allowed = max_tokens,
                "max_tokens 超出上下文预算，已夹取"
            );
        }
        let budget = max_ctx.saturating_sub(max_tokens).max(1);
        if ids.len() > budget {
            let cut = ids.len() - budget;
            ids.drain(..cut);
            warn!(
                target: "gen",
                truncated = cut,
                budget,
                "提示词超出上下文预算，已丢弃最早的若干 Token"
            );
        }
        let prompt_len = ids.len();
        max_tokens = max_tokens.min(max_ctx.saturating_sub(prompt_len).max(1));
        let opts = GenOptions { max_tokens, ..opts };
        info!(
            target: "gen",
            prompt_tokens = prompt_len,
            max_tokens = opts.max_tokens,
            temperature = opts.temperature,
            top_p = opts.top_p,
            seed = ?opts.seed,
            enable_thinking = opts.enable_thinking,
            "开始生成"
        );

        let rng = match opts.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        Ok(Generation {
            inner,
            session: self.session.clone(),
            slot: self.slot.clone(),
            id,
            cancel,
            input_ids: ids,
            prompt_len,
            produced: 0,
            new_ids: Vec::new(),
            emitted: String::new(),
            rng,
            finished: false,
            finish_reason: "stop",
            opts,
        })
    }
}

/// 一次生成任务：每步产出一个 Token 并返回增量文本，返回 None 表示结束
pub struct Generation {
    inner: Arc<EngineInner>,
    session: Arc<Mutex<Session>>,
    /// 生成槽，任务结束时释放，让排队的任务接手
    slot: Arc<GenSlot>,
    id: u64,
    cancel: Arc<AtomicBool>,
    input_ids: Vec<u32>,
    prompt_len: usize,
    produced: usize,
    new_ids: Vec<u32>,
    emitted: String,
    rng: StdRng,
    finished: bool,
    finish_reason: &'static str,
    opts: GenOptions,
}

impl Generation {
    #[allow(dead_code)]
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }
    pub fn completion_tokens(&self) -> usize {
        self.produced
    }
    pub fn finish_reason(&self) -> &'static str {
        self.finish_reason
    }
    #[allow(dead_code)]
    pub fn text(&self) -> &str {
        &self.emitted
    }

    /// 标记生成结束（length / stop / cancel）并统一记录输出
    fn finish(&mut self, reason: &'static str) {
        self.finished = true;
        self.finish_reason = reason;
        info!(
            target: "gen",
            reason,
            completion_tokens = self.produced,
            output = %self.emitted,
            "生成结束"
        );
    }

    /// 推进一步：Ok(Some(delta)) 表示有进展（delta 可能为空串），Ok(None) 表示已结束
    pub fn step(&mut self) -> Result<Option<String>> {
        if self.finished {
            return Ok(None);
        }
        if self.cancel.load(Ordering::Relaxed) {
            warn!(
                target: "gen",
                completion_tokens = self.produced,
                "生成被取消，提前结束"
            );
            self.finish("cancel");
            return Ok(None);
        }
        if self.produced >= self.opts.max_tokens {
            self.finish("length");
            return Ok(None);
        }
        // 上下文兜底：再生成一个 Token 就会超出 RoPE 缓存上限时直接收尾
        if self.input_ids.len() >= self.inner.max_context {
            warn!(
                target: "gen",
                limit = self.inner.max_context,
                "上下文已达上限，提前结束生成"
            );
            self.finish("length");
            return Ok(None);
        }

        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        let layers = self.inner.cfg.num_hidden_layers;
        // 会话缓存不可用（层数不符 / 被其它任务改写 / 已追平当前上下文）就整段重新 prefill
        // 注意用 >= ：session 与 input_ids 等长会让待计算的片段为空，前向会得到空序列
        if session.caches.len() != layers
            || session.tokens.len() >= self.input_ids.len()
            || !self.input_ids.starts_with(&session.tokens)
        {
            if !session.tokens.is_empty() {
                debug!(
                    target: "session",
                    cached = session.tokens.len(),
                    needed = self.input_ids.len(),
                    "会话缓存不可复用，重新 prefill"
                );
            }
            session.tokens.clear();
            session.caches = vec![None; layers];
        }

        let pos_offset = session.tokens.len();
        let feed: Vec<u32> = self.input_ids[pos_offset..].to_vec();
        let logits = {
            let mut caches = std::mem::take(&mut session.caches);
            let out = forward_chunked(&self.inner, &mut caches, pos_offset, &feed);
            session.caches = caches;
            out?
        };
        session.tokens.extend_from_slice(&feed);
        drop(session);

        let next = sample_token(
            &logits,
            self.opts.temperature,
            self.opts.top_p,
            &mut self.rng,
        )?;

        let cfg = &self.inner.cfg;
        if next == cfg.eos_token_id || Some(next) == cfg.pad_token_id {
            debug!(target: "gen", stop_token_id = next, "遇到结束符");
            self.finish("stop");
            return Ok(None);
        }

        self.input_ids.push(next);
        self.new_ids.push(next);
        self.produced += 1;

        // 增量解码：整体解码后取新增后缀，避免半个 UTF-8 字符被提前吐出
        let full = self
            .inner
            .tokenizer
            .decode(&self.new_ids, true)
            .unwrap_or_default();
        let delta = if full.len() > self.emitted.len() && full.starts_with(&self.emitted) {
            let d = full[self.emitted.len()..].to_string();
            self.emitted = full;
            d
        } else {
            String::new()
        };
        Ok(Some(delta))
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        // 释放生成槽，让排队的任务（HTTP 并发请求 / TUI 新提交）接手
        self.slot.release(self.id);
    }
}

/// 分块前向：单次前向的 Token 数不超过上下文上限，避免 RoPE 缓存越界
fn forward_chunked(
    inner: &EngineInner,
    caches: &mut Vec<Option<(Tensor, Tensor)>>,
    pos: usize,
    feed: &[u32],
) -> Result<Tensor> {
    let mut pos = pos;
    let mut logits: Option<Tensor> = None;
    let mut rest = feed;
    while !rest.is_empty() {
        let room = inner.max_context.saturating_sub(pos);
        if room == 0 {
            anyhow::bail!(
                "上下文已耗尽：已缓存 {} 个 Token，仍有 {} 个 Token 待处理",
                pos,
                rest.len()
            );
        }
        let n = rest.len().min(room);
        let input = Tensor::new(&rest[..n], &inner.device)?.unsqueeze(0)?;
        logits = Some(inner.model.forward(&input, pos, caches).with_context(|| {
            format!("模型前向推理失败: pos={pos} chunk={n} ctx={}", inner.max_context)
        })?);
        pos += n;
        rest = &rest[n..];
    }
    logits.context("待计算的 Token 片段为空")
}

/// temperature <= 0 走贪婪解码；否则按 top_p 做核采样
fn sample_token(logits: &Tensor, temperature: f64, top_p: f64, rng: &mut StdRng) -> Result<u32> {
    let logits = logits.to_dtype(DType::F32)?;
    // lm_head 输出为 [1, vocab]，统一降成 [vocab]
    let logits = if logits.rank() > 1 { logits.i(0)? } else { logits };
    if temperature <= 0.0 {
        return Ok(logits.argmax(D::Minus1)?.to_scalar::<u32>()?);
    }
    let logits = (&logits / temperature)?;
    let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
    let probs: Vec<f32> = probs.to_vec1()?;

    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));

    let top_p = top_p.clamp(0.0, 1.0) as f32;
    let mut cumulative = 0.0f32;
    let mut cutoff = order.len().min(1);
    for (rank, &idx) in order.iter().enumerate() {
        cumulative += probs[idx];
        cutoff = rank + 1;
        if cumulative >= top_p {
            break;
        }
    }
    let candidates = &order[..cutoff];
    let total: f32 = candidates.iter().map(|&i| probs[i]).sum();
    if total <= 0.0 {
        return Ok(candidates[0] as u32);
    }
    let mut target: f32 = rng.gen_range(0.0..total);
    for &idx in candidates {
        target -= probs[idx];
        if target <= 0.0 {
            return Ok(idx as u32);
        }
    }
    Ok(candidates[candidates.len() - 1] as u32)
}
