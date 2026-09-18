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

/// 一次生成任务的采样参数
///
/// 由命令行参数或 HTTP 请求字段组装而成，贯穿 `Engine` → `Generation` 全链路。
#[derive(Debug, Clone, Copy)]
pub struct GenOptions {
    /// 本次最多生成多少个 Token；引擎会按剩余上下文再夹一次
    pub max_tokens: usize,
    /// 采样温度；`<= 0` 时退化为贪婪解码（直接取 argmax）
    pub temperature: f64,
    /// 核采样（top-p）概率阈值，取值 (0, 1]
    pub top_p: f64,
    /// 随机种子；`Some` 时采样结果可复现，`None` 用系统熵
    pub seed: Option<u64>,
    /// 是否开启官方模板的思考模式（`enable_thinking`）
    pub enable_thinking: bool,
}

/// 未显式指定时的默认生成长度：给足上下文，避免长回答被拦腰截断
pub const DEFAULT_MAX_TOKEN: usize = 8192;

/// 默认采样参数：温度 0.7、top_p 0.95、不固定种子、关闭思考模式
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

/// 一次完整生成的结果（非流式接口直接把它序列化进响应）
#[derive(Debug, Clone)]
pub struct GenResult {
    /// 生成的完整文本
    pub text: String,
    /// 提示词占用的 Token 数
    pub prompt_tokens: usize,
    /// 实际生成的 Token 数
    pub completion_tokens: usize,
    /// 结束原因：`stop`（遇到 EOS）/ `length`（到达上限）/ `cancel`（被中断）
    pub finish_reason: &'static str,
    /// 整轮生成耗时（毫秒）
    pub elapsed_ms: u128,
}

/// 单条会话的 KV-Cache 状态（TUI / HTTP 共用一份，多轮对话靠前缀复用避免重复 prefill）
struct Session {
    /// 已缓存的完整 Token 序列；新请求以它为前缀时可只算增量部分
    tokens: Vec<u32>,
    /// 每层一份 `(K, V)` 缓存；`None` 表示该层尚未计算
    caches: Vec<Option<(Tensor, Tensor)>>,
}

/// 生成槽：同一时刻只允许一个生成任务占用共享的 KV-Cache 会话
///
/// HTTP 并发请求、TUI 取消后仍在收尾的旧任务都必须排队，
/// 否则多个任务会交错改写同一份会话，导致上下文错乱、反复重新 prefill。
struct GenSlot {
    /// 当前占用者的任务 id；`None` 表示空闲
    owner: Mutex<Option<u64>>,
    /// 用于唤醒排队任务的条件变量
    cvar: Condvar,
}

impl GenSlot {
    /// 抢占生成槽：已被占用时阻塞等待，直到前一个任务释放
    ///
    /// # 参数
    /// - `id`：本次任务的唯一 id（由 `Engine::next_id` 分配）
    fn acquire(&self, id: u64) {
        let mut owner = self.owner.lock().unwrap_or_else(|e| e.into_inner());
        while owner.is_some() {
            owner = self.cvar.wait(owner).unwrap_or_else(|e| e.into_inner());
        }
        *owner = Some(id);
    }

    /// 释放生成槽；只有持有者本人能释放，避免误放别人的锁
    ///
    /// # 参数
    /// - `id`：调用方的任务 id，与 `owner` 不符时直接忽略
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

/// 只读的模型资源，包在 `Arc` 里可在线程间安全共享
struct EngineInner {
    /// 模型本体（含 RoPE 缓存与全部解码层）
    model: SparkModel,
    /// 分词器，用于编码提示词与增量解码输出
    tokenizer: Tokenizer,
    /// 从 `config.json` 解析出的结构参数
    cfg: SparkConfig,
    /// 计算设备（CUDA 或 CPU）
    device: Device,
    /// 本地 `chat_template.jinja` 路径
    template: PathBuf,
    /// 全局系统提示词（`--system`），每次生成前自动前置
    system: Option<String>,
    /// 上下文上限（Prompt + 生成）
    max_context: usize,
    /// 模型短名（仓库 ID 的最后一段），用于 `/api/tags` 等接口
    model_id: String,
    /// 权重文件总大小（字节）
    size_bytes: u64,
    /// 权重文件最后修改时间（Unix 秒）
    modified_secs: u64,
    /// 实际使用的计算精度（如 `BF16`）
    dtype: String,
}

/// 诊断日志中单段文本保留的头部 / 尾部字符数
///
/// 超长对话每轮都会被截断，若不裁剪，一条日志就能写进几十 KB
const LOG_TEXT_HEAD_TAIL: usize = 1000;

/// 把 Token 序列解码回文本
///
/// # 参数
/// - `tokenizer`：模型分词器
/// - `ids`：待解码的 Token id 切片
///
/// # 返回
/// 解码后的文本；解码失败时返回 `<解码失败: …>`，不中断调用方
fn decode_ids(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    tokenizer
        .decode(ids, true)
        .unwrap_or_else(|e| format!("<解码失败: {e}>"))
}

/// 超长文本压缩成「头部 + 中间省略说明 + 尾部」，短文本原样返回
///
/// 提示词截断、HTTP 异常请求体等场景都会往日志里塞大段文本，
/// 不裁剪的话一条日志就能写进几十 KB。阈值由 [`LOG_TEXT_HEAD_TAIL`] 控制。
///
/// # 参数
/// - `text`：原始文本
///
/// # 返回
/// 长度不超过 `2 × LOG_TEXT_HEAD_TAIL + 省略说明` 的预览文本
pub(crate) fn preview_text(text: &str) -> String {
    let total = text.chars().count();
    if total <= LOG_TEXT_HEAD_TAIL * 2 {
        return text.to_string();
    }
    let omitted = total - LOG_TEXT_HEAD_TAIL * 2;
    let head: String = text.chars().take(LOG_TEXT_HEAD_TAIL).collect();
    let tail: String = text.chars().skip(total - LOG_TEXT_HEAD_TAIL).collect();
    format!("{head}\n…（中间省略 {omitted} 字符）…\n{tail}")
}

/// 把对话消息拼成便于阅读的原文，供诊断日志使用
///
/// 输出形如 `[system] 你是助手\n[user] 你好`，用于提示词被截断时
/// 还原客户端最初提交的原始输入（含 system 提示）。
///
/// # 参数
/// - `messages`：归一化后的消息列表
fn format_messages(messages: &[Message]) -> String {
    messages
        .iter()
        .map(|m| format!("[{}] {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 统计权重文件的总大小与最后修改时间（供 `/api/tags`、`/api/show` 展示）
///
/// 单个文件读取失败（缺失 / 权限不足）时静默跳过，只统计能读到的部分。
///
/// # 参数
/// - `paths`：Safetensors 分片路径列表
///
/// # 返回
/// `(总字节数, 最晚修改时间的 Unix 秒)`；全部读不到时为 `(0, 0)`
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

/// 推理引擎句柄：`Clone` 代价很低，内部全部是 `Arc`
///
/// TUI 与 HTTP 服务都持有一份，模型权重只加载一次。
#[derive(Clone)]
pub struct Engine {
    /// 只读的模型资源
    inner: Arc<EngineInner>,
    /// 全局唯一的 KV-Cache 会话，由生成槽串行保护
    session: Arc<Mutex<Session>>,
    /// 生成槽：保证同一时刻只有一个生成任务改写会话
    slot: Arc<GenSlot>,
    /// 任务 id 发号器
    next_id: Arc<AtomicU64>,
}

impl Engine {
    /// 下载（或复用缓存）模型文件并完成加载
    ///
    /// 全流程：准备文件 → 选择设备 → 解析 config → 读分词器 → 内存映射权重
    /// → 构建模型 → 统计元信息。每一步都会打 `target: "model"` / `"app"` 日志。
    ///
    /// # 参数
    /// - `repo_id`：Hugging Face 仓库 ID，如 `XHToken/Spark-X2.5-1.7B`
    /// - `shard_count`：仓库缺少 index.json 时按此数推断分片文件名
    /// - `dtype`：`None` 时自动选择（CUDA=BF16，CPU=F16）
    /// - `max_context`：上下文上限，决定 RoPE 缓存长度与预算裁剪
    /// - `system`：全局系统提示词，每次生成前自动前置
    ///
    /// # 返回
    /// 加载完成的 `Arc<Engine>`；任一环节失败（下载 / 解析 / 权重缺失）返回错误
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

    /// 模型短名（仓库 ID 的最后一段），用于接口回显与日志
    pub fn model_id(&self) -> &str {
        &self.inner.model_id
    }

    /// 当前生效的上下文上限（Prompt + 生成）
    pub fn max_context(&self) -> usize {
        self.inner.max_context
    }

    /// 当前加载模型的元信息（大小、精度、结构参数等）
    ///
    /// 供 Ollama 的 `/api/tags`、`/api/ps`、`/api/show` 及 `/api/health` 使用。
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
    ///
    /// TUI 的 `/clear`、`/reset` 与取消生成后都会调用：
    /// 被中断的生成会让缓存停在半截状态，必须整体丢弃才能继续正确复用前缀。
    pub fn reset(&self) {
        let mut session = self.session.lock().unwrap();
        let dropped = session.tokens.len();
        session.tokens.clear();
        session.caches.clear();
        debug!(target: "session", dropped_tokens = dropped, "已重置 KV-Cache");
    }

    /// 阻塞式完整生成，通过 `on_delta` 回调推送增量文本
    ///
    /// 内部就是「创建任务 → 循环 `step()` 直到返回 `None`」，
    /// 期间独占生成槽，返回时通过 `Drop` 自动释放。
    ///
    /// # 参数
    /// - `history`：对话历史（不含 system，引擎会自动前置 `--system`）
    /// - `opts`：采样参数
    /// - `cancel`：中断标志，置位后在下一个 Token 立即以 `finish_reason = "cancel"` 收尾
    /// - `on_delta`：增量文本回调（空串不会回调）
    ///
    /// # 返回
    /// `GenResult`（含完整文本、Token 统计、结束原因与耗时）
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

    /// 创建一个可逐步推进的生成任务（供 SSE / NDJSON 流式接口使用）
    ///
    /// 等价于 `start_generation_with_cancel(..., 不可取消的标志)`。
    ///
    /// # 参数
    /// - `history`：对话历史
    /// - `opts`：采样参数
    ///
    /// # 返回
    /// 已抢占生成槽的 `Generation`；丢弃它即释放槽位
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
        // 先分配 id 再排队：保证释放时用的是同一个 id
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

    /// 生成任务的真正构造过程：渲染模板 → 编码 → 裁剪预算 → 建随机数发生器
    ///
    /// 仅在已持有生成槽（`acquire` 成功）后调用；失败时由调用方负责 `release`。
    ///
    /// # 参数
    /// - `id`：已分配的任务 id
    /// - `history`：对话历史（不含 system）
    /// - `opts`：采样参数（内部会按剩余上下文夹一次 `max_tokens`）
    /// - `cancel`：中断标志
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
        // 否则前向推理时 RoPE 缓存会被越界切片，直接导致推理失败。
        // 提示词优先：只有提示词自身超限才丢弃最早的 Token，剩余空间全部留给生成
        let max_ctx = inner.max_context.max(2);
        let prompt_budget = max_ctx - 1; // 至少留一个位置给生成
        if ids.len() > prompt_budget {
            let cut = ids.len() - prompt_budget;
            // 先解码再裁剪：日志要同时留下被丢弃的部分与真正送进模型的部分
            let dropped = decode_ids(&inner.tokenizer, &ids[..cut]);
            let retained = decode_ids(&inner.tokenizer, &ids[cut..]);
            let original = format_messages(&messages);
            ids.drain(..cut);
            warn!(
                target: "gen",
                truncated = cut,
                budget = prompt_budget,
                dropped_chars = dropped.chars().count(),
                retained_chars = retained.chars().count(),
                original_input = %preview_text(&original),
                dropped_text = %preview_text(&dropped),
                retained_text = %preview_text(&retained),
                "提示词超出上下文预算，已丢弃最早的若干 Token"
            );
        }
        let prompt_len = ids.len();
        let allowed = max_ctx - prompt_len;
        let max_tokens = opts.max_tokens.max(1).min(allowed);
        if max_tokens < opts.max_tokens {
            // 客户端常常直接塞一个远超上下文的 max_tokens，属常规情况，不打扰日志
            debug!(
                target: "gen",
                requested = opts.max_tokens,
                allowed,
                prompt_tokens = prompt_len,
                context = max_ctx,
                "max_tokens 超出剩余上下文预算，已夹取"
            );
        }
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
    /// 只读的模型资源
    inner: Arc<EngineInner>,
    /// 共享的 KV-Cache 会话
    session: Arc<Mutex<Session>>,
    /// 生成槽，任务结束时释放，让排队的任务接手
    slot: Arc<GenSlot>,
    /// 本任务的 id，用于释放槽位时校验身份
    id: u64,
    /// 中断标志
    cancel: Arc<AtomicBool>,
    /// 提示词 Token + 已生成 Token 的完整序列
    input_ids: Vec<u32>,
    /// 提示词长度（`input_ids` 的前缀长度）
    prompt_len: usize,
    /// 已生成的 Token 数
    produced: usize,
    /// 仅已生成部分的 Token，用于增量解码
    new_ids: Vec<u32>,
    /// 已吐出的完整文本，用于计算增量后缀
    emitted: String,
    /// 采样用的随机数发生器（可由 `seed` 复现）
    rng: StdRng,
    /// 是否已结束
    finished: bool,
    /// 结束原因：`stop` / `length` / `cancel`
    finish_reason: &'static str,
    /// 实际生效的采样参数（`max_tokens` 已被夹过）
    opts: GenOptions,
}

impl Generation {
    /// 提示词占用的 Token 数
    #[allow(dead_code)]
    pub fn prompt_tokens(&self) -> usize {
        self.prompt_len
    }

    /// 已生成的 Token 数
    pub fn completion_tokens(&self) -> usize {
        self.produced
    }

    /// 结束原因：`stop` / `length` / `cancel`
    pub fn finish_reason(&self) -> &'static str {
        self.finish_reason
    }

    /// 已生成的完整文本
    #[allow(dead_code)]
    pub fn text(&self) -> &str {
        &self.emitted
    }

    /// 标记生成结束（length / stop / cancel）并统一记录输出
    ///
    /// # 参数
    /// - `reason`：`"stop"`（遇到 EOS）/ `"length"`（到上限）/ `"cancel"`（被中断）
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

    /// 推进一步：产出一个 Token 并返回增量文本
    ///
    /// 这是流式接口的核心：OpenAI SSE、Anthropic SSE、Ollama NDJSON 都在循环里调它。
    /// 内部处理了取消、长度上限、上下文兜底、KV-Cache 复用判断、分块前向与采样。
    ///
    /// # 返回
    /// - `Ok(Some(delta))`：刚产出的增量文本；**可能为空串**（尚未凑齐一个完整字符），
    ///   调用方应继续循环而不是当作结束
    /// - `Ok(None)`：已结束，`finish_reason()` 可取原因
    /// - `Err`：前向推理或采样失败
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

/// 任务被丢弃时释放生成槽
impl Drop for Generation {
    fn drop(&mut self) {
        // 释放生成槽，让排队的任务（HTTP 并发请求 / TUI 新提交）接手
        self.slot.release(self.id);
    }
}

/// 分块前向：单次前向的 Token 数不超过上下文上限，避免 RoPE 缓存越界
///
/// 长提示词的 prefill 会被切成多段逐段推进，每段起点位置 `pos` 随之累加，
/// 只保留最后一段的 logits（下一步采样只需要最后一个位置）。
///
/// # 参数
/// - `inner`：只读模型资源
/// - `caches`：各层的 KV-Cache（就地更新）
/// - `pos`：本批 Token 的起始绝对位置
/// - `feed`：待计算的 Token 片段
///
/// # 返回
/// 最后一段的 logits（形状 `[1, vocab]`）；上下文耗尽或前向失败时返回错误
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

/// 从 logits 采样下一个 Token
///
/// - `temperature <= 0`：贪婪解码，直接取 argmax（结果确定）
/// - 否则：先按温度缩放 + softmax，再做 top-p 核采样（概率按截断后的总和重新归一）
///
/// # 参数
/// - `logits`：`lm_head` 的输出，形状 `[1, vocab]` 或 `[vocab]`
/// - `temperature`：温度，越大分布越平
/// - `top_p`：核采样阈值，会被夹到 [0, 1]
/// - `rng`：随机数发生器（`seed` 固定时可复现）
///
/// # 返回
/// 采样得到的 Token id
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
