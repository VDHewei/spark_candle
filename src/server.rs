//! 兼容 OpenAI Chat Completions 与 Anthropic Messages 的 HTTP 服务

use crate::chat::Message;
use crate::cli::ServeArgs;
use crate::engine::{preview_text, Engine, GenOptions, Generation};
use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::convert::Infallible;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, error, info, warn};

/// 所有 HTTP handler 共享的状态
#[derive(Clone)]
struct AppState {
    /// 推理引擎（内部全为 `Arc`，克隆代价很低）
    engine: Arc<Engine>,
    /// 命令行给出的默认采样参数；请求未覆盖的字段用它兜底
    default: GenOptions,
}

// ==========================================
// 请求 / 响应数据结构
// ==========================================
/// 消息内容：OpenAI 允许纯字符串，Anthropic 允许分块数组，这里用 untagged 兼容两者
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ContentValue {
    /// 纯文本内容（OpenAI 风格）
    Text(String),
    /// 分块内容（Anthropic 风格，如 `[{"type":"text","text":"..."}]`）
    Parts(Vec<ContentPart>),
}

/// Anthropic 风格的一个内容块
#[derive(Debug, Deserialize)]
struct ContentPart {
    /// 文本块内容；非文本块（如图片）为 `None`
    #[serde(default)]
    text: Option<String>,
}

impl ContentValue {
    /// 折叠成纯文本：分块形式取所有 `text` 并用换行连接，非文本块忽略
    fn to_text(&self) -> String {
        match self {
            ContentValue::Text(s) => s.clone(),
            ContentValue::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.text.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

/// 两种协议共用的入参消息结构（`role` 缺省为 user）
#[derive(Debug, Deserialize)]
struct ApiMessage {
    /// 角色：`system` / `user` / `assistant`
    #[serde(default = "default_role")]
    role: String,
    /// 消息内容；缺失视为空串
    #[serde(default)]
    content: Option<ContentValue>,
}

/// `role` 字段缺失时的默认值
fn default_role() -> String {
    "user".to_string()
}

/// OpenAI `/v1/chat/completions` 的请求体
#[derive(Debug, Deserialize)]
struct OaiChatRequest {
    /// 模型名；`None` 时用当前加载的模型
    #[serde(default)]
    model: Option<String>,
    /// 对话消息列表
    #[serde(default)]
    messages: Vec<ApiMessage>,
    /// 是否以 SSE 流式返回
    #[serde(default)]
    stream: bool,
    /// 最大生成 Token 数；`None` 用命令行默认值
    #[serde(default)]
    max_tokens: Option<usize>,
    /// 采样温度；`None` 用命令行默认值
    #[serde(default)]
    temperature: Option<f64>,
    /// 核采样阈值；`None` 用命令行默认值
    #[serde(default)]
    top_p: Option<f64>,
}

/// Anthropic `/v1/messages` 的请求体
///
/// 与 OpenAI 的差异：`system` 是独立的顶层字段，而不是 messages 里的一条。
#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    /// 模型名；`None` 时用当前加载的模型
    #[serde(default)]
    model: Option<String>,
    /// 最大生成 Token 数（Anthropic 协议里是必填，这里容错处理）
    #[serde(default)]
    max_tokens: Option<usize>,
    /// 顶层 system 提示，可以是字符串或分块数组
    #[serde(default)]
    system: Option<ContentValue>,
    /// 对话消息列表
    #[serde(default)]
    messages: Vec<ApiMessage>,
    /// 是否以 SSE 流式返回
    #[serde(default)]
    stream: bool,
    /// 采样温度；`None` 用命令行默认值
    #[serde(default)]
    temperature: Option<f64>,
    /// 核采样阈值；`None` 用命令行默认值
    #[serde(default)]
    top_p: Option<f64>,
}

/// Token 用量统计（OpenAI 风格）
#[derive(Debug, Serialize)]
struct Usage {
    /// 提示词 Token 数
    prompt_tokens: usize,
    /// 生成 Token 数
    completion_tokens: usize,
    /// 两者之和
    total_tokens: usize,
}

/// 非流式响应中的一条消息
#[derive(Debug, Serialize)]
struct OaiRespMessage {
    /// 固定为 `assistant`
    role: String,
    /// 完整回复文本
    content: String,
}

/// OpenAI 响应的一个 choice（本服务只返回 1 个）
#[derive(Debug, Serialize)]
struct OaiChoice {
    /// 固定为 0
    index: usize,
    /// 回复消息
    message: OaiRespMessage,
    /// `stop` / `length`
    finish_reason: String,
}

/// OpenAI `/v1/chat/completions` 的非流式响应
#[derive(Debug, Serialize)]
struct OaiChatResponse {
    /// `chatcmpl-<随机十六进制>`
    id: String,
    /// 固定为 `chat.completion`
    object: String,
    /// 创建时间（Unix 秒）
    created: u64,
    /// 实际使用的模型名
    model: String,
    /// 候选回复列表
    choices: Vec<OaiChoice>,
    /// Token 用量
    usage: Usage,
}

/// SSE 增量：只含本次新增的字段，未变化的字段省略
#[derive(Debug, Serialize)]
struct OaiDelta {
    /// 首帧带 `assistant`，后续帧省略
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    /// 新增文本；结束帧省略
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

/// SSE 帧中的一个 choice
#[derive(Debug, Serialize)]
struct OaiChunkChoice {
    /// 固定为 0
    index: usize,
    /// 增量内容
    delta: OaiDelta,
    /// 仅结束帧非 `None`
    finish_reason: Option<String>,
}

/// OpenAI 流式响应的一帧（`chat.completion.chunk`）
#[derive(Debug, Serialize)]
struct OaiChunk {
    /// 与本次请求同 id
    id: String,
    /// 固定为 `chat.completion.chunk`
    object: String,
    /// 创建时间（Unix 秒）
    created: u64,
    /// 实际使用的模型名
    model: String,
    /// 候选增量列表
    choices: Vec<OaiChunkChoice>,
}

/// Anthropic 响应中的一个文本块
#[derive(Debug, Serialize)]
struct AnthropicTextBlock {
    /// 固定为 `text`
    #[serde(rename = "type")]
    kind: String,
    /// 回复文本
    text: String,
}

/// Anthropic `/v1/messages` 的非流式响应
#[derive(Debug, Serialize)]
struct AnthropicResponse {
    /// `msg_<随机十六进制>`
    id: String,
    /// 固定为 `message`
    #[serde(rename = "type")]
    kind: String,
    /// 固定为 `assistant`
    role: String,
    /// 实际使用的模型名
    model: String,
    /// 内容块列表
    content: Vec<AnthropicTextBlock>,
    /// `end_turn` / `max_tokens`
    stop_reason: String,
    /// 命中停止序列时才有值，本服务恒为 `None`
    stop_sequence: Option<String>,
    /// Token 用量
    usage: AnthropicUsage,
}

/// Anthropic 风格的 Token 用量
#[derive(Debug, Serialize)]
struct AnthropicUsage {
    /// 输入 Token 数
    input_tokens: usize,
    /// 输出 Token 数
    output_tokens: usize,
}

/// `/v1/models` 中的单张模型卡片
#[derive(Debug, Serialize)]
struct ModelCard {
    /// 模型 ID
    id: String,
    /// 固定为 `model`
    object: String,
    /// 创建时间（Unix 秒）
    created: u64,
    /// 固定为 `spark-candle`
    owned_by: String,
}

/// `/v1/models` 的响应
#[derive(Debug, Serialize)]
struct ModelsResponse {
    /// 固定为 `list`
    object: String,
    /// 模型卡片列表（本服务只有一张）
    data: Vec<ModelCard>,
}

/// OpenAI 风格的错误响应外层
#[derive(Debug, Serialize)]
struct ErrorBody {
    /// 错误详情
    error: ErrorDetail,
}

/// OpenAI 风格的错误详情
#[derive(Debug, Serialize)]
struct ErrorDetail {
    /// 人类可读的错误信息
    message: String,
    /// 错误类型，如 `invalid_request_error`
    #[serde(rename = "type")]
    kind: String,
}

// ==========================================
// 工具函数
// ==========================================
/// 当前 Unix 时间戳（秒）；系统时钟异常时返回 0
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// 生成 16 位十六进制随机后缀，用于拼 `chatcmpl-…` / `msg_…` 这类 id
fn random_suffix() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// 把 Unix 时间戳转成 RFC3339（UTC）字符串，如 `2026-09-18T07:33:26.123456789Z`
///
/// 自己实现是为了避免额外引入时间库：days → 年月日走的是
/// Howard Hinnant 的 `civil_from_days` 算法。
///
/// # 参数
/// - `secs`：Unix 秒
/// - `nanos`：秒内的纳秒部分（不足 9 位时左侧补 0）
fn rfc3339(secs: u64, nanos: u32) -> String {
    // days → 年月日：Howard Hinnant 的 civil_from_days 算法，避免引入时间库
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as i64;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        y,
        m,
        d,
        sod / 3600,
        sod % 3600 / 60,
        sod % 60,
        nanos
    )
}

/// 当前时刻的 RFC3339 字符串（Ollama 的 `created_at` 字段用）
fn now_rfc3339() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(d.as_secs(), d.subsec_nanos())
}

/// 毫秒转纳秒（Ollama 的 `*_duration` 字段单位为纳秒）
///
/// # 参数
/// - `ms`：毫秒
fn as_nanos(ms: u128) -> u128 {
    ms.saturating_mul(1_000_000)
}

/// 生成 OpenAI 风格的错误响应
///
/// # 参数
/// - `status`：HTTP 状态码
/// - `message`：错误信息（会放进 `error.message`）
fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            kind: "invalid_request_error".to_string(),
        },
    };
    (status, Json(body)).into_response()
}

/// 生成 SSE 流里的错误事件（用于流式中途推理失败）
///
/// 错误类型为 `server_error`，因为响应头已经发出，无法再改状态码。
///
/// # 参数
/// - `message`：错误信息
fn error_event(message: impl Into<String>) -> Result<Event, Infallible> {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            kind: "server_error".to_string(),
        },
    };
    Ok(Event::default().data(serde_json::to_string(&body).unwrap_or_default()))
}

/// 把 API 入参消息归一化成引擎的 `Message`
///
/// 过滤规则：只保留 `user` / `assistant` / `system` 三种角色，且丢弃空白内容。
///
/// # 参数
/// - `messages`：API 入参消息列表
fn to_messages(messages: &[ApiMessage]) -> Vec<Message> {
    messages
        .iter()
        .filter(|m| matches!(m.role.as_str(), "user" | "assistant" | "system"))
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.as_ref().map(|c| c.to_text()).unwrap_or_default(),
        })
        .filter(|m| !m.content.trim().is_empty())
        .collect()
}

/// 组装请求消息：可选的 system 提示前置 + 过滤后的 messages
///
/// # 参数
/// - `system`：顶层 system 字段（Anthropic / Ollama 有），`None` 时跳过
/// - `messages`：API 入参消息列表
fn build_messages(system: Option<&str>, messages: &[ApiMessage]) -> Vec<Message> {
    let mut out = Vec::new();
    if let Some(text) = system.map(str::trim) {
        if !text.is_empty() {
            out.push(Message {
                role: "system".to_string(),
                content: text.to_string(),
            });
        }
    }
    out.extend(to_messages(messages));
    out
}

/// 组装并校验请求消息；为空时记 WARN 并直接从当前函数返回 400 响应
///
/// # 参数
/// - `$api`：日志里的接口名（如 `"anthropic"`）
/// - `$system`：顶层 system 文本
/// - `$messages`：API 入参消息列表
macro_rules! messages_or_400 {
    ($api:expr, $system:expr, $messages:expr) => {{
        let msgs = build_messages($system, $messages);
        if msgs.is_empty() {
            warn!(target: "api", api = $api, "请求被拒绝：messages 为空");
            return error_response(StatusCode::BAD_REQUEST, "messages 不能为空");
        }
        msgs
    }};
}

/// 启动流式生成（在 `block_in_place` 中等待生成槽）；失败时记 ERROR 并返回 500
///
/// # 参数
/// - `$api`：日志里的接口名
/// - `$state`：`AppState`
/// - `$messages`：归一化后的消息
/// - `$opts`：采样参数
macro_rules! generation_or_500 {
    ($api:expr, $state:expr, $messages:expr, $opts:expr) => {{
        match tokio::task::block_in_place(|| $state.engine.start_generation($messages, $opts)) {
            Ok(gen) => gen,
            Err(e) => {
                error!(target: "api", api = $api, error = %e, "推理失败");
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}"));
            }
        }
    }};
}

/// 非流式路径的推理失败：记 ERROR 日志并生成 500 响应
///
/// # 参数
/// - `api`：日志里的接口名
/// - `e`：底层错误
fn inference_error(api: &'static str, e: impl std::fmt::Display) -> Response {
    error!(target: "api", api, error = %e, "推理失败");
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("推理失败: {e}"),
    )
}

/// 记录一次 API 调用的输入
///
/// 完整内容以 `debug` 级别记录（`target: "api"`，字段 `input`），
/// `info` 只记条数与参数，避免生产环境日志被长对话撑爆。
///
/// # 参数
/// - `api`：接口名（如 `openai` / `anthropic` / `ollama.chat`）
/// - `model`：实际使用的模型名
/// - `messages`：归一化后的消息
/// - `opts`：本次生效的采样参数
/// - `stream`：是否流式
fn log_input(api: &'static str, model: &str, messages: &[Message], opts: &GenOptions, stream: bool) {
    let brief = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join(" || ");
    info!(
        target: "api",
        api,
        model = %model,
        messages = messages.len(),
        stream,
        max_tokens = opts.max_tokens,
        temperature = opts.temperature,
        top_p = opts.top_p,
        "收到请求"
    );
    debug!(target: "api", api, input = %brief, "请求内容");
}

/// 记录一次 API 调用的输出统计（`target: "api"` 的 info 日志）
///
/// # 参数
/// - `api`：接口名
/// - `res`：生成结果
fn log_output(api: &'static str, res: &crate::engine::GenResult) {
    info!(
        target: "api",
        api,
        prompt_tokens = res.prompt_tokens,
        completion_tokens = res.completion_tokens,
        finish_reason = res.finish_reason,
        elapsed_ms = res.elapsed_ms,
        "回复完成"
    );
}

/// 把引擎内部结束原因映射成各家协议约定的字面量
///
/// # 参数
/// - `reason`：引擎的 `finish_reason`（`stop` / `length` / `cancel`）
/// - `length`：协议里表示「到达上限」的字面量
/// - `stop`：协议里表示「正常结束」的字面量
///
/// # 返回
/// `reason == "length"` 时返回 `length`，否则（含 `cancel`）返回 `stop`
fn map_finish_reason(reason: &str, length: &'static str, stop: &'static str) -> &'static str {
    match reason {
        "length" => length,
        _ => stop,
    }
}

/// 引擎结束原因 → Anthropic 的 `stop_reason`（`max_tokens` / `end_turn`）
fn anthropic_stop_reason(reason: &str) -> &'static str {
    map_finish_reason(reason, "max_tokens", "end_turn")
}

impl AppState {
    /// 用请求字段覆盖默认采样参数，未提供的字段沿用命令行默认值
    ///
    /// # 参数
    /// - `max_tokens`：OpenAI / Anthropic 的 `max_tokens`，`None` 用默认值
    /// - `temperature`：采样温度，`None` 用默认值
    /// - `top_p`：核采样阈值，`None` 用默认值
    fn gen_options(
        &self,
        max_tokens: Option<usize>,
        temperature: Option<f64>,
        top_p: Option<f64>,
    ) -> GenOptions {
        GenOptions {
            max_tokens: max_tokens.unwrap_or(self.default.max_tokens),
            temperature: temperature.unwrap_or(self.default.temperature),
            top_p: top_p.unwrap_or(self.default.top_p),
            seed: self.default.seed,
            enable_thinking: self.default.enable_thinking,
        }
    }

    /// 决定响应里回显的模型名：请求未指定时用当前加载的模型
    ///
    /// # 参数
    /// - `requested`：请求里的 `model` 字段
    fn model_name(&self, requested: Option<String>) -> String {
        requested.unwrap_or_else(|| self.engine.model_id().to_string())
    }

    /// 阻塞式完整生成
    ///
    /// 推理是同步的 CPU/GPU 密集任务，用 `block_in_place` 包住，
    /// 告诉 Tokio 当前线程会阻塞，避免把异步工作线程占死。
    ///
    /// # 参数
    /// - `messages`：归一化后的消息
    /// - `opts`：采样参数
    ///
    /// # 返回
    /// 生成结果；抢占生成槽失败或前向出错时返回错误
    fn generate_blocking(
        &self,
        messages: &[Message],
        opts: GenOptions,
    ) -> anyhow::Result<crate::engine::GenResult> {
        tokio::task::block_in_place(|| {
            self.engine
                .generate_blocking(messages, opts, Arc::new(AtomicBool::new(false)), |_| {})
        })
    }
}

// ==========================================
// OpenAI 兼容接口
// ==========================================
/// OpenAI `/api/v1/chat/completions`：非流式返回 JSON，流式返回 SSE
///
/// 流程：归一化 messages → 组装采样参数 → 记输入日志 →
/// 非流式走 `generate_blocking`，流式则创建 `Generation` 并用
/// `stream::unfold` 按 `Role → Running… → Tail → Done` 的阶段推进。
///
/// # 参数
/// - `state`：共享状态（引擎 + 默认采样参数）
/// - `req`：OpenAI 风格请求体
///
/// # 返回
/// 200 JSON / SSE 流；messages 为空 400，推理失败 500
async fn openai_chat(State(state): State<AppState>, Json(req): Json<OaiChatRequest>) -> Response {
    let messages = to_messages(&req.messages);
    if messages.is_empty() {
        warn!(target: "api", api = "openai", "请求被拒绝：messages 为空");
        return error_response(StatusCode::BAD_REQUEST, "messages 不能为空");
    }
    let opts = state.gen_options(req.max_tokens, req.temperature, req.top_p);
    let model = state.model_name(req.model.clone());
    log_input("openai", &model, &messages, &opts, req.stream);
    let id = format!("chatcmpl-{}", random_suffix());
    let created = now_secs();

    if !req.stream {
        let result = state.generate_blocking(&messages, opts);
        return match result {
            Ok(res) => {
                log_output("openai", &res);
                Json(OaiChatResponse {
                    id,
                    object: "chat.completion".to_string(),
                    created,
                    model,
                    choices: vec![OaiChoice {
                        index: 0,
                        message: OaiRespMessage {
                            role: "assistant".to_string(),
                            content: res.text,
                        },
                        finish_reason: res.finish_reason.to_string(),
                    }],
                    usage: Usage {
                        prompt_tokens: res.prompt_tokens,
                        completion_tokens: res.completion_tokens,
                        total_tokens: res.prompt_tokens + res.completion_tokens,
                    },
                })
                .into_response()
            }
            Err(e) => inference_error("openai", e),
        };
    }

    let gen = generation_or_500!("openai", state, &messages, opts);

    /// SSE 推进阶段：首帧角色 → 逐帧增量 → 结束帧 → `[DONE]` → 收尾
    enum Phase {
        /// 首帧：只带 `role: assistant`
        Role(Generation),
        /// 持续产出增量文本
        Running(Generation),
        /// 生成结束，输出 `finish_reason`
        Tail(&'static str),
        /// 发送 `[DONE]`
        Done,
        /// 流结束
        End,
    }

    let sse = stream::unfold(Phase::Role(gen), move |phase| {
        let id = id.clone();
        let model = model.clone();
        async move {
            match phase {
                Phase::Role(gen) => {
                    let chunk = OaiChunk {
                        id,
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model,
                        choices: vec![OaiChunkChoice {
                            index: 0,
                            delta: OaiDelta {
                                role: Some("assistant".to_string()),
                                content: Some(String::new()),
                            },
                            finish_reason: None,
                        }],
                    };
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                        ),
                        Phase::Running(gen),
                    ))
                }
                Phase::Running(mut gen) => {
                    let step = tokio::task::block_in_place(|| gen.step());
                    match step {
                        Ok(Some(delta)) => {
                            if delta.is_empty() {
                                Some((
                                    Ok::<Event, Infallible>(Event::default().comment("keep-alive")),
                                    Phase::Running(gen),
                                ))
                            } else {
                                let chunk = OaiChunk {
                                    id,
                                    object: "chat.completion.chunk".to_string(),
                                    created,
                                    model,
                                    choices: vec![OaiChunkChoice {
                                        index: 0,
                                        delta: OaiDelta {
                                            role: None,
                                            content: Some(delta),
                                        },
                                        finish_reason: None,
                                    }],
                                };
                                Some((
                                    Ok::<Event, Infallible>(
                                        Event::default()
                                            .data(serde_json::to_string(&chunk).unwrap_or_default()),
                                    ),
                                    Phase::Running(gen),
                                ))
                            }
                        }
                        Ok(None) => {
                            info!(
                                target: "api",
                                api = "openai",
                                completion_tokens = gen.completion_tokens(),
                                finish_reason = gen.finish_reason(),
                                "流式回复完成"
                            );
                            Some((
                                Ok::<Event, Infallible>(Event::default().comment("done")),
                                Phase::Tail(gen.finish_reason()),
                            ))
                        }
                        Err(e) => {
                            error!(target: "api", api = "openai", error = %e, "推理失败");
                            Some((error_event(format!("推理失败: {e}")), Phase::End))
                        }
                    }
                }
                Phase::Tail(reason) => {
                    let chunk = OaiChunk {
                        id,
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model,
                        choices: vec![OaiChunkChoice {
                            index: 0,
                            delta: OaiDelta {
                                role: None,
                                content: None,
                            },
                            finish_reason: Some(reason.to_string()),
                        }],
                    };
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()),
                        ),
                        Phase::Done,
                    ))
                }
                Phase::Done => Some((
                    Ok::<Event, Infallible>(Event::default().data("[DONE]")),
                    Phase::End,
                )),
                Phase::End => None,
            }
        }
    });

    Sse::new(sse).into_response()
}

// ==========================================
// Anthropic 兼容接口
// ==========================================
/// Anthropic `/api/v1/messages`：非流式返回 JSON，流式返回 SSE
///
/// 流式阶段为 `message_start → content_block_start → content_block_delta…
/// → content_block_stop → message_delta → message_stop`。
///
/// # 参数
/// - `state`：共享状态
/// - `req`：Anthropic 风格请求体（`system` 为独立顶层字段）
///
/// # 返回
/// 200 JSON / SSE 流；messages 为空 400，推理失败 500
async fn anthropic_messages(
    State(state): State<AppState>,
    Json(req): Json<AnthropicRequest>,
) -> Response {
    let system = req.system.as_ref().map(|c| c.to_text());
    let messages = messages_or_400!("anthropic", system.as_deref(), &req.messages);

    let opts = state.gen_options(req.max_tokens, req.temperature, req.top_p);
    let model = state.model_name(req.model.clone());
    log_input("anthropic", &model, &messages, &opts, req.stream);
    let id = format!("msg_{}", random_suffix());

    if !req.stream {
        let result = state.generate_blocking(&messages, opts);
        return match result {
            Ok(res) => {
                log_output("anthropic", &res);
                Json(AnthropicResponse {
                id,
                kind: "message".to_string(),
                role: "assistant".to_string(),
                model,
                content: vec![AnthropicTextBlock {
                    kind: "text".to_string(),
                    text: res.text,
                }],
                stop_reason: anthropic_stop_reason(res.finish_reason).to_string(),
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: res.prompt_tokens,
                    output_tokens: res.completion_tokens,
                },
            })
                .into_response()
            }
            Err(e) => inference_error("anthropic", e),
        };
    }

    let gen = generation_or_500!("anthropic", state, &messages, opts);

    /// SSE 推进阶段，严格对应 Anthropic 的事件顺序
    enum Phase {
        /// `message_start`
        MessageStart(Generation),
        /// `content_block_start`
        BlockStart(Generation),
        /// 持续 `content_block_delta`
        Running(Generation),
        /// `content_block_stop`，携带（结束原因, 已生成 Token 数）
        BlockStop(&'static str, usize),
        /// `message_delta`（含 stop_reason 与用量）
        MessageDelta,
        /// `message_stop`
        MessageStop,
        /// 流结束
        End,
    }

    let sse = stream::unfold(Phase::MessageStart(gen), move |phase| {
        let id = id.clone();
        let model = model.clone();
        async move {
            match phase {
                Phase::MessageStart(gen) => {
                    let payload = serde_json::json!({
                        "type": "message_start",
                        "message": {
                            "id": id,
                            "type": "message",
                            "role": "assistant",
                            "model": model,
                            "content": [],
                            "stop_reason": null,
                            "stop_sequence": null,
                            "usage": {"input_tokens": 0, "output_tokens": 0}
                        }
                    });
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("message_start")
                                .data(payload.to_string()),
                        ),
                        Phase::BlockStart(gen),
                    ))
                }
                Phase::BlockStart(gen) => {
                    let payload = serde_json::json!({
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""}
                    });
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("content_block_start")
                                .data(payload.to_string()),
                        ),
                        Phase::Running(gen),
                    ))
                }
                Phase::Running(mut gen) => {
                    let step = tokio::task::block_in_place(|| gen.step());
                    match step {
                        Ok(Some(delta)) => {
                            if delta.is_empty() {
                                Some((
                                    Ok::<Event, Infallible>(
                                        Event::default().comment("keep-alive"),
                                    ),
                                    Phase::Running(gen),
                                ))
                            } else {
                                let payload = serde_json::json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": delta}
                                });
                                Some((
                                    Ok::<Event, Infallible>(
                                        Event::default()
                                            .event("content_block_delta")
                                            .data(payload.to_string()),
                                    ),
                                    Phase::Running(gen),
                                ))
                            }
                        }
                        Ok(None) => {
                            info!(
                                target: "api",
                                api = "anthropic",
                                completion_tokens = gen.completion_tokens(),
                                finish_reason = gen.finish_reason(),
                                "流式回复完成"
                            );
                            let payload =
                                serde_json::json!({"type": "content_block_stop", "index": 0});
                            Some((
                                Ok::<Event, Infallible>(
                                    Event::default()
                                        .event("content_block_stop")
                                        .data(payload.to_string()),
                                ),
                                Phase::BlockStop(gen.finish_reason(), gen.completion_tokens()),
                            ))
                        }
                        Err(e) => {
                            error!(target: "api", api = "anthropic", error = %e, "推理失败");
                            Some((error_event(format!("推理失败: {e}")), Phase::End))
                        }
                    }
                }
                Phase::BlockStop(reason, completion) => {
                    let payload = serde_json::json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": anthropic_stop_reason(reason), "stop_sequence": null},
                        "usage": {"output_tokens": completion}
                    });
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default()
                                .event("message_delta")
                                .data(payload.to_string()),
                        ),
                        Phase::MessageDelta,
                    ))
                }
                Phase::MessageDelta => {
                    let payload = serde_json::json!({"type": "message_stop"});
                    Some((
                        Ok::<Event, Infallible>(
                            Event::default().event("message_stop").data(payload.to_string()),
                        ),
                        Phase::MessageStop,
                    ))
                }
                Phase::MessageStop => Some((
                    Ok::<Event, Infallible>(Event::default().comment("end")),
                    Phase::End,
                )),
                Phase::End => None,
            }
        }
    });

    Sse::new(sse).into_response()
}

// ==========================================
// Ollama 兼容接口（/api/tags、/api/chat、/api/generate …）
// ==========================================
/// 对外报告的兼容版本号，客户端仅用于可用性判断
const OLLAMA_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Ollama 请求里的 `options` 字段
#[derive(Debug, Deserialize)]
struct OllamaOptions {
    /// 采样温度
    #[serde(default)]
    temperature: Option<f64>,
    /// 核采样阈值
    #[serde(default)]
    top_p: Option<f64>,
    /// Ollama 用 num_predict 表示最大生成 Token 数
    #[serde(default)]
    num_predict: Option<usize>,
    /// 随机种子；`None` 时沿用命令行默认
    #[serde(default)]
    seed: Option<u64>,
}

impl OllamaOptions {
    /// 把 Ollama 的 options 覆盖到基础采样参数上
    ///
    /// # 参数
    /// - `base`：命令行默认参数
    ///
    /// # 返回
    /// 覆盖后的新参数（`enable_thinking` 始终沿用默认值）
    fn apply(&self, base: GenOptions) -> GenOptions {
        GenOptions {
            max_tokens: self.num_predict.unwrap_or(base.max_tokens),
            temperature: self.temperature.unwrap_or(base.temperature),
            top_p: self.top_p.unwrap_or(base.top_p),
            seed: self.seed.or(base.seed),
            enable_thinking: base.enable_thinking,
        }
    }
}

/// Ollama `/api/chat` 的请求体
#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    /// 模型名；`None` 时用当前加载的模型
    #[serde(default)]
    model: Option<String>,
    /// 对话消息列表
    #[serde(default)]
    messages: Vec<ApiMessage>,
    /// Ollama 缺省即为流式输出
    #[serde(default)]
    stream: Option<bool>,
    /// system 提示（字符串或分块数组）
    #[serde(default)]
    system: Option<ContentValue>,
    /// 采样参数覆盖项
    #[serde(default)]
    options: Option<OllamaOptions>,
}

/// Ollama `/api/generate` 的请求体（单轮补全）
#[derive(Debug, Deserialize)]
struct OllamaGenerateRequest {
    /// 模型名；`None` 时用当前加载的模型
    #[serde(default)]
    model: Option<String>,
    /// 补全提示词（必填，空白会被拒）
    #[serde(default)]
    prompt: String,
    /// system 提示（此处为纯字符串）
    #[serde(default)]
    system: Option<String>,
    /// Ollama 缺省即为流式输出
    #[serde(default)]
    stream: Option<bool>,
    /// 采样参数覆盖项
    #[serde(default)]
    options: Option<OllamaOptions>,
}

/// Ollama `/api/show` 的请求体（只需要 model 名）
#[derive(Debug, Deserialize)]
struct OllamaModelRequest {
    /// 模型名；本服务忽略它，始终返回当前加载的模型
    #[serde(default)]
    model: Option<String>,
}

/// 由模型信息派生出稳定的伪 digest（客户端只做展示与比对，不参与校验）
fn ollama_digest(info: &crate::engine::ModelInfo) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    info.id.hash(&mut h);
    info.size_bytes.hash(&mut h);
    info.modified_secs.hash(&mut h);
    let v = h.finish();
    format!(
        "sha256:{:016x}{:016x}{:016x}{:016x}",
        v,
        v.rotate_left(17),
        v.rotate_left(41),
        v ^ 0x9E37_79B9_7F4A_7C15
    )
}

/// /api/tags、/api/ps 中描述模型的 details 字段
fn ollama_details(info: &crate::engine::ModelInfo) -> serde_json::Value {
    json!({
        "parent_model": "",
        "format": "safetensors",
        "family": "spark",
        "families": ["spark"],
        "parameter_size": "",
        "quantization_level": info.dtype,
    })
}

/// /api/show 中的 model_info 字段
fn ollama_model_info(info: &crate::engine::ModelInfo) -> serde_json::Value {
    json!({
        "general.architecture": "spark",
        "general.file_type": info.dtype,
        "spark.block_count": info.layers,
        "spark.attention.head_count": info.heads,
        "spark.attention.head_count_kv": info.kv_heads,
        "spark.embedding_length": info.head_dim * info.heads,
        "spark.context_length": info.max_context,
        "spark.vocab_size": info.vocab,
    })
}

/// 引擎结束原因 → Ollama 的 `done_reason`（`length` / `stop`）
fn ollama_done_reason(reason: &str) -> &'static str {
    map_finish_reason(reason, "length", "stop")
}

/// Ollama 结束帧的统计字段（Token 计数与耗时），chat / generate 共用
fn ollama_done_stats(gen: &Generation, elapsed: u128) -> serde_json::Value {
    json!({
        "done_reason": ollama_done_reason(gen.finish_reason()),
        "prompt_eval_count": gen.prompt_tokens(),
        "eval_count": gen.completion_tokens(),
        "total_duration": elapsed,
        "load_duration": 0,
        "prompt_eval_duration": 0,
        "eval_duration": elapsed,
    })
}

/// 把生成过程包装成 Ollama 风格的 NDJSON 流（一行一个 JSON 对象）
fn ollama_stream(
    model: String,
    gen: Generation,
    delta_payload: fn(&str) -> serde_json::Value,
    done_payload: fn(&Generation, u128) -> serde_json::Value,
) -> Body {
    // Generation 体积较大，装箱后枚举各分支大小才均衡
    enum Phase {
        Running(Box<Generation>),
        End,
    }

    let started = Instant::now();
    let s = stream::unfold(Phase::Running(Box::new(gen)), move |phase| {
        let model = model.clone();
        async move {
            let mut gen = match phase {
                Phase::Running(g) => g,
                Phase::End => return None,
            };
            loop {
                match tokio::task::block_in_place(|| gen.step()) {
                    // 空增量（未凑齐一个完整字符）不推流
                    Ok(Some(text)) if text.is_empty() => continue,
                    Ok(Some(text)) => {
                        let mut obj = json!({"model": model, "created_at": now_rfc3339(), "done": false});
                        merge(&mut obj, delta_payload(&text));
                        return Some((ndjson_line(&obj), Phase::Running(gen)));
                    }
                    Ok(None) => {
                        info!(
                            target: "api",
                            api = "ollama",
                            completion_tokens = gen.completion_tokens(),
                            finish_reason = gen.finish_reason(),
                            "流式回复完成"
                        );
                        let mut obj = json!({"model": model, "created_at": now_rfc3339(), "done": true});
                        merge(&mut obj, done_payload(&gen, as_nanos(started.elapsed().as_millis())));
                        return Some((ndjson_line(&obj), Phase::End));
                    }
                    Err(e) => {
                        error!(target: "api", api = "ollama", error = %e, "推理失败");
                        let obj = json!({
                            "model": model,
                            "created_at": now_rfc3339(),
                            "error": format!("推理失败: {e}"),
                            "done": true,
                        });
                        return Some((ndjson_line(&obj), Phase::End));
                    }
                }
            }
        }
    });

    Body::from_stream(s)
}

/// 把 `extra` 的顶层字段合并进 `base`（同名覆盖）
///
/// 只在两边都是对象时生效，否则静默忽略——用于给统一的
/// `{model, created_at, done}` 骨架拼上各接口独有的字段。
fn merge(base: &mut serde_json::Value, extra: serde_json::Value) {
    if let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            base.insert(k.clone(), v.clone());
        }
    }
}

/// 把一个 JSON 对象序列化成一行 NDJSON（末尾带 `\n`）
///
/// # 参数
/// - `obj`：待输出的对象
fn ndjson_line(obj: &serde_json::Value) -> Result<Bytes, Infallible> {
    let mut line = serde_json::to_string(obj).unwrap_or_default();
    line.push('\n');
    Ok(Bytes::from(line))
}

/// 给流式响应套上 `application/x-ndjson` 响应头
fn ndjson_response(body: Body) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/x-ndjson"),
        )],
        body,
    )
        .into_response()
}

/// Ollama 客户端用根路径探测服务是否在线
async fn ollama_root() -> &'static str {
    "Ollama is running"
}

/// Ollama `/api/version`：返回兼容版本号（取自 Cargo 包版本）
async fn ollama_version() -> Json<serde_json::Value> {
    Json(json!({ "version": OLLAMA_VERSION }))
}

/// Ollama `/api/tags`：列出已加载的模型（含大小、digest、details）
async fn ollama_tags(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.engine.model_info();
    let name = format!("{}:latest", info.id);
    info!(target: "api", api = "ollama.tags", model = %name, "查询模型列表");
    Json(json!({
        "models": [{
            "name": name,
            "model": name,
            "modified_at": rfc3339(info.modified_secs, 0),
            "size": info.size_bytes,
            "digest": ollama_digest(&info),
            "details": ollama_details(&info),
        }]
    }))
}

/// Ollama `/api/ps`：模拟「正在运行的模型」列表，过期时间写死为 5 分钟后
async fn ollama_ps(State(state): State<AppState>) -> Json<serde_json::Value> {
    let info = state.engine.model_info();
    let name = format!("{}:latest", info.id);
    Json(json!({
        "models": [{
            "name": name,
            "model": name,
            "size": info.size_bytes,
            "digest": ollama_digest(&info),
            "expires_at": rfc3339(now_secs() + 300, 0),
            "size_vram": info.size_bytes,
            "details": ollama_details(&info),
        }]
    }))
}

/// Ollama `/api/show`：返回模型详情（`details` / `model_info` / `capabilities`）
///
/// 忽略请求里的 `model`（本服务只加载了一个模型），始终返回当前模型的信息。
async fn ollama_show(
    State(state): State<AppState>,
    Json(req): Json<OllamaModelRequest>,
) -> Response {
    let info = state.engine.model_info();
    info!(target: "api", api = "ollama.show", model = %state.model_name(req.model), "查询模型信息");
    Json(json!({
        "license": "",
        "modelfile": "",
        "parameters": format!("num_ctx {}", info.max_context),
        "template": "{{ .Prompt }}",
        "system": "",
        "details": ollama_details(&info),
        "model_info": ollama_model_info(&info),
        "capabilities": ["completion", "chat"],
    }))
    .into_response()
}

/// Ollama `/api/embeddings`：当前模型不支持，固定返回 501
async fn ollama_embeddings() -> Response {
    error_response(
        StatusCode::NOT_IMPLEMENTED,
        "当前模型不支持 embeddings 接口",
    )
}

/// Ollama /api/chat：与 OpenAI 接口共用同一套引擎与生成槽
async fn ollama_chat(State(state): State<AppState>, Json(req): Json<OllamaChatRequest>) -> Response {
    let system = req.system.as_ref().map(|c| c.to_text());
    let messages = messages_or_400!("ollama.chat", system.as_deref(), &req.messages);

    let stream = req.stream.unwrap_or(true);
    let opts = match &req.options {
        Some(o) => o.apply(state.default),
        None => state.default,
    };
    let model = state.model_name(req.model.clone());
    log_input("ollama.chat", &model, &messages, &opts, stream);

    if !stream {
        let result = state.generate_blocking(&messages, opts);
        return match result {
            Ok(res) => {
                log_output("ollama.chat", &res);
                let elapsed = as_nanos(res.elapsed_ms);
                Json(json!({
                    "model": model,
                    "created_at": now_rfc3339(),
                    "message": {"role": "assistant", "content": res.text},
                    "done": true,
                    "done_reason": ollama_done_reason(res.finish_reason),
                    "prompt_eval_count": res.prompt_tokens,
                    "eval_count": res.completion_tokens,
                    "total_duration": elapsed,
                    "load_duration": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": elapsed,
                }))
                .into_response()
            }
            Err(e) => inference_error("ollama.chat", e),
        };
    }

    let gen = generation_or_500!("ollama.chat", state, &messages, opts);

    fn delta(text: &str) -> serde_json::Value {
        json!({ "message": {"role": "assistant", "content": text} })
    }
    fn done(gen: &Generation, elapsed: u128) -> serde_json::Value {
        let mut obj = json!({ "message": {"role": "assistant", "content": ""} });
        merge(&mut obj, ollama_done_stats(gen, elapsed));
        obj
    }

    ndjson_response(ollama_stream(model, gen, delta, done))
}

/// Ollama /api/generate：把 prompt + system 组装成单轮对话后走同一条生成链路
async fn ollama_generate(
    State(state): State<AppState>,
    Json(req): Json<OllamaGenerateRequest>,
) -> Response {
    let prompt = req.prompt.trim().to_string();
    if prompt.is_empty() {
        warn!(target: "api", api = "ollama.generate", "请求被拒绝：prompt 为空");
        return error_response(StatusCode::BAD_REQUEST, "prompt 不能为空");
    }
    let mut messages = build_messages(req.system.as_deref(), &[]);
    messages.push(Message {
        role: "user".to_string(),
        content: prompt,
    });

    let stream = req.stream.unwrap_or(true);
    let opts = match &req.options {
        Some(o) => o.apply(state.default),
        None => state.default,
    };
    let model = state.model_name(req.model.clone());
    log_input("ollama.generate", &model, &messages, &opts, stream);

    if !stream {
        let result = state.generate_blocking(&messages, opts);
        return match result {
            Ok(res) => {
                log_output("ollama.generate", &res);
                let elapsed = as_nanos(res.elapsed_ms);
                Json(json!({
                    "model": model,
                    "created_at": now_rfc3339(),
                    "response": res.text,
                    "done": true,
                    "done_reason": ollama_done_reason(res.finish_reason),
                    "context": [],
                    "prompt_eval_count": res.prompt_tokens,
                    "eval_count": res.completion_tokens,
                    "total_duration": elapsed,
                    "load_duration": 0,
                    "prompt_eval_duration": 0,
                    "eval_duration": elapsed,
                }))
                .into_response()
            }
            Err(e) => inference_error("ollama.generate", e),
        };
    }

    let gen = generation_or_500!("ollama.generate", state, &messages, opts);

    fn delta(text: &str) -> serde_json::Value {
        json!({ "response": text })
    }
    fn done(gen: &Generation, elapsed: u128) -> serde_json::Value {
        let mut obj = json!({ "response": "", "context": [] });
        merge(&mut obj, ollama_done_stats(gen, elapsed));
        obj
    }

    ndjson_response(ollama_stream(model, gen, delta, done))
}

// ==========================================
// 其它端点与启动
// ==========================================
/// `/api/v1/models`：OpenAI 风格的模型列表（只有当前加载的这一个）
async fn models(State(state): State<AppState>) -> Json<ModelsResponse> {
    Json(ModelsResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.engine.model_id().to_string(),
            object: "model".to_string(),
            created: now_secs(),
            owned_by: "spark-candle".to_string(),
        }],
    })
}

/// `/api/health`：健康检查，返回状态、模型名与上下文上限
async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.engine.model_id(),
        "max_context": state.engine.max_context(),
    }))
}

/// 记录请求体时允许读取的最大字节数（与 axum 的 Json 默认上限一致）
const LOG_BODY_LIMIT: usize = 2 * 1024 * 1024;

/// 记录每次 HTTP 访问的方法、URI、状态码与耗时
///
/// 请求体只能被消费一次：先读进内存再原样回填给下游 handler，
/// 请求失败时把 URI 与请求体一并落日志，便于还原客户端到底发了什么
async fn trace_request(req: Request<Body>, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let started = Instant::now();

    // 声明了超大体量的请求直接跳过读取，交给 handler 按自己的上限拒绝
    let oversized = req
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .is_some_and(|len| len > LOG_BODY_LIMIT);

    let (req, body_text) = if method == Method::GET || method == Method::HEAD || oversized {
        let note = oversized.then(|| format!("<请求体超过 {LOG_BODY_LIMIT} 字节，未记录>"));
        (req, note)
    } else {
        let (parts, body) = req.into_parts();
        match axum::body::to_bytes(body, LOG_BODY_LIMIT).await {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                (Request::from_parts(parts, Body::from(bytes)), Some(text))
            }
            Err(_) => {
                warn!(
                    target: "http",
                    %method, %uri, limit = LOG_BODY_LIMIT,
                    "请求体超过上限，已拒绝"
                );
                return error_response(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    format!("请求体超过 {LOG_BODY_LIMIT} 字节上限"),
                );
            }
        }
    };

    let res = next.run(req).await;
    let elapsed_ms = started.elapsed().as_millis();
    let status = res.status().as_u16();
    // 失败时才带上请求体：成功路径的入参已由 api 目标的日志覆盖
    let body = body_text.as_deref().unwrap_or("");
    if res.status().is_success() {
        info!(target: "http", %method, %uri, status, elapsed_ms, "请求完成");
    } else {
        warn!(
            target: "http",
            %method, %uri, status, elapsed_ms,
            body = %preview_text(body),
            "请求异常"
        );
    }
    res
}

/// 简单的 CORS 中间件 + OPTIONS 预检处理
///
/// 放行全部来源与方法，并加 `x-accel-buffering: no`
/// 防止 Nginx 之类反向代理缓冲 SSE 流。
async fn cors(req: Request<Body>, next: Next) -> Response {
    let mut res = if req.method() == Method::OPTIONS {
        let mut res = Response::new(Body::empty());
        *res.status_mut() = StatusCode::NO_CONTENT;
        res
    } else {
        next.run(req).await
    };
    let headers = res.headers_mut();
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        header::HeaderValue::from_static("*"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        header::HeaderValue::from_static("GET, POST, OPTIONS"),
    );
    headers.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        header::HeaderValue::from_static("*"),
    );
    headers.insert(
        header::HeaderName::from_static("x-accel-buffering"),
        header::HeaderValue::from_static("no"),
    );
    res
}

/// 等待 Ctrl+C（HTTPS 与 HTTP 两种服务都用它做优雅退出）
async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    println!("\n【系统提示】收到退出信号，服务正在关闭…");
}

/// 安装 rustls 默认加密后端；进程内只需一次，已安装则跳过
fn install_tls_provider() {
    if rustls::crypto::CryptoProvider::get_default().is_none() {
        let _ = rustls::crypto::ring::default_provider().install_default();
    }
}

/// 打印服务启动横幅（列出各协议端点）
///
/// # 参数
/// - `addr`：`host:port`
/// - `scheme`：`http` 或 `https`
fn print_banner(addr: &str, scheme: &str) {
    println!("\n【系统提示】{} 服务已启动: {}://{}", scheme.to_uppercase(), scheme, addr);
    println!("  · OpenAI     POST {scheme}://{addr}/api/v1/chat/completions");
    println!("  · Anthropic  POST {scheme}://{addr}/api/v1/messages");
    println!("  · 模型列表   GET  {scheme}://{addr}/api/v1/models");
    println!("  · 健康检查   GET  {scheme}://{addr}/api/health");
    println!("  · Ollama     GET  {scheme}://{addr}/api/tags");
    println!("               POST {scheme}://{addr}/api/chat");
    println!("               POST {scheme}://{addr}/api/generate\n");
}

/// HTTP 服务入口：建路由 → 挂中间件 → 按是否配置 TLS 选择 HTTPS / HTTP
///
/// 中间件自下而上：`cors` 在外层（先处理 OPTIONS），`trace_request` 在内层。
/// 指定了 `--tls-cert` / `--tls-key` 时用 `axum-server` + rustls 提供 HTTPS，
/// 收到 Ctrl+C 后有 10 秒优雅收尾时间。
///
/// # 参数
/// - `engine`：已加载的引擎
/// - `args`：`serve` 子命令参数（地址、端口、TLS、默认采样参数）
///
/// # 返回
/// 绑定端口失败、TLS 证书加载失败或服务异常退出时返回错误
pub async fn run(engine: Arc<Engine>, args: ServeArgs) -> anyhow::Result<()> {
    let state = AppState {
        engine,
        default: args.common.gen_options(None),
    };

    let app = Router::new()
        .route("/api/health", get(health))
        .route("/api/v1/models", get(models))
        .route("/api/v1/chat/completions", post(openai_chat))
        .route("/api/v1/messages", post(anthropic_messages))
        // Ollama 兼容端点
        .route("/", get(ollama_root))
        .route("/api/version", get(ollama_version))
        .route("/api/tags", get(ollama_tags))
        .route("/api/ps", get(ollama_ps))
        .route("/api/show", post(ollama_show))
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/embeddings", post(ollama_embeddings))
        .with_state(state)
        .layer(middleware::from_fn(cors))
        .layer(middleware::from_fn(trace_request));

    let addr = format!("{}:{}", args.host, args.port);
    match args.tls_pair()? {
        // 指定了证书与私钥：以 HTTPS（rustls）对外提供服务
        Some((cert, key)) => {
            install_tls_provider();
            let config = axum_server::tls_rustls::RustlsConfig::from_pem_file(cert, key)
                .await
                .map_err(|e| {
                    anyhow::anyhow!("加载 TLS 证书失败（{cert:?} / {key:?}）: {e}")
                })?;
            let handle = axum_server::Handle::new();
            let shutdown = handle.clone();
            tokio::spawn(async move {
                shutdown_signal().await;
                // 给已建立的连接 10 秒收尾时间
                shutdown.graceful_shutdown(Some(std::time::Duration::from_secs(10)));
            });
            print_banner(&addr, "https");
            axum_server::bind_rustls(addr.parse()?, config)
                .handle(handle)
                .serve(app.into_make_service())
                .await?;
        }
        None => {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            print_banner(&addr, "http");
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?;
        }
    }
    Ok(())
}
