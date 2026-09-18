//! 兼容 OpenAI Chat Completions 与 Anthropic Messages 的 HTTP 服务

use crate::chat::Message;
use crate::cli::ServeArgs;
use crate::engine::{Engine, GenOptions, Generation};
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

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    default: GenOptions,
}

// ==========================================
// 请求 / 响应数据结构
// ==========================================
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ContentValue {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(default)]
    text: Option<String>,
}

impl ContentValue {
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

#[derive(Debug, Deserialize)]
struct ApiMessage {
    #[serde(default = "default_role")]
    role: String,
    #[serde(default)]
    content: Option<ContentValue>,
}

fn default_role() -> String {
    "user".to_string()
}

#[derive(Debug, Deserialize)]
struct OaiChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ApiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    max_tokens: Option<usize>,
    #[serde(default)]
    system: Option<ContentValue>,
    #[serde(default)]
    messages: Vec<ApiMessage>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct OaiRespMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OaiChoice {
    index: usize,
    message: OaiRespMessage,
    finish_reason: String,
}

#[derive(Debug, Serialize)]
struct OaiChatResponse {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OaiChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct OaiDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Serialize)]
struct OaiChunkChoice {
    index: usize,
    delta: OaiDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
struct OaiChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<OaiChunkChoice>,
}

#[derive(Debug, Serialize)]
struct AnthropicTextBlock {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

#[derive(Debug, Serialize)]
struct AnthropicResponse {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    role: String,
    model: String,
    content: Vec<AnthropicTextBlock>,
    stop_reason: String,
    stop_sequence: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelCard {
    id: String,
    object: String,
    created: u64,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    object: String,
    data: Vec<ModelCard>,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: String,
}

// ==========================================
// 工具函数
// ==========================================
fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

fn random_suffix() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// 把 Unix 时间戳转成 RFC3339（UTC）字符串，如 2026-09-18T07:33:26.123456789Z
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

fn now_rfc3339() -> String {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    rfc3339(d.as_secs(), d.subsec_nanos())
}

/// 毫秒转纳秒（Ollama 的 duration 字段单位为纳秒）
fn as_nanos(ms: u128) -> u128 {
    ms.saturating_mul(1_000_000)
}

fn error_response(status: StatusCode, message: impl Into<String>) -> Response {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            kind: "invalid_request_error".to_string(),
        },
    };
    (status, Json(body)).into_response()
}

fn error_event(message: impl Into<String>) -> Result<Event, Infallible> {
    let body = ErrorBody {
        error: ErrorDetail {
            message: message.into(),
            kind: "server_error".to_string(),
        },
    };
    Ok(Event::default().data(serde_json::to_string(&body).unwrap_or_default()))
}

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

/// 组装并校验请求消息；为空时直接返回 400 响应
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

/// 启动流式生成；失败时记录日志并直接返回 500 响应
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

/// 推理失败：记录日志并生成 500 响应
fn inference_error(api: &'static str, e: impl std::fmt::Display) -> Response {
    error!(target: "api", api, error = %e, "推理失败");
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("推理失败: {e}"),
    )
}

/// 记录一次 API 调用的输入（内容以 debug 级别记录，避免 info 日志过大）
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

/// 记录一次 API 调用的输出统计
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

/// 把内部结束原因映射成各家协议约定的字面量
fn map_finish_reason(reason: &str, length: &'static str, stop: &'static str) -> &'static str {
    match reason {
        "length" => length,
        _ => stop,
    }
}

fn anthropic_stop_reason(reason: &str) -> &'static str {
    map_finish_reason(reason, "max_tokens", "end_turn")
}

impl AppState {
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

    fn model_name(&self, requested: Option<String>) -> String {
        requested.unwrap_or_else(|| self.engine.model_id().to_string())
    }

    /// 阻塞式完整生成：等待生成槽空闲，避免占用异步运行时线程
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

    enum Phase {
        Role(Generation),
        Running(Generation),
        Tail(&'static str),
        Done,
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

    enum Phase {
        MessageStart(Generation),
        BlockStart(Generation),
        Running(Generation),
        BlockStop(&'static str, usize),
        MessageDelta,
        MessageStop,
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

#[derive(Debug, Deserialize)]
struct OllamaOptions {
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_p: Option<f64>,
    /// Ollama 用 num_predict 表示最大生成 Token 数
    #[serde(default)]
    num_predict: Option<usize>,
    #[serde(default)]
    seed: Option<u64>,
}

impl OllamaOptions {
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

#[derive(Debug, Deserialize)]
struct OllamaChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<ApiMessage>,
    /// Ollama 缺省即为流式输出
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    system: Option<ContentValue>,
    #[serde(default)]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
struct OllamaGenerateRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Deserialize)]
struct OllamaModelRequest {
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

fn merge(base: &mut serde_json::Value, extra: serde_json::Value) {
    if let (Some(base), Some(extra)) = (base.as_object_mut(), extra.as_object()) {
        for (k, v) in extra {
            base.insert(k.clone(), v.clone());
        }
    }
}

fn ndjson_line(obj: &serde_json::Value) -> Result<Bytes, Infallible> {
    let mut line = serde_json::to_string(obj).unwrap_or_default();
    line.push('\n');
    Ok(Bytes::from(line))
}

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

async fn ollama_version() -> Json<serde_json::Value> {
    Json(json!({ "version": OLLAMA_VERSION }))
}

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

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "model": state.engine.model_id(),
        "max_context": state.engine.max_context(),
    }))
}

/// 记录每次 HTTP 访问的方法、路径、状态码与耗时
async fn trace_request(req: Request<Body>, next: Next) -> Response {
    let method = req.method().clone();
    let uri = req.uri().clone();
    let started = Instant::now();
    let res = next.run(req).await;
    let elapsed_ms = started.elapsed().as_millis();
    let status = res.status().as_u16();
    if res.status().is_success() {
        info!(target: "http", %method, %uri, status, elapsed_ms, "请求完成");
    } else {
        warn!(target: "http", %method, %uri, status, elapsed_ms, "请求异常");
    }
    res
}

/// 简单的 CORS + OPTIONS 预检处理
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

/// 打印服务启动横幅（scheme 为 http / https）
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
