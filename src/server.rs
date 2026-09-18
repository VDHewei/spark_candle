//! 兼容 OpenAI Chat Completions 与 Anthropic Messages 的 HTTP 服务

use crate::chat::Message;
use crate::cli::ServeArgs;
use crate::engine::{Engine, GenOptions, Generation};
use axum::{
    body::Body,
    extract::State,
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{sse::Event, IntoResponse, Response, Sse},
    routing::{get, post},
    Json, Router,
};
use futures_util::stream;
use serde::{Deserialize, Serialize};
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

fn anthropic_stop_reason(reason: &str) -> &'static str {
    match reason {
        "length" => "max_tokens",
        _ => "end_turn",
    }
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
        // 生成槽可能被其它请求占用，等待过程放在 block_in_place 中，避免阻塞异步运行时
        let result = tokio::task::block_in_place(|| {
            state
                .engine
                .generate_blocking(&messages, opts, Arc::new(AtomicBool::new(false)), |_| {})
        });
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
            Err(e) => {
                error!(target: "api", api = "openai", error = %e, "推理失败");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}"))
            }
        };
    }

    let gen = match tokio::task::block_in_place(|| state.engine.start_generation(&messages, opts)) {
        Ok(g) => g,
        Err(e) => {
            error!(target: "api", api = "openai", error = %e, "推理失败");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}"));
        }
    };

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
    let mut messages = Vec::new();
    if let Some(system) = &req.system {
        let text = system.to_text();
        if !text.trim().is_empty() {
            messages.push(Message {
                role: "system".to_string(),
                content: text,
            });
        }
    }
    messages.extend(to_messages(&req.messages));
    if messages.is_empty() {
        warn!(target: "api", api = "anthropic", "请求被拒绝：messages 为空");
        return error_response(StatusCode::BAD_REQUEST, "messages 不能为空");
    }

    let opts = state.gen_options(req.max_tokens, req.temperature, req.top_p);
    let model = state.model_name(req.model.clone());
    log_input("anthropic", &model, &messages, &opts, req.stream);
    let id = format!("msg_{}", random_suffix());

    if !req.stream {
        let result = tokio::task::block_in_place(|| {
            state
                .engine
                .generate_blocking(&messages, opts, Arc::new(AtomicBool::new(false)), |_| {})
        });
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
            Err(e) => {
                error!(target: "api", api = "anthropic", error = %e, "推理失败");
                error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}"))
            }
        };
    }

    let gen = match tokio::task::block_in_place(|| state.engine.start_generation(&messages, opts)) {
        Ok(g) => g,
        Err(e) => {
            error!(target: "api", api = "anthropic", error = %e, "推理失败");
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, format!("推理失败: {e}"));
        }
    };

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

pub async fn run(engine: Arc<Engine>, args: ServeArgs) -> anyhow::Result<()> {
    let state = AppState {
        engine,
        default: args.common.gen_options(None),
    };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(openai_chat))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
        .layer(middleware::from_fn(cors))
        .layer(middleware::from_fn(trace_request));

    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    println!("\n【系统提示】HTTP 服务已启动: http://{}", addr);
    println!("  · OpenAI     POST http://{addr}/v1/chat/completions");
    println!("  · Anthropic  POST http://{addr}/v1/messages");
    println!("  · 模型列表   GET  http://{addr}/v1/models");
    println!("  · 健康检查   GET  http://{addr}/health\n");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}
