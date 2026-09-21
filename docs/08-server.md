# 08 · `src/server.rs`

基于 axum 的 HTTP 层，同时兼容 **OpenAI Chat Completions**、**Anthropic Messages**、**Ollama** 三套协议。
按文件中的编号小节说明。

---

## 共享状态

### `struct AppState`

| 字段 | 说明 |
| --- | --- |
| `engine: Arc<Engine>` | 推理引擎 |
| `default: GenOptions` | 命令行默认采样参数，请求未覆盖的字段用它兜底 |

方法：

- `gen_options(max_tokens, temperature, top_p)` —— 用请求字段覆盖默认值
- `model_name(requested)` —— 请求没给 `model` 时用当前加载的模型名
- `generate_blocking(messages, opts)` —— 包在 `tokio::task::block_in_place` 里的完整生成
  （推理是同步密集任务，必须告诉 Tokio 当前线程会阻塞，否则会占死工作线程）

---

## 工具类结构体

| 结构体 | 说明 |
| --- | --- |
| `ContentValue` | `untagged` 枚举：`Text(String)`（OpenAI）/ `Parts(Vec<ContentPart>)`（Anthropic） |
| `ContentPart` | Anthropic 的一个内容块；只取 `text`，非文本块忽略 |
| `ApiMessage` | 两种协议共用的入参消息，`role` 缺省 `user` |
| `OaiChatRequest` / `AnthropicRequest` | 两家协议的请求体（Anthropic 的 `system` 是独立顶层字段） |
| `Usage` / `OaiRespMessage` / `OaiChoice` / `OaiChatResponse` | OpenAI 非流式响应 |
| `OaiDelta` / `OaiChunkChoice` / `OaiChunk` | OpenAI SSE 帧 |
| `AnthropicTextBlock` / `AnthropicResponse` / `AnthropicUsage` | Anthropic 响应 |
| `ModelCard` / `ModelsResponse` | `/v1/models` |
| `ErrorBody` / `ErrorDetail` | OpenAI 风格错误体（`invalid_request_error` / `server_error`） |
| `OllamaOptions` / `OllamaChatRequest` / `OllamaGenerateRequest` / `OllamaModelRequest` | Ollama 请求体 |

`ContentValue::to_text()` 把分块内容折叠成纯文本（用 `\n` 连接）。

---

## 工具函数

| 函数 | 作用 |
| --- | --- |
| `now_secs()` | 当前 Unix 秒 |
| `random_suffix()` | 16 位十六进制，用于 `chatcmpl-…` / `msg_…` |
| `rfc3339(secs, nanos)` | 自实现 RFC3339（Howard Hinnant 的 `civil_from_days`），避免引入时间库 |
| `now_rfc3339()` | 当前时刻的 RFC3339（Ollama `created_at`） |
| `as_nanos(ms)` | 毫秒 → 纳秒（Ollama 的 `*_duration`） |
| `error_response(status, msg)` | OpenAI 风格错误响应 |
| `error_event(msg)` | 流式中途失败时推的 SSE 错误事件（`server_error`） |
| `to_messages(&[ApiMessage])` | 归一化：只留 user/assistant/system，丢弃空白内容 |
| `build_messages(system, &[ApiMessage])` | system 前置 + `to_messages` |
| `inference_error(api, e)` | 记 ERROR 并返回 500 |
| `log_input(...)` | info 记条数与参数，debug 记完整内容 |
| `log_output(api, res)` | info 记输出统计 |
| `map_finish_reason(reason, length, stop)` | 引擎原因 → 协议字面量 |
| `anthropic_stop_reason(reason)` | → `max_tokens` / `end_turn` |
| `ollama_done_reason(reason)` | → `length` / `stop` |

### 两个宏

| 宏 | 作用 |
| --- | --- |
| `messages_or_400!` | 组装消息；为空时记 WARN（`target: "api"`）并**从当前函数返回** 400 |
| `generation_or_500!` | 在 `block_in_place` 里启动流式生成；失败记 ERROR 并返回 500 |

用宏而不是函数，是因为它们需要 `return` 穿透到调用方 handler。

---

## OpenAI 接口

### `async fn openai_chat(...) -> Response`

`POST /api/v1/chat/completions`

- 非流式：`generate_blocking` → `OaiChatResponse`
- 流式：`Phase` 状态机推进 SSE

```
Role     → 首帧 {"delta":{"role":"assistant","content":""}}
Running  → 每帧 {"delta":{"content":"增量"}}；空增量发 : keep-alive 注释
Tail     → {"delta":{},"finish_reason":"stop|length"}
Done     → data: [DONE]
End      → 流结束
```

## Anthropic 接口

### `async fn anthropic_messages(...) -> Response`

`POST /api/v1/messages`，严格按 Anthropic 的事件顺序：

```
message_start → content_block_start → content_block_delta*
              → content_block_stop → message_delta → message_stop
```

`system` 从顶层字段取，`messages` 走 `messages_or_400!`。

---

## Ollama 接口

| 函数 | 路径 | 说明 |
| --- | --- | --- |
| `ollama_root` | `GET /` | 返回 `Ollama is running` |
| `ollama_version` | `GET /api/version` | 返回 Cargo 包版本 |
| `ollama_tags` | `GET /api/tags` | 模型列表（含 digest、details） |
| `ollama_ps` | `GET /api/ps` | 模拟运行中模型，过期时间 +5 分钟 |
| `ollama_show` | `POST /api/show` | 模型详情；忽略请求里的 `model` |
| `ollama_chat` | `POST /api/chat` | 多轮对话，默认流式 |
| `ollama_generate` | `POST /api/generate` | 单轮补全，`prompt` 为空返回 400 |
| `ollama_embeddings` | `POST /api/embeddings` | 固定 501 |

辅助函数：

| 函数 | 作用 |
| --- | --- |
| `ollama_digest(info)` | 由 id / 大小 / mtime 派生稳定伪 digest（客户端只做展示比对） |
| `ollama_details(info)` | `/api/tags`、`/api/ps` 的 `details` 字段 |
| `ollama_model_info(info)` | `/api/show` 的 `model_info` 字段 |
| `ollama_done_stats(gen, elapsed)` | 结束帧的 Token 计数与耗时（chat / generate 共用） |
| `ollama_stream(model, gen, delta_payload, done_payload)` | 把生成过程包成 NDJSON 流 |
| `merge(base, extra)` | 合并两个 JSON 对象的顶层字段 |
| `ndjson_line(obj)` | 序列化成一行 NDJSON |
| `ndjson_response(body)` | 套 `application/x-ndjson` 响应头 |

`ollama_stream` 用两个函数指针（`delta_payload` / `done_payload`）区分
chat（`message.content`）与 generate（`response`）的字段差异，避免复制整套状态机。
`Generation` 体积较大，装箱后枚举各分支大小才均衡。

---

## 其它端点

| 函数 | 路径 | 说明 |
| --- | --- | --- |
| `models` | `GET /api/v1/models` | 只有当前加载的这一个模型 |
| `health` | `GET /api/health` | `status` / `model` / `max_context` |

---

## 中间件

### `async fn trace_request(req, next) -> Response`

`const LOG_BODY_LIMIT: usize = 2 * 1024 * 1024;`

`Body` 只能消费一次，所以中间件先读进内存、再**原样回填**给下游 handler：

```
GET / HEAD 或 Content-Length > 2MB  → 不读体（超大只记占位说明）
其余                                → to_bytes(body, 2MB)
  成功 → 文本留档，Bytes 回填
  失败 → 记 WARN + 返回 413
handler 执行后：
  成功 → info:  method / uri / status / elapsed_ms
  失败 → warn:  同上 + body（经 preview_text 裁剪）
```

这条 WARN 是排查「客户端到底发了什么」的主要手段。

### `async fn cors(req, next) -> Response`

- `OPTIONS` 直接返回 `204`，不进 handler
- 放行全部来源 / 方法 / 头
- 额外加 `x-accel-buffering: no`，防止 Nginx 之类反向代理缓冲 SSE

中间件顺序（`.layer` 后加的在外层）：`cors` 在外，`trace_request` 在内。

---

## 启动

| 函数 | 作用 |
| --- | --- |
| `shutdown_signal()` | 等 Ctrl+C |
| `install_tls_provider()` | 安装 rustls 默认加密后端；已安装则跳过 |
| `print_banner(addr, scheme)` | 打印各协议端点 |
| `pub async fn run(engine, args)` | 建路由 → 挂中间件 → 选 HTTPS / HTTP |

路由表：

```
GET  /api/health
GET  /api/v1/models
POST /api/v1/chat/completions     OpenAI
POST /api/v1/messages             Anthropic
GET  /                            Ollama 存活
GET  /api/version
GET  /api/tags
GET  /api/ps
POST /api/show
POST /api/chat
POST /api/generate
POST /api/embeddings
```

- 有 TLS：`axum_server::bind_rustls` + `Handle`，Ctrl+C 后 **10 秒**优雅收尾
- 无 TLS：`axum::serve(listener, app).with_graceful_shutdown(...)`

---

## 常见改动

| 需求 | 改哪里 |
| --- | --- |
| 加新端点 | `run()` 里加 `.route(...)` + 一个 handler |
| 放宽请求体上限 | `LOG_BODY_LIMIT`（同时影响日志与 413 阈值） |
| 改 CORS 策略 | `cors` 中间件（目前是全放行） |
| 支持新的结束原因 | `map_finish_reason` 及各协议的包装函数 |
| 让流式带 usage 统计 | 各 `Phase::Tail` / `BlockStop` 分支 |
