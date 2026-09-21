# 10 · HTTP 接口参考

服务默认监听 `127.0.0.1:8000`（`--host` / `--port`），可用 `--tls-cert` + `--tls-key` 切到 HTTPS。
所有 POST 接口都要求 `Content-Type: application/json`，请求体上限 **2 MB**（超出返回 `413`）。

`model` 字段可省略，省略时回显当前加载的模型名。

---

## 通用

### `GET /api/health`

```json
{"status":"ok","model":"Spark-X2.5-1.7B","max_context":8192}
```

### `GET /api/v1/models`

```json
{"object":"list","data":[{"id":"Spark-X2.5-1.7B","object":"model","created":1758182286,"owned_by":"spark-candle"}]}
```

### 错误响应

```json
{"error":{"message":"messages 不能为空","type":"invalid_request_error"}}
```

| 状态码 | 常见原因 |
| --- | --- |
| `400` | `messages` 为空 / `prompt` 为空 |
| `404` | 路径不存在 |
| `413` | 请求体超过 2 MB |
| `500` | 推理失败 |
| `501` | `/api/embeddings`（不支持） |

---

## OpenAI Chat Completions

### `POST /api/v1/chat/completions`

请求字段：

| 字段 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | string | 当前模型 | 仅用于回显 |
| `messages` | array | — | `[{"role":"user"|"assistant"|"system","content":"..."|[{"type":"text","text":"..."}]}]` |
| `stream` | bool | `false` | true 时返回 SSE |
| `max_tokens` | int | `--max-tokens` | 超过剩余上下文会被夹取 |
| `temperature` | float | `--temperature` | `0` = 贪婪 |
| `top_p` | float | `--top-p` | 核采样阈值 |

**非流式响应**

```json
{
  "id":"chatcmpl-3f2a...","object":"chat.completion","created":1758182286,
  "model":"Spark-X2.5-1.7B",
  "choices":[{"index":0,"message":{"role":"assistant","content":"你好！"},"finish_reason":"stop"}],
  "usage":{"prompt_tokens":12,"completion_tokens":8,"total_tokens":20}
}
```

**流式响应（SSE）**

```
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

data: {"…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

- 空增量（还没凑齐一个完整字符）会发 `: keep-alive` 注释帧，客户端应忽略
- `finish_reason`：`stop`（正常结束或被取消）/ `length`（到达上限）

```bash
curl http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'
```

客户端配置：`base_url = http://127.0.0.1:8000/api/v1`，`api_key` 任意。

---

## Anthropic Messages

### `POST /api/v1/messages`

请求字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model` | string | 仅用于回显 |
| `max_tokens` | int | 最大生成 Token 数 |
| `system` | string 或分块数组 | 顶层 system 提示 |
| `messages` | array | 同 OpenAI（不含 system） |
| `stream` | bool | true 时返回 SSE |
| `temperature` / `top_p` | float | 覆盖默认值 |

**非流式响应**

```json
{
  "id":"msg_9a1c…","type":"message","role":"assistant","model":"Spark-X2.5-1.7B",
  "content":[{"type":"text","text":"你好！"}],
  "stop_reason":"end_turn","stop_sequence":null,
  "usage":{"input_tokens":12,"output_tokens":8}
}
```

**流式事件顺序**

```
event: message_start          data: {"type":"message_start","message":{…}}
event: content_block_start    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
event: content_block_delta    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你好"}}
event: content_block_stop     data: {"type":"content_block_stop","index":0}
event: message_delta          data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":8}}
event: message_stop           data: {"type":"message_stop"}
```

`stop_reason`：`end_turn` / `max_tokens`。

---

## Ollama

### `GET /`

```
Ollama is running
```

### `GET /api/version`

```json
{"version":"0.1.0"}
```

取自 Cargo 包版本（`OLLAMA_VERSION`）。

### `GET /api/tags` · `GET /api/ps`

```json
{"models":[{"name":"Spark-X2.5-1.7B:latest","model":"Spark-X2.5-1.7B:latest",
  "modified_at":"2026-09-18T07:33:26.000000000Z","size":3543348019,
  "digest":"sha256:…","details":{"format":"safetensors","family":"spark","quantization_level":"BF16",…}}]}
```

`/api/ps` 额外带 `expires_at`（当前时间 +5 分钟）与 `size_vram`。

### `POST /api/show`

```json
{"license":"","modelfile":"","parameters":"num_ctx 8192","template":"{{ .Prompt }}","system":"",
 "details":{…},"model_info":{"general.architecture":"spark","spark.block_count":24,…},
 "capabilities":["completion","chat"]}
```

### `POST /api/chat`

请求字段：`model`、`messages`、`system`、`stream`（**缺省 true**）、
`options`（`temperature` / `top_p` / `num_predict` / `seed`）。

NDJSON，一行一个对象：

```json
{"model":"Spark-X2.5-1.7B","created_at":"2026-09-18T08:58:06.123456789Z","done":false,"message":{"role":"assistant","content":"你好"}}
{"model":"Spark-X2.5-1.7B","created_at":"…","done":true,"message":{"role":"assistant","content":""},
 "done_reason":"stop","prompt_eval_count":12,"eval_count":8,
 "total_duration":1234000000,"load_duration":0,"prompt_eval_duration":0,"eval_duration":1234000000}
```

非流式时返回单个 JSON 对象（字段同上，`done: true`）。

### `POST /api/generate`

请求字段：`model`、`prompt`（必填，空白返回 400）、`system`、`stream`、`options`。

与 `/api/chat` 的差别：增量放在 `response` 字段而不是 `message.content`，
结束帧额外带 `context: []`。

### `POST /api/embeddings`

固定返回 `501`：当前模型不支持 embeddings。

---

## 结束原因对照表

| 引擎内部 | OpenAI | Anthropic | Ollama |
| --- | --- | --- | --- |
| `stop`（遇到 EOS） | `stop` | `end_turn` | `stop` |
| `length`（到达上限） | `length` | `max_tokens` | `length` |
| `cancel`（被取消） | `stop` | `end_turn` | `stop` |

> 各家协议都没有「已取消」状态，因此取消只能表现为正常结束。

## 并发与排队

所有请求共享一份 KV-Cache 会话，生成槽会把并发请求**排队**而不是拒绝。
因此吞吐约等于单路串行；若客户端有并发需求，建议在自己一侧串行化，或直接起多个进程
（每个进程加载一份权重，显存占用会翻倍）。
