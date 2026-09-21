# spark_candle

基于 **Candle（Rust）** 的 **Spark-X2.5** 本地推理运行时，提供两种前端：

- `spark_candle chat` —— 终端 TUI 多轮对话（默认子命令）
- `spark_candle serve` —— 兼容 **OpenAI Chat Completions**、**Anthropic Messages**、**Ollama** 三种协议的 HTTP 服务

[English](./README.md) | 简体中文 | [完整文档](./docs/README.md)

---

## 特性

- **不依赖 Python 运行时** —— 单个可执行文件，权重通过 `safetensors` 内存映射加载
- **混合注意力** —— 滑动窗口层 + 全注意力层混排，GQA 分组查询、逐头输出门控、partial RoPE
- **跨轮次复用 KV-Cache** —— 多轮对话每轮只 prefill 新增的 Token
- **流式输出** —— OpenAI / Anthropic 走 SSE，Ollama 走 NDJSON
- **HTTPS** —— 通过 `--tls-cert` / `--tls-key` 启用（rustls + ring）
- **结构化 JSONL 日志** —— 一行一个 JSON，按天滚动；请求失败时记录 URI 与请求体

---

## 环境要求

| 项目 | 说明 |
| --- | --- |
| Rust | edition 2021，建议使用较新的 stable 工具链 |
| GPU（可选） | 自动探测 CUDA，不可用时回退 CPU |
| 磁盘 | 默认 1.7B 权重约 3.5 GB（HF 缓存目录） |

权重从 Hugging Face 拉取。代码默认设置 `HF_ENDPOINT=https://hf-mirror.com`（国内镜像），
若你有直连环境，可注释掉 `src/model.rs` 中该行。

---

## 编译

```bash
cargo build --release
# 产物：target/release/spark_candle
```

---

## 快速开始

### 终端对话（TUI）

```bash
cargo run --release -- chat
# 等价于
cargo run --release
```

| 按键 | 功能 |
| --- | --- |
| `Enter` | 发送 |
| `Alt+Enter` | 换行 |
| `↑` / `↓` / `PgUp` / `PgDn` | 滚动对话区 |
| `Esc` / `Ctrl+C` | 生成中取消；空闲时退出 |
| `/clear` | 清空会话历史 **并** 丢弃 KV-Cache |
| `/reset` | 仅重置 KV-Cache |
| `/help` | 显示快捷键 |
| `/exit` `/quit` `/q` | 退出 |

### HTTP 服务

```bash
cargo run --release -- serve --host 0.0.0.0 --port 8000
```

---

## 命令行参数

`chat` 与 `serve` 共用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `XHToken/Spark-X2.5-1.7B` | Hugging Face 仓库 ID |
| `--shards` | `2` | 仓库缺少 `index.json` 时按此数推断分片名 |
| `--dtype` | 自动（CUDA=bf16，CPU=f16） | `f16` / `bf16` / `f32` |
| `--max-context` | `8192` | 上下文上限（Prompt + 生成） |
| `--max-tokens` | `8192` | 默认单次生成最大 Token 数 |
| `--temperature` | `0.7` | 采样温度，`0` 为贪婪解码 |
| `--top-p` | `0.95` | 核采样阈值 |
| `--seed` | 无 | 固定随机种子，便于复现 |
| `--thinking` | `false` | 开启官方模板的思考模式 |
| `--system` | 无 | 全局 system 提示，每次请求自动前置 |
| `--log-dir` | `logs` | 日志目录（按天滚动） |
| `--log-level` | `info` | `error`/`warn`/`info`/`debug`/`trace`，可被 `RUST_LOG` 覆盖 |

`serve` 专属参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--tls-cert` | 无 | PEM 证书（需与 `--tls-key` 同时给出才启用 HTTPS） |
| `--tls-key` | 无 | PEM 私钥（PKCS#1 / PKCS#8 / SEC1） |

---

## HTTP 接口一览

| 方法 | 路径 | 协议 | 流式 |
| --- | --- | --- | --- |
| GET | `/api/health` | 健康检查 | – |
| GET | `/api/v1/models` | OpenAI | – |
| POST | `/api/v1/chat/completions` | OpenAI Chat Completions | SSE（`stream: true`） |
| POST | `/api/v1/messages` | Anthropic Messages | SSE（`stream: true`） |
| GET | `/` | Ollama 存活探测（返回 `Ollama is running`） | – |
| GET | `/api/version` | Ollama | – |
| GET | `/api/tags`、`/api/ps` | Ollama | – |
| POST | `/api/show` | Ollama | – |
| POST | `/api/chat` | Ollama | NDJSON（默认开启） |
| POST | `/api/generate` | Ollama | NDJSON（默认开启） |
| POST | `/api/embeddings` | Ollama | 501，当前模型不支持 |

### 示例

```bash
# OpenAI，非流式
curl http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"spark","messages":[{"role":"user","content":"你好"}],"max_tokens":512}'

# OpenAI，流式
curl -N http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}],"stream":true}'

# Anthropic
curl http://127.0.0.1:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"max_tokens":512,"system":"请简洁回答。","messages":[{"role":"user","content":"你好"}]}'

# Ollama
curl -N http://127.0.0.1:8000/api/chat \
  -d '{"model":"spark","messages":[{"role":"user","content":"你好"}]}'
```

任何 OpenAI 兼容客户端把 `base_url` 指向 `http://127.0.0.1:8000/api/v1` 即可直接使用
（Ollama 客户端指向 `/api`）。

---

## 注意事项

- **同一时刻只跑一个生成任务。** 所有请求共享一份 KV-Cache 会话，生成槽会把并发请求
  排队（不是拒绝），避免上下文交错污染。
- **预算分配提示词优先。** 当 `prompt + max_tokens` 超过 `--max-context` 时，
  从前往后丢弃提示词 Token（WARN 日志会同时打印原始输入、被丢弃部分与剩余部分），
  剩余空间全部留给生成；客户端塞进过大的 `max_tokens` 只会被静默夹取（debug 级日志）。
- **请求体超过 2 MB** 直接返回 `413`。
- 取消（`cancel`）在协议层面只能表现为 `stop` / `end_turn` / `stop`，各家协议没有
  “已取消”这个状态。

---

## 日志

JSONL 格式，一行一个 JSON 对象，按天滚动：`logs/spark.log.jsonl.YYYY-MM-DD`
（目录由 `--log-dir` 指定）。自定义字段被展平到顶层，例如：

```json
{"timestamp":"2026-09-18T08:58:06.665892Z","level":"INFO","message":"收到请求","target":"api","api":"openai","model":"spark","messages":2,"stream":false,"max_tokens":8192,"temperature":0.7,"top_p":0.95}
```

常用 `target`：`app`、`model`、`gen`、`session`、`api`、`http`、`tui`。

```bash
RUST_LOG=debug cargo run --release -- serve   # 覆盖 --log-level
```

---

## 项目结构

```
src/
  main.rs     程序入口，分派到 chat / serve
  cli.rs      clap 命令行参数定义
  logging.rs  JSONL tracing 订阅者（按天滚动）
  chat.rs     Message 结构 + Jinja 对话模板渲染
  model.rs    Spark-X2.5 的 Candle 实现 + 权重下载
  engine.rs   模型加载、KV-Cache 会话、采样、流式生成
  server.rs   OpenAI / Anthropic / Ollama HTTP 层（axum）
  tui.rs      终端界面（ratatui + crossterm）
docs/         分文件、分功能的说明文档
```

---

## 文档

完整文档见 [docs/README.md](./docs/README.md)：

- [架构总览](./docs/01-architecture.md)
- [main.rs](./docs/02-main.md) · [cli.rs](./docs/03-cli.md) · [logging.rs](./docs/04-logging.md)
- [chat.rs](./docs/05-chat.md) · [model.rs](./docs/06-model.md) · [engine.rs](./docs/07-engine.md)
- [server.rs](./docs/08-server.md) · [tui.rs](./docs/09-tui.md)
- [HTTP 接口参考](./docs/10-http-api.md)

---

## 许可

模型权重的许可请参照模型仓库（`XHToken/Spark-X2.5-1.7B`）的说明。
