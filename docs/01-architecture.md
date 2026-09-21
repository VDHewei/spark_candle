# 01 · 整体架构

## 模块分层

```
┌─────────────────────────────────────────────────────────┐
│  main.rs          入口：解析 CLI，分派 chat / serve       │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
        ┌───────▼────────┐        ┌───────▼────────┐
        │  tui.rs        │        │  server.rs     │
        │  终端界面       │        │  HTTP 服务      │
        └───────┬────────┘        └───────┬────────┘
                │                         │
                └───────────┬─────────────┘
                            │
                ┌───────────▼────────────┐
                │  engine.rs             │
                │  加载 / KV-Cache / 采样  │
                └───────────┬────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼──────┐  ┌─────────▼────────┐  ┌───────▼──────┐
│  model.rs    │  │  chat.rs         │  │  logging.rs  │
│  Candle 实现 │  │  Jinja 模板渲染   │  │  JSONL 日志  │
└──────────────┘  └──────────────────┘  └──────────────┘
```

- `cli.rs` 只负责参数定义，被 `main.rs` 消费
- `engine.rs` 是唯一的「推理真相来源」，TUI 与 HTTP 都只调它
- `model.rs` 只被 `engine.rs` 使用，不感知任何协议
- `logging.rs` 被 `main.rs` 初始化一次，其余模块只管打日志

## 启动时序（两种模式共用）

```
Cli::parse()
  └─ init_logging(common, console)       // TUI: console=false；serve: console=true
  └─ load_engine(common, mode)
       └─ Engine::load(repo_id, shards, dtype, max_context, system)
            ├─ prepare_model_files()     // 下载 / 复用 HF 缓存
            ├─ Device::new_cuda(0) 或 Cpu
            ├─ 解析 config.json → SparkConfig
            ├─ Tokenizer::from_file()
            ├─ VarBuilder::from_mmaped_safetensors()   // 零拷贝映射
            └─ SparkModel::load()
  └─ tui::run()  |  server::run()
```

`serve` 模式在加载模型**之前**先校验 TLS 参数（`args.tls_pair()`），
避免耗时下载完成后才因为证书路径写错而报错。

## 一次请求的完整数据流

```
HTTP 请求
  │
  ├─ middleware: cors          // 处理 OPTIONS 预检 + 放行 CORS
  ├─ middleware: trace_request // 读取 body（≤2MB）→ 原样回填 → 失败时记日志
  │
  ├─ handler（openai_chat / anthropic_messages / ollama_chat / ollama_generate）
  │    ├─ Json<XxxRequest> 反序列化
  │    ├─ to_messages() / build_messages()    // 归一化成 Vec<Message>
  │    ├─ state.gen_options(...)              // 请求字段覆盖命令行默认
  │    └─ log_input()                         // target="api"
  │
  ├─ engine.start_generation()                // 抢占生成槽 + 渲染模板 + 编码
  │    ├─ apply_chat_template() 或 build_prompt_fallback()
  │    ├─ tokenizer.encode()
  │    ├─ 上下文预算裁剪（见下）
  │    └─ 返回 Generation
  │
  ├─ 循环 gen.step()
  │    ├─ 复用 KV-Cache 前缀 → forward_chunked() → sample_token()
  │    └─ 增量解码（整体解码后取新增后缀，避免半个 UTF-8 字符）
  │
  └─ 按协议包装：SSE（OpenAI/Anthropic）或 NDJSON（Ollama）
```

## 并发模型：单生成槽

`Engine` 内部只有**一份** KV-Cache 会话（`Session`），因此：

- `GenSlot` 用 `Mutex<Option<u64>>` + `Condvar` 实现「同一时刻仅一个生成任务」
- 新任务调用 `slot.acquire(id)` 会阻塞等待；`Generation` 被 `Drop` 时 `release(id)`
- HTTP 侧用 `tokio::task::block_in_place` 包裹，避免占死异步工作线程
- 结果是**并发请求排队而不是报错**；代价是吞吐等于单路串行

为什么要串行：多个任务交错改写同一份 `Session` 会导致上下文错乱、反复重新 prefill。

## KV-Cache 复用规则

`Generation::step()` 每步都会判断缓存能否复用：

| 条件 | 处理 |
| --- | --- |
| 层数不符（`caches.len() != layers`） | 整段重新 prefill |
| 缓存已追平输入（`session.tokens.len() >= input_ids.len()`） | 整段重新 prefill |
| 输入不再以缓存为前缀（前缀被改写） | 整段重新 prefill |
| 其余情况 | 只前向 `input_ids[pos_offset..]` 这一段 |

TUI 取消 / `/clear` / `/reset` 都会调 `Engine::reset()` 丢弃缓存，
因为中断后缓存停在半截状态，继续复用会得到错乱上下文。

## 上下文预算

```
max_ctx      = --max-context（默认 8192）
prompt_budget= max_ctx - 1          // 至少留一个位置给生成
若 prompt > prompt_budget：从前往后丢弃最早的 Token（WARN，记录原文/丢弃/剩余）
allowed      = max_ctx - prompt_len
max_tokens   = min(请求值, allowed) // 差值仅在 debug 级记录
```

设计取舍：**提示词优先**。客户端（尤其是各类前端默认配置）常常直接传
`max_tokens: 32000`，若按「给提示词留一半」的旧策略，几乎每个请求都会被夹到
`max_ctx/2` 并刷 WARN；现在只有提示词本身超限才告警。

## 采样

| 输入 | 行为 |
| --- | --- |
| `temperature <= 0` | 贪婪解码，`argmax` |
| `temperature > 0` | softmax(logits / T) → 按概率降序累加到 `top_p` → 在候选集内按权重随机 |
| `seed` 给定 | `StdRng::seed_from_u64`，结果可复现 |

## 日志埋点

| target | 谁在打 | 典型事件 |
| --- | --- | --- |
| `app` | `main.rs` / `engine.rs` | 加载开始、加载完成、TUI/服务启动 |
| `model` | `engine.rs` / `model.rs` | 硬件后端、配置解析、权重读取 |
| `gen` | `engine.rs` | 开始生成、生成结束、提示词截断、上下文触顶、取消 |
| `session` | `engine.rs` | KV-Cache 重置、缓存不可复用 |
| `api` | `server.rs` | 收到请求（info）、请求内容（debug）、回复完成、推理失败 |
| `http` | `server.rs` | 请求完成（info）、请求异常（warn，含 URI + body）、请求体超限 |
| `tui` | `tui.rs` | 用户输入、内置命令、取消生成、生成失败 |

两条值得记住的诊断日志：

1. **`target: "gen"`，「提示词超出上下文预算，已丢弃最早的若干 Token」**
   字段：`truncated`、`budget`、`dropped_chars`、`retained_chars`、
   `original_input`（原始输入）、`dropped_text`（被丢掉的文本）、
   `retained_text`（真正送进模型的文本）。超长文本按头尾各 1000 字符裁剪
   （`LOG_TEXT_HEAD_TAIL`）。

2. **`target: "http"`，「请求异常」**
   字段：`method`、`uri`、`status`、`elapsed_ms`、`body`。
   客户端发了畸形报文 / 打错路径时，靠这条还原现场。
