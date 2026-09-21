# 07 · `src/engine.rs`

推理引擎：模型加载、KV-Cache 会话管理、生成槽、采样、可逐步推进的流式生成。
TUI 与 HTTP 服务**只**通过本模块触达模型。

---

## 数据结构

### `pub struct GenOptions`（`Copy`）

| 字段 | 说明 |
| --- | --- |
| `max_tokens` | 本次最多生成多少 Token；引擎会按剩余上下文再夹一次 |
| `temperature` | `<= 0` 走贪婪解码 |
| `top_p` | 核采样阈值，内部会夹到 [0, 1] |
| `seed` | `Some` 时可复现 |
| `enable_thinking` | 官方模板思考模式 |

`pub const DEFAULT_MAX_TOKEN: usize = 8192;` —— CLI 与 `Default` 共用同一个常量。

### `pub struct GenResult`

`text` / `prompt_tokens` / `completion_tokens` / `finish_reason` / `elapsed_ms`。
非流式接口直接把它序列化进响应。

### `pub struct ModelInfo`

给 Ollama `/api/tags`、`/api/ps`、`/api/show` 用的元信息：id、权重大小、修改时间、
精度、上下文上限、层数、头数、KV 头数、head_dim、词表大小。

### `struct Session`（私有）

- `tokens: Vec<u32>` —— 已缓存的完整 Token 序列
- `caches: Vec<Option<(Tensor, Tensor)>>` —— 每层一份 `(K, V)`

### `struct GenSlot`（私有）

`Mutex<Option<u64>>` + `Condvar` 实现的单槽锁：

- `acquire(id)` —— 被占用时阻塞等待
- `release(id)` —— **只有持有者本人**能释放，避免误放别人的锁

### `struct EngineInner`（私有，包在 `Arc` 里）

模型、分词器、配置、设备、模板路径、system 提示、`max_context`、模型短名、
权重大小与修改时间、精度字符串。全部只读。

### `pub struct Engine`（`Clone`）

```rust
inner: Arc<EngineInner>
session: Arc<Mutex<Session>>
slot: Arc<GenSlot>
next_id: Arc<AtomicU64>
```

克隆代价很低，TUI 与 HTTP 各持一份，权重只加载一次。

### `pub struct Generation`

一次生成任务的全部可变状态。关键的几个字段：

| 字段 | 说明 |
| --- | --- |
| `input_ids` | 提示词 Token + 已生成 Token 的完整序列 |
| `prompt_len` | 提示词长度（`input_ids` 前缀） |
| `new_ids` | 仅已生成部分，用于增量解码 |
| `emitted` | 已吐出的完整文本，用于算增量后缀 |
| `rng` | `seed` 固定时可复现 |
| `finish_reason` | `stop` / `length` / `cancel` |

`Drop` 时释放生成槽 —— 这是排队任务能被唤醒的唯一途径，
所以**不能**把 `Generation` 长期持有不放。

---

## `Engine` 的方法

### `pub fn load(repo_id, shard_count, dtype, max_context, system) -> Result<Arc<Engine>>`

```
prepare_model_files()
  → Device::new_cuda(0) 否则 Cpu
  → 读 config.json → SparkConfig（打 target="model" 日志）
  → Tokenizer::from_file()
  → dtype 未指定时：CUDA → BF16，CPU → F16
  → VarBuilder::from_mmaped_safetensors()
  → SparkModel::load(vb, &cfg, max_context)
  → weight_stats() 统计大小与 mtime
  → 组装 EngineInner / 空 Session / 空 GenSlot / id 发号器
```

完成后打 `target: "app"` 的「模型加载完成」（含耗时）。

### `model_id()` / `max_context()` / `model_info()`

只读访问器，供 HTTP 接口回显。

### `pub fn reset()`

清空会话（丢弃 KV-Cache）。TUI 的 `/clear`、`/reset` 与取消生成后都会调用 ——
被中断的生成会让缓存停在半截状态，必须整体丢弃。

### `pub fn generate_blocking(history, opts, cancel, on_delta) -> Result<GenResult>`

「创建任务 → 循环 `step()` 直到 `None`」的便捷封装，期间独占生成槽。
`on_delta` 只在增量非空时回调。

### `pub fn start_generation(history, opts) -> Result<Generation>`

流式接口用。等价于 `start_generation_with_cancel(..., 不可取消标志)`。

### `pub fn start_generation_with_cancel(history, opts, cancel) -> Result<Generation>`

**会阻塞**等待生成槽空出。HTTP 侧必须包在 `tokio::task::block_in_place` 里，
TUI 侧跑在独立线程里。失败时立刻 `release`，不留死槽。

### `fn build_generation(id, history, opts, cancel)`（私有）

```
前置 --system（如有）
  → apply_chat_template()    失败则降级 build_prompt_fallback()（WARN）
  → tokenizer.encode()
  → 上下文预算裁剪（见下）
  → 按 seed 建 StdRng
  → 构造 Generation
```

#### 上下文预算（重要）

```
max_ctx       = max(--max-context, 2)
prompt_budget = max_ctx - 1
若 prompt > prompt_budget：丢弃最早的 Token，并打 WARN
allowed       = max_ctx - prompt_len
max_tokens    = min(请求 max_tokens, allowed)   // 差值仅 debug 级
```

WARN 日志字段（提示词超限时）：

| 字段 | 内容 |
| --- | --- |
| `truncated` | 丢弃的 Token 数 |
| `budget` | 提示词预算 |
| `dropped_chars` / `retained_chars` | 丢弃 / 剩余文本的字符数 |
| `original_input` | **原始输入**（`format_messages`，含 system） |
| `dropped_text` | 被丢掉的那部分文本 |
| `retained_text` | 真正送进模型的提示词 |

三者都会经过 `preview_text` 裁剪（头尾各 1000 字符）。

设计取舍：**提示词优先**。旧策略「给提示词留一半上下文」会把 `max_tokens: 32000`
一律夹到 `max_ctx/2` 并刷 WARN，而这是很多前端的默认行为；现在只有提示词本身超限才告警。

---

## `Generation` 的方法

### `pub fn step(&mut self) -> Result<Option<String>>`

流式生成的核心。返回：

- `Ok(Some(delta))` —— 刚产出的增量，**可能为空串**（还没凑齐一个完整字符），调用方应继续循环
- `Ok(None)` —— 已结束，`finish_reason()` 可取原因
- `Err` —— 前向或采样失败

每步的执行顺序：

1. 已结束 → `None`
2. `cancel` 置位 → 记 WARN，`finish("cancel")`，返回 `None`
3. `produced >= max_tokens` → `finish("length")`
4. `input_ids.len() >= max_context` → 记 WARN，`finish("length")`（RoPE 缓存兜底）
5. 取会话锁，**判断缓存能否复用**（见下）
6. `forward_chunked()` 前向
7. `sample_token()` 采样
8. 命中 EOS / PAD → `finish("stop")`
9. 追加 Token，**增量解码**：整体解码 `new_ids` 后取 `emitted` 之后的后缀
   （避免半个 UTF-8 字符被提前吐出）

### KV-Cache 复用判断

以下任一成立就整段重新 prefill：

- `session.caches.len() != layers`
- `session.tokens.len() >= input_ids.len()`（等长会让待算片段为空，前向得到空序列）
- `!input_ids.starts_with(&session.tokens)`

否则只前向 `input_ids[pos_offset..]`。重新 prefill 时打 `target: "session"` 的 debug 日志。

### `fn finish(&mut self, reason)`

统一收尾并打 `target: "gen"` 的 info 日志（含 `reason`、`completion_tokens`、`output`）。

---

## 自由函数

### `fn forward_chunked(inner, caches, pos, feed) -> Result<Tensor>`

长提示词的 prefill 会被切成多段：每段长度 `min(剩余, max_ctx - pos)`，
`pos` 随之累加，**只保留最后一段的 logits**。上下文耗尽时报错并说明已缓存多少、
还剩多少待处理。

### `fn sample_token(logits, temperature, top_p, rng) -> Result<u32>`

| 条件 | 行为 |
| --- | --- |
| `temperature <= 0` | `argmax` 贪婪解码 |
| 否则 | `softmax(logits / T)` → 按概率降序累加到 `top_p` 得到候选集 → 候选集内按权重轮盘赌 |

概率总和为 0（极端数值）时退化为取候选集第一个。

### 日志辅助

| 函数 | 作用 |
| --- | --- |
| `decode_ids(tokenizer, ids)` | 把 Token 序列解码回文本（失败返回 `<解码失败: …>`） |
| `preview_text(text)` | 超长文本压成「头 1000 + 省略说明 + 尾 1000」（`pub(crate)`，`server.rs` 也用） |
| `format_messages(messages)` | `[role] content` 逐行拼接，用于记录原始输入 |
| `weight_stats(paths)` | 统计权重总大小与最晚修改时间 |

`const LOG_TEXT_HEAD_TAIL: usize = 1000;` —— 想让日志里看到更多原文就调大它。

---

## 常见改动

| 需求 | 改哪里 |
| --- | --- |
| 调整日志里保留的原文长度 | `LOG_TEXT_HEAD_TAIL` |
| 改上下文预算策略 | `build_generation` 里的 `prompt_budget` / `allowed` 计算 |
| 加 repetition penalty 等采样策略 | `sample_token` |
| 支持多会话（并发） | 把 `Session` 从单个 `Arc<Mutex<..>>` 换成按 session_id 的 `HashMap`，并放宽 `GenSlot` |
| 换默认生成长度 | `DEFAULT_MAX_TOKEN` |
