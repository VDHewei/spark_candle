# 04 · `src/logging.rs`

基于 `tracing` + `tracing-subscriber` + `tracing-appender` 的日志初始化。
格式固定为 **JSONL**（一行一个 JSON 对象），按天滚动写入 `--log-dir`。

## 设计要点

| 点 | 做法 | 原因 |
| --- | --- | --- |
| 格式 | `.json().flatten_event(true)` | 自定义字段展平到顶层，便于 `jq` / 日志系统解析 |
| ANSI | `.with_ansi(false)` | 文件里不能出现颜色转义序列 |
| span | `.with_current_span(false).with_span_list(false)` | 本服务没有用 span，去掉冗余字段 |
| 写入 | `non_blocking(appender)` | 日志写入在后台线程，不阻塞推理 |
| TUI | `console = false` | stdout 被 ratatui 独占，打日志会刷花界面 |

## 结构体

### `pub struct LogGuard(Option<WorkerGuard>)`

持有 `tracing-appender` 的后台写入线程句柄。

- **必须存活到程序结束**：`main` 里 `let _log_guard = init_logging(...)?;`
- `Drop` 时丢弃内部 `WorkerGuard`，触发最后一次 flush，把缓冲区里剩余的日志落盘
- 若在 `main` 里写成 `let _ = init_logging(...)`，守卫会立刻析构，可能丢日志

## 函数

### `fn json_layer<S, W>(writer: W) -> impl Layer<S>`

构造一个 JSON 输出层。泛型参数：

- `S`：`tracing::Subscriber` 实现
- `W`：`MakeWriter`，可以是非阻塞文件 writer，也可以是 `std::io::stdout`

同一个函数被复用于「写文件」和「写控制台」两条路径，保证两边格式一致。

### `pub fn init(log_dir: &Path, level: &str, console: bool) -> Result<LogGuard>`

| 参数 | 说明 |
| --- | --- |
| `log_dir` | 日志目录，不存在时自动创建 |
| `level` | 默认过滤级别，`RUST_LOG` 存在时被覆盖 |
| `console` | 是否额外输出到 stdout |

流程：

```
create_dir_all(log_dir)
  → rolling::daily(log_dir, "spark.log.jsonl")   // logs/spark.log.jsonl.YYYY-MM-DD
  → non_blocking(appender) → (writer, guard)
  → EnvFilter::try_from_default_env() 或 EnvFilter::new(level)
  → registry().with(filter).with(json_layer(file))
  → console == true 时再叠一层 json_layer(stdout)
  → init()
```

> `init()` 进程内只能调用一次，重复调用会 panic。

## 输出样例

```json
{"timestamp":"2026-09-18T08:58:06.665892Z","level":"INFO","message":"收到请求","target":"api","api":"openai","model":"spark","messages":2,"stream":false,"max_tokens":8192,"temperature":0.7,"top_p":0.95}
{"timestamp":"2026-09-18T08:58:07.102341Z","level":"WARN","message":"请求异常","target":"http","method":"POST","uri":"/api/v1/chat/completions","status":400,"elapsed_ms":1,"body":"{\"model\":\"spark\",\"messages\":[]}"}
```

## 常用命令

```bash
# 看今天的全部 WARN 及以上
jq 'select(.level=="WARN" or .level=="ERROR")' logs/spark.log.jsonl.2026-09-18

# 只看某个 target
jq 'select(.target=="gen")' logs/spark.log.jsonl.2026-09-18

# 临时开 debug（会覆盖 --log-level）
RUST_LOG=spark_candle=debug cargo run --release -- serve
```
