# 02 · `src/main.rs`

程序入口。职责只有三件：声明模块、解析命令行、把控制权交给 `tui::run` 或 `server::run`。

## 模块清单

```rust
mod chat;     // 消息结构 + Jinja 模板
mod cli;      // 命令行参数
mod engine;   // 推理引擎
mod logging;  // 日志
mod model;    // Candle 模型实现 + 权重下载
mod server;   // HTTP 服务
mod tui;      // 终端界面
```

全部是私有模块（本 crate 是二进制 crate，没有 lib 目标）。

## 函数

### `fn init_logging(common: &CommonArgs, console: bool) -> Result<LogGuard>`

按共用参数初始化日志，并打印一行「日志文件目录: xxx」到 stdout。

- `common`：提供 `--log-dir` / `--log-level`
- `console`：是否**同时**输出到 stdout。TUI 模式必须传 `false`——终端已被界面独占，
  往 stdout 打日志会把画面刷花
- 返回 `LogGuard`，`main` 里用 `let _log_guard = ...` 绑住，程序退出时自动 flush

### `fn load_engine(common: &CommonArgs, mode: &str) -> Result<Arc<Engine>>`

按共用参数加载模型，TUI / HTTP 共用。加载前先打一条启动日志：

```rust
tracing::info!(target: "app", model, mode, max_context, max_tokens, "开始加载模型");
```

其中 `max_tokens` 取的是 `common.gen_options(None).max_tokens`，
即**已经被上下文夹过之后**的真实生效值，便于事后核对「配了 8192 却只有 4096 上下文」这类问题。

- `mode`：`"chat"` 或 `"serve"`，仅用于日志区分

### `fn main() -> Result<()>`

```
Cli::parse()
  → cli.command.unwrap_or(Command::Chat(ChatArgs::default()))
```

| 子命令 | 行为 |
| --- | --- |
| `Chat(args)` | 日志只落盘 → 加载模型 → `tui::run(engine, args)` |
| `Serve(args)` | 先 `args.tls_pair()?` 校验 TLS → 日志同时进控制台 → 加载模型 → 建 Tokio 多线程运行时 → `rt.block_on(server::run(engine, args))` |

不带子命令时等价于 `chat`。

> 注意：`serve` 需要先构造 Tokio 运行时再 `block_on`，因为 `main` 本身不是 `#[tokio::main]`；
> 这样 TUI 模式完全不引入异步运行时开销。

## 失败路径

以下情况会从 `main` 直接返回错误并以非零退出码结束：

- 日志目录无法创建（`logging::init`）
- 模型文件下载失败 / `config.json` 解析失败 / 权重形状不符（`engine::Engine::load`）
- `--tls-cert` 与 `--tls-key` 只给了其中一个（`ServeArgs::tls_pair`）
- Tokio 运行时构建失败、端口绑定失败（`server::run`）
