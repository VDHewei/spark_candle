# 03 · `src/cli.rs`

用 `clap` 的 derive API 定义全部命令行参数。本文件不含业务逻辑，只做三件额外的事：
TLS 参数配对校验、dtype 解析、构造 `GenOptions`。

## 常量

| 常量 | 值 | 说明 |
| --- | --- | --- |
| `DEFAULT_MAX_CONTEXT` | `8192` | `--max-context` 默认值，同时用于 `Default for CommonArgs` |
| `DEFAULT_MAX_TOKEN`（来自 `engine`） | `8192` | `--max-tokens` 默认值 |

两者刻意共用 `engine::DEFAULT_MAX_TOKEN`，避免「clap 默认 512、`Default` 里 8192」这种
带不带子命令行为不一致的问题。

## 结构体

### `struct Cli`

顶层入口，只有一个可选子命令。

| 字段 | 说明 |
| --- | --- |
| `command: Option<Command>` | `None` 时 `main` 按 `Command::Chat` 处理 |

### `enum Command`

| 变体 | 说明 |
| --- | --- |
| `Chat(ChatArgs)` | 终端 TUI 多轮对话（默认子命令） |
| `Serve(ServeArgs)` | HTTP 服务 |

### `struct CommonArgs`

`chat` 与 `serve` 共用，通过 `#[command(flatten)]` 展平到子命令层级。

| 字段 | CLI | 默认 | 说明 |
| --- | --- | --- | --- |
| `model` | `--model` | `XHToken/Spark-X2.5-1.7B` | HF 仓库 ID |
| `shards` | `--shards` | `2` | 无 `index.json` 时的分片数 |
| `dtype` | `--dtype` | 无 | `f16` / `bf16` / `f32` |
| `max_context` | `--max-context` | `8192` | 上下文上限（Prompt + 生成） |
| `system` | `--system` | 无 | 全局 system 提示 |
| `max_tokens` | `--max-tokens` | `8192` | 默认单次生成上限 |
| `temperature` | `--temperature` | `0.7` | `0` = 贪婪 |
| `top_p` | `--top-p` | `0.95` | 核采样阈值 |
| `seed` | `--seed` | 无 | 固定种子，便于复现 |
| `thinking` | `--thinking` | `false` | 官方模板的 `enable_thinking` |
| `log_dir` | `--log-dir` | `logs` | 日志目录 |
| `log_level` | `--log-level` | `info` | 可被 `RUST_LOG` 覆盖 |

### `struct ChatArgs`

只含展平的 `common`。手写 `Default` 是为了支持「不带子命令直接启动」的场景。

### `struct ServeArgs`

| 字段 | CLI | 默认 | 说明 |
| --- | --- | --- | --- |
| `common` | — | — | 展平的共用参数 |
| `host` | `--host` | `127.0.0.1` | 监听地址 |
| `port` | `--port` | `8000` | 监听端口 |
| `tls_cert` | `--tls-cert` | 无 | PEM 证书（可含完整链） |
| `tls_key` | `--tls-key` | 无 | PEM 私钥（PKCS#1/8、SEC1） |

## 函数与方法

### `impl Default for CommonArgs`

手写实现，保证**不带子命令**启动时（`Cli::command == None`）拿到的默认值与
显式写 `chat` 完全一致。

### `ServeArgs::tls_pair(&self) -> anyhow::Result<Option<(&PathBuf, &PathBuf)>>`

| 入参组合 | 结果 |
| --- | --- |
| 两个都给了 | `Ok(Some((cert, key)))` → 启用 HTTPS |
| 两个都没给 | `Ok(None)` → 明文 HTTP |
| 只给一个 | `Err("启用 HTTPS 需要同时指定 --tls-cert 与 --tls-key")` |

`main` 在**加载模型之前**调用它，避免耗时的权重下载后才报错。

### `CommonArgs::to_dtype(&self) -> anyhow::Result<Option<candle_core::DType>>`

把 `--dtype` 字符串映射成 `DType`；`None` 表示「未指定，交给引擎按设备自动选」
（CUDA → BF16，CPU → F16）。取值非法时报错并列出可选项。

### `CommonArgs::gen_options(&self, max_tokens: Option<usize>) -> GenOptions`

构造生成参数：

- `max_tokens` 传入值优先，否则用 `--max-tokens`
- 无论哪种，都会被 `min(max_context - 1)` 夹一次 —— 生成预算不可能超过上下文上限，
  在 CLI 层就夹掉，避免「配了 8192 生成 / 4096 上下文」的矛盾配置一路跑到引擎里才被裁剪
- `seed` / `enable_thinking` 直接取自命令行

## 修改默认值的正确姿势

1. 改 `DEFAULT_MAX_CONTEXT` 或 `engine::DEFAULT_MAX_TOKEN` 常量
2. 不要只改 `#[arg(default_value_t = ...)]` —— `Default for CommonArgs` 是手写的，
   两处必须保持一致，否则带不带子命令行为不同
