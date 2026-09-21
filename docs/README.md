# spark_candle 文档索引

本文档按「文件 + 功能」组织，逐模块说明 `src/` 下每个文件、结构体与关键函数的职责。

> 代码里每个函数 / 结构体都带有 `///` 文档注释（`cargo doc` 可直接生成），
> 本文是它们的「人话版」说明与上下游关系梳理。

## 目录

| 文档 | 内容 |
| --- | --- |
| [01-architecture.md](./01-architecture.md) | 整体架构、数据流、并发模型、上下文预算、日志埋点 |
| [02-main.md](./02-main.md) | `src/main.rs`：程序入口与两种模式的启动流程 |
| [03-cli.md](./03-cli.md) | `src/cli.rs`：命令行参数、默认值、TLS 校验、dtype 解析 |
| [04-logging.md](./04-logging.md) | `src/logging.rs`：JSONL 日志初始化与按天滚动 |
| [05-chat.md](./05-chat.md) | `src/chat.rs`：消息结构与 Jinja 对话模板渲染 |
| [06-model.md](./06-model.md) | `src/model.rs`：Spark-X2.5 的 Candle 实现与权重下载 |
| [07-engine.md](./07-engine.md) | `src/engine.rs`：加载、KV-Cache 会话、生成槽、采样、流式步进 |
| [08-server.md](./08-server.md) | `src/server.rs`：OpenAI / Anthropic / Ollama HTTP 层 |
| [09-tui.md](./09-tui.md) | `src/tui.rs`：终端界面、按键、后台生成线程 |
| [10-http-api.md](./10-http-api.md) | HTTP 接口参考（请求/响应样例与字段说明） |

## 速查

- 想改**默认参数** → [03-cli.md](./03-cli.md)
- 想改**上下文 / max_tokens 的裁剪策略** → [07-engine.md](./07-engine.md)
- 想加**新的 HTTP 接口** → [08-server.md](./08-server.md)
- 想排查**日志看不懂** → [01-architecture.md#日志埋点](./01-architecture.md#日志埋点)
- 想接**客户端** → [10-http-api.md](./10-http-api.md)
