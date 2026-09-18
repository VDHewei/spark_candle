//! 日志初始化：JSONL 格式（一行一条 JSON），按天滚动写入 logs/ 目录
//!
//! · 每条日志为一个完整 JSON 对象，字段被展平到顶层（timestamp/level/target/message + 自定义字段）
//! · TUI 模式只写文件（stdout 被终端界面独占，不能输出日志）
//! · serve 模式同时写文件与控制台

use anyhow::{Context, Result};
use std::path::Path;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    fmt::{self, MakeWriter},
    layer::SubscriberExt,
    registry::LookupSpan,
    util::SubscriberInitExt,
    EnvFilter,
};

/// 构造 JSONL 输出层：一条日志一行 JSON，自定义字段展平到顶层
///
/// 关闭 ANSI 与 span 信息，保证日志文件可以被 `jq` / 日志系统直接解析。
///
/// # 类型参数
/// - `S`：`tracing` 的 Subscriber 实现
/// - `W`：写入目标（文件非阻塞 writer 或 `std::io::stdout`）
fn json_layer<S, W>(writer: W) -> impl tracing_subscriber::Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
    W: for<'a> MakeWriter<'a> + 'static,
{
    fmt::layer()
        .json()
        .with_writer(writer)
        .with_ansi(false)
        .with_target(true)
        .with_current_span(false)
        .with_span_list(false)
        .flatten_event(true)
}

/// 日志守卫：必须存活到程序结束
///
/// 内部持有 `tracing_appender` 的后台写入线程句柄，
/// 一旦被丢弃会触发最后一次 flush，把缓冲区里剩余的日志落盘。
/// `main` 里用 `let _log_guard = ...` 绑定即可。
pub struct LogGuard(Option<WorkerGuard>);

/// 析构时触发后台写入线程的最后一次 flush
impl Drop for LogGuard {
    fn drop(&mut self) {
        // 丢弃 WorkerGuard 会触发后台写入线程最后一次 flush
        drop(self.0.take());
    }
}

/// 初始化全局日志（进程内只能调用一次，重复调用会 panic）
///
/// # 参数
/// - `log_dir`：日志目录，不存在时自动创建；文件名 `spark.log.jsonl.YYYY-MM-DD`
/// - `level`：默认过滤级别（error / warn / info / debug / trace），可被 `RUST_LOG` 覆盖
/// - `console`：是否额外输出到 stdout；TUI 模式必须传 `false`（终端被界面独占）
///
/// # 返回
/// `LogGuard`，需存活到程序结束以保证日志完整刷盘
pub fn init(log_dir: &Path, level: &str, console: bool) -> Result<LogGuard> {
    std::fs::create_dir_all(log_dir)
        .with_context(|| format!("创建日志目录失败: {}", log_dir.display()))?;

    // 每天一个文件：logs/spark.log.YYYY-MM-DD
    let appender = tracing_appender::rolling::daily(log_dir, "spark.log.jsonl");
    let (non_blocking, guard) = tracing_appender::non_blocking(appender);

    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level.to_string()));

    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(json_layer(non_blocking));

    if console {
        registry.with(json_layer(std::io::stdout)).init();
    } else {
        registry.init();
    }

    Ok(LogGuard(Some(guard)))
}
