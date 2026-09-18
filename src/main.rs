//! spark_candle：Spark-X2.5 本地推理 CLI
//!
//! · `spark_candle chat`  启动终端 TUI 多轮对话（默认）
//! · `spark_candle serve` 启动 OpenAI / Anthropic 兼容 HTTP 服务

mod chat;
mod cli;
mod engine;
mod logging;
mod model;
mod server;
mod tui;

use anyhow::Result;
use clap::Parser;
use cli::{ChatArgs, Cli, Command, CommonArgs};
use engine::Engine;
use logging::LogGuard;
use std::path::Path;
use std::sync::Arc;

/// 按共用参数初始化日志，返回值需存活到程序结束
fn init_logging(common: &CommonArgs, console: bool) -> Result<LogGuard> {
    let guard = logging::init(Path::new(&common.log_dir), &common.log_level, console)?;
    println!("【系统提示】日志文件目录: {}", common.log_dir);
    Ok(guard)
}

/// 按共用参数加载模型（TUI / HTTP 两种模式共用）
fn load_engine(common: &CommonArgs, mode: &str) -> Result<Arc<Engine>> {
    tracing::info!(
        target: "app",
        model = %common.model,
        mode,
        max_context = common.max_context,
        "开始加载模型"
    );
    Engine::load(
        &common.model,
        common.shards,
        common.to_dtype()?,
        common.max_context,
        common.system.clone(),
    )
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Command::Chat(ChatArgs::default())) {
        Command::Chat(args) => {
            // TUI 独占终端，日志只落盘不打印到控制台
            let _log_guard = init_logging(&args.common, false)?;
            tracing::info!(
                target: "app",
                log_level = %args.common.log_level,
                "启动 TUI 会话"
            );
            let engine = load_engine(&args.common, "chat")?;
            tui::run(engine, args)
        }
        Command::Serve(args) => {
            let _log_guard = init_logging(&args.common, true)?;
            tracing::info!(
                target: "app",
                host = %args.host,
                port = args.port,
                "启动 HTTP 服务"
            );
            let engine = load_engine(&args.common, "serve")?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(server::run(engine, args))
        }
    }
}
