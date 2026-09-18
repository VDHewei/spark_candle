//! 命令行参数定义

use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "spark_candle",
    version,
    about = "Spark-X2.5 本地推理：TUI 交互对话 + OpenAI/Anthropic 兼容 HTTP 服务"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// 启动终端 TUI 多轮对话（默认子命令）
    Chat(ChatArgs),
    /// 启动兼容 OpenAI / Anthropic 协议的 HTTP 服务
    Serve(ServeArgs),
}

#[derive(Args, Debug, Clone)]
pub struct CommonArgs {
    /// Hugging Face 仓库 ID
    #[arg(long, default_value = "XHToken/Spark-X2.5-1.7B")]
    pub model: String,

    /// 仓库缺少 index.json 时的权重分片数
    #[arg(long, default_value_t = 2)]
    pub shards: usize,

    /// 权重精度：f16 / bf16 / f32（默认 CUDA=bf16，CPU=f16）
    #[arg(long, value_name = "DTYPE")]
    pub dtype: Option<String>,

    /// 上下文上限（Prompt + 生成）
    #[arg(long, default_value_t = 4096)]
    pub max_context: usize,

    /// 默认系统提示词
    #[arg(long)]
    pub system: Option<String>,

    /// 单次生成的最大 Token 数
    #[arg(long, default_value_t = 512)]
    pub max_tokens: usize,

    /// 采样温度，0 表示贪婪解码
    #[arg(long, default_value_t = 0.7)]
    pub temperature: f64,

    /// 核采样概率阈值
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f64,

    /// 随机种子（便于复现）
    #[arg(long)]
    pub seed: Option<u64>,

    /// 开启模型的思考模式（官方模板的 enable_thinking）
    #[arg(long, default_value_t = false)]
    pub thinking: bool,

    /// 日志目录（按天滚动，文件名为 spark.log.YYYY-MM-DD）
    #[arg(long, default_value = "logs")]
    pub log_dir: String,

    /// 日志级别：error / warn / info / debug / trace（可用 RUST_LOG 覆盖）
    #[arg(long, default_value = "info")]
    pub log_level: String,
}

#[derive(Args, Debug, Default)]
pub struct ChatArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}

impl Default for CommonArgs {
    fn default() -> Self {
        Self {
            model: "XHToken/Spark-X2.5-1.7B".to_string(),
            shards: 2,
            dtype: None,
            max_context: 4096,
            system: None,
            max_tokens: 512,
            temperature: 0.7,
            top_p: 0.95,
            seed: None,
            thinking: false,
            log_dir: "logs".to_string(),
            log_level: "info".to_string(),
        }
    }
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    #[command(flatten)]
    pub common: CommonArgs,

    /// 监听地址
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// 监听端口
    #[arg(long, default_value_t = 8000)]
    pub port: u16,
}

impl CommonArgs {
    pub fn to_dtype(&self) -> anyhow::Result<Option<candle_core::DType>> {
        match self.dtype.as_deref() {
            None => Ok(None),
            Some("f16") => Ok(Some(candle_core::DType::F16)),
            Some("bf16") => Ok(Some(candle_core::DType::BF16)),
            Some("f32") => Ok(Some(candle_core::DType::F32)),
            Some(other) => anyhow::bail!("不支持的 dtype: {}（可选 f16/bf16/f32）", other),
        }
    }

    pub fn gen_options(&self, max_tokens: Option<usize>) -> crate::engine::GenOptions {
        crate::engine::GenOptions {
            max_tokens: max_tokens.unwrap_or(self.max_tokens),
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            enable_thinking: self.thinking,
        }
    }
}
