//! 命令行参数定义

use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

use crate::engine::DEFAULT_MAX_TOKEN;

/// 默认上下文上限（Prompt + 生成）
const DEFAULT_MAX_CONTEXT: usize = 32000;//

#[derive(Parser, Debug)]
#[command(
    name = "spark_candle",
    version,
    about = "Spark-X2.5 本地推理：TUI 交互对话 + OpenAI/Anthropic 兼容 HTTP 服务"
)]
/// 顶层命令行入口：只含一个可选子命令
///
/// 不带子命令时等价于 `spark_candle chat`，直接进入 TUI 对话。
pub struct Cli {
    /// 子命令；`None` 时按 `Command::Chat` 处理
    #[command(subcommand)]
    pub command: Option<Command>,
}

/// 可用子命令
#[derive(Subcommand, Debug)]
pub enum Command {
    /// 启动终端 TUI 多轮对话（默认子命令）
    Chat(ChatArgs),
    /// 启动兼容 OpenAI / Anthropic / Ollama 协议的 HTTP 服务
    Serve(ServeArgs),
}

/// `chat` 与 `serve` 共用的模型 / 采样 / 日志参数
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
    #[arg(long, default_value_t = DEFAULT_MAX_CONTEXT)]
    pub max_context: usize,

    /// 默认系统提示词
    #[arg(long)]
    pub system: Option<String>,

    /// 单次生成的最大 Token 数；超过上下文上限时会自动夹取
    #[arg(long, default_value_t = DEFAULT_MAX_TOKEN)]
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

/// `chat` 子命令的参数：目前只有共用参数，后续可在此扩展 TUI 专属选项
#[derive(Args, Debug, Default)]
pub struct ChatArgs {
    /// 展平的共用参数（`#[command(flatten)]` 后与子命令同级）
    #[command(flatten)]
    pub common: CommonArgs,
}

/// 手写 `Default`：TUI 不带子命令启动时（Cli::command == None）用它补齐默认值
impl Default for CommonArgs {
    fn default() -> Self {
        Self {
            model: "XHToken/Spark-X2.5-1.7B".to_string(),
            shards: 2,
            dtype: None,
            max_context: DEFAULT_MAX_CONTEXT,
            system: None,
            max_tokens: DEFAULT_MAX_TOKEN,
            temperature: 0.7,
            top_p: 0.95,
            seed: None,
            thinking: false,
            log_dir: "logs".to_string(),
            log_level: "info".to_string(),
        }
    }
}

/// `serve` 子命令的参数：共用参数 + 监听地址 / 端口 / TLS
#[derive(Args, Debug)]
pub struct ServeArgs {
    /// 展平的共用参数
    #[command(flatten)]
    pub common: CommonArgs,

    /// 监听地址
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// 监听端口
    #[arg(long, default_value_t = 8000)]
    pub port: u16,

    /// TLS 证书文件（PEM，可含完整证书链）；与 --tls-key 同时指定时启用 HTTPS
    #[arg(long, value_name = "FILE")]
    pub tls_cert: Option<PathBuf>,

    /// TLS 私钥文件（PEM，支持 PKCS#1 / PKCS#8 / SEC1）
    #[arg(long, value_name = "FILE")]
    pub tls_key: Option<PathBuf>,
}

impl ServeArgs {
    /// 校验并返回 TLS 证书 / 私钥路径；两者必须成对出现，都缺省则走明文 HTTP
    ///
    /// # 返回
    /// - `Ok(Some((cert, key)))`：两个参数都给了，启用 HTTPS
    /// - `Ok(None)`：两个参数都没给，走明文 HTTP
    /// - `Err`：只给了其中一个，属于配置错误
    ///
    /// 在加载模型之前就调用它，避免耗时的权重下载后才报错。
    pub fn tls_pair(&self) -> anyhow::Result<Option<(&PathBuf, &PathBuf)>> {
        match (&self.tls_cert, &self.tls_key) {
            (Some(cert), Some(key)) => Ok(Some((cert, key))),
            (None, None) => Ok(None),
            _ => anyhow::bail!("启用 HTTPS 需要同时指定 --tls-cert 与 --tls-key"),
        }
    }
}

impl CommonArgs {
    /// 把 `--dtype` 字符串解析成 Candle 的精度类型
    ///
    /// # 返回
    /// - `Ok(None)`：未指定，由引擎按设备自动选择（CUDA=BF16，CPU=F16）
    /// - `Ok(Some(dtype))`：显式指定的 f16 / bf16 / f32
    /// - `Err`：取值不合法
    pub fn to_dtype(&self) -> anyhow::Result<Option<candle_core::DType>> {
        match self.dtype.as_deref() {
            None => Ok(None),
            Some("f16") => Ok(Some(candle_core::DType::F16)),
            Some("bf16") => Ok(Some(candle_core::DType::BF16)),
            Some("f32") => Ok(Some(candle_core::DType::F32)),
            Some(other) => anyhow::bail!("不支持的 dtype: {}（可选 f16/bf16/f32）", other),
        }
    }

    /// 由命令行参数构造生成参数
    ///
    /// # 参数
    /// - `max_tokens`：单次请求覆盖值（HTTP 接口传入），`None` 表示用命令行默认值
    ///
    /// # 说明
    /// 生成预算不可能超过上下文上限，这里先按 `max_context - 1` 夹一次，
    /// 避免配了「8192 生成 / 4096 上下文」这类矛盾配置还跑到引擎里才被裁剪。
    pub fn gen_options(&self, max_tokens: Option<usize>) -> crate::engine::GenOptions {
        // 生成预算不可能超过上下文上限，这里先按上下文夹一次，避免运行时再被裁剪
        let max_tokens = max_tokens
            .unwrap_or(self.max_tokens)
            .min(self.max_context.saturating_sub(1).max(1));
        crate::engine::GenOptions {
            max_tokens,
            temperature: self.temperature,
            top_p: self.top_p,
            seed: self.seed,
            enable_thinking: self.thinking,
        }
    }
}
