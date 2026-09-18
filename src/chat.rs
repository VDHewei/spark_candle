//! 对话消息结构与官方 chat_template.jinja 渲染

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 标准对话消息（同时用于 TUI、OpenAI 与 Anthropic 接口）
///
/// 两种协议的消息在进入引擎前都会被归一化成这个结构，
/// 引擎只认 `role`（system / user / assistant）与 `content` 两个字段。
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    /// 消息角色：`system` / `user` / `assistant`
    pub role: String,
    /// 消息正文（纯文本；多模态的分块内容在 API 层已折叠成文本）
    pub content: String,
}

/// 使用下载的 jinja 模板动态拼接对话
///
/// 优先走模型仓库自带的 `chat_template.jinja`，以保证与官方
/// `tokenizer.apply_chat_template` 的输出完全一致。
///
/// # 参数
/// - `template_path`：本地 `chat_template.jinja` 的路径（由 `prepare_model_files` 下载）
/// - `messages`：归一化后的对话消息列表（不含 system 时由调用方自行前置）
/// - `add_generation_prompt`：是否在末尾追加 `<|Bot|>` 引导模型开始回答
/// - `enable_thinking`：官方模板的思考模式开关，对应 `--thinking`
///
/// # 返回
/// 渲染完成的完整提示词文本；模板读取或渲染失败时返回错误，
/// 调用方（`Engine::build_generation`）会降级到 `build_prompt_fallback`。
pub fn apply_chat_template(
    template_path: &Path,
    messages: &[Message],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String> {
    let template_content =
        std::fs::read_to_string(template_path).context("读取本地 chat_template.jinja 失败")?;

    let mut env = Environment::new();
    // 模板里用到的 raise_exception 不是 minijinja 内置函数，这里手动补齐
    env.add_function(
        "raise_exception",
        |msg: String| -> std::result::Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env.add_template("chat_template", &template_content)?;
    let template = env.get_template("chat_template")?;

    let prompt = template
        .render(context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
            bos_token => "<｜begin▁of▁sentence｜>",
            eos_token => "<｜end▁of▁sentence｜>",
        })
        .context("Jinja 模板动态渲染 Prompt 失败")?;

    Ok(prompt)
}

/// Jinja 渲染不可用时的兜底：Spark-X2.5 的原生对话格式
///
/// 手工复刻官方模板的输出，保证即使模板文件缺失也能正常对话
/// （只是可能缺少官方的 `enable_thinking` 等扩展能力）。
///
/// 生成的结构形如：
/// `<｜start▁of▁sentence｜><|System|>…<|User|>问题<|Bot|>`
///
/// # 参数
/// - `messages`：归一化后的对话消息列表
///
/// # 返回
/// 拼接好的提示词，末尾一定带 `<|Bot|>` 以便模型续写
pub fn build_prompt_fallback(messages: &[Message]) -> String {
    let mut prompt = String::from("<｜start▁of▁sentence｜><|System|>\nyou are a helpful assistant.");
    for m in messages {
        match m.role.as_str() {
            "system" => {
                prompt.push_str("\n\n");
                prompt.push_str(&m.content);
            }
            "user" => {
                prompt.push_str("<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>");
                prompt.push_str(&m.content);
            }
            "assistant" => {
                prompt.push_str("<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>");
                prompt.push_str(&m.content);
            }
            _ => {}
        }
    }
    prompt.push_str("<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>");
    prompt
}
