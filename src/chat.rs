//! 对话消息结构与官方 chat_template.jinja 渲染

use anyhow::{Context, Result};
use minijinja::{context, Environment};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// 标准对话消息（同时用于 TUI、OpenAI 与 Anthropic 接口）
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// 使用下载的 jinja 模板动态拼接对话
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
