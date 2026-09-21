# 05 · `src/chat.rs`

负责「消息归一化之后的文本表示」与「官方对话模板渲染」。只有两个结构体、两个函数，
但它是 TUI、OpenAI、Anthropic、Ollama 四条入口的公共收敛点。

## 结构体

### `pub struct Message`

```rust
pub struct Message {
    pub role: String,     // system / user / assistant
    pub content: String,  // 纯文本
}
```

- 同时实现了 `Serialize` / `Deserialize`，可直接喂给 minijinja 的 `context!`
- 多模态的分块内容（`[{"type":"text",...}]`）在 `server.rs` 层就已折叠成纯文本
- 引擎只认这两个字段，协议差异在这里被彻底抹平

## 函数

### `pub fn apply_chat_template(...) -> Result<String>`

```rust
apply_chat_template(
    template_path: &Path,          // 本地 chat_template.jinja
    messages: &[Message],
    add_generation_prompt: bool,   // 是否在末尾加 <|Bot|>
    enable_thinking: bool,         // --thinking
) -> Result<String>
```

流程：

1. 读取 `chat_template.jinja`（由 `prepare_model_files` 下载到本地缓存）
2. 建 minijinja `Environment`，并**手动注册** `raise_exception` 函数 —— 官方模板会用到它，
   而 minijinja 没有内置
3. 渲染时注入变量：`messages`、`add_generation_prompt`、`enable_thinking`、
   `bos_token`（`<｜begin▁of▁sentence｜>`）、`eos_token`（`<｜end▁of▁sentence｜>`）
4. 返回完整提示词

失败（文件缺失 / 模板语法错误 / 渲染异常）时返回 `Err`，
`Engine::build_generation` 会捕获并降级到 `build_prompt_fallback`。

### `pub fn build_prompt_fallback(messages: &[Message]) -> String`

Jinja 不可用时的兜底，手工复刻官方模板的输出结构：

```
<｜start▁of▁sentence｜><|System|>
you are a helpful assistant.

{system 内容}
<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|User|>{user 内容}
<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>{assistant 内容}
<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>
```

- 末尾一定有 `<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>`，引导模型续写
- 不支持 `enable_thinking` 等官方模板扩展能力，但保证「能聊」
- `role` 不在 `system` / `user` / `assistant` 三者之内时静默跳过

## 常见改动

| 需求 | 改哪里 |
| --- | --- |
| 换默认 system 提示 | `build_prompt_fallback` 里的 `you are a helpful assistant.`，或用 `--system` |
| 模板变量不够用 | 在 `context!{}` 里补，并确保 `server.rs`/`engine.rs` 传得进来 |
| 模板里用了别的自定义函数 | 仿照 `raise_exception` 用 `env.add_function` 注册 |
