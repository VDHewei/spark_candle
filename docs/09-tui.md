# 09 · `src/tui.rs`

基于 ratatui + crossterm 的终端多轮对话界面。

## 设计要点

| 点 | 做法 | 原因 |
| --- | --- | --- |
| 生成不阻塞界面 | 生成跑在独立线程，通过 `mpsc::channel` 回传增量 | 主线程只管收增量 + 渲染 |
| 取消后不串台 | 每个事件带**世代号**（`gen`），只接受与当前世代一致的事件 | 被取消但仍在收尾的旧任务不会污染界面 |
| 终端必定恢复 | `run()` 用 `result` 变量兜住，退出前统一反向操作 | 出错也不会把终端留在 raw mode |
| 中文光标不跑偏 | 自己按「显示宽度 + 折行」算光标位置 | CJK 占 2 列，按字节算会偏移 |

## 数据结构

### `enum WorkerEvent`

| 变体 | 载荷 | 说明 |
| --- | --- | --- |
| `Delta(u64, String)` | (世代号, 增量文本) | 流式输出 |
| `Done(u64, Result<GenResult, String>)` | (世代号, 结果) | 生成结束 |

### `struct ViewItem`

界面上的一条气泡：`role`（`你` / `Spark` / `系统`）+ `text`（流式过程中不断追加）。

### `struct App`

| 字段 | 说明 |
| --- | --- |
| `engine` | 引擎句柄 |
| `opts` | 采样参数（启动时由 `args.common.gen_options(None)` 生成） |
| `history` | 对话历史（`Vec<Message>`），提交时整体克隆给后台线程 |
| `view` | 界面气泡 |
| `input` | 输入框内容 |
| `scroll` / `follow` | 滚动位置与「是否自动贴底」 |
| `status` | 状态栏文本 |
| `busy` | 是否生成中 |
| `gen` | 世代号，每次提交或取消自增 |
| `cancel` | 当前任务的中断标志 |
| `tx` / `rx` | 与后台线程通信的通道 |

### `const HELP`

状态栏常驻的快捷键提示。

## 函数

### `pub fn run(engine: Arc<Engine>, args: ChatArgs) -> Result<()>`

```
enable_raw_mode()
  → EnterAlternateScreen
  → Terminal::new(CrosstermBackend::new(stdout))
  → event_loop(...)          // 结果存进 result，不提前 return
  → disable_raw_mode()
  → LeaveAlternateScreen
  → show_cursor()
  → result
```

### `fn event_loop(terminal, engine, args) -> Result<()>`

每轮 50ms 轮询：

1. `event::poll(50ms)` 有按键则处理（见下表），只处理 `KeyEventKind::Press`
2. `try_recv` 把所有待处理事件收干净
   - `Delta`：世代号匹配才追加到最后一个气泡
   - `Done`：世代号匹配才更新 `busy`、写历史、更新状态栏；失败时记 `target: "tui"` 的 error
3. `terminal.draw(|f| ui(f, &app))`

| 按键 | 行为 |
| --- | --- |
| `Enter` | 提交（`submit`） |
| `Alt+Enter` | 换行 |
| `Backspace` | 删一个字符 |
| 普通字符 | 追加（排除带 Ctrl/Alt 的组合） |
| `↑` / `PgUp` | 上滚，关闭自动贴底 |
| `↓` / `PgDn` | 下滚 |
| `Esc` / `Ctrl+C` | 生成中 → `cancel`；空闲 → 退出 |
| `/exit` `/quit` `/q` | 退出 |
| `/clear` | 清历史 + 清界面 + `engine.reset()` |
| `/reset` | 仅 `engine.reset()` |
| `/help` | 追加一条系统气泡 |

### `fn cancel(app: &mut App)`

顺序很关键：

```
app.gen += 1                         // 让后台增量作废
app.busy = false
app.cancel.store(true)               // 先让后台线程尽快收尾
app.engine.reset()                   // 再丢掉半截状态的 KV-Cache
移除空的「Spark」气泡
撤回刚加入历史的用户消息              // 避免会话错位
```

### `fn submit(app: &mut App) -> bool`

- `busy` 或空输入 → `false`
- 内置命令 → 处理完返回（`/exit` 返回 `true`）
- 普通文本 → 入历史、插入「你」和空的「Spark」气泡、`gen += 1`、
  新建 `cancel` 标志、起线程跑 `generate_blocking`，回调里发 `Delta`，结束发 `Done`

### `fn ui(f: &mut Frame, app: &App)`

三块区域（`Min(5)` / `Length(4)` / `Length(1)`）：

1. **对话区** —— 自建行列表 + `scroll((scroll, 0))`；`follow` 时贴底
2. **输入区** —— 带 `Wrap`，光标位置按显示宽度推算，越界则不设置
3. **状态栏** —— `busy` 时黄色，否则深灰

### `fn build_lines(app, width) -> Vec<Line<'static>>`

自己折行（而不是用 `Paragraph::wrap`）是为了拿到精确总行数，从而算出滚动上限。
`Spark` 气泡为空时显示 `…`，表示「正在生成」。

### `fn wrap_line(line, width) -> Vec<String>`

按显示宽度切行，CJK 算 2 列。

### `fn char_width(c) -> usize`

覆盖的 Unicode 区间：Hangul Jamo、CJK 部首与汉字、韩文音节、CJK 兼容表意文字、
CJK 兼容形式、全角 ASCII、全角符号、CJK 扩展 B。粗略但够用。

---

## 常见改动

| 需求 | 改哪里 |
| --- | --- |
| 加内置命令 | `submit` 的 `match text.as_str()` |
| 改配色 | `build_lines` 里的 `Color::Cyan` / `Green` / `DarkGray` |
| 改布局比例 | `ui` 里的 `Constraint` |
| 加滚动到底部快捷键 | `event_loop` 的按键分支（设 `follow = true`） |
| 界面里显示 Token 统计 | `Done` 分支里已拼进 `app.status`，可直接扩展 |
