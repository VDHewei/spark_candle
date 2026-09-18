//! 终端 TUI 多轮对话界面（ratatui + crossterm）

use crate::chat::Message;
use crate::cli::ChatArgs;
use crate::engine::{Engine, GenOptions, GenResult};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use tracing::{error, info, warn};

/// 后台线程推送的事件，携带世代号以便丢弃被取消的生成任务
enum WorkerEvent {
    Delta(u64, String),
    Done(u64, std::result::Result<GenResult, String>),
}

struct ViewItem {
    role: String,
    text: String,
}

struct App {
    engine: Arc<Engine>,
    opts: GenOptions,
    history: Vec<Message>,
    view: Vec<ViewItem>,
    input: String,
    scroll: u16,
    follow: bool,
    status: String,
    busy: bool,
    /// 世代号：每次提交或取消自增，用于丢弃过期后台任务的增量
    gen: u64,
    /// 当前生成任务的中断标志：置位后后台线程会在下一个 Token 立即收尾
    cancel: Arc<AtomicBool>,
    rx: Receiver<WorkerEvent>,
    tx: Sender<WorkerEvent>,
}

const HELP: &str = "Enter 发送 · Alt+Enter 换行 · ↑/↓ 或 PgUp/PgDn 滚动 · /clear 清空 · /reset 重置缓存 · /exit 退出";

pub fn run(engine: Arc<Engine>, args: ChatArgs) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(&mut terminal, engine, args);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    engine: Arc<Engine>,
    args: ChatArgs,
) -> Result<()> {
    let (tx, rx) = channel::<WorkerEvent>();
    let mut app = App {
        engine,
        opts: args.common.gen_options(None),
        history: Vec::new(),
        view: vec![ViewItem {
            role: "系统".to_string(),
            text: format!(
                "已加载 {}，上下文上限 {}。输入 /help 查看快捷键。",
                args.common.model, args.common.max_context
            ),
        }],
        input: String::new(),
        scroll: 0,
        follow: true,
        status: HELP.to_string(),
        busy: false,
        gen: 0,
        cancel: Arc::new(AtomicBool::new(false)),
        rx,
        tx,
    };

    loop {
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match (key.code, key.modifiers) {
                    (KeyCode::Char('c'), m) | (KeyCode::Char('C'), m)
                        if m.contains(KeyModifiers::CONTROL) =>
                    {
                        // 生成中先取消当前任务，空闲时才退出
                        if app.busy {
                            cancel(&mut app);
                        } else {
                            return Ok(());
                        }
                    }
                    (KeyCode::Esc, _) => {
                        if app.busy {
                            cancel(&mut app);
                        } else {
                            return Ok(());
                        }
                    }
                    (KeyCode::Enter, m) if m.contains(KeyModifiers::ALT) => {
                        app.input.push('\n')
                    }
                    (KeyCode::Enter, _) => {
                        if submit(&mut app) {
                            return Ok(())
                        }
                    }
                    (KeyCode::Backspace, _) => {
                        app.input.pop();
                    }
                    (KeyCode::Char(c), m)
                        if !m.contains(KeyModifiers::CONTROL) && !m.contains(KeyModifiers::ALT) =>
                    {
                        app.input.push(c)
                    }
                    (KeyCode::Up, _) | (KeyCode::PageUp, _) => {
                        app.follow = false;
                        app.scroll = app.scroll.saturating_sub(1);
                    }
                    (KeyCode::Down, _) | (KeyCode::PageDown, _) => {
                        app.scroll = app.scroll.saturating_add(1);
                    }
                    _ => {}
                }
            }
        }

        // 拉取后台生成的增量（世代号不匹配的直接丢弃）
        while let Ok(ev) = app.rx.try_recv() {
            match ev {
                WorkerEvent::Delta(gen, delta) => {
                    if gen == app.gen {
                        if let Some(last) = app.view.last_mut() {
                            last.text.push_str(&delta);
                        }
                    }
                }
                WorkerEvent::Done(gen, res) => {
                    if gen != app.gen {
                        continue;
                    }
                    app.busy = false;
                    match res {
                        Ok(r) => {
                            app.history.push(Message {
                                role: "assistant".to_string(),
                                content: r.text.clone(),
                            });
                            app.status = format!(
                                "完成：{} 输入 / {} 输出 / {:.1}s · {}",
                                r.prompt_tokens,
                                r.completion_tokens,
                                r.elapsed_ms as f64 / 1000.0,
                                HELP
                            );
                        }
                        Err(e) => {
                            error!(target: "tui", error = %e, "生成失败");
                            app.status = format!("生成失败：{} · {}", e, HELP);
                        }
                    }
                }
            }
        }

        terminal.draw(|f| ui(f, &app))?;
    }
}

/// 取消当前生成：自增世代号让后台增量作废，并丢弃 KV-Cache
fn cancel(app: &mut App) {
    warn!(target: "tui", produced = app.view.last().map(|v| v.text.chars().count()).unwrap_or(0), "用户取消生成");
    app.gen += 1;
    app.busy = false;
    // 先让后台线程尽快收尾，再丢弃被污染的 KV-Cache
    app.cancel.store(true, Ordering::Relaxed);
    app.engine.reset();
    if let Some(last) = app.view.last_mut() {
        if last.role == "Spark" && last.text.is_empty() {
            app.view.pop();
        }
    }
    // 撤回刚刚加入历史的用户消息，避免会话错位
    if app.history.last().map(|m| m.role.as_str()) == Some("user") {
        app.history.pop();
    }
    app.status = format!("已取消本次生成。 · {}", HELP);
}

/// 处理回车提交，返回 true 表示退出 TUI
fn submit(app: &mut App) -> bool {
    if app.busy {
        return false;
    }
    let text = app.input.trim().to_string();
    if text.is_empty() {
        return false;
    }
    app.input.clear();

    match text.as_str() {
        "/exit" | "/quit" | "/q" => {
            info!(target: "tui", command = %text, "退出会话");
            return true;
        }
        "/clear" => {
            app.history.clear();
            app.view.clear();
            app.engine.reset();
            app.status = format!("已清空会话。 · {}", HELP);
            info!(target: "tui", command = "/clear", "清空会话历史");
            return false;
        }
        "/reset" => {
            app.engine.reset();
            app.status = format!("已重置 KV-Cache。 · {}", HELP);
            info!(target: "tui", command = "/reset", "重置 KV-Cache");
            return false;
        }
        "/help" => {
            app.view.push(ViewItem {
                role: "系统".to_string(),
                text: HELP.to_string(),
            });
            return false;
        }
        _ => {}
    }

    info!(target: "tui", role = "user", text = %text, "用户输入");
    app.history.push(Message {
        role: "user".to_string(),
        content: text.clone(),
    });
    app.view.push(ViewItem {
        role: "你".to_string(),
        text,
    });
    app.view.push(ViewItem {
        role: "Spark".to_string(),
        text: String::new(),
    });
    app.follow = true;
    app.busy = true;
    app.gen += 1;
    app.status = "生成中…（Esc / Ctrl+C 取消）".to_string();

    let cancel = Arc::new(AtomicBool::new(false));
    app.cancel = cancel.clone();

    let engine = app.engine.clone();
    let history = app.history.clone();
    let opts = app.opts;
    let tx = app.tx.clone();
    let gen = app.gen;
    thread::spawn(move || {
        let res = engine
            .generate_blocking(&history, opts, cancel, |delta| {
                let _ = tx.send(WorkerEvent::Delta(gen, delta.to_string()));
            })
            .map_err(|e| e.to_string());
        let _ = tx.send(WorkerEvent::Done(gen, res));
    });

    false
}

fn ui(f: &mut Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(f.area());

    // ---- 对话区 ----
    let block = Block::default()
        .borders(Borders::ALL)
        .title(format!(" Spark-X2.5 · {} ", app.engine.model_id()));
    let inner = block.inner(chunks[0]);
    let width = inner.width.max(1) as usize;
    let lines = build_lines(app, width);
    let total = lines.len() as u16;
    let paragraph = Paragraph::new(lines).block(block);
    let max_scroll = total.saturating_sub(inner.height);
    let scroll = if app.follow {
        max_scroll
    } else {
        app.scroll.min(max_scroll)
    };
    f.render_widget(paragraph.scroll((scroll, 0)), chunks[0]);

    // ---- 输入区 ----
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(if app.busy { " 输入（生成中） " } else { " 输入 " });
    let input_inner = input_block.inner(chunks[1]);
    let prompt = Paragraph::new(app.input.as_str())
        .wrap(Wrap { trim: false })
        .block(input_block);
    f.render_widget(prompt, chunks[1]);

    // 按「显示宽度 + 折行」推算光标位置，中文输入时才不会跑偏
    let input_width = input_inner.width.max(1) as usize;
    let (mut row, mut col) = (0usize, 0usize);
    for ch in app.input.chars() {
        if ch == '\n' {
            row += 1;
            col = 0;
            continue;
        }
        let w = char_width(ch);
        if col + w > input_width {
            row += 1;
            col = 0;
        }
        col += w;
    }
    let cursor_x = input_inner.x + col as u16;
    let cursor_y = input_inner.y + row as u16;
    if cursor_y < input_inner.y + input_inner.height && cursor_x < input_inner.x + input_inner.width
    {
        f.set_cursor_position((cursor_x, cursor_y));
    }

    // ---- 状态栏 ----
    let status_style = if app.busy {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    f.render_widget(
        Paragraph::new(Line::from(app.status.clone())).style(status_style),
        chunks[2],
    );
}

/// 把消息渲染为按给定宽度折行后的文本行（自行折行以便精确控制滚动）
fn build_lines(app: &App, width: usize) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();
    for item in &app.view {
        let (label, color) = match item.role.as_str() {
            "你" => ("你", Color::Cyan),
            "Spark" => ("Spark", Color::Green),
            _ => ("系统", Color::DarkGray),
        };
        lines.push(Line::from(Span::styled(
            format!("{}:", label),
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        let text = if item.text.is_empty() && item.role == "Spark" {
            "…".to_string()
        } else {
            item.text.clone()
        };
        for seg in text.split('\n') {
            if seg.is_empty() {
                lines.push(Line::from(""));
                continue;
            }
            for chunk in wrap_line(&format!("  {seg}"), width) {
                lines.push(Line::from(Span::styled(chunk, Style::default().fg(Color::White))));
            }
        }
    }
    lines
}

fn wrap_line(line: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_width = 0usize;
    for ch in line.chars() {
        let w = char_width(ch);
        if current_width + w > width && !current.is_empty() {
            out.push(std::mem::take(&mut current));
            current_width = 0;
        }
        current.push(ch);
        current_width += w;
    }
    out.push(current);
    out
}

/// 粗略的显示宽度：CJK / 全角算 2，其余算 1
fn char_width(c: char) -> usize {
    let u = c as u32;
    let wide = (0x1100..=0x115F).contains(&u)
        || (0x2E80..=0xA4CF).contains(&u)
        || (0xAC00..=0xD7A3).contains(&u)
        || (0xF900..=0xFAFF).contains(&u)
        || (0xFE30..=0xFE6F).contains(&u)
        || (0xFF00..=0xFF60).contains(&u)
        || (0xFFE0..=0xFFE6).contains(&u)
        || (0x20000..=0x3FFFD).contains(&u);
    if wide {
        2
    } else {
        1
    }
}
