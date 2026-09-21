# spark_candle

Local inference runtime for **Spark-X2.5** (Candle / Rust), with two front-ends:

- `spark_candle chat` — a terminal TUI for multi-turn conversation (default subcommand)
- `spark_candle serve` — an HTTP server that speaks **OpenAI Chat Completions**, **Anthropic Messages** and **Ollama** protocols

English | [简体中文](./README_CN.md) | [Full documentation](./docs/README.md)

---

## Features

- **No Python, no server-side runtime** — a single static binary, weights are memory-mapped (`safetensors`)
- **Hybrid attention** — sliding-window layers + full-attention layers, GQA, per-head output gating, partial RoPE
- **KV-Cache reuse across turns** — only the newly added tokens are prefilled on every turn
- **Streaming** — SSE (OpenAI / Anthropic) and NDJSON (Ollama)
- **HTTPS** — optional TLS via `--tls-cert` / `--tls-key` (rustls + ring)
- **Structured JSONL logs** — one JSON object per line, rotated daily, request body recorded on failures

---

## Requirements

| Item | Note |
| --- | --- |
| Rust | edition 2021, a recent stable toolchain |
| GPU (optional) | CUDA is used automatically when available, otherwise CPU |
| Disk | ~3.5 GB for the default 1.7B checkpoint (HF cache) |

Weights are fetched from Hugging Face. The code sets `HF_ENDPOINT=https://hf-mirror.com` by default;
comment that line out in `src/model.rs` if you have direct access.

---

## Build

```bash
cargo build --release
# binary: target/release/spark_candle
```

---

## Quick start

### Terminal chat (TUI)

```bash
cargo run --release -- chat
# or simply
cargo run --release
```

| Key | Action |
| --- | --- |
| `Enter` | send |
| `Alt+Enter` | newline |
| `↑` / `↓` / `PgUp` / `PgDn` | scroll the transcript |
| `Esc` / `Ctrl+C` | cancel the running generation (or exit when idle) |
| `/clear` | clear history **and** KV-Cache |
| `/reset` | reset KV-Cache only |
| `/help` | show the key bindings |
| `/exit` `/quit` `/q` | quit |

### HTTP server

```bash
cargo run --release -- serve --host 0.0.0.0 --port 8000
```

---

## Command-line reference

Shared options (`chat` and `serve`):

| Option | Default | Description |
| --- | --- | --- |
| `--model` | `XHToken/Spark-X2.5-1.7B` | Hugging Face repo id |
| `--shards` | `2` | shard count used when the repo has no `index.json` |
| `--dtype` | auto (`bf16` on CUDA, `f16` on CPU) | `f16` / `bf16` / `f32` |
| `--max-context` | `8192` | context ceiling (prompt + completion) |
| `--max-tokens` | `8192` | default max generated tokens per request |
| `--temperature` | `0.7` | `0` = greedy decoding |
| `--top-p` | `0.95` | nucleus sampling threshold |
| `--seed` | — | fixed seed for reproducible sampling |
| `--thinking` | `false` | enable the official template's thinking mode |
| `--system` | — | global system prompt, prepended to every request |
| `--log-dir` | `logs` | log directory (daily rolling) |
| `--log-level` | `info` | `error` / `warn` / `info` / `debug` / `trace`; `RUST_LOG` wins |

`serve`-only options:

| Option | Default | Description |
| --- | --- | --- |
| `--host` | `127.0.0.1` | listen address |
| `--port` | `8000` | listen port |
| `--tls-cert` | — | PEM certificate (enables HTTPS together with `--tls-key`) |
| `--tls-key` | — | PEM private key (PKCS#1 / PKCS#8 / SEC1) |

---

## HTTP endpoints

| Method | Path | Protocol | Streaming |
| --- | --- | --- | --- |
| GET | `/api/health` | health probe | – |
| GET | `/api/v1/models` | OpenAI | – |
| POST | `/api/v1/chat/completions` | OpenAI Chat Completions | SSE (`stream: true`) |
| POST | `/api/v1/messages` | Anthropic Messages | SSE (`stream: true`) |
| GET | `/` | Ollama liveness (`Ollama is running`) | – |
| GET | `/api/version` | Ollama | – |
| GET | `/api/tags`, `/api/ps` | Ollama | – |
| POST | `/api/show` | Ollama | – |
| POST | `/api/chat` | Ollama | NDJSON (default on) |
| POST | `/api/generate` | Ollama | NDJSON (default on) |
| POST | `/api/embeddings` | Ollama | 501, not supported |

### Examples

```bash
# OpenAI, non-streaming
curl http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"spark","messages":[{"role":"user","content":"Hello"}],"max_tokens":512}'

# OpenAI, streaming
curl -N http://127.0.0.1:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":true}'

# Anthropic
curl http://127.0.0.1:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"max_tokens":512,"system":"Be terse.","messages":[{"role":"user","content":"Hi"}]}'

# Ollama
curl -N http://127.0.0.1:8000/api/chat \
  -d '{"model":"spark","messages":[{"role":"user","content":"Hi"}]}'
```

Any OpenAI-compatible client works — just point `base_url` at
`http://127.0.0.1:8000/api/v1` (OpenAI) or `/api` (Ollama).

---

## Notes & limitations

- **One generation at a time.** All requests share a single KV-Cache session, so a
  generation slot serialises concurrent requests; they are queued, not rejected.
- **Prompt wins the budget.** If `prompt + max_tokens` exceeds `--max-context`, the
  prompt is truncated from the front (a `WARN` is logged with the dropped and retained
  text) and everything left over is given to generation. A too-large `max_tokens` is
  clamped silently (`debug` level).
- **Request bodies larger than 2 MB** are rejected with `413`.
- `cancel` is reported as `stop` / `end_turn` / `stop` on the wire (protocols have no
  "cancelled" state).

---

## Logging

JSONL, one object per line, rotated daily: `logs/spark.log.jsonl.YYYY-MM-DD`
(`--log-dir`). Custom fields are flattened to the top level, e.g.

```json
{"timestamp":"2026-09-18T08:58:06.665892Z","level":"INFO","message":"收到请求","target":"api","api":"openai","model":"spark","messages":2,"stream":false,"max_tokens":8192,"temperature":0.7,"top_p":0.95}
```

Useful `target`s: `app`, `model`, `gen`, `session`, `api`, `http`, `tui`.

```bash
RUST_LOG=debug cargo run --release -- serve   # overrides --log-level
```

---

## Project layout

```
src/
  main.rs     entry point, dispatches to chat / serve
  cli.rs      clap argument definitions
  logging.rs  JSONL tracing subscriber (daily rolling)
  chat.rs     Message struct + Jinja chat template rendering
  model.rs    Spark-X2.5 Candle implementation + weight download
  engine.rs   model loading, KV-Cache session, sampling, streaming generation
  server.rs   OpenAI / Anthropic / Ollama HTTP layer (axum)
  tui.rs      terminal UI (ratatui + crossterm)
docs/         per-file and per-feature documentation
```

---

## Documentation

See [docs/README.md](./docs/README.md) for the full documentation:

- [Architecture](./docs/01-architecture.md)
- [main.rs](./docs/02-main.md) · [cli.rs](./docs/03-cli.md) · [logging.rs](./docs/04-logging.md)
- [chat.rs](./docs/05-chat.md) · [model.rs](./docs/06-model.md) · [engine.rs](./docs/07-engine.md)
- [server.rs](./docs/08-server.md) · [tui.rs](./docs/09-tui.md)
- [HTTP API reference](./docs/10-http-api.md)

---

## License

Check the model repository (`XHToken/Spark-X2.5-1.7B`) for model weight licensing.
