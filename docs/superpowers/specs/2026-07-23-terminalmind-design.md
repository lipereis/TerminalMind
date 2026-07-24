# TerminalMind Design Spec

**Date:** 2026-07-23  
**Status:** Approved (pending user review of this file)  
**Product:** Personal Research Assistant CLI for an AI Engineering portfolio

## Goal

Build a production-minded Python CLI that:

1. Accepts research queries and returns **strict OpenAI Structured Outputs** validated by Pydantic.
2. Optionally grounds answers in locally **ingested** `.txt` / `.md` files via chunk + keyword retrieval (RAG-lite).
3. Persists sessions and exports markdown reports.
4. Demonstrates modular design, Loguru logging, Rich UX, and Pytest coverage without live API calls in unit tests.

## Non-goals (v1)

- Live web search APIs (Tavily/Bing/etc.)
- Vector embeddings / vector DB
- Multi-turn chat sessions in the CLI
- CI pipelines (can add later)
- Binary / PDF ingest

## Decisions locked

| Topic | Choice |
|-------|--------|
| Research mode | LLM + ingested files; LLM-only fallback with Rich warning if store empty |
| Data location | `~/.terminalmind/` default; global `--data-dir` override |
| Retrieval | Chunk + naive keyword overlap; top chunks until `max_context_chars` |
| Architecture | Thin Typer CLI + fat `ResearchAgent` |
| Ingest formats | `.txt` and `.md` only |
| Structured output | OpenAI `beta.chat.completions.parse` with `response_format=ResearchAnswer` |

## Architecture

```
CLI (Typer + Rich)
        │
        ▼
ResearchAgent          # orchestration
  ├── retrieval        # chunk + keyword score (private helpers)
  ├── OpenAI client    # structured parse → ResearchAnswer
  └── Storage          # filesystem under data_dir
```

### Package layout

```
terminalmind/
├── __init__.py
├── main.py              # Typer app + commands
├── config.py            # Pydantic Settings
├── core/
│   ├── __init__.py
│   ├── agent.py         # ResearchAgent
│   └── schemas.py       # Pydantic models
├── utils/
│   ├── __init__.py
│   └── storage.py       # FS persistence + exporters
└── tests/
    ├── __init__.py
    └── test_agent.py
```

Also ship: `pyproject.toml`, `.env.example`, `README.md`, `.gitignore`.

Console entrypoint: `terminalmind` → `terminalmind.main:app`.

## Components

### `config.Settings` (Pydantic Settings)

- `openai_api_key: str` (from `OPENAI_API_KEY`)
- `openai_model: str` = `"gpt-4o-mini"`
- `data_dir: Path` = `~/.terminalmind` (overridable)
- `log_level: str` = `"INFO"`
- `max_context_chars: int` = `12000`
- Chunk defaults: ~800 chars, ~100 overlap (constants or settings fields)

### Schemas (`core/schemas.py`)

**LLM output (strict):**

```python
class ResearchAnswer(BaseModel):
    summary: str
    key_points: list[str]
    follow_ups: list[str]
```

**Internal (storage / agent, not model response_format):**

- `IngestRecord` — id, source_path, stored_path, content_hash, ingested_at, char_count
- `HistoryEntry` — id, query, answer (`ResearchAnswer`), created_at, used_ingest: bool, chunk_ids: list[str]
- `Chunk` — id, source_id, text, index

### `ResearchAgent`

Owns orchestration:

1. Load settings / ensure data dirs
2. Ingest file (delegate persistence to Storage)
3. Retrieve relevant chunks for a query
4. Call OpenAI with structured parse
5. Persist history + report

### `Storage` (`utils/storage.py`)

Filesystem only. No LLM calls.

Data dir layout:

```
<data_dir>/
├── history.jsonl
├── ingest/
│   ├── files/           # copied sources
│   ├── chunks.jsonl
│   └── manifest.json
├── reports/
│   └── <session-id>.md
└── logs/
    └── terminalmind.log
```

## CLI commands

Global options: `--data-dir`, `--verbose` (DEBUG logging).

| Command | Behavior |
|---------|----------|
| `terminalmind search "query"` | Retrieve (or warn LLM-only) → structured answer → Rich panels → append history → write markdown report |
| `terminalmind ingest "path"` | Validate `.txt`/`.md` → hash/idempotent skip → copy → chunk → update manifest |
| `terminalmind history` | Rich table: Time, Query, Summary snippet, Context (Y/N) — newest first |

## Data flows

### ingest

1. Validate path exists and suffix is `.txt` or `.md`.
2. Compute SHA-256; if hash already in `manifest.json`, report already ingested and exit 0.
3. Copy into `ingest/files/`.
4. Split into chunks (~800 chars, ~100 overlap; prefer paragraph boundaries when possible).
5. Append chunks to `chunks.jsonl`; update `manifest.json`.
6. Loguru info + Rich success.

### search

1. Load chunks. If none: Rich warning; proceed LLM-only (`used_ingest=False`).
2. Else: tokenize query and chunk text as lowercase alphanumeric word tokens (`[a-z0-9]+`); score by overlap count (ties broken by earlier chunk index); select top-k until `max_context_chars`.
3. Build messages with optional context block.
4. `client.beta.chat.completions.parse(..., response_format=ResearchAnswer)`.
5. Render Summary panel, key-point bullets, follow-up bullets via Rich.
6. Append `HistoryEntry` to `history.jsonl`; write `reports/<id>.md`.
7. Exit 0.

### history

1. Read `history.jsonl` (skip corrupt lines with warn).
2. Sort newest-first; render Rich Table.

## Errors & logging

- Loguru to stderr and rotating file at `data_dir/logs/terminalmind.log`.
- Missing `OPENAI_API_KEY` → exit 1 with clear `.env` guidance.
- Bad ingest path / unsupported type → exit 1, no traceback to user.
- OpenAI auth / rate limit / timeout → catch SDK exceptions; log full exception; Rich red message; exit 1.
- Structured parse / validation failure → log; tell user model returned invalid shape; exit 1.
- Corrupt JSONL/manifest lines → warn, skip or backup-rename and re-init empty when file wholly unreadable.
- No bare `except:`.

## Testing

- Pytest with `tmp_path`; no live OpenAI in unit tests.
- Mock OpenAI client; assert `ResearchAnswer` parsing / agent return shape.
- Chunk + keyword scoring: fixture text → expected top chunk.
- Ingest idempotency: same content twice → single manifest entry.
- History append + load.
- Empty ingest → `used_ingest=False` path.
- Settings: env / `--data-dir` override.
- `pyproject.toml` configures `[tool.pytest.ini_options]` with `testpaths = ["tests"]`.

## Dependencies (`pyproject.toml`)

Runtime: `typer`, `rich`, `pydantic`, `pydantic-settings`, `openai`, `loguru`, `python-dotenv`.

Dev: `pytest`, (optional) `pytest-cov`.

Python: `>=3.11`.

## Success criteria

- Three commands work end-to-end with a real key (manual smoke).
- Unit tests pass without network.
- Structured responses always match `ResearchAnswer`.
- Portfolio README explains ingest → search → history loop in a few steps.
