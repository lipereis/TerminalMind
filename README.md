# TerminalMind

Personal Research Assistant CLI for AI Engineering portfolios. Queries an LLM with **strict Structured Outputs** (Pydantic), optionally grounds answers in locally ingested `.txt` / `.md` files via keyword RAG-lite, and keeps session history + markdown reports.

## Install

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Unix:
# source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
```

## Free API (Gemini)

1. Get a free key: https://aistudio.google.com/apikey
2. Put it in `.env` (already templated in `.env.example`):

```env
OPENAI_API_KEY=your-gemini-api-key
OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
OPENAI_MODEL=gemini-flash-latest
```

Uses Google’s OpenAI-compatible endpoint — same SDK, no OpenAI billing.

## Usage

```bash
terminalmind ingest ./notes.md
terminalmind search "What are the main claims?"
terminalmind history
terminalmind --data-dir ./tmp-data search "..."
terminalmind --verbose search "debug run"
```

Flow: **ingest** → **search** (structured summary / key points / follow-ups) → **history**.

Empty ingest store: search warns and falls back to LLM-only.

## Architecture

Thin Typer CLI → `ResearchAgent` → OpenAI-compatible client (`parse` + JSON-schema fallback) + `Storage` under `~/.terminalmind/` (or `--data-dir`).

Design: `docs/superpowers/specs/2026-07-23-terminalmind-design.md`

## Test

```bash
pytest -v
```
