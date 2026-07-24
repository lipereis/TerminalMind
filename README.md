# TerminalMind

Personal Research Assistant CLI for AI Engineering portfolios. Queries OpenAI with **strict Structured Outputs** (Pydantic), optionally grounds answers in locally ingested `.txt` / `.md` files via keyword RAG-lite, and keeps session history + markdown reports.

## Install

```bash
python -m venv .venv
# Windows Git Bash:
source .venv/Scripts/activate
# Unix:
# source .venv/bin/activate

pip install -e ".[dev]"
cp .env.example .env
# set OPENAI_API_KEY in .env
```

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

Thin Typer CLI → `ResearchAgent` → OpenAI `beta.chat.completions.parse` + `Storage` under `~/.terminalmind/` (or `--data-dir`).

Design: `docs/superpowers/specs/2026-07-23-terminalmind-design.md`

## Test

```bash
pytest -v
```
