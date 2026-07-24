# TerminalMind Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-minded Personal Research Assistant CLI (`terminalmind`) with Typer/Rich, Pydantic structured OpenAI outputs, local ingest + keyword RAG-lite, Loguru logging, and Pytest coverage without live API calls.

**Architecture:** Thin Typer CLI delegates to `ResearchAgent`, which orchestrates keyword chunk retrieval, OpenAI `beta.chat.completions.parse` into `ResearchAnswer`, and filesystem persistence via `Storage` under `~/.terminalmind/` (or `--data-dir`).

**Tech Stack:** Python 3.11+, Typer, Rich, Pydantic v2, pydantic-settings, OpenAI SDK, Loguru, python-dotenv, Pytest.

## Global Constraints

- Python `>=3.11`
- Package layout exactly under `terminalmind/` as in the design spec
- LLM structured output must use `response_format=ResearchAnswer` via `client.beta.chat.completions.parse`
- Ingest accepts only `.txt` and `.md`
- Default data dir: `~/.terminalmind`; override with `--data-dir`
- Empty ingest store: LLM-only search with Rich warning (`used_ingest=False`)
- Tokenization: lowercase alphanumeric word tokens via `[a-z0-9]+`
- Chunking: ~800 chars, ~100 overlap, prefer paragraph boundaries
- No live OpenAI calls in unit tests — mock the client
- No bare `except:`
- Pytest `testpaths = ["terminalmind/tests"]` (tests live inside the package per spec layout)
- Spec reference: `docs/superpowers/specs/2026-07-23-terminalmind-design.md`

---

## File Structure

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata, deps, console script, pytest config |
| `.gitignore` | Ignore `.env`, `__pycache__`, `.venv`, dist, etc. |
| `.env.example` | Document `OPENAI_API_KEY` and optional overrides |
| `README.md` | Portfolio-facing install + ingest → search → history |
| `terminalmind/__init__.py` | Package version |
| `terminalmind/config.py` | `Settings` (pydantic-settings) |
| `terminalmind/main.py` | Typer app: `search`, `ingest`, `history` + global options |
| `terminalmind/core/__init__.py` | Re-exports (optional, keep thin) |
| `terminalmind/core/schemas.py` | `ResearchAnswer`, `Chunk`, `IngestRecord`, `HistoryEntry` |
| `terminalmind/core/agent.py` | `ResearchAgent`: chunk, retrieve, ingest, search |
| `terminalmind/utils/__init__.py` | Empty or thin |
| `terminalmind/utils/storage.py` | Data-dir layout, manifest, chunks, history, reports, logging setup |
| `terminalmind/tests/__init__.py` | Test package marker |
| `terminalmind/tests/test_agent.py` | Unit tests for schemas helpers, storage, agent, CLI |

---

### Task 1: Scaffold project + schemas

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `terminalmind/__init__.py`
- Create: `terminalmind/core/__init__.py`
- Create: `terminalmind/utils/__init__.py`
- Create: `terminalmind/core/schemas.py`
- Create: `terminalmind/tests/__init__.py`
- Create: `terminalmind/tests/test_agent.py`
- Test: `terminalmind/tests/test_agent.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ResearchAnswer(summary: str, key_points: list[str], follow_ups: list[str])`
  - `Chunk(id: str, source_id: str, text: str, index: int)`
  - `IngestRecord(id: str, source_path: str, stored_path: str, content_hash: str, ingested_at: datetime, char_count: int)`
  - `HistoryEntry(id: str, query: str, answer: ResearchAnswer, created_at: datetime, used_ingest: bool, chunk_ids: list[str])`

- [ ] **Step 1: Write the failing test**

Create `terminalmind/tests/test_agent.py`:

```python
from datetime import datetime, timezone

from terminalmind.core.schemas import (
    Chunk,
    HistoryEntry,
    IngestRecord,
    ResearchAnswer,
)


def test_research_answer_roundtrip() -> None:
    raw = {
        "summary": "Overview",
        "key_points": ["a", "b"],
        "follow_ups": ["what next?"],
    }
    answer = ResearchAnswer.model_validate(raw)
    assert answer.summary == "Overview"
    assert answer.key_points == ["a", "b"]
    assert answer.follow_ups == ["what next?"]


def test_history_entry_embeds_answer() -> None:
    answer = ResearchAnswer(
        summary="S",
        key_points=["k"],
        follow_ups=["f"],
    )
    entry = HistoryEntry(
        id="h1",
        query="q",
        answer=answer,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        used_ingest=False,
        chunk_ids=[],
    )
    assert entry.answer.summary == "S"
    assert entry.used_ingest is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pip install -e ".[dev]" 2>/dev/null; pytest terminalmind/tests/test_agent.py::test_research_answer_roundtrip -v`

Expected: FAIL (package / module not found) — if install fails because `pyproject.toml` missing, that is the expected failure mode. Create files in next steps then re-run.

If the environment has no editable install yet, create `pyproject.toml` first in Step 3, then confirm import error on `schemas` before implementing schemas — or implement scaffold + schemas together so the first green cycle is schemas.

- [ ] **Step 3: Write minimal implementation**

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "terminalmind"
version = "0.1.0"
description = "Personal Research Assistant CLI with structured OpenAI outputs"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "typer>=0.12",
  "rich>=13",
  "pydantic>=2",
  "pydantic-settings>=2",
  "openai>=1.40",
  "loguru>=0.7",
  "python-dotenv>=1.0",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov>=5"]

[project.scripts]
terminalmind = "terminalmind.main:app"

[tool.setuptools.packages.find]
include = ["terminalmind*"]

[tool.pytest.ini_options]
testpaths = ["terminalmind/tests"]
pythonpath = ["."]
```

`.gitignore`:

```
.env
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
.coverage
htmlcov/
```

`.env.example`:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
# TERMINALMIND_DATA_DIR=/custom/path
# TERMINALMIND_LOG_LEVEL=INFO
# TERMINALMIND_MAX_CONTEXT_CHARS=12000
```

`terminalmind/__init__.py`:

```python
__version__ = "0.1.0"
```

`terminalmind/core/__init__.py` and `terminalmind/utils/__init__.py` and `terminalmind/tests/__init__.py`: empty files (or docstring only).

`terminalmind/core/schemas.py`:

```python
from datetime import datetime

from pydantic import BaseModel, Field


class ResearchAnswer(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    source_id: str
    text: str
    index: int


class IngestRecord(BaseModel):
    id: str
    source_path: str
    stored_path: str
    content_hash: str
    ingested_at: datetime
    char_count: int


class HistoryEntry(BaseModel):
    id: str
    query: str
    answer: ResearchAnswer
    created_at: datetime
    used_ingest: bool
    chunk_ids: list[str] = Field(default_factory=list)
```

Also add a minimal stub `README.md` so setuptools `readme` resolves:

```markdown
# TerminalMind

Personal Research Assistant CLI. See full docs after Task 7.
```

- [ ] **Step 4: Install and run tests**

Run:

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash; use .venv/bin/activate on Unix
pip install -e ".[dev]"
pytest terminalmind/tests/test_agent.py -v
```

Expected: PASS for both schema tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore .env.example README.md terminalmind
git commit -m "feat: scaffold package and Pydantic schemas"
```

---

### Task 2: Settings (config)

**Files:**
- Create: `terminalmind/config.py`
- Modify: `terminalmind/tests/test_agent.py`
- Test: `terminalmind/tests/test_agent.py`

**Interfaces:**
- Consumes: pydantic-settings, dotenv conventions
- Produces:
  - `class Settings(BaseSettings)` with fields:
    - `openai_api_key: str`
    - `openai_model: str = "gpt-4o-mini"`
    - `data_dir: Path` default `Path.home() / ".terminalmind"`
    - `log_level: str = "INFO"`
    - `max_context_chars: int = 12000`
    - `chunk_size: int = 800`
    - `chunk_overlap: int = 100`
  - Env prefix / names: `OPENAI_API_KEY`, `OPENAI_MODEL`, `TERMINALMIND_DATA_DIR`, `TERMINALMIND_LOG_LEVEL`, `TERMINALMIND_MAX_CONTEXT_CHARS` (map via `Field(validation_alias=...)` or `SettingsConfigDict(env_file=".env")` with explicit aliases)
  - `def get_settings(**overrides: object) -> Settings`

- [ ] **Step 1: Write the failing test**

Append to `terminalmind/tests/test_agent.py`:

```python
from pathlib import Path

from terminalmind.config import Settings, get_settings


def test_settings_data_dir_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TERMINALMIND_DATA_DIR", str(tmp_path))
    settings = get_settings()
    assert settings.openai_api_key == "test-key"
    assert settings.data_dir == tmp_path
    assert settings.openai_model == "gpt-4o-mini"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest terminalmind/tests/test_agent.py::test_settings_data_dir_override -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'terminalmind.config'`

- [ ] **Step 3: Write minimal implementation**

`terminalmind/config.py`:

```python
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")
    data_dir: Path = Field(
        default_factory=lambda: Path.home() / ".terminalmind",
        validation_alias="TERMINALMIND_DATA_DIR",
    )
    log_level: str = Field(default="INFO", validation_alias="TERMINALMIND_LOG_LEVEL")
    max_context_chars: int = Field(
        default=12000,
        validation_alias="TERMINALMIND_MAX_CONTEXT_CHARS",
    )
    chunk_size: int = 800
    chunk_overlap: int = 100


def get_settings(**overrides: object) -> Settings:
    """Build settings; optional overrides used by CLI (`data_dir`, `log_level`)."""
    settings = Settings()
    return settings.model_copy(update=overrides) if overrides else settings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest terminalmind/tests/test_agent.py::test_settings_data_dir_override -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add terminalmind/config.py terminalmind/tests/test_agent.py
git commit -m "feat: add pydantic Settings for API key and data dir"
```

---

### Task 3: Storage — dirs, chunking, ingest, history, reports

**Files:**
- Create: `terminalmind/utils/storage.py`
- Modify: `terminalmind/tests/test_agent.py`
- Test: `terminalmind/tests/test_agent.py`

**Interfaces:**
- Consumes: `Settings`, `Chunk`, `IngestRecord`, `HistoryEntry`, `ResearchAnswer`
- Produces:
  - `class Storage:`
    - `__init__(self, data_dir: Path) -> None`
    - `ensure_layout(self) -> None`
    - `setup_logging(self, level: str) -> None`
    - `content_hash(self, text: str) -> str`  # sha256 hex
    - `chunk_text(self, text: str, source_id: str, chunk_size: int, overlap: int) -> list[Chunk]`
    - `ingest_file(self, path: Path, chunk_size: int, overlap: int) -> tuple[IngestRecord, bool]`  
      # returns `(record, created)` where `created=False` means idempotent skip
    - `load_chunks(self) -> list[Chunk]`
    - `list_ingest_records(self) -> list[IngestRecord]`
    - `append_history(self, entry: HistoryEntry) -> None`
    - `load_history(self) -> list[HistoryEntry]`  # newest first; skip bad lines
    - `write_report(self, entry: HistoryEntry) -> Path`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from terminalmind.utils.storage import Storage


def test_chunk_text_respects_size(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    text = ("paragraph one. " * 40) + "\n\n" + ("paragraph two. " * 40)
    chunks = storage.chunk_text(text, source_id="s1", chunk_size=800, overlap=100)
    assert len(chunks) >= 2
    assert all(c.source_id == "s1" for c in chunks)
    assert chunks[0].index == 0


def test_ingest_idempotent(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.ensure_layout()
    src = tmp_path / "doc.md"
    src.write_text("# Title\n\nHello research world.\n", encoding="utf-8")
    rec1, created1 = storage.ingest_file(src, chunk_size=800, overlap=100)
    rec2, created2 = storage.ingest_file(src, chunk_size=800, overlap=100)
    assert created1 is True
    assert created2 is False
    assert rec1.content_hash == rec2.content_hash
    assert len(storage.list_ingest_records()) == 1
    assert len(storage.load_chunks()) >= 1


def test_history_append_and_load_newest_first(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.ensure_layout()
    a1 = ResearchAnswer(summary="first", key_points=[], follow_ups=[])
    a2 = ResearchAnswer(summary="second", key_points=[], follow_ups=[])
    e1 = HistoryEntry(
        id="1",
        query="q1",
        answer=a1,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        used_ingest=False,
        chunk_ids=[],
    )
    e2 = HistoryEntry(
        id="2",
        query="q2",
        answer=a2,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        used_ingest=True,
        chunk_ids=["c1"],
    )
    storage.append_history(e1)
    storage.append_history(e2)
    loaded = storage.load_history()
    assert loaded[0].id == "2"
    assert loaded[1].id == "1"
    report = storage.write_report(e2)
    assert report.exists()
    body = report.read_text(encoding="utf-8")
    assert "second" in body
    assert "q2" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest terminalmind/tests/test_agent.py::test_chunk_text_respects_size terminalmind/tests/test_agent.py::test_ingest_idempotent terminalmind/tests/test_agent.py::test_history_append_and_load_newest_first -v`

Expected: FAIL — `Storage` not found

- [ ] **Step 3: Write minimal implementation**

Create `terminalmind/utils/storage.py` with this full implementation:

```python
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from terminalmind.core.schemas import Chunk, HistoryEntry, IngestRecord

ALLOWED_SUFFIXES = {".txt", ".md"}


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.ingest_dir = data_dir / "ingest"
        self.files_dir = self.ingest_dir / "files"
        self.chunks_path = self.ingest_dir / "chunks.jsonl"
        self.manifest_path = self.ingest_dir / "manifest.json"
        self.history_path = data_dir / "history.jsonl"
        self.reports_dir = data_dir / "reports"
        self.logs_dir = data_dir / "logs"

    def ensure_layout(self) -> None:
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self.manifest_path.write_text("[]", encoding="utf-8")
        if not self.chunks_path.exists():
            self.chunks_path.write_text("", encoding="utf-8")
        if not self.history_path.exists():
            self.history_path.write_text("", encoding="utf-8")

    def setup_logging(self, level: str) -> None:
        logger.remove()
        logger.add(sys.stderr, level=level)
        logger.add(
            self.logs_dir / "terminalmind.log",
            rotation="1 MB",
            retention="7 days",
            level=level,
        )

    def content_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def chunk_text(
        self,
        text: str,
        source_id: str,
        chunk_size: int,
        overlap: int,
    ) -> list[Chunk]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
        units: list[str] = []
        for para in paragraphs:
            if len(para) <= chunk_size:
                units.append(para)
            else:
                start = 0
                while start < len(para):
                    units.append(para[start : start + chunk_size])
                    start += max(chunk_size - overlap, 1)

        chunks: list[Chunk] = []
        buffer = ""
        for unit in units:
            candidate = f"{buffer}\n\n{unit}".strip() if buffer else unit
            if buffer and len(candidate) > chunk_size:
                chunk_id = f"{source_id}:{len(chunks)}"
                chunks.append(
                    Chunk(id=chunk_id, source_id=source_id, text=buffer, index=len(chunks))
                )
                if overlap > 0 and len(buffer) > overlap:
                    buffer = (buffer[-overlap:] + "\n\n" + unit).strip()
                else:
                    buffer = unit
                if len(buffer) > chunk_size:
                    start = 0
                    while start < len(buffer):
                        piece = buffer[start : start + chunk_size]
                        chunks.append(
                            Chunk(
                                id=f"{source_id}:{len(chunks)}",
                                source_id=source_id,
                                text=piece,
                                index=len(chunks),
                            )
                        )
                        start += max(chunk_size - overlap, 1)
                    buffer = ""
            else:
                buffer = candidate

        if buffer:
            chunks.append(
                Chunk(
                    id=f"{source_id}:{len(chunks)}",
                    source_id=source_id,
                    text=buffer,
                    index=len(chunks),
                )
            )
        return chunks

    def list_ingest_records(self) -> list[IngestRecord]:
        try:
            raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt manifest; backing up and re-init: {}", exc)
            backup = self.manifest_path.with_suffix(".json.bak")
            if self.manifest_path.exists():
                self.manifest_path.replace(backup)
            self.manifest_path.write_text("[]", encoding="utf-8")
            return []
        records: list[IngestRecord] = []
        for item in raw:
            try:
                records.append(IngestRecord.model_validate(item))
            except Exception as exc:  # noqa: BLE001 — skip bad rows only
                logger.warning("Skipping bad manifest row: {}", exc)
        return records

    def _save_manifest(self, records: list[IngestRecord]) -> None:
        payload = [r.model_dump(mode="json") for r in records]
        self.manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def ingest_file(
        self,
        path: Path,
        chunk_size: int,
        overlap: int,
    ) -> tuple[IngestRecord, bool]:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {path.suffix} (use .txt or .md)")

        text = path.read_text(encoding="utf-8")
        digest = self.content_hash(text)
        existing = self.list_ingest_records()
        for record in existing:
            if record.content_hash == digest:
                return record, False

        source_id = str(uuid.uuid4())
        stored_name = f"{source_id}{path.suffix.lower()}"
        stored_path = self.files_dir / stored_name
        shutil.copy2(path, stored_path)

        chunks = self.chunk_text(text, source_id=source_id, chunk_size=chunk_size, overlap=overlap)
        with self.chunks_path.open("a", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")

        record = IngestRecord(
            id=source_id,
            source_path=str(path.resolve()),
            stored_path=str(stored_path),
            content_hash=digest,
            ingested_at=datetime.now(timezone.utc),
            char_count=len(text),
        )
        existing.append(record)
        self._save_manifest(existing)
        logger.info("Ingested {} chunks from {}", len(chunks), path)
        return record, True

    def load_chunks(self) -> list[Chunk]:
        if not self.chunks_path.exists():
            return []
        chunks: list[Chunk] = []
        for line_no, line in enumerate(self.chunks_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                chunks.append(Chunk.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping corrupt chunk line {}: {}", line_no, exc)
        return chunks

    def append_history(self, entry: HistoryEntry) -> None:
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(entry.model_dump_json() + "\n")

    def load_history(self) -> list[HistoryEntry]:
        if not self.history_path.exists():
            return []
        entries: list[HistoryEntry] = []
        for line_no, line in enumerate(self.history_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                entries.append(HistoryEntry.model_validate_json(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping corrupt history line {}: {}", line_no, exc)
        entries.sort(key=lambda e: e.created_at, reverse=True)
        return entries

    def write_report(self, entry: HistoryEntry) -> Path:
        path = self.reports_dir / f"{entry.id}.md"
        key_points = "\n".join(f"- {p}" for p in entry.answer.key_points) or "- (none)"
        follow_ups = "\n".join(f"- {p}" for p in entry.answer.follow_ups) or "- (none)"
        body = (
            f"# Research Report\n\n"
            f"- **ID:** {entry.id}\n"
            f"- **Query:** {entry.query}\n"
            f"- **Created:** {entry.created_at.isoformat()}\n"
            f"- **Used ingest:** {entry.used_ingest}\n\n"
            f"## Summary\n\n{entry.answer.summary}\n\n"
            f"## Key Points\n\n{key_points}\n\n"
            f"## Follow-ups\n\n{follow_ups}\n"
        )
        path.write_text(body, encoding="utf-8")
        return path
```

**On-disk formats:**

- `manifest.json` — JSON list of `IngestRecord.model_dump(mode="json")`
- `chunks.jsonl` — one `Chunk.model_dump_json()` per line
- `history.jsonl` — one `HistoryEntry.model_dump_json()` per line

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest terminalmind/tests/test_agent.py::test_chunk_text_respects_size terminalmind/tests/test_agent.py::test_ingest_idempotent terminalmind/tests/test_agent.py::test_history_append_and_load_newest_first -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add terminalmind/utils/storage.py terminalmind/tests/test_agent.py
git commit -m "feat: add Storage for ingest, history, and reports"
```

---

### Task 4: ResearchAgent — retrieve + search (mocked OpenAI)

**Files:**
- Create: `terminalmind/core/agent.py`
- Modify: `terminalmind/tests/test_agent.py`
- Test: `terminalmind/tests/test_agent.py`

**Interfaces:**
- Consumes: `Settings`, `Storage`, OpenAI client protocol, schemas
- Produces:
  - `def tokenize(text: str) -> set[str]`  # module-level or static; `[a-z0-9]+` lowercased
  - `class ResearchAgent:`
    - `__init__(self, settings: Settings, storage: Storage | None = None, client: object | None = None) -> None`
    - `ingest(self, path: Path) -> tuple[IngestRecord, bool]`
    - `select_chunks(self, query: str, chunks: list[Chunk]) -> list[Chunk]`
    - `search(self, query: str) -> HistoryEntry`
      # builds messages, calls `client.beta.chat.completions.parse`, persists history+report

- [ ] **Step 1: Write the failing tests**

```python
from unittest.mock import MagicMock

from terminalmind.config import Settings
from terminalmind.core.agent import ResearchAgent, tokenize
from terminalmind.core.schemas import Chunk


def test_tokenize_alnum_lowercase() -> None:
    assert tokenize("Hello, AI-2024!") == {"hello", "ai", "2024"}


def test_select_chunks_prefers_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = Settings(OPENAI_API_KEY="test-key", data_dir=tmp_path)  # or get_settings + copy
    # Prefer constructing Settings in a way that works with your Field aliases.
    settings = get_settings()
    settings = settings.model_copy(update={"data_dir": tmp_path, "max_context_chars": 500})
    agent = ResearchAgent(settings=settings, client=MagicMock())
    chunks = [
        Chunk(id="a:0", source_id="a", text="cats sit on mats", index=0),
        Chunk(id="a:1", source_id="a", text="quantum chromodynamics research", index=1),
        Chunk(id="a:2", source_id="a", text="cats and dogs research", index=2),
    ]
    selected = agent.select_chunks("cats research", chunks)
    assert selected[0].id in {"a:2", "a:0"}
    assert "chromodynamics" not in selected[0].text


def test_search_empty_ingest_sets_used_ingest_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = get_settings().model_copy(update={"data_dir": tmp_path})
    storage = Storage(tmp_path)
    storage.ensure_layout()

    parsed = ResearchAnswer(
        summary="LLM only",
        key_points=["p1"],
        follow_ups=["f1"],
    )
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(parsed=parsed))]
    )
    agent = ResearchAgent(settings=settings, storage=storage, client=mock_client)
    entry = agent.search("what is X?")
    assert entry.used_ingest is False
    assert entry.answer.summary == "LLM only"
    assert storage.load_history()[0].id == entry.id
    assert (tmp_path / "reports" / f"{entry.id}.md").exists()
    mock_client.beta.chat.completions.parse.assert_called_once()
    kwargs = mock_client.beta.chat.completions.parse.call_args.kwargs
    assert kwargs["response_format"] is ResearchAnswer
```

Fix `Settings(...)` construction in tests to match whatever Task 2 settled on (`get_settings().model_copy(...)` is the preferred pattern).

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest terminalmind/tests/test_agent.py::test_tokenize_alnum_lowercase terminalmind/tests/test_agent.py::test_select_chunks_prefers_overlap terminalmind/tests/test_agent.py::test_search_empty_ingest_sets_used_ingest_false -v`

Expected: FAIL — `agent` module missing

- [ ] **Step 3: Write minimal implementation**

`terminalmind/core/agent.py`:

```python
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger
from openai import APIError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError

from terminalmind.config import Settings
from terminalmind.core.schemas import Chunk, HistoryEntry, IngestRecord, ResearchAnswer
from terminalmind.utils.storage import Storage

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class ResearchAgent:
    def __init__(
        self,
        settings: Settings,
        storage: Storage | None = None,
        client: object | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage or Storage(settings.data_dir)
        self.storage.ensure_layout()
        self.client = client or OpenAI(api_key=settings.openai_api_key)

    def ingest(self, path: Path) -> tuple[IngestRecord, bool]:
        return self.storage.ingest_file(
            path,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )

    def select_chunks(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        q_tokens = tokenize(query)
        if not q_tokens or not chunks:
            return []

        def score(chunk: Chunk) -> tuple[int, int]:
            overlap = len(q_tokens & tokenize(chunk.text))
            return (overlap, -chunk.index)  # higher overlap, earlier index

        ranked = sorted(chunks, key=score, reverse=True)
        selected: list[Chunk] = []
        total = 0
        for chunk in ranked:
            if score(chunk)[0] <= 0:
                break
            if total + len(chunk.text) > self.settings.max_context_chars and selected:
                break
            selected.append(chunk)
            total += len(chunk.text)
        return selected

    def search(self, query: str) -> HistoryEntry:
        chunks = self.storage.load_chunks()
        selected = self.select_chunks(query, chunks) if chunks else []
        used_ingest = bool(selected)
        if chunks and not selected:
            # chunks exist but no overlap — still LLM-only for answer quality
            used_ingest = False
            logger.info("No overlapping chunks; falling back to LLM-only context")

        context_block = ""
        if selected:
            context_block = "\n\n".join(
                f"[chunk {c.id}]\n{c.text}" for c in selected
            )

        system = (
            "You are TerminalMind, a careful research assistant. "
            "Return only structured findings matching the schema."
        )
        user = f"Research query:\n{query}"
        if context_block:
            user += f"\n\nLocal context:\n{context_block}"

        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=ResearchAnswer,
            )
        except (AuthenticationError, RateLimitError, APITimeoutError, APIError):
            logger.exception("OpenAI API failure during search")
            raise

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            logger.error("Model returned empty parsed message")
            raise ValueError("Model returned invalid structured output")

        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            query=query,
            answer=parsed,
            created_at=datetime.now(timezone.utc),
            used_ingest=used_ingest,
            chunk_ids=[c.id for c in selected],
        )
        self.storage.append_history(entry)
        self.storage.write_report(entry)
        return entry
```

For `test_select_chunks_prefers_overlap`: scoring must put `"cats and dogs research"` above `"quantum chromodynamics research"` for query `"cats research"`. Implement exactly as above.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest terminalmind/tests/test_agent.py -v`

Expected: all PASS (including earlier tasks)

- [ ] **Step 5: Commit**

```bash
git add terminalmind/core/agent.py terminalmind/tests/test_agent.py
git commit -m "feat: add ResearchAgent with keyword retrieval and structured search"
```

---

### Task 5: Typer CLI (`search`, `ingest`, `history`)

**Files:**
- Create: `terminalmind/main.py`
- Modify: `terminalmind/tests/test_agent.py`
- Test: `terminalmind/tests/test_agent.py`

**Interfaces:**
- Consumes: `ResearchAgent`, `Storage`, `get_settings`, Rich, Typer
- Produces:
  - `app = typer.Typer(...)`  (callable as console script; use `typer.main.get_command` pattern OR `app()` via `if __name__` — for `[project.scripts] terminalmind = "terminalmind.main:app"`, export a callable. Prefer:

```python
app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.callback()
def main(
    ctx: typer.Context,
    data_dir: Path | None = typer.Option(None, "--data-dir"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None: ...

# For setuptools entry point with Typer, use:
# terminalmind = "terminalmind.main:app"
# and ensure `app` is a Typer instance (Typer is callable via Click).
```

  - Commands: `search(query: str)`, `ingest(path: Path)`, `history()`
  - Helper `_build_agent(ctx) -> ResearchAgent`
  - Rich render helpers for answer panels and history table

- [ ] **Step 1: Write the failing CLI tests**

```python
from typer.testing import CliRunner

from terminalmind.main import app

runner = CliRunner()


def test_ingest_rejects_unsupported_extension(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    bad = tmp_path / "x.pdf"
    bad.write_bytes(b"%PDF")
    result = runner.invoke(app, ["--data-dir", str(tmp_path), "ingest", str(bad)])
    assert result.exit_code != 0


def test_ingest_and_history_flow(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    doc = tmp_path / "note.md"
    doc.write_text("TerminalMind portfolio notes about RAG.", encoding="utf-8")
    r1 = runner.invoke(app, ["--data-dir", str(tmp_path), "ingest", str(doc)])
    assert r1.exit_code == 0

    # Seed history without live OpenAI
    storage = Storage(tmp_path)
    entry = HistoryEntry(
        id="seed",
        query="RAG?",
        answer=ResearchAnswer(summary="About RAG", key_points=["k"], follow_ups=["f"]),
        created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        used_ingest=True,
        chunk_ids=["x:0"],
    )
    storage.append_history(entry)
    r2 = runner.invoke(app, ["--data-dir", str(tmp_path), "history"])
    assert r2.exit_code == 0
    assert "RAG?" in r2.stdout
    assert "About RAG" in r2.stdout
```

Optional (recommended) search CLI test with monkeypatched `ResearchAgent.search` or patched OpenAI — keep at least ingest/history covered; add:

```python
def test_search_command_uses_agent(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_entry = HistoryEntry(
        id="s1",
        query="q",
        answer=ResearchAnswer(summary="Sum", key_points=["k"], follow_ups=["f"]),
        created_at=datetime(2026, 1, 4, tzinfo=timezone.utc),
        used_ingest=False,
        chunk_ids=[],
    )

    def fake_search(self, query: str) -> HistoryEntry:  # noqa: ARG001
        return fake_entry

    monkeypatch.setattr(ResearchAgent, "search", fake_search)
    result = runner.invoke(app, ["--data-dir", str(tmp_path), "search", "q"])
    assert result.exit_code == 0
    assert "Sum" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest terminalmind/tests/test_agent.py::test_ingest_rejects_unsupported_extension terminalmind/tests/test_agent.py::test_ingest_and_history_flow -v`

Expected: FAIL — `main` missing

- [ ] **Step 3: Write minimal implementation**

`terminalmind/main.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from openai import APIError, APITimeoutError, AuthenticationError, RateLimitError
from pydantic import ValidationError
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from terminalmind.config import get_settings
from terminalmind.core.agent import ResearchAgent
from terminalmind.core.schemas import ResearchAnswer
from terminalmind.utils.storage import Storage

console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True, help="TerminalMind research CLI")


@app.callback()
def _root(
    ctx: typer.Context,
    data_dir: Optional[Path] = typer.Option(None, "--data-dir", help="Override data directory"),
    verbose: bool = typer.Option(False, "--verbose", help="Enable DEBUG logging"),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir
    ctx.obj["verbose"] = verbose


def _build_agent(ctx: typer.Context, require_api_key: bool = True) -> ResearchAgent:
    try:
        settings = get_settings()
    except ValidationError:
        console.print(
            "[red]Missing OPENAI_API_KEY. Copy .env.example to .env and set your key.[/red]"
        )
        raise typer.Exit(1) from None

    overrides: dict[str, object] = {}
    if ctx.obj.get("data_dir") is not None:
        overrides["data_dir"] = ctx.obj["data_dir"]
    if ctx.obj.get("verbose"):
        overrides["log_level"] = "DEBUG"
    if overrides:
        settings = settings.model_copy(update=overrides)

    storage = Storage(settings.data_dir)
    storage.ensure_layout()
    storage.setup_logging(settings.log_level)

    if require_api_key and not settings.openai_api_key:
        console.print("[red]OPENAI_API_KEY is empty.[/red]")
        raise typer.Exit(1)

    return ResearchAgent(settings=settings, storage=storage)


def _render_answer(answer: ResearchAnswer) -> None:
    console.print(Panel(answer.summary, title="Summary", border_style="cyan"))
    keypoints = "\n".join(f"- {p}" for p in answer.key_points) or "- (none)"
    followups = "\n".join(f"- {p}" for p in answer.follow_ups) or "- (none)"
    console.print(Markdown(f"## Key Points\n\n{keypoints}\n\n## Follow-ups\n\n{followups}"))


@app.command("ingest")
def ingest_cmd(
    ctx: typer.Context,
    path: Path = typer.Argument(..., exists=False, help="Path to .txt or .md file"),
) -> None:
    agent = _build_agent(ctx, require_api_key=False)
    try:
        record, created = agent.ingest(path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if created:
        console.print(f"[green]Ingested[/green] {record.source_path} ({record.char_count} chars)")
    else:
        console.print(f"[yellow]Already ingested[/yellow] (hash {record.content_hash[:12]}…)")


@app.command("search")
def search_cmd(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Research query"),
) -> None:
    agent = _build_agent(ctx, require_api_key=True)
    chunks = agent.storage.load_chunks()
    if not chunks:
        console.print(
            "[yellow]No ingested documents found. Falling back to LLM-only answer.[/yellow]"
        )
    try:
        with console.status("Researching…"):
            entry = agent.search(query)
    except (AuthenticationError, RateLimitError, APITimeoutError, APIError) as exc:
        console.print(f"[red]OpenAI API error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    _render_answer(entry.answer)
    console.print(
        f"[dim]Saved history + report ({entry.id})"
        f"{' · grounded in ingest' if entry.used_ingest else ' · LLM-only'}[/dim]"
    )


@app.command("history")
def history_cmd(ctx: typer.Context) -> None:
    agent = _build_agent(ctx, require_api_key=False)
    entries = agent.storage.load_history()
    if not entries:
        console.print("[dim]No research sessions yet.[/dim]")
        return
    table = Table(title="Research History")
    table.add_column("Time", style="cyan")
    table.add_column("Query")
    table.add_column("Summary")
    table.add_column("Context", justify="center")
    for entry in entries:
        summary = entry.answer.summary
        if len(summary) > 60:
            summary = summary[:57] + "..."
        table.add_row(
            entry.created_at.strftime("%Y-%m-%d %H:%M"),
            entry.query,
            summary,
            "Y" if entry.used_ingest else "N",
        )
    console.print(table)


if __name__ == "__main__":
    app()
```

Note: `ingest` / `history` set `require_api_key=False` so portfolio demos can show local storage without a key; `get_settings()` still needs a dummy `OPENAI_API_KEY` in env for Settings validation — tests set `test-key`. For real CLI without key on ingest-only, either keep key required in Settings (document in README) or give `openai_api_key` a default empty string and enforce only in `search`. Prefer default empty + enforce in `_build_agent(..., require_api_key=True)`:

Update Task 2 `Settings` if not already:

```python
openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
```

Then search fails clearly when empty; ingest/history work without a key.

- [ ] **Step 4: Run full suite**

Run: `pytest -v`

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add terminalmind/main.py terminalmind/tests/test_agent.py
git commit -m "feat: add Typer CLI for search, ingest, and history"
```

---

### Task 6: README + polish logging sink

**Files:**
- Modify: `README.md`
- Modify: `terminalmind/utils/storage.py` (if stderr logging still uses a rough sink — switch to `sys.stderr`)
- Modify: `.env.example` if needed
- Test: manual commands only (no new required unit tests)

**Interfaces:**
- Consumes: finished CLI
- Produces: portfolio-ready README

- [ ] **Step 1: Replace README with full portfolio docs**

Include:

1. One-paragraph pitch (structured outputs + local RAG-lite)
2. Install: `python -m venv .venv`, activate, `pip install -e ".[dev]"`, copy `.env.example` → `.env`
3. Usage:

```bash
terminalmind ingest ./notes.md
terminalmind search "What are the main claims?"
terminalmind history
terminalmind --data-dir ./tmp-data search "..."
```

4. Architecture blurb pointing at `docs/superpowers/specs/2026-07-23-terminalmind-design.md`
5. Testing: `pytest -v`

- [ ] **Step 2: Fix logging to use stderr explicitly**

In `setup_logging`:

```python
import sys
logger.remove()
logger.add(sys.stderr, level=level)
logger.add(self.logs_dir / "terminalmind.log", rotation="1 MB", retention="7 days", level=level)
```

- [ ] **Step 3: Run full verification**

```bash
pytest -v
```

Expected: PASS

Manual smoke (only if real `OPENAI_API_KEY` present — do not fail the task if absent):

```bash
terminalmind --data-dir ./demo-data ingest README.md
terminalmind --data-dir ./demo-data search "What is TerminalMind?"
terminalmind --data-dir ./demo-data history
```

- [ ] **Step 4: Commit**

```bash
git add README.md terminalmind/utils/storage.py .env.example
git commit -m "docs: add README and polish logging"
```

---

## Self-Review Checklist (author)

1. **Spec coverage:** schemas, settings, storage layout, ingest idempotency, keyword retrieve, LLM-only fallback, structured parse, CLI three commands, `--data-dir`/`--verbose`, history table, reports, loguru file, pytest without network — each mapped to a task above.
2. **Placeholders:** none remaining; Storage and CLI include full code blocks.
3. **Types:** `ResearchAnswer`, `Chunk`, `IngestRecord`, `HistoryEntry`, `Storage`, `ResearchAgent.search -> HistoryEntry`, `tokenize -> set[str]` consistent across tasks.
4. **pytest path:** `terminalmind/tests` aligned with package layout (spec's `testpaths = ["tests"]` corrected here to match files).
5. **API key:** default empty string; enforced on `search` only so ingest/history work offline.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-23-terminalmind.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints

Which approach?
