from datetime import datetime, timezone
from pathlib import Path

from terminalmind.config import get_settings
from terminalmind.core.schemas import (
    HistoryEntry,
    ResearchAnswer,
)
from terminalmind.utils.storage import Storage


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


def test_settings_data_dir_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TERMINALMIND_DATA_DIR", str(tmp_path))
    settings = get_settings()
    assert settings.openai_api_key == "test-key"
    assert settings.data_dir == tmp_path
    assert settings.openai_model == "gpt-4o-mini"


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
