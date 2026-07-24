from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from typer.testing import CliRunner

from terminalmind.config import get_settings
from terminalmind.core.agent import ResearchAgent, tokenize
from terminalmind.core.schemas import (
    Chunk,
    HistoryEntry,
    ResearchAnswer,
)
from terminalmind.main import app
from terminalmind.utils.storage import Storage

runner = CliRunner()


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
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
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


def test_tokenize_alnum_lowercase() -> None:
    assert tokenize("Hello, AI-2024!") == {"hello", "ai", "2024"}


def test_select_chunks_prefers_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = get_settings().model_copy(
        update={"data_dir": tmp_path, "max_context_chars": 500}
    )
    agent = ResearchAgent(settings=settings, client=MagicMock())
    chunks = [
        Chunk(id="a:0", source_id="a", text="cats sit on mats", index=0),
        Chunk(id="a:1", source_id="a", text="quantum chromodynamics research", index=1),
        Chunk(id="a:2", source_id="a", text="cats and dogs research", index=2),
    ]
    selected = agent.select_chunks("cats research", chunks)
    assert selected[0].id in {"a:2", "a:0"}
    assert "chromodynamics" not in selected[0].text


def test_search_empty_ingest_sets_used_ingest_false(
    tmp_path: Path, monkeypatch
) -> None:
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
