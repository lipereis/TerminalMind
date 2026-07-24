from datetime import datetime, timezone

from terminalmind.core.schemas import (
    HistoryEntry,
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
