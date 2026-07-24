from datetime import datetime

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Provenance for a grounded answer — chunk id + short snippet."""

    chunk_id: str
    snippet: str = ""


class ResearchAnswer(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    sources: list[SourceCitation] = Field(default_factory=list)


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
