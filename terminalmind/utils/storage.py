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
                chunks.append(
                    Chunk(
                        id=f"{source_id}:{len(chunks)}",
                        source_id=source_id,
                        text=buffer,
                        index=len(chunks),
                    )
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

        chunks = self.chunk_text(
            text, source_id=source_id, chunk_size=chunk_size, overlap=overlap
        )
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
        for line_no, line in enumerate(
            self.chunks_path.read_text(encoding="utf-8").splitlines(), 1
        ):
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
        for line_no, line in enumerate(
            self.history_path.read_text(encoding="utf-8").splitlines(), 1
        ):
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
