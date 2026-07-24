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
        if client is not None:
            self.client = client
        elif settings.openai_base_url:
            self.client = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
        else:
            self.client = OpenAI(api_key=settings.openai_api_key)

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
            return (overlap, -chunk.index)

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

    def _structured_answer(self, messages: list[dict[str, str]]) -> ResearchAnswer:
        """Parse into ResearchAnswer via OpenAI Structured Outputs, with JSON fallback."""
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.settings.openai_model,
                messages=messages,
                response_format=ResearchAnswer,
            )
            parsed = completion.choices[0].message.parsed
            if parsed is not None:
                return parsed
            logger.warning("parse() returned empty; falling back to JSON schema")
        except (AuthenticationError, RateLimitError, APITimeoutError):
            raise
        except (AttributeError, TypeError, APIError) as exc:
            logger.warning("Structured parse unavailable ({}); using JSON fallback", exc)

        completion = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "research_answer",
                    "strict": True,
                    "schema": ResearchAnswer.model_json_schema(),
                },
            },
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Model returned invalid structured output")
        return ResearchAnswer.model_validate_json(content)

    def search(self, query: str) -> HistoryEntry:
        chunks = self.storage.load_chunks()
        selected = self.select_chunks(query, chunks) if chunks else []
        used_ingest = bool(selected)
        if chunks and not selected:
            logger.info("No overlapping chunks; falling back to LLM-only context")

        context_block = ""
        if selected:
            context_block = "\n\n".join(f"[chunk {c.id}]\n{c.text}" for c in selected)

        system = (
            "You are TerminalMind, a careful research assistant. "
            "Return only structured findings matching the schema."
        )
        user = f"Research query:\n{query}"
        if context_block:
            user += f"\n\nLocal context:\n{context_block}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            parsed = self._structured_answer(messages)
        except (AuthenticationError, RateLimitError, APITimeoutError, APIError):
            logger.exception("LLM API failure during search")
            raise

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
