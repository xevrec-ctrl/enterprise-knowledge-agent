"""Local Markdown retrieval used by the first project version.

The index is deliberately model-free so the project can be tested before an API key is
configured. It ranks Markdown sections with BM25. A vector retriever can later implement
the same ``search`` interface without changing the Agent tool contract.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize English words and Chinese character n-grams without extra packages."""
    tokens: list[str] = []
    for segment in _TOKEN_PATTERN.findall(text.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", segment):
            tokens.extend(segment)
            tokens.extend(
                segment[index : index + 2] for index in range(len(segment) - 1)
            )
        else:
            tokens.append(segment)
    return tokens


@dataclass(frozen=True)
class KnowledgeChunk:
    """One searchable Markdown section."""

    source: str
    title: str
    section: str
    content: str

    @property
    def citation(self) -> str:
        """Return the stable citation marker for this section."""
        return f"【来源：{self.source}#{self.section}】"


@dataclass(frozen=True)
class SearchResult:
    """Ranked evidence returned by the knowledge base."""

    chunk: KnowledgeChunk
    score: float

    def as_dict(self) -> dict[str, str | float]:
        """Serialize evidence into the tool response schema."""
        return {
            "source": self.chunk.source,
            "section": self.chunk.section,
            "citation": self.chunk.citation,
            "score": round(self.score, 4),
            "content": self.chunk.content,
        }


def _load_markdown(path: Path) -> list[KnowledgeChunk]:
    title = path.stem
    section = "概述"
    lines: list[str] = []
    chunks: list[KnowledgeChunk] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if content:
            chunks.append(
                KnowledgeChunk(
                    source=path.name,
                    title=title,
                    section=section,
                    content=content,
                )
            )
        lines.clear()

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
            section = "概述"
        elif line.startswith("## "):
            flush()
            section = line[3:].strip()
        elif line:
            lines.append(line)
    flush()
    return chunks


class KnowledgeBase:
    """Load Markdown documents and perform deterministic BM25 retrieval."""

    def __init__(self, directory: Path | str = DEFAULT_KNOWLEDGE_DIR) -> None:
        """Build an in-memory index from all Markdown files in a directory."""
        self.directory = Path(directory)
        self.chunks = self._load_chunks()
        self._tokens = [tokenize(self._searchable_text(chunk)) for chunk in self.chunks]
        self._term_counts = [Counter(tokens) for tokens in self._tokens]
        self._document_frequency = self._build_document_frequency()
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )

    @staticmethod
    def _searchable_text(chunk: KnowledgeChunk) -> str:
        return f"{chunk.title} {chunk.section} {chunk.content}"

    def _load_chunks(self) -> list[KnowledgeChunk]:
        if not self.directory.exists():
            return []
        chunks: list[KnowledgeChunk] = []
        for path in sorted(self.directory.glob("*.md")):
            chunks.extend(_load_markdown(path))
        return chunks

    def _build_document_frequency(self) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for tokens in self._tokens:
            frequencies.update(set(tokens))
        return frequencies

    def search(self, query: str, top_k: int = 3) -> list[SearchResult]:
        """Return the most relevant sections for a query."""
        query_tokens = tokenize(query)
        if not query_tokens or not self.chunks:
            return []

        results: list[SearchResult] = []
        document_count = len(self.chunks)
        k1 = 1.5
        b = 0.75

        for index, chunk in enumerate(self.chunks):
            token_counts = self._term_counts[index]
            document_length = len(self._tokens[index]) or 1
            score = 0.0
            for token in query_tokens:
                frequency = token_counts.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self._document_frequency[token]
                inverse_frequency = math.log(
                    1
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + k1 * (
                    1 - b + b * document_length / (self._average_length or 1.0)
                )
                score += inverse_frequency * (frequency * (k1 + 1)) / denominator

            heading_tokens = set(tokenize(f"{chunk.title} {chunk.section}"))
            score += 0.35 * len(set(query_tokens) & heading_tokens)
            if score > 0:
                results.append(SearchResult(chunk=chunk, score=score))

        results.sort(
            key=lambda item: (-item.score, item.chunk.source, item.chunk.section)
        )
        return results[: max(1, min(top_k, 5))]


@lru_cache(maxsize=1)
def get_default_knowledge_base() -> KnowledgeBase:
    """Return the process-wide knowledge base index."""
    return KnowledgeBase()
