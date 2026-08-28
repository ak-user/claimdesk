"""Splits the benefits booklet into one chunk per numbered section.

Policy documents are written in numbered sections, and members and examiners refer to them that
way ("section 4.2"). Chunking on those headings keeps every chunk self-contained and gives each
answer a citation a person can check, which fixed-size windows would not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEADING = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.+?)\s*$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"])")
MAX_WORDS = 180


@dataclass(frozen=True)
class Chunk:
    section: str
    title: str
    parent_title: str
    text: str

    @property
    def label(self) -> str:
        return f"{self.section} {self.title}"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def snippet(text: str, limit: int = 240) -> str:
    """Shortens text at a word boundary for display."""
    return text if len(text) <= limit else text[: text.rfind(" ", 0, limit)] + "…"


def parse_booklet(markdown: str) -> list[Chunk]:
    """Returns one chunk per section that has body text, in document order."""
    chunks: list[Chunk] = []
    parent_title = ""
    section: str | None = None
    title = ""
    parent_for_section = ""
    body: list[str] = []

    def flush() -> None:
        text = " ".join(line.strip() for line in body if line.strip())
        if section is not None and text:
            chunks.extend(_split_long(Chunk(section, title, parent_for_section, text)))

    for line in markdown.splitlines():
        match = HEADING.match(line)
        if match:
            flush()
            level, number, heading = match.groups()
            body = []
            section, title = number, heading
            if level == "##":
                parent_title = heading
                parent_for_section = ""
            else:
                parent_for_section = parent_title
            continue
        if section is not None and not line.startswith(">"):
            body.append(line)
    flush()
    return chunks


def _split_long(chunk: Chunk) -> list[Chunk]:
    """Keeps sections short enough to cite precisely. The demo booklet never needs it."""
    if len(chunk.text.split()) <= MAX_WORDS:
        return [chunk]
    parts: list[list[str]] = [[]]
    for sentence in split_sentences(chunk.text):
        if parts[-1] and len(" ".join(parts[-1] + [sentence]).split()) > MAX_WORDS:
            parts.append([])
        parts[-1].append(sentence)
    return [
        Chunk(
            f"{chunk.section}" if i == 0 else f"{chunk.section}-{i + 1}",
            chunk.title,
            chunk.parent_title,
            " ".join(part),
        )
        for i, part in enumerate(parts)
    ]
