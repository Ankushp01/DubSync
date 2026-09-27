"""Natural TTS chunking for DubSync."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


MIN_TARGET_WORDS = 45
TARGET_WORDS = 55
MAX_WORDS = 80


@dataclass
class TTSChunk:
    """A natural-language chunk prepared for TTS."""

    source_segment_indices: list[int]
    source_text: str
    translated_text: str
    word_count: int

    # Exact timing is only available when the chunk maps to
    # complete Stage 2 segments.
    start: float | None = None
    end: float | None = None

    @property
    def target_duration(self) -> float | None:
        """Return target duration when exact timing is available."""
        if self.start is None or self.end is None:
            return None
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = {
            "source_segment_indices": self.source_segment_indices,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
            "word_count": self.word_count,
        }

        if self.start is not None and self.end is not None:
            data["start"] = self.start
            data["end"] = self.end
            data["target_duration"] = self.target_duration

        return data


def build_tts_chunks(
    segments: list[dict[str, Any]],
    *,
    min_target_words: int = MIN_TARGET_WORDS,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
) -> list[TTSChunk]:
    """
    Combine Stage 2 dialogue into natural TTS chunks.

    Rules:
    - Aim for roughly 50–60 words.
    - Prefer complete sentences.
    - Never split a sentence merely to hit the word target.
    - Allow chunks to grow toward 70–80 words when necessary.
    - Allow a short final chunk.
    - Do not invent sentence-level timestamps.
    """

    if not segments:
        return []

    units = _build_sentence_units(segments)

    chunks: list[TTSChunk] = []
    current: list[dict[str, Any]] = []
    current_words = 0

    for unit in units:
        unit_words = _word_count(unit["translated_text"])

        if not current:
            current.append(unit)
            current_words = unit_words
            continue

        proposed_words = current_words + unit_words

        # Still within our preferred target.
        if proposed_words <= target_words:
            current.append(unit)
            current_words = proposed_words
            continue

        # We already have a good-sized chunk.
        if current_words >= min_target_words:
            chunks.append(_make_chunk(current))
            current = [unit]
            current_words = unit_words
            continue

        # Current chunk is short, so allow it to grow.
        if proposed_words <= max_words:
            current.append(unit)
            current_words = proposed_words
            continue

        # Adding this sentence would make the chunk too large.
        chunks.append(_make_chunk(current))
        current = [unit]
        current_words = unit_words

    if current:
        chunks.append(_make_chunk(current))

    return chunks


def _build_sentence_units(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Split translated text into sentence units while retaining source IDs."""

    units: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        text = segment["translated_text"].strip()

        if not text:
            continue

        sentences = _split_sentences(text)

        for sentence_index, sentence in enumerate(sentences):
            sentence = sentence.strip()

            if not sentence:
                continue

            units.append(
                {
                    "segment_index": index,
                    "source_text": segment["source_text"],
                    "translated_text": sentence,
                    "is_complete_segment": (
                        len(sentences) == 1
                    ),
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                }
            )

    return units


def _split_sentences(text: str) -> list[str]:
    """Split text at strong sentence boundaries."""

    parts = re.split(
        r"(?<=[.!?।])\s+",
        text.strip(),
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def _word_count(text: str) -> int:
    """Count whitespace-separated words."""

    return len(text.split())


def _make_chunk(
    units: list[dict[str, Any]],
) -> TTSChunk:
    """Combine sentence units into one TTS chunk."""

    translated_text = " ".join(
        unit["translated_text"]
        for unit in units
    )

    source_text = " ".join(
        unit["source_text"]
        for unit in units
    )

    segment_indices = sorted(
        set(unit["segment_index"] for unit in units)
    )

    # Only assign exact timing when the chunk contains complete
    # Stage 2 segments. If a Stage 2 segment was split between
    # multiple TTS chunks, timing is intentionally left unknown.
    complete_segment_indices = _complete_segment_indices(
        units
    )

    start = None
    end = None

    if len(segment_indices) == 1:
        # A single Stage 2 segment may have been split into
        # multiple sentence chunks, so we need to check whether
        # all its text belongs to this chunk.
        if _contains_entire_segment(units):
            start = units[0].get("start")
            end = units[0].get("end")

    elif complete_segment_indices == segment_indices:
        start = units[0].get("start")
        end = units[-1].get("end")

    return TTSChunk(
        source_segment_indices=segment_indices,
        source_text=source_text,
        translated_text=translated_text,
        word_count=_word_count(translated_text),
        start=start,
        end=end,
    )


def _contains_entire_segment(
    units: list[dict[str, Any]],
) -> bool:
    """
    Determine whether all sentence units from the same Stage 2
    segment are contained in this chunk.

    The units carry a private marker indicating whether they
    represent the complete translated segment.
    """

    return all(
        unit["is_complete_segment"]
        for unit in units
    )


def _complete_segment_indices(
    units: list[dict[str, Any]],
) -> list[int]:
    """Return Stage 2 segment IDs represented completely in this chunk."""

    return sorted(
        set(
            unit["segment_index"]
            for unit in units
            if unit["is_complete_segment"]
        )
    )