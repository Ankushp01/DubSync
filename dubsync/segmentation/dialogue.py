"""Group timestamped ASR fragments into natural dialogue turns."""

from __future__ import annotations

from dubsync.models import TimedText


SENTENCE_ENDINGS = (".", "?", "!", "。", "？", "！")


def _ends_sentence(text: str) -> bool:
    """Return True when the fragment ends at a sentence boundary."""
    return text.rstrip().endswith(SENTENCE_ENDINGS)


def group_dialogue(
    segments: list[TimedText],
    *,
    max_gap_seconds: float,
    max_characters: int,
    min_duration_seconds: float = 3.0,
    preferred_max_duration_seconds: float = 12.0,
    hard_max_duration_seconds: float = 18.0,
) -> list[TimedText]:
    """
    Merge ASR fragments into natural dialogue turns.

    Priorities:
    1. Never cross a large silence gap.
    2. Prefer sentence boundaries.
    3. Keep turns around the preferred duration.
    4. Never exceed the hard duration or character limit.
    5. Keep very short conversational fragments from becoming
       unnecessarily isolated.
    """

    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative.")

    if max_characters < 1:
        raise ValueError("max_characters must be positive.")

    if min_duration_seconds < 0:
        raise ValueError("min_duration_seconds must be non-negative.")

    if preferred_max_duration_seconds < min_duration_seconds:
        raise ValueError(
            "preferred_max_duration_seconds must be >= min_duration_seconds."
        )

    if hard_max_duration_seconds < preferred_max_duration_seconds:
        raise ValueError(
            "hard_max_duration_seconds must be >= preferred_max_duration_seconds."
        )

    groups: list[TimedText] = []

    for segment in segments:
        if segment.end < segment.start:
            raise ValueError("A transcript segment ends before it starts.")

        if not groups:
            groups.append(segment)
            continue

        previous = groups[-1]
        current_duration = previous.end - previous.start
        gap = segment.start - previous.end

        proposed_text = f"{previous.text} {segment.text}".strip()
        proposed_duration = segment.end - previous.start

        # 1. A real silence gap starts a new dialogue turn.
        if gap > max_gap_seconds:
            groups.append(segment)
            continue

        # 2. Hard safety limits always start a new group.
        if (
            proposed_duration > hard_max_duration_seconds
            or len(proposed_text) > max_characters
        ):
            groups.append(segment)
            continue

        # 3. If the previous fragment already ended a sentence,
        #    normally preserve that boundary.
        #
        #    Exception: if the current group is very short, allow the
        #    next fragment to join it. This handles dialogue such as:
        #    "You haven't?" + "No, re-watch..."
        if (
            _ends_sentence(previous.text)
            and current_duration >= min_duration_seconds
        ):
            groups.append(segment)
            continue

        # 4. Once we are around the preferred duration, prefer
        #    starting a new group rather than creating a long turn.
        if current_duration >= preferred_max_duration_seconds:
            groups.append(segment)
            continue

        # 5. Otherwise merge the fragments.
        groups[-1] = TimedText(
            start=previous.start,
            end=segment.end,
            text=proposed_text,
        )

    return groups