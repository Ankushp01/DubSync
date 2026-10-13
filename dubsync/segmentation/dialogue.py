"""Smart dialogue segmentation and natural utterance grouping for DubSync.

Architecture:
ASR -> speaker / pause / linguistic analysis -> SMART DIALOGUE SEGMENTATION
-> 8–10 sec preferred segments (12 sec hard ceiling) -> translation -> TTS
"""

from __future__ import annotations

import math
import re
from typing import Optional

from dubsync.models import TimedText

# Core Target Constants
PREFERRED_SEGMENT_MIN_SECONDS = 8.0
PREFERRED_SEGMENT_MAX_SECONDS = 10.0
HARD_MAX_SEGMENT_SECONDS = 12.0
MEANINGFUL_PAUSE_SECONDS = 0.75

SENTENCE_ENDINGS = (".", "?", "!", "。", "？", "！")
CLAUSE_PUNCTUATION = (";", ":", "—", ",", "-", "…")

# Common English and multilingual coordinating / subordinating conjunctions
CLAUSE_CONJUNCTIONS = (
    r"\b(?:and|but|so|because|although|though|when|while|where|since|if|then|or|yet|which|that)\b"
)


def _ends_sentence(text: str) -> bool:
    """Return True when text ends with strong sentence termination punctuation."""
    return text.rstrip().endswith(SENTENCE_ENDINGS)


def _find_linguistic_split_point(
    text: str,
    target_ratio: float,
    min_ratio: float = 0.25,
    max_ratio: float = 0.75,
) -> int:
    """Find the best natural linguistic split point (character index) in text.

    Priority hierarchy:
    1. Sentence boundary: '. ', '? ', '! '
    2. Major clause punctuation: '; ', ': ', '— '
    3. Comma or dash: ', ', ' - '
    4. Clause conjunctions: ' and ', ' but ', ' so ', ' because ', etc.
    5. Word boundary nearest target ratio.
    """
    length = len(text)
    if length < 10:
        return length // 2

    target_idx = int(length * target_ratio)
    min_idx = max(1, int(length * min_ratio))
    max_idx = min(length - 1, int(length * max_ratio))

    candidates: list[tuple[float, int]] = []

    # Find all whitespace positions within [min_idx, max_idx]
    for m in re.finditer(r"\s+", text):
        idx = m.start()
        if idx < min_idx or idx > max_idx:
            continue

        prefix = text[:idx].rstrip()
        suffix = text[m.end():].lstrip()
        if not prefix or not suffix:
            continue

        # Proximity score penalty (0 to 50 based on distance from target_idx)
        dist = abs(idx - target_idx) / length
        proximity_penalty = dist * 40.0

        score = 0.0

        # Category 1: Sentence boundary
        if prefix.endswith(SENTENCE_ENDINGS):
            score = 100.0 - proximity_penalty
        # Category 2: Semicolon / colon / dash
        elif prefix.endswith((";", ":", "—")):
            score = 80.0 - proximity_penalty
        # Category 3: Comma
        elif prefix.endswith((",", "…", "-")):
            score = 65.0 - proximity_penalty
        # Category 4: Clause conjunctions at start of suffix or end of prefix
        elif re.match(CLAUSE_CONJUNCTIONS, suffix, re.IGNORECASE):
            score = 55.0 - proximity_penalty
        # Category 5: General word boundary
        else:
            score = 25.0 - proximity_penalty

        candidates.append((score, m.end()))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    # Fallback: nearest space to target_idx
    best_space = -1
    best_dist = float("inf")
    for m in re.finditer(r"\s+", text):
        idx = m.start()
        d = abs(idx - target_idx)
        if d < best_dist:
            best_dist = d
            best_space = m.end()

    if best_space != -1:
        return best_space
    return target_idx


def split_long_segment(
    segment: TimedText,
    hard_max_seconds: float = HARD_MAX_SEGMENT_SECONDS,
    preferred_min_seconds: float = PREFERRED_SEGMENT_MIN_SECONDS,
    preferred_max_seconds: float = PREFERRED_SEGMENT_MAX_SECONDS,
) -> list[TimedText]:
    """Recursively split a segment longer than hard_max_seconds at natural linguistic boundaries.

    Guarantees every output segment has duration <= hard_max_seconds without arbitrary mid-word cuts.
    """
    duration = segment.end - segment.start
    if duration <= hard_max_seconds:
        return [segment]

    # Determine desired number of subsegments
    num_splits = math.ceil(duration / preferred_max_seconds)
    if num_splits < 2:
        num_splits = 2

    # Target first piece duration: aim for preferred range (e.g. 8-10s) or balanced fraction
    target_first_dur = min(preferred_max_seconds, duration / num_splits)
    target_ratio = target_first_dur / duration

    # Allow split between 3.0s from start and (hard_max_seconds - 0.5s)
    min_ratio = max(0.15, 3.0 / duration)
    max_ratio = min(0.85, (hard_max_seconds - 0.5) / duration)

    split_char_idx = _find_linguistic_split_point(
        segment.text,
        target_ratio=target_ratio,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
    )

    text1 = segment.text[:split_char_idx].strip()
    text2 = segment.text[split_char_idx:].strip()

    if not text1 or not text2:
        # Emergency word-level fallback if regex failed
        words = segment.text.split()
        if len(words) >= 2:
            half = len(words) // 2
            text1 = " ".join(words[:half])
            text2 = " ".join(words[half:])
            split_char_idx = len(text1) + 1
        else:
            return [segment]

    t_split = round(
        segment.start + (split_char_idx / max(1, len(segment.text))) * duration,
        3,
    )
    # Ensure t_split is strictly inside boundaries
    t_split = max(round(segment.start + 1.0, 3), min(round(segment.end - 1.0, 3), t_split))

    part1 = TimedText(
        start=segment.start,
        end=t_split,
        text=text1,
        speaker=segment.speaker,
    )
    part2 = TimedText(
        start=t_split,
        end=segment.end,
        text=text2,
        speaker=segment.speaker,
    )

    results = []
    # If part1 still somehow exceeds hard max, split it
    if (part1.end - part1.start) > hard_max_seconds:
        results.extend(split_long_segment(part1, hard_max_seconds, preferred_min_seconds, preferred_max_seconds))
    else:
        results.append(part1)

    # If part2 exceeds hard max, recursively split it
    if (part2.end - part2.start) > hard_max_seconds:
        results.extend(split_long_segment(part2, hard_max_seconds, preferred_min_seconds, preferred_max_seconds))
    else:
        results.append(part2)

    return results


def group_dialogue(
    segments: list[TimedText],
    *,
    max_gap_seconds: float = 0.75,
    max_characters: int = 400,
    preferred_min_duration_seconds: float = PREFERRED_SEGMENT_MIN_SECONDS,
    preferred_max_duration_seconds: float = PREFERRED_SEGMENT_MAX_SECONDS,
    hard_max_duration_seconds: float = HARD_MAX_SEGMENT_SECONDS,
    meaningful_pause_seconds: float = MEANINGFUL_PAUSE_SECONDS,
    min_duration_seconds: float | None = None,
) -> list[TimedText]:
    """Smart dialogue segmentation: merges continuous speech towards 8-10s with a 12s hard ceiling.

    Principles:
    1. Target Range: 8–10 seconds preferred. 12 seconds hard ceiling.
    2. Naturalness: Shorter segments are preserved when there is a natural pause, speaker change,
       or conversational turn. Short utterances are NEVER padded with artificial silence.
    3. Merging: Consecutive compatible segments from the same speaker with no meaningful pause
       are merged until approximately 8–10 seconds.
    4. Splitting: Any utterance exceeding 12 seconds is split at the nearest natural linguistic
       boundary (sentence, clause, comma, conjunction) rather than an arbitrary mid-word cut.
    """
    if not segments:
        return []

    # Effective parameters
    effective_pause = min(max_gap_seconds, meaningful_pause_seconds)
    pref_min = preferred_min_duration_seconds
    pref_max = preferred_max_duration_seconds
    hard_max = hard_max_duration_seconds

    # Step 1: Pre-process and split any single ASR segment longer than 12s
    preprocessed: list[TimedText] = []
    for s in segments:
        if s.end < s.start:
            raise ValueError(f"Segment ends before it starts: {s.start} -> {s.end}")
        dur = s.end - s.start
        if dur > hard_max:
            preprocessed.extend(
                split_long_segment(
                    s,
                    hard_max_seconds=hard_max,
                    preferred_min_seconds=pref_min,
                    preferred_max_seconds=pref_max,
                )
            )
        else:
            preprocessed.append(s)

    # Step 2: Smart merging of consecutive segments
    groups: list[TimedText] = []

    for seg in preprocessed:
        if not groups:
            groups.append(seg)
            continue

        prev = groups[-1]
        prev_dur = prev.end - prev.start
        gap = seg.start - prev.end
        combined_dur = seg.end - prev.start
        combined_text = f"{prev.text} {seg.text}".strip()

        # Rule 1: Speaker change -> Never merge
        if prev.speaker is not None and seg.speaker is not None and prev.speaker != seg.speaker:
            groups.append(seg)
            continue

        # Rule 2: Meaningful silence pause -> Never merge across pause
        if gap >= effective_pause:
            groups.append(seg)
            continue

        # Rule 3: Hard ceilings (12s duration or max characters) -> Never exceed
        if combined_dur > hard_max or len(combined_text) > max_characters:
            groups.append(seg)
            continue

        # Rule 4: Previous group is already in or above preferred range (>= 8.0s)
        if prev_dur >= pref_min:
            # If previous ended a sentence, keep it separate as a complete, well-timed turn
            if _ends_sentence(prev.text):
                groups.append(seg)
                continue

            # If previous did not end a sentence, merge up to pref_max (10s),
            # or up to 12s if this segment completes the sentence
            if combined_dur <= pref_max:
                pass  # Merge to preserve continuous utterance
            elif combined_dur <= hard_max and _ends_sentence(seg.text):
                pass  # Merge beyond 10s up to 12s to complete natural utterance
            else:
                groups.append(seg)
                continue

        # Rule 5: Previous group is below preferred min (< 8.0s)
        else:
            # If previous did not end a sentence: it's an unfinished fragment -> always merge if <= 12s
            if not _ends_sentence(prev.text):
                pass  # Merge

            # If previous ended a sentence:
            else:
                # Merge when combined duration fits naturally within 8–10 seconds (or <= pref_max)
                if combined_dur <= pref_max:
                    pass  # Merge into natural conversational chunk
                elif combined_dur <= hard_max and prev_dur < 3.0:
                    # Avoid leaving a very short isolated fragment (e.g. "Simple.")
                    pass
                else:
                    # Already a substantial natural utterance (e.g. 5–7s) and combining exceeds 10s
                    groups.append(seg)
                    continue

        # Merge segment into previous group
        groups[-1] = TimedText(
            start=prev.start,
            end=seg.end,
            text=combined_text,
            speaker=prev.speaker or seg.speaker,
        )

    # Step 3: Sanitize and guarantee clean non-overlapping timeline boundaries
    sanitized: list[TimedText] = []
    for g in groups:
        start = max(0.0, float(g.start))
        end = max(start + 0.1, float(g.end))
        if sanitized:
            prev_seg = sanitized[-1]
            if start < prev_seg.end:
                start = prev_seg.end
                if end <= start:
                    end = start + 0.1
        sanitized.append(
            TimedText(
                start=round(start, 3),
                end=round(end, 3),
                text=g.text,
                speaker=g.speaker,
            )
        )

    return sanitized