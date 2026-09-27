from __future__ import annotations

import json
from pathlib import Path

from jiwer import (
    wer,
    process_words,
)


def _load_asr_text(stage1_manifest: str | Path) -> str:
    """Load and concatenate raw ASR segment text."""
    manifest_path = Path(stage1_manifest).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    segments = manifest["asr_segments"]

    return " ".join(
        segment["text"].strip()
        for segment in segments
        if segment.get("text", "").strip()
    )


def _load_reference_text(reference_path: str | Path) -> str:
    """Load ground-truth transcript from a UTF-8 text file."""
    path = Path(reference_path).resolve()

    return path.read_text(encoding="utf-8").strip()


def evaluate_stage1(
    stage1_manifest: str | Path,
    reference_transcript: str | Path | None = None,
) -> dict:
    """
    Evaluate Stage 1 ASR using Word Error Rate.

    A ground-truth reference transcript is required for WER.
    """

    hypothesis = _load_asr_text(stage1_manifest)

    summary = {
        "metric": "stage1_asr_wer",
        "reference_available": reference_transcript is not None,
        "wer": None,
        "substitutions": None,
        "deletions": None,
        "insertions": None,
        "reference_word_count": None,
        "hypothesis_word_count": len(hypothesis.split()),
    }

    if reference_transcript is None:
        summary["status"] = "reference_transcript_required"
        return summary

    reference = _load_reference_text(reference_transcript)

    if not reference:
        summary["status"] = "reference_transcript_empty"
        return summary

    alignment = process_words(
        reference,
        hypothesis,
    )

    summary.update(
        {
            "status": "evaluated",
            "wer": wer(reference, hypothesis),
            "substitutions": alignment.substitutions,
            "deletions": alignment.deletions,
            "insertions": alignment.insertions,
            "reference_word_count": len(reference.split()),
        }
    )

    return summary