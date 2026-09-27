from __future__ import annotations

import json
from pathlib import Path

import sacrebleu


def _load_translation_hypothesis(
    stage2_manifest: str | Path,
) -> list[str]:
    """Load translated text from Stage 2 output."""
    manifest_path = Path(stage2_manifest).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    return [
        segment["translated_text"].strip()
        for segment in manifest["dialogue_segments"]
    ]


def _load_reference_translation(
    reference_path: str | Path,
) -> list[str]:
    """Load one reference translation per line."""
    path = Path(reference_path).resolve()

    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_stage2(
    stage2_manifest: str | Path,
    reference_translation: str | Path | None = None,
) -> dict:
    """
    Evaluate Stage 2 translation using BLEU and chrF.

    The reference file must contain one human-verified translation
    per dialogue segment, in the same order as the Stage 2 manifest.
    """

    hypothesis = _load_translation_hypothesis(stage2_manifest)

    summary = {
        "metric": "stage2_translation_quality",
        "reference_available": reference_translation is not None,
        "status": None,
        "segments_total": len(hypothesis),
        "segments_evaluated": 0,
        "bleu": None,
        "chrf": None,
        "segment_metrics": [],
    }

    if reference_translation is None:
        summary["status"] = "reference_translation_required"
        return summary

    references = _load_reference_translation(reference_translation)

    if not references:
        summary["status"] = "reference_translation_empty"
        return summary

    if len(references) != len(hypothesis):
        raise ValueError(
            f"Reference segment count ({len(references)}) does not match "
            f"hypothesis segment count ({len(hypothesis)})."
        )

    # Corpus-level metrics.
    bleu = sacrebleu.corpus_bleu(
        hypothesis,
        [references],
    )

    chrf = sacrebleu.corpus_chrf(
        hypothesis,
        [references],
    )

    # Segment-level metrics.
    segment_metrics = []

    for index, (hyp, ref) in enumerate(
        zip(hypothesis, references),
        start=1,
    ):
        segment_bleu = sacrebleu.sentence_bleu(
            hyp,
            [ref],
        )

        segment_chrf = sacrebleu.sentence_chrf(
            hyp,
            [ref],
        )

        segment_metrics.append(
            {
                "index": index,
                "bleu": segment_bleu.score,
                "chrf": segment_chrf.score,
            }
        )

    summary.update(
        {
            "status": "evaluated",
            "segments_evaluated": len(hypothesis),
            "bleu": bleu.score,
            "chrf": chrf.score,
            "segment_metrics": segment_metrics,
        }
    )

    return summary