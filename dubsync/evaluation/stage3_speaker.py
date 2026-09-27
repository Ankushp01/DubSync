from __future__ import annotations

import json
from pathlib import Path

from speechbrain.inference.speaker import SpeakerRecognition


MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


def evaluate_stage3_speaker(
    stage3_manifest: str | Path,
    device: str = "cuda",
) -> dict:
    """
    Evaluate speaker similarity between the reference speaker and
    each generated TTS segment.

    Uses SpeechBrain ECAPA-TDNN speaker verification with cosine
    similarity.
    """

    manifest_path = Path(stage3_manifest).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    reference_audio = Path(manifest["reference_audio"])

    if not reference_audio.is_file():
        raise FileNotFoundError(
            f"Reference audio not found: {reference_audio}"
        )
            
    model_dir = (
        Path(__file__).resolve().parents[2]
        / "models"
        / "spkrec-ecapa-voxceleb"
    )

    model = SpeakerRecognition.from_hparams(
        source=str(model_dir),
        savedir=str(model_dir),
        run_opts={"device": device},
    )

    results = []

    for segment in manifest["segments"]:
        index = int(segment["index"])

        audio_path = manifest_path.parent / segment["audio"]

        if not audio_path.is_file():
            results.append(
                {
                    "index": index,
                    "audio": segment["audio"],
                    "status": "missing_audio",
                }
            )
            continue

        score, prediction = model.verify_files(
            str(reference_audio),
            str(audio_path),
        )

        score_value = float(score.squeeze().item())

        results.append(
            {
                "index": index,
                "audio": segment["audio"],
                "speaker_similarity": score_value,
                "same_speaker_prediction": bool(
                    prediction.squeeze().item()
                ),
                "status": "evaluated",
            }
        )

    valid = [
        item
        for item in results
        if item["status"] == "evaluated"
    ]

    scores = [
        item["speaker_similarity"]
        for item in valid
    ]

    return {
        "metric": "stage3_speaker_similarity",
        "model": MODEL_SOURCE,
        "similarity_metric": "cosine_similarity",
        "reference_audio": str(reference_audio),
        "segments_total": len(manifest["segments"]),
        "segments_evaluated": len(valid),
        "segments_missing_audio": (
            len(manifest["segments"]) - len(valid)
        ),
        "mean_speaker_similarity": (
            sum(scores) / len(scores)
            if scores
            else None
        ),
        "min_speaker_similarity": (
            min(scores)
            if scores
            else None
        ),
        "max_speaker_similarity": (
            max(scores)
            if scores
            else None
        ),
        "segments": results,
    }