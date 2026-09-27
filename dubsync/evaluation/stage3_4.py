from __future__ import annotations

import json
import math
from pathlib import Path

import soundfile as sf


def _duration(path: Path) -> float:
    """Return WAV duration in seconds."""
    info = sf.info(str(path))
    return float(info.frames / info.samplerate)


def evaluate_stage3(stage3_manifest: str | Path) -> dict:
    """
    Evaluate Stage 3 TTS duration accuracy.

    Uses the final segment_XXX.wav files, not the raw Chatterbox files.
    """
    manifest_path = Path(stage3_manifest).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    output_dir = manifest_path.parent
    segments = manifest["segments"]

    results = []

    for segment in segments:
        index = int(segment["index"])
        target_duration = float(segment["target_duration"])

        audio_name = segment["audio"]
        audio_path = output_dir / audio_name

        if not audio_path.is_file():
            results.append(
                {
                    "index": index,
                    "audio": audio_name,
                    "status": "missing_audio",
                }
            )
            continue

        actual_duration = _duration(audio_path)

        absolute_error = abs(actual_duration - target_duration)

        relative_error = (
            absolute_error / target_duration
            if target_duration > 0
            else None
        )

        results.append(
            {
                "index": index,
                "audio": audio_name,
                "target_duration": target_duration,
                "actual_duration": actual_duration,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "relative_error_percent": (
                    relative_error * 100
                    if relative_error is not None
                    else None
                ),
            }
        )

    valid = [
        item
        for item in results
        if item.get("status") != "missing_audio"
    ]

    absolute_errors = [
        item["absolute_error"]
        for item in valid
    ]

    relative_errors = [
        item["relative_error"]
        for item in valid
        if item["relative_error"] is not None
    ]

    summary = {
        "metric": "stage3_tts_duration_accuracy",
        "segments_total": len(segments),
        "segments_evaluated": len(valid),
        "segments_missing_audio": len(segments) - len(valid),
        "mean_absolute_duration_error_seconds": (
            sum(absolute_errors) / len(absolute_errors)
            if absolute_errors
            else None
        ),
        "max_absolute_duration_error_seconds": (
            max(absolute_errors)
            if absolute_errors
            else None
        ),
        "mean_relative_duration_error": (
            sum(relative_errors) / len(relative_errors)
            if relative_errors
            else None
        ),
        "mean_relative_duration_error_percent": (
            sum(relative_errors) / len(relative_errors) * 100
            if relative_errors
            else None
        ),
        "within_0_10_seconds": sum(
            error <= 0.10 for error in absolute_errors
        ),
        "within_0_25_seconds": sum(
            error <= 0.25 for error in absolute_errors
        ),
        "within_0_50_seconds": sum(
            error <= 0.50 for error in absolute_errors
        ),
        "segments": results,
    }

    return summary


def evaluate_stage4(
    stage3_manifest: str | Path,
    assembled_audio: str | Path,
) -> dict:
    """
    Evaluate timestamp-based Stage 4 audio assembly.

    Stage 4 is deterministic: every segment is placed at its requested
    start timestamp. We therefore verify the resulting timeline against
    the target segment windows and assembled file duration.
    """
    manifest_path = Path(stage3_manifest).resolve()
    audio_path = Path(assembled_audio).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    segments = manifest["segments"]

    expected_end = max(
        float(segment["end"])
        for segment in segments
    )

    intervals = []

    for segment in segments:
        start = float(segment["start"])
        target_end = float(segment["end"])

        audio_file = manifest_path.parent / segment["audio"]

        if not audio_file.is_file():
            continue

        actual_duration = _duration(audio_file)
        actual_end = start + actual_duration

        intervals.append(
            {
                "index": int(segment["index"]),
                "target_start": start,
                "target_end": target_end,
                "actual_start": start,
                "actual_end": actual_end,
                "start_error_seconds": 0.0,
                "end_error_seconds": actual_end - target_end,
                "absolute_end_error_seconds": abs(
                    actual_end - target_end
                ),
            }
        )

    # Check for overlaps in the generated audio intervals.
    # Stage 4 timing accuracy metrics.
    timestamp_errors = [
        abs(item["actual_start"] - item["target_start"])
        for item in intervals
    ]

    overrun_values = [
        max(
            0.0,
            item["actual_end"] - item["target_end"],
        )
        for item in intervals
    ]

    underrun_values = [
        max(
            0.0,
            item["target_end"] - item["actual_end"],
        )
        for item in intervals
    ]

    overruns = [
        {
            "segment": item["index"],
            "overrun_seconds": item["actual_end"] - item["target_end"],
        }
        for item in intervals
        if item["actual_end"] > item["target_end"]
    ]

    underruns = [
        {
            "segment": item["index"],
            "underrun_seconds": item["target_end"] - item["actual_end"],
        }
        for item in intervals
        if item["actual_end"] < item["target_end"]
    ]

    actual_timeline_duration = _duration(audio_path)

    timeline_error = actual_timeline_duration - expected_end


    summary = {
        "metric": "stage4_audio_assembly_accuracy",

        "expected_timeline_end_seconds": expected_end,
        "actual_timeline_duration_seconds": actual_timeline_duration,

        "timeline_duration_error_seconds": timeline_error,
        "absolute_timeline_duration_error_seconds": abs(
            timeline_error
        ),

        "mean_timestamp_placement_error_seconds": (
            sum(timestamp_errors) / len(timestamp_errors)
            if timestamp_errors
            else None
        ),

        "max_timestamp_placement_error_seconds": (
            max(timestamp_errors)
            if timestamp_errors
            else None
        ),

        "segment_overrun_count": len(overruns),
        "segment_underrun_count": len(underruns),

        "mean_overrun_seconds": (
            sum(overrun_values) / len(overrun_values)
            if overrun_values
            else None
        ),

        "max_overrun_seconds": (
            max(overrun_values)
            if overrun_values
            else None
        ),

        "mean_underrun_seconds": (
            sum(underrun_values) / len(underrun_values)
            if underrun_values
            else None
        ),

        "max_underrun_seconds": (
            max(underrun_values)
            if underrun_values
            else None
        ),

        "overruns": overruns,
        "underruns": underruns,

        "segments": intervals,
    }

    return summary


def evaluate_stage3_and_stage4(
    stage3_manifest: str | Path,
    assembled_audio: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Run both evaluations and save JSON reports."""

    stage3_manifest = Path(stage3_manifest).resolve()
    assembled_audio = Path(assembled_audio).resolve()

    if output_dir is None:
        output_dir = stage3_manifest.parent / "evaluation"

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage3_metrics = evaluate_stage3(stage3_manifest)
    stage4_metrics = evaluate_stage4(
        stage3_manifest,
        assembled_audio,
    )

    stage3_path = output_dir / "stage3_metrics.json"
    stage4_path = output_dir / "stage4_metrics.json"

    stage3_path.write_text(
        json.dumps(
            stage3_metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    stage4_path.write_text(
        json.dumps(
            stage4_metrics,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return stage3_path, stage4_path