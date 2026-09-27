from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import soundfile as sf


def _read_wav(path: Path) -> tuple[dict, np.ndarray]:
    path = Path(path)

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    if audio.ndim != 1:
        raise ValueError(
            f"Expected mono audio, got shape {audio.shape}: {path}"
        )

    # Float32 -> int16 PCM
    audio = np.clip(audio, -1.0, 1.0)
    audio = (audio * 32767.0).astype(np.int16)

    params = {
        "channels": 1,
        "sample_rate": int(sample_rate),
        "sample_width": 2,
    }

    return params, audio


def assemble_stage3_audio(
    stage3_path: str | Path,
    output_path: str | Path,
) -> Path:

    stage3_path = Path(stage3_path).resolve()
    output_path = Path(output_path).resolve()

    if not stage3_path.exists():
        raise FileNotFoundError(
            f"Stage 3 manifest not found: {stage3_path}"
        )

    with stage3_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    segments = manifest.get("segments", [])

    if not segments:
        raise ValueError(
            "Stage 3 manifest contains no segments."
        )

    first_audio = stage3_path.parent / segments[0]["audio"]

    if not first_audio.exists():
        raise FileNotFoundError(
            f"Audio file not found: {first_audio}"
        )

    params, _ = _read_wav(first_audio)

    sample_rate = params["sample_rate"]

    # Determine complete timeline from Stage 3 target timings.
    timeline_end = max(
        float(segment["start"]) +
        float(segment["target_duration"])
        for segment in segments
    )

    total_frames = round(
        timeline_end * sample_rate
    )

    # Start with silence.
    assembled = np.zeros(
        total_frames,
        dtype=np.int16,
    )

    for segment in segments:

        audio_path = (
            stage3_path.parent /
            segment["audio"]
        )

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Segment audio not found: {audio_path}"
            )

        segment_params, audio = _read_wav(
            audio_path
        )

        if (
            segment_params["sample_rate"]
            != sample_rate
        ):
            raise ValueError(
                f"Sample-rate mismatch in "
                f"{audio_path}: "
                f"{segment_params['sample_rate']} "
                f"!= {sample_rate}"
            )

        start = float(segment["start"])

        start_frame = round(
            start * sample_rate
        )

        end_frame = start_frame + len(audio)

        # Keep audio inside final timeline.
        end_frame = min(
            end_frame,
            total_frames,
        )

        if end_frame <= start_frame:
            continue

        audio_length = (
            end_frame - start_frame
        )

        assembled[
            start_frame:end_frame
        ] = audio[:audio_length]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write standard 16-bit PCM WAV.
    with wave.open(
        str(output_path),
        "wb",
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        wf.writeframes(
            assembled.tobytes()
        )

    actual_duration = (
        len(assembled) / sample_rate
    )

    print(
        f"Sample rate: {sample_rate} Hz"
    )

    print(
        f"Timeline duration: "
        f"{timeline_end:.3f} s"
    )

    print(
        f"Output duration: "
        f"{actual_duration:.3f} s"
    )

    print(
        f"Output: {output_path}"
    )

    return output_path