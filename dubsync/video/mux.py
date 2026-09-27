from __future__ import annotations

import subprocess
from pathlib import Path


def mux_dubbed_audio(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    ffmpeg_executable: str | Path,
) -> Path:

    video_path = Path(video_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_path = Path(output_path).resolve()
    ffmpeg_executable = Path(ffmpeg_executable).resolve()

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {video_path}"
        )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Dubbed audio not found: {audio_path}"
        )

    if not ffmpeg_executable.exists():
        raise FileNotFoundError(
            f"FFmpeg not found: {ffmpeg_executable}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(ffmpeg_executable),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),

        # Keep original video untouched.
        "-map",
        "0:v:0",

        # Use assembled dubbed audio.
        "-map",
        "1:a:0",

        # No video re-encoding.
        "-c:v",
        "copy",

        # Encode audio to AAC for MP4 compatibility.
        "-c:a",
        "aac",
        "-b:a",
        "192k",

        # Output duration follows the shorter stream.
        "-shortest",

        str(output_path),
    ]

    print("Muxing dubbed audio with original video...")

    subprocess.run(
        command,
        check=True,
    )

    print(f"Output: {output_path}")

    return output_path