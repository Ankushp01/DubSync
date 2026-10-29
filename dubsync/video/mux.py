from __future__ import annotations

import re
import subprocess
from pathlib import Path


def _probe_duration(media_path: Path, ffmpeg_executable: Path) -> float:
    ffprobe_candidate = (
        ffmpeg_executable.parent / "ffprobe.exe"
        if ffmpeg_executable.suffix.lower() == ".exe"
        else ffmpeg_executable.parent / "ffprobe"
    )
    if ffprobe_candidate.is_file():
        cmd = [
            str(ffprobe_candidate),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        val = res.stdout.strip()
        if val:
            return float(val)

    cmd = [str(ffmpeg_executable), "-i", str(media_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", res.stderr)
    if m:
        h, m_m, s = float(m.group(1)), float(m.group(2)), float(m.group(3))
        return h * 3600 + m_m * 60 + s
    return 0.0


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

    video_dur = _probe_duration(video_path, ffmpeg_executable)
    audio_dur = _probe_duration(audio_path, ffmpeg_executable)

    if audio_dur > video_dur + 0.05:
        pad_sec = round(audio_dur - video_dur, 3)
        print(f"Dubbed audio ({audio_dur:.2f}s) exceeds video ({video_dur:.2f}s). Extending video by {pad_sec:.2f}s (cloning final frame)...")
        command = [
            str(ffmpeg_executable),
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[0:v]tpad=stop_mode=clone:stop_duration={pad_sec}[v]",
            "-map",
            "[v]",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(output_path),
        ]
    else:
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
