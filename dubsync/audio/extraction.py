"""Audio extraction using a separately installed FFmpeg executable."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess


class AudioExtractionError(RuntimeError):
    """Raised when FFmpeg cannot extract a usable working audio file."""


def extract_audio(
    video_path: str | Path,
    audio_path: str | Path,
    *,
    ffmpeg_executable: str = "ffmpeg",
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    """Extract lossless mono WAV suitable for ASR, overwriting only its target."""
    source = Path(video_path)
    target = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    executable = _find_executable(ffmpeg_executable)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable, "-y", "-i", str(source), "-vn", "-ac", str(channels),
        "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(target),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        detail = completed.stderr.strip()[-1000:]
        raise AudioExtractionError(f"FFmpeg audio extraction failed. {detail}")
    return target


def _find_executable(value: str) -> str:
    if Path(value).is_file():
        return value
    found = shutil.which(value)
    if found:
        return found
    raise AudioExtractionError(
        "FFmpeg was not found. Install FFmpeg and add it to PATH, or set "
        "[ffmpeg].executable to its full path in configs/local.toml."
    )
