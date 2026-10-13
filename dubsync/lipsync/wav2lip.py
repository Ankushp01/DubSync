"""Wav2Lip-HD lip synchronization execution wrapper."""
from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WAV2LIP_DIR = PROJECT_ROOT / "Wav2Lip-HD"
WAV2LIP_PYTHON = PROJECT_ROOT / ".wav2lip-venv" / "Scripts" / "python.exe"


def run_wav2lip_hd(
    video_path: str | Path,
    audio_path: str | Path,
    output_path: str | Path,
    progress_callback: Optional[Callable[[str, int], None]] = None,
    resize_factor: int = 1,
) -> Path:
    """Execute Wav2Lip-HD inference in the dedicated .wav2lip-venv environment.

    Args:
        video_path: Source input video file.
        audio_path: Dubbed timeline-aligned audio file.
        output_path: Target path for the lip-synced MP4 video.
        progress_callback: Optional callback receiving (status_message, progress_percent).
        resize_factor: Factor to reduce resolution (default 1).

    Returns:
        Path to the generated lip-synchronized video.
    """
    video = Path(video_path).resolve()
    audio = Path(audio_path).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not video.is_file():
        raise FileNotFoundError(f"Input video not found: {video}")
    if not audio.is_file():
        raise FileNotFoundError(f"Input audio not found: {audio}")
    if not WAV2LIP_PYTHON.is_file():
        raise FileNotFoundError(f"Wav2Lip Python interpreter not found: {WAV2LIP_PYTHON}")
    if not WAV2LIP_DIR.is_dir():
        raise FileNotFoundError(f"Wav2Lip-HD directory not found: {WAV2LIP_DIR}")

    # Checkpoint path for standard Wav2Lip only
    checkpoint = WAV2LIP_DIR / "checkpoints" / "wav2lip.pth"
    if not checkpoint.is_file():
        checkpoint = WAV2LIP_DIR / "wav2lip_models" / "wav2lip.pth"

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Wav2Lip checkpoint not found: {checkpoint}")

    cmd = [
        str(WAV2LIP_PYTHON),
        "inference.py",
        "--checkpoint_path", str(checkpoint),
        "--face", str(video),
        "--audio", str(audio),
        "--outfile", str(output),
        "--resize_factor", str(resize_factor),
        "--no_sr",
        "--no_segmentation",
    ]

    logger.info(f"Starting Vanilla Wav2Lip inference: {' '.join(cmd)}")
    if progress_callback:
        progress_callback("Initializing Vanilla Wav2Lip pipeline...", 91)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        cwd=str(WAV2LIP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    tqdm_pattern = re.compile(r"(\d+)%\|.*\|\s*(\d+)/(\d+)")

    lines = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            line_str = line.strip()
            lines.append(line_str)
            logger.debug(f"[Wav2Lip] {line_str}")

            if progress_callback:
                if "Using cuda for inference" in line_str:
                    progress_callback("CUDA detected. Initializing vanilla Wav2Lip inference...", 91)
                elif "Loading standard Vanilla Wav2Lip model" in line_str:
                    progress_callback("Loading standard Wav2Lip neural model...", 93)
                elif "Standard Vanilla Wav2Lip model loaded" in line_str:
                    progress_callback("Vanilla Wav2Lip model ready (no ESRGAN/segmentation). Generating frames...", 95)
                elif "Encoding final lip-synced video" in line_str:
                    progress_callback("Encoding final lip-synced video with FFmpeg...", 98)
                else:
                    match = tqdm_pattern.search(line_str)
                    if match:
                        current_pct = int(match.group(1))
                        # Map 0-100% of batch progress into 95-98% overall progress
                        mapped_progress = 95 + int((current_pct / 100.0) * 3)
                        progress_callback(
                            f"Generating lip-synced frames ({match.group(2)}/{match.group(3)} batches)...",
                            mapped_progress,
                        )

    return_code = process.wait()
    if return_code != 0:
        tail = "\n".join(lines[-30:])
        raise RuntimeError(
            f"Wav2Lip-HD inference failed with exit code {return_code}.\n"
            f"Output log:\n{tail}"
        )

    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(
            f"Wav2Lip-HD completed with code 0 but output file does not exist or is empty: {output}"
        )

    logger.info(f"Wav2Lip-HD inference complete: {output} ({output.stat().st_size} bytes)")
    if progress_callback:
        progress_callback("Lip synchronization completed successfully", 99)

    return output
