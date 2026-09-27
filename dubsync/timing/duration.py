"""Audio duration utilities for DubSync."""

from __future__ import annotations

from pathlib import Path

import torchaudio


def get_audio_duration(audio_path: str | Path) -> float:
    """Return audio duration in seconds."""

    audio_path = Path(audio_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    info = torchaudio.info(str(audio_path))

    if info.sample_rate <= 0:
        raise ValueError(
            f"Invalid sample rate for audio: {audio_path}"
        )

    return info.num_frames / info.sample_rate


def duration_error(
    generated_duration: float,
    target_duration: float,
) -> float:
    """Return generated - target duration."""

    return generated_duration - target_duration


def duration_error_ratio(
    generated_duration: float,
    target_duration: float,
) -> float:
    """Return relative duration error."""

    if target_duration <= 0:
        return 0.0

    return (
        generated_duration - target_duration
    ) / target_duration


def is_acceptable(
    generated_duration: float,
    target_duration: float,
    tolerance_seconds: float = 0.75,
) -> bool:
    """Return whether duration is within the allowed absolute tolerance."""

    return abs(
        generated_duration - target_duration
    ) <= tolerance_seconds