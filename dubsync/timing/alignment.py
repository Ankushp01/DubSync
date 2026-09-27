"""Audio duration alignment utilities for DubSync."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass(frozen=True)
class AlignmentResult:
    """Result of comparing generated audio against its target duration."""

    status: str
    generated_duration: float
    target_duration: float
    ratio: float
    difference_ratio: float


# Duration thresholds.
ACCEPTABLE_RATIO = 0.10
MAX_TIME_STRETCH_RATIO = 0.25


def analyze_duration(
    generated_duration: float,
    target_duration: float,
) -> AlignmentResult:
    """Determine how generated audio should be handled."""

    if generated_duration <= 0:
        raise ValueError(
            "generated_duration must be greater than zero."
        )

    if target_duration <= 0:
        raise ValueError(
            "target_duration must be greater than zero."
        )

    ratio = target_duration / generated_duration

    difference_ratio = abs(
        generated_duration - target_duration
    ) / target_duration

    if difference_ratio <= ACCEPTABLE_RATIO:
        status = "acceptable"

    elif difference_ratio <= MAX_TIME_STRETCH_RATIO:
        status = "time_stretch"

    else:
        status = "needs_rewrite"

    return AlignmentResult(
        status=status,
        generated_duration=generated_duration,
        target_duration=target_duration,
        ratio=ratio,
        difference_ratio=difference_ratio,
    )


def time_stretch_audio(
    input_path: str | Path,
    output_path: str | Path,
    target_duration: float,
) -> float:
    """Time-stretch audio to approximately the target duration."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Audio file not found: {input_path}"
        )

    if target_duration <= 0:
        raise ValueError(
            "target_duration must be greater than zero."
        )

    waveform, sample_rate = torchaudio.load(
        str(input_path)
    )

    generated_duration = (
        waveform.shape[-1] / sample_rate
    )

    if generated_duration <= 0:
        raise ValueError(
            f"Audio contains no samples: {input_path}"
        )

    # Speed factor:
    #
    # target / generated
    #
    # > 1.0 means the audio needs to become slower.
    # < 1.0 means the audio needs to become faster.
    speed_factor = (
        generated_duration / target_duration
    )

    stretched = torchaudio.functional.phase_vocoder(
        waveform,
        rate=speed_factor,
        phase_advance=torch.linspace(
            0,
            torch.pi * 512,
            513,
            device=waveform.device,
        ),
    )

    # phase_vocoder changes the number of samples.
    # Resample the result slightly so that the final file
    # lands exactly on the requested duration.
    target_samples = round(
        target_duration * sample_rate
    )

    stretched = torchaudio.functional.resample(
        stretched,
        orig_freq=sample_rate,
        new_freq=sample_rate,
    )

    current_samples = stretched.shape[-1]

    if current_samples > target_samples:
        stretched = stretched[:, :target_samples]

    elif current_samples < target_samples:
        padding = torch.zeros(
            stretched.shape[0],
            target_samples - current_samples,
            dtype=stretched.dtype,
            device=stretched.device,
        )

        stretched = torch.cat(
            [stretched, padding],
            dim=-1,
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torchaudio.save(
        str(output_path),
        stretched.cpu(),
        sample_rate,
    )

    return stretched.shape[-1] / sample_rate

if __name__ == "__main__":
    tests = [
        (7.28, 5.56),
        (4.32, 3.40),
        (7.04, 6.06),
        (5.12, 11.00),
        (4.44, 4.02),
        (2.88, 3.40),
    ]

    for generated, target in tests:
        result = analyze_duration(
            generated_duration=generated,
            target_duration=target,
        )

        print(
            f"{generated:.2f}s -> {target:.2f}s | "
            f"{result.status} | "
            f"ratio={result.ratio:.3f} | "
            f"difference={result.difference_ratio:.1%}"
        )