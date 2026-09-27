"""Chatterbox Multilingual TTS backend for DubSync."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import torchaudio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

from dubsync.timing.duration import (
    duration_error,
    duration_error_ratio,
    get_audio_duration,
    is_acceptable,
)
from dubsync.timing.gemini_adapter import (
    GeminiDurationAdapter,
)


# ------------------------------------------------------------------
# Windows UTF-8 console handling
# ------------------------------------------------------------------

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")


class ChatterboxTTS:
    """Chatterbox Multilingual V3 TTS backend."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "cuda",
    ):
        self.model_dir = (
            Path(model_dir).resolve()
            if model_dir is not None
            else None
        )

        self.device = device

        print("Loading Chatterbox Multilingual V3...")

        self.model = ChatterboxMultilingualTTS.from_pretrained(
            device=torch.device(device),
            t3_model="v3",
        )

        print("Chatterbox Multilingual V3 loaded.")

    def prepare_voice(
        self,
        reference_audio: str | Path,
        *,
        exaggeration: float = 0.5,
    ) -> None:
        """Prepare speaker conditioning from reference audio."""

        reference_audio = Path(reference_audio).resolve()

        if not reference_audio.is_file():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio}"
            )

        print("Preparing speaker conditioning...")

        self.model.prepare_conditionals(
            str(reference_audio),
            exaggeration=exaggeration,
        )

        print("Speaker conditioning ready.")

    def generate(
        self,
        text: str,
        *,
        language_id: str,
        reference_audio: str | Path,
    ):
        """Generate speech using the reference voice."""

        reference_audio = Path(reference_audio).resolve()

        if not reference_audio.is_file():
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio}"
            )

        return self.model.generate(
            text=text,
            language_id=language_id,
            audio_prompt_path=str(reference_audio),
        )

    def save(
        self,
        wav,
        output_path: str | Path,
    ) -> float:
        """Save generated audio and return its duration."""

        output_path = Path(output_path).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torchaudio.save(
            str(output_path),
            wav.cpu(),
            self.model.sr,
        )

        return wav.shape[-1] / self.model.sr


def generate_segment(
    tts: ChatterboxTTS,
    text: str,
    language_id: str,
    reference_audio: str | Path,
    output_path: str | Path,
) -> float:
    """Generate one TTS segment and return its duration."""

    output_path = Path(output_path).resolve()

    generation_start = time.time()

    wav = tts.generate(
        text,
        language_id=language_id,
        reference_audio=reference_audio,
    )

    generation_time = time.time() - generation_start

    duration = tts.save(
        wav,
        output_path,
    )

    print(
        f"Generated TTS: {output_path.name}"
    )

    print(
        f"Duration: {duration:.2f} seconds"
    )

    print(
        f"Generation time: {generation_time:.2f} seconds"
    )

    return duration


def create_reference_audio(
    source_audio: str | Path,
    reference_audio: str | Path,
    *,
    target_sample_rate: int = 24000,
) -> Path:
    """Create a Chatterbox reference audio file from source audio."""

    source_audio = Path(source_audio).resolve()
    reference_audio = Path(reference_audio).resolve()

    if not source_audio.is_file():
        raise FileNotFoundError(
            f"Source audio not found: {source_audio}"
        )

    print(
        f"Creating reference audio from: {source_audio}"
    )

    wav, sample_rate = torchaudio.load(
        str(source_audio)
    )

    # Convert to mono.
    if wav.shape[0] > 1:
        wav = wav.mean(
            dim=0,
            keepdim=True,
        )

    # Resample to Chatterbox's 24 kHz reference format.
    if sample_rate != target_sample_rate:
        wav = torchaudio.functional.resample(
            wav,
            sample_rate,
            target_sample_rate,
        )

    reference_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torchaudio.save(
        str(reference_audio),
        wav,
        target_sample_rate,
    )

    print(
        f"Created reference audio: {reference_audio}"
    )

    print(
        f"Reference sample rate: "
        f"{target_sample_rate} Hz"
    )

    print(
        f"Reference duration: "
        f"{wav.shape[-1] / target_sample_rate:.2f} seconds"
    )

    return reference_audio


# ------------------------------------------------------------------
# Final timing correction
# ------------------------------------------------------------------

def _atempo_filter(speed_factor: float) -> str:
    """
    Build an FFmpeg atempo filter.

    FFmpeg supports 0.5-2.0 per atempo filter.
    Chaining filters allows larger corrections.
    """

    if speed_factor <= 0:
        raise ValueError(
            f"Invalid speed factor: {speed_factor}"
        )

    filters = []

    remaining = speed_factor

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining:.8f}")

    return ",".join(filters)


def time_stretch_to_target(
    input_audio: str | Path,
    output_audio: str | Path,
    target_duration: float,
    *,
    ffmpeg_executable: str = "ffmpeg",
) -> float:
    """
    Pitch-preserving final timing correction using FFmpeg atempo.

    This should only be used for relatively small residual timing
    errors after Gemini text adaptation.
    """

    input_audio = Path(input_audio).resolve()
    output_audio = Path(output_audio).resolve()

    current_duration = get_audio_duration(
        input_audio
    )

    if current_duration <= 0:
        raise ValueError(
            f"Invalid audio duration: {input_audio}"
        )

    # atempo speed factor:
    #
    # factor > 1 => faster / shorter
    # factor < 1 => slower / longer
    #
    speed_factor = (
        current_duration / target_duration
    )

    filter_expression = _atempo_filter(
        speed_factor
    )

    output_audio.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(input_audio),
        "-filter:a",
        filter_expression,
        "-vn",
        str(output_audio),
    ]

    print(
        f"Applying final time correction: "
        f"{current_duration:.2f}s -> "
        f"{target_duration:.2f}s"
    )

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )

    final_duration = get_audio_duration(
        output_audio
    )

    return final_duration


def run_stage3(
    stage2_path: str | Path,
    output_dir: str | Path,
    reference_audio: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
    language_id: str = "hi",
    device: str = "cuda",
    ffmpeg_executable: str = "ffmpeg",
    enable_gemini_adaptation: bool = True,
    max_gemini_attempts: int = 2,
    timing_tolerance: float = 0.75,
    final_stretch_limit: float = 0.15,
) -> Path:
    """
    Generate TTS audio for every Stage 2 dialogue segment.

    Pipeline:

        translation
            ↓
        Chatterbox
            ↓
        duration measurement
            ↓
        Gemini adaptation if necessary
            ↓
        Chatterbox regeneration
            ↓
        small final time correction
    """

    stage2_path = Path(stage2_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not stage2_path.is_file():
        raise FileNotFoundError(
            f"Stage 2 file not found: {stage2_path}"
        )

    # ---------------------------------------------------------
    # Reference audio
    # ---------------------------------------------------------

    if reference_audio is None:

        source_audio = (
            output_dir
            / "source_16khz_mono.wav"
        )

        reference_audio = (
            output_dir
            / "reference.wav"
        )

        if not source_audio.is_file():
            raise FileNotFoundError(
                f"Source audio not found: {source_audio}"
            )

        create_reference_audio(
            source_audio,
            reference_audio,
        )

    else:

        reference_audio = Path(
            reference_audio
        ).resolve()

        if not reference_audio.is_file():
            raise FileNotFoundError(
                f"Reference audio not found: "
                f"{reference_audio}"
            )

    # ---------------------------------------------------------
    # Load Stage 2
    # ---------------------------------------------------------

    with stage2_path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    segments = data["dialogue_segments"]

    # ---------------------------------------------------------
    # Load Chatterbox
    # ---------------------------------------------------------

    tts = ChatterboxTTS(
        model_dir=model_dir,
        device=device,
    )

    # ---------------------------------------------------------
    # Load Gemini only if enabled
    # ---------------------------------------------------------

    gemini = None

    if enable_gemini_adaptation:

        try:

            gemini = GeminiDurationAdapter()

            print(
                "Gemini duration adaptation enabled."
            )

        except Exception as exc:

            print(
                "WARNING: Gemini duration adaptation "
                f"disabled: {exc}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    # ---------------------------------------------------------
    # Generate every dialogue segment
    # ---------------------------------------------------------

    for index, segment in enumerate(
        segments,
        start=1,
    ):

        original_text = segment[
            "translated_text"
        ]

        current_text = original_text

        source_text = segment[
            "source_text"
        ]

        start = float(
            segment["start"]
        )

        end = float(
            segment["end"]
        )

        target_duration = end - start

        output_file = (
            output_dir
            / f"segment_{index:03d}.wav"
        )

        raw_output_file = (
            output_dir
            / f"segment_{index:03d}_raw.wav"
        )

        print("\n" + "=" * 80)

        print(
            f"TTS SEGMENT "
            f"{index}/{len(segments)}"
        )

        print("=" * 80)

        print(
            f"Source: {source_text}"
        )

        print(
            f"Translation: {current_text}"
        )

        print(
            f"Target duration: "
            f"{target_duration:.2f} seconds"
        )

        # -----------------------------------------------------
        # Initial Chatterbox generation
        # -----------------------------------------------------

        generation_start = time.time()

        wav = tts.generate(
            current_text,
            language_id=language_id,
            reference_audio=reference_audio,
        )

        generation_time = (
            time.time()
            - generation_start
        )

        generated_duration = tts.save(
            wav,
            output_file,
        )

        # Preserve original generation if we need adaptation.
        shutil.copy2(
            output_file,
            raw_output_file,
        )

        gemini_attempts = 0
        adapted = False

        # -----------------------------------------------------
        # Gemini adaptation loop
        # -----------------------------------------------------

        while (
            gemini is not None
            and not is_acceptable(
                generated_duration,
                target_duration,
                timing_tolerance,
            )
            and gemini_attempts < max_gemini_attempts
        ):

            gemini_attempts += 1

            print("\n" + "-" * 70)

            print(
                f"Gemini adaptation "
                f"{gemini_attempts}/"
                f"{max_gemini_attempts}"
            )

            print(
                f"Current duration: "
                f"{generated_duration:.2f}s"
            )

            print(
                f"Target duration: "
                f"{target_duration:.2f}s"
            )

            # -------------------------------------------------
            # Ask Gemini to rewrite the text.
            # -------------------------------------------------

            revised_text = gemini.adapt(
                source_text=source_text,
                translated_text=current_text,
                target_duration=target_duration,
                generated_duration=generated_duration,
                target_language="Hindi",
            )

            print(
                f"Gemini revised text:\n"
                f"{revised_text}"
            )

            current_text = revised_text

            # -------------------------------------------------
            # Generate again using same reference voice.
            # -------------------------------------------------

            generation_start = time.time()

            wav = tts.generate(
                current_text,
                language_id=language_id,
                reference_audio=reference_audio,
            )

            generation_time += (
                time.time()
                - generation_start
            )

            generated_duration = tts.save(
                wav,
                output_file,
            )

            adapted = True

            print(
                f"New generated duration: "
                f"{generated_duration:.2f}s"
            )

        # -----------------------------------------------------
        # Final duration correction
        # -----------------------------------------------------

        duration_error = (
            generated_duration
            - target_duration
        )

        duration_ratio = (
            duration_error / target_duration
            if target_duration > 0
            else 0.0
        )

        # Only use time-stretch when the remaining error is
        # reasonably small. We do NOT want to stretch something
        # like 5.6s -> 11s.
        stretch_ratio = abs(
            duration_ratio
        )

        final_duration = generated_duration
        time_stretched = False

        if (
            not is_acceptable(
                generated_duration,
                target_duration,
                timing_tolerance,
            )
            and stretch_ratio <= final_stretch_limit
        ):

            corrected_file = (
                output_dir
                / f"segment_{index:03d}_corrected.wav"
            )

            final_duration = time_stretch_to_target(
                output_file,
                corrected_file,
                target_duration,
                ffmpeg_executable=ffmpeg_executable,
            )

            # Replace the normal segment with corrected version.
            shutil.move(
                corrected_file,
                output_file,
            )

            time_stretched = True

        # -----------------------------------------------------
        # Final measurements
        # -----------------------------------------------------

        final_duration = get_audio_duration(
            output_file
        )

        final_error = (
            final_duration
            - target_duration
        )

        final_error_ratio = (
            final_error / target_duration
            if target_duration > 0
            else 0.0
        )

        if is_acceptable(
            final_duration,
            target_duration,
            timing_tolerance,
        ):

            timing_status = "acceptable"

        else:

            timing_status = "needs_review"

        print(
            f"\nFinal duration: "
            f"{final_duration:.2f} seconds"
        )

        print(
            f"Final difference: "
            f"{final_error:+.2f} seconds"
        )

        print(
            f"Timing status: "
            f"{timing_status}"
        )

        print(
            f"Gemini adaptations: "
            f"{gemini_attempts}"
        )

        print(
            f"Final time stretch: "
            f"{time_stretched}"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} seconds"
        )

        # -----------------------------------------------------
        # Manifest
        # -----------------------------------------------------

        results.append(
            {
                "index": index,
                "start": start,
                "end": end,

                "target_duration": target_duration,

                "original_translated_text": (
                    original_text
                ),

                "final_translated_text": (
                    current_text
                ),

                "text_was_adapted": adapted,

                "gemini_attempts": (
                    gemini_attempts
                ),

                "raw_generated_duration": (
                    get_audio_duration(
                        raw_output_file
                    )
                ),

                "generated_duration": (
                    generated_duration
                ),

                "final_duration": (
                    final_duration
                ),

                "generation_time": (
                    generation_time
                ),

                "duration_error": (
                    final_error
                ),

                "duration_error_ratio": (
                    final_error_ratio
                ),

                "time_stretched": (
                    time_stretched
                ),

                "timing_status": (
                    timing_status
                ),

                "source_text": (
                    source_text
                ),

                "translated_text": (
                    current_text
                ),

                "audio": (
                    output_file.name
                ),

                "raw_audio": (
                    raw_output_file.name
                ),
            }
        )

    # ---------------------------------------------------------
    # Write Stage 3 manifest
    # ---------------------------------------------------------

    manifest_path = (
        output_dir
        / "stage3_tts.json"
    )

    manifest = {
        "schema_version": 2,
        "language": language_id,
        "reference_audio": str(
            reference_audio
        ),
        "tts_model": (
            "Chatterbox Multilingual V3"
        ),
        "duration_adapter": (
            "Gemini"
            if gemini is not None
            else None
        ),
        "timing_units": "seconds",
        "segments": results,
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)

    print("STAGE 3 COMPLETE")

    print("=" * 80)

    print(
        f"Manifest: {manifest_path}"
    )

    return manifest_path