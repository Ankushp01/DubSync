"""Chatterbox Multilingual TTS backend for DubSync."""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import torch
import torchaudio

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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

        # Chatterbox Multilingual V3 must be loaded
        # through from_pretrained().
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


def run_stage3(
    stage2_path: str | Path,
    output_dir: str | Path,
    reference_audio: str | Path | None = None,
    *,
    model_dir: str | Path | None = None,
    language_id: str = "hi",
    device: str = "cuda",
) -> Path:
    """Generate TTS audio for every Stage 2 dialogue segment."""

    stage2_path = Path(stage2_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not stage2_path.is_file():
        raise FileNotFoundError(
            f"Stage 2 file not found: {stage2_path}"
        )

    # ---------------------------------------------------------
    # Reference audio
    # ---------------------------------------------------------
    #
    # If the user supplied a reference audio file, use it.
    #
    # Otherwise automatically create:
    #
    # source_16khz_mono.wav
    #          ↓
    #     24 kHz mono
    #          ↓
    #      reference.wav
    #
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

        text = segment["translated_text"]

        start = float(segment["start"])
        end = float(segment["end"])

        target_duration = end - start

        output_file = (
            output_dir
            / f"segment_{index:03d}.wav"
        )

        print("\n" + "=" * 80)

        print(
            f"TTS SEGMENT "
            f"{index}/{len(segments)}"
        )

        print("=" * 80)

        print(text)

        print(
            f"Target duration: "
            f"{target_duration:.2f} seconds"
        )

        generation_start = time.time()

        wav = tts.generate(
            text,
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

        duration_error = (
            generated_duration
            - target_duration
        )

        duration_error_ratio = (
            duration_error / target_duration
            if target_duration > 0
            else 0.0
        )

        TIMING_TOLERANCE = 0.75

        if abs(duration_error) <= TIMING_TOLERANCE:
            timing_status = "acceptable"
        else:
            timing_status = "needs_review"

        print(
            f"Generated duration: "
            f"{generated_duration:.2f} seconds"
        )

        print(
            f"Difference: "
            f"{duration_error:+.2f} seconds"
        )

        print(
            f"Timing status: "
            f"{timing_status}"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} seconds"
        )

        results.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "target_duration": target_duration,
                "generated_duration": generated_duration,
                "generation_time": generation_time,
                "duration_error": duration_error,
                "duration_error_ratio": duration_error_ratio,
                "timing_status": timing_status,
                "source_text": segment["source_text"],
                "translated_text": text,
                "audio": output_file.name,
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
        "schema_version": 1,
        "language": language_id,
        "reference_audio": str(reference_audio),
        "tts_model": "Chatterbox Multilingual V3",
        "segments": results,
        "timing_units": "seconds",
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