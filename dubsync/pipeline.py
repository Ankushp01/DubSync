"""Pipeline orchestration for transcription and translation."""
from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess

from dubsync.audio.extraction import extract_audio
from dubsync.config import AppConfig
from dubsync.models import Transcript
from dubsync.segmentation.dialogue import group_dialogue
from dubsync.stt.whisper import WhisperTranscriber
from .translation.translator import (
    IndicTrans2Translator,
)
from dubsync.audio.assembly import assemble_stage3_audio
from dubsync.video.mux import mux_dubbed_audio

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TTS_PYTHON = (
    PROJECT_ROOT
    / ".tts-venv"
    / "Scripts"
    / "python.exe"
)

TTS_WORKER = (
    Path(__file__).resolve().parent
    / "tts"
    / "chatterbox_worker.py"
)


def run_stage1(video_path: str | Path, config: AppConfig) -> Path:
    """Generate a stable Stage 1 JSON artifact and return its path."""
    video = Path(video_path).resolve()

    job_dir = config.paths.output_dir / video.stem

    source_audio_path = job_dir / "source_16khz_mono.wav"
    transcript_path = job_dir / "stage1_transcript.json"

    extract_audio(
        video,
        source_audio_path,
        ffmpeg_executable=config.ffmpeg.executable,
        sample_rate=config.audio.sample_rate,
        channels=config.audio.channels,
    )

    transcriber = WhisperTranscriber(
        device=config.runtime.device,
        compute_type=config.runtime.compute_type,
        model_name=config.whisper.model_name,
        model_path=config.whisper.model_path,
    )

    transcript = transcriber.transcribe(
        source_audio_path,
        language=config.whisper.language,
        beam_size=config.whisper.beam_size,
        vad_filter=config.whisper.vad_filter,
    )

    _write_stage1(transcript_path, transcript, config)

    return transcript_path


def run_stage2(
    stage1_path: str | Path,
    config: AppConfig,
) -> Path:
    """Translate Stage 1 dialogue segments and write Stage 2 JSON."""

    stage1_file = Path(stage1_path).resolve()

    if not stage1_file.is_file():
        raise FileNotFoundError(
            f"Stage 1 transcript not found: "
            f"{stage1_file}"
        )

    stage2_path = (
        stage1_file.parent
        / "stage2_translation.json"
    )

    data = json.loads(
        stage1_file.read_text(
            encoding="utf-8"
        )
    )

    translator = IndicTrans2Translator(
        source_language=(
            config.translation.source_language
        ),
        target_language=(
            config.translation.target_language
        ),
    )

    source_texts = [
        segment["text"]
        for segment in data["dialogue_segments"]
    ]

    translations = translator.translate_many(
        source_texts
    )

    translated_segments = []

    for segment, translated_text in zip(
        data["dialogue_segments"],
        translations,
    ):
        translated_segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "source_text": segment["text"],
                "translated_text": translated_text,
            }
        )

    payload = {
        "schema_version": 1,
        "source_language": (
            config.translation.source_language
        ),
        "target_language": (
            config.translation.target_language
        ),
        "translation_model": (
            config.translation.model_name
        ),
        "dialogue_segments": translated_segments,
        "timing_units": "seconds",
    }

    stage2_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return stage2_path

def run_stage3(
    stage2_path: str | Path,
    config: AppConfig,
    reference_audio: str | Path | None = None,
) -> Path:
    """Run Chatterbox Stage 3 in the dedicated .tts-venv."""

    stage2_file = Path(stage2_path).resolve()

    if not stage2_file.is_file():
        raise FileNotFoundError(
            f"Stage 2 translation not found: {stage2_file}"
        )

    if not TTS_PYTHON.exists():
        raise FileNotFoundError(
            f"Chatterbox Python environment not found: "
            f"{TTS_PYTHON}"
        )

    if not TTS_WORKER.exists():
        raise FileNotFoundError(
            f"Chatterbox worker not found: "
            f"{TTS_WORKER}"
        )

    output_dir = stage2_file.parent


    payload = {
        "stage2_path": str(stage2_file),
        "output_dir": str(output_dir),
        "reference_audio": (
            str(Path(reference_audio).resolve())
            if reference_audio is not None
            else None
        ),
        "model_dir": str(config.tts.model_dir),
        "language_id": config.tts.language_id,
        "device": config.tts.device,
        "ffmpeg_executable": config.ffmpeg.executable,
        "enable_gemini_adaptation": True,
        "max_gemini_attempts": 2,
    }
        
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    process = subprocess.run(
        [
            str(TTS_PYTHON),
            str(TTS_WORKER),
        ],
        input=json.dumps(
            payload,
            ensure_ascii=False,
        ),
        text=True,
        capture_output=True,
        encoding="utf-8",
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    if process.returncode != 0:
        raise RuntimeError(
            "Chatterbox Stage 3 failed.\n\n"
            f"STDOUT:\n{process.stdout}\n\n"
            f"STDERR:\n{process.stderr}"
        )

    try:
        result = json.loads(
            process.stdout.strip().splitlines()[-1]
        )
    except (json.JSONDecodeError, IndexError) as exc:
        raise RuntimeError(
            "Chatterbox worker returned invalid JSON.\n\n"
            f"STDOUT:\n{process.stdout}\n\n"
            f"STDERR:\n{process.stderr}"
        ) from exc

    manifest_path = result.get("manifest_path")

    if not manifest_path:
        raise RuntimeError(
            "Chatterbox worker did not return "
            "'manifest_path'."
        )

    return Path(manifest_path)

def run_stage4(
    stage3_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Assemble Stage 3 TTS segments into one timeline-aligned WAV.
    """

    stage3_path = Path(stage3_path).resolve()
    output_dir = Path(output_dir).resolve()

    output_path = (
        output_dir
        / "audio"
        / "dubbed_audio.wav"
    )

    return assemble_stage3_audio(
        stage3_path=stage3_path,
        output_path=output_path,
    )

def run_stage5(
    video_path: str | Path,
    dubbed_audio_path: str | Path,
    output_dir: str | Path,
    ffmpeg_executable: str | Path,
) -> Path:
    output_dir = Path(output_dir).resolve()

    output_path = (
        output_dir
        / "video"
        / "dubbed_video.mp4"
    )

    return mux_dubbed_audio(
        video_path=video_path,
        audio_path=dubbed_audio_path,
        output_path=output_path,
        ffmpeg_executable=ffmpeg_executable,
    )

def _write_stage1(
    path: Path,
    transcript: Transcript,
    config: AppConfig,
) -> None:
    dialogue = group_dialogue(
        transcript.segments,
        max_gap_seconds=config.segmentation.max_gap_seconds,
        max_characters=config.segmentation.max_characters,
        min_duration_seconds=config.segmentation.min_duration_seconds,
        preferred_max_duration_seconds=config.segmentation.preferred_max_duration_seconds,
        hard_max_duration_seconds=config.segmentation.hard_max_duration_seconds,
    )

    payload = {
        "schema_version": 1,
        "source_language": transcript.language,
        "asr_segments": [
            item.to_dict()
            for item in transcript.segments
        ],
        "dialogue_segments": [
            item.to_dict()
            for item in dialogue
        ],
        "timing_units": "seconds",
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
