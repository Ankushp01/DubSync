"""Typed configuration for the local pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class PathsConfig:
    output_dir: Path
    model_dir: Path


@dataclass(frozen=True)
class RuntimeConfig:
    device: str
    compute_type: str


@dataclass(frozen=True)
class FfmpegConfig:
    executable: str


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int


@dataclass(frozen=True)
class WhisperConfig:
    model_name: str
    model_path: str
    language: str | None
    beam_size: int
    vad_filter: bool


@dataclass
class SegmentationConfig:
    max_gap_seconds: float
    max_characters: int
    min_duration_seconds: float
    preferred_max_duration_seconds: float
    hard_max_duration_seconds: float

@dataclass(frozen=True)
class TranslationConfig:
    model_name: str
    source_language: str
    target_language: str

@dataclass(frozen=True)
class TTSConfig:
    model_dir: str
    language_id: str
    device: str

@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    runtime: RuntimeConfig
    ffmpeg: FfmpegConfig
    audio: AudioConfig
    whisper: WhisperConfig
    segmentation: SegmentationConfig
    translation: TranslationConfig
    tts: TTSConfig


def load_config(config_path: str | Path) -> AppConfig:
    """Load TOML configuration, resolving relative paths from its parent."""
    path = Path(config_path).resolve()
    with path.open("rb") as file:
        data = tomllib.load(file)
    root = path.parent.parent
    paths = data["paths"]
    whisper = data["whisper"]
    translation = data["translation"]
    tts = data["tts"]
    return AppConfig(
        paths=PathsConfig(
            output_dir=_resolve(root, paths["output_dir"]),
            model_dir=_resolve(root, paths["model_dir"]),
        ),
        runtime=RuntimeConfig(**data["runtime"]),
        ffmpeg=FfmpegConfig(**data["ffmpeg"]),
        audio=AudioConfig(**data["audio"]),
        whisper=WhisperConfig(
            model_name=whisper["model_name"],
            model_path=whisper["model_path"],
            language=whisper["language"] or None,
            beam_size=whisper["beam_size"],
            vad_filter=whisper["vad_filter"],
        ),
        segmentation=SegmentationConfig(**data["segmentation"]),
        translation=TranslationConfig(
            model_name=translation["model_name"],
            source_language=translation["source_language"],
            target_language=translation["target_language"],
        ),
        tts=TTSConfig(
            model_dir=tts["model_dir"],
            language_id=tts["language_id"],
            device=tts["device"],
        ),
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path
