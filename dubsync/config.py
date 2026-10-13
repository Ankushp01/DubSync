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
    max_gap_seconds: float = 0.75
    max_characters: int = 400
    preferred_min_duration_seconds: float = 8.0
    preferred_max_duration_seconds: float = 10.0
    hard_max_duration_seconds: float = 12.0
    meaningful_pause_seconds: float = 0.75
    min_duration_seconds: float = 8.0

    def __init__(
        self,
        max_gap_seconds: float = 0.75,
        max_characters: int = 400,
        preferred_min_duration_seconds: float = 8.0,
        preferred_max_duration_seconds: float = 10.0,
        hard_max_duration_seconds: float = 12.0,
        meaningful_pause_seconds: float = 0.75,
        min_duration_seconds: float | None = None,
        **kwargs,
    ):
        self.max_gap_seconds = max_gap_seconds
        self.max_characters = max_characters
        self.preferred_min_duration_seconds = preferred_min_duration_seconds
        self.preferred_max_duration_seconds = preferred_max_duration_seconds
        self.hard_max_duration_seconds = hard_max_duration_seconds
        self.meaningful_pause_seconds = meaningful_pause_seconds
        self.min_duration_seconds = (
            min_duration_seconds
            if min_duration_seconds is not None
            else preferred_min_duration_seconds
        )

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
