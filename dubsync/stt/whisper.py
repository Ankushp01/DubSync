import inspect
import os
from pathlib import Path

from faster_whisper import WhisperModel

from dubsync.models import TimedText, Transcript


def resolve_cached_whisper_snapshot(model_name: str = "small") -> Path | None:
    """Locate cached local snapshot of Faster-Whisper model in HuggingFace cache."""
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_folder = f"models--Systran--faster-whisper-{model_name}"
    snapshots_dir = cache_dir / repo_folder / "snapshots"
    if snapshots_dir.is_dir():
        for s in sorted(list(snapshots_dir.iterdir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if s.is_dir() and (s / "model.bin").exists() and (s / "config.json").exists():
                return s
    return None


def ensure_whisper_model_cached(model_name: str = "small") -> Path:
    """Ensure Faster-Whisper model is cached ahead of time during preflight/startup."""
    cached = resolve_cached_whisper_snapshot(model_name)
    if cached is not None:
        return cached

    from faster_whisper import download_model
    downloaded_path = download_model(model_name)
    return Path(downloaded_path)


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "cuda",
        compute_type: str = "int8_float16",
        model_factory=None,
    ):
        local_files_only = False
        if model_path:
            model_source = str(model_path)
            local_files_only = True
        else:
            cached_snapshot = resolve_cached_whisper_snapshot(model_name)
            if cached_snapshot is not None:
                model_source = str(cached_snapshot)
                local_files_only = True
            else:
                model_source = model_name
                local_files_only = False

        factory = model_factory or WhisperModel

        kwargs = {
            "device": device,
            "compute_type": compute_type,
        }
        try:
            sig = inspect.signature(factory)
            if "local_files_only" in sig.parameters:
                kwargs["local_files_only"] = local_files_only
        except Exception:
            pass

        self.model = factory(
            model_source,
            **kwargs,
        )

    def transcribe(
        self,
        audio_path: str,
        *,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> Transcript:
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
        )

        transcript_segments = []

        for segment in segments:
            text = segment.text.strip()

            if not text:
                continue

            transcript_segments.append(
                TimedText(
                    start=round(segment.start, 3),
                    end=round(segment.end, 3),
                    text=text,
                )
            )

        return Transcript(
            language=info.language,
            segments=transcript_segments,
        )