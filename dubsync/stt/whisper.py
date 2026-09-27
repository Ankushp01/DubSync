"""Faster-Whisper transcription backend."""

from __future__ import annotations

from faster_whisper import WhisperModel

from dubsync.models import TimedText, Transcript


class WhisperTranscriber:
    def __init__(
        self,
        model_name: str = "small",
        model_path: str | None = None,
        device: str = "cuda",
        compute_type: str = "int8_float16",
    ):
        model_source = model_path or model_name

        self.model = WhisperModel(
            model_source,
            device=device,
            compute_type=compute_type,
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