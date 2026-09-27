"""Small, serializable data types shared by pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TimedText:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class Transcript:
    language: str | None
    segments: list[TimedText]

    def to_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
        }
