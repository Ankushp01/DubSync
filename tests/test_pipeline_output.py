import json
import tempfile
import unittest
from pathlib import Path

from dubsync.config import (
    AppConfig, AudioConfig, FfmpegConfig, PathsConfig, RuntimeConfig,
    SegmentationConfig, WhisperConfig,
)
from dubsync.models import TimedText, Transcript
from dubsync.pipeline import _write_stage1


def make_config(root: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(root / "output", root / "models"),
        runtime=RuntimeConfig("cpu", "int8"), ffmpeg=FfmpegConfig("ffmpeg"),
        audio=AudioConfig(16000, 1),
        whisper=WhisperConfig("base", "", None, 1, False),
        segmentation=SegmentationConfig(0.75, 280),
    )


class Stage1OutputTests(unittest.TestCase):
    def test_keeps_asr_and_dialogue_timestamps_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage1_transcript.json"
            _write_stage1(path, Transcript("en", [
                TimedText(0.0, 1.0, "One"), TimedText(1.2, 2.0, "two"),
            ]), make_config(Path(directory)))
            result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["asr_segments"][1]["start"], 1.2)
        self.assertEqual(result["dialogue_segments"], [
            {"start": 0.0, "end": 2.0, "text": "One two"},
        ])
