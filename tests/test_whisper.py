import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from dubsync.stt.whisper import WhisperTranscriber


class FakeModel:
    def transcribe(self, *_args, **_kwargs):
        return iter([
            SimpleNamespace(start=0.0, end=1.25, text=" hello "),
            SimpleNamespace(start=1.5, end=2.0, text="world"),
        ]), SimpleNamespace(language="en")


class WhisperTranscriberTests(unittest.TestCase):
    def test_maps_faster_whisper_segments_to_timed_text(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav") as audio:
            transcriber = WhisperTranscriber(
                device="cpu", compute_type="int8", model_name="unused",
                model_factory=lambda *_args, **_kwargs: FakeModel(),
            )
            result = transcriber.transcribe(audio.name, language=None, beam_size=1, vad_filter=False)
        self.assertEqual(result.language, "en")
        self.assertEqual(result.segments[0].text, "hello")
        self.assertEqual(result.segments[1].end, 2.0)
