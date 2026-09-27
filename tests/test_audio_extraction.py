import unittest
from unittest.mock import patch

from dubsync.audio.extraction import AudioExtractionError, _find_executable


class AudioExtractionTests(unittest.TestCase):
    @patch("dubsync.audio.extraction.shutil.which", return_value=None)
    def test_reports_missing_ffmpeg_clearly(self, _which) -> None:
        with self.assertRaisesRegex(AudioExtractionError, "FFmpeg was not found"):
            _find_executable("ffmpeg")
