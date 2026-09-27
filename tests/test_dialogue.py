import unittest

from dubsync.models import TimedText
from dubsync.segmentation.dialogue import group_dialogue


class DialogueGroupingTests(unittest.TestCase):
    def test_merges_close_segments_and_preserves_outer_timestamps(self) -> None:
        grouped = group_dialogue([
            TimedText(0.0, 1.0, "Hello"),
            TimedText(1.4, 2.0, "world."),
            TimedText(3.0, 4.0, "Separate turn."),
        ], max_gap_seconds=0.75, max_characters=100)
        self.assertEqual(grouped, [
            TimedText(0.0, 2.0, "Hello world."),
            TimedText(3.0, 4.0, "Separate turn."),
        ])
