import unittest

from dubsync.models import TimedText
from dubsync.segmentation.dialogue import (
    HARD_MAX_SEGMENT_SECONDS,
    PREFERRED_SEGMENT_MAX_SECONDS,
    PREFERRED_SEGMENT_MIN_SECONDS,
    group_dialogue,
    split_long_segment,
)


class DialogueSegmentationTests(unittest.TestCase):
    """Explicit tests for Dialogue Segmentation according to DubSync specification."""

    def test_a_multiple_consecutive_asr_fragments_merge_toward_8_to_10_seconds(self) -> None:
        """Test A: Multiple consecutive ASR fragments from one continuous utterance

        -> merge toward 8–10 seconds.
        """
        fragments = [
            TimedText(0.0, 2.1, "I was going"),
            TimedText(2.1, 4.0, "to ask her"),
            TimedText(4.0, 6.3, "but I didn't"),
            TimedText(6.3, 8.7, "have the courage."),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 8.7)
        self.assertEqual(
            result[0].text,
            "I was going to ask her but I didn't have the courage.",
        )
        # Verify duration is in the preferred 8-10s range
        dur = result[0].end - result[0].start
        self.assertTrue(PREFERRED_SEGMENT_MIN_SECONDS <= dur <= PREFERRED_SEGMENT_MAX_SECONDS)

    def test_b_same_speaker_meaningful_pause_no_merge(self) -> None:
        """Test B: Same speaker but meaningful pause

        -> do not merge across the pause.
        """
        fragments = [
            TimedText(0.0, 4.0, "This is the first sentence."),
            TimedText(5.2, 8.5, "This is the second sentence after a clear pause."),  # gap = 1.2s >= 0.75s
        ]
        result = group_dialogue(fragments, meaningful_pause_seconds=0.75)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 4.0)
        self.assertEqual(result[1].start, 5.2)
        self.assertEqual(result[1].end, 8.5)

    def test_c_speaker_a_to_speaker_b_never_merge(self) -> None:
        """Test C: Speaker A -> Speaker B

        -> never merge.
        """
        fragments = [
            TimedText(0.0, 3.0, "What do you think about the project?", speaker="Speaker_A"),
            TimedText(3.1, 7.0, "I think it is performing exceptionally well.", speaker="Speaker_B"),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].speaker, "Speaker_A")
        self.assertEqual(result[1].speaker, "Speaker_B")
        self.assertEqual(result[0].end, 3.0)
        self.assertEqual(result[1].start, 3.1)

    def test_d_natural_utterance_of_5_seconds_kept_as_5_seconds(self) -> None:
        """Test D: Natural utterance of 5 seconds

        -> keep as 5 seconds; do not extend.
        """
        fragments = [
            TimedText(0.0, 5.0, "This is a natural complete utterance."),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 5.0)
        dur = result[0].end - result[0].start
        self.assertEqual(dur, 5.0)

    def test_e_natural_utterance_of_9_seconds_kept_as_one_segment(self) -> None:
        """Test E: Natural utterance of 9 seconds

        -> keep as one segment.
        """
        fragments = [
            TimedText(0.0, 9.0, "This is a nine second natural speech segment that fits right in."),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 9.0)
        dur = result[0].end - result[0].start
        self.assertEqual(dur, 9.0)

    def test_f_natural_utterance_of_11_seconds_kept_as_one_segment(self) -> None:
        """Test F: Natural utterance of 11 seconds

        -> keep as one segment.
        """
        fragments = [
            TimedText(0.0, 11.0, "This is an eleven second complete continuous sentence under twelve seconds."),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 11.0)
        self.assertTrue(result[0].end - result[0].start <= HARD_MAX_SEGMENT_SECONDS)

    def test_g_natural_utterance_of_14_to_18_seconds_split_linguistically(self) -> None:
        """Test G: Natural utterance of 14–18 seconds

        -> split at a natural linguistic boundary, keeping each segment <=12s.
        """
        # 16-second single long sentence with a natural comma and conjunction clause
        long_segment = TimedText(
            0.0,
            16.0,
            "I forced myself to speak in my Insta stories, and at that time I was not confident so I was practicing.",
        )
        result = group_dialogue([long_segment])
        self.assertEqual(len(result), 2)
        for seg in result:
            dur = seg.end - seg.start
            # Every resulting segment must be <= 12.0s
            self.assertTrue(dur <= HARD_MAX_SEGMENT_SECONDS, f"Segment {seg} exceeded 12s hard ceiling")
        # Check that it split cleanly at the clause boundary
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[-1].end, 16.0)
        self.assertTrue(result[0].text.endswith(","))

    def test_h_short_sentence_followed_by_another_merges_into_preferred_range(self) -> None:
        """Test H: Very short sentence followed immediately by another sentence from the same speaker

        -> merge when the combined utterance naturally fits within 8–10 seconds.
        """
        fragments = [
            TimedText(0.0, 2.0, "How did I improve?"),
            TimedText(2.2, 3.2, "Simple."),
            TimedText(3.3, 8.8, "I practiced speaking directly to the camera every single day."),
        ]
        result = group_dialogue(fragments)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].start, 0.0)
        self.assertEqual(result[0].end, 8.8)
        self.assertEqual(
            result[0].text,
            "How did I improve? Simple. I practiced speaking directly to the camera every single day.",
        )
        dur = result[0].end - result[0].start
        self.assertTrue(PREFERRED_SEGMENT_MIN_SECONDS <= dur <= PREFERRED_SEGMENT_MAX_SECONDS)


if __name__ == "__main__":
    unittest.main()
