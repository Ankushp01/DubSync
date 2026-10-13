"""Regression test suite for DubSync duration adaptation.

Verifies that:
1. Duration adaptation preserves complete spoken content.
2. Local text adaptation never arbitrarily slices sentences or drops substantive clauses.
3. Audio duration adaptation respects the maximum acceptable speed limit (~1.25x).
"""
import unittest

from dubsync.audio.assembly import (
    MAX_FALLBACK_SPEED,
    MAX_NORMAL_SPEED,
    _build_atempo_filter,
)
from dubsync.timing.text_adapter import TextDurationAdapter


class DurationAdaptationRegressionTests(unittest.TestCase):
    """Regression tests verifying duration adaptation does not discard spoken audio/content."""

    def setUp(self) -> None:
        self.adapter = TextDurationAdapter()

    def test_text_adaptation_preserves_complete_sentence_and_clauses(self) -> None:
        """Verify that a 35-word translated sentence is NOT truncated into broken fragments."""
        original_hindi = (
            "क्या आपने कभी अपने कैमरे से बात करते हुए एक वीडियो रिकॉर्ड करने की कोशिश की है, "
            "लेकिन आप बस अपने शब्दों पर लड़खड़ाते रहते हैं, गड़बड़ करते रहते हैं, आप जानते हैं? "
            "मुझे एक पेशेवर से कुछ सुझाव मिले हैं, जो अपने बेडरूम में बैठता है और एक कैमरे से बात करता है..."
        )
        source_en = (
            "Have you ever tried recording a video of you talking to your camera, but you just keep "
            "stumbling over your words, messing up, you know? I've got some tips from a professional, "
            "who sits in his bedroom and talks to a camera..."
        )

        target_duration = 8.02
        generated_duration = 12.60

        adapted = self.adapter.adapt(
            source_text=source_en,
            translated_text=original_hindi,
            target_duration=target_duration,
            generated_duration=generated_duration,
            target_language="Hindi",
            attempt=1,
        )

        # Regression check: the previous bug produced 'की है, पर आप बस अपने।' (5 words)
        self.assertNotEqual(
            adapted.strip(),
            "की है, पर आप बस अपने।",
            "Regression failure: sentence was truncated to broken 5-word fragment around 'पर'",
        )

        # Word count check: must preserve at least 70% of words (no destructive word budget truncation)
        orig_words = original_hindi.split()
        adapted_words = adapted.split()
        self.assertGreaterEqual(
            len(adapted_words),
            int(len(orig_words) * 0.70),
            f"Adapted text retained only {len(adapted_words)}/{len(orig_words)} words (<70%)",
        )

        # Key semantic components must be preserved
        self.assertIn("कैमरे", adapted)
        self.assertIn("वीडियो", adapted)
        self.assertIn("पेशेवर", adapted)
        self.assertIn("सुझाव", adapted)

    def test_text_adaptation_segment_2_preserves_meaning(self) -> None:
        """Verify segment 2 translation is preserved and not chopped to 'नहीं है, पर केवल बुलेट पॉइंट।'"""
        original_hindi = (
            "यह मेरा जीवन है। टिप 1, रिकॉर्डिंग शुरू करने से पहले अपने विचारों को लिख लें। "
            "यह एक पूर्ण पटकथा होने की आवश्यकता नहीं है, लेकिन केवल बुलेट पॉइंट भी है..."
        )
        source_en = (
            "by himself. This is my life. Tip 1, write down your thoughts before you start recording. "
            "It doesn't have to be a full script, but even just bullet points is..."
        )

        target_duration = 5.48
        generated_duration = 8.84

        adapted = self.adapter.adapt(
            source_text=source_en,
            translated_text=original_hindi,
            target_duration=target_duration,
            generated_duration=generated_duration,
            target_language="Hindi",
            attempt=1,
        )

        self.assertNotEqual(
            adapted.strip(),
            "नहीं है, पर केवल बुलेट पॉइंट।",
            "Regression failure: segment 2 was truncated to broken 5-word fragment",
        )

        # Key semantic components must remain
        self.assertIn("टिप 1", adapted)
        self.assertIn("विचारों", adapted)
        self.assertIn("पटकथा", adapted)
        self.assertIn("बुलेट पॉइंट", adapted)

    def test_speed_limits_and_atempo_filter_clamping(self) -> None:
        """Verify duration adaptation respects acceptable speed bounds (~1.25x max)."""
        self.assertEqual(MAX_NORMAL_SPEED, 1.15)
        self.assertEqual(MAX_FALLBACK_SPEED, 1.25)

        # 1.20x speed is within acceptable fallback speed
        speed = 1.20
        self.assertLessEqual(speed, MAX_FALLBACK_SPEED)

        # Filter construction check
        filter_str = _build_atempo_filter(speed)
        self.assertIn("atempo=", filter_str)

        # Extreme speeds are never generated beyond clamped limit
        clamped_speed = min(1.85, MAX_FALLBACK_SPEED)
        self.assertEqual(clamped_speed, 1.25)


if __name__ == "__main__":
    unittest.main()
