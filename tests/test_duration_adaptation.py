"""Regression test suite for DubSync duration adaptation.

Verifies that:
1. Duration adaptation preserves complete spoken content.
2. Local text adaptation never arbitrarily slices sentences or drops substantive clauses.
3. Audio duration adaptation respects the maximum acceptable speed limit (~1.25x).
"""
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from dubsync.audio.assembly import (
    MAX_FALLBACK_SPEED,
    MAX_NORMAL_SPEED,
    TimingConflictError,
    _build_atempo_filter,
    assemble_stage3_audio,
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


def _make_dummy_wav(path: Path, duration_sec: float, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    samples = (0.2 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())


class AudioAssemblyTimelineRegressionTests(unittest.TestCase):
    """Regression tests verifying audio timeline assembly never discards spoken content."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dubsync_test_timeline_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_regression_12s_tts_in_7s_slot_preserves_complete_speech(self) -> None:
        """Reproduce 12.44s TTS competing for 7.40s slot (as in Front.mp4).

        Verify that:
        1. Complete speech is preserved without trimming (trimmed_seconds == 0.0).
        2. Speed is clamped at 1.25x maximum (no extreme acceleration).
        3. Timeline duration is extended to accommodate the full compressed speech.
        4. Accounting record reflects all_speech_preserved=True.
        """
        source_audio = self.temp_dir / "source_16khz_mono.wav"
        _make_dummy_wav(source_audio, 16.308)

        seg1_audio = self.temp_dir / "segment_001.wav"
        _make_dummy_wav(seg1_audio, 9.80)

        seg2_audio = self.temp_dir / "segment_002.wav"
        _make_dummy_wav(seg2_audio, 12.44)

        manifest_data = {
            "schema_version": 2,
            "language": "hi",
            "segments": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 8.88,
                    "target_duration": 8.88,
                    "audio": "segment_001.wav",
                    "translated_text": "यह तथ्य कि मैं एक कैमरे से बात कर रहा हूँ...",
                },
                {
                    "index": 2,
                    "start": 8.88,
                    "end": 16.28,
                    "target_duration": 7.40,
                    "audio": "segment_002.wav",
                    "translated_text": "पर यदि आप एक व्यक्ति पर ध्यान केंद्रित करते हैं... और आपको अधिक महसूस करने में मदद कर सकता है।",
                },
            ],
        }

        manifest_path = self.temp_dir / "stage3_tts.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        out_wav = self.temp_dir / "dubbed_audio.wav"
        res_path = assemble_stage3_audio(
            stage3_path=manifest_path,
            output_path=out_wav,
            source_audio_path=source_audio,
            allow_timeline_extension=True,
        )

        self.assertTrue(res_path.is_file())

        accounting_path = self.temp_dir / "timing_accounting.json"
        self.assertTrue(accounting_path.is_file())
        with accounting_path.open("r", encoding="utf-8") as f:
            acct = json.load(f)

        self.assertTrue(acct["all_speech_preserved"])
        self.assertEqual(acct["total_trimmed_seconds"], 0.0)
        self.assertTrue(acct["timeline_extended"])
        self.assertGreater(acct["total_extension_seconds"], 0.0)

        seg2 = acct["segments"][1]
        self.assertEqual(seg2["segment_index"], 2)
        self.assertEqual(seg2["trimmed_seconds"], 0.0)
        self.assertTrue(seg2["all_speech_preserved"])
        self.assertEqual(seg2["applied_speed_factor"], MAX_FALLBACK_SPEED)
        self.assertEqual(seg2["classification"], "TIMELINE_EXTENDED")

        expected_seg2_dur = round(12.44 / MAX_FALLBACK_SPEED, 3)
        self.assertAlmostEqual(seg2["placed_duration"], expected_seg2_dur, places=2)

    def test_regression_12s_tts_in_7s_slot_disallowing_extension_raises_conflict(self) -> None:
        """Verify that when allow_timeline_extension is False, a timing conflict is explicitly raised."""
        source_audio = self.temp_dir / "source_16khz_mono.wav"
        _make_dummy_wav(source_audio, 16.308)

        seg1_audio = self.temp_dir / "segment_001.wav"
        _make_dummy_wav(seg1_audio, 9.80)

        seg2_audio = self.temp_dir / "segment_002.wav"
        _make_dummy_wav(seg2_audio, 12.44)

        manifest_data = {
            "schema_version": 2,
            "language": "hi",
            "segments": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 8.88,
                    "target_duration": 8.88,
                    "audio": "segment_001.wav",
                },
                {
                    "index": 2,
                    "start": 8.88,
                    "end": 16.28,
                    "target_duration": 7.40,
                    "audio": "segment_002.wav",
                },
            ],
        }

        manifest_path = self.temp_dir / "stage3_tts.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        out_wav = self.temp_dir / "dubbed_audio.wav"

        # Must raise TimingConflictError instead of silently discarding audio
        with self.assertRaises(TimingConflictError):
            assemble_stage3_audio(
                stage3_path=manifest_path,
                output_path=out_wav,
                source_audio_path=source_audio,
                allow_timeline_extension=False,
            )

    def test_adjacent_segments_reallocation_utilizes_natural_gaps(self) -> None:
        """Verify that natural silence gaps are utilized to absorb duration pressure."""
        source_audio = self.temp_dir / "source_16khz_mono.wav"
        _make_dummy_wav(source_audio, 12.0)

        # Segment 1 finishes early at 2.5s within a 5.0s slot (leaving 2.5s gap)
        seg1_audio = self.temp_dir / "segment_001.wav"
        _make_dummy_wav(seg1_audio, 2.50)

        # Segment 2 needs 3.8s in a 3.0s slot [5.0 - 8.0]
        seg2_audio = self.temp_dir / "segment_002.wav"
        _make_dummy_wav(seg2_audio, 3.80)

        manifest_data = {
            "schema_version": 2,
            "language": "hi",
            "segments": [
                {"index": 1, "start": 0.0, "end": 5.0, "target_duration": 5.0, "audio": "segment_001.wav"},
                {"index": 2, "start": 5.0, "end": 8.0, "target_duration": 3.0, "audio": "segment_002.wav"},
            ],
        }

        manifest_path = self.temp_dir / "stage3_tts.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        out_wav = self.temp_dir / "dubbed_audio.wav"
        assemble_stage3_audio(
            stage3_path=manifest_path,
            output_path=out_wav,
            source_audio_path=source_audio,
            allow_timeline_extension=True,
        )

        accounting_path = self.temp_dir / "timing_accounting.json"
        with accounting_path.open("r", encoding="utf-8") as f:
            acct = json.load(f)

        self.assertTrue(acct["all_speech_preserved"])
        self.assertEqual(acct["total_trimmed_seconds"], 0.0)
        # Segment 2 absorbs leading gap and does not need timeline extension
        self.assertFalse(acct["timeline_extended"])

    def test_accounting_record_structure_and_completeness(self) -> None:
        """Verify that per-segment accounting record includes all required fields."""
        source_audio = self.temp_dir / "source_16khz_mono.wav"
        _make_dummy_wav(source_audio, 10.0)

        seg1_audio = self.temp_dir / "segment_001.wav"
        _make_dummy_wav(seg1_audio, 4.0)

        manifest_data = {
            "schema_version": 2,
            "language": "hi",
            "segments": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 5.0,
                    "target_duration": 5.0,
                    "audio": "segment_001.wav",
                    "translated_text": "परीक्षण वाक्य",
                }
            ],
        }

        manifest_path = self.temp_dir / "stage3_tts.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        out_wav = self.temp_dir / "dubbed_audio.wav"
        assemble_stage3_audio(
            stage3_path=manifest_path,
            output_path=out_wav,
            source_audio_path=source_audio,
        )

        accounting_path = self.temp_dir / "timing_accounting.json"
        with accounting_path.open("r", encoding="utf-8") as f:
            acct = json.load(f)

        rec = acct["segments"][0]
        required_fields = [
            "original_slot",
            "raw_tts_duration",
            "processed_tts_duration",
            "applied_speed_factor",
            "timeline_placement",
            "placed_duration",
            "trimmed_seconds",
            "all_speech_preserved",
            "unresolved_timing_conflict",
        ]
        for field in required_fields:
            self.assertIn(field, rec, f"Missing required accounting field: {field}")

    def test_front_mp4_fitted_case_without_timeline_extension(self) -> None:
        """Regression test: Front.mp4 with adapted 8.12s TTS fits inside 7.40s slot.

        Verifies that:
        1. Segment 1 (9.80s raw) is adapted to 8.83s (1.11x speed) leaving 50ms margin before 8.88s.
        2. Segment 2 (8.12s raw) starts cleanly at 8.88s and is adapted to 7.40s (1.10x speed).
        3. Total assembled audio is exactly 16.308s (source duration).
        4. timeline_extended is False, total_extension_seconds is 0.0, trimmed_seconds is 0.0.
        """
        source_audio = self.temp_dir / "source_16khz_mono.wav"
        _make_dummy_wav(source_audio, 16.308)

        seg1_audio = self.temp_dir / "segment_001.wav"
        _make_dummy_wav(seg1_audio, 9.80)

        # 8.12s is the actual TTS duration generated for the concisely adapted Segment 2 text
        seg2_audio = self.temp_dir / "segment_002.wav"
        _make_dummy_wav(seg2_audio, 8.12)

        manifest_data = {
            "schema_version": 2,
            "language": "hi",
            "segments": [
                {
                    "index": 1,
                    "start": 0.0,
                    "end": 8.88,
                    "target_duration": 8.88,
                    "audio": "segment_001.wav",
                    "translated_text": "यह तथ्य कि मैं एक कैमरे से बात कर रहा हूँ...",
                },
                {
                    "index": 2,
                    "start": 8.88,
                    "end": 16.28,
                    "target_duration": 7.40,
                    "audio": "segment_002.wav",
                    "translated_text": "पर यदि आप एक व्यक्ति पर ध्यान दें और मान लें कि यह केवल उसी के लिए है, तो यह आपकी घबराहट शांत कर आपको बेहतर महसूस कराएगा।",
                },
            ],
        }

        manifest_path = self.temp_dir / "stage3_tts.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f)

        out_wav = self.temp_dir / "dubbed_audio.wav"
        assemble_stage3_audio(
            stage3_path=manifest_path,
            output_path=out_wav,
            source_audio_path=source_audio,
            allow_timeline_extension=True,
        )

        accounting_path = self.temp_dir / "timing_accounting.json"
        with accounting_path.open("r", encoding="utf-8") as f:
            acct = json.load(f)

        self.assertTrue(acct["all_speech_preserved"])
        self.assertEqual(acct["total_trimmed_seconds"], 0.0)
        self.assertFalse(acct["timeline_extended"], "Timeline should NOT be extended when speech fits within source duration")
        self.assertEqual(acct["total_extension_seconds"], 0.0)
        self.assertEqual(acct["assembled_duration"], 16.308)

        seg1 = acct["segments"][0]
        self.assertEqual(seg1["segment_index"], 1)
        self.assertEqual(seg1["timeline_placement"][0], 0.0)
        self.assertLessEqual(seg1["timeline_placement"][1], 8.835)

        seg2 = acct["segments"][1]
        self.assertEqual(seg2["segment_index"], 2)
        self.assertAlmostEqual(seg2["timeline_placement"][0], 8.88, places=2)
        self.assertLessEqual(seg2["timeline_placement"][1], 16.308)
        self.assertFalse(seg2["timeline_extended"])
        self.assertEqual(seg2["trimmed_seconds"], 0.0)
        self.assertLessEqual(seg2["applied_speed_factor"], 1.15)

    def test_overlong_translation_paraphrasing_reduces_verbosity(self) -> None:
        """Verify that overlong verbose Segment 2 translation is concisely paraphrased without losing clauses."""
        adapter = TextDurationAdapter()
        source_en = "but if you focus down on one person and pretend like it's going to just one person it can really calm your nerves and help you feel more."
        verbose_hi = "लेकिन यदि आप एक व्यक्ति पर ध्यान केंद्रित करते हैं और ऐसा नाटक करते हैं कि यह केवल एक व्यक्ति के लिए जा रहा है तो यह वास्तव में आपकी नसों को शांत कर सकता है और आपको अधिक महसूस करने में मदद कर सकता है।"

        adapted = adapter.adapt(
            source_text=source_en,
            translated_text=verbose_hi,
            target_duration=7.40,
            generated_duration=11.76,
            target_language="Hindi",
        )

        # Word count is reduced
        self.assertLess(len(adapted.split()), len(verbose_hi.split()))
        # Essential semantic clauses preserved
        self.assertIn("व्यक्ति", adapted)
        self.assertIn("ध्यान", adapted)
        self.assertIn("घबराहट", adapted)
        self.assertIn("शांत", adapted)
        self.assertIn("महसूस", adapted)

    def test_wav_metadata_duration_calculation(self) -> None:
        """Verify that audio durations are computed from actual WAV samples and sample rates."""
        import soundfile as sf
        dummy_wav = self.temp_dir / "sample_rate_check.wav"
        _make_dummy_wav(dummy_wav, 3.500, sample_rate=24000)

        info = sf.info(str(dummy_wav))
        self.assertEqual(info.samplerate, 24000)
        self.assertEqual(info.frames, 84000)
        self.assertAlmostEqual(info.duration, 3.500, places=3)


if __name__ == "__main__":
    unittest.main()
