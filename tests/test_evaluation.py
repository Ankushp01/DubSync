"""Tests for evaluation logic, media integrity, and regression protection."""
from __future__ import annotations

import json
from pathlib import Path
import unittest

BACKEND_DIR = Path(__file__).resolve().parent.parent
ENGLISH_EVAL_DIR = BACKEND_DIR / "output" / "English_3_evaluation"
FRONT_EVAL_DIR = BACKEND_DIR / "output" / "Front_evaluation"


class TestEvaluationLogic(unittest.TestCase):
    def test_english_3_evaluation_artifacts_unchanged(self):
        """Regression protection: Ensure previous English 3 evaluation artifacts exist and are unmodified."""
        self.assertTrue(ENGLISH_EVAL_DIR.is_dir(), "English_3_evaluation directory must exist.")
        expected_files = [
            "evaluation_summary.json",
            "media_integrity.json",
            "segment_metrics.csv",
            "translation_review.csv",
            "evaluation_methodology.md",
            "evaluation_report.md",
            "generate_evaluation.py",
        ]
        for f in expected_files:
            p = ENGLISH_EVAL_DIR / f
            self.assertTrue(p.is_file(), f"Expected artifact {f} missing in English_3_evaluation.")
            self.assertGreater(p.stat().st_size, 0, f"Artifact {f} is unexpectedly empty.")

    def test_front_evaluation_artifacts_exist(self):
        """Ensure all required Front.mp4 evaluation artifacts exist and are non-empty."""
        self.assertTrue(FRONT_EVAL_DIR.is_dir(), "Front_evaluation directory must exist.")
        required_files = [
            "evaluation_summary.json",
            "media_integrity.json",
            "segment_metrics.csv",
            "translation_review.csv",
            "human_evaluation_template.csv",
            "evaluation_methodology.md",
            "evaluation_report.md",
            "generate_evaluation.py",
        ]
        for f in required_files:
            p = FRONT_EVAL_DIR / f
            self.assertTrue(p.is_file(), f"Required artifact {f} missing in Front_evaluation.")
            self.assertGreater(p.stat().st_size, 0, f"Artifact {f} is unexpectedly empty.")

    def test_unavailable_metrics_are_null_not_zero(self):
        """Ensure unavailable metrics are explicitly null and have descriptive reasons, never zero-filled."""
        summary_path = FRONT_EVAL_DIR / "evaluation_summary.json"
        self.assertTrue(summary_path.is_file())
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        unmeasured = summary.get("unmeasured_or_null_metrics", {})
        keys_to_check = [
            "asr_word_error_rate_wer",
            "asr_character_error_rate_cer",
            "translation_bleu_score",
            "translation_chrf_score",
            "quantitative_lipsync_syncnet_lse_d",
            "quantitative_lipsync_syncnet_lse_c",
            "mean_opinion_score_mos",
        ]
        for k in keys_to_check:
            self.assertIn(k, unmeasured, f"Key {k} missing from unmeasured_or_null_metrics.")
            self.assertIsNone(unmeasured[k], f"Metric {k} must be null, but was {unmeasured[k]}.")

        # Check that reasons are informative strings
        self.assertIn("asr_wer_reason", unmeasured)
        self.assertIn("unavailable", unmeasured["asr_wer_reason"].lower())
        self.assertIn("quantitative_lipsync_lse_d_reason", unmeasured)
        self.assertIn("unavailable", unmeasured["quantitative_lipsync_lse_d_reason"].lower())

    def test_media_integrity_duration_match(self):
        """Assert media duration match and clean container decodes across audio and video."""
        media_path = FRONT_EVAL_DIR / "media_integrity.json"
        self.assertTrue(media_path.is_file())
        with open(media_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dur_comp = data["duration_comparisons"]
        delta_s = dur_comp["wav2lip_video_to_audio_delta_seconds"]
        self.assertLess(delta_s, 0.2, f"Wav2Lip video to audio mismatch was {delta_s}s, expected < 0.2s.")
        self.assertTrue(data["assembled_audio_wav"]["decode_check"]["clean_decode"])
        self.assertTrue(data["final_wav2lip_video_mp4"]["decode_check"]["clean_decode"])

    def test_tts_content_preservation_and_raw_durations(self):
        """Test TTS content-preservation invariant and verify raw-vs-processed durations are tracked."""
        summary_path = FRONT_EVAL_DIR / "evaluation_summary.json"
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        timing = summary["pipeline_timing_metrics"]
        self.assertEqual(timing["total_dialogue_segments"], 2)
        self.assertGreater(timing["raw_tts_duration_sum_seconds"], 0)
        self.assertGreater(timing["placed_speech_duration_sum_seconds"], 0)
        self.assertEqual(timing["overlapping_segments_count"], 0)
        self.assertEqual(timing["segments_requiring_trimming"], 0)
        self.assertEqual(timing["total_trimmed_audio_seconds"], 0.0)
        self.assertTrue(timing["all_speech_preserved"])

        # Content preservation invariant
        measured = summary.get("measured_metrics", {})
        self.assertTrue(measured.get("all_speech_preserved"))
        self.assertEqual(measured.get("trimmed_seconds"), 0.0)
        self.assertTrue(measured.get("missing_clause_audible_verified"))

    def test_corrupt_or_missing_media_handling(self):
        """Verify that probe and decode functions handle non-existent media safely."""
        from output.Front_evaluation.generate_evaluation import decode_check
        fake_path = BACKEND_DIR / "input" / "non_existent_file_xyz123.mp4"
        res = decode_check(fake_path)
        self.assertFalse(res["clean_decode"])
        self.assertNotEqual(res["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
