"""
Unit tests for the Wav2Lip frame-extension logic introduced to fix the
frame-0 restart defect.

These tests exercise the frame-selection strategy in pure Python without
requiring a GPU, CUDA, or a real Wav2Lip model.  They verify:

  1. When mel steps equal source frames → no extension, frame list unchanged.
  2. When mel steps exceed source frames → last frame repeated for extra steps.
  3. Frame 0 is never returned after the original final frame in an extended run.
  4. The last source frame is the exact object used for every extension step.
  5. Frame exhaustion mid-batch is handled without raising StopIteration.
  6. A full normal-duration video (no extension) behaves exactly as before.
  7. Output frame count equals mel step count in both cases.
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# Minimal stub of the inference.py frame-extension logic
# (extracted so tests run without importing inference.py, which pulls
#  torch, face_detection, and other GPU-level dependencies at module scope)
# ---------------------------------------------------------------------------

def _build_extended_frames(source_frames, total_mels):
    """
    Mirror of the logic now inside datagen() in inference.py:

        if total_mels > total_source:
            extended_frames = source_frames + [source_frames[-1]] * extension_count
        else:
            extended_frames = source_frames
    """
    if not source_frames:
        raise ValueError('No frames could be read from the input video/image.')
    total_source = len(source_frames)
    if total_mels > total_source:
        extension_count = total_mels - total_source
        return source_frames + [source_frames[-1]] * extension_count
    return source_frames


def _simulate_frame_selection(source_frames, total_mels):
    """
    Simulate the datagen loop frame selection, returning the list of frames
    that would be used (one per mel step) without running real inference.
    """
    extended = _build_extended_frames(source_frames, total_mels)
    return [extended[i] for i in range(total_mels)]


def compute_target_frames(total_source: int, audio_duration: float, fps: float) -> int:
    """Mirror of the target frame selection logic in Wav2Lip-HD inference.py:
    Preserves complete source video frames when audio matches within float margin,
    extends with last-frame hold when audio exceeds video, and targets audio frames otherwise.
    """
    required_audio_frames = int(round(audio_duration * fps))
    if abs(total_source - required_audio_frames) <= 2:
        return total_source
    elif required_audio_frames > total_source:
        return required_audio_frames
    else:
        return required_audio_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(n, h=10, w=10):
    """Return n synthetic frames (numpy arrays with pixel value i % 256)."""
    return [np.full((h, w, 3), i % 256, dtype=np.uint8) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFrameExtensionLogic(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. No extension needed (mel steps == source frames)
    # ------------------------------------------------------------------
    def test_no_extension_when_mels_equal_source(self):
        frames = _make_frames(10)
        selected = _simulate_frame_selection(frames, total_mels=10)
        self.assertEqual(len(selected), 10)
        for i, (orig, sel) in enumerate(zip(frames, selected)):
            self.assertTrue(np.array_equal(orig, sel),
                            f'Frame {i} mismatch — extension should be a no-op')

    # ------------------------------------------------------------------
    # 2. Extension when mel steps exceed source frames
    # ------------------------------------------------------------------
    def test_extension_holds_last_frame(self):
        frames = _make_frames(5)
        selected = _simulate_frame_selection(frames, total_mels=8)
        self.assertEqual(len(selected), 8)
        # First 5 frames must be the originals
        for i in range(5):
            self.assertTrue(np.array_equal(selected[i], frames[i]),
                            f'Frame {i} should be the original source frame')
        # Frames 5, 6, 7 must all equal the last source frame (frame 4)
        for i in range(5, 8):
            self.assertTrue(np.array_equal(selected[i], frames[-1]),
                            f'Frame {i} should be a hold of the last source frame')

    # ------------------------------------------------------------------
    # 3. Frame 0 is never replayed after the original final frame
    # ------------------------------------------------------------------
    def test_frame_zero_never_replayed_after_end(self):
        frames = _make_frames(4)
        selected = _simulate_frame_selection(frames, total_mels=10)
        frame0_value = frames[0][0, 0, 0]   # pixel value = 0
        last_value = frames[-1][0, 0, 0]    # pixel value = 3
        for i in range(4, 10):
            actual_value = selected[i][0, 0, 0]
            self.assertNotEqual(actual_value, frame0_value,
                                f'Frame {i}: frame 0 was replayed (value {actual_value})')
            self.assertEqual(actual_value, last_value,
                             f'Frame {i}: expected last-frame hold (value {last_value}), '
                             f'got {actual_value}')

    # ------------------------------------------------------------------
    # 4. The held frames are the exact same object as source_frames[-1]
    # ------------------------------------------------------------------
    def test_held_frames_are_same_object_as_last_source_frame(self):
        frames = _make_frames(3)
        extended = _build_extended_frames(frames, total_mels=6)
        for i in range(3, 6):
            self.assertIs(extended[i], frames[-1],
                          f'Extended frame {i} should be the same object as frames[-1]')

    # ------------------------------------------------------------------
    # 5. Frame exhaustion never raises StopIteration
    # ------------------------------------------------------------------
    def test_no_stop_iteration_on_exhaustion(self):
        frames = _make_frames(2)
        try:
            selected = _simulate_frame_selection(frames, total_mels=100)
        except StopIteration:
            self.fail('StopIteration was raised — the old restart bug is present')
        self.assertEqual(len(selected), 100)

    # ------------------------------------------------------------------
    # 6. Normal-duration video (mels < source frames) is unchanged
    # ------------------------------------------------------------------
    def test_normal_video_no_extension(self):
        # More source frames than mel steps → use first total_mels frames only
        frames = _make_frames(20)
        selected = _simulate_frame_selection(frames, total_mels=12)
        self.assertEqual(len(selected), 12)
        for i in range(12):
            self.assertTrue(np.array_equal(selected[i], frames[i]),
                            f'Frame {i} mismatch in normal (no-extension) path')

    # ------------------------------------------------------------------
    # 7. Output frame count always equals mel step count
    # ------------------------------------------------------------------
    def test_output_frame_count_equals_mel_step_count(self):
        for source_count, mel_count in [(5, 5), (5, 3), (5, 8), (50, 60), (10, 10)]:
            frames = _make_frames(source_count)
            selected = _simulate_frame_selection(frames, total_mels=mel_count)
            self.assertEqual(len(selected), mel_count,
                             f'source={source_count}, mels={mel_count}: '
                             f'expected {mel_count} frames, got {len(selected)}')

    # ------------------------------------------------------------------
    # 8. Empty source raises ValueError (not StopIteration or IndexError)
    # ------------------------------------------------------------------
    def test_empty_source_raises_value_error(self):
        with self.assertRaises(ValueError):
            _build_extended_frames([], total_mels=5)

    # ------------------------------------------------------------------
    # 9. Exactly-at-boundary: mels == source (off-by-one guard)
    # ------------------------------------------------------------------
    def test_boundary_exactly_equal_mels_and_source(self):
        frames = _make_frames(391)   # matches real Front.mp4 frame count
        extended = _build_extended_frames(frames, total_mels=391)
        self.assertEqual(len(extended), 391)
        # Must be the original list returned unchanged (same object references)
        for i in range(391):
            self.assertIs(extended[i], frames[i],
                          f'Frame {i} should be the exact original object')

    # ------------------------------------------------------------------
    # 10. Front.mp4 realistic scenario: 391 source → 450 mel steps
    # ------------------------------------------------------------------
    def test_realistic_front_mp4_scenario(self):
        """
        Front.mp4 has 391 source frames.
        18.887s audio at 23.976fps → ~452 mel steps.
        Wav2Lip quantises to batches of 4, giving 450 output frames.
        """
        source_count = 391
        mel_count = 450
        frames = _make_frames(source_count)
        selected = _simulate_frame_selection(frames, total_mels=mel_count)

        self.assertEqual(len(selected), mel_count)

        # Frames 0-390 must match originals (by object identity)
        for i in range(source_count):
            self.assertIs(selected[i], frames[i],
                          f'Frame {i} should be the original source frame object')

        # Frames 391-449 must all be the held final frame (by object identity)
        for i in range(source_count, mel_count):
            self.assertIs(selected[i], frames[-1],
                          f'Frame {i} should be last-frame hold (object identity)')

    # ------------------------------------------------------------------
    # 11. Source video slightly longer than dubbed audio (Phase 1 case)
    # ------------------------------------------------------------------
    def test_source_video_slightly_longer_than_dubbed_audio(self):
        fps = 23.976023976023978
        total_source = 391
        audio_dur = 16.300
        target = compute_target_frames(total_source, audio_dur, fps)
        self.assertEqual(target, 391, "Expected target frames to preserve full source frames")
        frames = _make_frames(total_source)
        selected = _simulate_frame_selection(frames, total_mels=target)
        self.assertEqual(len(selected), 391)
        for i in range(391):
            self.assertIs(selected[i], frames[i])

    # ------------------------------------------------------------------
    # 12. Audio ends within source video duration (zero extension)
    # ------------------------------------------------------------------
    def test_audio_ends_within_source_video_duration(self):
        fps = 23.976
        total_source = 391
        audio_dur = 16.290
        target = compute_target_frames(total_source, audio_dur, fps)
        self.assertEqual(target, 391)
        extended = _build_extended_frames(_make_frames(total_source), total_mels=target)
        self.assertEqual(len(extended), total_source, "No extension should occur")

    # ------------------------------------------------------------------
    # 13. Audio genuinely exceeds source video duration
    # ------------------------------------------------------------------
    def test_audio_genuinely_exceeds_source_video_duration(self):
        fps = 23.976023976023978
        total_source = 391
        audio_dur = 18.887
        target = compute_target_frames(total_source, audio_dur, fps)
        expected_frames = int(round(18.887 * fps))
        self.assertEqual(target, expected_frames)
        self.assertGreater(target, total_source)
        frames = _make_frames(total_source)
        extended = _build_extended_frames(frames, total_mels=target)
        self.assertEqual(len(extended), target)
        # Verify tail frames are held
        for i in range(total_source, target):
            self.assertIs(extended[i], frames[-1])

    # ------------------------------------------------------------------
    # 14. Repeated execution without cumulative padding or duration shift
    # ------------------------------------------------------------------
    def test_repeated_execution_without_cumulative_padding(self):
        fps = 23.976023976023978
        total_source = 391
        audio_dur = 16.300
        for _ in range(5):
            target = compute_target_frames(total_source, audio_dur, fps)
            self.assertEqual(target, 391)
            frames = _make_frames(total_source)
            selected = _simulate_frame_selection(frames, total_mels=target)
            self.assertEqual(len(selected), 391)

    # ------------------------------------------------------------------
    # 15. Timestamp and frame rate consistency
    # ------------------------------------------------------------------
    def test_timestamp_and_frame_rate_consistency(self):
        fps = 23.976023976023978
        target_frames = 391
        dur = target_frames / fps
        self.assertAlmostEqual(dur, 16.307958, places=4)
        last_pts = (target_frames - 1) / fps
        self.assertAlmostEqual(last_pts, 16.26625, places=3)


if __name__ == '__main__':
    unittest.main()
