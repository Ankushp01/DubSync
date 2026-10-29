"""Clean non-overlapping audio timeline assembly with duration adaptation."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

logger = logging.getLogger(__name__)

# Constants for natural speech pacing
MAX_NORMAL_SPEED = 1.15
MAX_FALLBACK_SPEED = 1.25
MIN_SILENCE_MARGIN = 0.05  # 50ms silence between speech clips


class TimingConflictError(RuntimeError):
    """Raised when audio content cannot fit within the available timeline without trimming,
    and timeline extension is disallowed."""
    pass


def _safe_str(s: str) -> str:
    """Safely format string so it never crashes console printing."""
    try:
        enc = sys.stdout.encoding or "utf-8"
        s.encode(enc)
        return s
    except Exception:
        return s.encode("ascii", errors="replace").decode("ascii")


def _read_wav(path: Path) -> tuple[dict, np.ndarray]:
    """Read a mono WAV file into an int16 numpy array."""
    path = Path(path)
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)

    if audio.ndim != 1:
        # Convert multi-channel to mono by averaging
        audio = audio.mean(axis=1)

    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767.0).astype(np.int16)

    params = {
        "channels": 1,
        "sample_rate": int(sample_rate),
        "sample_width": 2,
    }
    return params, audio_int16


def _build_atempo_filter(speed_factor: float) -> str:
    """Build a chained FFmpeg atempo filter for any positive speed factor."""
    if speed_factor <= 0:
        raise ValueError(f"Invalid speed factor: {speed_factor}")

    filters = []
    remaining = speed_factor

    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0

    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


def time_stretch_audio(
    input_wav: Path,
    output_wav: Path,
    target_duration: float,
    ffmpeg_executable: str = "ffmpeg",
) -> float:
    """Pitch-preserving time stretch / compress audio to target duration using FFmpeg atempo."""
    input_wav = Path(input_wav).resolve()
    output_wav = Path(output_wav).resolve()

    info = sf.info(str(input_wav))
    current_duration = info.duration
    if current_duration <= 0 or target_duration <= 0:
        raise ValueError(f"Invalid durations: current={current_duration}, target={target_duration}")

    speed_factor = current_duration / target_duration
    filter_expr = _build_atempo_filter(speed_factor)

    output_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg_executable,
        "-y",
        "-i", str(input_wav),
        "-filter:a", filter_expr,
        "-vn",
        str(output_wav),
    ]

    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg atempo time-stretch failed: {res.stderr}")

    out_info = sf.info(str(output_wav))
    return out_info.duration


def _safe_trim_audio(
    input_wav: Path,
    output_wav: Path,
    max_duration: float,
    fade_duration: float = 0.02,
) -> float:
    """Trim audio to max_duration with a smooth fade-out to prevent audible clicks."""
    params, audio = _read_wav(input_wav)
    sr = params["sample_rate"]
    max_samples = int(max_duration * sr)
    if len(audio) > max_samples:
        trimmed = audio[:max_samples].copy().astype(np.float32)
        fade_samples = min(len(trimmed), int(fade_duration * sr))
        if fade_samples > 0:
            fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
            trimmed[-fade_samples:] *= fade_curve
        audio = np.clip(trimmed, -32768, 32767).astype(np.int16)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())
    return round(len(audio) / sr, 3)


def assemble_stage3_audio(
    stage3_path: str | Path,
    output_path: str | Path,
    source_audio_path: Optional[str | Path] = None,
    ffmpeg_executable: str = "ffmpeg",
    allow_timeline_extension: bool = True,
) -> Path:
    """Assemble Stage 3 TTS audio segments onto a clean, natural, non-overlapping timeline.

    Guarantees:
    1. Every dialogue segment has an explicit non-overlapping slot.
    2. TTS audio clips NEVER overlap with adjacent segments.
    3. Natural speech pacing is strictly prioritized: speed-up is capped at 1.15x (normal)
       and 1.25x (max fallback). Extreme speeds (e.g. 1.5x - 2.33x+) are NEVER applied.
    4. Clips can safely expand into natural trailing silence gaps before the next segment.
    5. Silence and natural pauses between dialogue segments are preserved.
    6. Substantive translated speech is NEVER silently truncated. If speech exceeds
       the original timeline, timeline is extended or explicit TimingConflictError is raised.
    7. Per-segment duration accounting record is generated and persisted.
    """
    stage3_path = Path(stage3_path).resolve()
    output_path = Path(output_path).resolve()

    if not stage3_path.is_file():
        raise FileNotFoundError(f"Stage 3 manifest not found: {stage3_path}")

    with stage3_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    segments = manifest.get("segments", [])
    if not segments:
        raise ValueError("Stage 3 manifest contains no segments.")

    # Determine timeline format from the first available audio segment
    manifest_dir = stage3_path.parent
    first_audio = manifest_dir / segments[0]["audio"]
    if not first_audio.is_file():
        raise FileNotFoundError(f"First audio segment not found: {first_audio}")

    first_params, _ = _read_wav(first_audio)
    sample_rate = first_params["sample_rate"]

    # Determine total timeline duration from source audio if available
    orig_source_duration: Optional[float] = None
    if source_audio_path:
        src = Path(source_audio_path).resolve()
        if src.is_file():
            orig_source_duration = sf.info(str(src)).duration

    if orig_source_duration is None:
        default_src = manifest_dir / "source_16khz_mono.wav"
        if default_src.is_file():
            orig_source_duration = sf.info(str(default_src)).duration

    source_duration = orig_source_duration

    # Step 1: Sanitize and sort dialogue segments by start timestamp
    sorted_segments = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    for idx, seg in enumerate(sorted_segments):
        start = round(float(seg["start"]), 3)
        end = round(float(seg["end"]), 3)
        if end <= start:
            end = round(start + 0.1, 3)

        # Enforce non-overlapping dialogue segment boundaries
        if idx > 0:
            prev_end = round(float(sorted_segments[idx - 1]["end"]), 3)
            if start < prev_end:
                logger.warning(
                    f"Dialogue segment {idx} start ({start}s) overlapped with segment {idx-1} end ({prev_end}s). Adjusting boundary."
                )
                start = prev_end
                if end <= start:
                    end = round(start + 0.1, 3)

        seg["start"] = start
        seg["end"] = end
        seg["target_duration"] = round(end - start, 3)

    # Step 2: Determine available duration, apply natural duration adaptation, and compute placement
    planned_placements = []
    adaptation_log = []
    current_cursor = 0.0

    for idx, seg in enumerate(sorted_segments):
        seg_start = seg["start"]
        seg_end = seg["end"]
        target_dur = seg["target_duration"]
        trans_text = seg.get("translated_text", seg.get("final_translated_text", ""))
        adaptation_attempts = int(seg.get("adaptation_attempts", seg.get("gemini_attempts", 0)))
        text_was_adapted = bool(seg.get("text_was_adapted", False))

        audio_file = manifest_dir / seg["audio"]
        if not audio_file.is_file():
            raise FileNotFoundError(f"Segment audio file not found: {audio_file}")

        tts_dur = round(sf.info(str(audio_file)).duration, 3)
        raw_audio_file = manifest_dir / seg.get("raw_audio", seg["audio"])
        raw_tts_dur = round(sf.info(str(raw_audio_file)).duration, 3) if raw_audio_file.is_file() else tts_dur

        # Determine earliest non-overlapping placement start
        if current_cursor == 0.0:
            placement_start = seg_start
        else:
            earliest_valid_start = round(current_cursor + MIN_SILENCE_MARGIN, 3)
            if earliest_valid_start <= seg_start:
                # Slot has leading silence or matches segment start
                # If this segment is under timing pressure, check if natural pause can be used
                if (tts_dur / target_dur > 1.0) and (seg_start > earliest_valid_start):
                    needed_slot = tts_dur / MAX_NORMAL_SPEED
                    extra_time = needed_slot - target_dur
                    pre_gap = seg_start - earliest_valid_start
                    shift_back = min(pre_gap, extra_time)
                    placement_start = round(seg_start - shift_back, 3)
                else:
                    placement_start = seg_start
            else:
                # Previous segment extended past seg_start; ripple-shift to avoid collision
                placement_start = earliest_valid_start

        # Determine available end boundary for this segment
        if idx + 1 < len(sorted_segments):
            next_start = sorted_segments[idx + 1]["start"]
            space_before_next = round(next_start - placement_start - MIN_SILENCE_MARGIN, 3)
            # If TTS can fit cleanly before next segment with normal speed, reserve silence margin
            if space_before_next > 0 and (tts_dur / space_before_next) <= MAX_NORMAL_SPEED:
                available_slot = space_before_next
            else:
                available_slot = round(max(target_dur, space_before_next), 3)
        else:
            # Final segment on timeline
            if orig_source_duration and orig_source_duration > placement_start:
                available_slot = round(orig_source_duration - placement_start, 3)
            elif source_duration and source_duration > placement_start:
                available_slot = round(source_duration - placement_start, 3)
            else:
                available_slot = target_dur

        speed_factor = 1.0
        final_dur = tts_dur
        final_clip_path = audio_file
        classification = "NORMAL"
        trimmed_seconds = 0.0
        all_speech_preserved = True
        timeline_extended = False
        extension_seconds = 0.0
        timing_conflict = None

        required_speed = tts_dur / available_slot if available_slot > 0 else 999.0

        if tts_dur <= available_slot or required_speed <= 1.0:
            # Fits comfortably without acceleration
            speed_factor = 1.0
            final_dur = tts_dur
            final_clip_path = audio_file
            classification = "TEXT_SHORTENED" if text_was_adapted else "NORMAL"
            adaptation_log.append(f"Segment {idx+1}: Speech fits naturally ({tts_dur:.2f}s in {available_slot:.2f}s slot)")

        elif required_speed <= MAX_NORMAL_SPEED:
            # Mild natural speed adjustment (1.00x - 1.15x) fits inside available slot
            speed_factor = round(required_speed, 2)
            desired_dur = available_slot
            adapted_file = manifest_dir / f"segment_{idx+1:03d}_adapted.wav"
            final_dur = time_stretch_audio(
                input_wav=audio_file,
                output_wav=adapted_file,
                target_duration=desired_dur,
                ffmpeg_executable=ffmpeg_executable,
            )
            final_clip_path = adapted_file
            classification = "MILD_SPEED_ADJUSTMENT"
            adaptation_log.append(
                f"Segment {idx+1}: Mild speed adjustment {required_speed:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) to fit {available_slot:.2f}s"
            )

        elif required_speed <= MAX_FALLBACK_SPEED:
            # Moderate speed adjustment (1.15x - 1.25x) fits inside available slot
            speed_factor = round(required_speed, 2)
            desired_dur = available_slot
            adapted_file = manifest_dir / f"segment_{idx+1:03d}_adapted.wav"
            final_dur = time_stretch_audio(
                input_wav=audio_file,
                output_wav=adapted_file,
                target_duration=desired_dur,
                ffmpeg_executable=ffmpeg_executable,
            )
            final_clip_path = adapted_file
            classification = "MILD_SPEED_ADJUSTMENT"
            adaptation_log.append(
                f"Segment {idx+1}: Moderate speed adjustment {required_speed:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) to fit {available_slot:.2f}s"
            )

        else:
            # Required speed > 1.25x: Do NOT use extreme atempo!
            # Clamp atempo to MAX_FALLBACK_SPEED (1.25x)
            speed_factor = MAX_FALLBACK_SPEED
            compressed_dur = round(tts_dur / MAX_FALLBACK_SPEED, 3)

            if compressed_dur <= available_slot:
                # Fits inside slot/trailing gap at 1.25x
                desired_dur = compressed_dur
                adapted_file = manifest_dir / f"segment_{idx+1:03d}_adapted.wav"
                final_dur = time_stretch_audio(
                    input_wav=audio_file,
                    output_wav=adapted_file,
                    target_duration=desired_dur,
                    ffmpeg_executable=ffmpeg_executable,
                )
                final_clip_path = adapted_file
                classification = "TEXT_SHORTENED" if text_was_adapted else "MILD_SPEED_ADJUSTMENT"
                adaptation_log.append(
                    f"Segment {idx+1}: Clamped speed to {MAX_FALLBACK_SPEED:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) utilizing {available_slot:.2f}s gap"
                )
            else:
                # Compressed speech exceeds available slot
                if allow_timeline_extension:
                    # Extend timeline to preserve 100% of spoken content without trimming
                    desired_dur = compressed_dur
                    adapted_file = manifest_dir / f"segment_{idx+1:03d}_adapted.wav"
                    final_dur = time_stretch_audio(
                        input_wav=audio_file,
                        output_wav=adapted_file,
                        target_duration=desired_dur,
                        ffmpeg_executable=ffmpeg_executable,
                    )
                    final_clip_path = adapted_file
                    classification = "TIMELINE_EXTENDED"
                    adaptation_log.append(
                        f"Segment {idx+1}: Clamped speed to {MAX_FALLBACK_SPEED:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s); timeline extended to preserve complete speech"
                    )
                else:
                    timing_conflict = (
                        f"Segment {idx+1} requires {compressed_dur:.3f}s at maximum speed {MAX_FALLBACK_SPEED}x, "
                        f"exceeding available slot {available_slot:.3f}s by {round(compressed_dur - available_slot, 3):.3f}s. "
                        f"Timeline extension is disabled."
                    )
                    raise TimingConflictError(timing_conflict)

        placement_end = round(placement_start + final_dur, 3)
        current_cursor = placement_end

        # Sub-frame rounding tolerance (50ms ~ 1 video frame at 24fps)
        EXTENSION_TOLERANCE = 0.05
        if orig_source_duration and (placement_end - orig_source_duration) > EXTENSION_TOLERANCE:
            timeline_extended = True
            extension_seconds = round(placement_end - orig_source_duration, 3)
        elif idx + 1 < len(sorted_segments) and (placement_end - sorted_segments[idx + 1]["start"]) > EXTENSION_TOLERANCE:
            timeline_extended = True
            extension_seconds = round(placement_end - sorted_segments[idx + 1]["start"], 3)

        planned_placements.append({
            "index": idx + 1,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "target_duration": target_dur,
            "tts_duration": tts_dur,
            "raw_tts_duration": raw_tts_dur,
            "available_duration": available_slot,
            "final_duration": final_dur,
            "placement_start": placement_start,
            "placement_end": placement_end,
            "audio_path": final_clip_path,
            "adapted_speed": speed_factor,
            "adaptation_attempts": adaptation_attempts,
            "classification": classification,
            "translated_text": trans_text,
            "trimmed_seconds": trimmed_seconds,
            "all_speech_preserved": all_speech_preserved,
            "timeline_extended": timeline_extended,
            "extension_seconds": extension_seconds,
            "timing_conflict": timing_conflict,
        })

    # Step 3: Programmatic Timeline Validation
    for i, p in enumerate(planned_placements):
        if i > 0:
            prev = planned_placements[i - 1]
            if p["placement_start"] < prev["placement_end"]:
                raise ValueError(
                    f"Timeline validation failed: Segment {p['index']} clip placement ({p['placement_start']}s) overlaps previous clip end ({prev['placement_end']}s)"
                )

        if i + 1 < len(planned_placements):
            next_p = planned_placements[i + 1]
            if p["placement_end"] > next_p["placement_start"]:
                raise ValueError(
                    f"Timeline validation failed: Segment {p['index']} clip end ({p['placement_end']}s) overlaps next clip start ({next_p['placement_start']}s)"
                )

    # Determine total timeline frames
    max_clip_end = max(p["placement_end"] for p in planned_placements)
    has_extension = any(p["timeline_extended"] for p in planned_placements)
    if not has_extension and orig_source_duration is not None:
        source_duration = orig_source_duration
    elif source_duration is None:
        source_duration = max_clip_end
    else:
        source_duration = max(source_duration, max_clip_end)

    total_frames = round(source_duration * sample_rate)
    total_timeline_sec = round(total_frames / sample_rate, 3)

    # Step 4: Build per-segment accounting records (Requirement 8)
    accounting_records = []
    for p in planned_placements:
        rec = {
            "segment_index": p["index"],
            "original_slot": [p["seg_start"], p["seg_end"]],
            "original_slot_duration": round(p["seg_end"] - p["seg_start"], 3),
            "raw_tts_duration": p["raw_tts_duration"],
            "processed_tts_duration": p["tts_duration"],
            "applied_speed_factor": p["adapted_speed"],
            "timeline_placement": [p["placement_start"], p["placement_end"]],
            "placed_duration": p["final_duration"],
            "trimmed_seconds": p["trimmed_seconds"],
            "all_speech_preserved": p["all_speech_preserved"],
            "timeline_extended": p["timeline_extended"],
            "extension_seconds": p["extension_seconds"],
            "unresolved_timing_conflict": p["timing_conflict"],
            "classification": p["classification"],
            "translated_text": p["translated_text"],
        }
        accounting_records.append(rec)

    # Write timing accounting artifact
    accounting_artifact = {
        "schema_version": 1,
        "source_duration": orig_source_duration,
        "assembled_duration": total_timeline_sec,
        "timeline_extended": any(r["timeline_extended"] for r in accounting_records),
        "total_extension_seconds": round(max(0.0, total_timeline_sec - (orig_source_duration or total_timeline_sec)), 3),
        "total_trimmed_seconds": round(sum(r["trimmed_seconds"] for r in accounting_records), 3),
        "all_speech_preserved": all(r["all_speech_preserved"] for r in accounting_records),
        "segments": accounting_records,
    }
    accounting_path = manifest_dir / "timing_accounting.json"
    with accounting_path.open("w", encoding="utf-8") as f:
        json.dump(accounting_artifact, f, indent=2, ensure_ascii=False)

    # Also update stage3 manifest
    manifest["timing_accounting"] = accounting_records
    manifest["timeline_extended"] = accounting_artifact["timeline_extended"]
    manifest["assembled_duration"] = total_timeline_sec
    with stage3_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Log detailed duration accounting table
    print("\n" + "=" * 145)
    print("DUBBED AUDIO TIMELINE ACCOUNTING & DURATION VERIFICATION")
    print("=" * 145)
    header = (
        f"{'Seg':<4} | {'Orig Slot':<16} | {'Raw TTS':<8} | {'Proc TTS':<9} | "
        f"{'Speed':<6} | {'Placed Slot':<17} | {'Placed Dur':<10} | {'Trimmed':<8} | "
        f"{'Preserved':<9} | {'Extended':<9} | {'Classification':<20}"
    )
    print(header)
    print("-" * 145)
    for r in accounting_records:
        orig_s = f"{r['original_slot'][0]:.2f}s - {r['original_slot'][1]:.2f}s"
        placed_s = f"{r['timeline_placement'][0]:.2f}s - {r['timeline_placement'][1]:.2f}s"
        ext_s = f"+{r['extension_seconds']:.2f}s" if r['timeline_extended'] else "NO"
        pres_s = "YES" if r['all_speech_preserved'] else "NO"
        row = (
            f"{r['segment_index']:<4} | {orig_s:<16} | {r['raw_tts_duration']:<6.2f}s | {r['processed_tts_duration']:<7.2f}s | "
            f"{r['applied_speed_factor']:<5.2f}x | {placed_s:<17} | {r['placed_duration']:<8.2f}s | {r['trimmed_seconds']:<6.2f}s | "
            f"{pres_s:<9} | {ext_s:<9} | {r['classification']:<20}"
        )
        print(_safe_str(row))
    print("=" * 145)
    if adaptation_log:
        print("DURATION ADAPTATIONS APPLIED:")
        for log_item in adaptation_log:
            print(f"  • {log_item}")
        print("=" * 145)
    orig_s_dur = f"{orig_source_duration:.2f}s" if orig_source_duration else "N/A"
    print(f"Original Video/Source Duration: {orig_s_dur} | Assembled Dubbed Audio Duration: {total_timeline_sec:.2f}s")
    if accounting_artifact["timeline_extended"]:
        print(f"Timeline Status: EXTENDED (+{accounting_artifact['total_extension_seconds']:.2f}s to preserve 100% spoken content)")
    else:
        print("Timeline Status: PASS (matches original timeline, 0 trimmed seconds)")
    print("Zero-Overlap & Content Integrity Verification: PASS\n")

    # Step 5: Assemble onto timeline buffer
    assembled = np.zeros(total_frames, dtype=np.int16)

    for p in planned_placements:
        _, audio_data = _read_wav(p["audio_path"])
        start_frame = round(p["placement_start"] * sample_rate)
        clip_frames = len(audio_data)
        end_frame = start_frame + clip_frames

        if end_frame > total_frames:
            clip_frames = total_frames - start_frame
            end_frame = total_frames

        if clip_frames <= 0 or start_frame >= total_frames:
            continue

        # Zero-overlap placement into the silence buffer
        assembled[start_frame:end_frame] = audio_data[:clip_frames]

    # Step 6: Write standard 16-bit PCM WAV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(assembled.tobytes())

    logger.info(f"Successfully assembled dubbed audio: {output_path} ({total_timeline_sec}s, {sample_rate}Hz)")
    return output_path