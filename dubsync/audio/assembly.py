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
) -> Path:
    """Assemble Stage 3 TTS audio segments onto a clean, natural, non-overlapping timeline.

    Guarantees:
    1. Every dialogue segment has an explicit non-overlapping slot.
    2. TTS audio clips NEVER overlap with adjacent segments.
    3. Natural speech pacing is strictly prioritized: speed-up is capped at 1.15x (normal)
       and 1.25x (max fallback). Extreme speeds (e.g. 1.5x - 2.33x+) are NEVER applied.
    4. Clips can safely expand into natural trailing silence gaps before the next segment.
    5. Silence and natural pauses between dialogue segments are preserved.
    6. Comprehensive duration quality check & classification table is logged.
    7. Final audio duration matches the source video/audio duration.
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
    source_duration: Optional[float] = None
    if source_audio_path:
        src = Path(source_audio_path).resolve()
        if src.is_file():
            source_duration = sf.info(str(src)).duration

    if source_duration is None:
        default_src = manifest_dir / "source_16khz_mono.wav"
        if default_src.is_file():
            source_duration = sf.info(str(default_src)).duration

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

        # Determine when next speech segment begins
        if idx + 1 < len(sorted_segments):
            next_start = sorted_segments[idx + 1]["start"]
        else:
            next_start = source_duration if source_duration else round(seg_start + max(target_dur, tts_dur) + 1.0, 3)

        # Maximum duration this clip could extend into without encroaching on next speech
        max_extendable_dur = round(max(target_dur, next_start - seg_start - MIN_SILENCE_MARGIN), 3)

        speed_factor = 1.0
        final_dur = tts_dur
        final_clip_path = audio_file
        classification = "NORMAL"

        if tts_dur <= target_dur:
            # Fits within original slot without any acceleration
            speed_factor = 1.0
            final_dur = tts_dur
            classification = "TEXT_SHORTENED" if text_was_adapted else "NORMAL"
        else:
            # Clip is longer than target_dur. Check required speed.
            required_speed = tts_dur / target_dur

            if required_speed <= MAX_NORMAL_SPEED:
                # Mild natural speed adjustment (1.00x - 1.15x) fits exactly inside target_dur
                speed_factor = round(required_speed, 2)
                desired_dur = target_dur
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
                    f"Segment {idx+1}: Mild speed adjustment {required_speed:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) to fit {target_dur:.2f}s"
                )

            elif required_speed <= MAX_FALLBACK_SPEED:
                # Moderate speed adjustment (1.15x - 1.25x) fits inside target_dur
                speed_factor = round(required_speed, 2)
                desired_dur = target_dur
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
                    f"Segment {idx+1}: Moderate speed adjustment {required_speed:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) to fit {target_dur:.2f}s"
                )

            else:
                # Required speed > 1.25x! DO NOT use excessive atempo!
                # Prioritize natural speech: clamp atempo to MAX_FALLBACK_SPEED (1.25x)
                # and extend into trailing silence gap before next segment.
                speed_factor = MAX_FALLBACK_SPEED
                compressed_dur = round(tts_dur / MAX_FALLBACK_SPEED, 3)

                if compressed_dur <= max_extendable_dur:
                    # Fits comfortably inside trailing silence gap!
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
                        f"Segment {idx+1}: Clamped speed to {MAX_FALLBACK_SPEED:.2f}x ({tts_dur:.2f}s -> {final_dur:.2f}s) utilizing {max_extendable_dur:.2f}s silence gap"
                    )
                else:
                    # Compressed audio still exceeds trailing gap.
                    # Compress at 1.25x and smoothly trim/fade the trailing audio to avoid overlap.
                    adapted_file = manifest_dir / f"segment_{idx+1:03d}_adapted.wav"
                    time_stretch_audio(
                        input_wav=audio_file,
                        output_wav=adapted_file,
                        target_duration=compressed_dur,
                        ffmpeg_executable=ffmpeg_executable,
                    )
                    trimmed_file = manifest_dir / f"segment_{idx+1:03d}_trimmed.wav"
                    final_dur = _safe_trim_audio(
                        input_wav=adapted_file,
                        output_wav=trimmed_file,
                        max_duration=max_extendable_dur,
                    )
                    final_clip_path = trimmed_file
                    classification = "DURATION_WARNING"
                    adaptation_log.append(
                        f"Segment {idx+1}: DURATION_WARNING: Clamped to {MAX_FALLBACK_SPEED:.2f}x and trimmed to {max_extendable_dur:.2f}s to avoid overlap"
                    )

        placement_start = seg_start
        placement_end = round(placement_start + final_dur, 3)

        planned_placements.append({
            "index": idx + 1,
            "seg_start": seg_start,
            "seg_end": seg_end,
            "target_duration": target_dur,
            "tts_duration": tts_dur,
            "available_duration": target_dur,
            "final_duration": final_dur,
            "placement_start": placement_start,
            "placement_end": placement_end,
            "audio_path": final_clip_path,
            "adapted_speed": speed_factor,
            "adaptation_attempts": adaptation_attempts,
            "classification": classification,
            "translated_text": trans_text,
        })

    # Step 3: Programmatic Timeline Validation
    for i, p in enumerate(planned_placements):
        if i > 0:
            prev = planned_placements[i - 1]
            if p["seg_start"] < prev["seg_end"]:
                raise ValueError(
                    f"Timeline validation failed: Segment {p['index']} start ({p['seg_start']}s) < previous segment end ({prev['seg_end']}s)"
                )
            if p["placement_start"] < prev["placement_end"]:
                raise ValueError(
                    f"Timeline validation failed: Segment {p['index']} clip placement ({p['placement_start']}s) overlaps previous clip end ({prev['placement_end']}s)"
                )

        if i + 1 < len(planned_placements):
            next_p = planned_placements[i + 1]
            if p["placement_end"] > next_p["seg_start"]:
                raise ValueError(
                    f"Timeline validation failed: Segment {p['index']} clip end ({p['placement_end']}s) extends into next segment start ({next_p['seg_start']}s)"
                )

    # Determine total timeline frames
    max_clip_end = max(p["placement_end"] for p in planned_placements)
    if source_duration is None:
        source_duration = max(max_clip_end, max(p["seg_end"] for p in planned_placements))
    else:
        source_duration = max(source_duration, max_clip_end)

    total_frames = round(source_duration * sample_rate)
    total_timeline_sec = round(total_frames / sample_rate, 3)

    # Step 4: Log detailed duration quality check and classification table
    print("\n" + "=" * 135)
    print("DUBBED AUDIO DURATION QUALITY CHECK & TIMELINE VALIDATION")
    print("=" * 135)
    header = (
        f"{'Segment':<7} | {'Start':<6} | {'End':<6} | {'Avail Dur':<9} | "
        f"{'Translated Text':<30} | {'TTS Dur':<7} | {'Final Dur':<9} | "
        f"{'Speed Adj':<9} | {'Adapt Attempts':<14} | {'Classification':<22}"
    )
    print(header)
    print("-" * 135)
    for p in planned_placements:
        txt = _safe_str(p["translated_text"])
        short_txt = (txt[:27] + "...") if len(txt) > 30 else txt
        row = (
            f"{p['index']:<7} | {p['seg_start']:<6.2f} | {p['seg_end']:<6.2f} | "
            f"{p['available_duration']:<7.2f}s | {short_txt:<30} | "
            f"{p['tts_duration']:<5.2f}s | {p['final_duration']:<7.2f}s | "
            f"{p['adapted_speed']:<7.2f}x | {p['adaptation_attempts']:<14} | "
            f"{p['classification']:<22}"
        )
        print(_safe_str(row))
    print("=" * 135)
    if adaptation_log:
        print("DURATION ADAPTATIONS APPLIED:")
        for log_item in adaptation_log:
            print(f"  • {log_item}")
        print("=" * 135)
    print(f"Total Source Duration: {source_duration:.2f}s | Assembled Audio Duration: {total_timeline_sec:.2f}s")
    print("Timeline Verification Status: PASS (0 overlaps, natural speech pacing, silence preserved)\n")

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