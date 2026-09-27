from __future__ import annotations

import json
import wave
from pathlib import Path


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as wf:
        params = wf.getparams()
        frames = wf.readframes(wf.getnframes())

    return params, frames


def assemble_stage3_audio(
    manifest_path: str | Path,
    output_path: str | Path,
) -> Path:
    """
    Assemble Stage 3 WAV segments onto their original timeline.

    Each segment is positioned using its `start` timestamp from
    stage3_tts.json.

    The output duration is determined by the final segment's `end`
    timestamp.
    """

    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Stage 3 manifest not found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    segments = manifest["segments"]

    if not segments:
        raise ValueError("Stage 3 manifest contains no segments.")

    manifest_dir = manifest_path.parent

    # ---------------------------------------------------------
    # Read first WAV to establish audio format.
    # ---------------------------------------------------------

    first_audio = (
        manifest_dir
        / segments[0]["audio"]
    )

    if not first_audio.is_file():
        raise FileNotFoundError(
            f"Segment audio not found: {first_audio}"
        )

    params, _ = _read_wav(first_audio)

    sample_rate = params.framerate
    channels = params.nchannels
    sample_width = params.sampwidth

    if channels != 1:
        raise ValueError(
            "Stage 4 currently expects mono TTS segments."
        )

    # ---------------------------------------------------------
    # Determine final timeline duration.
    # ---------------------------------------------------------

    timeline_end = max(
        float(segment["end"])
        for segment in segments
    )

    total_frames = round(
        timeline_end * sample_rate
    )

    # PCM silence.
    silence_frame = (
        b"\x00"
        * sample_width
        * channels
    )

    assembled = bytearray(
        silence_frame * total_frames
    )

    # ---------------------------------------------------------
    # Place every segment at its timestamp.
    # ---------------------------------------------------------

    for segment in segments:

        audio_path = (
            manifest_dir
            / segment["audio"]
        )

        if not audio_path.is_file():
            raise FileNotFoundError(
                f"Segment audio not found: {audio_path}"
            )

        segment_params, audio_data = _read_wav(
            audio_path
        )

        if segment_params.framerate != sample_rate:
            raise ValueError(
                f"Sample-rate mismatch in "
                f"{audio_path.name}: "
                f"{segment_params.framerate} != "
                f"{sample_rate}"
            )

        if segment_params.nchannels != channels:
            raise ValueError(
                f"Channel mismatch in "
                f"{audio_path.name}"
            )

        if segment_params.sampwidth != sample_width:
            raise ValueError(
                f"Sample-width mismatch in "
                f"{audio_path.name}"
            )

        start = float(segment["start"])

        start_frame = round(
            start * sample_rate
        )

        available_frames = (
            total_frames - start_frame
        )

        segment_frames = (
            len(audio_data)
            // (sample_width * channels)
        )

        frames_to_copy = min(
            segment_frames,
            available_frames,
        )

        byte_start = (
            start_frame
            * sample_width
            * channels
        )

        byte_end = (
            byte_start
            + frames_to_copy
            * sample_width
            * channels
        )

        assembled[
            byte_start:byte_end
        ] = audio_data[
            :frames_to_copy
            * sample_width
            * channels
        ]

        actual_duration = (
            segment_frames / sample_rate
        )

        print(
            f"Segment {segment['index']}: "
            f"{start:.3f}s -> "
            f"{start + actual_duration:.3f}s "
            f"({actual_duration:.3f}s)"
        )

    # ---------------------------------------------------------
    # Write final WAV.
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with wave.open(
        str(output_path),
        "wb",
    ) as wf:

        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)

        wf.writeframes(assembled)

    final_duration = (
        total_frames / sample_rate
    )

    print()
    print("=" * 80)
    print("STAGE 4 AUDIO ASSEMBLY COMPLETE")
    print("=" * 80)
    print(f"Output: {output_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {final_duration:.3f} seconds")

    return output_path






# """Audio timeline assembly for DubSync."""
# from __future__ import annotations

# import json
# from pathlib import Path

# import torch
# import torchaudio


# def assemble_stage3_audio(
#     stage3_manifest: str | Path,
#     tts_dir: str | Path,
#     output_path: str | Path,
# ) -> Path:
#     """Place Stage 3 WAVs on the original dialogue timeline."""

#     stage3_manifest = Path(stage3_manifest).resolve()
#     tts_dir = Path(tts_dir).resolve()
#     output_path = Path(output_path).resolve()

#     if not stage3_manifest.is_file():
#         raise FileNotFoundError(
#             f"Stage 3 manifest not found: {stage3_manifest}"
#         )

#     with stage3_manifest.open("r", encoding="utf-8") as file:
#         data = json.load(file)

#     segments = data["segments"]

#     if not segments:
#         raise ValueError("Stage 3 manifest contains no segments.")

#     loaded = []

#     sample_rate = None
#     total_samples = 0

#     for segment in segments:
#         audio_path = tts_dir / segment["audio"]

#         if not audio_path.is_file():
#             raise FileNotFoundError(
#                 f"TTS audio not found: {audio_path}"
#             )

#         wav, sr = torchaudio.load(str(audio_path))

#         if wav.shape[0] != 1:
#             raise ValueError(
#                 f"Expected mono audio: {audio_path}"
#             )

#         if sample_rate is None:
#             sample_rate = sr
#         elif sr != sample_rate:
#             raise ValueError(
#                 f"Sample-rate mismatch: {audio_path} "
#                 f"has {sr}, expected {sample_rate}"
#             )

#         start = float(segment["start"])
#         start_sample = round(start * sample_rate)

#         end_sample = start_sample + wav.shape[-1]

#         total_samples = max(
#             total_samples,
#             end_sample,
#         )

#         loaded.append(
#             (
#                 start_sample,
#                 wav,
#                 segment,
#             )
#         )

#     timeline = torchaudio.functional.resample(
#         torch.zeros(1, total_samples),
#         sample_rate,
#         sample_rate,
#     )

#     for start_sample, wav, segment in loaded:
#         end_sample = start_sample + wav.shape[-1]

#         timeline[:, start_sample:end_sample] += wav

#     output_path.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )

#     torchaudio.save(
#         str(output_path),
#         timeline,
#         sample_rate,
#     )

#     duration = timeline.shape[-1] / sample_rate

#     print(f"Created: {output_path}")
#     print(f"Sample rate: {sample_rate} Hz")
#     print(f"Duration: {duration:.2f} seconds")

#     return output_path