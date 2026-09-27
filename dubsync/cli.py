"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from dubsync.config import load_config
from dubsync.pipeline import (
    run_stage1,
    run_stage2,
    run_stage3,
    run_stage4,
    run_stage5,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DubSync local pipeline"
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Input video path",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.toml"),
    )

    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "4", "5", "all"],
        default="1",
        help="Pipeline stage to run",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.stage == "1":
        result = run_stage1(args.video, config)
        print(f"Stage 1 complete: {result}")

    elif args.stage == "2":
        stage1_path = (
            config.paths.output_dir
            / args.video.stem
            / "stage1_transcript.json"
        )

        result = run_stage2(stage1_path, config)
        print(f"Stage 2 complete: {result}")

    elif args.stage == "3":
        stage2_path = (
            config.paths.output_dir
            / args.video.stem
            / "stage2_translation.json"
        )

        result = run_stage3(stage2_path, config)
        print(f"Stage 3 complete: {result}")

    elif args.stage == "4":
        output_dir = (
            config.paths.output_dir
            / args.video.stem
        )

        stage3_path = (
            output_dir
            / "stage3_tts.json"
        )

        result = run_stage4(
            stage3_path,
            output_dir,
        )

        print(f"Stage 4 complete: {result}")

    elif args.stage == "5":
        output_dir = (
            config.paths.output_dir
            / args.video.stem
        )

        dubbed_audio_path = (
            output_dir
            / "audio"
            / "dubbed_audio.wav"
        )

        result = run_stage5(
            args.video,
            dubbed_audio_path,
            output_dir,
            config.ffmpeg.executable,
        )

        print(f"Stage 5 complete: {result}")

    else:
        # Stage 1
        stage1_path = run_stage1(args.video, config)
        print(f"Stage 1 complete: {stage1_path}")

        # Stage 2
        stage2_path = run_stage2(stage1_path, config)
        print(f"Stage 2 complete: {stage2_path}")

        # Stage 3
        stage3_path = run_stage3(stage2_path, config)
        print(f"Stage 3 complete: {stage3_path}")

        # Stage 4
        output_dir = (
            config.paths.output_dir
            / args.video.stem
        )

        stage4_path = run_stage4(
            stage3_path,
            output_dir,
        )
        print(f"Stage 4 complete: {stage4_path}")

        # Stage 5
        stage5_path = run_stage5(
            args.video,
            stage4_path,
            output_dir,
            config.ffmpeg.executable,
        )
        print(f"Stage 5 complete: {stage5_path}")


if __name__ == "__main__":
    main()