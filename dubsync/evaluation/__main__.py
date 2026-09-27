from __future__ import annotations

import argparse
from pathlib import Path

from dubsync.evaluation.stage3_4 import evaluate_stage3_and_stage4


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    output_dir = args.output_dir.resolve()

    stage3_manifest = output_dir / "stage3_tts.json"
    assembled_audio = (
        output_dir
        / "audio"
        / "dubbed_audio.wav"
    )

    stage3_path, stage4_path = evaluate_stage3_and_stage4(
        stage3_manifest,
        assembled_audio,
        output_dir / "evaluation",
    )

    print(f"Stage 3 metrics: {stage3_path}")
    print(f"Stage 4 metrics: {stage4_path}")


if __name__ == "__main__":
    main()