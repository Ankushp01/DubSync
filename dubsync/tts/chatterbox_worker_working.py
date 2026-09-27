import json
import sys
from pathlib import Path

from dubsync.tts.chatterbox_backend import (
    ChatterboxTTS,
    generate_segment,
    run_stage3,
)


def main():
    payload = json.load(sys.stdin)

    operation = payload.get(
        "operation",
        "stage3",
    )

    if operation == "generate":
        # import chatterbox.mtl_tts

        print("=== WORKER DEBUG ===", file=sys.stderr)
        print(f"Python: {sys.executable}", file=sys.stderr)
        print(
            f"Chatterbox: {chatterbox.mtl_tts.__file__}",
            file=sys.stderr,
        )
        print(
            f"Reference: {payload['reference_audio']}",
            file=sys.stderr,
        )
        print(
            f"Text: {payload['text']}",
            file=sys.stderr,
        )
        print(
            f"Language: {payload['language_id']}",
            file=sys.stderr,
        )
        print(
            f"Device: {payload['device']}",
            file=sys.stderr,
        )

        tts = ChatterboxTTS(
            model_dir=payload.get("model_dir"),
            device=payload["device"],
        )

        print(
            f"Model device: {tts.model.device}",
            file=sys.stderr,
        )
        print(
            f"Model sample rate: {tts.model.sr}",
            file=sys.stderr,
        )

        tts.prepare_voice(
            payload["reference_audio"]
        )

        print(
            "Speaker conditioning prepared.",
            file=sys.stderr,
        )

        wav = tts.generate(
            payload["text"],
            language_id=payload["language_id"],
            reference_audio=payload.get("reference_audio"),
        )

        output_path = Path(
            payload["output_path"]
        ).resolve()

        duration = tts.save(
            wav,
            output_path,
        )

        print(
            json.dumps(
                {
                    "results": [
                        {
                            "audio_path": str(output_path),
                            "duration": duration,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        )

        return

    if operation == "stage3":

        manifest_path = run_stage3(
            stage2_path=payload["stage2_path"],
            output_dir=payload["output_dir"],
            reference_audio=payload["reference_audio"],
            model_dir=payload.get("model_dir"),
            language_id=payload["language_id"],
            device=payload["device"],
        )

        json.dump(
            {
                "manifest_path": str(
                    Path(
                        manifest_path
                    ).resolve()
                )
            },
            sys.stdout,
            ensure_ascii=False,
        )

        return

    raise ValueError(
        f"Unknown worker operation: {operation}"
    )


if __name__ == "__main__":
    main()