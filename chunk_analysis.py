import json
import re
from pathlib import Path


STAGE2_PATH = Path(
    r"output\american\stage2_translation.json"
)

TARGET_MIN_WORDS = 50
TARGET_MAX_WORDS = 60


def split_sentences(text: str) -> list[str]:
    """Split Hindi text while preserving sentence punctuation."""
    parts = re.findall(
        r"[^।?!]+[।?!]+|[^।?!]+$",
        text,
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


def count_words(text: str) -> int:
    return len(text.split())


def build_chunks(segments):
    chunks = []

    current_sentences = []
    current_segments = []
    current_words = 0

    for segment_index, segment in enumerate(
        segments,
        start=1,
    ):
        sentences = split_sentences(
            segment["translated_text"]
        )

        for sentence in sentences:
            sentence_words = count_words(sentence)

            # If adding this sentence would take us
            # beyond the preferred range, close the
            # current chunk first.
            if (
                current_sentences
                and current_words >= TARGET_MIN_WORDS
                and current_words + sentence_words
                > TARGET_MAX_WORDS
            ):
                chunks.append(
                    {
                        "segments": current_segments.copy(),
                        "text": " ".join(
                            current_sentences
                        ),
                        "words": current_words,
                    }
                )

                current_sentences = []
                current_segments = []
                current_words = 0

            current_sentences.append(sentence)

            if segment_index not in current_segments:
                current_segments.append(
                    segment_index
                )

            current_words += sentence_words

    # Final chunk
    if current_sentences:
        chunks.append(
            {
                "segments": current_segments.copy(),
                "text": " ".join(
                    current_sentences
                ),
                "words": current_words,
            }
        )

    return chunks


data = json.loads(
    STAGE2_PATH.read_text(
        encoding="utf-8"
    )
)

segments = data["dialogue_segments"]

chunks = build_chunks(segments)


print()
print("=" * 80)
print("PROPOSED TTS CHUNKS")
print("=" * 80)

for index, chunk in enumerate(
    chunks,
    start=1,
):
    segment_numbers = ", ".join(
        str(number)
        for number in chunk["segments"]
    )

    print()
    print("-" * 80)
    print(f"CHUNK {index}")
    print("-" * 80)

    print(
        f"Source segments: {segment_numbers}"
    )

    print(
        f"Words: {chunk['words']}"
    )

    print()
    print(chunk["text"])

print()
print("=" * 80)
print(f"TOTAL CHUNKS: {len(chunks)}")
print("=" * 80)
