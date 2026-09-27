import json
from pathlib import Path
from google import genai

INPUT = Path(r"output\american_30s\stage1_transcript.json")
OUTPUT = Path(r"output\american_30s\gemini_duration_test.json")

client = genai.Client()

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for i, seg in enumerate(data["dialogue_segments"], 1):
    source = seg["text"]
    start = float(seg["start"])
    end = float(seg["end"])
    duration = end - start

    prompt = f"""
You are adapting dialogue for an AI video dubbing system.

English original:
{source}

Original dialogue duration:
{duration:.2f} seconds

Translate and adapt the dialogue into natural, conversational Hindi
for spoken dubbing.

Requirements:

- The English original is the source of truth for meaning.
- Preserve the complete meaning and intent.
- You may ADD natural words, phrases, or connective expressions if
  necessary to make the spoken dialogue fit the target duration.
- You may REMOVE redundant words or use more concise phrasing if
  necessary to make it shorter.
- You may restructure the sentence completely if that produces more
  natural spoken Hindi.
- Do not add new facts, opinions, or information.
- Do not remove important information or meaning.
- Do not translate word-for-word when that sounds unnatural.
- The result should sound like something a native Hindi speaker would
  naturally say aloud.
- Aim for approximately {duration:.2f} seconds at a normal speaking pace.
- Prioritize naturalness and meaning over exact duration.
- Return ONLY the final Hindi dialogue.
"""

    try:
        response = client.interactions.create(
            model="gemini-3.6-flash",
            input=prompt,
        )
    except Exception as e:
        print(f"Gemini request failed: {e}")
        break

    hindi = response.output_text.strip()

    results.append({
        "segment": i,
        "start": start,
        "end": end,
        "target_duration": duration,
        "source_text": source,
        "translated_text": hindi,
    })

    print(f"\n--- Segment {i} ---")
    print(f"Target: {duration:.2f}s")
    print("EN:", source)
    print("HI:", hindi)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        {
            "schema_version": 1,
            "model": "gemini-3.6-flash",
            "segments": results,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

print(f"\nSaved: {OUTPUT}")