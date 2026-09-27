import json
from google import genai

INPUT = r"output\american_30s\stage1_transcript.json"

client = genai.Client()

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, seg in enumerate(data["dialogue_segments"], 1):
    source = seg["text"]

    prompt = f"""
Translate the following English dialogue into natural, conversational Hindi
for an AI dubbing video.

Requirements:
- Preserve the original meaning exactly.
- Use natural spoken Hindi, not formal/literary Hindi.
- Do not add information.
- Do not omit information.
- Keep it easy to speak aloud.
- Prefer natural phrasing over literal word-for-word translation.
- Return ONLY the Hindi translation.

English:
{source}
"""

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    print(f"\n--- Segment {i} ---")
    print("EN:", source)
    print("HI:", response.output_text.strip())