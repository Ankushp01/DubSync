import json
from google import genai

INPUT = r"output\american_30s\stage2_translation.json"

client = genai.Client()

with open(INPUT, "r", encoding="utf-8") as f:
    data = json.load(f)

for i, seg in enumerate(data["dialogue_segments"], 1):
    source = seg["source_text"]
    indic = seg["translated_text"]

    prompt = f"""
You are adapting a machine-translated Hindi sentence for an AI dubbing system.

English original:
{source}

IndicTrans2 Hindi translation:
{indic}

Rewrite the Hindi translation into natural, conversational spoken Hindi.

Rules:
- The English original is the source of truth for meaning.
- Preserve the complete meaning of the English original.
- Use the IndicTrans2 translation as a translation reference.
- Fix unnatural, literal, or awkward Hindi.
- Make the result sound natural when spoken aloud.
- Do not add information that is not present in the English.
- Do not remove information from the English.
- Do not explain your changes.
- Return ONLY the final Hindi sentence.
"""

    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt,
    )

    print(f"\n--- Segment {i} ---")
    print("EN :", source)
    print("MT :", indic)
    print("AD :", response.output_text.strip())