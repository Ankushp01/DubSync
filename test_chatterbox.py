import json
import time
from pathlib import Path

import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


STAGE2_FILE = Path(r"output\american\stage2_translation.json")
OUTPUT_FILE = Path(r"output\american\test_tts_segment_01.wav")

MODEL_DIR = Path(
    r"C:\Users\ankus\.cache\huggingface\hub\models--ResembleAI--chatterbox\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)


print("Loading Stage 3...")

with STAGE2_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

segment = data["dialogue_segments"][0]

text = segment["translated_text"]
target_duration = segment["end"] - segment["start"]

print("\n" + "=" * 80)
print("TEXT")
print("=" * 80)
print(text)

print(f"\nTarget duration: {target_duration:.2f} seconds")


print("\nLoading Chatterbox Multilingual...")

model = ChatterboxMultilingualTTS.from_local(
    ckpt_dir=str(MODEL_DIR),
    device="cuda",
)

print("Chatterbox loaded.")


print("\nGenerating speech...")

start_time = time.time()

REFERENCE_AUDIO = Path(
    r"output\american\voice_reference.wav"
)

wav = model.generate(
    text=text,
    language_id="hi",
    audio_prompt_path=str(REFERENCE_AUDIO),
)

generation_time = time.time() - start_time

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

torchaudio.save(
    str(OUTPUT_FILE),
    wav.cpu(),
    model.sr,
)

generated_duration = wav.shape[-1] / model.sr

print("\n" + "=" * 80)
print("TTS RESULT")
print("=" * 80)
print(f"Output: {OUTPUT_FILE}")
print(f"Generated duration: {generated_duration:.2f} seconds")
print(f"Target duration:    {target_duration:.2f} seconds")
print(f"Difference:         {generated_duration - target_duration:+.2f} seconds")
print(f"Generation time:    {generation_time:.2f} seconds")
print("=" * 80)