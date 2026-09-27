import wave

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq


MODEL_ID = "Oriserve/Whisper-Hindi2Hinglish-Apex"
AUDIO_FILE = r"output\chadtag\source_16khz_mono.wav"


device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Device: {device}")
print(f"Dtype: {dtype}")
print(f"Loading: {MODEL_ID}")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(device)
model.eval()

print("Model loaded.")
print("Loading audio...")

with wave.open(AUDIO_FILE, "rb") as wav:
    sample_rate = wav.getframerate()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    frames = wav.readframes(wav.getnframes())

print(
    f"Audio: {sample_rate} Hz, "
    f"{channels} channel(s), "
    f"{sample_width * 8}-bit"
)

import numpy as np

audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

if channels > 1:
    audio = audio.reshape(-1, channels).mean(axis=1)

if sample_rate != 16000:
    raise ValueError(
        f"Expected 16000 Hz audio, got {sample_rate} Hz"
    )

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
)

input_features = inputs.input_features.to(
    device=device,
    dtype=dtype,
)

print("Transcribing...")

with torch.inference_mode():
    generated_ids = model.generate(
        input_features,
        max_new_tokens=444,
    )

result = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)[0]

print("\n" + "=" * 80)
print("APEX TRANSCRIPT")
print("=" * 80)
print(result)
print("=" * 80)
