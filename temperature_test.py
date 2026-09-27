from pathlib import Path

import torchaudio

from dubsync.tts.chatterbox_backend import ChatterboxTTS


MODEL_DIR = (
    r"C:\Users\ankus\.cache\huggingface\hub"
    r"\models--ResembleAI--chatterbox"
    r"\snapshots\5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"
)

REFERENCE_AUDIO = (
    r"output\american_30s\source_16khz_mono.wav"
)

OUTPUT_DIR = Path(
    r"output\american_30s\tts_temperature_test"
)

TEXT = (
    "लेकिन, अगर मैं एक चीज़ को दूसरों पर प्राथमिकता देता,"
    " तो वह आपके होंठ हिलाना होगा।"
)


tts = ChatterboxTTS(
    model_dir=MODEL_DIR,
    device="cuda",
)

tts.prepare_voice(REFERENCE_AUDIO)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

for temperature in (0.8, 0.6, 0.4):

    print()
    print("=" * 50)
    print(f"Temperature: {temperature}")
    print("=" * 50)

    wav = tts.model.generate(
        text=TEXT,
        language_id="hi",
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=temperature,
        repetition_penalty=2.0,
        min_p=0.05,
        top_p=1.0,
    )

    output_path = (
        OUTPUT_DIR
        / f"temp_{str(temperature).replace('.', '_')}.wav"
    )

    torchaudio.save(
        str(output_path),
        wav.cpu(),
        tts.model.sr,
    )

    duration = wav.shape[-1] / tts.model.sr

    print(f"Saved: {output_path}")
    print(f"Duration: {duration:.2f}s")