# DubSync AI

DubSync is a staged local video-dubbing pipeline. Stage 1 is implemented:

`video -> 16 kHz mono WAV -> Faster-Whisper ASR -> timestamped dialogue JSON`

Later stages (translation, duration adaptation, TTS, synchronization, lip sync,
and assembly) deliberately remain out of scope until Stage 1 is working locally.

## Prerequisites

- Windows, Python 3.11, and an NVIDIA GPU driver. The detected RTX 3050 4 GB is
  supported using the `int8_float16` configuration.
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on `PATH`.
  Alternatively set `ffmpeg.executable` to the executable's full path in
  `configs/local.toml`.
- A CUDA-compatible Faster-Whisper/CTranslate2 installation. Do not install
  PyTorch solely for Stage 1; Faster-Whisper uses CTranslate2 directly.
- For Windows GPU inference, install the project-pinned CUDA 12 cuBLAS and
  cuDNN 8 runtime wheels from `requirements-gpu-windows.txt`. They are not the
  full CUDA Toolkit and do not modify the NVIDIA display driver.

## Setup

Use the explicit Python 3.11 interpreter if `python` is not on PATH:

```powershell
& "C:\Users\ankus\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu-windows.txt
Copy-Item configs\default.toml configs\local.toml
```

Model weights are never stored in source modules. By default Faster-Whisper
downloads the selected `small` model into its cache on first run. To keep it in
the project-level model directory, download/convert it separately and set
`whisper.model_path` in `configs/local.toml`.

## Run Stage 1

```powershell
python -m dubsync input\your-video.mp4 --config configs\local.toml
```

The artifacts are written to `output\your-video\`:

- `source_16khz_mono.wav` – the ASR working audio.
- `stage1_transcript.json` – source ASR segments plus dialogue groupings, all
  timestamps in seconds. Both segment sets are retained for later alignment.

## Test

```powershell
python -m unittest discover -s tests -v
```

The tests do not load Whisper or a model. They verify timestamp mapping,
dialogue-boundary behavior, and the actionable FFmpeg error path.

## GPU memory behavior

Whisper is created only within the transcription step, then immediately
released and its CUDA cache is emptied. No later-stage model is loaded during
Stage 1. Start with the configured `small` model and `int8_float16`; reduce to
`base` or set `compute_type = "int8"` if the 4 GB GPU cannot accommodate the
model on the installed CUDA stack.
