"""IndicTrans2 worker.

This script is executed using:

C:\\Workpalace\\DubSync\\.indictrans-venv\\Scripts\\python.exe
"""

from __future__ import annotations

import json
import sys

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from IndicTransToolkit.processor import IndicProcessor


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")


import os
from pathlib import Path

MODEL_NAME = (
    "ai4bharat/indictrans2-en-indic-dist-200M"
)


def _resolve_indictrans_model_source() -> tuple[str, bool]:
    """Find local cached snapshot of IndicTrans2 to prevent network stalls."""
    cache_dir = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_folder = "models--" + MODEL_NAME.replace("/", "--")
    snapshots_dir = cache_dir / repo_folder / "snapshots"
    if snapshots_dir.is_dir():
        for s in sorted(list(snapshots_dir.iterdir()), key=lambda p: p.stat().st_mtime, reverse=True):
            if s.is_dir() and (s / "config.json").exists() and (s / "model.safetensors").exists():
                return str(s), True
    return MODEL_NAME, False


def load_model():
    """Load IndicTrans2 and its processor."""

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model_source, local_files_only = _resolve_indictrans_model_source()

    print(
        f"Loading IndicTrans2: {model_source} (local_files_only={local_files_only})",
        file=sys.stderr,
    )

    print(
        f"Device: {device}",
        file=sys.stderr,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    model.to(device)
    model.eval()

    processor = IndicProcessor(
        inference=True
    )

    return (
        tokenizer,
        model,
        processor,
        device,
    )


def translate_batch(
    texts: list[str],
    source_language: str,
    target_language: str,
) -> list[str]:
    """Translate a batch of texts."""

    if not texts:
        return []

    (
        tokenizer,
        model,
        processor,
        device,
    ) = load_model()

    processed = processor.preprocess_batch(
        texts,
        src_lang=source_language,
        tgt_lang=target_language,
    )

    inputs = tokenizer(
        processed,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        generated_tokens = model.generate(
            **inputs,
            num_beams=5,
            max_length=256,
        )

    generated_tokens = generated_tokens.cpu()

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    translations = processor.postprocess_batch(
        decoded,
        lang=target_language,
    )

    return [
        text.strip()
        for text in translations
    ]


def main():
    """Read JSON from stdin and write JSON to stdout."""

    payload = json.load(sys.stdin)

    texts = payload["texts"]
    source_language = payload["source_language"]
    target_language = payload["target_language"]

    translations = translate_batch(
        texts=texts,
        source_language=source_language,
        target_language=target_language,
    )

    json.dump(
        {
            "translations": translations
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()