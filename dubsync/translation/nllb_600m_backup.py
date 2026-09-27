"""NLLB translation backend."""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"

SOURCE_LANGUAGE = "eng_Latn"
TARGET_LANGUAGE = "hin_Deva"


class NLLBTranslator:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        source_language: str = SOURCE_LANGUAGE,
        target_language: str = TARGET_LANGUAGE,
    ):
        self.model_name = model_name
        self.source_language = source_language
        self.target_language = target_language

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading NLLB model: {self.model_name}")
        print(f"Device: {self.device}")
        print(
            f"Translation: "
            f"{self.source_language} -> {self.target_language}"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            src_lang=self.source_language,
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
        )

        self.model.to(self.device)
        self.model.eval()

        self.forced_bos_token_id = (
            self.tokenizer.convert_tokens_to_ids(
                self.target_language
            )
        )

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return ""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.forced_bos_token_id,
                max_length=512,
            )

        translated = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
        )[0]

        return translated.strip()