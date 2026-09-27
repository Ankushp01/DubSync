"""IndicTrans2 translation backend wrapper.

Runs IndicTrans2 through the dedicated .indictrans-venv
environment without importing IndicTrans2 into the main
DubSync environment.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Sequence


# C:\Workpalace\DubSync
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# C:\Workpalace\DubSync\.indictrans-venv\Scripts\python.exe
INDICTRANS_PYTHON = (
    PROJECT_ROOT
    / ".indictrans-venv"
    / "Scripts"
    / "python.exe"
)

# C:\Workpalace\DubSync\dubsync\translation\indictrans_worker.py
INDICTRANS_WORKER = (
    Path(__file__).resolve().parent
    / "indictrans_worker.py"
)


class IndicTrans2Translator:
    """Run IndicTrans2 in its dedicated virtual environment."""

    def __init__(
        self,
        source_language: str = "eng_Latn",
        target_language: str = "hin_Deva",
    ):
        self.source_language = source_language
        self.target_language = target_language

        if not INDICTRANS_PYTHON.exists():
            raise FileNotFoundError(
                "IndicTrans2 Python environment not found: "
                f"{INDICTRANS_PYTHON}"
            )

        if not INDICTRANS_WORKER.exists():
            raise FileNotFoundError(
                "IndicTrans2 worker not found: "
                f"{INDICTRANS_WORKER}"
            )

    def translate_many(
        self,
        texts: Sequence[str],
    ) -> list[str]:
        """Translate multiple texts using one IndicTrans2 process."""

        if not texts:
            return []

        payload = {
            "texts": list(texts),
            "source_language": self.source_language,
            "target_language": self.target_language,
        }

        process = subprocess.run(
            [
                str(INDICTRANS_PYTHON),
                str(INDICTRANS_WORKER),
            ],
            input=json.dumps(
                payload,
                ensure_ascii=False,
            ),
            text=True,
            capture_output=True,
            encoding="utf-8",
        )

        if process.returncode != 0:
            raise RuntimeError(
                "IndicTrans2 translation failed.\n\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            )

        try:
            result = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "IndicTrans2 worker returned invalid JSON.\n\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            ) from exc

        translations = result.get("translations")

        if not isinstance(translations, list):
            raise RuntimeError(
                "IndicTrans2 worker did not return a valid "
                "'translations' list."
            )

        if len(translations) != len(texts):
            raise RuntimeError(
                "IndicTrans2 returned an unexpected number of "
                f"translations: expected {len(texts)}, "
                f"got {len(translations)}"
            )

        return [
            str(text).strip()
            for text in translations
        ]

    def translate(self, text: str) -> str:
        """Translate a single string."""

        if not text or not text.strip():
            return ""

        return self.translate_many([text])[0]