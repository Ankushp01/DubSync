"""Gemini-based duration-aware translation rewriting."""

from __future__ import annotations

import os

from google import genai


DEFAULT_MODEL = "gemini-3.6-flash"


class DurationRewriter:
    """Rewrite translated dialogue to better fit a target duration."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
    ) -> None:
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY environment variable is not set."
            )

        self.model = model
        self.client = genai.Client(api_key=self.api_key)

    def rewrite(
        self,
        source_text: str,
        current_translation: str,
        target_duration: float,
        generated_duration: float,
        target_language: str = "Hindi",
    ) -> str:
        """Rewrite a translation to better fit the target duration."""

        if target_duration <= 0:
            raise ValueError("target_duration must be greater than zero.")

        if generated_duration <= 0:
            raise ValueError("generated_duration must be greater than zero.")

        prompt = f"""
You are adapting a translated dialogue line for video dubbing.

The original dialogue is:

{source_text}

The current {target_language} translation is:

{current_translation}

The target speaking duration is approximately:
{target_duration:.2f} seconds

The current generated speech duration is:
{generated_duration:.2f} seconds

Rewrite the {target_language} translation so that it can be spoken
naturally within approximately {target_duration:.2f} seconds.

The required duration reduction is approximately:
{max(0.0, (generated_duration - target_duration) / generated_duration * 100):.1f}%

Requirements:
- Preserve the meaning of the original dialogue.
- Preserve the essential intent, but do not preserve every word literally.
- This is spoken video dubbing, not a literal translation exercise.
- You may restructure the sentence completely if that makes it more natural
  and concise.
- Remove unnecessary pronouns, repetition, filler, and redundant phrasing.
- Prefer the shortest natural spoken form that still conveys the meaning.
- Do not add information that is not present in the original.
- The target duration is a hard practical constraint.
- If the current translation is too long, make a substantial reduction,
  not merely a small wording change.
- Do not explain your changes.
- Return ONLY the rewritten {target_language} sentence.
"""

        interaction = self.client.interactions.create(
            model=self.model,
            input=prompt,
        )

        text = interaction.output_text.strip()

        if not text:
            raise RuntimeError(
                "Gemini returned an empty duration rewrite."
            )

        return text