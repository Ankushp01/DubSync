"""Gemini-based text duration adaptation for DubSync."""

from __future__ import annotations

import os
import re

from google import genai


class GeminiDurationAdapter:
    """
    Use Gemini to rewrite translated text so that its spoken duration
    is closer to the target duration.

    Gemini does NOT generate audio.
    It only adapts the text.
    """

    def __init__(
        self,
        model: str | None = None,
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set."
            )

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.6-flash-lite",
            )
        )

        self.client = genai.Client(
            api_key=api_key
        )

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean Gemini's response so only the adapted text remains."""

        text = text.strip()

        # Remove markdown code fences.
        text = re.sub(
            r"^```(?:text|plaintext|hindi)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

        text = text.strip()

        # Remove surrounding quotation marks if Gemini added them.
        if (
            len(text) >= 2
            and text[0] in {'"', "'"}
            and text[-1] == text[0]
        ):
            text = text[1:-1].strip()

        return text

    def adapt(
        self,
        *,
        source_text: str,
        translated_text: str,
        target_duration: float,
        generated_duration: float,
        target_language: str = "Hindi",
    ) -> str:
        """
        Ask Gemini to adapt translated text to the required duration.
        """

        difference = generated_duration - target_duration

        if difference > 0:
            direction = "SHORTEN"
            instruction = """
The current spoken version is too long.

Rewrite the Hindi translation so that it can be spoken faster
because it contains fewer words/syllables.

Remove unnecessary wording, repetition, and verbosity.

Do NOT remove important meaning.
Do NOT introduce new information.
"""
        else:
            direction = "EXPAND"
            instruction = """
The current spoken version is too short.

Rewrite the Hindi translation so that it naturally takes longer
to speak.

You may use slightly fuller, more natural Hindi phrasing,
but you MUST preserve the original meaning.

Do NOT introduce new information.
Do NOT repeat the same idea unnecessarily.
"""

        prompt = f"""
You are adapting translated dialogue for an AI video dubbing system.

The original source dialogue is:

{source_text}

The current {target_language} translation is:

{translated_text}

Timing information:

Target duration: {target_duration:.2f} seconds
Current generated duration: {generated_duration:.2f} seconds
Difference: {difference:+.2f} seconds

Your task is to {direction} the translated dialogue.

{instruction}

Requirements:

1. Preserve the meaning of the original English dialogue.
2. Keep the dialogue natural spoken Hindi.
3. Do not add facts, explanations, or information that are not present
   in the original.
4. Do not translate word-for-word if that produces unnatural Hindi.
5. Preserve the conversational tone.
6. The revised text should move the spoken duration toward
   approximately {target_duration:.2f} seconds.
7. Do not mention timing, duration, translation, or these instructions.
8. Return ONLY the revised Hindi dialogue.
"""

        print(
            f"Gemini duration adaptation: {direction}"
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        result = response.text

        if not result:
            raise RuntimeError(
                "Gemini returned an empty response."
            )

        result = self._clean_response(result)

        if not result:
            raise RuntimeError(
                "Gemini returned empty text after cleaning."
            )

        return result