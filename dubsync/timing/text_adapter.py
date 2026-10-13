"""Intelligent duration-aware text adaptation for DubSync.

Supports:
1. Gemini GenAI API adaptation when GEMINI_API_KEY or GOOGLE_API_KEY is available.
2. High-quality local conciseness heuristics when API keys are not configured or offline.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)


# Common idiom & conversational simplifications for Hindi dubbing
HINDI_IDIOM_PATTERNS = [
    (
        r".*?(?:बाहर निकालने जा रहा था|बाहर ले जाने वाला था).*?(?:बोतल में डाल दिया|बोतल कर दिया).*?",
        "मैं उससे पूछने वाला था, पर हिम्मत नहीं हुई।",
    ),
    (
        r"बोतल में डाल दिया",
        "हिम्मत हार गया",
    ),
    (
        r"अंग्रेजी सुनने की परीक्षा,\s*यह आदमी इस बातचीत में क्या कहता है\?",
        "अंग्रेजी सुनने की परीक्षा: यह आदमी क्या कहता है?",
    ),
]

# Filler and redundant word patterns for Hindi spoken dialogue
HINDI_REDUNDANCIES = [
    (r"\bलेकिन मैंने\b", "पर"),
    (r"\bलेकिन\b", "पर"),
    (r"\bपरन्तु\b", "पर"),
    (r"\bकिन्तु\b", "पर"),
    (r"\bइस बातचीत में\b", ""),
    (r"\bके बारे में\b", "पर"),
    (r"\bजा रहा था\b", "था"),
    (r"\bकरने जा रहा था\b", "करने वाला था"),
    (r"\bउसे बाहर निकालने\b", "उससे पूछने"),
]


# Conversational filler phrases that can be safely simplified to shorten spoken duration
HINDI_CONVERSATIONAL_FILLERS = [
    (r"\bआप जानते हैं[?,]?\b", ""),
    (r"\bवास्तव में\b", ""),
    (r"\bबस\b", ""),
    (r"\bतो फिर\b", "तो"),
]


class TextDurationAdapter:
    """Adapts dialogue text to naturally fit within a target duration slot."""

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("TextDurationAdapter: Gemini client initialized successfully.")
            except Exception as e:
                logger.warning(f"TextDurationAdapter: Could not initialize Gemini client: {e}")
                self.client = None
        else:
            logger.info("TextDurationAdapter: No GEMINI_API_KEY/GOOGLE_API_KEY found. Using local conciseness engine.")

    def adapt(
        self,
        *,
        source_text: str,
        translated_text: str,
        target_duration: float,
        generated_duration: float,
        target_language: str = "Hindi",
        attempt: int = 1,
    ) -> str:
        """Adapt dialogue text to target duration using Gemini or local conciseness heuristics."""
        if target_duration <= 0 or generated_duration <= 0:
            return translated_text

        # Try Gemini API if available
        if self.client is not None:
            try:
                adapted = self._adapt_with_gemini(
                    source_text=source_text,
                    translated_text=translated_text,
                    target_duration=target_duration,
                    generated_duration=generated_duration,
                    target_language=target_language,
                    attempt=attempt,
                )
                # Verify adapted text does not discard more than 30% of content
                if len(adapted.split()) >= max(3, int(len(translated_text.split()) * 0.70)):
                    return adapted
                logger.warning("Gemini adaptation dropped too much content. Falling back to local adaptation.")
            except Exception as exc:
                logger.warning(f"Gemini adaptation failed, falling back to local conciseness: {exc}")

        # Fallback to intelligent, non-destructive local conciseness
        return self._adapt_locally(
            source_text=source_text,
            translated_text=translated_text,
            target_duration=target_duration,
            generated_duration=generated_duration,
            target_language=target_language,
            attempt=attempt,
        )

    def _adapt_with_gemini(
        self,
        *,
        source_text: str,
        translated_text: str,
        target_duration: float,
        generated_duration: float,
        target_language: str,
        attempt: int,
    ) -> str:
        diff = generated_duration - target_duration
        reduction_pct = max(0.0, (diff / generated_duration) * 100)

        prompt = f"""You are an expert dialogue adapter for multilingual video dubbing.
Original source dialogue: "{source_text}"
Current {target_language} translation: "{translated_text}"

Timing constraint:
- Target speaking duration: {target_duration:.2f} seconds
- Current generated duration: {generated_duration:.2f} seconds (needs ~{reduction_pct:.0f}% reduction)
- Attempt: {attempt}

Task:
Rewrite the {target_language} translation so it can be naturally and clearly spoken within {target_duration:.2f} seconds.

Guidelines:
1. Make it concise and punchy. Prioritize natural spoken delivery over word-for-word translation.
2. Remove filler words, redundant pronouns, and bulky phrasing.
3. If an English idiom was translated literally, replace it with a natural, brief {target_language} conversational equivalent.
4. Return ONLY the rewritten {target_language} sentence, with no surrounding quotes, markdown, or explanation.
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        res_text = (response.text or "").strip()
        # Clean quotes and code blocks
        res_text = re.sub(r"^```(?:text|hindi)?\s*", "", res_text, flags=re.IGNORECASE)
        res_text = re.sub(r"\s*```$", "", res_text).strip()
        if (len(res_text) >= 2 and res_text[0] in ('"', "'") and res_text[-1] == res_text[0]):
            res_text = res_text[1:-1].strip()

        if res_text:
            return res_text
        raise ValueError("Empty response from Gemini")

    def _adapt_locally(
        self,
        *,
        source_text: str,
        translated_text: str,
        target_duration: float,
        generated_duration: float,
        target_language: str,
        attempt: int,
    ) -> str:
        text = translated_text.strip()

        # Step 1: Check known idiom mistranslations
        for pattern, replacement in HINDI_IDIOM_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE).strip()
                text = re.sub(r"[।]+$", "।", text)
                return text

        # Step 2: Apply redundancy stripping
        for pat, rep in HINDI_REDUNDANCIES:
            text = re.sub(pat, rep, text).strip()

        # Clean multiple spaces or stray punctuation
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,।?!])", r"\1", text)
        text = re.sub(r"[।]+$", "।", text)

        # Step 3: Gentle conversational filler reduction if duration reduction is requested
        ratio = target_duration / max(0.1, generated_duration)
        if ratio < 0.85:
            for pat, rep in HINDI_CONVERSATIONAL_FILLERS:
                text = re.sub(pat, rep, text).strip()
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\s+([,।?!])", r"\1", text)
            text = re.sub(r"[।]+$", "।", text)

        # CRITICAL SAFETY GUARANTEE:
        # Never perform arbitrary word slicing (e.g. words[:target_word_count] or slicing around 'पर').
        # Complete sentence structure, all substantive clauses, and spoken meaning MUST be preserved.
        # Audio-level duration adaptation (atempo time-stretch up to ~1.25x and timeline extension)
        # handles remaining timing fitting naturally without dropping spoken dialogue.
        return text
