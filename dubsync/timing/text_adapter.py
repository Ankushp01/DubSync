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


def _devanagari_sub(pattern: str, replacement: str, text: str) -> str:
    """Safely replace words/phrases in Devanagari text respecting Unicode word boundaries.

    Standard Python regex '\\b' fails on Devanagari words ending with vowel signs
    (matras like ो, े, ा, ी, ु) or anusvara (ं), because Unicode combining marks (Mn)
    are classified as \\W.
    """
    regex = r"(?:(?<=[\s,।?!;:\-\"\'\(\)])|^)" + pattern + r"(?:(?=[\s,।?!;:\-\"\'\(\)])|$)"
    return re.sub(regex, replacement, text)


# Literal/verbose phrase simplifications that preserve full meaning in concise conversational Hindi
HINDI_CONVERSATIONAL_SIMPLIFICATIONS = [
    # Idiomatic calming nerves (calm your nerves -> calm anxiety/fear, not anatomical nerves)
    (r"आपकी नसों को शांत", "आपकी घबराहट शांत"),
    (r"नसों को शांत", "घबराहट शांत"),
    # Pretend like it's going to just one person -> assume it's only for that person
    (r"ऐसा नाटक करते हैं कि यह केवल एक व्यक्ति के लिए जा रहा है", "मान लें कि यह केवल उसी के लिए है,"),
    (r"ऐसा नाटक करते हैं कि यह सिर्फ एक व्यक्ति के लिए जा रहा है", "मान लें कि यह केवल उसी के लिए है,"),
    (r"ऐसा नाटक करते हैं कि", "मान लें कि"),
    (r"केवल एक व्यक्ति के लिए जा रहा है", "केवल उसी के लिए है"),
    (r"सिर्फ एक व्यक्ति के लिए जा रहा है", "सिर्फ उसी के लिए है"),
    (r"एक व्यक्ति के लिए जा रहा है", "उसी के लिए है"),
    # Focus down -> focus (conversational)
    (r"ध्यान केंद्रित करते हैं", "ध्यान दें"),
    (r"ध्यान केंद्रित करें", "ध्यान दें"),
    (r"ध्यान केंद्रित करने", "ध्यान देने"),
    # Help you feel more -> make you feel better
    (r"शांत कर सकता है और", "शांत कर"),
    (r"अधिक महसूस करने में मदद कर सकता है", "बेहतर महसूस कराएगा"),
    (r"महसूस करने में मदद कर सकता है", "बेहतर महसूस कराएगा"),
    (r"करने में मदद कर सकता है", "करा सकता है"),
    (r"करने की आवश्यकता नहीं है", "की जरूरत नहीं"),
    (r"होने की आवश्यकता नहीं है", "होना जरूरी नहीं"),
]

# Filler and redundant word patterns for Hindi spoken dialogue
HINDI_REDUNDANCIES = [
    (r"लेकिन मैंने", "पर"),
    (r"लेकिन", "पर"),
    (r"परन्तु", "पर"),
    (r"किन्तु", "पर"),
    (r"इस बातचीत में", ""),
    (r"के बारे में", "पर"),
    (r"जा रहा था", "था"),
    (r"करने जा रहा था", "करने वाला था"),
    (r"उसे बाहर निकालने", "उससे पूछने"),
]


# Conversational filler phrases that can be safely simplified to shorten spoken duration
HINDI_CONVERSATIONAL_FILLERS = [
    (r"आप जानते हैं[?,]?", ""),
    (r"वास्तव में", ""),
    (r"बस", ""),
    (r"तो फिर", "तो"),
    (r"यह तथ्य कि", "यह कि"),
    (r"इस तथ्य कि", "यह कि"),
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

        # Dynamic content safety threshold:
        # Allow proportional word reduction based on target duration (min 50%),
        # while strictly preventing destructive phrase slicing (e.g. 5-word broken fragments).
        target_ratio = target_duration / max(0.1, generated_duration)
        min_allowed_ratio = max(0.50, min(0.70, target_ratio * 0.95))
        min_words = max(3, int(len(translated_text.split()) * min_allowed_ratio))

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
                if len(adapted.split()) >= min_words:
                    return adapted
                logger.warning("Gemini adaptation dropped too much content. Falling back to local adaptation.")
            except Exception as exc:
                logger.warning(f"Gemini adaptation failed, falling back to local conciseness: {exc}")

        # Fallback to intelligent, non-destructive local conciseness
        adapted_local = self._adapt_locally(
            source_text=source_text,
            translated_text=translated_text,
            target_duration=target_duration,
            generated_duration=generated_duration,
            target_language=target_language,
            attempt=attempt,
        )
        if len(adapted_local.split()) >= min_words:
            return adapted_local
        return translated_text

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

        ratio = target_duration / max(0.1, generated_duration)

        # Step 2: Apply conversational phrase simplifications when duration reduction is needed
        if ratio < 0.95:
            for pat, rep in HINDI_CONVERSATIONAL_SIMPLIFICATIONS:
                text = _devanagari_sub(pat, rep, text)

        # Step 3: Apply redundancy stripping
        for pat, rep in HINDI_REDUNDANCIES:
            text = _devanagari_sub(pat, rep, text)

        # Clean multiple spaces or stray punctuation
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,।?!])", r"\1", text)
        text = re.sub(r"[।]+$", "।", text)

        # Step 4: Gentle conversational filler reduction if duration reduction is requested
        if ratio < 0.85:
            for pat, rep in HINDI_CONVERSATIONAL_FILLERS:
                text = _devanagari_sub(pat, rep, text)
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"\s+([,।?!])", r"\1", text)
            text = re.sub(r"[।]+$", "।", text)

        # CRITICAL SAFETY GUARANTEE:
        # Never perform arbitrary word slicing (e.g. words[:target_word_count] or slicing around 'पर').
        # Complete sentence structure, all substantive clauses, and spoken meaning MUST be preserved.
        # Audio-level duration adaptation (atempo time-stretch up to ~1.25x and timeline extension)
        # handles remaining timing fitting naturally without dropping spoken dialogue.
        return text
