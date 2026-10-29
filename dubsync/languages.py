"""Central Language Capability Registry for DubSync.

Defines all known languages, model-specific code mappings, component capabilities
(ASR, Translation, TTS), and explicit language-pair support.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class PairStatus(str, Enum):
    SUPPORTED = "supported"        # Fully supported and usable end-to-end
    UNSUPPORTED = "unsupported"    # Not supported by active pipeline
    THEORETICAL = "theoretical"    # Theoretical model capability, not wired in pipeline


@dataclass(frozen=True)
class LanguageInfo:
    """Metadata and model mappings for a specific language."""
    code: str                     # Standard ISO 639-1 code (e.g., 'en', 'hi', 'mr')
    name: str                     # English display name (e.g., 'English', 'Hindi')
    native_name: str              # Native script display name (e.g., 'हिन्दी')
    whisper_code: str             # Code expected by Faster-Whisper (e.g., 'en', 'hi')
    indictrans_code: Optional[str] # BCP-47 tag for IndicTrans2 (e.g., 'eng_Latn', 'hin_Deva')
    nllb_code: Optional[str]      # BCP-47 tag for NLLB-200 (e.g., 'eng_Latn', 'hin_Deva')
    chatterbox_code: Optional[str]# Language ID for Chatterbox Multilingual V3 (e.g., 'en', 'hi')
    asr_supported: bool           # Whether Faster-Whisper can transcribe this language
    tts_supported: bool           # Whether active Chatterbox TTS can synthesize this language
    is_indic: bool = False        # Whether this is an Indian language


@dataclass(frozen=True)
class LanguagePair:
    """Explicit capability record for a source-to-target language pair."""
    source_language: str          # Source language code or 'auto'
    target_language: str          # Target language code
    usable: bool                  # Usable end-to-end in active pipeline
    status: PairStatus
    verified_locally: bool        # Verified locally on this hardware/installation
    translation_model: Optional[str] = None
    tts_model: Optional[str] = None
    reason: Optional[str] = None  # Human-readable explanation if unusable or theoretical


# ==============================================================================
# Language Registry Definition
# ==============================================================================

# Chatterbox Multilingual V3 supported languages:
# ar, da, de, el, en, es, fi, fr, he, hi, it, ja, ko, ms, nl, no, pl, pt, ru, sv, sw, tr, zh
CHATTERBOX_SUPPORTED_CODES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
    "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
    "sw", "tr", "zh"
}

# IndicTrans2 (en-indic) targets (22 scheduled Indian languages):
INDICTRANS2_TARGET_MAP = {
    "hi": "hin_Deva",
    "mr": "mar_Deva",
    "bn": "ben_Beng",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "pa": "pan_Guru",
    "or": "ory_Orya",
    "as": "asm_Beng",
    "ur": "urd_Arab",
    "sa": "san_Deva",
    "ne": "npi_Deva",
    "sd": "snd_Deva",
    "ks": "kas_Deva",
    "kok": "kok_Deva",
    "mai": "mai_Deva",
    "brx": "brx_Deva",
    "doi": "doi_Deva",
    "mni": "mni_Beng",
    "sat": "sat_Olck",
}

LANGUAGES: Dict[str, LanguageInfo] = {
    "en": LanguageInfo(
        code="en",
        name="English",
        native_name="English",
        whisper_code="en",
        indictrans_code="eng_Latn",
        nllb_code="eng_Latn",
        chatterbox_code="en",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
    "hi": LanguageInfo(
        code="hi",
        name="Hindi",
        native_name="हिन्दी",
        whisper_code="hi",
        indictrans_code="hin_Deva",
        nllb_code="hin_Deva",
        chatterbox_code="hi",
        asr_supported=True,
        tts_supported=True,
        is_indic=True,
    ),
    "mr": LanguageInfo(
        code="mr",
        name="Marathi",
        native_name="मराठी",
        whisper_code="mr",
        indictrans_code="mar_Deva",
        nllb_code="mar_Deva",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "bn": LanguageInfo(
        code="bn",
        name="Bengali",
        native_name="বাংলা",
        whisper_code="bn",
        indictrans_code="ben_Beng",
        nllb_code="ben_Beng",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "ta": LanguageInfo(
        code="ta",
        name="Tamil",
        native_name="தமிழ்",
        whisper_code="ta",
        indictrans_code="tam_Taml",
        nllb_code="tam_Taml",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "te": LanguageInfo(
        code="te",
        name="Telugu",
        native_name="తెలుగు",
        whisper_code="te",
        indictrans_code="tel_Telu",
        nllb_code="tel_Telu",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "gu": LanguageInfo(
        code="gu",
        name="Gujarati",
        native_name="ગુજરાતી",
        whisper_code="gu",
        indictrans_code="guj_Gujr",
        nllb_code="guj_Gujr",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "kn": LanguageInfo(
        code="kn",
        name="Kannada",
        native_name="ಕನ್ನಡ",
        whisper_code="kn",
        indictrans_code="kan_Knda",
        nllb_code="kan_Knda",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "ml": LanguageInfo(
        code="ml",
        name="Malayalam",
        native_name="മലയാളം",
        whisper_code="ml",
        indictrans_code="mal_Mlym",
        nllb_code="mal_Mlym",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "pa": LanguageInfo(
        code="pa",
        name="Punjabi",
        native_name="ਪੰਜਾਬੀ",
        whisper_code="pa",
        indictrans_code="pan_Guru",
        nllb_code="pan_Guru",
        chatterbox_code=None,  # Not supported by Chatterbox V3
        asr_supported=True,
        tts_supported=False,
        is_indic=True,
    ),
    "es": LanguageInfo(
        code="es",
        name="Spanish",
        native_name="Español",
        whisper_code="es",
        indictrans_code=None,  # IndicTrans2 only supports Indic languages
        nllb_code="spa_Latn",
        chatterbox_code="es",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
    "fr": LanguageInfo(
        code="fr",
        name="French",
        native_name="Français",
        whisper_code="fr",
        indictrans_code=None,
        nllb_code="fra_Latn",
        chatterbox_code="fr",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
    "de": LanguageInfo(
        code="de",
        name="German",
        native_name="Deutsch",
        whisper_code="de",
        indictrans_code=None,
        nllb_code="deu_Latn",
        chatterbox_code="de",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
    "ja": LanguageInfo(
        code="ja",
        name="Japanese",
        native_name="日本語",
        whisper_code="ja",
        indictrans_code=None,
        nllb_code="jpn_Jpan",
        chatterbox_code="ja",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
    "ko": LanguageInfo(
        code="ko",
        name="Korean",
        native_name="한국어",
        whisper_code="ko",
        indictrans_code=None,
        nllb_code="kor_Hang",
        chatterbox_code="ko",
        asr_supported=True,
        tts_supported=True,
        is_indic=False,
    ),
}


# ==============================================================================
# Capability Resolution Logic
# ==============================================================================

INDICTRANS2_MODEL_ID = "ai4bharat/indictrans2-en-indic-dist-200M"
CHATTERBOX_MODEL_ID = "ResembleAI/chatterbox-v3"
WHISPER_MODEL_ID = "Systran/faster-whisper-small"


def get_language(code: str) -> Optional[LanguageInfo]:
    """Retrieve language metadata by ISO code."""
    if not code:
        return None
    return LANGUAGES.get(code.lower().strip())


def list_languages() -> List[LanguageInfo]:
    """List all registered languages."""
    return list(LANGUAGES.values())


def evaluate_language_pair(source: str, target: str) -> LanguagePair:
    """Evaluate whether a language pair is currently usable in the configured pipeline.

    A pair is usable ONLY if:
    1. ASR can transcribe the source language (or auto-detect it).
    2. The active translation route can translate source -> target.
    3. The active TTS model can synthesize speech in the target language.
    """
    src_norm = source.lower().strip()
    tgt_norm = target.lower().strip()

    # Special handling for auto source
    if src_norm == "auto":
        # Target must be Hindi in currently configured pipeline because
        # IndicTrans2 requires English and Chatterbox only does Hindi among Indic languages.
        if tgt_norm == "hi":
            return LanguagePair(
                source_language="auto",
                target_language="hi",
                usable=True,
                status=PairStatus.SUPPORTED,
                verified_locally=True,
                translation_model=INDICTRANS2_MODEL_ID,
                tts_model=CHATTERBOX_MODEL_ID,
                reason="Automatic source detection with English dialogue translated to Hindi.",
            )
        else:
            tgt_lang = get_language(tgt_norm)
            tgt_name = tgt_lang.name if tgt_lang else tgt_norm
            return LanguagePair(
                source_language="auto",
                target_language=tgt_norm,
                usable=False,
                status=PairStatus.UNSUPPORTED,
                verified_locally=False,
                reason=f"Auto-detection is only configured for Hindi target output. Target '{tgt_name}' is not usable in current pipeline.",
            )

    src_lang = get_language(src_norm)
    tgt_lang = get_language(tgt_norm)

    if not src_lang:
        return LanguagePair(
            source_language=src_norm,
            target_language=tgt_norm,
            usable=False,
            status=PairStatus.UNSUPPORTED,
            verified_locally=False,
            reason=f"Unknown source language code '{src_norm}'.",
        )

    if not tgt_lang:
        return LanguagePair(
            source_language=src_norm,
            target_language=tgt_norm,
            usable=False,
            status=PairStatus.UNSUPPORTED,
            verified_locally=False,
            reason=f"Unknown target language code '{tgt_norm}'.",
        )

    # Currently verified pipeline: en -> hi
    if src_norm == "en" and tgt_norm == "hi":
        return LanguagePair(
            source_language="en",
            target_language="hi",
            usable=True,
            status=PairStatus.SUPPORTED,
            verified_locally=True,
            translation_model=INDICTRANS2_MODEL_ID,
            tts_model=CHATTERBOX_MODEL_ID,
            reason="Verified English-to-Hindi neural dubbing pipeline.",
        )

    # Case: English to other Indic languages (mr, bn, ta, te, etc.)
    # IndicTrans2 translates en -> indic, BUT Chatterbox V3 lacks Indic voice synthesis for these.
    if src_norm == "en" and tgt_lang.is_indic and tgt_norm != "hi":
        return LanguagePair(
            source_language="en",
            target_language=tgt_norm,
            usable=False,
            status=PairStatus.THEORETICAL,
            verified_locally=False,
            translation_model=INDICTRANS2_MODEL_ID,
            tts_model=None,
            reason=f"IndicTrans2 can translate English to {tgt_lang.name}, but Chatterbox Multilingual V3 lacks {tgt_lang.name} speech synthesis.",
        )

    # Case: English to European/East Asian languages (es, fr, de, ja, ko)
    # Chatterbox V3 can speak these, BUT IndicTrans2 only translates Indian languages!
    # NLLB-200 could theoretically translate these, but is not currently wired as the active pipeline stage.
    if src_norm == "en" and not tgt_lang.is_indic and tgt_norm != "en":
        return LanguagePair(
            source_language="en",
            target_language=tgt_norm,
            usable=False,
            status=PairStatus.THEORETICAL,
            verified_locally=False,
            translation_model="facebook/nllb-200-distilled-600M (inactive)",
            tts_model=CHATTERBOX_MODEL_ID,
            reason=f"Chatterbox V3 supports {tgt_lang.name} voice synthesis, but active translation model (IndicTrans2) only translates into Indian languages. Requires NLLB routing.",
        )

    # Non-English source language
    if src_norm != "en":
        return LanguagePair(
            source_language=src_norm,
            target_language=tgt_norm,
            usable=False,
            status=PairStatus.UNSUPPORTED,
            verified_locally=False,
            reason=f"Non-English source '{src_lang.name}' is not supported by the active one-way English-to-Indic translation model ({INDICTRANS2_MODEL_ID}).",
        )

    # Same language (en -> en)
    if src_norm == "en" and tgt_norm == "en":
        return LanguagePair(
            source_language="en",
            target_language="en",
            usable=False,
            status=PairStatus.UNSUPPORTED,
            verified_locally=False,
            reason="Source and target languages are identical (English). Translation is not applicable.",
        )

    return LanguagePair(
        source_language=src_norm,
        target_language=tgt_norm,
        usable=False,
        status=PairStatus.UNSUPPORTED,
        verified_locally=False,
        reason=f"Language pair '{src_norm}' -> '{tgt_norm}' is not supported by the pipeline.",
    )


def validate_language_pair(source: str, target: str) -> Tuple[bool, Optional[str]]:
    """Validate whether requested language pair can be submitted for a dubbing job.

    Returns (is_valid, error_message).
    """
    pair = evaluate_language_pair(source, target)
    if not pair.usable:
        return False, pair.reason or f"Language pair '{source}' -> '{target}' is not supported."
    return True, None


def resolve_indictrans_tag(language_code: str) -> Optional[str]:
    """Map language code to IndicTrans2 BCP-47 language tag."""
    lang = get_language(language_code)
    return lang.indictrans_code if lang else None


def resolve_chatterbox_tag(language_code: str) -> Optional[str]:
    """Map language code to Chatterbox language identifier."""
    lang = get_language(language_code)
    return lang.chatterbox_code if lang else None


def resolve_whisper_code(language_code: str) -> Optional[str]:
    """Map language code to Faster-Whisper language code."""
    if language_code.lower() == "auto":
        return None
    lang = get_language(language_code)
    return lang.whisper_code if lang else None


def get_capabilities_payload() -> Dict[str, Any]:
    """Generate structured response describing all language capabilities."""
    languages_out = []
    for l in LANGUAGES.values():
        languages_out.append({
            "code": l.code,
            "name": l.name,
            "native_name": l.native_name,
            "asr_supported": l.asr_supported,
            "tts_supported": l.tts_supported,
            "is_indic": l.is_indic,
            "whisper_code": l.whisper_code,
            "indictrans_tag": l.indictrans_code,
            "chatterbox_tag": l.chatterbox_code,
        })

    # Evaluate common pairs
    pairs_out = []
    # Auto source
    for tgt in LANGUAGES.keys():
        pair = evaluate_language_pair("auto", tgt)
        pairs_out.append({
            "source": pair.source_language,
            "target": pair.target_language,
            "usable": pair.usable,
            "status": pair.status.value,
            "verified_locally": pair.verified_locally,
            "translation_model": pair.translation_model,
            "tts_model": pair.tts_model,
            "reason": pair.reason,
        })
    # English source
    for tgt in LANGUAGES.keys():
        pair = evaluate_language_pair("en", tgt)
        pairs_out.append({
            "source": pair.source_language,
            "target": pair.target_language,
            "usable": pair.usable,
            "status": pair.status.value,
            "verified_locally": pair.verified_locally,
            "translation_model": pair.translation_model,
            "tts_model": pair.tts_model,
            "reason": pair.reason,
        })

    usable_pairs = [p for p in pairs_out if p["usable"]]

    return {
        "active_pipeline": {
            "asr_model": WHISPER_MODEL_ID,
            "translation_model": INDICTRANS2_MODEL_ID,
            "tts_model": CHATTERBOX_MODEL_ID,
            "lipsync_model": "Wav2Lip + S3FD (Vanilla)",
        },
        "languages": languages_out,
        "supported_pairs": usable_pairs,
        "all_evaluated_pairs": pairs_out,
        "supported_source_languages": [
            {"code": "auto", "name": "Auto Detect (Recommended)"},
            {"code": "en", "name": "English"},
        ],
        "supported_target_languages_for_source": {
            "auto": ["hi"],
            "en": ["hi"],
        },
        "verified_locally": ["en->hi", "auto->hi"],
        "notes": (
            "Currently verified production pipeline supports English to Hindi dubbing. "
            "IndicTrans2 supports additional Indian target languages (mr, bn, ta, te, etc.), "
            "but Chatterbox Multilingual V3 lacks voice models for them. "
            "Chatterbox supports Spanish, French, and German, but IndicTrans2 only translates Indian languages."
        ),
    }
