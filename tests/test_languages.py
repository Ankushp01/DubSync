"""Unit and integration tests for Language Capability Registry and API endpoints."""
import unittest
from fastapi.testclient import TestClient

from api.auth.security import create_access_token
from api.main import app
from dubsync.languages import (
    LANGUAGES,
    PairStatus,
    evaluate_language_pair,
    get_capabilities_payload,
    get_language,
    list_languages,
    resolve_chatterbox_tag,
    resolve_indictrans_tag,
    resolve_whisper_code,
    validate_language_pair,
)


class LanguageCapabilityRegistryTests(unittest.TestCase):
    def test_language_code_mapping(self):
        """Verify language code mappings to Whisper, IndicTrans2, and Chatterbox."""
        # English
        self.assertEqual(resolve_whisper_code("en"), "en")
        self.assertEqual(resolve_indictrans_tag("en"), "eng_Latn")
        self.assertEqual(resolve_chatterbox_tag("en"), "en")

        # Hindi
        self.assertEqual(resolve_whisper_code("hi"), "hi")
        self.assertEqual(resolve_indictrans_tag("hi"), "hin_Deva")
        self.assertEqual(resolve_chatterbox_tag("hi"), "hi")

        # Marathi: IndicTrans2 tag exists, but Chatterbox tag is None
        self.assertEqual(resolve_whisper_code("mr"), "mr")
        self.assertEqual(resolve_indictrans_tag("mr"), "mar_Deva")
        self.assertIsNone(resolve_chatterbox_tag("mr"))

        # Spanish: Chatterbox tag exists, but IndicTrans2 tag is None (non-Indic)
        self.assertEqual(resolve_whisper_code("es"), "es")
        self.assertIsNone(resolve_indictrans_tag("es"))
        self.assertEqual(resolve_chatterbox_tag("es"), "es")

    def test_registry_consistency(self):
        """Verify registry entries contain valid codes, names, and flags."""
        languages = list_languages()
        self.assertGreater(len(languages), 5)

        seen_codes = set()
        for lang in languages:
            self.assertNotIn(lang.code, seen_codes, f"Duplicate language code: {lang.code}")
            seen_codes.add(lang.code)
            self.assertTrue(lang.code.islower())
            self.assertTrue(len(lang.name) > 0)
            self.assertTrue(len(lang.native_name) > 0)
            self.assertTrue(len(lang.whisper_code) > 0)
            self.assertIsInstance(lang.asr_supported, bool)
            self.assertIsInstance(lang.tts_supported, bool)
            self.assertIsInstance(lang.is_indic, bool)

    def test_supported_language_pair_validation(self):
        """Verify that English-to-Hindi and Auto-to-Hindi are supported and verified."""
        # en -> hi
        pair_en_hi = evaluate_language_pair("en", "hi")
        self.assertTrue(pair_en_hi.usable)
        self.assertEqual(pair_en_hi.status, PairStatus.SUPPORTED)
        self.assertTrue(pair_en_hi.verified_locally)
        self.assertEqual(pair_en_hi.translation_model, "ai4bharat/indictrans2-en-indic-dist-200M")
        self.assertEqual(pair_en_hi.tts_model, "ResembleAI/chatterbox-v3")

        valid, err = validate_language_pair("en", "hi")
        self.assertTrue(valid)
        self.assertIsNone(err)

        # auto -> hi
        pair_auto_hi = evaluate_language_pair("auto", "hi")
        self.assertTrue(pair_auto_hi.usable)
        self.assertEqual(pair_auto_hi.status, PairStatus.SUPPORTED)
        self.assertTrue(pair_auto_hi.verified_locally)

        valid_auto, err_auto = validate_language_pair("auto", "hi")
        self.assertTrue(valid_auto)
        self.assertIsNone(err_auto)

    def test_unsupported_language_pairs_prevent_false_claims(self):
        """Verify that unsupported or incomplete pairs are explicitly rejected with reasons."""
        # en -> mr: translation works, but Chatterbox TTS missing voice
        pair_en_mr = evaluate_language_pair("en", "mr")
        self.assertFalse(pair_en_mr.usable)
        self.assertEqual(pair_en_mr.status, PairStatus.THEORETICAL)
        self.assertIn("Chatterbox Multilingual V3 lacks Marathi", pair_en_mr.reason)

        valid, err = validate_language_pair("en", "mr")
        self.assertFalse(valid)
        self.assertIn("Chatterbox Multilingual V3 lacks Marathi", err)

        # en -> es: Chatterbox works, but IndicTrans2 lacks Spanish translation
        pair_en_es = evaluate_language_pair("en", "es")
        self.assertFalse(pair_en_es.usable)
        self.assertEqual(pair_en_es.status, PairStatus.THEORETICAL)
        self.assertIn("IndicTrans2", pair_en_es.reason)
        self.assertIn("Spanish", pair_en_es.reason)

        # Non-English source: es -> hi
        pair_es_hi = evaluate_language_pair("es", "hi")
        self.assertFalse(pair_es_hi.usable)
        self.assertEqual(pair_es_hi.status, PairStatus.UNSUPPORTED)
        self.assertIn("Non-English source", pair_es_hi.reason)

        # Identity pair: en -> en
        pair_en_en = evaluate_language_pair("en", "en")
        self.assertFalse(pair_en_en.usable)
        self.assertIn("identical", pair_en_en.reason)

        # auto -> mr: only Hindi is supported for auto source in current pipeline
        pair_auto_mr = evaluate_language_pair("auto", "mr")
        self.assertFalse(pair_auto_mr.usable)

    def test_automatic_vs_manual_source_resolution(self):
        """Verify that 'auto' resolves to None for Whisper VAD detection, whereas 'en' locks code."""
        self.assertIsNone(resolve_whisper_code("auto"))
        self.assertEqual(resolve_whisper_code("en"), "en")
        self.assertEqual(resolve_whisper_code("hi"), "hi")


class LanguageApiEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_get_languages_capabilities_endpoint(self):
        """Verify GET /api/v1/languages/capabilities response structure."""
        response = self.client.get("/api/v1/languages/capabilities")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("active_pipeline", data)
        self.assertEqual(data["active_pipeline"]["asr_model"], "Systran/faster-whisper-small")
        self.assertEqual(data["active_pipeline"]["translation_model"], "ai4bharat/indictrans2-en-indic-dist-200M")
        self.assertEqual(data["active_pipeline"]["tts_model"], "ResembleAI/chatterbox-v3")

        self.assertIn("languages", data)
        self.assertIn("supported_pairs", data)
        self.assertIn("supported_source_languages", data)
        self.assertIn("supported_target_languages_for_source", data)
        self.assertIn("verified_locally", data)

        # Confirm en -> hi is in supported_pairs
        supported_pairs = data["supported_pairs"]
        en_hi_pairs = [p for p in supported_pairs if p["source"] == "en" and p["target"] == "hi"]
        self.assertEqual(len(en_hi_pairs), 1)
        self.assertTrue(en_hi_pairs[0]["usable"])

        # Confirm target mapping
        self.assertIn("hi", data["supported_target_languages_for_source"]["en"])
        self.assertIn("hi", data["supported_target_languages_for_source"]["auto"])

    def test_get_language_pairs_endpoint(self):
        """Verify GET /api/v1/languages/pairs endpoint."""
        response = self.client.get("/api/v1/languages/pairs")
        self.assertEqual(response.status_code, 200)
        pairs = response.json()
        self.assertIsInstance(pairs, list)
        self.assertTrue(all(p["usable"] is True for p in pairs))

    def test_job_creation_rejects_unsupported_language_pair(self):
        """Verify POST /api/v1/jobs rejects unsupported language pair with HTTP 422."""
        fake_video = b"fake video bytes"
        files = {
            "video": ("test.mp4", fake_video, "video/mp4"),
        }
        data = {
            "source_language": "en",
            "target_language": "mr",  # Unsupported TTS
        }
        auth_headers = {"Authorization": f"Bearer {create_access_token('usr_lang_test', 'lang@example.com')}"}
        response = self.client.post("/api/v1/jobs", files=files, data=data, headers=auth_headers)
        self.assertEqual(response.status_code, 422)
        detail = response.json().get("detail", "")
        self.assertIn("Unsupported language pair", detail)
        self.assertIn("Chatterbox Multilingual V3 lacks Marathi speech synthesis", detail)

    def test_job_creation_rejects_spanish_translation_without_nllb(self):
        """Verify POST /api/v1/jobs rejects target='es' with HTTP 422."""
        fake_video = b"fake video bytes"
        files = {
            "video": ("test.mp4", fake_video, "video/mp4"),
        }
        data = {
            "source_language": "en",
            "target_language": "es",
        }
        auth_headers = {"Authorization": f"Bearer {create_access_token('usr_lang_test', 'lang@example.com')}"}
        response = self.client.post("/api/v1/jobs", files=files, data=data, headers=auth_headers)
        self.assertEqual(response.status_code, 422)
        detail = response.json().get("detail", "")
        self.assertIn("Unsupported language pair", detail)
        self.assertIn("IndicTrans2", detail)


if __name__ == "__main__":
    unittest.main()
