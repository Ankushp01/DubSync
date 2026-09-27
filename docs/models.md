# Model Dependencies

This document records the external model repositories used during
development/testing of DubSync V1 and later lip-sync integration.

## Translation

### Primary — IndicTrans2

Repository commit used:
`4f08e39cc6bf13cd62e2445dc725f22bff1a9219`

Role:
Primary translation backend for DubSync V1.

### Fallback — NLLB

Role:
Fallback translation backend when IndicTrans2 cannot be used or fails.

---

## TTS — Chatterbox Multilingual

Model:
Chatterbox Multilingual

Purpose:
Target-language speech generation with speaker/reference-audio conditioning.

Notes:
The Chatterbox implementation is integrated into the DubSync codebase.
Model weights are not stored in this repository.

---

## Lip Synchronization — Wav2Lip-HD

Repository:
Wav2Lip-HD

Repository commit used:
`e9716e14dc6d74c0957be7a9f3210e29eafea73f`

Purpose:
Lip synchronization of the original video with the generated dubbed audio.

Status:
Not part of the working V1 pipeline. Kept as the subsequent lip-sync stage.

---

## MuseTalk

Repository:
MuseTalk

Repository commit used:
`0a89dec45a0192b824e3cf4daf96c239440c5ed8`

Purpose:
Alternative/advanced lip-sync backend evaluated for later integration.

Status:
Not part of the working V1 pipeline.

---

## Important

Model weights and external repositories are intentionally not committed
to the main DubSync repository.

The repository commits above identify the exact external source versions
used during development.