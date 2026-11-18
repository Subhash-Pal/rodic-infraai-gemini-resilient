# RODIC InfraAI — Hosted POC

AI-assisted infrastructure visual inspection POC for the
RODIC InfraAI Innovation Challenge 2026.

## Application

Streamlit-based multimodal infrastructure inspection application.

The application combines:

- Domain-trained computer vision
- CODEBRIM defect classification
- SDNET2018 crack classification
- Gemini multimodal visual evidence
- Evidence arbitration
- Defect regions (bounding boxes) on one or many images
- Structured JSON export per image
- Indicative IRC / MoRTH-referenced repair guidance
- Gemini key rotation with a computer-vision-only fallback
- Human engineering validation boundary

### What each feature is, and is not

| Feature | What it does | What it is not |
|---|---|---|
| Defect regions | Scores 224 px tiles with both models, groups hot tiles into boxes, tightens crack boxes with a classical dark-line search | Not a trained object detector; coarse, unvalidated; may miss or over-box |
| Multiple images | Up to 10 images per run, batch summary table, one combined JSON | Not a bulk pipeline for thousands of images |
| JSON export | `rodic.inspection.v1` per image: evidence by source, regions, Gemini status, arbitration, recommendations, review flags, limits; image hash only | Contains no image, no API key |
| Repair guidance | Severity class and indicative actions per detected class, naming IRC / MoRTH references | Not a design or repair specification; standards are named, not quoted; crack width is not measured, so crack severity is provisional |
| Key rotation | Spreads Gemini calls over several keys, skips a rate-limited key at once, falls back to CV-only evidence | Does not add quota if the keys share one Google Cloud project |

## Responsible AI Boundary

The system organizes and reconciles visual evidence.

It does **not**:

- determine structural safety
- provide engineering certification
- make final engineering dispositions
- treat model confidence as calibrated probability
- average confidence values across independent evidence sources

Qualified human inspection remains responsible for engineering
significance and final disposition.

## Credential Configuration

No Gemini API credential is stored in this repository.

Configure these Streamlit secrets (see `.streamlit/secrets.toml.example`). Only
`GEMINI_API_KEY` is required; the others are optional extra keys used for rotation:

`GEMINI_API_KEY`, `GEMINI_API_KEY1`, `GEMINI_API_KEY2`, `GEMINI_API_KEY3`,
`GEMINI_API_KEY4`, `GEMINI_API_KEY5`

Free-tier Gemini limits apply per Google Cloud project and per model. Keys created in the
same project share one quota, so rotation only helps with keys from different projects.
Rotation settings (cooldowns, key names) are in `app/gemini_config.json` under `key_rotation`.

When every key and model is unavailable, the app does not stop: it shows computer-vision
evidence only, marks Gemini as `unavailable` in the table and in the JSON, and the
**Retry Gemini** button tries again (failed calls are not cached).

## Deployment

This repository is intended for Streamlit Community Cloud deployment
(main file: `app/app.py`; Python 3.11 or newer; CPU only).

See `docs/DEPLOYMENT.md` for the merge, secrets and smoke-test checklist.

## Tests and CI

```
pip install -r app/requirements.txt pytest
python -m pytest -q
```

The suite covers key rotation (fake clients, no network), region grouping, JSON export,
the recommender, the committed checkpoints on CPU (skipped when torch is missing) and a
headless start-up check of the app. GitHub Actions runs the same checks on every push and
pull request (`.github/workflows/ci.yml`).
