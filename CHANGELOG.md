# Changelog

## 1.1.0 (feature branch `feature/batch-json-irc-keyring`)

Added
- Gemini key rotation (`GEMINI_API_KEY` to `GEMINI_API_KEY5`) with per-key, per-model cooldowns,
  round-robin load spreading and a computer-vision-only fallback with a Retry button.
- Defect regions (bounding boxes) from tile scores of both models, crack boxes tightened with an
  OpenCV dark-line search; image-wide detections reported as not localized.
- Multiple image upload (up to 10) with a batch summary table.
- Structured JSON export per image and for the batch (`rodic.inspection.v1`).
- Indicative IRC / MoRTH-referenced severity and repair guidance (ported from Rodic-infra-Colab v2).
- Tests, GitHub Actions CI, `.streamlit/secrets.toml.example`, `docs/DEPLOYMENT.md`.

Changed
- Model loading no longer downloads ImageNet weights before applying the checkpoint
  (`pretrained=False`): faster cold start and no network need for the backbone.
- Evidence arbitration now counts an SDNET2018 crack as computer-vision evidence. The earlier app
  ignored SDNET there, which contradicted its own Computer-Vision Evidence panel.
- `analyze_image_with_gemini` gained optional `api_key`, `models_override` and
  `fail_fast_on_quota` arguments; defaults keep the original single-key behaviour.

Known limits
- Regions are tile-aligned and unvalidated; no trained detector.
- Crack width is not measured; crack severity is provisional.
- Tested on synthetic images and unit tests only; no real-image validation yet.
