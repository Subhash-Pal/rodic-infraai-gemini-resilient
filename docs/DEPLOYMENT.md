# Deployment checklist (Streamlit Community Cloud)

The feature work lives on branch `feature/batch-json-irc-keyring`. Streamlit redeploys when the
tracked branch changes, so nothing here is live until you merge into the branch the app tracks.

## Before merging
1. CI is green on the pull request (lint, secret check, tests, real checkpoints on CPU).
2. In Streamlit Cloud > app > Settings > Secrets, set `GEMINI_API_KEY` and any of
   `GEMINI_API_KEY1` to `GEMINI_API_KEY5` (format: `.streamlit/secrets.toml.example`).
   Use keys from different Google Cloud projects, otherwise rotation gains nothing.
3. Optional: deploy the branch as a second app (Streamlit Cloud > New app > branch) and test it
   before touching the live app.

## Smoke test after deploying (about 5 minutes)
1. Open the app; sidebar shows `Gemini keys configured: N` and `App version 1.1.0`.
2. Upload two images (one with a visible crack, one clean). Expect a batch summary with 2 rows.
3. Open the crack image: boxes on the Annotated tab, region list, repair guidance with the
   disclaimer, review flags, and a JSON download.
4. Download the batch JSON and open it: `schema_version` is `rodic.inspection.v1`, no keys, no
   image data.
5. Offline fallback: temporarily remove all keys (or untick "Use Gemini"); the app must still return
   CV evidence and show Gemini as unavailable or skipped.

## Rollback
Revert the merge commit (or redeploy the previous commit `0c2d89b`). No data is stored by the app.

## Limits to state when presenting
Tested on synthetic images and unit tests. Regions are coarse and unvalidated. Severity text is
indicative and not a design or safety judgement. No accuracy figures are claimed.
