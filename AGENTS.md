# Modaic Repository Notes

## Batch Overview

`modaic.batch` is the batch inference subsystem for DSPy-based predictors.

- Public entrypoints live in `src/modaic-sdk/modaic/batch`.
- Top-level `modaic.batch.abatch()` now accepts grouped inputs as `list[tuple[dspy.Predict, list[dict]]]` and submits one provider batch request for the whole group.
- `modaic.Predict.abatch()` keeps the old single-predict shape and delegates to the grouped top-level API.
- Runtime batch artifacts are stored under `<modaic_cache>/batch`.
  Temp transport files live in `<modaic_cache>/batch/tmp`.
  Final persisted results live in `<modaic_cache>/batch/<batch_id>/<predict_index>.duckdb`.

## When Editing `modaic.batch`

- Keep the separation between DSPy-facing adapter logic in `batch.py` and provider-facing transport logic in `batch/clients/`.
- Preserve the invariant that grouped `abatch()` does one initial provider submit for all predict/input groups.
- `ChatAdapter` fallback is intentionally aggregated into one second retry batch for all failed rows, not one retry per predictor.
- For detailed architecture, storage, and design notes, read `src/modaic-sdk/modaic/batch/AGENTS.md`.
