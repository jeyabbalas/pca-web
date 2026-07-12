---
name: verify
description: Build, launch, and drive the pca-web demo app to verify library or demo changes end-to-end in a real browser.
---

# Verifying pca-web changes

The runtime surface for this repo is the Vite demo app in `examples/demo/` — it exercises the library's public API (PCA, IncrementalPCA, worker proxies, model persistence) through real UI flows.

## Build / launch

- `npm run demo:dev` (repo root) — builds the library (`tsc -p tsconfig.build.json`) then starts Vite. Ready line: `➜ Local: http://localhost:517X/` (port auto-increments if 5173 is taken; grep the log for the actual URL).
- `npm run demo:build` — full typecheck (`tsc --noEmit`, strict) + production build. Run this for a fast static gate before driving the app.
- `npx biome check .` — repo-wide lint/format gate; `npx biome check --write examples/demo` to fix.

## Drive (browser automation)

- Panels top to bottom: 1·Data (dataset radios + dtype), 2·Model & execution (estimator/solver/nComponents/mode), 3·Run (Fit/Abort, progress, embedding scatter), 4·Results (scree, scoreSamples histogram, eigen-tiles for digits, reconstruction explorer + MSE curve), 5·Model persistence (JSON/IndexedDB).
- Click Fit and wait ~2-4 s; the "ran on … — N ms" green line confirms completion. Worker mode is the default path.
- **Gotcha:** panel heights change when controls appear (synthetic sliders, IPCA batchSize), moving the Fit button — click elements by `find` ref, not remembered coordinates. The viewport size can also drift between screenshots.
- Fits are seeded (`seed` input + fixed preset seeds), so refitting the same config must reproduce the identical scatter — a cheap determinism check.
- Console: the demo `console.error`s fit failures; filter for `Error|failed` and expect only errors you deliberately provoked.

## Flows worth driving after a change

- One Fit per affected dataset preset (digits, synthetic, swissroll, trefoil, circles, outliers) — check scatter shape/colors, scree, outlier note, MSE curve.
- Estimator toggle: IncrementalPCA needs integer nComponents ≤ min(batchSize, nFeatures); dataset radio changes clamp integer nComponents down to the preset's feature count.
- Abort mid-fit (worker mode), dtype float32 refit, and a model export → import round-trip when touching serialization.
