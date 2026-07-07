# pca-web build plan / status

Decisions locked with the user (2026-07-07):
- **camelCase API** (`nComponents`, `svdSolver`, `explainedVarianceRatio`, …), mapping table in README.
- **MIT license** (LICENSE already present).
- Data in: `number[][]` or typed array + shape; out: typed-array-backed `Matrix` with `.toArray()`. Compute dtype follows input dtype.
- Core `PCA`/`IncrementalPCA` synchronous CPU; `pca-web/webgpu` exports async GPU-accelerated path (feature-detected, injectable device, CPU fallback).
- Dev stack: TypeScript 6 (tsc build, NodeNext ESM), Vitest 4, Biome 2, Playwright, esbuild (test-page bundling), Python venv (sklearn 1.9.0, numpy 2.5.1) for fixtures.

## Phases

- [x] 0. Scaffold: configs, deps, venv, plan docs
- [x] 1. Read sklearn 1.9.0 source → docs/SPEC_NOTES.md (the porting spec)
- [x] 2. Numerics core (src/numeric/): gemm/syrk, Golub–Reinsch SVD, symmetric eigh,
      Householder QR, partial-pivot LU (permute_l), Cholesky/slogdet/inv, Lanczos
      truncated SVD (arpack equivalent), numpy MT19937 RandomState replica, svd_flip,
      lgamma. Unit tests vs numpy fixtures.
- [x] 3. Fixture oracle: python/generate_fixtures.py → fixtures/ (JSON + .bin),
      PCA + IncrementalPCA cases + numerics + RNG stream fixtures. Committed.
- [x] 4. PCA class (all solvers/modes/methods/attributes). Parity 80/80.
- [x] 5. IncrementalPCA. Parity 16/16 incl. per-step partial_fit state.
- [x] 6. Parity suite green — 126 tests (parity + API behavior + RNG + numerics);
      per-class tolerances documented in tests, observed maxima printed per run.
- [x] 7. WebGPU backend (subpath export) + CPU fallback equivalence.
      fma-exact products + integer-binned accumulation (Metal fast-math
      destroys classic EFTs — probed and documented in LESSONS.md).
- [x] 8. Playwright browser/GPU harness; ran for real (apple/metal-3,
      headless): 41/41 GPU cases green, GEMM self-check 5.3e-13.
- [x] 9. Benchmarks CPU vs GPU: 2.7×–7.7× at 50k×200…5k×2000.
- [x] 10. README (API mapping, solver guide, measured tolerances, fixture
      regen) + packaging verified via npm pack + clean-room import smoke.
- [x] 11. Final verifier-subagent audit (2 independent fresh-context auditors) + status report.

## Layout

```
src/
  index.ts  types.ts  matrix.ts  validation.ts  pca.ts  incremental-pca.ts
  numeric/ (blas, svd, eigh, qr, lu, cholesky, lanczos, randomized, rng, svdflip, special, stats)
  webgpu/ (index, device, kernels/wgsl, ops, gpu-pca)
python/generate_fixtures.py  fixtures/  tests/  tests/browser/  bench/
```
