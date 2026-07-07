# Lessons learned (build log)

One entry per lesson; update rather than duplicate; delete anything proven wrong.

- **sklearn 1.9 uses `svd_flip(u_based_decision=False)` for every PCA solver** — sign comes
  from the max-|.|-element of each row of Vt (first occurrence on ties), not from U. Older
  writeups describe u-based flipping; that changed in 1.5. Confirmed by reading installed
  1.9.0 source.
- **`transform` centers after projection** (`X @ Vt.T - mean @ Vt.T`), so it differs from
  `(X-mean) @ Vt.T` at roundoff level, and `fit_transform` (U*S) differs from
  `transform(fit(X))` at roundoff too. Fixtures capture both outputs separately.
- **IncrementalPCA keeps mean_/var_ in float64 even for float32 X** because
  `_incremental_mean_and_var` sums with float64 accumulators; components_ stay float32.
- **`np.sign(0.0) == 0`** — svd_flip would zero out an all-zero Vt row; keep in mind for
  rank-deficient edge fixtures.
- **numpy legacy gauss (polar) returns `f*x2` first and caches `f*x1`** — order matters for
  stream-exact RNG replication.
- **Never hand-annotate expected behavior a fixture generator can record.** The auto-solver
  expectations were hand-written flags, and one was wrong (80×500 sits exactly on sklearn's
  `max(shape) <= 500` → full boundary, not randomized). Recording `est._fit_svd_solver`
  makes the fixture self-truthing; the wrong guess cost a debugging round.
- **Comparisons beyond the numerical rank are meaningless everywhere, not just in PCA
  components**: QR's Q columns and LU's PL columns past a roundoff-zero pivot, and
  `inverse_transform` of out-of-span test data when rank < k, are all implementation-defined
  basis choices. LAPACK and a correct port legitimately differ there; parity tests must mask
  by rank (from the fixture's singular values / R diagonal) instead of loosening tolerances.
- **Unnormalized power iterations (`power_iteration_normalizer='none'`) are chaotically
  ill-conditioned by design**: trailing directions carry (σi/σ1)^(2·nIter+1) ≈ 1e-13 relative
  weight, and sklearn's own components move ~2e-4 abs under a 1-ulp input perturbation
  (measured). Parity there gets its own tolerance class, calibrated against that measurement —
  don't chase bit-agreement that sklearn itself can't reproduce.
- **noise_variance_ = (totalVar − Σev)/(min−k) is a cancellation** — its relative error is
  ev's relative error amplified by ~totalVar/noiseVar. Tolerances for it must be looser than
  for ev itself (atol ~1e-11 for roundoff-zeros, rtol ~1e-4 for randomized).
- **Biome's `noFocusedTests` flags any `.fit(` call** (Jasmine's focused `it` is named `fit`) —
  for an sklearn-style API the rule must be off.
