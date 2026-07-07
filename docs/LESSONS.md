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
