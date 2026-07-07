# sklearn 1.9.0 porting spec (extracted from installed source)

Ground truth: `.venv/lib/python3.14/site-packages/sklearn` — sklearn 1.9.0, numpy 2.5.1, scipy 1.18.0.
This file records the exact semantics `pca-web` must reproduce. Line refs are to installed source.

## PCA._fit dispatch (_pca.py)

- Validation: dtype float64 kept, float32 kept, anything else → float64. 2D required. NaN/inf rejected.
- `n_components` resolution when `None`: `min(n_samples, n_features)` for all solvers except arpack, which uses `min(shape) - 1`.
- `auto` solver selection (dense):
  1. `n_features <= 1000 and n_samples >= 10 * n_features` → `covariance_eigh`
  2. `max(shape) <= 500 or n_components == 'mle'` → `full`
  3. `1 <= n_components < 0.8 * min(shape)` → `randomized`
  4. else (incl. float n_components in (0,1)) → `full`
- full/covariance_eigh → `_fit_full`; arpack/randomized → `_fit_truncated`.

## PCA._fit_full

- `'mle'` requires `n_samples >= n_features`, else ValueError.
- int n_components must satisfy `0 <= n <= min(shape)` (0 allowed).
- `mean_ = X.mean(axis=0)` (numpy mean, input dtype accumulation).
- full: center (copy if `copy=True` else in place), `scipy.linalg.svd(X_centered, full_matrices=False)` (gesdd), `explained_variance_ = S**2/(n-1)`.
- covariance_eigh: `C = X.T @ X; C -= n * mean_ outer mean_; C /= (n-1)`; `eigh` ascending → flip both; clip eigenvals `< 0` to `0`; `S = sqrt(eigvals*(n-1))`; `Vt = eigenvecs.T`; `U = None`.
- **Sign convention: `svd_flip(U, Vt, u_based_decision=False)` for ALL solvers in 1.9** — per row of Vt, find argmax(|row|) (first occurrence), multiply row of Vt and column of U by `sign(that element)`. `np.sign(0.0) == 0` (zero row would be zeroed — edge).
- `total_var = sum(explained_variance_)` over ALL min(shape) values (before truncation).
- `'mle'` → `_infer_dimension(explained_variance_, n_samples)`.
- float in (0,1): `n = searchsorted(cumsum(ratio), n_components, side='right') + 1`.
- `noise_variance_ = mean(explained_variance_[n:])` if `n < min(shape)` else `0.0`.
- Attributes are truncated copies; `singular_values_` copy of `S[:n]`.
- `fit_transform`: returns `U[:, :n] * S[:n]`, or if whiten `U[:, :n] * sqrt(n_samples - 1)`. For covariance_eigh (U is None) falls back to `_transform(X, x_is_centered)`.

## PCA._fit_truncated (arpack | randomized)

- n_components must be int; `1 <= n <= min(shape)`; arpack strictly `n < min(shape)`.
- `random_state = check_random_state(...)`: int → `np.random.RandomState(seed)` (MT19937).
- `mean_ = X.mean(axis=0)`; center (copy if copy else in place).
- arpack: `v0 = random_state.uniform(-1, 1, min(shape))`; `svds(X_centered, k, tol, v0)`; reverse order: `S = S[::-1]`; `svd_flip(U[:, ::-1], Vt[::-1], u_based_decision=False)`. Converged results = exact truncated SVD (tol=0 → machine precision).
- randomized: `_randomized_svd(X_centered, n_components, n_oversamples, n_iter=iterated_power, power_iteration_normalizer, flip_sign=False, random_state)`, then `svd_flip(U, Vt, u_based_decision=False)`.
- `explained_variance_ = S**2/(n-1)`; `total_var = sum(X_centered**2)/(n-1)` (X_centered squared IN PLACE — with copy=False user input is destroyed); `ratio = ev/total_var`.
- `noise_variance_ = (total_var - sum(ev)) / (min(shape) - n)` if `n < min(shape)` else `0.0`.

## _randomized_svd + _randomized_range_finder (extmath.py)

- `n_random = n_components + n_oversamples`.
- `n_iter == 'auto'` → `7 if n_components < 0.1*min(shape) else 4`.
- `transpose == 'auto'` → `n_samples < n_features`; if transpose, work on `M.T`.
- Range finder: `Q = random_state.normal(size=(M.shape[1], n_random))` (fills C-order; downcast to f32 AFTER generation if M is f32).
- normalizer auto: `n_iter <= 2` → 'none' else 'LU'. LU = `scipy.linalg.lu(permute_l=True)` → use P@L. QR = economic.
- Power iterations: `for _ in range(n_iter): Q,_ = norm(M @ Q); Q,_ = norm(M.T @ Q)`; then `Q,_ = qr_economic(M @ Q)`.
- `B = Q.T @ M`; SVD(B, gesdd); `U = Q @ Uhat`.
- PCA passes flip_sign=False. Return truncation: not transposed → `(U[:, :k], s[:k], Vt[:k, :])`; transposed → `(Vt[:k].T, s[:k], U[:, :k].T)`.

## _BasePCA

- `transform`: `X @ components_.T` then `-= mean_ @ components_.T` (centering AFTER projection — replicate the op order). whiten: `scale = sqrt(explained_variance_); scale[scale < eps(dtype)] = eps; X /= scale`.
- `inverse_transform`: whiten → `X @ (sqrt(ev)[:,None] * components_) + mean_`; else `X @ components_ + mean_`.
- `get_covariance`: `comp = whiten ? components_*sqrt(ev)[:,None] : components_`; `evd = where(ev > nv, ev - nv, 0)`; `cov = (comp.T * evd) @ comp`; `cov[diag] += nv`.
- `get_precision`: `n_components_==0` → `eye/nv`; `nv==0` → `inv(get_covariance())` (LU inverse); else Woodbury: `prec = comp @ comp.T / nv; prec[diag] += 1/evd; prec = comp.T @ inv(prec) @ comp; prec /= -nv²; prec[diag] += 1/nv`.
- `get_feature_names_out`: `[f"{classname.lower()}{i}"]` → `pca0…`, `incrementalpca0…`.
- `score_samples`: `Xr = X - mean_`; `ll = -0.5 * rowsum(Xr * (Xr @ precision)) - 0.5*(p*log(2π) - fast_logdet(precision))`; `fast_logdet` = slogdet, `-inf` if sign ≤ 0. `score` = mean.

## Minka MLE (_assess_dimension / _infer_dimension)

- Operates on `explained_variance_` (length p), n_samples. `ll[0] = -inf`; ranks 1..p-1; return argmax (first max).
- eps=1e-15; `spectrum[rank-1] < eps` → -inf; `pu = -rank*log(2) + Σ_{i=1..rank} [lgamma((p-i+1)/2) - log(π)*(p-i+1)/2]`; `pl = -n/2 * Σ log(spectrum[:rank])`; `v = max(eps, Σ spectrum[rank:]/(p-rank))`; `pv = -log(v)*n*(p-rank)/2`; `m = p*rank - rank(rank+1)/2`; `pp = log(2π)*(m+rank)/2`; `pa = Σ_{i<rank, j>i..p-1} log((sp[i]-sp[j])*(1/sp_[j]-1/sp_[i])) + log(n)` where `sp_` has tail replaced by v; `ll = pu+pl+pv+pp - pa/2 - rank*log(n)/2`.

## IncrementalPCA

- Params: n_components (int≥1 | None), whiten, copy, batch_size (int≥1 | None). NO random_state/svd_solver.
- `fit`: resets `components_=None, n_samples_seen_=0, mean_=0.0, var_=0.0, ...`; validate (copy per flag); `batch_size_ = 5*n_features` if None; `gen_batches(n, batch_size_, min_batch_size=n_components or 0)`; per batch partial_fit(check_input=False).
- `gen_batches`: full slices while `end + min_batch_size <= n`, else skip (tail merged); final `slice(start, n)` if remainder.
- `partial_fit`: `first_pass = not hasattr(self, 'components_')` (False when called from fit!). n_components_ resolution: None → min(batch shape) on first, else keep `components_.shape[0]`; errors: `n_components > n_features` (always), `n_components > n_samples and first_pass`. Shape-change error if components_.shape[0] != n_components_.
- `_incremental_mean_and_var(X, mean_, var_, repeat(n_seen, p))`:
  - last_sample_count cast to float64. `last_sum = last_mean * count`. `new_sum = sum(X, axis=0)` **accumulated in float64 even for f32 X**; mean_/var_ end up float64 always.
  - `updated_mean = (last_sum + new_sum)/updated_count`.
  - `T = new_sum/new_count; temp = X - T` (promotes to f64); `correction = sum(temp, axis=0)`; `new_unnorm_var = sum(temp², axis=0) - correction²/new_count`; `updated_unnorm = last_var*last_count + new_unnorm + (last_count/new_count)/updated_count * (last_sum*new_count/last_count - new_sum)²`; where last_count==0 → new_unnorm; `updated_var = updated_unnorm/updated_count`.
- Whitening step: if `n_samples_seen_ == 0`: `X -= col_mean`. Else: `X -= col_batch_mean` (batch mean!); `mean_correction = sqrt(n_seen/n_total * n_batch) * (mean_ - col_batch_mean)` (one row); stack rows: `[singular_values_[:,None] * components_; X; mean_correction]`.
- `SVD(stacked, full_matrices=False)`; `svd_flip(u_based_decision=False)`; `explained_variance = S²/(n_total-1)`; `explained_variance_ratio = S²/sum(col_var * n_total)`.
- Truncate to n_components_; `noise_variance_ = mean(explained_variance[n_components_:])` if `n_components_ not in (batch_n_samples, n_features)` else 0.0 (batch_n_samples = PRE-stack row count of this batch).
- `transform` for dense input: NOT batched (plain _BasePCA.transform).
- dtype: X validated to f64/f32; mean_/var_ float64; components_ follow X dtype.

## numpy RandomState (MT19937 legacy) — needed for randomized/arpack parity

- Seeding with int < 2³²: `init_genrand(seed)`: `mt[0]=s; mt[i] = 1812433253*(mt[i-1]^(mt[i-1]>>30)) + i (mod 2³²)`. (≥2³² → init_by_array — implement too.)
- genrand_uint32: standard MT19937 with tempering (u=11, s=7 &0x9d2c5680, t=15 &0xefc60000, l=18); refill block of 624.
- random_sample: `a = next()>>5, b = next()>>6; (a*67108864 + b)/9007199254740992`.
- uniform(lo,hi): `lo + (hi-lo)*random_sample()`, C-order fill.
- gauss (polar, cached): draw x1,x2 = 2*rs-1 until `0 < r2 = x1²+x2² < 1`; `f = sqrt(-2 log(r2)/r2)`; **return f*x2 first**, cache f*x1 (has_gauss flag). `normal(size)` fills C-order.

## Numerical notes for parity tolerances

- scipy full SVD = LAPACK gesdd; our Golub–Reinsch converges to same values ~1e-14 rel (f64). Sign fixed by svd_flip; degenerate/близкие singular values → subspace ambiguity: avoid exact ties in fixtures except dedicated edge cases (compare reconstructions there).
- `np.mean/np.sum` use pairwise summation in input dtype (PCA), but float64 accumulators in IncrementalPCA stats. We accumulate in f64 and cast — differences absorbed by tolerance (verify empirically per dtype).
- f32 fixtures need loose tolerances (~1e-3 rel on variances, ~1e-2 on components for randomized); f64 tight (~1e-9..1e-12).
- get_precision with noise_variance_==0 and rank-deficient covariance (n < p, all components) is numerically singular — exclude from fixture comparisons.
- PCA whiten transform clips scale at eps(dtype); zero-variance components produce large-but-finite values — fixtures capture actual sklearn output.
