/**
 * Principal component analysis matching `sklearn.decomposition.PCA` (1.9).
 *
 * Naming is idiomatic TypeScript camelCase; every option and fitted attribute
 * maps 1:1 to its sklearn counterpart (`nComponents` ↔ `n_components`,
 * `explainedVarianceRatio` ↔ `explained_variance_ratio_`, …). Numerics follow
 * sklearn's exact computation order, including its deterministic
 * `svd_flip(u_based_decision=False)` sign convention.
 */
import { BasePCA, castTo, promoteDtype } from './base.js';
import { asMatrix, Matrix, type MatrixInput } from './matrix.js';
import { syrkTChunk, syrkTMirror } from './numeric/blas.js';
import { eigh } from './numeric/eigh.js';
import { lanczosSvdSteps } from './numeric/lanczos.js';
import { slogdet } from './numeric/lu.js';
import { inferDimension } from './numeric/mle.js';
import { computeBigGemm, randomizedSvdSteps } from './numeric/randomized.js';
import { checkRandomState, RandomState } from './numeric/rng.js';
import { centerInPlace, colMeans, cumsum, searchsortedRight } from './numeric/stats.js';
import { type SvdResult, svd } from './numeric/svd.js';
import { svdFlipVBased } from './numeric/svdflip.js';
import {
  type FitAsyncOptions,
  type FitObserver,
  makeReporter,
  type PCAFitSnapshot,
  type ProgressReporter,
  projectForSnapshot,
  toFloat64Copy,
} from './progress.js';
import { driveAsync, driveSync } from './scheduling.js';
import { dtypeOf, epsFor, type FloatArray } from './types.js';
import { assertAllFinite, checkFeatureCount } from './validation.js';

export type SvdSolver = 'auto' | 'full' | 'covariance_eigh' | 'arpack' | 'randomized';
export type PowerIterationNormalizer = 'auto' | 'QR' | 'LU' | 'none';

export interface PCAOptions {
  /**
   * Number of components to keep. An integer ≥ 0, a fraction in (0, 1) to
   * select by explained-variance ratio (full/covariance_eigh solvers),
   * 'mle' for Minka's MLE (full/covariance_eigh), or null (default) to keep
   * min(nSamples, nFeatures) components (min − 1 for arpack).
   */
  nComponents?: number | 'mle' | null;
  /** When false, fit may overwrite the training Matrix's data (like sklearn's copy=False). */
  copy?: boolean;
  /** Scale transformed output to unit component-wise variance. */
  whiten?: boolean;
  svdSolver?: SvdSolver;
  /**
   * Accepted for API parity with sklearn's arpack tolerance. The built-in
   * Lanczos solver always converges the requested triplets to machine
   * precision (equivalent to sklearn's default tol=0.0).
   */
  tol?: number;
  /** Power-iteration count for the randomized solver ('auto' → 7 or 4 like sklearn). */
  iteratedPower?: number | 'auto';
  /** Extra random test vectors for the randomized solver's range finder. */
  nOversamples?: number;
  powerIterationNormalizer?: PowerIterationNormalizer;
  /** Integer seed (or RandomState) making arpack/randomized reproducible — sklearn-stream compatible. */
  randomState?: number | RandomState | null;
}

interface ResolvedOptions {
  nComponents: number | 'mle' | null;
  copy: boolean;
  whiten: boolean;
  svdSolver: SvdSolver;
  tol: number;
  iteratedPower: number | 'auto';
  nOversamples: number;
  powerIterationNormalizer: PowerIterationNormalizer;
  randomState: number | RandomState | null;
}

interface FitResult {
  /** Left singular vectors from the fit, null for covariance_eigh. */
  u: Float64Array | null;
  uCols: number;
  s: Float64Array;
  x: Matrix;
}

const SOLVERS: readonly SvdSolver[] = ['auto', 'full', 'covariance_eigh', 'arpack', 'randomized'];
const NORMALIZERS: readonly PowerIterationNormalizer[] = ['auto', 'QR', 'LU', 'none'];

/**
 * sklearn 1.9's auto-solver heuristic (`PCA._fit`), as a pure function so the
 * WebGPU frontend dispatches identically. `nComponents` is the raw option
 * (null → min(shape) for the heuristic, like sklearn's unset default).
 */
export function resolveSvdSolver(
  rows: number,
  cols: number,
  nComponents: number | 'mle' | null,
  requested: SvdSolver,
): Exclude<SvdSolver, 'auto'> {
  if (requested !== 'auto') {
    return requested;
  }
  const minDim = Math.min(rows, cols);
  const nc = nComponents === null ? minDim : nComponents;
  // Tall-and-skinny problems are best handled by precomputing the covariance.
  if (cols <= 1000 && rows >= 10 * cols) {
    return 'covariance_eigh';
  }
  if (Math.max(rows, cols) <= 500 || nc === 'mle') {
    return 'full';
  }
  if (typeof nc === 'number' && nc >= 1 && nc < 0.8 * minDim) {
    return 'randomized';
  }
  // Also the case of nComponents in (0, 1).
  return 'full';
}

/**
 * Fit-time nComponents/solver compatibility checks, shared by the CPU and
 * WebGPU frontends. Message text mirrors the sklearn errors it was ported
 * from (camelCased).
 */
export function validateNcForSolver(
  nc: number | 'mle',
  n: number,
  p: number,
  solver: Exclude<SvdSolver, 'auto'>,
): void {
  const minDim = Math.min(n, p);
  if (solver === 'full' || solver === 'covariance_eigh') {
    if (nc === 'mle') {
      if (n < p) {
        throw new Error("nComponents='mle' is only supported if nSamples >= nFeatures");
      }
    } else if (!(nc >= 0 && nc <= minDim)) {
      throw new Error(
        `nComponents=${nc} must be between 0 and min(nSamples, nFeatures)=${minDim} with svdSolver='${solver}'`,
      );
    }
    return;
  }
  if (typeof nc !== 'number') {
    throw new Error(`nComponents='${nc}' cannot be a string with svdSolver='${solver}'`);
  }
  if (!(Number.isInteger(nc) && nc >= 1 && nc <= minDim)) {
    throw new Error(
      `nComponents=${nc} must be an integer between 1 and min(nSamples, nFeatures)=${minDim} with svdSolver='${solver}'`,
    );
  }
  if (solver === 'arpack' && nc === minDim) {
    throw new Error(
      `nComponents=${nc} must be strictly less than min(nSamples, nFeatures)=${minDim} with svdSolver='arpack'`,
    );
  }
}

function validateOptions(o: PCAOptions): ResolvedOptions {
  const nComponents = o.nComponents === undefined ? null : o.nComponents;
  if (nComponents !== null && nComponents !== 'mle') {
    if (typeof nComponents !== 'number' || Number.isNaN(nComponents)) {
      throw new Error(`nComponents must be a number, 'mle', or null; got ${nComponents}`);
    }
    const isInt = Number.isInteger(nComponents);
    if (isInt ? nComponents < 0 : !(nComponents > 0 && nComponents < 1)) {
      throw new Error(
        `nComponents must be an integer >= 0, a float in (0, 1), 'mle', or null; got ${nComponents}`,
      );
    }
  }
  const svdSolver = o.svdSolver ?? 'auto';
  if (!SOLVERS.includes(svdSolver)) {
    throw new Error(`svdSolver must be one of ${SOLVERS.join(', ')}; got '${svdSolver}'`);
  }
  const tol = o.tol ?? 0.0;
  if (!(typeof tol === 'number' && tol >= 0)) {
    throw new Error(`tol must be a number >= 0; got ${tol}`);
  }
  const iteratedPower = o.iteratedPower ?? 'auto';
  if (iteratedPower !== 'auto' && !(Number.isInteger(iteratedPower) && iteratedPower >= 0)) {
    throw new Error(`iteratedPower must be 'auto' or an integer >= 0; got ${iteratedPower}`);
  }
  const nOversamples = o.nOversamples ?? 10;
  if (!(Number.isInteger(nOversamples) && nOversamples >= 1)) {
    throw new Error(`nOversamples must be an integer >= 1; got ${nOversamples}`);
  }
  const powerIterationNormalizer = o.powerIterationNormalizer ?? 'auto';
  if (!NORMALIZERS.includes(powerIterationNormalizer)) {
    throw new Error(
      `powerIterationNormalizer must be one of ${NORMALIZERS.join(', ')}; got '${powerIterationNormalizer}'`,
    );
  }
  return {
    nComponents,
    copy: o.copy ?? true,
    whiten: o.whiten ?? false,
    svdSolver,
    tol,
    iteratedPower,
    nOversamples,
    powerIterationNormalizer,
    randomState: o.randomState ?? null,
  };
}

export class PCA extends BasePCA {
  private readonly opts: ResolvedOptions;

  private nSamples_ = 0;
  private fitSvdSolver_: Exclude<SvdSolver, 'auto'> = 'full';

  constructor(options: PCAOptions = {}) {
    super();
    this.opts = validateOptions(options);
    this.whitenOpt = this.opts.whiten;
  }

  /** sklearn's `n_samples_`. */
  get nSamples(): number {
    this.assertFitted();
    return this.nSamples_;
  }

  /** The solver actually used by fit (sklearn's `_fit_svd_solver`). */
  get resolvedSvdSolver(): Exclude<SvdSolver, 'auto'> {
    this.assertFitted();
    return this.fitSvdSolver_;
  }

  // ------------------------------------------------------------------
  // Fitting
  // ------------------------------------------------------------------

  fit(X: MatrixInput, observer?: FitObserver): this {
    driveSync(this._fitSteps(asMatrix(X), observer), observer?.signal);
    return this;
  }

  /** Fit and return the embedding of X — sklearn's `fit_transform` fast path. */
  fitTransform(X: MatrixInput, observer?: FitObserver): Matrix {
    const xm = asMatrix(X);
    const r = driveSync(this._fitSteps(xm, observer), observer?.signal);
    return this.fitTransformFromResult(r);
  }

  /**
   * Non-blocking fit: runs the exact same steps as `fit`, time-sliced on
   * the event loop (`budgetMs` of work per slice), so the UI stays
   * responsive. Results are bit-identical to the synchronous fit.
   */
  async fitAsync(X: MatrixInput, options: FitAsyncOptions = {}): Promise<this> {
    await driveAsync(this._fitSteps(asMatrix(X), options), {
      budgetMs: options.budgetMs,
      signal: options.signal,
    });
    return this;
  }

  /** Non-blocking fitTransform — the same U·S fast path as the sync version. */
  async fitTransformAsync(X: MatrixInput, options: FitAsyncOptions = {}): Promise<Matrix> {
    const xm = asMatrix(X);
    const r = await driveAsync(this._fitSteps(xm, options), {
      budgetMs: options.budgetMs,
      signal: options.signal,
    });
    return this.fitTransformFromResult(r);
  }

  /** The U·S fast path shared by fitTransform and fitTransformAsync. */
  private fitTransformFromResult(r: FitResult): Matrix {
    const n = r.x.rows;
    const k = this.nComponents_;
    if (r.u !== null) {
      // X_new = U * S (or U * sqrt(n−1) when whitening).
      const out = new Float64Array(n * k);
      if (this.opts.whiten) {
        const f = Math.sqrt(n - 1);
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < k; c++) {
            out[i * k + c] = r.u[i * r.uCols + c] * f;
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < k; c++) {
            out[i * k + c] = r.u[i * r.uCols + c] * r.s[c];
          }
        }
      }
      return new Matrix(castTo(out, this.dtype), n, k);
    }
    // covariance_eigh computes no U at fit time.
    return this.transformCore(r.x, false);
  }

  /**
   * @internal The fit as a step generator — every `yield` is a suspension
   * and abort checkpoint for the drivers (sync drain, time-sliced async,
   * worker). Runs the exact statement sequence of the classic fit. Progress
   * events (including the identical sequence for sync and async drives) are
   * emitted synchronously between steps; on any exit before completion —
   * abort, input error, callback throw — the estimator ends up unfitted.
   */
  *_fitSteps(X: Matrix, observer?: FitObserver): Generator<void, FitResult, void> {
    if (this.fitting) {
      throw new Error(
        'This PCA instance is already fitting; concurrent fits on one instance are not supported',
      );
    }
    // Pre-fit validation: failures here leave any previous model intact.
    if (X.rows < 1 || X.cols < 1) {
      throw new Error(
        `Found array with shape (${X.rows}, ${X.cols}); PCA requires at least 1 sample and 1 feature`,
      );
    }
    assertAllFinite(X, 'PCA.fit');
    this.fitting = true;
    let completed = false;
    try {
      this.dtype = X.dtype;

      const minDim = Math.min(X.rows, X.cols);
      const ncOpt = this.opts.nComponents;
      const solver = resolveSvdSolver(X.rows, X.cols, ncOpt, this.opts.svdSolver);
      const nc: number | 'mle' =
        ncOpt === null ? (solver !== 'arpack' ? minDim : minDim - 1) : ncOpt;
      this.fitSvdSolver_ = solver;

      const reporter = makeReporter(observer, {
        estimator: 'PCA',
        solver,
        nRows: X.rows,
        whiten: this.opts.whiten,
      });

      let r: FitResult;
      if (solver === 'full' || solver === 'covariance_eigh') {
        r = yield* this.fitFullSteps(X, nc, reporter);
      } else {
        r = yield* this.fitTruncatedSteps(X, nc, solver, reporter);
      }
      if (reporter) {
        reporter.emit({
          phase: 'finalize',
          step: 1,
          totalSteps: 1,
          snapshot: this.finalizeSnapshot(reporter, r),
        });
      }
      completed = true;
      return r;
    } finally {
      this.fitting = false;
      if (!completed) {
        // A failed or aborted fit leaves no (possibly inconsistent) model.
        this.fitted = false;
      }
    }
  }

  private *fitFullSteps(
    X: Matrix,
    nc: number | 'mle',
    reporter: ProgressReporter | null,
    rawGram?: Float64Array,
  ): Generator<void, FitResult, void> {
    const n = X.rows;
    const p = X.cols;
    const minDim = Math.min(n, p);
    const solver = this.fitSvdSolver_;

    validateNcForSolver(nc, n, p, solver);

    const meanF64 = colMeans(X.data, n, p);

    let u: Float64Array | null;
    let uCols = 0;
    let s: Float64Array;
    let vt: Float64Array;
    let explainedVariance: Float64Array;

    if (solver === 'full') {
      const xc = this.opts.copy ? X.data.slice() : X.data;
      centerInPlace(xc, n, p, meanF64);
      reporter?.emit({ phase: 'decompose', step: 0, totalSteps: null });
      yield;
      const dec = svd(xc, n, p);
      u = dec.u;
      uCols = minDim;
      s = dec.s;
      vt = dec.vt;
      explainedVariance = new Float64Array(minDim);
      for (let i = 0; i < minDim; i++) {
        explainedVariance[i] = (s[i] * s[i]) / (n - 1);
      }
    } else {
      // covariance_eigh: form the Gram matrix and center it afterwards,
      // avoiding any copy or mutation of X. The WebGPU frontend passes the
      // Gram in precomputed; everything downstream is shared.
      const c = rawGram ?? (yield* this.gramSteps(X, reporter));
      for (let i = 0; i < p; i++) {
        for (let j = 0; j < p; j++) {
          c[i * p + j] = (c[i * p + j] - n * meanF64[i] * meanF64[j]) / (n - 1);
        }
      }
      reporter?.emit({ phase: 'decompose', step: 0, totalSteps: null });
      yield;
      const dec = eigh(c, p);
      // Ascending → descending; clip tiny negatives (PSD by construction).
      const evals = new Float64Array(p);
      for (let i = 0; i < p; i++) {
        const v = dec.values[p - 1 - i];
        evals[i] = v < 0 ? 0 : v;
      }
      explainedVariance = evals;
      s = new Float64Array(p);
      vt = new Float64Array(p * p);
      for (let i = 0; i < p; i++) {
        s[i] = Math.sqrt(evals[i] * (n - 1));
        for (let j = 0; j < p; j++) {
          vt[i * p + j] = dec.vectors[j * p + (p - 1 - i)];
        }
      }
      u = null;
    }

    // Deterministic sign convention (sklearn 1.9: v-based for all solvers).
    svdFlipVBased(u, n, vt, u === null ? p : minDim, p);

    let totalVar = 0;
    for (let i = 0; i < explainedVariance.length; i++) {
      totalVar += explainedVariance[i];
    }
    const ratio = new Float64Array(explainedVariance.length);
    for (let i = 0; i < explainedVariance.length; i++) {
      ratio[i] = explainedVariance[i] / totalVar;
    }

    let k: number;
    if (nc === 'mle') {
      k = inferDimension(explainedVariance, n);
    } else if (nc > 0 && nc < 1) {
      k = searchsortedRight(cumsum(ratio), nc) + 1;
    } else {
      k = nc;
    }

    this.noiseVariance_ = 0;
    if (k < minDim) {
      let acc = 0;
      for (let i = k; i < explainedVariance.length; i++) {
        acc += explainedVariance[i];
      }
      this.noiseVariance_ = acc / (explainedVariance.length - k);
    }

    this.storeFitted(X, meanF64, k, vt, explainedVariance, ratio, s);
    return { u, uCols, s, x: X };
  }

  /**
   * The XᵀX Gram product in row chunks, yielding between chunks. Chunked
   * accumulation is strictly row-sequential, so the result is bitwise
   * identical to one monolithic syrkT pass.
   */
  private *gramSteps(
    X: Matrix,
    reporter: ProgressReporter | null,
  ): Generator<void, Float64Array, void> {
    const n = X.rows;
    const p = X.cols;
    const c = new Float64Array(p * p);
    const chunkRows = Math.max(64, Math.ceil(2 ** 22 / (p * p)));
    const totalSteps = Math.ceil(n / chunkRows);
    let step = 0;
    for (let start = 0; start < n; start += chunkRows) {
      const end = Math.min(n, start + chunkRows);
      syrkTChunk(X.data, p, start, end, c);
      step++;
      reporter?.emit({ phase: 'gram', step, totalSteps });
      yield;
    }
    syrkTMirror(c, p);
    return c;
  }

  private *fitTruncatedSteps(
    X: Matrix,
    nc: number | 'mle',
    solver: 'arpack' | 'randomized',
    reporter: ProgressReporter | null,
  ): Generator<void, FitResult, void> {
    const n = X.rows;
    const p = X.cols;
    const minDim = Math.min(n, p);

    validateNcForSolver(nc, n, p, solver);
    const k = nc as number;

    const rng = checkRandomState(this.opts.randomState);

    const meanF64 = colMeans(X.data, n, p);
    const xc = this.opts.copy ? X.data.slice() : X.data;
    centerInPlace(xc, n, p, meanF64);

    let dec: SvdResult;
    if (solver === 'arpack') {
      const v0 = new Float64Array(minDim);
      rng.uniform(-1, 1, v0);
      // scipy's svds returns ascending order and sklearn reverses it; our
      // Lanczos yields the same converged triplets already descending.
      dec = yield* lanczosSvdSteps(xc, n, p, k, v0, rng, reporter);
    } else {
      const gen = randomizedSvdSteps(n, p, k, {
        nOversamples: this.opts.nOversamples,
        nIter: this.opts.iteratedPower,
        powerIterationNormalizer: this.opts.powerIterationNormalizer,
        rng,
        float32Stream: this.dtype === 'float32',
        hooks: reporter,
      });
      let step = gen.next();
      while (!step.done) {
        yield;
        step = gen.next(computeBigGemm(xc, n, p, step.value));
      }
      dec = step.value;
    }
    return this.finishTruncated(X, xc, meanF64, k, dec);
  }

  /** Post-decomposition tail of the truncated solvers (flip, variances, store). */
  private finishTruncated(
    X: Matrix,
    xc: FloatArray,
    meanF64: Float64Array,
    nc: number,
    dec: SvdResult,
  ): FitResult {
    const n = X.rows;
    const p = X.cols;
    const minDim = Math.min(n, p);
    const { u, s, vt } = dec;
    svdFlipVBased(u, n, vt, nc, p);

    const explainedVariance = new Float64Array(nc);
    for (let i = 0; i < nc; i++) {
      explainedVariance[i] = (s[i] * s[i]) / (n - 1);
    }
    // Total variance of the centered data. sklearn squares X_centered in
    // place — destroying the caller's data when copy=false — replicated here.
    let totalVar = 0;
    if (xc !== X.data) {
      for (let i = 0; i < xc.length; i++) {
        totalVar += xc[i] * xc[i];
      }
    } else {
      for (let i = 0; i < xc.length; i++) {
        xc[i] *= xc[i];
        totalVar += xc[i];
      }
    }
    totalVar /= n - 1;

    const ratio = new Float64Array(nc);
    for (let i = 0; i < nc; i++) {
      ratio[i] = explainedVariance[i] / totalVar;
    }
    this.noiseVariance_ = 0;
    if (nc < minDim) {
      let acc = totalVar;
      for (let i = 0; i < nc; i++) {
        acc -= explainedVariance[i];
      }
      this.noiseVariance_ = acc / (minDim - nc);
    }

    this.storeFitted(X, meanF64, nc, vt, explainedVariance, ratio, s);
    return { u, uCols: nc, s, x: X };
  }

  // ------------------------------------------------------------------
  // WebGPU frontend bridges (internal API — not part of the public surface)
  // ------------------------------------------------------------------

  /**
   * @internal Completes a covariance_eigh fit from an externally computed
   * raw Gram matrix XᵀX (uncentered, float64, p×p). Used by the WebGPU
   * frontend; all semantics downstream of the Gram product are shared with
   * the CPU path. `rawGram` is consumed (mutated in place).
   */
  _fitGram(X: Matrix, rawGram: Float64Array): void {
    this.prepareFit(X, 'covariance_eigh');
    const minDim = Math.min(X.rows, X.cols);
    const ncOpt = this.opts.nComponents;
    driveSync(this.fitFullSteps(X, ncOpt === null ? minDim : ncOpt, null, rawGram));
  }

  /**
   * @internal Completes a randomized fit from an externally computed
   * decomposition of the centered data. `xc` must be the centered training
   * buffer (X.data itself when copy=false, matching sklearn's destructive
   * semantics) and `meanF64` the original column means.
   */
  _fitDecomposed(X: Matrix, xc: FloatArray, meanF64: Float64Array, dec: SvdResult): void {
    this.prepareFit(X, 'randomized');
    const ncOpt = this.opts.nComponents;
    const nc = ncOpt === null ? Math.min(X.rows, X.cols) : ncOpt;
    validateNcForSolver(nc, X.rows, X.cols, 'randomized');
    this.finishTruncated(X, xc, meanF64, nc as number, dec);
  }

  /** @internal Option access for the WebGPU frontend (read-only). */
  _resolvedOptions(): Readonly<ResolvedOptions> {
    return this.opts;
  }

  /** The fitCore preamble shared by the internal fit bridges. */
  private prepareFit(X: Matrix, solver: Exclude<SvdSolver, 'auto'>): void {
    if (X.rows < 1 || X.cols < 1) {
      throw new Error(
        `Found array with shape (${X.rows}, ${X.cols}); PCA requires at least 1 sample and 1 feature`,
      );
    }
    assertAllFinite(X, 'PCA.fit');
    this.dtype = X.dtype;
    this.fitSvdSolver_ = solver;
  }

  /**
   * The final model as a snapshot (fresh float64 copies of the stored
   * attributes — float32 fits produce float32-rounded values, faithfully
   * reflecting the model). Scores use U·S when the solver produced a U
   * (`fitTransform`'s math) and one projection GEMM for covariance_eigh,
   * whose fit never mutates X.
   */
  private finalizeSnapshot(reporter: ProgressReporter, r: FitResult): PCAFitSnapshot | undefined {
    if (!reporter.snapshotsEnabled) {
      return undefined;
    }
    const comp = this.components_ as Matrix;
    const k = this.nComponents_;
    const p = this.nFeaturesIn_;
    const snapshot: PCAFitSnapshot = {
      components: new Matrix(toFloat64Copy(comp.data), k, p),
      singularValues: toFloat64Copy(this.singularValues_ as FloatArray),
      explainedVariance: toFloat64Copy(this.explainedVariance_ as FloatArray),
    };
    if (reporter.scoresRequested) {
      const n = r.x.rows;
      if (r.u !== null) {
        const scores = new Float64Array(n * k);
        const f = Math.sqrt(n - 1);
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < k; c++) {
            const uv = r.u[i * r.uCols + c];
            scores[i * k + c] = this.opts.whiten ? uv * f : uv * r.s[c];
          }
        }
        snapshot.scores = new Matrix(scores, n, k);
      } else {
        const ev = this.explainedVariance_ as FloatArray;
        snapshot.scores = projectForSnapshot(
          r.x.data,
          n,
          p,
          comp.data,
          k,
          this.mean_ as FloatArray,
          ev,
          this.opts.whiten,
          epsFor(dtypeOf(ev)),
        );
      }
    }
    return snapshot;
  }

  private storeFitted(
    X: Matrix,
    meanF64: Float64Array,
    k: number,
    vt: Float64Array,
    explainedVariance: Float64Array,
    ratio: Float64Array,
    s: Float64Array,
  ): void {
    const p = X.cols;
    const dt = this.dtype;
    this.nSamples_ = X.rows;
    this.nFeaturesIn_ = p;
    this.nComponents_ = k;
    this.components_ = new Matrix(castTo(vt.slice(0, k * p), dt), k, p);
    this.explainedVariance_ = castTo(explainedVariance.slice(0, k), dt);
    this.explainedVarianceRatio_ = castTo(ratio.slice(0, k), dt);
    this.singularValues_ = castTo(s.slice(0, k), dt);
    this.mean_ = castTo(meanF64, dt);
    this.fitted = true;
  }

  // ------------------------------------------------------------------
  // Probabilistic PCA log-likelihood (PCA-only in sklearn, too)
  // ------------------------------------------------------------------

  /** Per-sample log-likelihood under the Tipping–Bishop probabilistic PCA model. */
  scoreSamples(X: MatrixInput): FloatArray {
    this.assertFitted();
    const xm = asMatrix(X);
    assertAllFinite(xm, 'PCA.scoreSamples');
    checkFeatureCount(xm, this.nFeaturesIn_, this.estimatorName);
    const meanArr = this.mean_ as FloatArray;
    const n = xm.rows;
    const p = xm.cols;
    const prec = this.precisionF64();
    const [sign, logdet] = slogdet(prec, p);
    const fastLogdet = sign > 0 ? logdet : Number.NEGATIVE_INFINITY;
    const constTerm = 0.5 * (p * Math.log(2 * Math.PI) - fastLogdet);
    const out = new Float64Array(n);
    const xr = new Float64Array(p);
    const proj = new Float64Array(p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) {
        xr[j] = xm.data[off + j] - meanArr[j];
      }
      proj.fill(0);
      for (let a = 0; a < p; a++) {
        const v = xr[a];
        if (v !== 0) {
          const poff = a * p;
          for (let j = 0; j < p; j++) {
            proj[j] += v * prec[poff + j];
          }
        }
      }
      let acc = 0;
      for (let j = 0; j < p; j++) {
        acc += xr[j] * proj[j];
      }
      out[i] = -0.5 * acc - constTerm;
    }
    return castTo(out, promoteDtype(xm.dtype, this.dtype));
  }

  /** Mean log-likelihood of all samples — sklearn's `score`. */
  score(X: MatrixInput): number {
    const ll = this.scoreSamples(X);
    let acc = 0;
    for (let i = 0; i < ll.length; i++) {
      acc += ll[i];
    }
    return acc / ll.length;
  }
}

export { RandomState };
