/**
 * Incremental PCA matching `sklearn.decomposition.IncrementalPCA` (1.9):
 * constant-memory fitting via `partialFit` batches (Ross et al. 2008), so the
 * dataset never needs to reside in memory at once.
 */
import { BasePCA, castTo } from './base.js';
import { asMatrix, Matrix, type MatrixInput } from './matrix.js';
import { assertValidModel, type IncrementalPCAModel } from './model.js';
import { colMeans, incrementalMeanAndVar } from './numeric/stats.js';
import { svd } from './numeric/svd.js';
import { svdFlipVBased } from './numeric/svdflip.js';
import {
  type FitAsyncOptions,
  type FitObserver,
  makeReporter,
  type PCAFitSnapshot,
  projectForSnapshot,
  toFloat64Copy,
} from './progress.js';
import { driveAsync, driveSync } from './scheduling.js';
import { type Dtype, dtypeOf, epsFor, type FloatArray } from './types.js';
import { assertAllFinite, checkFeatureCount } from './validation.js';

export interface IncrementalPCAOptions {
  /** Number of components to keep (integer ≥ 1), or null to use min(batch shape). */
  nComponents?: number | null;
  /** Scale transformed output to unit component-wise variance. */
  whiten?: boolean;
  /** When false, fit/partialFit may overwrite the input Matrix's data. */
  copy?: boolean;
  /** Rows per batch for fit(); null → 5 × nFeatures, like sklearn. */
  batchSize?: number | null;
}

interface ResolvedOptions {
  nComponents: number | null;
  whiten: boolean;
  copy: boolean;
  batchSize: number | null;
}

function validateOptions(o: IncrementalPCAOptions): ResolvedOptions {
  const nComponents = o.nComponents ?? null;
  if (nComponents !== null && !(Number.isInteger(nComponents) && nComponents >= 1)) {
    throw new Error(`nComponents must be an integer >= 1 or null; got ${nComponents}`);
  }
  const batchSize = o.batchSize ?? null;
  if (batchSize !== null && !(Number.isInteger(batchSize) && batchSize >= 1)) {
    throw new Error(`batchSize must be an integer >= 1 or null; got ${batchSize}`);
  }
  return {
    nComponents,
    whiten: o.whiten ?? false,
    copy: o.copy ?? true,
    batchSize,
  };
}

/** sklearn's `gen_batches`: full slices, with a too-small tail merged into the last one. */
function* genBatches(
  n: number,
  batchSize: number,
  minBatchSize: number,
): Generator<[number, number]> {
  let start = 0;
  const nFull = Math.floor(n / batchSize);
  for (let i = 0; i < nFull; i++) {
    const end = start + batchSize;
    if (end + minBatchSize > n) {
      continue;
    }
    yield [start, end];
    start = end;
  }
  if (start < n) {
    yield [start, n];
  }
}

export class IncrementalPCA extends BasePCA {
  protected override readonly estimatorName: string = 'IncrementalPCA';
  private readonly opts: ResolvedOptions;

  private var_: Float64Array | null = null;
  private meanF64_: Float64Array | null = null;
  private nSamplesSeen_ = 0;
  private batchSize_: number | null = null;
  /** Mirrors sklearn's hasattr(self, 'components_') — set by fit() before batching. */
  private hasComponentsAttr_ = false;

  constructor(options: IncrementalPCAOptions = {}) {
    super();
    this.opts = validateOptions(options);
    this.whitenOpt = this.opts.whiten;
  }

  /** Per-feature running variance (always float64, like sklearn) — `var_`. */
  get variance(): Float64Array {
    this.assertFitted();
    return this.var_ as Float64Array;
  }

  /** Samples processed so far — sklearn's `n_samples_seen_`. */
  get nSamplesSeen(): number {
    this.assertFitted();
    return this.nSamplesSeen_;
  }

  /** Batch size used by fit — sklearn's `batch_size_` (only set by fit()). */
  get batchSize(): number {
    if (this.batchSize_ === null) {
      throw new Error("batchSize is only available after fit() (sklearn's batch_size_)");
    }
    return this.batchSize_;
  }

  // ------------------------------------------------------------------
  // Fitting
  // ------------------------------------------------------------------

  /** Fit in minibatches of `batchSize` rows — sklearn's `IncrementalPCA.fit`. */
  fit(X: MatrixInput, observer?: FitObserver): this {
    driveSync(this._fitSteps(asMatrix(X), observer), observer?.signal);
    return this;
  }

  /**
   * Non-blocking fit: identical batching, time-sliced on the event loop.
   * Results are bit-identical to the synchronous fit. An abort between
   * batches keeps the completed-batch model (the estimator stays usable).
   */
  async fitAsync(X: MatrixInput, options: FitAsyncOptions = {}): Promise<this> {
    await driveAsync(this._fitSteps(asMatrix(X), options), {
      budgetMs: options.budgetMs,
      signal: options.signal,
    });
    return this;
  }

  /**
   * @internal The batched fit as a step generator — one `yield` per batch
   * is the suspension/abort checkpoint for the drivers. Mirrors sklearn's
   * fit(): state is reset before validation, and each batch runs the
   * unchanged partialFit update.
   */
  *_fitSteps(X: Matrix, observer?: FitObserver): Generator<void, void, void> {
    if (this.fitting) {
      throw new Error(
        'This IncrementalPCA instance is already fitting; concurrent fits on one instance are not supported',
      );
    }
    this.fitting = true;
    try {
      // Reset all state, mirroring sklearn's fit().
      this.components_ = null;
      this.hasComponentsAttr_ = true; // sklearn sets self.components_ = None
      this.nSamplesSeen_ = 0;
      this.meanF64_ = null;
      this.var_ = null;
      this.singularValues_ = null;
      this.explainedVariance_ = null;
      this.explainedVarianceRatio_ = null;
      this.noiseVariance_ = 0;
      this.fitted = false;

      assertAllFinite(X, 'IncrementalPCA.fit');
      if (X.rows < 1 || X.cols < 1) {
        throw new Error(
          `Found array with shape (${X.rows}, ${X.cols}); at least 1 sample and 1 feature required`,
        );
      }
      const work = this.opts.copy ? X.copy() : X;
      const n = work.rows;
      const p = work.cols;
      this.nFeaturesIn_ = p;
      this.batchSize_ = this.opts.batchSize ?? 5 * p;

      // Materialized so totalSteps is known up front.
      const batches = [...genBatches(n, this.batchSize_, this.opts.nComponents ?? 0)];
      const reporter = makeReporter(observer, {
        estimator: 'IncrementalPCA',
        solver: 'incremental',
        nRows: n,
        whiten: this.opts.whiten,
      });
      for (let i = 0; i < batches.length; i++) {
        const [start, end] = batches[i];
        const rows = end - start;
        const batch = new Matrix(work.data.subarray(start * p, end * p), rows, p);
        this.partialFitCore(batch);
        if (reporter) {
          reporter.emit({
            phase: 'batch',
            step: i + 1,
            totalSteps: batches.length,
            snapshot: reporter.wantSnapshot(i + 1)
              ? this.batchSnapshot(X, end, reporter.scoresRequested)
              : undefined,
          });
        }
        yield;
      }
      if (reporter) {
        reporter.emit({
          phase: 'finalize',
          step: batches.length,
          totalSteps: batches.length,
          snapshot: reporter.snapshotsEnabled
            ? this.batchSnapshot(X, n, reporter.scoresRequested)
            : undefined,
        });
      }
    } finally {
      this.fitting = false;
    }
  }

  /**
   * Snapshot of the just-stored (batch-prefix) model, as fresh float64
   * copies. Scores project the original input's rows seen so far; with
   * `copy: false` the first batch's rows were destructively centered by
   * the update (sklearn's contract), making those scores approximate.
   */
  private batchSnapshot(X: Matrix, rowsSeen: number, wantScores: boolean): PCAFitSnapshot {
    const comp = this.components_ as Matrix;
    const k = comp.rows;
    const p = comp.cols;
    const snapshot: PCAFitSnapshot = {
      components: new Matrix(toFloat64Copy(comp.data), k, p),
      singularValues: toFloat64Copy(this.singularValues_ as FloatArray),
      explainedVariance: toFloat64Copy(this.explainedVariance_ as FloatArray),
    };
    if (wantScores) {
      const ev = this.explainedVariance_ as FloatArray;
      snapshot.scores = projectForSnapshot(
        X.data,
        rowsSeen,
        p,
        comp.data,
        k,
        this.mean_ as FloatArray,
        ev,
        this.whitenOpt,
        epsFor(dtypeOf(ev)),
      );
    }
    return snapshot;
  }

  /** Incremental fit on one batch — sklearn's `partial_fit`. All of X is one batch. */
  partialFit(X: MatrixInput): this {
    if (this.fitting) {
      throw new Error(
        'This IncrementalPCA instance is fitting; partialFit cannot run concurrently with fit',
      );
    }
    const xm = asMatrix(X);
    assertAllFinite(xm, 'IncrementalPCA.partialFit');
    if (xm.rows < 1 || xm.cols < 1) {
      throw new Error(
        `Found array with shape (${xm.rows}, ${xm.cols}); at least 1 sample and 1 feature required`,
      );
    }
    if (this.nFeaturesIn_ > 0 && this.hasComponentsAttr_) {
      checkFeatureCount(xm, this.nFeaturesIn_, this.estimatorName);
    } else {
      this.nFeaturesIn_ = xm.cols;
    }
    const work = this.opts.copy ? xm.copy() : xm;
    this.partialFitCore(work);
    return this;
  }

  /** Equivalent to fit(X).transform(X) (sklearn's TransformerMixin behavior). */
  fitTransform(X: MatrixInput, observer?: FitObserver): Matrix {
    this.fit(X, observer);
    return this.transform(X);
  }

  /** Non-blocking fitTransform — fitAsync followed by transform. */
  async fitTransformAsync(X: MatrixInput, options: FitAsyncOptions = {}): Promise<Matrix> {
    await this.fitAsync(X, options);
    return this.transform(X);
  }

  // ------------------------------------------------------------------
  // Model serialization
  // ------------------------------------------------------------------

  /**
   * The fitted model as a plain structured-clone-friendly object. Includes
   * the running statistics (`nSamplesSeen`, float64 mean/variance), so
   * `partialFit` resumes bit-exactly after rehydration.
   */
  toModel(): IncrementalPCAModel {
    const base = this.exportBaseModel();
    return {
      ...base,
      estimator: 'ipca',
      nSamplesSeen: this.nSamplesSeen_,
      variance: (this.var_ as Float64Array).slice(),
      batchSize: this.batchSize_,
      options: {
        nComponents: this.opts.nComponents,
        whiten: this.opts.whiten,
        copy: this.opts.copy,
        batchSize: this.opts.batchSize,
      },
    };
  }

  /**
   * Rehydrates a fitted IncrementalPCA from `toModel()` output (validated
   * first; arrays adopted without copying). Further `partialFit` calls
   * continue the stream exactly where the exported estimator left off.
   */
  static fromModel(model: IncrementalPCAModel): IncrementalPCA {
    assertValidModel(model, 'ipca');
    const ipca = new IncrementalPCA(model.options);
    ipca.importBaseModel(model);
    // mean_ and meanF64_ alias one array in live estimators; preserve that.
    ipca.meanF64_ = model.mean as Float64Array;
    ipca.var_ = model.variance;
    ipca.nSamplesSeen_ = model.nSamplesSeen;
    ipca.batchSize_ = model.batchSize;
    ipca.hasComponentsAttr_ = true;
    return ipca;
  }

  /** The whitening/update step; mutates `X`'s data (callers pass a copy unless copy=false). */
  private partialFitCore(X: Matrix): void {
    const firstPass = !this.hasComponentsAttr_;
    if (firstPass) {
      this.components_ = null;
      this.hasComponentsAttr_ = true;
    }
    const n = X.rows;
    const p = X.cols;

    // Resolve n_components exactly like sklearn.
    const ncOpt = this.opts.nComponents;
    if (ncOpt === null) {
      if (this.components_ === null) {
        this.nComponents_ = Math.min(n, p);
      } else {
        this.nComponents_ = this.components_.rows;
      }
    } else if (!(ncOpt <= p)) {
      throw new Error(
        `nComponents=${ncOpt} invalid for nFeatures=${p}, need more rows than columns for IncrementalPCA processing`,
      );
    } else if (ncOpt > n && firstPass) {
      throw new Error(
        `nComponents=${ncOpt} must be less or equal to the batch number of samples ${n} for the first partialFit call.`,
      );
    } else {
      this.nComponents_ = ncOpt;
    }
    if (this.components_ !== null && this.components_.rows !== this.nComponents_) {
      throw new Error(
        `Number of components has changed from ${this.components_.rows} to ` +
          `${this.nComponents_} between calls to partialFit! Try setting nComponents to a fixed value.`,
      );
    }
    const k = this.nComponents_;

    // Update running statistics (float64, like sklearn's float64 accumulators).
    const stats = incrementalMeanAndVar(X.data, n, p, this.meanF64_, this.var_, this.nSamplesSeen_);
    const nTotal = stats.count;

    // Build the matrix to decompose.
    let stacked: FloatArray;
    let stackedRows: number;
    let stackedDtype: Dtype;
    if (this.nSamplesSeen_ === 0) {
      // First step: just center the batch (in its own dtype, like sklearn).
      stackedDtype = X.dtype;
      stackedRows = n;
      stacked = X.data;
      for (let i = 0; i < n; i++) {
        const off = i * p;
        for (let j = 0; j < p; j++) {
          stacked[off + j] -= stats.mean[j];
        }
      }
    } else {
      // Later steps: numpy's vstack promotes to float64 because the
      // mean-correction row is float64 — replicated (a float32 model's
      // attributes really do become float64 after the second partial_fit).
      const comp = this.components_ as Matrix;
      const sv = this.singularValues_ as FloatArray;
      const colBatchMean = colMeans(X.data, n, p);
      stackedDtype = 'float64';
      stackedRows = k + n + 1;
      const st = new Float64Array(stackedRows * p);
      for (let c = 0; c < k; c++) {
        const f = sv[c];
        const off = c * p;
        for (let j = 0; j < p; j++) {
          st[off + j] = f * comp.data[off + j];
        }
      }
      // X -= col_batch_mean happens in the batch's dtype in sklearn.
      const f32 = X.dtype === 'float32';
      for (let i = 0; i < n; i++) {
        const src = i * p;
        const dst = (k + i) * p;
        for (let j = 0; j < p; j++) {
          const v = X.data[src + j] - colBatchMean[j];
          st[dst + j] = f32 ? Math.fround(v) : v;
        }
      }
      const corrScale = Math.sqrt((this.nSamplesSeen_ / nTotal) * n);
      const off = (k + n) * p;
      for (let j = 0; j < p; j++) {
        st[off + j] = corrScale * ((this.meanF64_ as Float64Array)[j] - colBatchMean[j]);
      }
      stacked = st;
    }

    const dec = svd(stacked, stackedRows, p);
    const minDim = Math.min(stackedRows, p);
    svdFlipVBased(dec.u, stackedRows, dec.vt, minDim, p);

    const explainedVariance = new Float64Array(minDim);
    let varTimesTotal = 0;
    for (let j = 0; j < p; j++) {
      varTimesTotal += stats.variance[j] * nTotal;
    }
    const ratio = new Float64Array(minDim);
    for (let i = 0; i < minDim; i++) {
      const s2 = dec.s[i] * dec.s[i];
      explainedVariance[i] = s2 / (nTotal - 1);
      ratio[i] = s2 / varTimesTotal;
    }

    this.nSamplesSeen_ = nTotal;
    this.meanF64_ = stats.mean;
    this.var_ = stats.variance;
    this.dtype = stackedDtype;
    this.components_ = new Matrix(castTo(dec.vt.slice(0, k * p), stackedDtype), k, p);
    this.singularValues_ = castTo(dec.s.slice(0, k), stackedDtype);
    this.explainedVariance_ = castTo(explainedVariance.slice(0, k), stackedDtype);
    this.explainedVarianceRatio_ = castTo(ratio.slice(0, k), stackedDtype);
    this.mean_ = stats.mean; // float64, like sklearn
    if (k !== n && k !== p) {
      let acc = 0;
      for (let i = k; i < minDim; i++) {
        acc += explainedVariance[i];
      }
      this.noiseVariance_ = acc / (minDim - k);
    } else {
      this.noiseVariance_ = 0;
    }
    this.fitted = true;
  }
}
