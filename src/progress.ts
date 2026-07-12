/**
 * Progress reporting for fits: the public observer/event types and the
 * internal reporter that turns solver checkpoints into `PCAFitProgress`
 * events with optional intermediate model snapshots.
 *
 * Layering: solver modules under `src/numeric/` import ONLY types from this
 * file (erased at compile time), keeping the numeric layer dependency-free.
 * The estimators construct the concrete `ProgressReporter` and thread it
 * through solver options as `SolverHooks`.
 */
import { Matrix } from './matrix.js';
import { matmulTransB } from './numeric/blas.js';
import type { SvdResult } from './numeric/svd.js';
import { svdFlipVBased } from './numeric/svdflip.js';
import type { AbortSignalLike } from './scheduling.js';
import type { FloatArray } from './types.js';

/** The solver a progress event originates from. */
export type FitSolverId = 'full' | 'covariance_eigh' | 'arpack' | 'randomized' | 'incremental';

/** The stage of the fit a progress event belongs to. */
export type FitPhase =
  | 'gram'
  | 'decompose'
  | 'power-iteration'
  | 'lanczos-step'
  | 'batch'
  | 'finalize';

/**
 * An intermediate (or final) model snapshot: the evolving principal axes
 * and, when requested, the embedding of the training rows. All arrays are
 * fresh float64 copies owned by the callback — safe to keep, mutate, or
 * transfer to another thread; they never alias solver or estimator state.
 */
export interface PCAFitSnapshot {
  /** k×p principal axes in sklearn's `svd_flip` sign convention. */
  components: Matrix;
  /** Singular values (length k, descending). */
  singularValues: Float64Array;
  /** Per-component variance, s²/(n−1) against the rows fitted so far. */
  explainedVariance: Float64Array;
  /**
   * rows×k embedding of the training rows (U·S, or U·√(n−1) when
   * whitening). Present only when `SnapshotOptions.scores` is set.
   */
  scores?: Matrix;
}

/** One progress event. Emitted synchronously from inside the fit. */
export interface PCAFitProgress {
  estimator: 'PCA' | 'IncrementalPCA';
  solver: FitSolverId;
  phase: FitPhase;
  /** Completed units of the current phase (0 marks entering a phase). */
  step: number;
  /** Units in the current phase, when known ahead of time. */
  totalSteps: number | null;
  /**
   * Overall fit progress in [0, 1] — monotone within a fit, exactly 1 at
   * `finalize` — or null while progress is indeterminate.
   */
  fraction: number | null;
  /** Intermediate model, when requested through `SnapshotOptions`. */
  snapshot?: PCAFitSnapshot;
  /** Solver-specific extras (arpack: `{basisSize, jmax, maxResidual}`). */
  detail?: Readonly<Record<string, number>>;
}

/** Opt-in intermediate model snapshots on progress events. */
export interface SnapshotOptions {
  /** Attach evolving components/singularValues/explainedVariance. */
  components?: boolean;
  /** Also attach `scores` (implies `components`). */
  scores?: boolean;
  /**
   * Attach snapshots only to every Nth step within a phase (default 1).
   * The `finalize` event always carries a snapshot when snapshots are
   * enabled. Snapshots cost roughly one extra solver step on the randomized
   * solver, so `every: 2` halves that overhead.
   */
  every?: number;
}

/** Observer options accepted by the synchronous `fit`/`fitTransform`. */
export interface FitObserver {
  onProgress?: (event: PCAFitProgress) => void;
  snapshot?: SnapshotOptions;
  signal?: AbortSignalLike;
}

/** Options accepted by `fitAsync`/`fitTransformAsync`. */
export interface FitAsyncOptions extends FitObserver {
  /** Milliseconds of solver work per event-loop slice (default 12). */
  budgetMs?: number;
}

// ---------------------------------------------------------------------
// Internal reporting machinery (not part of the public API surface)
// ---------------------------------------------------------------------

/** @internal What a solver reports at a checkpoint. */
export interface SolverProgressEmit {
  phase: FitPhase;
  step: number;
  totalSteps: number | null;
  /**
   * Fresh snapshot-only decomposition triplets, truncated to k, oriented
   * like the input matrix, unflipped. Ownership transfers to the reporter,
   * which applies the sign convention in place. Solvers must never pass
   * arrays that the ongoing fit still reads.
   */
  dec?: SvdResult | null;
  /** Pre-built snapshot (estimator-side emits: batches, finalize). */
  snapshot?: PCAFitSnapshot;
  detail?: Readonly<Record<string, number>>;
}

/**
 * @internal Reporting channel threaded through solver options. Solvers call
 * `wantSnapshot` before spending anything on a snapshot decomposition and
 * `emit` at every checkpoint. Both are purely synchronous — suspension is
 * the generators' job, not the reporter's.
 */
export interface SolverHooks {
  wantSnapshot(step: number): boolean;
  emit(e: SolverProgressEmit): void;
}

/**
 * Static phase → [start, end] fraction spans per solver. A phase with known
 * totalSteps interpolates linearly across its span; a single-shot span
 * (start === end) pins the bar while an indeterminate stage runs; a phase
 * with no span reports null (indeterminate). `finalize` is always 1.
 */
const PHASE_SPANS: Record<FitSolverId, Partial<Record<FitPhase, readonly [number, number]>>> = {
  full: {},
  covariance_eigh: { gram: [0, 0.85], decompose: [0.85, 0.85] },
  arpack: {},
  randomized: { 'power-iteration': [0, 0.9] },
  incremental: { batch: [0, 1] },
};

function fractionFor(
  solver: FitSolverId,
  phase: FitPhase,
  step: number,
  totalSteps: number | null,
): number | null {
  if (phase === 'finalize') {
    return 1;
  }
  const span = PHASE_SPANS[solver][phase];
  if (span === undefined) {
    return null;
  }
  const [lo, hi] = span;
  if (lo === hi) {
    return lo;
  }
  if (totalSteps === null || totalSteps <= 0) {
    return null;
  }
  const t = Math.min(1, Math.max(0, step / totalSteps));
  return lo + (hi - lo) * t;
}

/** @internal Fresh float64 copy of a possibly-float32 stored attribute. */
export function toFloat64Copy(a: FloatArray): Float64Array {
  return a instanceof Float64Array ? a.slice() : new Float64Array(a);
}

/**
 * @internal Float64 embedding of the first `n` rows of `x` under a possibly
 * intermediate model — `transform`'s math without the dtype casting:
 * project, remove the projected mean, and whiten with an eps clip.
 */
export function projectForSnapshot(
  x: FloatArray,
  n: number,
  p: number,
  components: FloatArray,
  k: number,
  mean: FloatArray,
  explainedVariance: FloatArray,
  whiten: boolean,
  evEps: number,
): Matrix {
  const xt = matmulTransB(x, components, n, p, k);
  const meanProj = new Float64Array(k);
  for (let c = 0; c < k; c++) {
    let acc = 0;
    const off = c * p;
    for (let j = 0; j < p; j++) {
      acc += mean[j] * components[off + j];
    }
    meanProj[c] = acc;
  }
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < k; c++) {
      xt[i * k + c] -= meanProj[c];
    }
  }
  if (whiten) {
    for (let c = 0; c < k; c++) {
      let scale = Math.sqrt(explainedVariance[c]);
      if (scale < evEps) {
        scale = evEps;
      }
      for (let i = 0; i < n; i++) {
        xt[i * k + c] /= scale;
      }
    }
  }
  return new Matrix(xt, n, k);
}

/** @internal Context a reporter needs to shape events and snapshots. */
export interface ReporterContext {
  estimator: 'PCA' | 'IncrementalPCA';
  solver: FitSolverId;
  /** Rows of the training matrix (variance scaling, score shapes). */
  nRows: number;
  whiten: boolean;
}

/**
 * @internal Turns solver checkpoints into observer events: computes the
 * overall fraction from the per-solver phase table and materializes
 * snapshots from fresh solver triplets. Callback exceptions propagate —
 * they abort the fit by design.
 */
export class ProgressReporter implements SolverHooks {
  private readonly onProgress: (event: PCAFitProgress) => void;
  private readonly wantComponents: boolean;
  private readonly wantScores: boolean;
  private readonly every: number;
  private readonly ctx: ReporterContext;

  constructor(
    onProgress: (event: PCAFitProgress) => void,
    snapshot: SnapshotOptions | undefined,
    ctx: ReporterContext,
  ) {
    this.onProgress = onProgress;
    this.wantScores = snapshot?.scores === true;
    this.wantComponents = snapshot?.components === true || this.wantScores;
    const every = snapshot?.every ?? 1;
    this.every = Number.isInteger(every) && every >= 1 ? every : 1;
    this.ctx = ctx;
  }

  /** True when any event may carry a snapshot. */
  get snapshotsEnabled(): boolean {
    return this.wantComponents;
  }

  /** True when snapshots should include `scores`. */
  get scoresRequested(): boolean {
    return this.wantScores;
  }

  wantSnapshot(step: number): boolean {
    return this.wantComponents && step >= 1 && step % this.every === 0;
  }

  emit(e: SolverProgressEmit): void {
    const event: PCAFitProgress = {
      estimator: this.ctx.estimator,
      solver: this.ctx.solver,
      phase: e.phase,
      step: e.step,
      totalSteps: e.totalSteps,
      fraction: fractionFor(this.ctx.solver, e.phase, e.step, e.totalSteps),
    };
    const snapshot = e.snapshot ?? (e.dec ? this.snapshotFromDec(e.dec) : undefined);
    if (snapshot !== undefined) {
      event.snapshot = snapshot;
    }
    if (e.detail !== undefined) {
      event.detail = e.detail;
    }
    this.onProgress(event);
  }

  /**
   * Builds a snapshot from fresh solver triplets, adopting their arrays:
   * applies the deterministic sign convention in place, then derives the
   * variances and (when requested) the U·S scores.
   */
  private snapshotFromDec(dec: SvdResult): PCAFitSnapshot {
    const n = this.ctx.nRows;
    const k = dec.s.length;
    const p = dec.vt.length / k;
    svdFlipVBased(dec.u, n, dec.vt, k, p);
    const explainedVariance = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      explainedVariance[i] = (dec.s[i] * dec.s[i]) / (n - 1);
    }
    const snapshot: PCAFitSnapshot = {
      components: new Matrix(dec.vt, k, p),
      singularValues: dec.s,
      explainedVariance,
    };
    if (this.wantScores) {
      const scores = new Float64Array(n * k);
      const f = Math.sqrt(n - 1);
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          const u = dec.u[i * k + c];
          scores[i * k + c] = this.ctx.whiten ? u * f : u * dec.s[c];
        }
      }
      snapshot.scores = new Matrix(scores, n, k);
    }
    return snapshot;
  }
}

/**
 * @internal Constructs a reporter only when there is a callback to feed —
 * a null reporter means solvers skip all snapshot work.
 */
export function makeReporter(
  observer: FitObserver | undefined,
  ctx: ReporterContext,
): ProgressReporter | null {
  const onProgress = observer?.onProgress;
  return onProgress ? new ProgressReporter(onProgress, observer?.snapshot, ctx) : null;
}
