/**
 * WebGPU-accelerated PCA with a numerically-equivalent CPU fallback.
 *
 * The GPU takes over exactly the large-GEMM hotspots — the XᵀX Gram product
 * of covariance_eigh, the power-iteration panel products of the randomized
 * solver (X stays resident on the device across iterations), and large
 * transform projections. Everything else (eigh, panel QR/LU, the small SVD,
 * sign flips, variance bookkeeping, validation) runs through the same code
 * as the CPU `PCA` class via internal bridges, so semantics cannot drift.
 *
 * When WebGPU is unavailable — or the fit is too small to benefit, or the
 * solver is inherently sequential (full, arpack) — fit() delegates to the
 * CPU implementation and produces bit-identical results. When the GPU is
 * used, results match the CPU path within the df64 GEMM accuracy measured
 * by the equivalence test suite (~1e-12 relative on components).
 */
import { castTo } from '../base.js';
import { asMatrix, Matrix, type MatrixInput } from '../matrix.js';
import type { PCAModel } from '../model.js';
import { randomizedSvdSteps } from '../numeric/randomized.js';
import { checkRandomState } from '../numeric/rng.js';
import { centerInPlace, colMeans } from '../numeric/stats.js';
import type { SvdResult } from '../numeric/svd.js';
import {
  PCA,
  type PCAOptions,
  resolveSvdSolver,
  type SvdSolver,
  validateNcForSolver,
} from '../pca.js';
import { type FitAsyncOptions, makeReporter } from '../progress.js';
import { throwIfAborted } from '../scheduling.js';
import type { FloatArray } from '../types.js';
import { assertAllFinite, assertMinSamplesForFit, checkFeatureCount } from '../validation.js';
import { GpuEngine, isWebGPUSupported, type WebGPUDeviceOptions } from './engine.js';

export interface WebGPUPCAOptions extends PCAOptions, WebGPUDeviceOptions {
  /**
   * Minimum element count (rows × cols) before the GPU path is used;
   * smaller problems run on the CPU, where they are faster anyway.
   * Default 1<<18 (≈262k elements).
   */
  minGpuElements?: number;
}

/** Wraps a GPU failure that must not trigger a CPU refit (input was mutated). */
class GpuFitUnrecoverableError extends Error {
  override readonly cause: unknown;
  constructor(cause: unknown) {
    super('WebGPU fit failed after in-place mutation');
    this.cause = cause;
  }
}

export class WebGPUPCA {
  private cpu: PCA;
  private readonly deviceOptions: WebGPUDeviceOptions;
  private readonly minGpuElements: number;
  private enginePromise: Promise<GpuEngine | null> | null = null;
  private engine: GpuEngine | null = null;
  private backend_: 'webgpu' | 'cpu' = 'cpu';
  private gpuFitting = false;

  constructor(options: WebGPUPCAOptions = {}) {
    const { device, powerPreference, minGpuElements, ...pcaOptions } = options;
    this.cpu = new PCA(pcaOptions);
    this.deviceOptions = { device, powerPreference };
    this.minGpuElements = minGpuElements ?? 1 << 18;
  }

  /** True when a WebGPU implementation is present in this environment. */
  static isSupported(): boolean {
    return isWebGPUSupported();
  }

  /** 'webgpu' when the last fit/transform used the GPU, else 'cpu'. */
  get backend(): 'webgpu' | 'cpu' {
    return this.backend_;
  }

  /** Adapter description of the engine in use, or null on the CPU fallback. */
  get gpuAdapterInfo(): string | null {
    return this.engine?.adapterInfo ?? null;
  }

  /** Measured df64 GEMM relative error from the device self-check (NaN on CPU). */
  get gpuMeasuredRelError(): number {
    return this.engine?.measuredRelError ?? Number.NaN;
  }

  /** Releases GPU resources (does not destroy an injected device). */
  dispose(): void {
    this.engine?.dispose();
    this.engine = null;
    this.enginePromise = null;
  }

  // ------------------------------------------------------------------
  // Fitting
  // ------------------------------------------------------------------

  async fit(X: MatrixInput, options: FitAsyncOptions = {}): Promise<this> {
    const xm = asMatrix(X);
    this.guardConcurrentFit();
    try {
      const route = await this.route(xm);
      if (route === 'cpu') {
        // The fallback is non-blocking, observable, and abortable too —
        // and still bit-identical to the plain PCA class.
        await this.cpu.fitAsync(xm, options);
        this.backend_ = 'cpu';
        return this;
      }
      try {
        if (route === 'covariance_eigh') {
          await this.fitGramGpu(xm, options);
        } else {
          await this.fitRandomizedGpu(xm, options);
        }
      } catch (err) {
        await this.recoverOnGpuError(xm, err, options);
      }
      return this;
    } finally {
      this.gpuFitting = false;
    }
  }

  async fitTransform(X: MatrixInput, options: FitAsyncOptions = {}): Promise<Matrix> {
    const xm = asMatrix(X);
    this.guardConcurrentFit();
    try {
      const route = await this.route(xm);
      if (route === 'cpu') {
        this.backend_ = 'cpu';
        return await this.cpu.fitTransformAsync(xm, options);
      }
      if (route === 'covariance_eigh') {
        try {
          await this.fitGramGpu(xm, options);
        } catch (err) {
          await this.recoverOnGpuError(xm, err, options);
        }
        // sklearn's fit_transform for covariance_eigh IS transform-after-fit
        // (no U is computed at fit time), so this matches exactly.
        return await this.transform(xm);
      }
      let dec: SvdResult;
      try {
        dec = await this.fitRandomizedGpu(xm, options);
      } catch (err) {
        await this.recoverOnGpuError(xm, err, options, 'skipFit');
        return await this.cpu.fitTransformAsync(xm, options);
      }
      // Randomized fast path: X_new = U·S (or U·√(n−1) when whitening),
      // exactly like PCA.fitTransform.
      const n = xm.rows;
      const k = this.cpu.nComponents;
      const out = new Float64Array(n * k);
      const whiten = this.cpu.whiten;
      const f = Math.sqrt(n - 1);
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          out[i * k + c] = whiten ? dec.u[i * k + c] * f : dec.u[i * k + c] * dec.s[c];
        }
      }
      return new Matrix(castTo(out, xm.dtype), n, k);
    } finally {
      this.gpuFitting = false;
    }
  }

  /** Concurrent fits on one instance would interleave GPU state; throw early. */
  private guardConcurrentFit(): void {
    if (this.gpuFitting) {
      throw new Error(
        'This WebGPUPCA instance is already fitting; concurrent fits on one instance are not supported',
      );
    }
    this.gpuFitting = true;
  }

  /**
   * Validates the input (before any GPU spend or in-place mutation) and
   * decides which backend the fit runs on. Mutates nothing.
   */
  private async route(xm: Matrix): Promise<'cpu' | 'covariance_eigh' | 'randomized'> {
    if (xm.rows < 1 || xm.cols < 1) {
      throw new Error(
        `Found array with shape (${xm.rows}, ${xm.cols}); PCA requires at least 1 sample and 1 feature`,
      );
    }
    assertMinSamplesForFit(xm, 'PCA');
    assertAllFinite(xm, 'PCA.fit');
    const o = this.cpu._resolvedOptions();
    const solver = resolveSvdSolver(xm.rows, xm.cols, o.nComponents, o.svdSolver);
    const engine = await this.ensureEngine();
    if (!this.gpuEligible(engine, xm, solver)) {
      return 'cpu';
    }
    // Fail doomed configurations before uploading anything.
    const nc = o.nComponents === null ? Math.min(xm.rows, xm.cols) : o.nComponents;
    validateNcForSolver(nc, xm.rows, xm.cols, solver as 'covariance_eigh' | 'randomized');
    return solver as 'covariance_eigh' | 'randomized';
  }

  /** GPU Gram product + shared CPU covariance_eigh tail. Mutates nothing on failure. */
  private async fitGramGpu(xm: Matrix, options: FitAsyncOptions): Promise<void> {
    const eng = this.engine as GpuEngine;
    const reporter = makeReporter(options, {
      estimator: 'PCA',
      solver: 'covariance_eigh',
      nRows: xm.rows,
      whiten: this.cpu.whiten,
    });
    throwIfAborted(options.signal);
    const gx = eng.upload(xm.data, xm.rows, xm.cols);
    try {
      reporter?.emit({ phase: 'gram', step: 0, totalSteps: 1 });
      const gram = await eng.syrk(gx);
      reporter?.emit({ phase: 'gram', step: 1, totalSteps: 1 });
      throwIfAborted(options.signal);
      reporter?.emit({ phase: 'decompose', step: 0, totalSteps: null });
      this.cpu._fitGram(xm, gram);
      this.backend_ = 'webgpu';
      if (reporter) {
        this.cpu._emitFinalize(reporter, xm, null, 0, null);
      }
    } finally {
      gx.destroy();
    }
  }

  /**
   * GPU-driven randomized fit: centers on the CPU (respecting copy
   * semantics), keeps X resident on the device, and drives the shared
   * algorithm generator with GPU GEMMs. Returns the (sign-flipped)
   * decomposition for the fitTransform fast path.
   */
  private async fitRandomizedGpu(xm: Matrix, options: FitAsyncOptions): Promise<SvdResult> {
    const eng = this.engine as GpuEngine;
    const o = this.cpu._resolvedOptions();
    const n = xm.rows;
    const p = xm.cols;
    const nc = (o.nComponents === null ? Math.min(n, p) : o.nComponents) as number;
    const reporter = makeReporter(options, {
      estimator: 'PCA',
      solver: 'randomized',
      nRows: n,
      whiten: this.cpu.whiten,
    });
    throwIfAborted(options.signal);
    const meanF64 = colMeans(xm.data, n, p);
    const xc: FloatArray = o.copy ? xm.data.slice() : xm.data;
    centerInPlace(xc, n, p, meanF64);
    const gx = eng.upload(xc, n, p);
    try {
      const rng = checkRandomState(o.randomState);
      // The reporter rides the solver's hooks: power-iteration events and
      // snapshot decompositions surface as ordinary GEMM requests below, so
      // X stays device-resident throughout.
      const gen = randomizedSvdSteps(n, p, nc, {
        nOversamples: o.nOversamples,
        nIter: o.iteratedPower,
        powerIterationNormalizer: o.powerIterationNormalizer,
        rng,
        float32Stream: xm.dtype === 'float32',
        hooks: reporter,
      });
      let step = gen.next();
      while (!step.done) {
        throwIfAborted(options.signal);
        const req = step.value;
        const result =
          req.op === 'mulA'
            ? await eng.mulA(gx, req.b, req.w)
            : req.op === 'mulAT'
              ? await eng.mulAT(gx, req.b, req.w)
              : await eng.mulTA(gx, req.b, req.w);
        step = gen.next(result);
      }
      const dec: SvdResult = step.value;
      this.cpu._fitDecomposed(xm, xc, meanF64, dec);
      this.backend_ = 'webgpu';
      if (reporter) {
        this.cpu._emitFinalize(reporter, xm, dec.u, nc, dec.s);
      }
      return dec;
    } catch (err) {
      if (!o.copy) {
        // X was centered in place; a silent CPU retry would re-center the
        // mutated buffer and corrupt the fit. Surface the GPU failure.
        throw new GpuFitUnrecoverableError(err);
      }
      throw err;
    } finally {
      gx.destroy();
    }
  }

  /**
   * CPU refit after a GPU failure — unless the input was already mutated,
   * or the "failure" is a cancellation: an abort must reject with the abort
   * error, never silently refit on the CPU.
   */
  private async recoverOnGpuError(
    xm: Matrix,
    err: unknown,
    options: FitAsyncOptions,
    mode?: 'skipFit',
  ): Promise<void> {
    if (err instanceof GpuFitUnrecoverableError) {
      throw err.cause;
    }
    if (options.signal?.aborted || (err as { name?: unknown } | null)?.name === 'AbortError') {
      throw err;
    }
    this.backend_ = 'cpu';
    if (mode !== 'skipFit') {
      await this.cpu.fitAsync(xm, options);
    }
  }

  // ------------------------------------------------------------------
  // Transforms (GPU projection for large inputs, CPU otherwise)
  // ------------------------------------------------------------------

  async transform(X: MatrixInput): Promise<Matrix> {
    const xm = asMatrix(X);
    const engine = this.engine;
    const comp = this.cpu.components; // asserts fitted
    if (
      engine === null ||
      xm.rows * xm.cols < this.minGpuElements ||
      !engine.canFit(xm.rows, xm.cols) ||
      !engine.canFit(comp.rows, comp.cols)
    ) {
      return this.cpu.transform(xm);
    }
    assertAllFinite(xm, 'PCA.transform');
    checkFeatureCount(xm, this.cpu.nFeaturesIn, 'PCA');
    const gx = engine.upload(xm.data, xm.rows, xm.cols);
    const gc = engine.upload(comp.data, comp.rows, comp.cols);
    try {
      const proj = await engine.project(gx, gc);
      return this.cpu._transformFromProjection(proj, xm.rows, xm.dtype, false);
    } catch {
      return this.cpu.transform(xm);
    } finally {
      gx.destroy();
      gc.destroy();
    }
  }

  /** CPU delegate (small k×p GEMM; not worth a device round-trip). */
  inverseTransform(X: MatrixInput): Matrix {
    return this.cpu.inverseTransform(X);
  }

  // ------------------------------------------------------------------
  // Fitted state and CPU-delegated methods
  // ------------------------------------------------------------------

  get components(): Matrix {
    return this.cpu.components;
  }
  get explainedVariance(): FloatArray {
    return this.cpu.explainedVariance;
  }
  get explainedVarianceRatio(): FloatArray {
    return this.cpu.explainedVarianceRatio;
  }
  get singularValues(): FloatArray {
    return this.cpu.singularValues;
  }
  get mean(): FloatArray {
    return this.cpu.mean;
  }
  get nComponents(): number {
    return this.cpu.nComponents;
  }
  get nSamples(): number {
    return this.cpu.nSamples;
  }
  get nFeaturesIn(): number {
    return this.cpu.nFeaturesIn;
  }
  get noiseVariance(): number {
    return this.cpu.noiseVariance;
  }
  get whiten(): boolean {
    return this.cpu.whiten;
  }
  get resolvedSvdSolver(): Exclude<SvdSolver, 'auto'> {
    return this.cpu.resolvedSvdSolver;
  }

  getCovariance(): Matrix {
    return this.cpu.getCovariance();
  }
  getPrecision(): Matrix {
    return this.cpu.getPrecision();
  }
  scoreSamples(X: MatrixInput): FloatArray {
    return this.cpu.scoreSamples(X);
  }
  score(X: MatrixInput): number {
    return this.cpu.score(X);
  }
  getFeatureNamesOut(): string[] {
    return this.cpu.getFeatureNamesOut();
  }

  // ------------------------------------------------------------------
  // Model serialization
  // ------------------------------------------------------------------

  /** The fitted model as a plain object — identical to the CPU class's. */
  toModel(): PCAModel {
    return this.cpu.toModel();
  }

  /**
   * Rehydrates a fitted WebGPUPCA. `options` supplies the device knobs
   * (powerPreference, injected device, minGpuElements); the PCA options
   * come from the model itself.
   */
  static fromModel(
    model: PCAModel,
    options: WebGPUDeviceOptions & { minGpuElements?: number } = {},
  ): WebGPUPCA {
    const gpu = new WebGPUPCA({ ...model.options, ...options });
    gpu.cpu = PCA.fromModel(model);
    return gpu;
  }

  // ------------------------------------------------------------------

  private async ensureEngine(): Promise<GpuEngine | null> {
    if (this.enginePromise === null) {
      this.enginePromise = GpuEngine.create(this.deviceOptions).then((e) => {
        this.engine = e;
        return e;
      });
    }
    return this.enginePromise;
  }

  private gpuEligible(engine: GpuEngine | null, xm: Matrix, solver: string): boolean {
    return (
      engine !== null &&
      (solver === 'covariance_eigh' || solver === 'randomized') &&
      xm.rows * xm.cols >= this.minGpuElements &&
      engine.canFit(xm.rows, xm.cols) &&
      engine.canFit(xm.cols, xm.cols)
    );
  }
}
