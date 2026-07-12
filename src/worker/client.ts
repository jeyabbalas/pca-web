/**
 * `pca-web/client` — main-thread async proxies for estimators running in a
 * pca-web worker: `WorkerPCA` and `WorkerIncrementalPCA`.
 *
 * Import discipline (enforced by a dist-graph test): this module pulls in
 * only matrix/types/validation/model/protocol — never estimator or solver
 * code — so apps that fit exclusively in the worker don't bundle the
 * solvers twice.
 *
 * Fitted attributes are mirrored synchronously on the client: every fit
 * piggybacks the resulting model on its response, so `components`,
 * `explainedVariance`, `exportModel()`, … work without a round-trip once
 * the fit promise resolves.
 */
import { asMatrix, Matrix, type MatrixInput } from '../matrix.js';
import {
  type AnyPCAModel,
  assertValidModel,
  type IncrementalPCAModel,
  type PCAModel,
} from '../model.js';
import type { PCAFitProgress, PCAFitSnapshot } from '../progress.js';
import type { AbortSignalLike } from '../scheduling.js';
import type { FloatArray } from '../types.js';
import { NotFittedError } from '../validation.js';
import {
  type CallRequest,
  isWorkerResponse,
  matrixToWire,
  type PCAWorkerLike,
  type ProgressOptions,
  type QueueableRequest,
  type WireError,
  type WireIPCAOptions,
  type WireMatrix,
  type WirePCAOptions,
  type WireProgress,
  type WireValue,
  type WorkerEstimatorInfo,
  type WorkerFactory,
  type WorkerMethod,
  wireToMatrix,
} from './protocol.js';

declare global {
  // Merged, never conflicting: DOM and Node declare the identical member.
  interface ImportMeta {
    url: string;
  }
}

// Module-scoped ambient declarations (the build has lib ES2022 only, no
// DOM). They emit nothing; the references below resolve to the host
// globals at runtime, and the literal `new Worker(new URL(...))` syntax
// stays intact for bundlers' static worker detection.
declare const Worker:
  | (new (
      url: unknown,
      options?: { type?: string; name?: string },
    ) => PCAWorkerLike)
  | undefined;
declare const URL: new (url: string, base?: string) => unknown;
declare const structuredClone: <T>(value: T) => T;

/** Thrown into in-flight calls when `terminate()` hard-kills the worker. */
export class WorkerTerminatedError extends Error {
  override readonly name = 'WorkerTerminatedError';

  constructor(message = 'The pca-web worker was terminated') {
    super(message);
  }
}

/** AbortError stand-in for hosts without DOMException. */
class ClientAbortError extends Error {
  override readonly name = 'AbortError';

  constructor(message = 'The operation was aborted') {
    super(message);
  }
}

function makeAbortError(message?: string): Error {
  const DE = (globalThis as { DOMException?: new (message?: string, name?: string) => Error })
    .DOMException;
  return DE !== undefined
    ? new DE(message ?? 'The operation was aborted', 'AbortError')
    : new ClientAbortError(message);
}

/** Rebuilds a worker-marshalled error, instanceof-correct where it matters. */
function reviveError(wire: WireError): Error {
  if (wire.name === 'NotFittedError') {
    const err = new NotFittedError('PCA');
    err.message = wire.message;
    return err;
  }
  if (wire.name === 'AbortError') {
    return makeAbortError(wire.message);
  }
  const err = new Error(wire.message);
  err.name = wire.name;
  return err;
}

/** Client-side abort wiring accepts a real AbortSignal when events are needed. */
interface EventfulAbortSignal extends AbortSignalLike {
  addEventListener?: (type: 'abort', listener: () => void, opts?: { once?: boolean }) => void;
  removeEventListener?: (type: 'abort', listener: () => void) => void;
}

/** Per-fit options for the worker proxies. */
export interface WorkerFitOptions {
  signal?: EventfulAbortSignal;
  /**
   * Transfer the input's buffer to the worker instead of copying (the
   * caller's array becomes unusable). Views are tight-sliced first, so a
   * subarray's parent buffer is never touched either way.
   */
  transfer?: boolean;
  onProgress?: (event: PCAFitProgress) => void;
  progress?: ProgressOptions;
}

/** Options shared by both proxies' constructors. */
export interface WorkerClientOptions {
  /**
   * The worker (or MessagePort, or Node worker_threads port) to run on, or
   * a factory producing one. Clients given the same port instance share
   * one connection. Omit to spawn the packaged default worker — see the
   * README's bundler recipes if your bundler cannot resolve it.
   */
  worker?: PCAWorkerLike | WorkerFactory;
  /** Default progress callback for all fits (per-call options override). */
  onProgress?: (event: PCAFitProgress) => void;
  /** Default progress tuning (throttle interval, snapshots). */
  progress?: ProgressOptions;
}

export interface WorkerPCAOptions extends WirePCAOptions, WorkerClientOptions {
  /** Where fits run: 'cpu' (default) or 'webgpu' (falls back to CPU). */
  backend?: 'cpu' | 'webgpu';
}

export interface WorkerIncrementalPCAOptions extends WireIPCAOptions, WorkerClientOptions {}

export type { WorkerEstimatorInfo } from './protocol.js';

// ---------------------------------------------------------------------
// Connection: request-id dispatch shared per worker instance
// ---------------------------------------------------------------------

interface PendingEntry {
  resolve: (r: { value: WireValue; model?: AnyPCAModel }) => void;
  reject: (err: Error) => void;
  onProgress?: (e: WireProgress) => void;
}

class Connection {
  readonly port: PCAWorkerLike;
  private readonly pending = new Map<number, PendingEntry>();
  private nextId = 1;
  private readyResolve: (() => void) | null = null;
  private readonly readyPromise: Promise<void>;
  private dead: Error | null = null;

  constructor(port: PCAWorkerLike) {
    this.port = port;
    this.readyPromise = new Promise((resolve) => {
      this.readyResolve = resolve;
    });
    port.addEventListener('message', this.listener);
    port.start?.();
  }

  private readonly listener = (ev: { data: unknown }): void => {
    const msg = ev.data;
    if (!isWorkerResponse(msg)) {
      return;
    }
    if (msg.t === 'ready') {
      this.readyResolve?.();
      this.readyResolve = null;
      return;
    }
    if (msg.t === 'progress') {
      const entry = this.pending.get(msg.id);
      if (entry?.onProgress !== undefined) {
        try {
          entry.onProgress(msg.event);
        } catch (err) {
          // A throwing user callback must not break dispatch; mirror the
          // main-thread contract (callback throw kills the fit) instead.
          this.pending.delete(msg.id);
          this.post({ t: 'abort', targetId: msg.id });
          entry.reject(err instanceof Error ? err : new Error(String(err)));
        }
      }
      return;
    }
    const entry = this.pending.get(msg.id);
    if (entry === undefined) {
      return;
    }
    this.pending.delete(msg.id);
    if (msg.t === 'result') {
      entry.resolve({ value: msg.value, model: msg.model });
    } else {
      entry.reject(reviveError(msg.error));
    }
  };

  allocId(): number {
    return this.nextId++;
  }

  ready(): Promise<void> {
    return this.dead !== null ? Promise.reject(this.dead) : this.readyPromise;
  }

  post(msg: { t: 'abort'; targetId: number }): void {
    if (this.dead === null) {
      this.port.postMessage(msg, []);
    }
  }

  request(
    msg: QueueableRequest,
    transfer: ArrayBufferLike[] = [],
    onProgress?: (e: WireProgress) => void,
  ): Promise<{ value: WireValue; model?: AnyPCAModel }> {
    if (this.dead !== null) {
      return Promise.reject(this.dead);
    }
    return new Promise((resolve, reject) => {
      this.pending.set(msg.id, { resolve, reject, onProgress });
      this.port.postMessage(msg, transfer);
    });
  }

  /** Hard-kill: rejects all in-flight calls and poisons future ones. */
  terminate(): void {
    if (this.dead !== null) {
      return;
    }
    this.dead = new WorkerTerminatedError();
    this.port.removeEventListener('message', this.listener);
    for (const entry of this.pending.values()) {
      entry.reject(new WorkerTerminatedError());
    }
    this.pending.clear();
    this.port.terminate?.();
    this.port.close?.();
  }
}

const CONNECTIONS = new WeakMap<PCAWorkerLike, Connection>();

function connectionFor(port: PCAWorkerLike): Connection {
  let conn = CONNECTIONS.get(port);
  if (conn === undefined) {
    conn = new Connection(port);
    CONNECTIONS.set(port, conn);
  }
  return conn;
}

function defaultWorkerFactory(): PCAWorkerLike {
  const explain = (reason: string): Error =>
    new Error(
      `pca-web/client: could not start the default worker (${reason}). ` +
        'Pass your own via { worker }: ' +
        "new Worker(new URL('pca-web/worker', import.meta.url), { type: 'module' }) works in " +
        "webpack 5 and native ESM; Vite dev needs `import PcaWorker from 'pca-web/worker?worker'` " +
        "(then { worker: () => new PcaWorker() }) or 'pca-web' in optimizeDeps.exclude; esbuild " +
        'needs pca-web/worker bundled as its own entry; in Node use worker_threads and ' +
        "attachPCAWorker(parentPort) from 'pca-web/worker'.",
    );
  if (typeof Worker !== 'function') {
    throw explain('no Worker constructor in this environment');
  }
  try {
    return new Worker(new URL('./worker.js', import.meta.url), {
      type: 'module',
      name: 'pca-web',
    });
  } catch (err) {
    throw explain(String(err));
  }
}

function wireToProgress(e: WireProgress): PCAFitProgress {
  const out: PCAFitProgress = {
    estimator: e.estimator,
    solver: e.solver,
    phase: e.phase,
    step: e.step,
    totalSteps: e.totalSteps,
    fraction: e.fraction,
  };
  if (e.detail !== undefined) {
    out.detail = e.detail;
  }
  if (e.snapshot !== undefined) {
    const snapshot: PCAFitSnapshot = {
      components: wireToMatrix(e.snapshot.components),
      singularValues: e.snapshot.singularValues,
      explainedVariance: e.snapshot.explainedVariance,
    };
    if (e.snapshot.scores !== undefined) {
      snapshot.scores = wireToMatrix(e.snapshot.scores);
    }
    out.snapshot = snapshot;
  }
  return out;
}

// ---------------------------------------------------------------------
// Shared proxy base
// ---------------------------------------------------------------------

let estCounter = 0;

abstract class WorkerEstimatorBase<M extends AnyPCAModel> {
  protected readonly conn: Connection;
  protected readonly estId: string;
  protected readonly ownsWorker: boolean;
  protected readonly defaultOnProgress?: (event: PCAFitProgress) => void;
  protected readonly defaultProgress?: ProgressOptions;
  protected model_: M | null = null;
  protected whitenOpt: boolean;
  private readonly createDone: Promise<unknown>;
  private disposed = false;

  protected constructor(
    estimator: 'pca' | 'ipca',
    backend: 'cpu' | 'webgpu',
    wireOptions: WirePCAOptions | WireIPCAOptions,
    client: WorkerClientOptions,
    hydrate?: M,
  ) {
    const rs = (wireOptions as WirePCAOptions).randomState;
    if (rs !== undefined && rs !== null && typeof rs !== 'number') {
      throw new Error(
        'WorkerPCA: randomState must be a numeric seed (a live RandomState cannot cross the worker boundary)',
      );
    }
    this.ownsWorker = client.worker === undefined || typeof client.worker === 'function';
    const port =
      client.worker === undefined
        ? defaultWorkerFactory()
        : typeof client.worker === 'function'
          ? client.worker()
          : client.worker;
    this.conn = connectionFor(port);
    this.defaultOnProgress = client.onProgress;
    this.defaultProgress = client.progress;
    this.whitenOpt = (wireOptions as WirePCAOptions).whiten ?? hydrate?.whiten ?? false;
    this.estId = `est-${++estCounter}`;
    const create: QueueableRequest = {
      t: 'create',
      id: this.conn.allocId(),
      est: this.estId,
      estimator,
      backend,
      options: wireOptions,
    };
    if (hydrate !== undefined) {
      create.model = hydrate;
      this.model_ = structuredClone(hydrate);
    }
    // Pipelined: sent immediately, never awaited here. Failures resurface
    // on ready() and on every subsequent call for this estimator id.
    this.createDone = this.conn.request(create);
    this.createDone.catch(() => {});
  }

  /** Resolves when the worker handshake and this estimator's create are done. */
  async ready(): Promise<void> {
    await this.conn.ready();
    await this.createDone;
  }

  /** Graceful shutdown of this estimator (terminates owned workers too). */
  async dispose(): Promise<void> {
    if (this.disposed) {
      return;
    }
    this.disposed = true;
    try {
      await this.conn.request({ t: 'dispose', id: this.conn.allocId(), est: this.estId });
    } finally {
      if (this.ownsWorker) {
        this.conn.terminate();
      }
    }
  }

  /**
   * Hard-kill the worker: all in-flight calls reject and subsequent calls
   * keep rejecting with WorkerTerminatedError.
   */
  terminate(): void {
    this.conn.terminate();
  }

  protected mirror(): M {
    if (this.model_ === null) {
      throw new NotFittedError('PCA');
    }
    return this.model_;
  }

  protected async call(
    method: WorkerMethod,
    payload: { x?: WireMatrix; model?: AnyPCAModel; returnModel?: boolean },
    transfer: ArrayBufferLike[] = [],
    fit?: WorkerFitOptions,
  ): Promise<WireValue> {
    if (this.disposed) {
      throw new Error('This worker estimator was disposed');
    }
    const signal = fit?.signal;
    if (signal?.aborted) {
      throw signal.reason ?? makeAbortError();
    }
    const id = this.conn.allocId();
    const msg: CallRequest = { t: 'call', id, est: this.estId, method, ...payload };
    const onProgress = fit?.onProgress ?? this.defaultOnProgress;
    const progressOptions = fit?.progress ?? this.defaultProgress;
    let progressSink: ((e: WireProgress) => void) | undefined;
    if (onProgress !== undefined || progressOptions !== undefined) {
      msg.progress = progressOptions ?? {};
      if (onProgress !== undefined) {
        progressSink = (e) => onProgress(wireToProgress(e));
      }
    }
    let onAbort: (() => void) | null = null;
    if (signal?.addEventListener !== undefined) {
      onAbort = () => this.conn.post({ t: 'abort', targetId: id });
      signal.addEventListener('abort', onAbort, { once: true });
    }
    try {
      const { value, model } = await this.conn.request(msg, transfer, progressSink);
      if (model !== undefined) {
        this.model_ = model as M;
        this.whitenOpt = model.whiten;
      }
      return value;
    } catch (err) {
      // Prefer the caller's abort reason over the marshalled AbortError.
      if (signal?.aborted && signal.reason !== undefined && signal.reason !== null) {
        throw signal.reason;
      }
      throw err;
    } finally {
      if (onAbort !== null) {
        signal?.removeEventListener?.('abort', onAbort);
      }
    }
  }

  protected wireX(
    X: MatrixInput,
    fit?: WorkerFitOptions,
  ): { x: WireMatrix; transfer: ArrayBufferLike[] } {
    const wire = matrixToWire(asMatrix(X));
    return { x: wire, transfer: fit?.transfer === true ? [wire.data.buffer] : [] };
  }

  protected async callMatrix(method: WorkerMethod, X?: MatrixInput): Promise<Matrix> {
    const payload: { x?: WireMatrix } = X === undefined ? {} : { x: matrixToWire(asMatrix(X)) };
    const value = await this.call(method, payload, []);
    return wireToMatrix(value as WireMatrix);
  }

  // ---- synchronous mirror (hydrated from the piggybacked model) ----

  get components(): Matrix {
    const m = this.mirror();
    return new Matrix(m.components, m.nComponents, m.nFeaturesIn);
  }

  get explainedVariance(): FloatArray {
    return this.mirror().explainedVariance;
  }

  get explainedVarianceRatio(): FloatArray {
    return this.mirror().explainedVarianceRatio;
  }

  get singularValues(): FloatArray {
    return this.mirror().singularValues;
  }

  get mean(): FloatArray {
    return this.mirror().mean;
  }

  get nComponents(): number {
    return this.mirror().nComponents;
  }

  get nFeaturesIn(): number {
    return this.mirror().nFeaturesIn;
  }

  get noiseVariance(): number {
    return this.mirror().noiseVariance;
  }

  get whiten(): boolean {
    return this.model_ !== null ? this.model_.whiten : this.whitenOpt;
  }

  getFeatureNamesOut(): string[] {
    const m = this.mirror();
    const prefix = m.estimator === 'ipca' ? 'incrementalpca' : 'pca';
    const names = new Array<string>(m.nComponents);
    for (let i = 0; i < m.nComponents; i++) {
      names[i] = `${prefix}${i}`;
    }
    return names;
  }

  /** The current model as an independent copy (sync, from the mirror). */
  exportModel(): M {
    return structuredClone(this.mirror());
  }

  /** Rehydrates worker and mirror from a serialized model. */
  async importModel(model: M): Promise<this> {
    await this.call('importModel', { model });
    this.model_ = structuredClone(model);
    this.whitenOpt = model.whiten;
    return this;
  }

  /** Which backend actually executes, plus adapter details. */
  async info(): Promise<WorkerEstimatorInfo> {
    return (await this.call('info', {})) as WorkerEstimatorInfo;
  }

  // ---- async delegated methods ----

  async transform(X: MatrixInput): Promise<Matrix> {
    return this.callMatrix('transform', X);
  }

  async inverseTransform(X: MatrixInput): Promise<Matrix> {
    return this.callMatrix('inverseTransform', X);
  }

  async getCovariance(): Promise<Matrix> {
    return this.callMatrix('getCovariance');
  }

  async getPrecision(): Promise<Matrix> {
    return this.callMatrix('getPrecision');
  }
}

// ---------------------------------------------------------------------
// Public proxies
// ---------------------------------------------------------------------

export class WorkerPCA extends WorkerEstimatorBase<PCAModel> {
  constructor(options: WorkerPCAOptions = {}, /** @internal use fromModel */ hydrate?: PCAModel) {
    const { worker, onProgress, progress, backend, ...wire } = options;
    super('pca', backend ?? 'cpu', wire, { worker, onProgress, progress }, hydrate);
  }

  /**
   * Creates a proxy already hydrated with a fitted model — the worker
   * adopts it (pipelined with create) and the sync mirror is available
   * immediately.
   */
  static fromModel(model: PCAModel, options: WorkerPCAOptions = {}): WorkerPCA {
    assertValidModel(model, 'pca');
    return new WorkerPCA(options, model);
  }

  async fit(X: MatrixInput, options: WorkerFitOptions = {}): Promise<this> {
    const { x, transfer } = this.wireX(X, options);
    await this.call('fit', { x, returnModel: true }, transfer, options);
    return this;
  }

  async fitTransform(X: MatrixInput, options: WorkerFitOptions = {}): Promise<Matrix> {
    const { x, transfer } = this.wireX(X, options);
    const value = await this.call('fitTransform', { x, returnModel: true }, transfer, options);
    return wireToMatrix(value as WireMatrix);
  }

  async scoreSamples(X: MatrixInput): Promise<FloatArray> {
    const { x } = this.wireX(X);
    const value = await this.call('scoreSamples', { x });
    return (value as { kind: 'array'; data: FloatArray }).data;
  }

  async score(X: MatrixInput): Promise<number> {
    const { x } = this.wireX(X);
    return (await this.call('score', { x })) as number;
  }

  get nSamples(): number {
    return this.mirror().nSamples;
  }

  /** The solver the last fit actually used. */
  get resolvedSvdSolver(): PCAModel['svdSolver'] {
    return this.mirror().svdSolver;
  }
}

export class WorkerIncrementalPCA extends WorkerEstimatorBase<IncrementalPCAModel> {
  constructor(
    options: WorkerIncrementalPCAOptions = {},
    /** @internal use fromModel */ hydrate?: IncrementalPCAModel,
  ) {
    const { worker, onProgress, progress, ...wire } = options;
    super('ipca', 'cpu', wire, { worker, onProgress, progress }, hydrate);
  }

  static fromModel(
    model: IncrementalPCAModel,
    options: WorkerIncrementalPCAOptions = {},
  ): WorkerIncrementalPCA {
    assertValidModel(model, 'ipca');
    return new WorkerIncrementalPCA(options, model);
  }

  async fit(X: MatrixInput, options: WorkerFitOptions = {}): Promise<this> {
    const { x, transfer } = this.wireX(X, options);
    await this.call('fit', { x, returnModel: true }, transfer, options);
    return this;
  }

  async fitTransform(X: MatrixInput, options: WorkerFitOptions = {}): Promise<Matrix> {
    const { x, transfer } = this.wireX(X, options);
    const value = await this.call('fitTransform', { x, returnModel: true }, transfer, options);
    return wireToMatrix(value as WireMatrix);
  }

  /** One incremental batch — sklearn's partial_fit, off-thread. */
  async partialFit(X: MatrixInput, options: WorkerFitOptions = {}): Promise<this> {
    const { x, transfer } = this.wireX(X, options);
    await this.call('partialFit', { x, returnModel: true }, transfer, options);
    return this;
  }

  get variance(): Float64Array {
    return this.mirror().variance;
  }

  get nSamplesSeen(): number {
    return this.mirror().nSamplesSeen;
  }

  get batchSize(): number {
    const b = this.mirror().batchSize;
    if (b === null) {
      throw new Error("batchSize is only available after fit() (sklearn's batch_size_)");
    }
    return b;
  }
}

export type {
  PCAWorkerLike,
  ProgressOptions,
  WireMatrix,
  WorkerFactory,
} from './protocol.js';
