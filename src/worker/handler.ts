/**
 * Worker-side request handler: a multi-instance estimator registry with a
 * strict global FIFO execution queue. Fits run through `fitAsync` — the
 * yield points are what let abort messages (handled out-of-band, straight
 * from the listener) take effect mid-fit; a synchronous fit would block the
 * worker's queue and make cancellation impossible.
 *
 * The WebGPU backend is loaded through a cached dynamic import only when an
 * estimator requests it, so CPU-only worker bundles stay lean.
 */
import { IncrementalPCA } from '../incremental-pca.js';
import type { Matrix } from '../matrix.js';
import type { AnyPCAModel, IncrementalPCAModel, PCAModel } from '../model.js';
import { PCA, type PCAOptions } from '../pca.js';
import type { FitAsyncOptions, PCAFitProgress } from '../progress.js';
import type { WebGPUPCA } from '../webgpu/gpu-pca.js';
import {
  type CallRequest,
  type CreateRequest,
  isWorkerRequest,
  matrixToWire,
  modelTransferList,
  PCA_WORKER_PROTOCOL_VERSION,
  type PCAWorkerPort,
  type PCAWorkerResponse,
  type QueueableRequest,
  tightArray,
  toWireError,
  type WireError,
  type WireIPCAOptions,
  type WirePCAOptions,
  type WireProgress,
  type WireSnapshot,
  type WireValue,
  type WorkerEstimatorInfo,
  wireToMatrix,
} from './protocol.js';

export interface AttachPCAWorkerOptions {
  /**
   * Event-loop slice budget for fits inside the worker (default 50 ms —
   * coarser than the main thread's 12 ms: nothing paints here, the slices
   * only exist so abort messages get processed).
   */
  budgetMs?: number;
}

interface MutableSignal {
  aborted: boolean;
  reason?: unknown;
}

type AnyEstimator = PCA | IncrementalPCA | WebGPUPCA;

interface Entry {
  estimator: 'pca' | 'ipca';
  requestedBackend: 'cpu' | 'webgpu';
  isGpu: boolean;
  deviceOptions: { powerPreference?: 'low-power' | 'high-performance'; minGpuElements?: number };
  inst: AnyEstimator | null;
  createError?: WireError;
}

interface QueueItem {
  msg: QueueableRequest;
  signal: MutableSignal | null;
}

interface CallOutcome {
  value: WireValue;
  model?: AnyPCAModel;
  transfer: ArrayBufferLike[];
}

let webgpuModulePromise: Promise<typeof import('../webgpu/index.js')> | null = null;

function webgpuModule(): Promise<typeof import('../webgpu/index.js')> {
  webgpuModulePromise ??= import('../webgpu/index.js').catch((err: unknown) => {
    webgpuModulePromise = null;
    throw new Error(
      `pca-web worker: failed to load the WebGPU backend (${String(err)}). ` +
        'If this worker was bundled as a classic script (iife), dynamic import is unavailable — ' +
        "configure the bundler to emit an ES-module worker (Vite: worker: { format: 'es' }) " +
        "or create the estimator with backend: 'cpu'.",
    );
  });
  return webgpuModulePromise;
}

function hasWebGPU(): boolean {
  const nav = (globalThis as { navigator?: { gpu?: unknown } }).navigator;
  return typeof nav?.gpu === 'object' && nav.gpu !== null;
}

function progressToWire(e: PCAFitProgress): { event: WireProgress; transfer: ArrayBufferLike[] } {
  const transfer: ArrayBufferLike[] = [];
  const event: WireProgress = {
    estimator: e.estimator,
    solver: e.solver,
    phase: e.phase,
    step: e.step,
    totalSteps: e.totalSteps,
    fraction: e.fraction,
  };
  if (e.detail !== undefined) {
    event.detail = e.detail;
  }
  if (e.snapshot !== undefined) {
    // Snapshot arrays are fresh copies by the reporter contract — they
    // double as the transfer list.
    const snap: WireSnapshot = {
      components: matrixToWire(e.snapshot.components),
      singularValues: tightArray(e.snapshot.singularValues),
      explainedVariance: tightArray(e.snapshot.explainedVariance),
    };
    transfer.push(
      snap.components.data.buffer,
      snap.singularValues.buffer,
      snap.explainedVariance.buffer,
    );
    if (e.snapshot.scores !== undefined) {
      snap.scores = matrixToWire(e.snapshot.scores);
      transfer.push(snap.scores.data.buffer);
    }
    event.snapshot = snap;
  }
  return { event, transfer };
}

/**
 * Timer-less progress throttle: latest-wins coalescing within a phase,
 * phase-boundary and finalize events always delivered, pending events
 * dropped when the terminal result supersedes them.
 */
class ProgressThrottle {
  private pending: PCAFitProgress | null = null;
  private lastPhase: string | null = null;
  private lastSentAt = Number.NEGATIVE_INFINITY;
  private seq = 0;

  constructor(
    private readonly post: (msg: PCAWorkerResponse, transfer?: ArrayBufferLike[]) => void,
    private readonly callId: number,
    private readonly minIntervalMs: number,
  ) {}

  readonly onProgress = (e: PCAFitProgress): void => {
    const now = Date.now();
    const boundary = this.lastPhase !== null && e.phase !== this.lastPhase;
    this.lastPhase = e.phase;
    if (boundary && this.pending !== null) {
      // Deliver the previous phase's final coalesced event before moving on.
      this.send(this.pending);
    }
    this.pending = null;
    if (boundary || e.phase === 'finalize' || now - this.lastSentAt >= this.minIntervalMs) {
      this.send(e);
    } else {
      this.pending = e;
    }
  };

  /** The terminal result/error supersedes any pending event. */
  settle(): void {
    this.pending = null;
  }

  private send(e: PCAFitProgress): void {
    const { event, transfer } = progressToWire(e);
    this.lastSentAt = Date.now();
    this.post({ t: 'progress', id: this.callId, seq: this.seq++, event }, transfer);
  }
}

function requireX(x: Matrix | null, method: string): Matrix {
  if (x === null) {
    throw new Error(`'${method}' requires input data X`);
  }
  return x;
}

function requireInst(entry: Entry): AnyEstimator {
  // Poisoned entries (createError) are rejected before dispatch.
  return entry.inst as AnyEstimator;
}

async function dispatchFit(entry: Entry, x: Matrix, opts: FitAsyncOptions): Promise<void> {
  const inst = requireInst(entry);
  if (entry.isGpu) {
    await (inst as WebGPUPCA).fit(x, opts);
  } else if (entry.estimator === 'pca') {
    await (inst as PCA).fitAsync(x, opts);
  } else {
    await (inst as IncrementalPCA).fitAsync(x, opts);
  }
}

async function dispatchFitTransform(
  entry: Entry,
  x: Matrix,
  opts: FitAsyncOptions,
): Promise<Matrix> {
  const inst = requireInst(entry);
  if (entry.isGpu) {
    return (inst as WebGPUPCA).fitTransform(x, opts);
  }
  if (entry.estimator === 'pca') {
    return (inst as PCA).fitTransformAsync(x, opts);
  }
  return (inst as IncrementalPCA).fitTransformAsync(x, opts);
}

async function rebuildFromModel(entry: Entry, model: AnyPCAModel): Promise<AnyEstimator> {
  if (entry.isGpu) {
    const mod = await webgpuModule();
    return mod.WebGPUPCA.fromModel(model as PCAModel, entry.deviceOptions);
  }
  if (entry.estimator === 'pca') {
    return PCA.fromModel(model as PCAModel);
  }
  return IncrementalPCA.fromModel(model as IncrementalPCAModel);
}

/**
 * Attaches the pca-web protocol to a message port (a dedicated worker's
 * global scope, a MessagePort, or Node's worker_threads port). Returns a
 * detach function. Posts `{t: 'ready'}` immediately so clients can await
 * the handshake.
 */
export function attachPCAWorker(
  port: PCAWorkerPort,
  options: AttachPCAWorkerOptions = {},
): () => void {
  const budgetMs = options.budgetMs ?? 50;
  const registry = new Map<string, Entry>();
  const queue: QueueItem[] = [];
  let running: QueueItem | null = null;
  let pumping = false;
  let detached = false;

  const post = (msg: PCAWorkerResponse, transfer?: ArrayBufferLike[]): void => {
    port.postMessage(msg, transfer ?? []);
  };

  const abortError = (): WireError => ({
    name: 'AbortError',
    message: 'The operation was aborted',
  });

  function handleAbort(targetId: number): void {
    if (running !== null && running.msg.id === targetId) {
      if (running.signal !== null) {
        running.signal.aborted = true;
      }
      return;
    }
    const idx = queue.findIndex((item) => item.msg.id === targetId);
    if (idx >= 0) {
      const [item] = queue.splice(idx, 1);
      post({ t: 'error', id: item.msg.id, error: abortError() });
    }
  }

  async function createEntry(msg: CreateRequest): Promise<Entry> {
    const opts = msg.options as WirePCAOptions;
    const { powerPreference, minGpuElements, ...pcaOptions } = opts;
    const entry: Entry = {
      estimator: msg.estimator,
      requestedBackend: msg.backend,
      isGpu: msg.backend === 'webgpu',
      deviceOptions: { powerPreference, minGpuElements },
      inst: null,
    };
    if (msg.backend === 'webgpu') {
      if (msg.estimator === 'ipca') {
        throw new Error("IncrementalPCA has no WebGPU backend; create it with backend: 'cpu'");
      }
      const mod = await webgpuModule();
      entry.inst = msg.model
        ? mod.WebGPUPCA.fromModel(msg.model as PCAModel, entry.deviceOptions)
        : new mod.WebGPUPCA({ ...pcaOptions, powerPreference, minGpuElements } as PCAOptions);
      return entry;
    }
    if (msg.estimator === 'pca') {
      entry.inst = msg.model
        ? PCA.fromModel(msg.model as PCAModel)
        : new PCA(pcaOptions as PCAOptions);
      return entry;
    }
    entry.inst = msg.model
      ? IncrementalPCA.fromModel(msg.model as IncrementalPCAModel)
      : new IncrementalPCA(msg.options as WireIPCAOptions);
    return entry;
  }

  function buildInfo(entry: Entry): WorkerEstimatorInfo {
    const gpu = entry.isGpu ? (requireInst(entry) as WebGPUPCA) : null;
    return {
      kind: 'info',
      estimator: entry.estimator,
      requestedBackend: entry.requestedBackend,
      backend: gpu !== null ? gpu.backend : 'cpu',
      gpuAdapterInfo: gpu !== null ? gpu.gpuAdapterInfo : null,
      webgpuAvailable: hasWebGPU(),
    };
  }

  async function runCall(
    entry: Entry,
    msg: CallRequest,
    throttle: ProgressThrottle | null,
    signal: MutableSignal,
  ): Promise<CallOutcome> {
    const inst = requireInst(entry);
    const x = msg.x !== undefined ? wireToMatrix(msg.x) : null;
    const fitOptions: FitAsyncOptions = { budgetMs, signal };
    if (throttle !== null) {
      fitOptions.onProgress = throttle.onProgress;
      if (msg.progress?.snapshot !== undefined) {
        fitOptions.snapshot = msg.progress.snapshot;
      }
    }
    const withModel = (outcome: CallOutcome): CallOutcome => {
      if (msg.returnModel === true) {
        outcome.model = inst.toModel();
      }
      return outcome;
    };
    switch (msg.method) {
      case 'fit': {
        await dispatchFit(entry, requireX(x, msg.method), fitOptions);
        return withModel({ value: null, transfer: [] });
      }
      case 'fitTransform': {
        const out = await dispatchFitTransform(entry, requireX(x, msg.method), fitOptions);
        const wire = matrixToWire(out);
        return withModel({ value: wire, transfer: [wire.data.buffer] });
      }
      case 'partialFit': {
        if (entry.estimator !== 'ipca') {
          throw new Error("partialFit requires an estimator created with estimator: 'ipca'");
        }
        (inst as IncrementalPCA).partialFit(requireX(x, msg.method));
        return withModel({ value: null, transfer: [] });
      }
      case 'transform': {
        const xm = requireX(x, msg.method);
        const out = entry.isGpu
          ? await (inst as WebGPUPCA).transform(xm)
          : (inst as PCA | IncrementalPCA).transform(xm);
        const wire = matrixToWire(out);
        return { value: wire, transfer: [wire.data.buffer] };
      }
      case 'inverseTransform': {
        const wire = matrixToWire(inst.inverseTransform(requireX(x, msg.method)));
        return { value: wire, transfer: [wire.data.buffer] };
      }
      case 'scoreSamples': {
        if (entry.estimator !== 'pca') {
          throw new Error('scoreSamples is only available on PCA (like sklearn)');
        }
        const out = tightArray((inst as PCA | WebGPUPCA).scoreSamples(requireX(x, msg.method)));
        return { value: { kind: 'array', data: out }, transfer: [out.buffer] };
      }
      case 'score': {
        if (entry.estimator !== 'pca') {
          throw new Error('score is only available on PCA (like sklearn)');
        }
        return { value: (inst as PCA | WebGPUPCA).score(requireX(x, msg.method)), transfer: [] };
      }
      case 'getCovariance': {
        const wire = matrixToWire(inst.getCovariance());
        return { value: wire, transfer: [wire.data.buffer] };
      }
      case 'getPrecision': {
        const wire = matrixToWire(inst.getPrecision());
        return { value: wire, transfer: [wire.data.buffer] };
      }
      case 'importModel': {
        if (msg.model === undefined) {
          throw new Error('importModel requires a model payload');
        }
        entry.inst = await rebuildFromModel(entry, msg.model);
        return { value: null, transfer: [] };
      }
      case 'info': {
        return { value: buildInfo(entry), transfer: [] };
      }
      default: {
        throw new Error(`Unknown method '${String(msg.method)}'`);
      }
    }
  }

  async function executeItem(item: QueueItem): Promise<void> {
    const msg = item.msg;
    if (msg.t === 'create') {
      try {
        registry.set(msg.est, await createEntry(msg));
        post({ t: 'result', id: msg.id, value: null });
      } catch (err) {
        const error = toWireError(err);
        // Poison the id so pipelined follow-up calls report the root cause.
        registry.set(msg.est, {
          estimator: msg.estimator,
          requestedBackend: msg.backend,
          isGpu: msg.backend === 'webgpu',
          deviceOptions: {},
          inst: null,
          createError: error,
        });
        post({ t: 'error', id: msg.id, error });
      }
      return;
    }
    if (msg.t === 'dispose') {
      const entry = registry.get(msg.est);
      registry.delete(msg.est);
      if (entry?.isGpu && entry.inst !== null) {
        (entry.inst as WebGPUPCA).dispose();
      }
      post({ t: 'result', id: msg.id, value: null });
      return;
    }
    const entry = registry.get(msg.est);
    if (entry === undefined) {
      post({
        t: 'error',
        id: msg.id,
        error: { name: 'Error', message: `Unknown estimator id '${msg.est}' (send create first)` },
      });
      return;
    }
    if (entry.createError !== undefined) {
      post({ t: 'error', id: msg.id, error: entry.createError });
      return;
    }
    const signal: MutableSignal = { aborted: false };
    item.signal = signal;
    const throttle =
      msg.progress !== undefined
        ? new ProgressThrottle(post, msg.id, msg.progress.minIntervalMs ?? 33)
        : null;
    try {
      const outcome = await runCall(entry, msg, throttle, signal);
      throttle?.settle();
      const response: PCAWorkerResponse = { t: 'result', id: msg.id, value: outcome.value };
      if (outcome.model !== undefined) {
        response.model = outcome.model;
        outcome.transfer.push(...modelTransferList(outcome.model));
      }
      post(response, outcome.transfer);
    } catch (err) {
      throttle?.settle();
      post({ t: 'error', id: msg.id, error: toWireError(err) });
    }
  }

  async function pump(): Promise<void> {
    if (pumping) {
      return;
    }
    pumping = true;
    try {
      while (queue.length > 0 && !detached) {
        const item = queue.shift() as QueueItem;
        running = item;
        await executeItem(item);
        running = null;
      }
    } finally {
      running = null;
      pumping = false;
    }
  }

  const listener = (ev: { data: unknown }): void => {
    const msg = ev.data;
    if (!isWorkerRequest(msg)) {
      return;
    }
    if (msg.t === 'abort') {
      // Out-of-band: never queued, takes effect immediately.
      handleAbort(msg.targetId);
      return;
    }
    queue.push({ msg, signal: null });
    void pump();
  };

  port.addEventListener('message', listener);
  port.start?.();
  post({ t: 'ready', protocolVersion: PCA_WORKER_PROTOCOL_VERSION });

  return () => {
    detached = true;
    port.removeEventListener('message', listener);
  };
}
