/**
 * Wire protocol shared by the worker handler (`pca-web/worker`) and the
 * client proxies (`pca-web/client`).
 *
 * Import discipline: this module (and everything the client pulls in) must
 * stay free of solver code — value imports are limited to matrix/types/
 * model/validation; estimator and progress types appear type-only (erased
 * at build time). This is what keeps `pca-web/client` a ~few-KB proxy while
 * `pca-web/worker` carries the estimators.
 */
import { Matrix } from '../matrix.js';
import type { AnyPCAModel } from '../model.js';
import type { PowerIterationNormalizer, SvdSolver } from '../pca.js';
import type { FitPhase, FitSolverId, SnapshotOptions } from '../progress.js';
import type { FloatArray } from '../types.js';

export const PCA_WORKER_PROTOCOL_VERSION = 1;

// ---------------------------------------------------------------------
// Structural endpoint types (no DOM/WebWorker lib required)
// ---------------------------------------------------------------------

/**
 * The message-port surface the protocol needs. Satisfied by a browser
 * `Worker` or `MessagePort` and by Node's `worker_threads` `MessagePort`
 * (an EventTarget since Node 15) — which is what the vitest suites use.
 * `transfer` is required (pass `[]`) so the signature matches the DOM
 * overload sets structurally.
 */
export interface PCAWorkerPort {
  postMessage(message: unknown, transfer: ArrayBufferLike[]): void;
  addEventListener(type: 'message', listener: (ev: { data: unknown }) => void): void;
  removeEventListener(type: 'message', listener: (ev: { data: unknown }) => void): void;
  /** MessagePorts need an explicit start() when using addEventListener. */
  start?(): void;
}

/** A port that may also be a terminatable/closable endpoint. */
export interface PCAWorkerLike extends PCAWorkerPort {
  terminate?(): unknown;
  close?(): void;
}

export type WorkerFactory = () => PCAWorkerLike;

// ---------------------------------------------------------------------
// Wire data
// ---------------------------------------------------------------------

export interface WireMatrix {
  kind: 'matrix';
  data: FloatArray;
  rows: number;
  cols: number;
}

/** A typed array that owns its entire buffer (safe to clone or transfer). */
export function tightArray<T extends FloatArray>(a: T): T {
  return a.byteOffset === 0 && a.byteLength === a.buffer.byteLength ? a : (a.slice() as T);
}

/**
 * Encodes a Matrix for the wire. The data is tight-sliced when it is a
 * view — cloning or transferring a subarray view would otherwise serialize
 * (or detach) its entire parent buffer.
 */
export function matrixToWire(m: Matrix): WireMatrix {
  return { kind: 'matrix', data: tightArray(m.data), rows: m.rows, cols: m.cols };
}

export function wireToMatrix(w: WireMatrix): Matrix {
  return new Matrix(w.data, w.rows, w.cols);
}

export function isWireMatrix(v: unknown): v is WireMatrix {
  return (
    typeof v === 'object' &&
    v !== null &&
    (v as { kind?: unknown }).kind === 'matrix' &&
    ((v as WireMatrix).data instanceof Float64Array ||
      (v as WireMatrix).data instanceof Float32Array)
  );
}

/** Marshalled error (Error instances do not structured-clone with names pre-ES2022 hosts). */
export interface WireError {
  name: string;
  message: string;
  stack?: string;
}

export function toWireError(err: unknown): WireError {
  if (err instanceof Error) {
    const wire: WireError = { name: err.name || 'Error', message: err.message };
    if (typeof err.stack === 'string') {
      wire.stack = err.stack;
    }
    return wire;
  }
  return { name: 'Error', message: String(err) };
}

// ---------------------------------------------------------------------
// Progress on the wire
// ---------------------------------------------------------------------

/** Snapshot payload with matrices in wire form (always transferred). */
export interface WireSnapshot {
  components: WireMatrix;
  singularValues: Float64Array;
  explainedVariance: Float64Array;
  scores?: WireMatrix;
}

/** `PCAFitProgress` as it crosses the thread boundary. */
export interface WireProgress {
  estimator: 'PCA' | 'IncrementalPCA';
  solver: FitSolverId;
  phase: FitPhase;
  step: number;
  totalSteps: number | null;
  fraction: number | null;
  snapshot?: WireSnapshot;
  detail?: Readonly<Record<string, number>>;
}

/** Client-side progress subscription options for a fit call. */
export interface ProgressOptions {
  /**
   * Worker-side throttle: at most one event per interval, latest-wins
   * within a phase; phase boundaries and finalize are never coalesced.
   * Default 33 ms; 0 delivers every event.
   */
  minIntervalMs?: number;
  snapshot?: SnapshotOptions;
}

// ---------------------------------------------------------------------
// Estimator options on the wire (serializable subsets)
// ---------------------------------------------------------------------

/** PCA options as they cross the wire — randomState must be a numeric seed. */
export interface WirePCAOptions {
  nComponents?: number | 'mle' | null;
  copy?: boolean;
  whiten?: boolean;
  svdSolver?: SvdSolver;
  tol?: number;
  iteratedPower?: number | 'auto';
  nOversamples?: number;
  powerIterationNormalizer?: PowerIterationNormalizer;
  randomState?: number | null;
  /** WebGPU backend knobs (ignored on the CPU backend). */
  powerPreference?: 'low-power' | 'high-performance';
  minGpuElements?: number;
}

export interface WireIPCAOptions {
  nComponents?: number | null;
  whiten?: boolean;
  copy?: boolean;
  batchSize?: number | null;
}

// ---------------------------------------------------------------------
// Requests (client → worker)
// ---------------------------------------------------------------------

export type WorkerMethod =
  | 'fit'
  | 'fitTransform'
  | 'partialFit'
  | 'transform'
  | 'inverseTransform'
  | 'scoreSamples'
  | 'score'
  | 'getCovariance'
  | 'getPrecision'
  | 'importModel'
  | 'info';

export interface CreateRequest {
  t: 'create';
  /** Request id (responded to like any call, enabling pipelining). */
  id: number;
  /** Client-chosen estimator id for all subsequent calls. */
  est: string;
  estimator: 'pca' | 'ipca';
  backend: 'cpu' | 'webgpu';
  options: WirePCAOptions | WireIPCAOptions;
  /** Hydrate the new estimator from a serialized model. */
  model?: AnyPCAModel;
}

export interface CallRequest {
  t: 'call';
  id: number;
  est: string;
  method: WorkerMethod;
  x?: WireMatrix;
  model?: AnyPCAModel;
  /** Subscribe this call to progress events. */
  progress?: ProgressOptions;
  /** Piggyback the fitted model on the result (keeps the client mirror fresh). */
  returnModel?: boolean;
}

/** Out-of-band: never enters the execution queue. */
export interface AbortRequest {
  t: 'abort';
  targetId: number;
}

export interface DisposeRequest {
  t: 'dispose';
  id: number;
  est: string;
}

export type PCAWorkerRequest = CreateRequest | CallRequest | AbortRequest | DisposeRequest;

/** Requests that enter the FIFO queue (abort is handled out-of-band). */
export type QueueableRequest = CreateRequest | CallRequest | DisposeRequest;

// ---------------------------------------------------------------------
// Responses (worker → client)
// ---------------------------------------------------------------------

export interface ReadyResponse {
  t: 'ready';
  protocolVersion: number;
}

export type WireValue =
  | null
  | number
  | string[]
  | WireMatrix
  | { kind: 'array'; data: FloatArray }
  | WorkerEstimatorInfo;

export interface ResultResponse {
  t: 'result';
  id: number;
  value: WireValue;
  /** Piggybacked model (fit/partialFit with returnModel). */
  model?: AnyPCAModel;
}

export interface ProgressResponse {
  t: 'progress';
  id: number;
  /** Monotonic per call — receivers may assert ordering. */
  seq: number;
  event: WireProgress;
}

export interface ErrorResponse {
  t: 'error';
  id: number;
  error: WireError;
}

export type PCAWorkerResponse = ReadyResponse | ResultResponse | ProgressResponse | ErrorResponse;

export interface WorkerEstimatorInfo {
  kind: 'info';
  estimator: 'pca' | 'ipca';
  requestedBackend: 'cpu' | 'webgpu';
  /** Backend the last fit actually ran on ('cpu' until a GPU fit happens). */
  backend: 'cpu' | 'webgpu';
  gpuAdapterInfo: string | null;
  webgpuAvailable: boolean;
}

// ---------------------------------------------------------------------
// Guards and transfer-list helpers
// ---------------------------------------------------------------------

const REQUEST_TAGS = new Set(['create', 'call', 'abort', 'dispose']);
const RESPONSE_TAGS = new Set(['ready', 'result', 'progress', 'error']);

export function isWorkerRequest(v: unknown): v is PCAWorkerRequest {
  return typeof v === 'object' && v !== null && REQUEST_TAGS.has((v as { t?: string }).t as string);
}

export function isWorkerResponse(v: unknown): v is PCAWorkerResponse {
  return (
    typeof v === 'object' && v !== null && RESPONSE_TAGS.has((v as { t?: string }).t as string)
  );
}

/** Buffers of a model's typed arrays (all tight by `toModel` contract). */
export function modelTransferList(model: AnyPCAModel): ArrayBufferLike[] {
  const buffers: ArrayBufferLike[] = [
    model.components.buffer,
    model.mean.buffer,
    model.explainedVariance.buffer,
    model.explainedVarianceRatio.buffer,
    model.singularValues.buffer,
  ];
  if (model.estimator === 'ipca') {
    buffers.push(model.variance.buffer);
  }
  return buffers;
}
