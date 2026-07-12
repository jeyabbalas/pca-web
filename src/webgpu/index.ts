/**
 * `pca-web/webgpu` — WebGPU-accelerated PCA.
 *
 * import { WebGPUPCA, isWebGPUSupported } from 'pca-web/webgpu';
 *
 * TypeScript: the declarations reference the standard WebGPU globals.
 * With `lib: ["dom"]` (TS ≥ 5.9) they resolve natively; Node-only
 * consumers add the types-only package @webgpu/types instead (see README).
 */

export { asMatrix, Matrix, type MatrixInput } from '../matrix.js';
export { PCA, type PCAOptions, type PowerIterationNormalizer, type SvdSolver } from '../pca.js';
export type {
  FitAsyncOptions,
  FitObserver,
  FitPhase,
  FitSolverId,
  PCAFitProgress,
  PCAFitSnapshot,
  SnapshotOptions,
} from '../progress.js';
export type { AbortSignalLike } from '../scheduling.js';
export { FitAbortError } from '../scheduling.js';
export { GpuEngine, isWebGPUSupported, type WebGPUDeviceOptions } from './engine.js';
export { WebGPUPCA, type WebGPUPCAOptions } from './gpu-pca.js';
