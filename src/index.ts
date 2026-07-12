/**
 * pca-web — scikit-learn-compatible PCA and IncrementalPCA for the browser
 * and Node.js, with zero runtime dependencies.
 *
 * WebGPU acceleration lives behind the `pca-web/webgpu` subpath export so
 * the core stays free of GPU code for bundlers.
 */

export type { IncrementalPCAOptions } from './incremental-pca.js';
export { IncrementalPCA } from './incremental-pca.js';
export type { MatrixInput } from './matrix.js';
export { asMatrix, Matrix } from './matrix.js';
export type {
  AnyPCAModel,
  IncrementalPCAModel,
  IncrementalPCAModelOptions,
  PCAModel,
  PCAModelBaseFields,
  PCAModelOptions,
} from './model.js';
export {
  assertValidModel,
  modelFromJSON,
  modelToJSON,
  PCA_MODEL_FORMAT_VERSION,
} from './model.js';
export type { PCAOptions, PowerIterationNormalizer, SvdSolver } from './pca.js';
export { PCA, RandomState } from './pca.js';
export type {
  FitAsyncOptions,
  FitObserver,
  FitPhase,
  FitSolverId,
  PCAFitProgress,
  PCAFitSnapshot,
  SnapshotOptions,
} from './progress.js';
export type { AbortSignalLike } from './scheduling.js';
export { FitAbortError } from './scheduling.js';
export type { Dtype, FloatArray } from './types.js';
export { NotFittedError } from './validation.js';
