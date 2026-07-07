/**
 * pca-web — scikit-learn-compatible PCA and IncrementalPCA for the browser
 * and Node.js, with zero runtime dependencies.
 *
 * WebGPU acceleration lives behind the `pca-web/webgpu` subpath export so
 * the core stays free of GPU code for bundlers.
 */
export { PCA, RandomState } from './pca.js';
export type { PCAOptions, PowerIterationNormalizer, SvdSolver } from './pca.js';
export { IncrementalPCA } from './incremental-pca.js';
export type { IncrementalPCAOptions } from './incremental-pca.js';
export { Matrix, asMatrix } from './matrix.js';
export type { MatrixInput } from './matrix.js';
export { NotFittedError } from './validation.js';
export type { Dtype, FloatArray } from './types.js';
