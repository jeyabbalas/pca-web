/**
 * `pca-web/webgpu` — WebGPU-accelerated PCA.
 *
 * import { WebGPUPCA, isWebGPUSupported } from 'pca-web/webgpu';
 */

export { asMatrix, Matrix, type MatrixInput } from '../matrix.js';
export { PCA, type PCAOptions, type PowerIterationNormalizer, type SvdSolver } from '../pca.js';
export { GpuEngine, isWebGPUSupported, type WebGPUDeviceOptions } from './engine.js';
export { WebGPUPCA, type WebGPUPCAOptions } from './gpu-pca.js';
