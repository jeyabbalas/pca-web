/**
 * Plain serializable model objects: the currency for worker transfer
 * (`postMessage`), `structuredClone`, and IndexedDB persistence — all of
 * which handle typed arrays natively. `modelToJSON`/`modelFromJSON` cover
 * text transports bit-exactly (JS number formatting is shortest-roundtrip);
 * structured clone / IndexedDB remain the primary, cheaper path.
 *
 * This module is part of the worker client's import graph, so it must stay
 * free of value imports from estimator/solver code (types only).
 */
import type { PowerIterationNormalizer, SvdSolver } from './pca.js';
import { constructorFor, type Dtype, type FloatArray } from './types.js';

export const PCA_MODEL_FORMAT_VERSION = 1;

/** Fields shared by both estimators' models. */
export interface PCAModelBaseFields {
  formatVersion: typeof PCA_MODEL_FORMAT_VERSION;
  /** Storage dtype of the fitted attributes ('float64' | 'float32'). */
  dtype: Dtype;
  nComponents: number;
  nFeaturesIn: number;
  whiten: boolean;
  noiseVariance: number;
  /** k×p row-major principal axes (tight copy). */
  components: FloatArray;
  /** Per-feature training mean, length p. */
  mean: FloatArray;
  /** Length k. */
  explainedVariance: FloatArray;
  /** Length k. */
  explainedVarianceRatio: FloatArray;
  /** Length k. */
  singularValues: FloatArray;
}

/**
 * Serializable PCA constructor options. A live `RandomState` instance
 * cannot be serialized and degrades to null (refits of a rehydrated
 * estimator are then unseeded); numeric seeds round-trip exactly.
 */
export interface PCAModelOptions {
  nComponents: number | 'mle' | null;
  copy: boolean;
  whiten: boolean;
  svdSolver: SvdSolver;
  tol: number;
  iteratedPower: number | 'auto';
  nOversamples: number;
  powerIterationNormalizer: PowerIterationNormalizer;
  randomState: number | null;
}

export interface IncrementalPCAModelOptions {
  nComponents: number | null;
  whiten: boolean;
  copy: boolean;
  batchSize: number | null;
}

export interface PCAModel extends PCAModelBaseFields {
  estimator: 'pca';
  nSamples: number;
  /** The solver the fit actually used (sklearn's `_fit_svd_solver`). */
  svdSolver: Exclude<SvdSolver, 'auto'>;
  options: PCAModelOptions;
}

export interface IncrementalPCAModel extends PCAModelBaseFields {
  estimator: 'ipca';
  nSamplesSeen: number;
  /** Per-feature running variance (always float64), length p. */
  variance: Float64Array;
  /** Set by fit(); null when the model was built via partialFit only. */
  batchSize: number | null;
  options: IncrementalPCAModelOptions;
}

export type AnyPCAModel = PCAModel | IncrementalPCAModel;

const RESOLVED_SOLVERS = ['full', 'covariance_eigh', 'arpack', 'randomized'] as const;

function fail(msg: string): never {
  throw new Error(`Invalid PCA model: ${msg}`);
}

function checkArray(
  value: unknown,
  name: string,
  length: number,
  dtype: Dtype | 'any',
): asserts value is FloatArray {
  if (!(value instanceof Float64Array) && !(value instanceof Float32Array)) {
    fail(`${name} must be a Float64Array or Float32Array`);
  }
  if (dtype !== 'any' && !(value instanceof constructorFor(dtype))) {
    fail(`${name} must be a ${dtype === 'float64' ? 'Float64Array' : 'Float32Array'}`);
  }
  if (value.length !== length) {
    fail(`${name} must have length ${length}, got ${value.length}`);
  }
}

function isPositiveInt(v: unknown): v is number {
  return typeof v === 'number' && Number.isInteger(v) && v >= 1;
}

/**
 * Structural validation of a (possibly deserialized or foreign) model
 * object. Throws with a specific message on the first problem found.
 */
export function assertValidModel(
  model: unknown,
  expected?: 'pca' | 'ipca',
): asserts model is AnyPCAModel {
  if (typeof model !== 'object' || model === null) {
    fail('not an object');
  }
  const m = model as Record<string, unknown>;
  if (m.formatVersion !== PCA_MODEL_FORMAT_VERSION) {
    fail(
      `unsupported formatVersion ${String(m.formatVersion)} (this build reads version ${PCA_MODEL_FORMAT_VERSION})`,
    );
  }
  if (m.estimator !== 'pca' && m.estimator !== 'ipca') {
    fail(`estimator must be 'pca' or 'ipca', got ${String(m.estimator)}`);
  }
  if (expected !== undefined && m.estimator !== expected) {
    fail(`expected a '${expected}' model, got '${String(m.estimator)}'`);
  }
  if (m.dtype !== 'float64' && m.dtype !== 'float32') {
    fail(`dtype must be 'float64' or 'float32', got ${String(m.dtype)}`);
  }
  const dtype = m.dtype as Dtype;
  if (typeof m.nComponents !== 'number' || !Number.isInteger(m.nComponents) || m.nComponents < 0) {
    fail(`nComponents must be a non-negative integer, got ${String(m.nComponents)}`);
  }
  if (!isPositiveInt(m.nFeaturesIn)) {
    fail(`nFeaturesIn must be a positive integer, got ${String(m.nFeaturesIn)}`);
  }
  const k = m.nComponents as number;
  const p = m.nFeaturesIn as number;
  if (typeof m.whiten !== 'boolean') {
    fail('whiten must be a boolean');
  }
  if (typeof m.noiseVariance !== 'number' || !Number.isFinite(m.noiseVariance)) {
    fail('noiseVariance must be a finite number');
  }
  checkArray(m.components, 'components', k * p, dtype);
  checkArray(m.explainedVariance, 'explainedVariance', k, dtype);
  checkArray(m.explainedVarianceRatio, 'explainedVarianceRatio', k, dtype);
  checkArray(m.singularValues, 'singularValues', k, dtype);
  if (typeof m.options !== 'object' || m.options === null) {
    fail('options must be an object');
  }
  if (m.estimator === 'pca') {
    checkArray(m.mean, 'mean', p, dtype);
    if (!isPositiveInt(m.nSamples)) {
      fail(`nSamples must be a positive integer, got ${String(m.nSamples)}`);
    }
    if (!RESOLVED_SOLVERS.includes(m.svdSolver as (typeof RESOLVED_SOLVERS)[number])) {
      fail(`svdSolver must be one of ${RESOLVED_SOLVERS.join(', ')}, got ${String(m.svdSolver)}`);
    }
  } else {
    // IncrementalPCA keeps float64 statistics regardless of the model dtype.
    checkArray(m.mean, 'mean', p, 'float64');
    checkArray(m.variance, 'variance', p, 'float64');
    if (!isPositiveInt(m.nSamplesSeen)) {
      fail(`nSamplesSeen must be a positive integer, got ${String(m.nSamplesSeen)}`);
    }
    if (m.batchSize !== null && !isPositiveInt(m.batchSize)) {
      fail(`batchSize must be a positive integer or null, got ${String(m.batchSize)}`);
    }
  }
}

// ---------------------------------------------------------------------
// JSON transport (bit-exact via shortest-roundtrip number formatting)
// ---------------------------------------------------------------------

const ARRAY_FIELDS = [
  'components',
  'mean',
  'explainedVariance',
  'explainedVarianceRatio',
  'singularValues',
  'variance',
] as const;

/**
 * Serializes a model to a JSON string, with typed arrays as plain number
 * arrays. JavaScript prints doubles in shortest-roundtrip form, so parsing
 * restores every value bit-exactly (float32 payloads are exact in float64).
 */
export function modelToJSON(model: AnyPCAModel): string {
  assertValidModel(model);
  const out: Record<string, unknown> = { ...model };
  for (const f of ARRAY_FIELDS) {
    const v = (model as unknown as Record<string, unknown>)[f];
    if (v instanceof Float64Array || v instanceof Float32Array) {
      out[f] = Array.from(v);
    }
  }
  return JSON.stringify(out);
}

/** Parses `modelToJSON` output back into a validated model object. */
export function modelFromJSON(json: string): AnyPCAModel {
  const raw = JSON.parse(json) as Record<string, unknown>;
  if (typeof raw !== 'object' || raw === null) {
    fail('JSON payload is not an object');
  }
  const dtype: Dtype = raw.dtype === 'float32' ? 'float32' : 'float64';
  const Ctor = constructorFor(dtype);
  for (const f of ARRAY_FIELDS) {
    const v = raw[f];
    if (Array.isArray(v)) {
      // IncrementalPCA keeps float64 running statistics regardless of the
      // model dtype; every other array carries the model dtype.
      const forceF64 = f === 'variance' || (f === 'mean' && raw.estimator === 'ipca');
      raw[f] = forceF64 ? Float64Array.from(v) : Ctor.from(v);
    }
  }
  assertValidModel(raw);
  return raw;
}
