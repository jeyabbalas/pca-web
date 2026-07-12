import type { Matrix } from './matrix.js';

/** Thrown when a method requiring a fitted model is called before fit. */
export class NotFittedError extends Error {
  constructor(estimator: string) {
    super(
      `This ${estimator} instance is not fitted yet. ` +
        `Call 'fit' with appropriate arguments before using this estimator.`,
    );
    this.name = 'NotFittedError';
  }
}

/** Rejects NaN/Infinity, mirroring sklearn's input validation. */
export function assertAllFinite(x: Matrix, context: string): void {
  const d = x.data;
  for (let i = 0; i < d.length; i++) {
    if (!Number.isFinite(d[i])) {
      throw new Error(`Input contains ${Number.isNaN(d[i]) ? 'NaN' : 'infinity'} (${context})`);
    }
  }
}

/**
 * Rejects fits on fewer than 2 samples. Every variance path divides by
 * n − 1, so a single-sample fit can only produce an all-NaN model. sklearn
 * permits it with a RuntimeWarning; with no warning channel here this is a
 * deliberate, documented deviation. Message text mirrors sklearn's
 * check_array(ensure_min_samples=2).
 */
export function assertMinSamplesForFit(x: Matrix, estimator: string): void {
  if (x.rows < 2) {
    throw new Error(
      `Found array with ${x.rows} sample(s) (shape=(${x.rows}, ${x.cols})) ` +
        `while a minimum of 2 is required by ${estimator}.`,
    );
  }
}

export function checkFeatureCount(x: Matrix, expected: number, estimator: string): void {
  if (x.cols !== expected) {
    throw new Error(
      `X has ${x.cols} features, but ${estimator} is expecting ${expected} features as input`,
    );
  }
}
