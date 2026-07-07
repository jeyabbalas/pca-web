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

export function checkFeatureCount(x: Matrix, expected: number, estimator: string): void {
  if (x.cols !== expected) {
    throw new Error(
      `X has ${x.cols} features, but ${estimator} is expecting ${expected} features as input`,
    );
  }
}
