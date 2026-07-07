/**
 * Column statistics and small array helpers. All accumulation is float64
 * regardless of the storage dtype of the inputs.
 */
import type { FloatArray } from '../types.js';

/** Per-column mean of X (m×n), float64 accumulation. */
export function colMeans(x: FloatArray, m: number, n: number): Float64Array {
  const sums = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      sums[j] += x[off + j];
    }
  }
  for (let j = 0; j < n; j++) {
    sums[j] /= m;
  }
  return sums;
}

/** Per-column sum of X (m×n), float64 accumulation. */
export function colSums(x: FloatArray, m: number, n: number): Float64Array {
  const sums = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      sums[j] += x[off + j];
    }
  }
  return sums;
}

/** X[i][j] -= mean[j], in place (rounds into X's dtype). */
export function centerInPlace(x: FloatArray, m: number, n: number, mean: Float64Array): void {
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      x[off + j] -= mean[j];
    }
  }
}

/** Sum of all squared entries divided by (m - 1) — total variance of centered X. */
export function totalVariance(xc: FloatArray, m: number, n: number): number {
  let s = 0;
  const len = m * n;
  for (let i = 0; i < len; i++) {
    const v = xc[i];
    s += v * v;
  }
  return s / (m - 1);
}

/** Cumulative sum (sequential, like xp.cumulative_sum). */
export function cumsum(a: FloatArray): Float64Array {
  const out = new Float64Array(a.length);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i];
    out[i] = s;
  }
  return out;
}

/** numpy searchsorted(a, v, side='right') for an ascending array. */
export function searchsortedRight(a: Float64Array, v: number): number {
  let lo = 0;
  let hi = a.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (a[mid] <= v) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}
