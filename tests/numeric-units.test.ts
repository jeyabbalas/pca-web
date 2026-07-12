/**
 * Direct unit tests for numeric helpers that the fixture-parity suites only
 * exercise indirectly: incremental statistics, the svd_flip sign convention,
 * Minka's MLE, and the small array utilities. References are pure-JS
 * two-pass computations, or values captured from the pinned scikit-learn
 * (python/requirements.txt) where noted.
 */
import { describe, expect, it } from 'vitest';
import { assessDimension, inferDimension } from '../src/numeric/mle.js';
import { RandomState } from '../src/numeric/rng.js';
import {
  colMeans,
  colSums,
  cumsum,
  incrementalMeanAndVar,
  searchsortedRight,
} from '../src/numeric/stats.js';
import { svdFlipVBased } from '../src/numeric/svdflip.js';
import { assertClose, assertScalarClose } from './helpers/compare.js';

const T = { atol: 1e-12, rtol: 1e-10 };

describe('incrementalMeanAndVar', () => {
  /** Two-pass population mean/variance (ddof=0), the definitionally correct reference. */
  function twoPass(x: Float64Array, m: number, n: number) {
    const mean = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        mean[j] += x[i * n + j];
      }
    }
    for (let j = 0; j < n; j++) {
      mean[j] /= m;
    }
    const variance = new Float64Array(n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        const d = x[i * n + j] - mean[j];
        variance[j] += d * d;
      }
    }
    for (let j = 0; j < n; j++) {
      variance[j] /= m;
    }
    return { mean, variance };
  }

  it('matches the two-pass reference over arbitrary batch splits', () => {
    const m = 40;
    const n = 5;
    const rng = new RandomState(3);
    const x = new Float64Array(m * n);
    rng.standardNormal(x);
    // varied per-column scales and offsets so a wrong accumulator shows up
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        x[i * n + j] = x[i * n + j] * (j + 1) + 10 * j;
      }
    }
    const ref = twoPass(x, m, n);

    const splitPlans = [
      [40],
      [10, 30],
      [13, 13, 13, 1],
      [2, 1, 1, 36],
      [2, ...new Array(38).fill(1)], // seed batch, then a stream of single rows
    ];
    for (const sizes of splitPlans) {
      let mean: Float64Array | null = null;
      let variance: Float64Array | null = null;
      let count = 0;
      let row = 0;
      for (const sz of sizes) {
        const batch = x.subarray(row * n, (row + sz) * n);
        const stats = incrementalMeanAndVar(batch, sz, n, mean, variance, count);
        mean = stats.mean;
        variance = stats.variance;
        count = stats.count;
        row += sz;
      }
      expect(count).toBe(m);
      assertClose(mean as Float64Array, ref.mean, T, `mean after splits [${sizes}]`);
      assertClose(variance as Float64Array, ref.variance, T, `variance after splits [${sizes}]`);
    }
  });
});

describe('svdFlipVBased', () => {
  it('makes the largest-|value| entry of each Vt row positive and flips the matching U column', () => {
    // row 0: max |.| is -4 at index 1 → row and U column 0 negated
    // row 1: max |.| is +0.5 at index 0 → unchanged
    const vt = Float64Array.from([1, -4, 2, 0.5, 0.2, -0.1]);
    const u = Float64Array.from([1, 2, 3, 4, 5, 6]); // 3×2
    svdFlipVBased(u, 3, vt, 2, 3);
    expect(Array.from(vt)).toEqual([-1, 4, -2, 0.5, 0.2, -0.1]);
    expect(Array.from(u)).toEqual([-1, 2, -3, 4, -5, 6]);
  });

  it('resolves ties by first occurrence, like np.argmax', () => {
    const vt = Float64Array.from([-2, 2]);
    svdFlipVBased(null, 0, vt, 1, 2);
    expect(Array.from(vt)).toEqual([2, -2]);

    const stable = Float64Array.from([2, -2]);
    svdFlipVBased(null, 0, stable, 1, 2);
    expect(Array.from(stable)).toEqual([2, -2]);
  });

  it('gives an all-zero row sign 0, zeroing the matching U column (numpy parity)', () => {
    const vt = Float64Array.from([0, 0, 3, 1]); // row 0 all-zero, row 1 positive
    const u = Float64Array.from([1, 2, 3, 4]); // 2×2
    svdFlipVBased(u, 2, vt, 2, 2);
    expect(Array.from(vt)).toEqual([0, 0, 3, 1]);
    expect(Array.from(u)).toEqual([0, 2, 0, 4]);
  });
});

describe("Minka MLE ('mle' nComponents)", () => {
  // Reference values from the pinned scikit-learn's _assess_dimension /
  // _infer_dimension on spectrum [5, 3, 0.5, 0.1, 0.05] with n=50.
  const spectrum = Float64Array.from([5.0, 3.0, 0.5, 0.1, 0.05]);
  const n = 50;

  it('matches sklearn _assess_dimension at every admissible rank', () => {
    const expected = [
      -42.007931644707206, 23.790666469851825, 47.43535336049514, 46.63958779804743,
    ];
    const tol = { atol: 1e-8, rtol: 1e-10 };
    for (let rank = 1; rank <= 4; rank++) {
      assertScalarClose(
        assessDimension(spectrum, rank, n),
        expected[rank - 1],
        tol,
        `rank ${rank}`,
      );
    }
    expect(inferDimension(spectrum, n)).toBe(3);
  });

  it('returns -Infinity for a near-zero eigenvalue at the tested rank', () => {
    const tiny = Float64Array.from([5.0, 3.0, 1e-16, 1e-17]);
    expect(assessDimension(tiny, 3, n)).toBe(Number.NEGATIVE_INFINITY);
    expect(inferDimension(tiny, n)).toBe(2);
  });

  it('rejects ranks outside [1, nFeatures - 1]', () => {
    expect(() => assessDimension(spectrum, 0, n)).toThrow(/rank/);
    expect(() => assessDimension(spectrum, spectrum.length, n)).toThrow(/rank/);
  });
});

describe('small array utilities', () => {
  it('colMeans/colSums accumulate float32 input in float64', () => {
    const x32 = Float32Array.from([1, 2, 3, 4, 5, 6]); // 3×2
    const sums = colSums(x32, 3, 2);
    const means = colMeans(x32, 3, 2);
    expect(sums).toBeInstanceOf(Float64Array);
    expect(Array.from(sums)).toEqual([9, 12]);
    expect(Array.from(means)).toEqual([3, 4]);
  });

  it('cumsum is sequential', () => {
    expect(Array.from(cumsum(Float64Array.from([1, 2, 3])))).toEqual([1, 3, 6]);
    expect(Array.from(cumsum(new Float64Array(0)))).toEqual([]);
  });

  it("searchsortedRight matches np.searchsorted(side='right')", () => {
    const a = Float64Array.from([1, 2, 2, 3]);
    expect(searchsortedRight(a, 0)).toBe(0);
    expect(searchsortedRight(a, 1)).toBe(1);
    expect(searchsortedRight(a, 1.5)).toBe(1);
    expect(searchsortedRight(a, 2)).toBe(3);
    expect(searchsortedRight(a, 3)).toBe(4);
    expect(searchsortedRight(a, 5)).toBe(4);
  });
});
