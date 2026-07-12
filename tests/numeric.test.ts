import { describe, expect, it } from 'vitest';
import { syrkT, syrkTChunk, syrkTMirror } from '../src/numeric/blas.js';
import { eigh } from '../src/numeric/eigh.js';
import { inverse, luFactor, permutedL, slogdet } from '../src/numeric/lu.js';
import { qrEconomic } from '../src/numeric/qr.js';
import { RandomState } from '../src/numeric/rng.js';
import { svd } from '../src/numeric/svd.js';
import { assertClose, assertScalarClose } from './helpers/compare.js';
import { getArray, loadSuite, num } from './helpers/fixtures.js';

const suite = loadSuite('numeric');
const T15 = { atol: 1e-12, rtol: 1e-11 };

describe('numeric kernels vs numpy/scipy fixtures', () => {
  for (const c of suite.cases) {
    it(`svd/qr/lu on ${c.id}`, () => {
      const aRef = c.arrays.A;
      const [m, n] = aRef.shape;
      const a = Float64Array.from(getArray(suite, aRef));

      const s = svd(a, m, n).s;
      assertClose(s, getArray(suite, c.arrays.svd_s), T15, `${c.id}: singular values`);

      // Numerical rank: factor columns beyond it are implementation-defined
      // for rank-deficient inputs (QR reflector directions and LU pivot
      // choices there are decided by ~1e-16 residual noise), so elementwise
      // comparison against scipy is only meaningful for the leading columns.
      const sRef = getArray(suite, c.arrays.svd_s);
      let rankA = 0;
      while (rankA < s.length && sRef[rankA] > sRef[0] * 1e-10) {
        rankA++;
      }
      const kDim = Math.min(m, n); // economy Q is m×kDim, R is kDim×n

      const { q, r } = qrEconomic(a, m, n);
      const rRef = getArray(suite, c.arrays.qr_r);
      assertClose(r, rRef, T15, `${c.id}: R`);
      const qRef = getArray(suite, c.arrays.qr_q);
      for (let j = 0; j < Math.min(kDim, rankA); j++) {
        const col = [];
        const colRef = [];
        for (let i = 0; i < m; i++) {
          col.push(q[i * kDim + j]);
          colRef.push(qRef[i * kDim + j]);
        }
        assertClose(col, colRef, T15, `${c.id}: Q column ${j}`);
      }

      const pl = permutedL(luFactor(a, m, n), m, n);
      const plRef = getArray(suite, c.arrays.lu_pl);
      for (let j = 0; j < Math.min(kDim, rankA); j++) {
        const col = [];
        const colRef = [];
        for (let i = 0; i < m; i++) {
          col.push(pl[i * kDim + j]);
          colRef.push(plRef[i * kDim + j]);
        }
        assertClose(col, colRef, T15, `${c.id}: PL column ${j}`);
      }

      if (c.arrays.sym) {
        const sym = Float64Array.from(getArray(suite, c.arrays.sym));
        const { values } = eigh(sym, n);
        assertClose(values, getArray(suite, c.arrays.eigh_values), T15, `${c.id}: eigh values`);

        const shifted = sym.slice();
        for (let i = 0; i < n; i++) {
          shifted[i * n + i] += 10;
        }
        assertClose(
          inverse(shifted, n),
          getArray(suite, c.arrays.inv_shifted),
          T15,
          `${c.id}: inverse`,
        );
        const [sign, ld] = slogdet(shifted, n);
        assertScalarClose(sign, num(c.scalars.slogdet_sign), T15, `${c.id}: slogdet sign`);
        assertScalarClose(ld, num(c.scalars.slogdet_logdet), T15, `${c.id}: slogdet value`);
      }
    });
  }
});

describe('syrkTChunk', () => {
  it('chunked accumulation is bitwise identical to one syrkT pass over arbitrary splits', () => {
    const rng = new RandomState(42);
    for (const [n, p] of [
      [17, 5],
      [64, 8],
      [100, 13],
    ] as const) {
      const a = new Float64Array(n * p);
      rng.standardNormal(a);
      const whole = syrkT(a, n, p);
      for (const splits of [
        [0, n],
        [0, 1, n],
        [0, 7, 7, n],
        [0, 3, 11, n - 1, n],
      ]) {
        const c = new Float64Array(p * p);
        for (let s = 0; s + 1 < splits.length; s++) {
          syrkTChunk(a, p, splits[s], splits[s + 1], c);
        }
        syrkTMirror(c, p);
        expect(c).toEqual(whole);
      }
    }
  });
});
