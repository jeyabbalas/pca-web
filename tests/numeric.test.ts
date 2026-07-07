import { describe, it } from 'vitest';
import { eigh } from '../src/numeric/eigh.js';
import { inverse, luFactor, permutedL, slogdet } from '../src/numeric/lu.js';
import { qrEconomic } from '../src/numeric/qr.js';
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

      const { q, r } = qrEconomic(a, m, n);
      assertClose(q, getArray(suite, c.arrays.qr_q), T15, `${c.id}: Q`);
      assertClose(r, getArray(suite, c.arrays.qr_r), T15, `${c.id}: R`);

      const pl = permutedL(luFactor(a, m, n), m, n);
      assertClose(pl, getArray(suite, c.arrays.lu_pl), T15, `${c.id}: PL`);

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
