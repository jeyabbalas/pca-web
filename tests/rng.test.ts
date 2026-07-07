import { describe, expect, it } from 'vitest';
import { RandomState } from '../src/numeric/rng.js';
import { getArray, loadSuite, num } from './helpers/fixtures.js';

const suite = loadSuite('rng');

describe('RandomState replica vs numpy fixtures', () => {
  for (const c of suite.cases) {
    it(`matches numpy streams for ${c.id}`, () => {
      const seed = num(c.params.seed);

      const uni = getArray(suite, c.arrays.uniform_m1_1);
      const rs1 = new RandomState(seed);
      const u = new Float64Array(uni.length);
      rs1.uniform(-1, 1, u);
      expect(Array.from(u)).toEqual(Array.from(uni)); // bit-exact

      const nrm = getArray(suite, c.arrays.standard_normal);
      const rs2 = new RandomState(seed);
      const g = new Float64Array(nrm.length);
      rs2.standardNormal(g);
      for (let i = 0; i < g.length; i++) {
        // Math.log may differ from C libm by <= 1 ulp.
        expect(Math.abs(g[i] - nrm[i])).toBeLessThanOrEqual(1e-13 * Math.abs(nrm[i]) + 1e-300);
      }

      const mixed = getArray(suite, c.arrays.mixed);
      const rs3 = new RandomState(seed);
      const parts = [
        new Float64Array(5),
        new Float64Array(3),
        new Float64Array(8),
        new Float64Array(4),
        new Float64Array(1),
      ];
      rs3.standardNormal(parts[0]);
      rs3.uniform(0, 1, parts[1]);
      rs3.standardNormal(parts[2]);
      rs3.uniform(-2, 3, parts[3]);
      rs3.standardNormal(parts[4]);
      const mix = Float64Array.from(parts.flatMap((p) => Array.from(p)));
      for (let i = 0; i < mix.length; i++) {
        expect(Math.abs(mix[i] - mixed[i])).toBeLessThanOrEqual(
          1e-13 * Math.abs(mixed[i]) + 1e-300,
        );
      }

      const mat = getArray(suite, c.arrays.normal_7x3_flat);
      const rs4 = new RandomState(seed);
      const m = new Float64Array(mat.length);
      rs4.standardNormal(m); // C-order flat fill == numpy normal(size=(7,3)).ravel()
      for (let i = 0; i < m.length; i++) {
        expect(Math.abs(m[i] - mat[i])).toBeLessThanOrEqual(1e-13 * Math.abs(mat[i]) + 1e-300);
      }
    });
  }

  it('rejects out-of-range seeds like numpy', () => {
    expect(() => new RandomState(-1)).toThrow();
    expect(() => new RandomState(2 ** 32)).toThrow();
    expect(() => new RandomState(1.5)).toThrow();
  });
});
