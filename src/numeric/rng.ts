/**
 * Replica of numpy's legacy `numpy.random.RandomState` for the streams PCA
 * needs: `uniform` and `normal` (the legacy Box–Muller *polar* Gaussian with
 * its one-value cache). Bit-compatible MT19937 core; floating-point results
 * match numpy to within 1–2 ulp (Math.log may differ from C libm by ≤1 ulp).
 *
 * This is what makes `svdSolver: 'randomized'` reproduce scikit-learn's
 * output for the same `randomState` seed: sklearn draws its Gaussian test
 * matrix from `RandomState(seed).normal(...)`.
 */

const N = 624;
const M = 397;
const MATRIX_A = 0x9908b0df;
const UPPER_MASK = 0x80000000;
const LOWER_MASK = 0x7fffffff;

export class RandomState {
  private mt = new Uint32Array(N);
  private mti = N + 1;
  private hasGauss = false;
  private gauss = 0;

  /**
   * Seeds like `numpy.random.RandomState(seed)`: an integer in [0, 2^32)
   * (numpy's legacy seeding rejects anything larger). Omitted seed draws one
   * from `Math.random()` (non-reproducible, mirroring sklearn's
   * `random_state=None`).
   */
  constructor(seed?: number | null) {
    if (seed === undefined || seed === null) {
      this.initGenrand((Math.random() * 0x100000000) >>> 0);
    } else {
      if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
        throw new RangeError(`randomState seed must be an integer in [0, 2**32 - 1], got ${seed}`);
      }
      this.initGenrand(seed);
    }
  }

  private initGenrand(s: number): void {
    const mt = this.mt;
    mt[0] = s >>> 0;
    for (let i = 1; i < N; i++) {
      const prev = mt[i - 1] ^ (mt[i - 1] >>> 30);
      // 1812433253 * prev + i (mod 2^32)
      mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
    this.mti = N;
    this.hasGauss = false;
    this.gauss = 0;
  }

  /** Next 32-bit unsigned integer from the MT19937 stream. */
  nextUint32(): number {
    const mt = this.mt;
    if (this.mti >= N) {
      let y: number;
      for (let kk = 0; kk < N - M; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + M] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      for (let kk = N - M; kk < N - 1; kk++) {
        y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
        mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      }
      y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
      mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ (y & 1 ? MATRIX_A : 0);
      this.mti = 0;
    }
    let y = mt[this.mti++];
    y ^= y >>> 11;
    y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
    y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
    y ^= y >>> 18;
    return y >>> 0;
  }

  /** numpy `random_sample()`: 53-bit uniform double in [0, 1). */
  randomSample(): number {
    const a = this.nextUint32() >>> 5;
    const b = this.nextUint32() >>> 6;
    return (a * 67108864 + b) / 9007199254740992;
  }

  /** numpy legacy `gauss`: polar Box–Muller with one-value cache. */
  nextGauss(): number {
    if (this.hasGauss) {
      const temp = this.gauss;
      this.hasGauss = false;
      this.gauss = 0;
      return temp;
    }
    let x1: number;
    let x2: number;
    let r2: number;
    do {
      x1 = 2.0 * this.randomSample() - 1.0;
      x2 = 2.0 * this.randomSample() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 === 0.0);
    const f = Math.sqrt((-2.0 * Math.log(r2)) / r2);
    this.gauss = f * x1;
    this.hasGauss = true;
    return f * x2;
  }

  /** Fills `out` with N(0,1) draws in C order, like `normal(size=...)`. */
  standardNormal(out: Float64Array): Float64Array {
    for (let i = 0; i < out.length; i++) {
      out[i] = this.nextGauss();
    }
    return out;
  }

  /** Fills `out` with `low + (high-low) * random_sample()` draws. */
  uniform(low: number, high: number, out: Float64Array): Float64Array {
    const scale = high - low;
    for (let i = 0; i < out.length; i++) {
      out[i] = low + scale * this.randomSample();
    }
    return out;
  }
}

/**
 * sklearn's `check_random_state`: an integer seeds a fresh RandomState, an
 * existing RandomState passes through, null/undefined gives an unseeded one.
 */
export function checkRandomState(state: number | RandomState | null | undefined): RandomState {
  if (state instanceof RandomState) {
    return state;
  }
  return new RandomState(state ?? null);
}
