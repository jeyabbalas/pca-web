/**
 * double-single ("ds") encoding: each float64 is carried as an unevaluated
 * sum of two float32 values (hi + lo), preserving ~48 significant bits.
 * WGSL has no f64; the GEMM kernels multiply ds pairs exactly via fma and
 * accumulate in exact integer bins (see kernels.ts).
 *
 * Matrices are prescaled by a power of two before encoding so their
 * magnitude is ~1: power-of-two scaling is exact in f64, keeps every
 * on-device grid in normal f32 range for any normal f64 input, and is
 * undone exactly on the result.
 */

/** Largest |x|; 0 for an empty or all-zero array. */
export function maxAbs(src: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < src.length; i++) {
    const a = Math.abs(src[i]);
    if (a > m) {
      m = a;
    }
  }
  return m;
}

/** Power-of-two exponent s with |x|·2^-s ≤ ~1 for |x| ≤ max (0 for max = 0). */
export function prescaleExponent(max: number): number {
  if (max === 0) {
    return 0;
  }
  return Math.ceil(Math.log2(max));
}

/** Packs values·2^-scaleExp into interleaved (hi, lo) f32 pairs. */
export function encodeDs(src: ArrayLike<number>, scaleExp: number): Float32Array {
  const scale = 2 ** -scaleExp; // exact for any reasonable exponent
  const out = new Float32Array(src.length * 2);
  for (let i = 0; i < src.length; i++) {
    const x = src[i] * scale; // exact: power-of-two multiply
    const hi = Math.fround(x);
    out[2 * i] = hi;
    out[2 * i + 1] = Math.fround(x - hi);
  }
  return out;
}
