/**
 * Dense row-major matrix kernels. Inputs may be Float64Array or Float32Array
 * (large data arrays keep their storage dtype); all accumulation happens in
 * float64 (JS numbers) and results are produced as Float64Array.
 */
import type { FloatArray } from '../types.js';

/** C (m×n) = A (m×k) @ B (k×n). */
export function matmul(
  a: FloatArray,
  b: FloatArray,
  m: number,
  k: number,
  n: number,
): Float64Array {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const cOff = i * n;
    const aOff = i * k;
    for (let p = 0; p < k; p++) {
      const av = a[aOff + p];
      if (av !== 0) {
        const bOff = p * n;
        for (let j = 0; j < n; j++) {
          c[cOff + j] += av * b[bOff + j];
        }
      }
    }
  }
  return c;
}

/** C (k1×k2) = Aᵀ @ B where A is (m×k1) and B is (m×k2). */
export function matmulTransA(
  a: FloatArray,
  b: FloatArray,
  m: number,
  k1: number,
  k2: number,
): Float64Array {
  const c = new Float64Array(k1 * k2);
  for (let r = 0; r < m; r++) {
    const aOff = r * k1;
    const bOff = r * k2;
    for (let i = 0; i < k1; i++) {
      const av = a[aOff + i];
      if (av !== 0) {
        const cOff = i * k2;
        for (let j = 0; j < k2; j++) {
          c[cOff + j] += av * b[bOff + j];
        }
      }
    }
  }
  return c;
}

/** C (m×n) = A (m×k) @ Bᵀ where B is (n×k). */
export function matmulTransB(
  a: FloatArray,
  b: FloatArray,
  m: number,
  k: number,
  n: number,
): Float64Array {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const aOff = i * k;
    const cOff = i * n;
    for (let j = 0; j < n; j++) {
      const bOff = j * k;
      let s = 0;
      for (let p = 0; p < k; p++) {
        s += a[aOff + p] * b[bOff + p];
      }
      c[cOff + j] = s;
    }
  }
  return c;
}

/**
 * Symmetric rank-k update: C (p×p) = Aᵀ @ A for A (n×p). Computes the upper
 * triangle and mirrors it, halving the flops vs a generic matmul.
 */
export function syrkT(a: FloatArray, n: number, p: number): Float64Array {
  const c = new Float64Array(p * p);
  for (let r = 0; r < n; r++) {
    const off = r * p;
    for (let i = 0; i < p; i++) {
      const av = a[off + i];
      if (av !== 0) {
        const cOff = i * p;
        for (let j = i; j < p; j++) {
          c[cOff + j] += av * a[off + j];
        }
      }
    }
  }
  for (let i = 0; i < p; i++) {
    for (let j = i + 1; j < p; j++) {
      c[j * p + i] = c[i * p + j];
    }
  }
  return c;
}

/** y (m) = A (m×n) @ x (n). */
export function matvec(a: FloatArray, x: FloatArray, m: number, n: number): Float64Array {
  const y = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    const off = i * n;
    let s = 0;
    for (let j = 0; j < n; j++) {
      s += a[off + j] * x[j];
    }
    y[i] = s;
  }
  return y;
}

/** y (n) = Aᵀ (n×m) @ x (m) where A is (m×n). */
export function matvecTransA(a: FloatArray, x: FloatArray, m: number, n: number): Float64Array {
  const y = new Float64Array(n);
  for (let i = 0; i < m; i++) {
    const off = i * n;
    const xv = x[i];
    if (xv !== 0) {
      for (let j = 0; j < n; j++) {
        y[j] += a[off + j] * xv;
      }
    }
  }
  return y;
}

export function dot(a: FloatArray, b: FloatArray, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) {
    s += a[i] * b[i];
  }
  return s;
}

export function nrm2(a: FloatArray, len: number): number {
  // Two-pass scaled norm for overflow safety.
  let amax = 0;
  for (let i = 0; i < len; i++) {
    const v = Math.abs(a[i]);
    if (v > amax) {
      amax = v;
    }
  }
  if (amax === 0 || !Number.isFinite(amax)) {
    return amax;
  }
  let s = 0;
  for (let i = 0; i < len; i++) {
    const v = a[i] / amax;
    s += v * v;
  }
  return amax * Math.sqrt(s);
}

/** Out-of-place transpose: returns Bᵀ (n×m) for B (m×n). */
export function transpose(a: FloatArray, m: number, n: number): Float64Array {
  const t = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      t[j * m + i] = a[off + j];
    }
  }
  return t;
}
