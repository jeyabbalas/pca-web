/**
 * LU decomposition with partial pivoting (LAPACK dgetrf conventions: pivot is
 * the first maximal |value|), plus the helpers PCA needs: `permute_l` output
 * matching `scipy.linalg.lu(A, permute_l=True)` (used as the randomized
 * solver's power-iteration normalizer), matrix inverse, and slogdet.
 */

export interface LuFactor {
  /** In-place L\U factors, m×n row-major. */
  lu: Float64Array;
  /** ipiv[j]: row j was swapped with row ipiv[j] at step j. */
  ipiv: Int32Array;
}

/** Factorizes a copy of `a` (m×n). Singular pivots are tolerated (column skipped). */
export function luFactor(a: Float64Array, m: number, n: number): LuFactor {
  const lu = a.slice();
  const k = Math.min(m, n);
  const ipiv = new Int32Array(k);

  for (let j = 0; j < k; j++) {
    // Partial pivot: first index of maximal |value| in column j, rows j..m-1.
    let p = j;
    let maxv = Math.abs(lu[j * n + j]);
    for (let i = j + 1; i < m; i++) {
      const v = Math.abs(lu[i * n + j]);
      if (v > maxv) {
        maxv = v;
        p = i;
      }
    }
    ipiv[j] = p;
    if (lu[p * n + j] !== 0) {
      if (p !== j) {
        for (let c = 0; c < n; c++) {
          const t = lu[j * n + c];
          lu[j * n + c] = lu[p * n + c];
          lu[p * n + c] = t;
        }
      }
      const inv = 1 / lu[j * n + j];
      for (let i = j + 1; i < m; i++) {
        lu[i * n + j] *= inv;
      }
      for (let i = j + 1; i < m; i++) {
        const l = lu[i * n + j];
        if (l !== 0) {
          for (let c = j + 1; c < n; c++) {
            lu[i * n + c] -= l * lu[j * n + c];
          }
        }
      }
    }
  }
  return { lu, ipiv };
}

/**
 * P@L from an LU factorization: the m×k unit-lower-trapezoidal factor with
 * the row permutation applied, as returned by scipy's `lu(permute_l=True)`.
 */
export function permutedL(f: LuFactor, m: number, n: number): Float64Array {
  const k = Math.min(m, n);
  const sigma = new Int32Array(m);
  for (let i = 0; i < m; i++) {
    sigma[i] = i;
  }
  for (let j = 0; j < k; j++) {
    const p = f.ipiv[j];
    const t = sigma[j];
    sigma[j] = sigma[p];
    sigma[p] = t;
  }
  const pl = new Float64Array(m * k);
  for (let i = 0; i < m; i++) {
    const dest = sigma[i] * k;
    const lim = Math.min(i, k - 1);
    for (let j = 0; j < lim; j++) {
      pl[dest + j] = f.lu[i * n + j];
    }
    if (i < k) {
      pl[dest + i] = 1;
    } else {
      pl[dest + lim] = f.lu[i * n + lim];
    }
  }
  return pl;
}

/** Solves A x = b in place using a square LU factorization. */
export function luSolveInPlace(f: LuFactor, n: number, b: Float64Array): void {
  const { lu, ipiv } = f;
  for (let j = 0; j < n; j++) {
    const p = ipiv[j];
    if (p !== j) {
      const t = b[j];
      b[j] = b[p];
      b[p] = t;
    }
  }
  // Forward substitution with unit lower triangle.
  for (let i = 1; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) {
      s -= lu[i * n + j] * b[j];
    }
    b[i] = s;
  }
  // Back substitution.
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let j = i + 1; j < n; j++) {
      s -= lu[i * n + j] * b[j];
    }
    b[i] = s / lu[i * n + i];
  }
}

/** Matrix inverse of a square `a` (n×n) via LU. Throws on an exactly singular
 * matrix, like scipy.linalg.inv. */
export function inverse(a: Float64Array, n: number): Float64Array {
  const f = luFactor(a, n, n);
  for (let j = 0; j < n; j++) {
    if (f.lu[j * n + j] === 0) {
      throw new Error('singular matrix: inverse cannot be computed');
    }
  }
  const inv = new Float64Array(n * n);
  const col = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    col.fill(0);
    col[j] = 1;
    luSolveInPlace(f, n, col);
    for (let i = 0; i < n; i++) {
      inv[i * n + j] = col[i];
    }
  }
  return inv;
}

/** numpy `slogdet` for a square matrix: [sign, log|det|]. */
export function slogdet(a: Float64Array, n: number): [number, number] {
  const f = luFactor(a, n, n);
  let sign = 1;
  let logdet = 0;
  for (let j = 0; j < n; j++) {
    if (f.ipiv[j] !== j) {
      sign = -sign;
    }
    const d = f.lu[j * n + j];
    if (d === 0) {
      return [0, Number.NEGATIVE_INFINITY];
    }
    if (d < 0) {
      sign = -sign;
    }
    logdet += Math.log(Math.abs(d));
  }
  return [sign, logdet];
}
