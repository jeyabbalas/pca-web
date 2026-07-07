/**
 * Symmetric eigendecomposition matching `numpy.linalg.eigh`: eigenvalues in
 * ascending order, eigenvectors in the columns of `vectors`.
 *
 * Householder tridiagonalization (tred2) followed by implicit-shift QL
 * iteration (tql2) — the classical EISPACK/JAMA pair (public domain),
 * written for flat row-major arrays.
 */

export interface EighResult {
  /** Eigenvalues, ascending, length n. */
  values: Float64Array;
  /** Eigenvectors, n×n row-major; column j pairs with values[j]. */
  vectors: Float64Array;
}

function hypot2(a: number, b: number): number {
  const aa = Math.abs(a);
  const ab = Math.abs(b);
  if (aa > ab) {
    const t = ab / aa;
    return aa * Math.sqrt(1 + t * t);
  }
  if (ab === 0) {
    return 0;
  }
  const t = aa / ab;
  return ab * Math.sqrt(1 + t * t);
}

const EPS = 2 ** -52;
const MAX_ITER = 100;

/** Householder reduction to symmetric tridiagonal form, with accumulation. */
function tred2(v: Float64Array, n: number, d: Float64Array, e: Float64Array): void {
  for (let j = 0; j < n; j++) {
    d[j] = v[(n - 1) * n + j];
  }

  for (let i = n - 1; i > 0; i--) {
    let scale = 0;
    let h = 0;
    for (let k = 0; k < i; k++) {
      scale += Math.abs(d[k]);
    }
    if (scale === 0) {
      e[i] = d[i - 1];
      for (let j = 0; j < i; j++) {
        d[j] = v[(i - 1) * n + j];
        v[i * n + j] = 0;
        v[j * n + i] = 0;
      }
    } else {
      // Generate Householder vector.
      for (let k = 0; k < i; k++) {
        d[k] /= scale;
        h += d[k] * d[k];
      }
      let f = d[i - 1];
      let g = Math.sqrt(h);
      if (f > 0) {
        g = -g;
      }
      e[i] = scale * g;
      h -= f * g;
      d[i - 1] = f - g;
      for (let j = 0; j < i; j++) {
        e[j] = 0;
      }
      // Apply similarity transformation to remaining columns.
      for (let j = 0; j < i; j++) {
        f = d[j];
        v[j * n + i] = f;
        g = e[j] + v[j * n + j] * f;
        for (let k = j + 1; k <= i - 1; k++) {
          g += v[k * n + j] * d[k];
          e[k] += v[k * n + j] * f;
        }
        e[j] = g;
      }
      f = 0;
      for (let j = 0; j < i; j++) {
        e[j] /= h;
        f += e[j] * d[j];
      }
      const hh = f / (h + h);
      for (let j = 0; j < i; j++) {
        e[j] -= hh * d[j];
      }
      for (let j = 0; j < i; j++) {
        f = d[j];
        g = e[j];
        for (let k = j; k <= i - 1; k++) {
          v[k * n + j] -= f * e[k] + g * d[k];
        }
        d[j] = v[(i - 1) * n + j];
        v[i * n + j] = 0;
      }
    }
    d[i] = h;
  }

  // Accumulate transformations.
  for (let i = 0; i < n - 1; i++) {
    v[(n - 1) * n + i] = v[i * n + i];
    v[i * n + i] = 1;
    const h = d[i + 1];
    if (h !== 0) {
      for (let k = 0; k <= i; k++) {
        d[k] = v[k * n + (i + 1)] / h;
      }
      for (let j = 0; j <= i; j++) {
        let g = 0;
        for (let k = 0; k <= i; k++) {
          g += v[k * n + (i + 1)] * v[k * n + j];
        }
        for (let k = 0; k <= i; k++) {
          v[k * n + j] -= g * d[k];
        }
      }
    }
    for (let k = 0; k <= i; k++) {
      v[k * n + (i + 1)] = 0;
    }
  }
  for (let j = 0; j < n; j++) {
    d[j] = v[(n - 1) * n + j];
    v[(n - 1) * n + j] = 0;
  }
  v[(n - 1) * n + (n - 1)] = 1;
  e[0] = 0;
}

/** Implicit-shift QL iteration on the tridiagonal, ascending order. */
function tql2(v: Float64Array, n: number, d: Float64Array, e: Float64Array): void {
  for (let i = 1; i < n; i++) {
    e[i - 1] = e[i];
  }
  e[n - 1] = 0;

  let f = 0;
  let tst1 = 0;
  for (let l = 0; l < n; l++) {
    tst1 = Math.max(tst1, Math.abs(d[l]) + Math.abs(e[l]));
    let m = l;
    while (m < n) {
      if (Math.abs(e[m]) <= EPS * tst1) {
        break;
      }
      m++;
    }
    if (m > l) {
      let iter = 0;
      do {
        iter++;
        if (iter > MAX_ITER) {
          throw new Error('eigh: failed to converge');
        }
        // Compute implicit shift.
        let g = d[l];
        let p = (d[l + 1] - g) / (2.0 * e[l]);
        let r = hypot2(p, 1.0);
        if (p < 0) {
          r = -r;
        }
        d[l] = e[l] / (p + r);
        d[l + 1] = e[l] * (p + r);
        const dl1 = d[l + 1];
        let h = g - d[l];
        for (let i = l + 2; i < n; i++) {
          d[i] -= h;
        }
        f += h;
        // Implicit QL transformation.
        p = d[m];
        let c = 1;
        let c2 = c;
        let c3 = c;
        const el1 = e[l + 1];
        let s = 0;
        let s2 = 0;
        for (let i = m - 1; i >= l; i--) {
          c3 = c2;
          c2 = c;
          s2 = s;
          g = c * e[i];
          h = c * p;
          r = hypot2(p, e[i]);
          e[i + 1] = s * r;
          s = e[i] / r;
          c = p / r;
          p = c * d[i] - s * g;
          d[i + 1] = h + s * (c * g + s * d[i]);
          // Accumulate transformation.
          for (let k = 0; k < n; k++) {
            h = v[k * n + (i + 1)];
            v[k * n + (i + 1)] = s * v[k * n + i] + c * h;
            v[k * n + i] = c * v[k * n + i] - s * h;
          }
        }
        p = (-s * s2 * c3 * el1 * e[l]) / dl1;
        e[l] = s * p;
        d[l] = c * p;
      } while (Math.abs(e[l]) > EPS * tst1);
    }
    d[l] += f;
    e[l] = 0;
  }

  // Sort eigenvalues ascending, carrying eigenvector columns along.
  for (let i = 0; i < n - 1; i++) {
    let k = i;
    let p = d[i];
    for (let j = i + 1; j < n; j++) {
      if (d[j] < p) {
        k = j;
        p = d[j];
      }
    }
    if (k !== i) {
      d[k] = d[i];
      d[i] = p;
      for (let j = 0; j < n; j++) {
        p = v[j * n + i];
        v[j * n + i] = v[j * n + k];
        v[j * n + k] = p;
      }
    }
  }
}

/**
 * Eigendecomposition of a symmetric matrix `a` (n×n row-major, read-only).
 * Only the values actually stored are used (the matrix is assumed exactly
 * symmetric, as sklearn's covariance construction guarantees).
 */
export function eigh(a: Float64Array, n: number): EighResult {
  const v = a.slice();
  const d = new Float64Array(n);
  const e = new Float64Array(n);
  if (n === 1) {
    return { values: Float64Array.of(a[0]), vectors: Float64Array.of(1) };
  }
  tred2(v, n, d, e);
  tql2(v, n, d, e);
  return { values: d, vectors: v };
}
