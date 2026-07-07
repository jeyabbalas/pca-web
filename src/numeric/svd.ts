/**
 * Economy ("thin") singular value decomposition of a dense matrix, matching
 * `scipy.linalg.svd(A, full_matrices=False)` up to floating-point roundoff
 * and the usual sign ambiguity of singular vectors.
 *
 * Golub–Reinsch algorithm: Householder bidiagonalization followed by
 * implicit-shift QR on the bidiagonal, with singular values sorted
 * descending and made non-negative. The core follows the classical
 * EISPACK/JAMA structure (public domain), written here for flat row-major
 * arrays. Requires m >= n; the exported wrapper transposes when m < n.
 */
import type { FloatArray } from '../types.js';
import { transpose } from './blas.js';

export interface SvdResult {
  /** Left singular vectors, m × k (row-major), k = min(m, n). */
  u: Float64Array;
  /** Singular values, length k, descending, non-negative. */
  s: Float64Array;
  /** Right singular vectors transposed, k × n (row-major). */
  vt: Float64Array;
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
const TINY = 2 ** -966;
const MAX_SWEEPS = 1000;

/**
 * Core Golub–Reinsch for m >= n on a float64 working copy `a` (m×n,
 * row-major, consumed as scratch). Returns U (m×n), s (n), V (n×n, columns
 * are right singular vectors).
 */
function svdBase(
  a: Float64Array,
  m: number,
  n: number,
): { u: Float64Array; s: Float64Array; v: Float64Array } {
  const nu = n; // = min(m, n) since m >= n
  const s = new Float64Array(n);
  const u = new Float64Array(m * nu);
  const v = new Float64Array(n * n);
  const e = new Float64Array(n);
  const work = new Float64Array(m);

  // Reduce a to bidiagonal form, storing the diagonal in s and the
  // super-diagonal in e, while collecting Householder vectors in a/u.
  const nct = Math.min(m - 1, n);
  const nrt = Math.max(0, Math.min(n - 2, m));
  const lim = Math.max(nct, nrt);
  for (let k = 0; k < lim; k++) {
    if (k < nct) {
      // Householder for column k: norm of a[k..m-1, k].
      let nrm = 0;
      for (let i = k; i < m; i++) {
        nrm = hypot2(nrm, a[i * n + k]);
      }
      s[k] = nrm;
      if (s[k] !== 0) {
        if (a[k * n + k] < 0) {
          s[k] = -s[k];
        }
        for (let i = k; i < m; i++) {
          a[i * n + k] /= s[k];
        }
        a[k * n + k] += 1.0;
      }
      s[k] = -s[k];
    }
    for (let j = k + 1; j < n; j++) {
      if (k < nct && s[k] !== 0) {
        let t = 0;
        for (let i = k; i < m; i++) {
          t += a[i * n + k] * a[i * n + j];
        }
        t = -t / a[k * n + k];
        for (let i = k; i < m; i++) {
          a[i * n + j] += t * a[i * n + k];
        }
      }
      e[j] = a[k * n + j];
    }
    if (k < nct) {
      for (let i = k; i < m; i++) {
        u[i * nu + k] = a[i * n + k];
      }
    }
    if (k < nrt) {
      // Householder for row k: norm of e[k+1..n-1].
      let nrm = 0;
      for (let i = k + 1; i < n; i++) {
        nrm = hypot2(nrm, e[i]);
      }
      e[k] = nrm;
      if (e[k] !== 0) {
        if (e[k + 1] < 0) {
          e[k] = -e[k];
        }
        for (let i = k + 1; i < n; i++) {
          e[i] /= e[k];
        }
        e[k + 1] += 1.0;
      }
      e[k] = -e[k];
      if (k + 1 < m && e[k] !== 0) {
        for (let i = k + 1; i < m; i++) {
          work[i] = 0;
        }
        for (let j = k + 1; j < n; j++) {
          const ej = e[j];
          for (let i = k + 1; i < m; i++) {
            work[i] += ej * a[i * n + j];
          }
        }
        for (let j = k + 1; j < n; j++) {
          const t = -e[j] / e[k + 1];
          for (let i = k + 1; i < m; i++) {
            a[i * n + j] += t * work[i];
          }
        }
      }
      for (let i = k + 1; i < n; i++) {
        v[i * n + k] = e[i];
      }
    }
  }

  // Set up the final bidiagonal matrix of order p.
  let p = n; // = Math.min(n, m + 1) since m >= n
  if (nct < n) {
    s[nct] = a[nct * n + nct];
  }
  if (nrt + 1 < p) {
    e[nrt] = a[nrt * n + (p - 1)];
  }
  e[p - 1] = 0;

  // Generate U.
  for (let j = nct; j < nu; j++) {
    for (let i = 0; i < m; i++) {
      u[i * nu + j] = 0;
    }
    u[j * nu + j] = 1;
  }
  for (let k = nct - 1; k >= 0; k--) {
    if (s[k] !== 0) {
      for (let j = k + 1; j < nu; j++) {
        let t = 0;
        for (let i = k; i < m; i++) {
          t += u[i * nu + k] * u[i * nu + j];
        }
        t = -t / u[k * nu + k];
        for (let i = k; i < m; i++) {
          u[i * nu + j] += t * u[i * nu + k];
        }
      }
      for (let i = k; i < m; i++) {
        u[i * nu + k] = -u[i * nu + k];
      }
      u[k * nu + k] += 1.0;
      for (let i = 0; i < k - 1; i++) {
        u[i * nu + k] = 0;
      }
    } else {
      for (let i = 0; i < m; i++) {
        u[i * nu + k] = 0;
      }
      u[k * nu + k] = 1;
    }
  }

  // Generate V.
  for (let k = n - 1; k >= 0; k--) {
    if (k < nrt && e[k] !== 0) {
      for (let j = k + 1; j < n; j++) {
        let t = 0;
        for (let i = k + 1; i < n; i++) {
          t += v[i * n + k] * v[i * n + j];
        }
        t = -t / v[(k + 1) * n + k];
        for (let i = k + 1; i < n; i++) {
          v[i * n + j] += t * v[i * n + k];
        }
      }
    }
    for (let i = 0; i < n; i++) {
      v[i * n + k] = 0;
    }
    v[k * n + k] = 1;
  }

  // Main iteration loop for the singular values.
  const pp = p - 1;
  let iter = 0;
  while (p > 0) {
    let k: number;
    let kase: number;
    if (iter > MAX_SWEEPS) {
      throw new Error('svd: failed to converge');
    }
    // kase = 1: s(p) and e[k-1] are negligible and k < p
    // kase = 2: s(k) is negligible and k < p
    // kase = 3: e[k-1] is negligible, k < p, and s(k)...s(p) are not
    //           negligible (QR step)
    // kase = 4: e(p-1) is negligible (convergence)
    for (k = p - 2; k >= -1; k--) {
      if (k === -1) {
        break;
      }
      if (Math.abs(e[k]) <= TINY + EPS * (Math.abs(s[k]) + Math.abs(s[k + 1]))) {
        e[k] = 0;
        break;
      }
    }
    if (k === p - 2) {
      kase = 4;
    } else {
      let ks: number;
      for (ks = p - 1; ks >= k; ks--) {
        if (ks === k) {
          break;
        }
        const t = (ks !== p ? Math.abs(e[ks]) : 0) + (ks !== k + 1 ? Math.abs(e[ks - 1]) : 0);
        if (Math.abs(s[ks]) <= TINY + EPS * t) {
          s[ks] = 0;
          break;
        }
      }
      if (ks === k) {
        kase = 3;
      } else if (ks === p - 1) {
        kase = 1;
      } else {
        kase = 2;
        k = ks;
      }
    }
    k++;

    if (kase === 1) {
      // Deflate negligible s(p).
      let f = e[p - 2];
      e[p - 2] = 0;
      for (let j = p - 2; j >= k; j--) {
        let t = hypot2(s[j], f);
        const cs = s[j] / t;
        const sn = f / t;
        s[j] = t;
        if (j !== k) {
          f = -sn * e[j - 1];
          e[j - 1] = cs * e[j - 1];
        }
        for (let i = 0; i < n; i++) {
          t = cs * v[i * n + j] + sn * v[i * n + (p - 1)];
          v[i * n + (p - 1)] = -sn * v[i * n + j] + cs * v[i * n + (p - 1)];
          v[i * n + j] = t;
        }
      }
    } else if (kase === 2) {
      // Split at negligible s(k).
      let f = e[k - 1];
      e[k - 1] = 0;
      for (let j = k; j < p; j++) {
        let t = hypot2(s[j], f);
        const cs = s[j] / t;
        const sn = f / t;
        s[j] = t;
        f = -sn * e[j];
        e[j] = cs * e[j];
        for (let i = 0; i < m; i++) {
          t = cs * u[i * nu + j] + sn * u[i * nu + (k - 1)];
          u[i * nu + (k - 1)] = -sn * u[i * nu + j] + cs * u[i * nu + (k - 1)];
          u[i * nu + j] = t;
        }
      }
    } else if (kase === 3) {
      // One QR step with Wilkinson-style shift.
      const scale = Math.max(
        Math.max(
          Math.max(Math.max(Math.abs(s[p - 1]), Math.abs(s[p - 2])), Math.abs(e[p - 2])),
          Math.abs(s[k]),
        ),
        Math.abs(e[k]),
      );
      const sp = s[p - 1] / scale;
      const spm1 = s[p - 2] / scale;
      const epm1 = e[p - 2] / scale;
      const sk = s[k] / scale;
      const ek = e[k] / scale;
      const b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2.0;
      const c = sp * epm1 * (sp * epm1);
      let shift = 0;
      if (b !== 0 || c !== 0) {
        shift = Math.sqrt(b * b + c);
        if (b < 0) {
          shift = -shift;
        }
        shift = c / (b + shift);
      }
      let f = (sk + sp) * (sk - sp) + shift;
      let g = sk * ek;
      // Chase zeros.
      for (let j = k; j < p - 1; j++) {
        let t = hypot2(f, g);
        let cs = f / t;
        let sn = g / t;
        if (j !== k) {
          e[j - 1] = t;
        }
        f = cs * s[j] + sn * e[j];
        e[j] = cs * e[j] - sn * s[j];
        g = sn * s[j + 1];
        s[j + 1] = cs * s[j + 1];
        for (let i = 0; i < n; i++) {
          t = cs * v[i * n + j] + sn * v[i * n + (j + 1)];
          v[i * n + (j + 1)] = -sn * v[i * n + j] + cs * v[i * n + (j + 1)];
          v[i * n + j] = t;
        }
        t = hypot2(f, g);
        cs = f / t;
        sn = g / t;
        s[j] = t;
        f = cs * e[j] + sn * s[j + 1];
        s[j + 1] = -sn * e[j] + cs * s[j + 1];
        g = sn * e[j + 1];
        e[j + 1] = cs * e[j + 1];
        if (j < m - 1) {
          for (let i = 0; i < m; i++) {
            t = cs * u[i * nu + j] + sn * u[i * nu + (j + 1)];
            u[i * nu + (j + 1)] = -sn * u[i * nu + j] + cs * u[i * nu + (j + 1)];
            u[i * nu + j] = t;
          }
        }
      }
      e[p - 2] = f;
      iter++;
    } else {
      // Convergence (kase === 4).
      if (s[k] <= 0) {
        s[k] = s[k] < 0 ? -s[k] : 0;
        for (let i = 0; i <= pp; i++) {
          v[i * n + k] = -v[i * n + k];
        }
      }
      // Bubble the converged value into descending order.
      while (k < pp) {
        if (s[k] >= s[k + 1]) {
          break;
        }
        const t = s[k];
        s[k] = s[k + 1];
        s[k + 1] = t;
        if (k < n - 1) {
          for (let i = 0; i < n; i++) {
            const tv = v[i * n + (k + 1)];
            v[i * n + (k + 1)] = v[i * n + k];
            v[i * n + k] = tv;
          }
        }
        if (k < m - 1) {
          for (let i = 0; i < m; i++) {
            const tu = u[i * nu + (k + 1)];
            u[i * nu + (k + 1)] = u[i * nu + k];
            u[i * nu + k] = tu;
          }
        }
        k++;
      }
      iter = 0;
      p--;
    }
  }

  return { u, s, v };
}

/**
 * Economy SVD of `a` (m×n, any shape). Input is read (not modified);
 * accepts Float32Array data (values are read exactly, all computation is
 * float64).
 */
export function svd(a: FloatArray, m: number, n: number): SvdResult {
  if (m >= n) {
    const work = Float64Array.from(a);
    const { u, s, v } = svdBase(work, m, n);
    // vt = v transposed (n×n) — rows are right singular vectors.
    const vt = transpose(v, n, n);
    return { u, s, vt };
  }
  // m < n: decompose Aᵀ (n×m) = U' S V'ᵀ, then A = V' S U'ᵀ.
  const at = transpose(a, m, n); // n×m
  const { u: up, s, v: vp } = svdBase(at, n, m);
  // U = V' (m×m); Vt = U'ᵀ (m×n).
  return { u: vp, s, vt: transpose(up, n, m) };
}
