/**
 * Truncated SVD via Golub–Kahan–Lanczos bidiagonalization with full
 * reorthogonalization — the `svdSolver: 'arpack'` equivalent. Like ARPACK at
 * tol=0, it converges the requested singular triplets to machine precision,
 * so its output matches `scipy.sparse.linalg.svds` (and hence sklearn's
 * arpack solver) up to floating-point noise and the sign convention that the
 * caller fixes afterwards with svd_flip.
 *
 * Memory: O((m+n) · j) for a Krylov basis of size j (typically a small
 * multiple of k), never O(m·n) beyond the input itself.
 */
import type { FloatArray } from '../types.js';
import { matvec, matvecTransA } from './blas.js';
import type { RandomState } from './rng.js';
import { type SvdResult, svd } from './svd.js';

const EPS = 2 ** -52;

function norm(v: Float64Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) {
    s += v[i] * v[i];
  }
  return Math.sqrt(s);
}

/** v -= (basis_j · v) basis_j for all j — classical Gram–Schmidt, run twice. */
function reorthogonalize(v: Float64Array, basis: Float64Array[]): void {
  for (let pass = 0; pass < 2; pass++) {
    for (let j = 0; j < basis.length; j++) {
      const b = basis[j];
      let d = 0;
      for (let i = 0; i < v.length; i++) {
        d += b[i] * v[i];
      }
      if (d !== 0) {
        for (let i = 0; i < v.length; i++) {
          v[i] -= d * b[i];
        }
      }
    }
  }
}

/** A fresh random unit vector orthogonal to `basis`, or null if the space is exhausted. */
function randomOrthogonal(
  dim: number,
  basis: Float64Array[],
  rng: RandomState,
): Float64Array | null {
  if (basis.length >= dim) {
    return null;
  }
  for (let attempt = 0; attempt < 5; attempt++) {
    const v = new Float64Array(dim);
    rng.standardNormal(v);
    reorthogonalize(v, basis);
    const nv = norm(v);
    if (nv > 1e-8 * Math.sqrt(dim)) {
      for (let i = 0; i < dim; i++) {
        v[i] /= nv;
      }
      return v;
    }
  }
  return null;
}

/**
 * Top-k singular triplets of `a` (m×n). `v0` seeds the start vector (length
 * min(m,n), like scipy's svds); the converged result does not depend on it.
 * Requires 1 <= k < min(m, n) (ARPACK's constraint, enforced by the caller).
 */
export function lanczosSvd(
  a: FloatArray,
  m: number,
  n: number,
  k: number,
  v0: Float64Array,
  rng: RandomState,
): SvdResult {
  // Run the recurrence with the v-side on the smaller dimension, mirroring
  // scipy operating on the smaller Gram operator.
  const wide = m < n;
  const rows = wide ? n : m; // u-side dimension
  const cols = wide ? m : n; // v-side dimension
  const mulA = (x: Float64Array): Float64Array =>
    wide ? matvecTransA(a, x, m, n) : matvec(a, x, m, n);
  const mulAT = (x: Float64Array): Float64Array =>
    wide ? matvec(a, x, m, n) : matvecTransA(a, x, m, n);

  const jmax = cols;
  const V: Float64Array[] = [];
  const U: Float64Array[] = [];
  const alphas: number[] = [];
  const betas: number[] = [];

  const v = Float64Array.from(v0);
  const v0norm = norm(v);
  if (v0norm === 0) {
    throw new Error('lanczosSvd: starting vector must be nonzero');
  }
  for (let i = 0; i < cols; i++) {
    v[i] /= v0norm;
  }
  V.push(v);

  let anormEst = 0;
  let result: SvdResult | null = null;

  for (let j = 0; j < jmax; j++) {
    // u_j = A v_j - beta_{j-1} u_{j-1}
    const u = mulA(V[j]);
    if (j > 0) {
      const b = betas[j - 1];
      const uPrev = U[j - 1];
      for (let i = 0; i < rows; i++) {
        u[i] -= b * uPrev[i];
      }
    }
    reorthogonalize(u, U);
    let alpha = norm(u);
    anormEst = Math.max(anormEst, alpha);
    if (alpha > EPS * anormEst * rows) {
      for (let i = 0; i < rows; i++) {
        u[i] /= alpha;
      }
      U.push(u);
    } else {
      // Breakdown: the u-side Krylov space is exhausted in this direction.
      alpha = 0;
      const fresh = randomOrthogonal(rows, U, rng);
      if (fresh === null) {
        // Full u-space captured; finish with what we have (post-loop).
        break;
      }
      U.push(fresh);
    }
    alphas.push(alpha);

    // v_{j+1} = Aᵀ u_j - alpha_j v_j
    const vn = mulAT(U[j]);
    const al = alphas[j];
    const vj = V[j];
    for (let i = 0; i < cols; i++) {
      vn[i] -= al * vj[i];
    }
    reorthogonalize(vn, V);
    let beta = norm(vn);

    const basisSize = j + 1;
    const canStop = basisSize >= k;
    const shouldCheck = canStop && (basisSize === jmax || (basisSize - k) % 3 === 0);
    if (shouldCheck) {
      const check = tryFinish(U, V, alphas, betas, beta, k, rows, cols, basisSize === jmax);
      if (check !== null) {
        result = check;
        break;
      }
    }

    if (beta > EPS * anormEst * cols) {
      for (let i = 0; i < cols; i++) {
        vn[i] /= beta;
      }
      V.push(vn);
    } else {
      beta = 0;
      const fresh = randomOrthogonal(cols, V, rng);
      if (fresh === null) {
        result = tryFinish(U, V, alphas, betas, 0, k, rows, cols, true);
        break;
      }
      V.push(fresh);
    }
    betas.push(beta);
  }

  if (result === null) {
    result = tryFinish(U, V, alphas, betas, 0, k, rows, cols, true);
    if (result === null) {
      throw new Error('lanczosSvd: failed to converge');
    }
  }

  if (!wide) {
    return result;
  }
  // We decomposed Aᵀ; swap sides back: A = V_op Σ U_opᵀ.
  const { u: uOp, s, vt: vtOp } = result;
  const uA = new Float64Array(m * k);
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < k; c++) {
      uA[i * k + c] = vtOp[c * m + i];
    }
  }
  const vtA = new Float64Array(k * n);
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < n; i++) {
      vtA[c * n + i] = uOp[i * k + c];
    }
  }
  return { u: uA, s, vt: vtA };
}

/**
 * Builds the bidiagonal B from (alphas, betas), takes its SVD, and returns
 * the top-k Ritz triplets if they are converged (residual = betaNext ·
 * |last row of P|), or unconditionally when `force` is set.
 */
function tryFinish(
  U: Float64Array[],
  V: Float64Array[],
  alphas: number[],
  betas: number[],
  betaNext: number,
  k: number,
  rows: number,
  cols: number,
  force: boolean,
): SvdResult | null {
  const J = alphas.length;
  if (J < k) {
    return null;
  }
  // Upper bidiagonal B (J×J): diag = alphas, superdiag = betas.
  const B = new Float64Array(J * J);
  for (let i = 0; i < J; i++) {
    B[i * J + i] = alphas[i];
    if (i + 1 < J) {
      B[i * J + i + 1] = betas[i];
    }
  }
  const { u: P, s, vt: Qt } = svd(B, J, J);
  const smax = s[0] > 0 ? s[0] : 1;
  if (!force) {
    for (let i = 0; i < k; i++) {
      const residual = Math.abs(betaNext * P[(J - 1) * J + i]);
      if (residual > 1e-13 * smax) {
        return null;
      }
    }
  }

  // uOut (rows×k) = U_mat @ P[:, :k]; U_mat columns are the U basis vectors.
  const uOut = new Float64Array(rows * k);
  for (let j = 0; j < J; j++) {
    const uj = U[j];
    for (let c = 0; c < k; c++) {
      const w = P[j * J + c];
      if (w !== 0) {
        for (let i = 0; i < rows; i++) {
          uOut[i * k + c] += w * uj[i];
        }
      }
    }
  }
  // vtOut (k×cols): row c = Σ_j Qt[c, j] * V[j].
  const vtOut = new Float64Array(k * cols);
  for (let c = 0; c < k; c++) {
    const off = c * cols;
    for (let j = 0; j < J; j++) {
      const w = Qt[c * J + j];
      if (w !== 0) {
        const vj = V[j];
        for (let i = 0; i < cols; i++) {
          vtOut[off + i] += w * vj[i];
        }
      }
    }
  }
  return { u: uOut, s: s.slice(0, k), vt: vtOut };
}
