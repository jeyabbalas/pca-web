/**
 * Economy QR decomposition via Householder reflections, matching
 * `scipy.linalg.qr(mode='economic')` (LAPACK dgeqrf/dorgqr) up to roundoff,
 * including the sign convention (R's diagonal is -sign(a_kk)·‖column‖), so
 * Q matches scipy's column-for-column on the same input.
 */

export interface QrResult {
  /** Q, m × k orthonormal columns, k = min(m, n). */
  q: Float64Array;
  /** R, k × n upper triangular. */
  r: Float64Array;
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

/** Economy QR of `a` (m×n row-major, read-only). */
export function qrEconomic(a: Float64Array, m: number, n: number): QrResult {
  const k = Math.min(m, n);
  const qr = a.slice();
  const rdiag = new Float64Array(k);

  for (let c = 0; c < k; c++) {
    // LAPACK dlarfg semantics: when nothing is stored below the diagonal
    // (norm of the subvector is zero), tau = 0 and H = I — no reflection,
    // and R keeps the diagonal entry's original sign.
    let nrmBelow = 0;
    for (let i = c + 1; i < m; i++) {
      nrmBelow = hypot2(nrmBelow, qr[i * n + c]);
    }
    if (nrmBelow === 0) {
      rdiag[c] = qr[c * n + c];
      qr[c * n + c] = 0; // flags "no reflector" for the Q accumulation
      continue;
    }
    let nrm = hypot2(qr[c * n + c], nrmBelow);
    if (qr[c * n + c] < 0) {
      nrm = -nrm;
    }
    for (let i = c; i < m; i++) {
      qr[i * n + c] /= nrm;
    }
    qr[c * n + c] += 1.0;
    for (let j = c + 1; j < n; j++) {
      let s = 0;
      for (let i = c; i < m; i++) {
        s += qr[i * n + c] * qr[i * n + j];
      }
      s = -s / qr[c * n + c];
      for (let i = c; i < m; i++) {
        qr[i * n + j] += s * qr[i * n + c];
      }
    }
    rdiag[c] = -nrm;
  }

  // R (k×n): upper triangle of the reduced matrix with rdiag on the diagonal.
  const r = new Float64Array(k * n);
  for (let i = 0; i < k; i++) {
    r[i * n + i] = rdiag[i];
    for (let j = i + 1; j < n; j++) {
      r[i * n + j] = qr[i * n + j];
    }
  }

  // Q (m×k): accumulate reflectors backwards.
  const q = new Float64Array(m * k);
  for (let c = k - 1; c >= 0; c--) {
    for (let i = 0; i < m; i++) {
      q[i * k + c] = 0;
    }
    q[c * k + c] = 1;
    for (let j = c; j < k; j++) {
      if (qr[c * n + c] !== 0) {
        let s = 0;
        for (let i = c; i < m; i++) {
          s += qr[i * n + c] * q[i * k + j];
        }
        s = -s / qr[c * n + c];
        for (let i = c; i < m; i++) {
          q[i * k + j] += s * qr[i * n + c];
        }
      }
    }
  }

  return { q, r };
}
