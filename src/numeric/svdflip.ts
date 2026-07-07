/**
 * sklearn's `svd_flip(u, v, u_based_decision=False)`: deterministic sign
 * convention for singular vectors. For each row of Vt, the entry with the
 * largest |value| (first occurrence, like np.argmax) is made positive; the
 * matching column of U gets the same factor. Uses Math.sign like np.sign,
 * so an all-zero row keeps sign 0 — exactly numpy's behavior.
 */
export function svdFlipVBased(
  u: Float64Array | null,
  uRows: number,
  vt: Float64Array,
  vtRows: number,
  vtCols: number,
): void {
  for (let i = 0; i < vtRows; i++) {
    const off = i * vtCols;
    let maxAbs = -1;
    let maxIdx = 0;
    for (let j = 0; j < vtCols; j++) {
      const a = Math.abs(vt[off + j]);
      if (a > maxAbs) {
        maxAbs = a;
        maxIdx = j;
      }
    }
    const sign = Math.sign(vt[off + maxIdx]);
    for (let j = 0; j < vtCols; j++) {
      vt[off + j] *= sign;
    }
    if (u !== null) {
      const uCols = vtRows;
      for (let r = 0; r < uRows; r++) {
        u[r * uCols + i] *= sign;
      }
    }
  }
}

/** `svd_flip(u, v, u_based_decision=True)`: sign from the columns of U. */
export function svdFlipUBased(
  u: Float64Array,
  uRows: number,
  uCols: number,
  vt: Float64Array | null,
  vtCols: number,
): void {
  for (let c = 0; c < uCols; c++) {
    let maxAbs = -1;
    let maxIdx = 0;
    for (let r = 0; r < uRows; r++) {
      const a = Math.abs(u[r * uCols + c]);
      if (a > maxAbs) {
        maxAbs = a;
        maxIdx = r;
      }
    }
    const sign = Math.sign(u[maxIdx * uCols + c]);
    for (let r = 0; r < uRows; r++) {
      u[r * uCols + c] *= sign;
    }
    if (vt !== null) {
      const off = c * vtCols;
      for (let j = 0; j < vtCols; j++) {
        vt[off + j] *= sign;
      }
    }
  }
}
