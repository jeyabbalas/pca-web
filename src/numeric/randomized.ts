/**
 * Port of sklearn's `_randomized_svd` (Halko et al.), matching the 1.9.0
 * implementation step for step: same transpose heuristic, same Gaussian test
 * matrix drawn from the numpy RandomState replica, same power-iteration
 * normalizers ('auto' → LU when n_iter > 2, else none), same final economic
 * QR and small SVD. With the same seed this reproduces sklearn's randomized
 * PCA output to floating-point accuracy.
 *
 * The algorithm is written once, as a generator that yields the three
 * products against the (large, possibly device-resident) input matrix and
 * receives their results; everything between yields — panel QR/LU, the small
 * SVD, truncation — is plain CPU code. `randomizedSvd` drives the generator
 * synchronously against the CPU BLAS; the WebGPU backend drives the same
 * generator asynchronously against device-resident buffers, so the two paths
 * cannot drift structurally.
 */
import type { FloatArray } from '../types.js';
import { matmul, matmulTransA, transpose } from './blas.js';
import { luFactor, permutedL } from './lu.js';
import { qrEconomic } from './qr.js';
import type { RandomState } from './rng.js';
import { type SvdResult, svd } from './svd.js';

export interface RandomizedSvdOptions {
  nOversamples: number;
  nIter: number | 'auto';
  powerIterationNormalizer: 'auto' | 'QR' | 'LU' | 'none';
  rng: RandomState;
  /** Round the Gaussian test matrix to float32, as sklearn does for float32 inputs. */
  float32Stream: boolean;
}

/**
 * A product request against the input matrix A (rows×cols):
 * - mulA:  A @ b   (b is cols×w, result rows×w)
 * - mulAT: Aᵀ @ b  (b is rows×w, result cols×w)
 * - mulTA: bᵀ @ A  (b is rows×w, result w×cols)
 */
export interface BigGemmRequest {
  op: 'mulA' | 'mulAT' | 'mulTA';
  b: Float64Array;
  w: number;
}

/** The randomized-SVD algorithm, independent of how A-products are computed. */
export function* randomizedSvdSteps(
  rows: number,
  cols: number,
  k: number,
  opts: RandomizedSvdOptions,
): Generator<BigGemmRequest, SvdResult, Float64Array> {
  const nRandom = k + opts.nOversamples;
  let nIter: number;
  if (opts.nIter === 'auto') {
    // "7 was found a good compromise for PCA" — sklearn #5299.
    nIter = k < 0.1 * Math.min(rows, cols) ? 7 : 4;
  } else {
    nIter = opts.nIter;
  }
  const transposed = rows < cols;
  const effRows = transposed ? cols : rows;
  const effCols = transposed ? rows : cols;

  // Aeff @ B and Aeffᵀ @ B without materializing the transpose.
  const mulAeff = (b: Float64Array, w: number): BigGemmRequest =>
    transposed ? { op: 'mulAT', b, w } : { op: 'mulA', b, w };
  const mulAeffT = (b: Float64Array, w: number): BigGemmRequest =>
    transposed ? { op: 'mulA', b, w } : { op: 'mulAT', b, w };

  // Gaussian test matrix, shape (effCols, nRandom), C-order draw.
  let q: Float64Array = new Float64Array(effCols * nRandom);
  opts.rng.standardNormal(q);
  if (opts.float32Stream) {
    for (let i = 0; i < q.length; i++) {
      q[i] = Math.fround(q[i]);
    }
  }
  let qWidth = nRandom;

  let normalizer = opts.powerIterationNormalizer;
  if (normalizer === 'auto') {
    normalizer = nIter <= 2 ? 'none' : 'LU';
  }

  const applyNormalizer = (m_: Float64Array, mRows: number, mCols: number): Float64Array => {
    if (normalizer === 'LU') {
      const f = luFactor(m_, mRows, mCols);
      return permutedL(f, mRows, mCols);
    }
    if (normalizer === 'QR') {
      return qrEconomic(m_, mRows, mCols).q;
    }
    return m_;
  };

  // Power iterations imprint the top singular vectors of Aeff onto Q.
  for (let it = 0; it < nIter; it++) {
    let t: Float64Array = yield mulAeff(q, qWidth); // effRows × qWidth
    t = applyNormalizer(t, effRows, qWidth);
    const tWidth = normalizer === 'none' ? qWidth : Math.min(effRows, qWidth);
    q = yield mulAeffT(t, tWidth); // effCols × tWidth
    q = applyNormalizer(q, effCols, tWidth);
    qWidth = normalizer === 'none' ? tWidth : Math.min(effCols, tWidth);
  }

  // Orthonormal basis of the sampled range.
  const proj = yield mulAeff(q, qWidth); // effRows × qWidth
  const qFinal = qrEconomic(proj, effRows, qWidth).q;
  const qCols = Math.min(effRows, qWidth);

  // B = Qᵀ Aeff (qCols × effCols), then SVD of the small B.
  let b: Float64Array;
  if (transposed) {
    // B = Qᵀ Aᵀ = (A Q)ᵀ.
    const aq = yield { op: 'mulA', b: qFinal, w: qCols };
    b = transpose(aq, rows, qCols);
  } else {
    b = yield { op: 'mulTA', b: qFinal, w: qCols };
  }
  const { u: uhat, s, vt } = svd(b, qCols, effCols);
  const nu = Math.min(qCols, effCols);
  const uFull = matmul(qFinal, uhat, effRows, qCols, nu); // effRows × nu

  // Truncate to k and undo the transpose, exactly like sklearn.
  if (!transposed) {
    const u = new Float64Array(rows * k);
    for (let i = 0; i < rows; i++) {
      for (let c = 0; c < k; c++) {
        u[i * k + c] = uFull[i * nu + c];
      }
    }
    return { u, s: s.slice(0, k), vt: vt.slice(0, k * effCols) };
  }
  // A = (Aeff)ᵀ = V Σ Uᵀ: U_A = Vt[:k].T (rows×k), Vt_A = (U[:, :k])ᵀ (k×cols).
  const uA = new Float64Array(rows * k);
  for (let i = 0; i < rows; i++) {
    for (let c = 0; c < k; c++) {
      uA[i * k + c] = vt[c * effCols + i];
    }
  }
  const vtA = new Float64Array(k * cols);
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < cols; i++) {
      vtA[c * cols + i] = uFull[i * nu + c];
    }
  }
  return { u: uA, s: s.slice(0, k), vt: vtA };
}

/** Computes one A-product request on the CPU BLAS. */
export function computeBigGemm(
  a: FloatArray,
  rows: number,
  cols: number,
  req: BigGemmRequest,
): Float64Array {
  if (req.op === 'mulA') {
    return matmul(a, req.b, rows, cols, req.w);
  }
  if (req.op === 'mulAT') {
    return matmulTransA(a, req.b, rows, cols, req.w);
  }
  return matmulTransA(req.b, a, rows, req.w, cols);
}

/** Top-k randomized SVD of `a` (rows×cols). Returns U (rows×k), s (k), Vt (k×cols). */
export function randomizedSvd(
  a: FloatArray,
  rows: number,
  cols: number,
  k: number,
  opts: RandomizedSvdOptions,
): SvdResult {
  const gen = randomizedSvdSteps(rows, cols, k, opts);
  let step = gen.next();
  while (!step.done) {
    step = gen.next(computeBigGemm(a, rows, cols, step.value));
  }
  return step.value;
}
