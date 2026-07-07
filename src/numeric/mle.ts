/**
 * Minka's MLE for choosing the number of PCA components — a direct port of
 * sklearn's `_assess_dimension` / `_infer_dimension`.
 *
 * Reference: Thomas P. Minka, "Automatic Choice of Dimensionality for PCA",
 * NIPS 2000.
 */
import { lgamma } from './special.js';

const LOG_PI = Math.log(Math.PI);
const LOG_2PI = Math.log(2 * Math.PI);

/** Log-likelihood that the data has rank `rank`, given the eigenvalue spectrum. */
export function assessDimension(spectrum: Float64Array, rank: number, nSamples: number): number {
  const p = spectrum.length;
  if (!(rank >= 1 && rank < p)) {
    throw new Error('the tested rank should be in [1, n_features - 1]');
  }
  const eps = 1e-15;
  if (spectrum[rank - 1] < eps) {
    // Tiny eigenvalue: log-likelihood would be tiny and numerically fragile.
    return Number.NEGATIVE_INFINITY;
  }

  let pu = -rank * Math.LN2;
  for (let i = 1; i <= rank; i++) {
    pu += lgamma((p - i + 1) / 2) - (LOG_PI * (p - i + 1)) / 2;
  }

  let pl = 0;
  for (let i = 0; i < rank; i++) {
    pl += Math.log(spectrum[i]);
  }
  pl = (-pl * nSamples) / 2;

  let v = 0;
  for (let i = rank; i < p; i++) {
    v += spectrum[i];
  }
  v = Math.max(eps, v / (p - rank));
  const pv = (-Math.log(v) * nSamples * (p - rank)) / 2;

  const m = p * rank - (rank * (rank + 1)) / 2;
  const pp = (LOG_2PI * (m + rank)) / 2;

  const spectrumFilled = Float64Array.from(spectrum);
  for (let i = rank; i < p; i++) {
    spectrumFilled[i] = v;
  }
  let pa = 0;
  const logN = Math.log(nSamples);
  for (let i = 0; i < rank; i++) {
    for (let j = i + 1; j < p; j++) {
      pa +=
        Math.log(
          (spectrum[i] - spectrum[j]) * (1.0 / spectrumFilled[j] - 1.0 / spectrumFilled[i]),
        ) + logN;
    }
  }

  return pu + pl + pv + pp - pa / 2 - (rank * logN) / 2;
}

/** argmax over rank of assessDimension, ranks 1..p-1 (never returns 0). */
export function inferDimension(spectrum: Float64Array, nSamples: number): number {
  const p = spectrum.length;
  let best = 0;
  let bestLl = Number.NEGATIVE_INFINITY;
  for (let rank = 1; rank < p; rank++) {
    const ll = assessDimension(spectrum, rank, nSamples);
    if (ll > bestLl) {
      bestLl = ll;
      best = rank;
    }
  }
  return best;
}
