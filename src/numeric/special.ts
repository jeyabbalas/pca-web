/**
 * Log-gamma via the Lanczos approximation (g = 7, 9 coefficients), accurate
 * to ~1e-15 relative for the positive arguments Minka's MLE uses.
 */
const LANCZOS_G = 7;
const LANCZOS_COEF = [
  0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
  -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
  1.5056327351493116e-7,
];
const LOG_SQRT_2PI = 0.9189385332046727; // log(sqrt(2*pi))

export function lgamma(x: number): number {
  if (x < 0.5) {
    // Reflection formula.
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  const z = x - 1;
  let a = LANCZOS_COEF[0];
  const t = z + LANCZOS_G + 0.5;
  for (let i = 1; i < LANCZOS_COEF.length; i++) {
    a += LANCZOS_COEF[i] / (z + i);
  }
  return LOG_SQRT_2PI + (z + 0.5) * Math.log(t) - t + Math.log(a);
}
