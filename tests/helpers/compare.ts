import { expect } from 'vitest';

export interface Tol {
  atol: number;
  rtol: number;
}

interface DiffStats {
  maxAbs: number;
  maxRel: number;
  /** worst violation of |a-e| <= atol + rtol*|e| measured as ratio to the bound */
  worstRatio: number;
  worstIndex: number;
  n: number;
}

function diffStats(actual: ArrayLike<number>, expected: ArrayLike<number>, tol: Tol): DiffStats {
  let maxAbs = 0;
  let maxRel = 0;
  let worstRatio = 0;
  let worstIndex = -1;
  for (let i = 0; i < expected.length; i++) {
    const a = actual[i];
    const e = expected[i];
    if (Number.isNaN(a) && Number.isNaN(e)) {
      continue;
    }
    const d = Math.abs(a - e);
    const bound = tol.atol + tol.rtol * Math.abs(e);
    const ratio = d / bound;
    if (d > maxAbs) {
      maxAbs = d;
    }
    if (Math.abs(e) > 0 && d / Math.abs(e) > maxRel) {
      maxRel = d / Math.abs(e);
    }
    if (ratio > worstRatio || (Number.isNaN(d) && worstRatio < Number.POSITIVE_INFINITY)) {
      worstRatio = Number.isNaN(d) ? Number.POSITIVE_INFINITY : ratio;
      worstIndex = i;
    }
  }
  return { maxAbs, maxRel, worstRatio, worstIndex, n: expected.length };
}

/** Global registry of observed diffs, printed after the run to document real tolerances. */
const observed = new Map<string, { maxAbs: number; maxRel: number; count: number }>();

export function observedReport(): string {
  const lines: string[] = [];
  const keys = [...observed.keys()].sort();
  for (const k of keys) {
    const o = observed.get(k) as { maxAbs: number; maxRel: number; count: number };
    lines.push(
      `${k.padEnd(44)} n=${String(o.count).padStart(4)} maxAbs=${o.maxAbs.toExponential(2)} maxRel=${o.maxRel.toExponential(2)}`,
    );
  }
  return lines.join('\n');
}

function record(group: string, s: DiffStats): void {
  const cur = observed.get(group) ?? { maxAbs: 0, maxRel: 0, count: 0 };
  cur.maxAbs = Math.max(cur.maxAbs, s.maxAbs);
  cur.maxRel = Math.max(cur.maxRel, s.maxRel);
  cur.count += 1;
  observed.set(group, cur);
}

/**
 * Asserts |actual - expected| <= atol + rtol*|expected| elementwise, with a
 * readable failure message, and records the observed maxima under `group`
 * for the end-of-run tolerance report.
 */
export function assertClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tol: Tol,
  label: string,
  group?: string,
): void {
  expect(actual.length, `${label}: length`).toBe(expected.length);
  const s = diffStats(actual, expected, tol);
  if (group) {
    record(group, s);
  }
  if (s.worstRatio > 1) {
    const i = s.worstIndex;
    throw new Error(
      `${label}: exceeded tolerance (atol=${tol.atol}, rtol=${tol.rtol}) at index ${i}: ` +
        `actual=${actual[i]}, expected=${expected[i]}, |diff|=${Math.abs(
          actual[i] - expected[i],
        )} (${s.worstRatio.toFixed(1)}x the bound); maxAbs=${s.maxAbs.toExponential(3)}, maxRel=${s.maxRel.toExponential(3)}`,
    );
  }
}

export function assertScalarClose(
  actual: number,
  expected: number,
  tol: Tol,
  label: string,
  group?: string,
): void {
  assertClose([actual], [expected], tol, label, group);
}
