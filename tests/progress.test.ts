import { describe, expect, it } from 'vitest';
import { IncrementalPCA } from '../src/incremental-pca.js';
import { Matrix } from '../src/matrix.js';
import { RandomState } from '../src/numeric/rng.js';
import { PCA } from '../src/pca.js';
import type { PCAFitProgress } from '../src/progress.js';
import { NotFittedError } from '../src/validation.js';

function gaussian(n: number, p: number, seed: number): Matrix {
  const data = new Float64Array(n * p);
  new RandomState(seed).standardNormal(data);
  return new Matrix(data, n, p);
}

function collect(): { events: PCAFitProgress[]; onProgress: (e: PCAFitProgress) => void } {
  const events: PCAFitProgress[] = [];
  return { events, onProgress: (e) => events.push(e) };
}

/** Non-null fractions must be non-decreasing and end at exactly 1. */
function assertFractionMonotone(events: PCAFitProgress[]): void {
  const fractions = events.map((e) => e.fraction).filter((f): f is number => f !== null);
  for (let i = 1; i < fractions.length; i++) {
    expect(fractions[i]).toBeGreaterThanOrEqual(fractions[i - 1]);
  }
  expect(fractions[fractions.length - 1]).toBe(1);
  expect(events[events.length - 1].phase).toBe('finalize');
  expect(events[events.length - 1].fraction).toBe(1);
}

/** Every row of a snapshot's components must have its max-|v| entry positive. */
function assertSignConvention(components: Matrix): void {
  for (let i = 0; i < components.rows; i++) {
    let maxAbs = -1;
    let maxVal = 0;
    for (let j = 0; j < components.cols; j++) {
      const v = components.get(i, j);
      if (Math.abs(v) > maxAbs) {
        maxAbs = Math.abs(v);
        maxVal = v;
      }
    }
    expect(maxVal).toBeGreaterThanOrEqual(0);
  }
}

describe('randomized solver progress', () => {
  const X = gaussian(120, 30, 7);

  it('emits one event per power iteration plus finalize, with known totals', () => {
    const { events, onProgress } = collect();
    new PCA({
      nComponents: 4,
      svdSolver: 'randomized',
      iteratedPower: 5,
      randomState: 0,
    }).fit(X, { onProgress });

    const power = events.filter((e) => e.phase === 'power-iteration');
    expect(power).toHaveLength(5);
    power.forEach((e, i) => {
      expect(e.estimator).toBe('PCA');
      expect(e.solver).toBe('randomized');
      expect(e.step).toBe(i + 1);
      expect(e.totalSteps).toBe(5);
      expect(e.snapshot).toBeUndefined();
    });
    expect(events).toHaveLength(6);
    assertFractionMonotone(events);
  });

  it('attaches snapshots with correct shapes, sign convention, and scores', () => {
    const { events, onProgress } = collect();
    const pca = new PCA({
      nComponents: 4,
      svdSolver: 'randomized',
      iteratedPower: 4,
      randomState: 42,
    });
    const scoresRef = pca.fitTransform(X, {
      onProgress,
      snapshot: { components: true, scores: true },
    });

    const power = events.filter((e) => e.phase === 'power-iteration');
    expect(power).toHaveLength(4);
    for (const e of power) {
      const snap = e.snapshot;
      expect(snap).toBeDefined();
      if (!snap) {
        continue;
      }
      expect(snap.components.rows).toBe(4);
      expect(snap.components.cols).toBe(30);
      expect(snap.components.data).toBeInstanceOf(Float64Array);
      expect(snap.singularValues).toHaveLength(4);
      expect(snap.explainedVariance).toHaveLength(4);
      expect(snap.scores?.rows).toBe(120);
      expect(snap.scores?.cols).toBe(4);
      assertSignConvention(snap.components);
    }

    // The finalize snapshot IS the final model (float64 fit → bitwise).
    const finalize = events[events.length - 1];
    expect(finalize.phase).toBe('finalize');
    expect(finalize.snapshot?.components.data).toEqual(pca.components.data);
    expect(finalize.snapshot?.singularValues).toEqual(pca.singularValues);
    expect(finalize.snapshot?.explainedVariance).toEqual(pca.explainedVariance);
    expect(finalize.snapshot?.scores?.data).toEqual(scoresRef.data);
  });

  it('gates snapshots with every, but finalize always carries one', () => {
    const { events, onProgress } = collect();
    new PCA({
      nComponents: 4,
      svdSolver: 'randomized',
      iteratedPower: 5,
      randomState: 0,
    }).fit(X, { onProgress, snapshot: { components: true, every: 2 } });

    const power = events.filter((e) => e.phase === 'power-iteration');
    expect(power.map((e) => (e.snapshot ? e.step : null)).filter((s) => s !== null)).toEqual([
      2, 4,
    ]);
    expect(events[events.length - 1].snapshot).toBeDefined();
    // components-only snapshots omit scores
    expect(events[events.length - 1].snapshot?.scores).toBeUndefined();
  });

  it('produces a bitwise-identical model with observer on or off (no RNG perturbation)', () => {
    const plain = new PCA({ nComponents: 4, svdSolver: 'randomized', randomState: 3 }).fit(X);
    const { onProgress } = collect();
    const observed = new PCA({ nComponents: 4, svdSolver: 'randomized', randomState: 3 }).fit(X, {
      onProgress,
      snapshot: { components: true, scores: true },
    });
    expect(observed.components.data).toEqual(plain.components.data);
    expect(observed.singularValues).toEqual(plain.singularValues);
    expect(observed.explainedVariance).toEqual(plain.explainedVariance);
    expect(observed.noiseVariance).toBe(plain.noiseVariance);
  });
});

describe('arpack solver progress', () => {
  const X = gaussian(300, 40, 11);

  it('emits residual checkpoints with monotone detail and indeterminate fraction', () => {
    const { events, onProgress } = collect();
    new PCA({ nComponents: 5, svdSolver: 'arpack', randomState: 0 }).fit(X, { onProgress });

    const steps = events.filter((e) => e.phase === 'lanczos-step');
    expect(steps.length).toBeGreaterThanOrEqual(1);
    let lastBasis = 0;
    for (const e of steps) {
      expect(e.totalSteps).toBeNull();
      expect(e.fraction).toBeNull();
      const d = e.detail;
      expect(d).toBeDefined();
      if (!d) {
        continue;
      }
      expect(d.basisSize).toBeGreaterThan(lastBasis);
      lastBasis = d.basisSize;
      expect(d.jmax).toBe(40);
      expect(Number.isFinite(d.maxResidual)).toBe(true);
    }
    assertFractionMonotone(events);
  });

  it('assembles Ritz-triplet snapshots on request', () => {
    const { events, onProgress } = collect();
    const pca = new PCA({ nComponents: 5, svdSolver: 'arpack', randomState: 0 });
    pca.fit(X, { onProgress, snapshot: { components: true, scores: true } });

    const withSnap = events.filter((e) => e.phase === 'lanczos-step' && e.snapshot);
    // The terminal checkpoint deliberately carries no snapshot; earlier ones do.
    for (const e of withSnap) {
      const snap = e.snapshot;
      if (!snap) {
        continue;
      }
      expect(snap.components.rows).toBe(5);
      expect(snap.components.cols).toBe(40);
      expect(snap.scores?.rows).toBe(300);
      expect(snap.scores?.cols).toBe(5);
      assertSignConvention(snap.components);
    }
    const finalize = events[events.length - 1];
    expect(finalize.snapshot?.components.data).toEqual(pca.components.data);
  });

  it('is bit-identical with hooks on or off', () => {
    const plain = new PCA({ nComponents: 5, svdSolver: 'arpack', randomState: 9 }).fit(X);
    const { onProgress } = collect();
    const observed = new PCA({ nComponents: 5, svdSolver: 'arpack', randomState: 9 }).fit(X, {
      onProgress,
      snapshot: { components: true, scores: true },
    });
    expect(observed.components.data).toEqual(plain.components.data);
    expect(observed.singularValues).toEqual(plain.singularValues);
  });
});

describe('covariance_eigh solver progress', () => {
  // p large enough that the 2^22-element chunk budget splits n=200 rows
  // into several gram chunks (chunkRows = max(64, ceil(2^22/256²)) = 64).
  const X = gaussian(200, 256, 5);

  it('emits gram chunks with known totals, a pinned decompose, and finalize', () => {
    const { events, onProgress } = collect();
    new PCA({ nComponents: 3, svdSolver: 'covariance_eigh' }).fit(X, { onProgress });

    const gram = events.filter((e) => e.phase === 'gram');
    expect(gram).toHaveLength(4); // ceil(200/64)
    gram.forEach((e, i) => {
      expect(e.step).toBe(i + 1);
      expect(e.totalSteps).toBe(4);
      expect(e.snapshot).toBeUndefined();
    });
    const decompose = events.filter((e) => e.phase === 'decompose');
    expect(decompose).toHaveLength(1);
    expect(decompose[0].fraction).toBe(0.85);
    assertFractionMonotone(events);
  });

  it('finalize snapshot scores match transform bitwise (no U at fit time)', () => {
    const { events, onProgress } = collect();
    const pca = new PCA({ nComponents: 3, svdSolver: 'covariance_eigh' });
    pca.fit(X, { onProgress, snapshot: { components: true, scores: true } });

    const finalize = events[events.length - 1];
    expect(finalize.snapshot?.components.data).toEqual(pca.components.data);
    expect(finalize.snapshot?.scores?.data).toEqual(pca.transform(X).data);
  });
});

describe('full solver progress', () => {
  const X = gaussian(50, 10, 1);

  it('emits one indeterminate decompose event and finalize', () => {
    const { events, onProgress } = collect();
    const pca = new PCA({ nComponents: 3, svdSolver: 'full' });
    const scoresRef = pca.fitTransform(X, {
      onProgress,
      snapshot: { components: true, scores: true },
    });

    expect(events.map((e) => e.phase)).toEqual(['decompose', 'finalize']);
    expect(events[0].fraction).toBeNull();
    expect(events[0].totalSteps).toBeNull();
    expect(events[1].fraction).toBe(1);
    expect(events[1].snapshot?.scores?.data).toEqual(scoresRef.data);
  });
});

describe('observer semantics', () => {
  const X = gaussian(80, 12, 2);

  it('sync fit emits the identical event sequence on repeated runs', () => {
    const run = () => {
      const { events, onProgress } = collect();
      new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 4 }).fit(X, { onProgress });
      return events.map((e) => ({ ...e }));
    };
    const a = run();
    const b = run();
    expect(a.length).toBe(b.length);
    a.forEach((e, i) => {
      expect(e.phase).toBe(b[i].phase);
      expect(e.step).toBe(b[i].step);
      expect(e.totalSteps).toBe(b[i].totalSteps);
      expect(e.fraction).toBe(b[i].fraction);
    });
  });

  it('a throwing onProgress propagates and leaves the estimator unfitted', () => {
    const pca = new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 });
    const boom = new Error('observer exploded');
    expect(() =>
      pca.fit(X, {
        onProgress: () => {
          throw boom;
        },
      }),
    ).toThrow(boom);
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('a throwing onProgress during refit invalidates the previous model too', () => {
    const pca = new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 });
    pca.fit(X);
    expect(pca.nComponents).toBe(3);
    expect(() =>
      pca.fit(X, {
        onProgress: () => {
          throw new Error('mid-refit');
        },
      }),
    ).toThrow('mid-refit');
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('re-entrant fit on the same instance throws', () => {
    const pca = new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 });
    expect(() =>
      pca.fit(X, {
        onProgress: () => {
          pca.fit(X);
        },
      }),
    ).toThrow(/already fitting/);
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('a fresh fit after a failed one succeeds', () => {
    const pca = new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 });
    expect(() =>
      pca.fit(X, {
        onProgress: () => {
          throw new Error('first attempt dies');
        },
      }),
    ).toThrow('first attempt dies');
    pca.fit(X);
    expect(pca.nComponents).toBe(3);
  });
});

describe('IncrementalPCA progress', () => {
  const X = gaussian(100, 8, 21);

  it('emits one event per batch with linear fraction, then finalize', () => {
    const { events, onProgress } = collect();
    new IncrementalPCA({ nComponents: 3, batchSize: 25 }).fit(X, { onProgress });

    const batches = events.filter((e) => e.phase === 'batch');
    expect(batches).toHaveLength(4);
    batches.forEach((e, i) => {
      expect(e.estimator).toBe('IncrementalPCA');
      expect(e.solver).toBe('incremental');
      expect(e.step).toBe(i + 1);
      expect(e.totalSteps).toBe(4);
      expect(e.fraction).toBeCloseTo((i + 1) / 4, 15);
    });
    assertFractionMonotone(events);
  });

  it('batch-i snapshots bitwise-equal a partialFit prefix reference', () => {
    const { events, onProgress } = collect();
    new IncrementalPCA({ nComponents: 3, batchSize: 25 }).fit(X, {
      onProgress,
      snapshot: { components: true, scores: true },
    });

    const batches = events.filter((e) => e.phase === 'batch');
    for (let i = 0; i < batches.length; i++) {
      const end = 25 * (i + 1);
      const ref = new IncrementalPCA({ nComponents: 3 });
      for (let b = 0; b < i + 1; b++) {
        const rows = new Matrix(X.data.slice(b * 25 * 8, (b + 1) * 25 * 8), 25, 8);
        ref.partialFit(rows);
      }
      const snap = batches[i].snapshot;
      expect(snap).toBeDefined();
      if (!snap) {
        continue;
      }
      expect(snap.components.data).toEqual(ref.components.data);
      expect(snap.singularValues).toEqual(ref.singularValues);
      const prefix = new Matrix(X.data.slice(0, end * 8), end, 8);
      expect(snap.scores?.data).toEqual(ref.transform(prefix).data);
    }
    // finalize carries the full-data snapshot
    const finalize = events[events.length - 1];
    expect(finalize.snapshot?.scores?.rows).toBe(100);
  });

  it('re-entrant fit and concurrent partialFit throw', () => {
    const ipca = new IncrementalPCA({ nComponents: 3, batchSize: 25 });
    expect(() =>
      ipca.fit(X, {
        onProgress: () => {
          ipca.partialFit(X);
        },
      }),
    ).toThrow(/cannot run concurrently/);
    const ipca2 = new IncrementalPCA({ nComponents: 3, batchSize: 25 });
    expect(() =>
      ipca2.fit(X, {
        onProgress: () => {
          ipca2.fit(X);
        },
      }),
    ).toThrow(/already fitting/);
  });
});
