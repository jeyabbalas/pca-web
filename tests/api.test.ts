/**
 * API-behavior tests: everything the fixture-parity suites can't express —
 * validation errors, copy semantics, input formats, dtype preservation,
 * determinism, and not-fitted guards. Error-message expectations mirror the
 * sklearn messages each check was ported from.
 */
import { describe, expect, it } from 'vitest';
import { IncrementalPCA, Matrix, NotFittedError, PCA, RandomState } from '../src/index.js';

function demoData(n = 24, p = 6, seed = 7): Matrix {
  const rng = new RandomState(seed);
  const data = new Float64Array(n * p);
  rng.standardNormal(data);
  // add feature offsets so centering is observable
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      data[i * p + j] += j;
    }
  }
  return new Matrix(data, n, p);
}

describe('PCA option validation', () => {
  it('rejects invalid constructor options', () => {
    expect(() => new PCA({ svdSolver: 'lobpcg' as never })).toThrow(/svdSolver must be one of/);
    expect(() => new PCA({ tol: -1 })).toThrow(/tol must be a number >= 0/);
    expect(() => new PCA({ nOversamples: 0 })).toThrow(/nOversamples/);
    expect(() => new PCA({ iteratedPower: -3 })).toThrow(/iteratedPower/);
    expect(() => new PCA({ iteratedPower: 2.5 })).toThrow(/iteratedPower/);
    expect(() => new PCA({ nComponents: -1 })).toThrow(/nComponents/);
    expect(() => new PCA({ nComponents: 'magic' as never })).toThrow(/nComponents/);
  });

  it('rejects fit-time nComponents/solver conflicts with sklearn semantics', () => {
    const X = demoData();
    expect(() => new PCA({ nComponents: 7, svdSolver: 'full' }).fit(X)).toThrow(
      /between 0 and min\(nSamples, nFeatures\)=6/,
    );
    // arpack requires nComponents strictly below min(n, p)
    expect(() => new PCA({ nComponents: 6, svdSolver: 'arpack' }).fit(X)).toThrow(/strictly less/);
    // truncated solvers take only integer nComponents
    expect(() => new PCA({ nComponents: 0.5, svdSolver: 'randomized' }).fit(X)).toThrow(/integer/);
    expect(() => new PCA({ nComponents: 'mle', svdSolver: 'arpack' }).fit(X)).toThrow(
      /cannot be a string/,
    );
    // mle needs nSamples >= nFeatures
    const wide = demoData(4, 9);
    expect(() => new PCA({ nComponents: 'mle', svdSolver: 'full' }).fit(wide)).toThrow(
      /nSamples >= nFeatures/,
    );
  });

  it('rejects non-finite input at fit and transform boundaries', () => {
    const X = demoData();
    const bad = X.copy();
    bad.data[5] = Number.NaN;
    expect(() => new PCA().fit(bad)).toThrow(/NaN/);
    bad.data[5] = Number.POSITIVE_INFINITY;
    expect(() => new PCA().fit(bad)).toThrow(/infinity/);
  });

  it('rejects empty input', () => {
    // [] is rejected at the Matrix boundary, before PCA's own shape check
    expect(() => new PCA().fit([])).toThrow(/at least one row/);
  });
});

describe('not-fitted guards', () => {
  it('throws NotFittedError from every post-fit API before fit', () => {
    const pca = new PCA({ nComponents: 2 });
    const X = demoData();
    expect(() => pca.transform(X)).toThrow(NotFittedError);
    expect(() => pca.inverseTransform(X)).toThrow(NotFittedError);
    expect(() => pca.components).toThrow(NotFittedError);
    expect(() => pca.explainedVariance).toThrow(NotFittedError);
    expect(() => pca.getCovariance()).toThrow(NotFittedError);
    expect(() => pca.getPrecision()).toThrow(NotFittedError);
    expect(() => pca.scoreSamples(X)).toThrow(NotFittedError);
    expect(() => pca.getFeatureNamesOut()).toThrow(NotFittedError);
    expect(() => new IncrementalPCA().transform(X)).toThrow(NotFittedError);
    expect(() => new IncrementalPCA().batchSize).toThrow(/after fit/);
  });

  it('enforces feature-count agreement after fit', () => {
    const pca = new PCA({ nComponents: 2 }).fit(demoData(24, 6));
    expect(() => pca.transform(demoData(5, 4))).toThrow(/expecting 6 features/);
    expect(() => pca.inverseTransform(demoData(5, 4))).toThrow(/nComponents/);
  });
});

describe('copy semantics', () => {
  it('leaves the input untouched with copy=true (default)', () => {
    const X = demoData();
    const snapshot = X.data.slice();
    new PCA({ nComponents: 3 }).fit(X);
    expect(X.data).toEqual(snapshot);
    new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 }).fit(X);
    expect(X.data).toEqual(snapshot);
  });

  it('centers the input in place with copy=false and svdSolver=full', () => {
    const X = demoData();
    const p = X.cols;
    const colMeans = new Float64Array(p);
    for (let i = 0; i < X.rows; i++) {
      for (let j = 0; j < p; j++) {
        colMeans[j] += X.data[i * p + j] / X.rows;
      }
    }
    const pca = new PCA({ nComponents: 3, svdSolver: 'full', copy: false }).fit(X);
    // column means of the mutated buffer are ~0 (X was centered in place)
    for (let j = 0; j < p; j++) {
      let s = 0;
      for (let i = 0; i < X.rows; i++) {
        s += X.data[i * p + j];
      }
      expect(Math.abs(s / X.rows)).toBeLessThan(1e-12);
    }
    // and mean_ holds the original (pre-mutation) column means
    for (let j = 0; j < p; j++) {
      expect(Math.abs(pca.mean[j] - colMeans[j])).toBeLessThan(1e-12);
    }
  });

  it('destroys the input with copy=false and a truncated solver (sklearn parity)', () => {
    const X = demoData();
    const snapshot = X.data.slice();
    new PCA({ nComponents: 2, svdSolver: 'randomized', randomState: 0, copy: false }).fit(X);
    expect(X.data).not.toEqual(snapshot);
  });
});

describe('input formats and dtype preservation', () => {
  it('accepts number[][] and matches Matrix input exactly', () => {
    const X = demoData(10, 3);
    const nested: number[][] = [];
    for (let i = 0; i < X.rows; i++) {
      nested.push(Array.from(X.row(i)));
    }
    const a = new PCA({ nComponents: 2 }).fit(X);
    const b = new PCA({ nComponents: 2 }).fit(nested);
    expect(a.components.data).toEqual(b.components.data);
    expect(a.singularValues).toEqual(b.singularValues);
  });

  it('preserves float32 in, float32 out (fitted attributes and transforms)', () => {
    const X64 = demoData(20, 4);
    const X32 = new Matrix(Float32Array.from(X64.data), 20, 4);
    const pca = new PCA({ nComponents: 2 }).fit(X32);
    expect(pca.components.data).toBeInstanceOf(Float32Array);
    expect(pca.explainedVariance).toBeInstanceOf(Float32Array);
    expect(pca.singularValues).toBeInstanceOf(Float32Array);
    expect(pca.mean).toBeInstanceOf(Float32Array);
    expect(pca.transform(X32).data).toBeInstanceOf(Float32Array);
    expect(pca.inverseTransform(pca.transform(X32)).data).toBeInstanceOf(Float32Array);

    const pca64 = new PCA({ nComponents: 2 }).fit(X64);
    expect(pca64.components.data).toBeInstanceOf(Float64Array);
    expect(pca64.transform(X64).data).toBeInstanceOf(Float64Array);
  });
});

describe('determinism and chaining', () => {
  it('is bitwise-deterministic for a fixed randomState', () => {
    const X = demoData(30, 8);
    const opts = { nComponents: 3, svdSolver: 'randomized' as const, randomState: 42 };
    const a = new PCA(opts).fit(X);
    const b = new PCA(opts).fit(X);
    expect(a.components.data).toEqual(b.components.data);
    expect(a.singularValues).toEqual(b.singularValues);

    const c = new PCA({ ...opts, randomState: 43 }).fit(X);
    expect(c.components.data).not.toEqual(a.components.data);
  });

  it('fit returns this; fitTransform agrees with fit().transform()', () => {
    const X = demoData(15, 5);
    const pca = new PCA({ nComponents: 3 });
    expect(pca.fit(X)).toBe(pca);
    const viaFitTransform = new PCA({ nComponents: 3 }).fitTransform(X);
    const viaTransform = pca.transform(X);
    for (let i = 0; i < viaTransform.data.length; i++) {
      expect(Math.abs(viaFitTransform.data[i] - viaTransform.data[i])).toBeLessThan(1e-10);
    }
  });

  it('whiten round-trips through inverseTransform at full rank', () => {
    const X = demoData(20, 4);
    const pca = new PCA({ nComponents: 4, whiten: true }).fit(X);
    const back = pca.inverseTransform(pca.transform(X));
    for (let i = 0; i < X.data.length; i++) {
      expect(Math.abs(back.data[i] - X.data[i])).toBeLessThan(1e-10);
    }
  });
});

describe('IncrementalPCA API behavior', () => {
  it('rejects invalid options and first-batch shape conflicts', () => {
    expect(() => new IncrementalPCA({ nComponents: 0 })).toThrow(/nComponents/);
    expect(() => new IncrementalPCA({ batchSize: 0 })).toThrow(/batchSize/);
    // nComponents > nFeatures is checked first (sklearn's check order) ...
    expect(() => new IncrementalPCA({ nComponents: 9 }).partialFit(demoData(20, 6))).toThrow(
      /invalid for nFeatures=6/,
    );
    // ... then nComponents > nSamples on the FIRST batch
    const ipca = new IncrementalPCA({ nComponents: 5 });
    expect(() => ipca.partialFit(demoData(4, 6))).toThrow(/less or equal to the batch number/);
  });

  it('rejects feature-count changes between partialFit calls', () => {
    const ipca = new IncrementalPCA({ nComponents: 2 });
    ipca.partialFit(demoData(10, 6));
    expect(() => ipca.partialFit(demoData(10, 5))).toThrow(/expecting 6 features/);
  });

  it('accumulates nSamplesSeen across partialFit calls and exposes batchSize after fit', () => {
    const ipca = new IncrementalPCA({ nComponents: 2 });
    ipca.partialFit(demoData(10, 6, 1));
    ipca.partialFit(demoData(7, 6, 2));
    expect(ipca.nSamplesSeen).toBe(17);

    const fitted = new IncrementalPCA({ nComponents: 2 }).fit(demoData(30, 6));
    expect(fitted.batchSize).toBe(30); // 5 * nFeatures
  });

  it('fit() resets state from previous fits (no leakage across refits)', () => {
    const ipca = new IncrementalPCA({ nComponents: 2 });
    ipca.fit(demoData(30, 6, 1));
    const first = ipca.components.data.slice();
    ipca.fit(demoData(30, 6, 1));
    expect(ipca.components.data).toEqual(first);
    expect(ipca.nSamplesSeen).toBe(30);
  });

  it('names features by estimator: pca vs incrementalpca', () => {
    const X = demoData(20, 3);
    expect(new PCA({ nComponents: 2 }).fit(X).getFeatureNamesOut()).toEqual(['pca0', 'pca1']);
    expect(new IncrementalPCA({ nComponents: 2 }).fit(X).getFeatureNamesOut()).toEqual([
      'incrementalpca0',
      'incrementalpca1',
    ]);
  });
});
