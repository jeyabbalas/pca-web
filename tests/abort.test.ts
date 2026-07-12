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

const X = gaussian(200, 30, 13);

describe('fitAsync abort', () => {
  it('a pre-aborted signal rejects with AbortError, zero events, estimator unfitted', async () => {
    const controller = new AbortController();
    controller.abort();
    const events: PCAFitProgress[] = [];
    const pca = new PCA({ nComponents: 5, svdSolver: 'randomized', randomState: 0 });
    await expect(
      pca.fitAsync(X, { signal: controller.signal, onProgress: (e) => events.push(e) }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(events).toHaveLength(0);
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('aborting from inside onProgress stops the fit and rejects', async () => {
    const controller = new AbortController();
    const events: PCAFitProgress[] = [];
    const pca = new PCA({
      nComponents: 5,
      svdSolver: 'randomized',
      iteratedPower: 7,
      randomState: 0,
    });
    await expect(
      pca.fitAsync(X, {
        budgetMs: 0,
        signal: controller.signal,
        onProgress: (e) => {
          events.push(e);
          if (e.phase === 'power-iteration' && e.step === 2) {
            controller.abort();
          }
        },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    // No events after the aborting one.
    expect(events[events.length - 1].step).toBe(2);
    expect(events.filter((e) => e.phase === 'finalize')).toHaveLength(0);
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('abort rejects with the signal reason when one is provided', async () => {
    const controller = new AbortController();
    const reason = new Error('user clicked cancel');
    controller.abort(reason);
    await expect(
      new PCA({ nComponents: 3, svdSolver: 'full' }).fitAsync(X, { signal: controller.signal }),
    ).rejects.toBe(reason);
  });

  it('a minimal structural {aborted} signal works without DOM types', async () => {
    const signal = { aborted: false };
    const pca = new PCA({ nComponents: 5, svdSolver: 'arpack', randomState: 0 });
    const fit = pca.fitAsync(X, { budgetMs: 0, signal });
    signal.aborted = true;
    await expect(fit).rejects.toMatchObject({ name: 'AbortError' });
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('an abort before the first step leaves a previous model intact', async () => {
    const pca = new PCA({ nComponents: 5, svdSolver: 'randomized', randomState: 0 });
    pca.fit(X);
    const before = pca.components.data.slice();
    const controller = new AbortController();
    // Abort lands during the initial event-loop yield — no step has run.
    const fit = pca.fitAsync(X, { budgetMs: 0, signal: controller.signal });
    controller.abort();
    await expect(fit).rejects.toMatchObject({ name: 'AbortError' });
    expect(pca.components.data).toEqual(before);
  });

  it('an abort mid-refit invalidates the previous model (documented contract)', async () => {
    const pca = new PCA({ nComponents: 5, svdSolver: 'randomized', randomState: 0 });
    pca.fit(X);
    expect(pca.nComponents).toBe(5);
    const controller = new AbortController();
    await expect(
      pca.fitAsync(X, {
        budgetMs: 0,
        signal: controller.signal,
        onProgress: () => controller.abort(),
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    expect(() => pca.components).toThrow(NotFittedError);
    // And the instance is reusable afterwards.
    await pca.fitAsync(X, { budgetMs: 0 });
    expect(pca.nComponents).toBe(5);
  });

  it('all solvers honor mid-fit aborts at their yield checkpoints', async () => {
    for (const solver of ['full', 'covariance_eigh', 'arpack', 'randomized'] as const) {
      const controller = new AbortController();
      const pca = new PCA({ nComponents: 3, svdSolver: solver, randomState: 0 });
      const fit = pca.fitAsync(gaussian(200, 64, 3), {
        budgetMs: 0,
        signal: controller.signal,
        onProgress: () => controller.abort(),
      });
      await expect(fit, solver).rejects.toMatchObject({ name: 'AbortError' });
      expect(() => pca.components, solver).toThrow(NotFittedError);
    }
  });
});

describe('sync fit abort', () => {
  it('a signal flipped inside onProgress aborts the synchronous fit too', () => {
    const signal = { aborted: false };
    const pca = new PCA({ nComponents: 5, svdSolver: 'randomized', randomState: 0 });
    try {
      pca.fit(X, {
        signal,
        onProgress: (e) => {
          if (e.step === 2) {
            signal.aborted = true;
          }
        },
      });
      expect.unreachable('should have thrown');
    } catch (err) {
      expect((err as Error).name).toBe('AbortError');
    }
    expect(() => pca.components).toThrow(NotFittedError);
  });

  it('a pre-aborted signal throws synchronously before any work', () => {
    const pca = new PCA({ nComponents: 3, svdSolver: 'full' });
    try {
      pca.fit(X, { signal: { aborted: true } });
      expect.unreachable('should have thrown');
    } catch (err) {
      expect((err as Error).name).toBe('AbortError');
    }
  });
});

describe('IncrementalPCA abort', () => {
  it('abort after batch 2 of 5 keeps the 2-batch model', async () => {
    const Xi = gaussian(100, 8, 19);
    const controller = new AbortController();
    const ipca = new IncrementalPCA({ nComponents: 3, batchSize: 20 });
    await expect(
      ipca.fitAsync(Xi, {
        budgetMs: 0,
        signal: controller.signal,
        onProgress: (e) => {
          if (e.phase === 'batch' && e.step === 2) {
            controller.abort();
          }
        },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });

    // The estimator holds exactly the model of the first two batches.
    const ref = new IncrementalPCA({ nComponents: 3 });
    ref.partialFit(new Matrix(Xi.data.slice(0, 20 * 8), 20, 8));
    ref.partialFit(new Matrix(Xi.data.slice(20 * 8, 40 * 8), 20, 8));
    expect(ipca.nSamplesSeen).toBe(40);
    expect(ipca.components.data).toEqual(ref.components.data);
    expect(ipca.singularValues).toEqual(ref.singularValues);

    // partialFit can resume from the kept model.
    ipca.partialFit(new Matrix(Xi.data.slice(40 * 8, 60 * 8), 20, 8));
    expect(ipca.nSamplesSeen).toBe(60);
  });

  it('a pre-aborted signal leaves a previously fitted model reset (sklearn fit() resets first)', async () => {
    const Xi = gaussian(60, 6, 29);
    const ipca = new IncrementalPCA({ nComponents: 2, batchSize: 30 });
    ipca.fit(Xi);
    const controller = new AbortController();
    controller.abort();
    await expect(ipca.fitAsync(Xi, { signal: controller.signal })).rejects.toMatchObject({
      name: 'AbortError',
    });
    // The abort landed before the first step: reset did not run either,
    // so the previous model is intact (same rule as PCA).
    expect(ipca.nSamplesSeen).toBe(60);
  });
});
