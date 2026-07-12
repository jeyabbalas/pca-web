/**
 * End-to-end tests of the WorkerPCA/WorkerIncrementalPCA client proxies
 * over a Node MessagePort (which satisfies PCAWorkerLike): results must be
 * bit-exact against the direct estimators, the sync mirror must behave,
 * and errors/aborts/termination must surface correctly.
 */
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { afterEach, describe, expect, it } from 'vitest';
import { IncrementalPCA } from '../src/incremental-pca.js';
import { Matrix } from '../src/matrix.js';
import { RandomState } from '../src/numeric/rng.js';
import { PCA, type PCAOptions, type PowerIterationNormalizer, type SvdSolver } from '../src/pca.js';
import type { PCAFitProgress } from '../src/progress.js';
import { NotFittedError } from '../src/validation.js';
import { WorkerIncrementalPCA, WorkerPCA, WorkerTerminatedError } from '../src/worker/client.js';
import { attachPCAWorker } from '../src/worker/handler.js';
import type { PCAWorkerLike } from '../src/worker/protocol.js';
import { getMatrix, loadSuite } from './helpers/fixtures.js';

function gaussian(n: number, p: number, seed: number): Matrix {
  const data = new Float64Array(n * p);
  new RandomState(seed).standardNormal(data);
  return new Matrix(data, n, p);
}

const closers: Array<() => void> = [];
afterEach(() => {
  for (const close of closers.splice(0)) {
    close();
  }
});

/** A worker "thread" over a MessageChannel — same protocol, same process. */
function workerPort(budgetMs = 0): PCAWorkerLike {
  const channel = new MessageChannel();
  const detach = attachPCAWorker(channel.port1, { budgetMs });
  closers.push(() => {
    detach();
    channel.port1.close();
    channel.port2.close();
  });
  return channel.port2 as unknown as PCAWorkerLike;
}

function toOptions(params: Record<string, unknown>): PCAOptions {
  const o: PCAOptions = {};
  if ('n_components' in params) {
    o.nComponents = params.n_components as PCAOptions['nComponents'];
  }
  if ('svd_solver' in params) {
    o.svdSolver = params.svd_solver as SvdSolver;
  }
  if ('whiten' in params) {
    o.whiten = params.whiten as boolean;
  }
  if ('iterated_power' in params) {
    o.iteratedPower = params.iterated_power as number | 'auto';
  }
  if ('n_oversamples' in params) {
    o.nOversamples = params.n_oversamples as number;
  }
  if ('power_iteration_normalizer' in params) {
    o.powerIterationNormalizer = params.power_iteration_normalizer as PowerIterationNormalizer;
  }
  if ('random_state' in params) {
    o.randomState = params.random_state as number | null;
  }
  if ('tol' in params) {
    o.tol = params.tol as number;
  }
  return o;
}

describe('WorkerPCA parity against direct PCA (fixture subset)', () => {
  const suite = loadSuite('pca');
  // Deterministic subset: solvers without RNG, or with a numeric seed.
  const cases = suite.cases
    .filter((c) => {
      const solver = (c.flags.expected_solver ?? c.params.svd_solver ?? 'auto') as string;
      return (
        solver === 'full' ||
        solver === 'covariance_eigh' ||
        typeof c.params.random_state === 'number'
      );
    })
    .filter((_, i) => i % 4 === 0) // every 4th — coverage without bloat
    .slice(0, 12);

  it('covers several solvers', () => {
    expect(cases.length).toBeGreaterThanOrEqual(6);
  });

  for (const c of cases) {
    it(`bit-exact vs direct on ${c.id}`, async () => {
      const X = getMatrix(suite, c.arrays.X);
      const options = toOptions(c.params);
      const direct = new PCA(options).fit(X);
      const { randomState, ...rest } = options;
      const worker = new WorkerPCA({
        ...rest,
        randomState: randomState as number | null | undefined,
        worker: workerPort(),
      });
      await worker.fit(X);

      expect(worker.nComponents).toBe(direct.nComponents);
      expect(worker.nFeaturesIn).toBe(direct.nFeaturesIn);
      expect(worker.nSamples).toBe(direct.nSamples);
      expect(worker.resolvedSvdSolver).toBe(direct.resolvedSvdSolver);
      expect(worker.noiseVariance).toBe(direct.noiseVariance);
      expect(worker.components.data).toEqual(direct.components.data);
      expect(worker.singularValues).toEqual(direct.singularValues);
      expect(worker.explainedVariance).toEqual(direct.explainedVariance);
      expect(worker.explainedVarianceRatio).toEqual(direct.explainedVarianceRatio);
      expect(worker.mean).toEqual(direct.mean);

      const wt = await worker.transform(X);
      expect(wt.data).toEqual(direct.transform(X).data);
    });
  }
});

describe('WorkerPCA behavior', () => {
  const X = gaussian(150, 20, 5);
  const opts = { nComponents: 4, svdSolver: 'randomized' as const, randomState: 0 };

  it('fitTransform, scoreSamples, score, covariance, precision round-trip', async () => {
    const direct = new PCA(opts);
    const directScores = direct.fitTransform(X);
    const worker = new WorkerPCA({ ...opts, worker: workerPort() });
    const scores = await worker.fitTransform(X);
    expect(scores.data).toEqual(directScores.data);
    expect(await worker.scoreSamples(X)).toEqual(direct.scoreSamples(X));
    expect(await worker.score(X)).toBe(direct.score(X));
    expect((await worker.getCovariance()).data).toEqual(direct.getCovariance().data);
    expect((await worker.getPrecision()).data).toEqual(direct.getPrecision().data);
    expect((await worker.inverseTransform(scores)).data).toEqual(
      direct.inverseTransform(directScores).data,
    );
    expect(worker.getFeatureNamesOut()).toEqual(direct.getFeatureNamesOut());
    const info = await worker.info();
    expect(info.backend).toBe('cpu');
    expect(info.webgpuAvailable).toBe(false);
  });

  it('streams progress with snapshots rehydrated as Matrix instances', async () => {
    const events: PCAFitProgress[] = [];
    const worker = new WorkerPCA({
      ...opts,
      iteratedPower: 5,
      worker: workerPort(),
    });
    await worker.fit(X, {
      onProgress: (e) => events.push(e),
      progress: { minIntervalMs: 0, snapshot: { components: true, scores: true } },
    });
    expect(events.filter((e) => e.phase === 'power-iteration')).toHaveLength(5);
    const last = events[events.length - 1];
    expect(last.phase).toBe('finalize');
    expect(last.fraction).toBe(1);
    expect(last.snapshot?.components).toBeInstanceOf(Matrix);
    expect(last.snapshot?.scores).toBeInstanceOf(Matrix);
    expect(last.snapshot?.components.data).toEqual(worker.components.data);
  });

  it('mirror getters throw NotFittedError before any fit resolves', () => {
    const worker = new WorkerPCA({ ...opts, worker: workerPort() });
    expect(() => worker.components).toThrow(NotFittedError);
    expect(() => worker.nSamples).toThrow(NotFittedError);
    expect(() => worker.exportModel()).toThrow(NotFittedError);
  });

  it('aborts mid-fit with an AbortError and worker-side unfitted state', async () => {
    const worker = new WorkerPCA({
      ...opts,
      iteratedPower: 40,
      worker: workerPort(),
    });
    const controller = new AbortController();
    await expect(
      worker.fit(X, {
        signal: controller.signal,
        onProgress: (e) => {
          if (e.step === 2) {
            controller.abort();
          }
        },
        progress: { minIntervalMs: 0 },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    await expect(worker.transform(X)).rejects.toThrow(NotFittedError);
    // The proxy stays usable: a fresh fit succeeds.
    await worker.fit(X);
    expect(worker.nComponents).toBe(4);
  });

  it('a pre-aborted signal rejects without posting the call', async () => {
    const worker = new WorkerPCA({ ...opts, worker: workerPort() });
    const controller = new AbortController();
    const reason = new Error('nope');
    controller.abort(reason);
    await expect(worker.fit(X, { signal: controller.signal })).rejects.toBe(reason);
  });

  it('transfer: true detaches the caller buffer; default copies', async () => {
    const worker = new WorkerPCA({ ...opts, worker: workerPort() });
    const copied = gaussian(50, 8, 9);
    await worker.fit(copied);
    expect(copied.data.byteLength).toBeGreaterThan(0); // copy by default

    const transferred = gaussian(50, 8, 9);
    await worker.fit(transferred, { transfer: true });
    expect(transferred.data.byteLength).toBe(0); // detached
    expect(worker.nComponents).toBe(4);
  });

  it('export/import round-trips through a second worker', async () => {
    const first = new WorkerPCA({ ...opts, worker: workerPort() });
    await first.fit(X);
    const model = first.exportModel();

    const second = new WorkerPCA({ worker: workerPort() });
    await second.importModel(model);
    expect(second.components.data).toEqual(first.components.data);
    expect((await second.transform(X)).data).toEqual((await first.transform(X)).data);

    // fromModel: mirror is available synchronously, before any await.
    const third = WorkerPCA.fromModel(model, { worker: workerPort() });
    expect(third.nComponents).toBe(4);
    expect(third.components.data).toEqual(first.components.data);
    expect((await third.transform(X)).data).toEqual((await first.transform(X)).data);
  });

  it('exportModel returns an independent copy of the mirror', async () => {
    const worker = new WorkerPCA({ ...opts, worker: workerPort() });
    await worker.fit(X);
    const model = worker.exportModel();
    model.components[0] = 777;
    expect(worker.components.data[0]).not.toBe(777);
  });

  it('terminate rejects in-flight calls with WorkerTerminatedError', async () => {
    const worker = new WorkerPCA({
      ...opts,
      iteratedPower: 60,
      worker: workerPort(),
    });
    const fit = worker.fit(gaussian(400, 60, 1));
    worker.terminate();
    await expect(fit).rejects.toBeInstanceOf(WorkerTerminatedError);
    await expect(worker.transform(X)).rejects.toBeInstanceOf(WorkerTerminatedError);
  });

  it('rejects a live RandomState in options with a clear message', () => {
    expect(
      () =>
        new WorkerPCA({
          randomState: new RandomState(1) as unknown as number,
          worker: workerPort(),
        }),
    ).toThrow(/numeric seed/);
  });

  it('two proxies can share one port (connection sharing)', async () => {
    const port = workerPort();
    const a = new WorkerPCA({ ...opts, worker: port });
    const b = new WorkerPCA({ nComponents: 2, svdSolver: 'full', worker: port });
    const [ra, rb] = await Promise.all([a.fit(X), b.fit(X)]);
    expect(ra.nComponents).toBe(4);
    expect(rb.nComponents).toBe(2);
    expect(a.components.data).not.toEqual(b.components.data);
  });
});

describe('WorkerIncrementalPCA', () => {
  const Xi = gaussian(90, 6, 31);
  const rows = (a: number, b: number) => new Matrix(Xi.data.slice(a * 6, b * 6), b - a, 6);

  it('partialFit streams bit-exactly and mirrors the running statistics', async () => {
    const direct = new IncrementalPCA({ nComponents: 2 });
    direct.partialFit(rows(0, 30));
    direct.partialFit(rows(30, 60));

    const worker = new WorkerIncrementalPCA({ nComponents: 2, worker: workerPort() });
    await worker.partialFit(rows(0, 30));
    await worker.partialFit(rows(30, 60));
    expect(worker.nSamplesSeen).toBe(60);
    expect(worker.components.data).toEqual(direct.components.data);
    expect(worker.variance).toEqual(direct.variance);
    expect(worker.mean).toEqual(direct.mean);
    expect((await worker.transform(Xi)).data).toEqual(direct.transform(Xi).data);
    expect(worker.getFeatureNamesOut()).toEqual(direct.getFeatureNamesOut());
  });

  it('fit matches, exposes batchSize, and emits batch progress', async () => {
    const direct = new IncrementalPCA({ nComponents: 2, batchSize: 30 }).fit(Xi);
    const events: PCAFitProgress[] = [];
    const worker = new WorkerIncrementalPCA({
      nComponents: 2,
      batchSize: 30,
      worker: workerPort(),
      onProgress: (e) => events.push(e),
      progress: { minIntervalMs: 0 },
    });
    await worker.fit(Xi);
    expect(worker.batchSize).toBe(30);
    expect(worker.components.data).toEqual(direct.components.data);
    expect(events.filter((e) => e.phase === 'batch')).toHaveLength(3);
    expect(events[events.length - 1].phase).toBe('finalize');
  });

  it('abort keeps the worker-side batch-prefix model; the mirror self-heals', async () => {
    // 10 batches of 9 rows; the abort posted at batch 1 lands within a
    // couple of the worker's yield checkpoints — a few more batches may
    // legitimately complete before it takes effect, but never all of them.
    const controller = new AbortController();
    const worker = new WorkerIncrementalPCA({
      nComponents: 2,
      batchSize: 9,
      worker: workerPort(),
    });
    await expect(
      worker.fit(Xi, {
        signal: controller.signal,
        progress: { minIntervalMs: 0 },
        onProgress: (e) => {
          if (e.phase === 'batch' && e.step === 1) {
            controller.abort();
          }
        },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    // The worker kept the completed-batch model; transform works there…
    const t = await worker.transform(Xi);
    expect(t.cols).toBe(2);
    // …and the next partialFit resumes from it and refreshes the mirror.
    await worker.partialFit(rows(0, 9));
    expect(worker.nSamplesSeen % 9).toBe(0);
    expect(worker.nSamplesSeen).toBeGreaterThanOrEqual(18);
    expect(worker.nSamplesSeen).toBeLessThan(99);
  });

  it('round-trips models into IncrementalPCA.fromModel and back', async () => {
    const worker = new WorkerIncrementalPCA({ nComponents: 2, worker: workerPort() });
    await worker.partialFit(rows(0, 45));
    const local = IncrementalPCA.fromModel(worker.exportModel());
    local.partialFit(rows(45, 90));

    await worker.partialFit(rows(45, 90));
    expect(worker.components.data).toEqual(local.components.data);
    expect(worker.nSamplesSeen).toBe(local.nSamplesSeen);
  });
});

describe('client bundle import discipline', () => {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  const clientJs = join(root, 'dist', 'worker', 'client.js');
  const allowed = new Set([
    'dist/matrix.js',
    'dist/types.js',
    'dist/validation.js',
    'dist/model.js',
    'dist/worker/client.js',
    'dist/worker/protocol.js',
  ]);

  it.skipIf(!existsSync(clientJs))(
    'dist/worker/client.js pulls in no estimator/solver/handler code',
    () => {
      const seen = new Set<string>();
      const visit = (file: string): void => {
        const rel = file.slice(root.length + 1);
        if (seen.has(rel)) {
          return;
        }
        seen.add(rel);
        expect(allowed.has(rel), `unexpected module in client graph: ${rel}`).toBe(true);
        const src = readFileSync(file, 'utf8');
        for (const m of src.matchAll(/from\s+['"](\.[^'"]+)['"]/g)) {
          visit(resolve(dirname(file), m[1]));
        }
        for (const m of src.matchAll(/import\s+['"](\.[^'"]+)['"]/g)) {
          visit(resolve(dirname(file), m[1]));
        }
      };
      visit(clientJs);
      // Sanity: the walk actually saw the graph.
      expect(seen.size).toBeGreaterThanOrEqual(4);
    },
  );
});
