/**
 * In-browser Web Worker battery, driven by tests/browser/run-worker.mjs:
 * a REAL `new Worker(...)` running the packaged entry, exercised through
 * the WorkerPCA/WorkerIncrementalPCA client proxies. Verifies progress
 * observed on the main thread, bit-equality against in-page estimators,
 * abort, transfer semantics, the default worker factory, and (as a smoke
 * report) whether WebGPU-in-worker executed.
 *
 * Results land on window.__PCA_WORKER_RESULTS__.
 */
import { IncrementalPCA, Matrix, PCA, RandomState } from '../../src/index.js';
import type { PCAFitProgress } from '../../src/progress.js';
import { WorkerIncrementalPCA, WorkerPCA } from '../../src/worker/client.js';
import type { PCAWorkerLike } from '../../src/worker/protocol.js';

interface CaseResult {
  id: string;
  pass: boolean;
  detail: string;
}

interface Results {
  cases: CaseResult[];
  failures: number;
  /** Backend the webgpu-smoke fit actually used, or null if it errored. */
  workerGpuBackend: string | null;
}

declare global {
  interface Window {
    __PCA_WORKER_RESULTS__?: Results;
  }
}

function gaussian(n: number, p: number, seed: number): Matrix {
  const rng = new RandomState(seed);
  const data = new Float64Array(n * p);
  rng.standardNormal(data);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      data[i * p + j] = data[i * p + j] * (1 + (j % 7)) + 2 * (j % 5);
    }
  }
  return new Matrix(data, n, p);
}

function bitEqual(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (!Object.is(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

function maxRelDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    const scale = Math.max(1e-30, Math.abs(b[i]));
    worst = Math.max(worst, d / scale);
  }
  return worst;
}

function spawn(): PCAWorkerLike {
  return new Worker('/worker-entry.js', {
    type: 'module',
    name: 'pca-web-test',
  }) as unknown as PCAWorkerLike;
}

async function main(): Promise<Results> {
  const results: Results = { cases: [], failures: 0, workerGpuBackend: null };
  const record = (id: string, pass: boolean, detail: string): void => {
    results.cases.push({ id, pass, detail });
    if (!pass) {
      results.failures++;
    }
  };

  const X = gaussian(300, 40, 11);
  const opts = { nComponents: 6, svdSolver: 'randomized' as const, randomState: 0 };

  // 1. results bit-equal to the in-page PCA
  try {
    const direct = new PCA(opts);
    const directScores = direct.fitTransform(X);
    const worker = new WorkerPCA({ ...opts, worker: spawn() });
    const scores = await worker.fitTransform(X);
    const pass =
      bitEqual(worker.components.data, direct.components.data) &&
      bitEqual(worker.singularValues, direct.singularValues) &&
      bitEqual(scores.data, directScores.data) &&
      bitEqual((await worker.transform(X)).data, direct.transform(X).data);
    record('worker.bit-equal', pass, pass ? 'components/scores/transform identical' : 'MISMATCH');
    worker.terminate();
  } catch (err) {
    record('worker.bit-equal', false, String(err));
  }

  // 2. progress observed on the main thread (with snapshots), fraction monotone
  try {
    const events: PCAFitProgress[] = [];
    const worker = new WorkerPCA({ ...opts, iteratedPower: 6, worker: spawn() });
    await worker.fit(X, {
      onProgress: (e) => events.push(e),
      progress: { minIntervalMs: 0, snapshot: { components: true, scores: true } },
    });
    const power = events.filter((e) => e.phase === 'power-iteration');
    const fractions = events.map((e) => e.fraction).filter((f): f is number => f !== null);
    const monotone = fractions.every((f, i) => i === 0 || f >= fractions[i - 1]);
    const last = events[events.length - 1];
    const pass =
      power.length === 6 &&
      monotone &&
      last.phase === 'finalize' &&
      last.fraction === 1 &&
      last.snapshot !== undefined &&
      last.snapshot.components instanceof Matrix &&
      last.snapshot.scores instanceof Matrix &&
      bitEqual(last.snapshot.components.data, worker.components.data);
    record(
      'worker.progress-main-thread',
      pass,
      `events=${events.length}, power=${power.length}, monotone=${monotone}, final snapshot=${last.snapshot !== undefined}`,
    );
    worker.terminate();
  } catch (err) {
    record('worker.progress-main-thread', false, String(err));
  }

  // 3. abort rejects with AbortError; the proxy remains usable.
  // The fit must span several of the worker's ~50ms slices so the abort
  // message has yield points to land on.
  try {
    const worker = new WorkerPCA({ ...opts, iteratedPower: 40, worker: spawn() });
    const controller = new AbortController();
    let name = 'none';
    try {
      await worker.fit(gaussian(1500, 300, 12), {
        signal: controller.signal,
        progress: { minIntervalMs: 0 },
        onProgress: (e) => {
          if (e.step >= 2) {
            controller.abort();
          }
        },
      });
    } catch (err) {
      name = (err as Error).name;
    }
    await worker.fit(X);
    const pass = name === 'AbortError' && worker.nComponents === 6;
    record('worker.abort', pass, `rejected with ${name}; refit ok=${worker.nComponents === 6}`);
    worker.terminate();
  } catch (err) {
    record('worker.abort', false, String(err));
  }

  // 4. transfer: true detaches the caller's buffer; default copies
  try {
    const worker = new WorkerPCA({ ...opts, worker: spawn() });
    const copied = gaussian(100, 10, 13);
    await worker.fit(copied);
    const copiedIntact = copied.data.byteLength > 0;
    const transferred = gaussian(100, 10, 13);
    await worker.fit(transferred, { transfer: true });
    const detached = transferred.data.byteLength === 0;
    record(
      'worker.transfer',
      copiedIntact && detached,
      `default copy intact=${copiedIntact}, transferred detached=${detached}`,
    );
    worker.terminate();
  } catch (err) {
    record('worker.transfer', false, String(err));
  }

  // 5. the default worker factory (new URL('./worker.js', import.meta.url))
  try {
    const worker = new WorkerPCA(opts); // no { worker } — default factory
    await worker.fit(X);
    const direct = new PCA(opts).fit(X);
    const pass = bitEqual(worker.components.data, direct.components.data);
    record('worker.default-factory', pass, pass ? 'spawned and bit-equal' : 'MISMATCH');
    worker.terminate();
  } catch (err) {
    record('worker.default-factory', false, String(err));
  }

  // 6. IncrementalPCA partialFit stream with a live mirror
  try {
    const Xi = gaussian(120, 8, 14);
    const rows = (a: number, b: number) => new Matrix(Xi.data.slice(a * 8, b * 8), b - a, 8);
    const direct = new IncrementalPCA({ nComponents: 3 });
    direct.partialFit(rows(0, 60));
    direct.partialFit(rows(60, 120));
    const worker = new WorkerIncrementalPCA({ nComponents: 3, worker: spawn() });
    await worker.partialFit(rows(0, 60));
    await worker.partialFit(rows(60, 120));
    const pass =
      worker.nSamplesSeen === 120 &&
      bitEqual(worker.components.data, direct.components.data) &&
      bitEqual(worker.variance, direct.variance) &&
      bitEqual((await worker.transform(Xi)).data, direct.transform(Xi).data);
    record('worker.ipca-stream', pass, `nSamplesSeen=${worker.nSamplesSeen}`);
    worker.terminate();
  } catch (err) {
    record('worker.ipca-stream', false, String(err));
  }

  // 7. WebGPU-in-worker smoke: report which backend executed
  try {
    const Xg = gaussian(2000, 300, 15); // 600k elements — over the GPU floor
    const gopts = { nComponents: 8, svdSolver: 'randomized' as const, randomState: 1 };
    const worker = new WorkerPCA({ ...gopts, backend: 'webgpu', worker: spawn() });
    await worker.fit(Xg);
    const info = await worker.info();
    results.workerGpuBackend = info.backend;
    const direct = new PCA(gopts).fit(Xg);
    const rel = maxRelDiff(worker.components.data, direct.components.data);
    // On the GPU: within df64 equivalence; on the CPU fallback: bit-equal.
    const pass = info.backend === 'webgpu' ? rel < 1e-6 : rel === 0;
    record(
      'worker.webgpu-smoke',
      pass,
      `backend=${info.backend} (webgpuAvailable=${info.webgpuAvailable}, adapter=${info.gpuAdapterInfo ?? 'n/a'}), maxRel vs CPU=${rel.toExponential(2)}`,
    );
    worker.terminate();
  } catch (err) {
    record('worker.webgpu-smoke', false, String(err));
  }

  return results;
}

main()
  .then((results) => {
    window.__PCA_WORKER_RESULTS__ = results;
  })
  .catch((err) => {
    window.__PCA_WORKER_RESULTS__ = {
      cases: [{ id: 'main', pass: false, detail: String(err) }],
      failures: 1,
      workerGpuBackend: null,
    };
  });
