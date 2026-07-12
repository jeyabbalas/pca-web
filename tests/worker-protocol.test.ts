/**
 * Protocol-level tests of the worker handler over a bare MessageChannel —
 * hand-rolled wire messages, no client class involved. Node's MessagePort
 * is an EventTarget and satisfies PCAWorkerPort structurally.
 */
import { afterEach, describe, expect, it } from 'vitest';
import { IncrementalPCA } from '../src/incremental-pca.js';
import { Matrix } from '../src/matrix.js';
import { RandomState } from '../src/numeric/rng.js';
import { PCA } from '../src/pca.js';
import { attachPCAWorker } from '../src/worker/handler.js';
import {
  type ErrorResponse,
  matrixToWire,
  type PCAWorkerRequest,
  type PCAWorkerResponse,
  type ProgressResponse,
  type ResultResponse,
  tightArray,
  type WireMatrix,
  type WorkerEstimatorInfo,
  wireToMatrix,
} from '../src/worker/protocol.js';

function gaussian(n: number, p: number, seed: number): Matrix {
  const data = new Float64Array(n * p);
  new RandomState(seed).standardNormal(data);
  return new Matrix(data, n, p);
}

class Harness {
  readonly channel = new MessageChannel();
  readonly detach: () => void;
  readonly messages: PCAWorkerResponse[] = [];
  private waiters: Array<() => void> = [];
  private nextId = 1;

  constructor(budgetMs = 0) {
    this.detach = attachPCAWorker(this.channel.port1, { budgetMs });
    this.channel.port2.addEventListener('message', (ev) => {
      this.messages.push((ev as MessageEvent).data as PCAWorkerResponse);
      const ws = this.waiters;
      this.waiters = [];
      for (const w of ws) {
        w();
      }
    });
    this.channel.port2.start();
  }

  id(): number {
    return this.nextId++;
  }

  send(msg: PCAWorkerRequest, transfer?: Transferable[]): void {
    this.channel.port2.postMessage(msg, transfer ?? []);
  }

  async waitFor<T extends PCAWorkerResponse>(pred: (m: PCAWorkerResponse) => boolean): Promise<T> {
    for (;;) {
      const found = this.messages.find(pred);
      if (found !== undefined) {
        return found as T;
      }
      await new Promise<void>((resolve) => {
        this.waiters.push(resolve);
      });
    }
  }

  settled(id: number): Promise<ResultResponse | ErrorResponse> {
    return this.waitFor((m) => (m.t === 'result' || m.t === 'error') && m.id === id);
  }

  progressOf(id: number): ProgressResponse[] {
    return this.messages.filter((m): m is ProgressResponse => m.t === 'progress' && m.id === id);
  }

  close(): void {
    this.detach();
    this.channel.port1.close();
    this.channel.port2.close();
  }
}

let harness: Harness | null = null;
afterEach(() => {
  harness?.close();
  harness = null;
});

const X = gaussian(150, 20, 3);

describe('worker protocol over MessageChannel', () => {
  it('posts a ready handshake on attach', async () => {
    harness = new Harness();
    const ready = await harness.waitFor((m) => m.t === 'ready');
    expect(ready).toMatchObject({ t: 'ready', protocolVersion: 1 });
  });

  it('pipelined create + fit piggybacks a model bit-equal to a direct fit', async () => {
    harness = new Harness();
    const createId = harness.id();
    const fitId = harness.id();
    // No round-trip between create and fit — both sent immediately.
    harness.send({
      t: 'create',
      id: createId,
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 4, svdSolver: 'randomized', randomState: 0 },
    });
    harness.send({
      t: 'call',
      id: fitId,
      est: 'e1',
      method: 'fit',
      x: matrixToWire(X),
      returnModel: true,
    });
    const created = await harness.settled(createId);
    expect(created.t).toBe('result');
    const fitted = (await harness.settled(fitId)) as ResultResponse;
    expect(fitted.t).toBe('result');
    expect(fitted.value).toBeNull();
    expect(fitted.model).toBeDefined();

    const direct = new PCA({ nComponents: 4, svdSolver: 'randomized', randomState: 0 }).fit(X);
    const restored = PCA.fromModel(fitted.model as never);
    expect(restored.components.data).toEqual(direct.components.data);
    expect(restored.singularValues).toEqual(direct.singularValues);
  });

  it('fitTransform and the query methods round-trip bit-exactly', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 3, svdSolver: 'full' },
    });
    const direct = new PCA({ nComponents: 3, svdSolver: 'full' });
    const directScores = direct.fitTransform(X);

    const ftId = harness.id();
    harness.send({ t: 'call', id: ftId, est: 'e1', method: 'fitTransform', x: matrixToWire(X) });
    const ft = (await harness.settled(ftId)) as ResultResponse;
    expect(wireToMatrix(ft.value as WireMatrix).data).toEqual(directScores.data);

    const cases = [
      ['transform', matrixToWire(X), direct.transform(X).data],
      ['inverseTransform', matrixToWire(directScores), direct.inverseTransform(directScores).data],
      ['getCovariance', undefined, direct.getCovariance().data],
      ['getPrecision', undefined, direct.getPrecision().data],
    ] as const;
    for (const [method, x, expected] of cases) {
      const id = harness.id();
      harness.send({ t: 'call', id, est: 'e1', method, x });
      const res = (await harness.settled(id)) as ResultResponse;
      expect(res.t, method).toBe('result');
      expect(wireToMatrix(res.value as WireMatrix).data, method).toEqual(expected);
    }

    const ssId = harness.id();
    harness.send({ t: 'call', id: ssId, est: 'e1', method: 'scoreSamples', x: matrixToWire(X) });
    const ss = (await harness.settled(ssId)) as ResultResponse;
    expect((ss.value as { kind: 'array'; data: Float64Array }).data).toEqual(
      direct.scoreSamples(X),
    );

    const scId = harness.id();
    harness.send({ t: 'call', id: scId, est: 'e1', method: 'score', x: matrixToWire(X) });
    const sc = (await harness.settled(scId)) as ResultResponse;
    expect(sc.value).toBe(direct.score(X));
  });

  it('streams every progress event with monotonic seq at minIntervalMs 0', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 3, svdSolver: 'randomized', iteratedPower: 5, randomState: 0 },
    });
    const fitId = harness.id();
    harness.send({
      t: 'call',
      id: fitId,
      est: 'e1',
      method: 'fit',
      x: matrixToWire(X),
      progress: { minIntervalMs: 0, snapshot: { components: true, scores: true } },
    });
    await harness.settled(fitId);
    const events = harness.progressOf(fitId);
    expect(events.map((e) => e.seq)).toEqual(events.map((_, i) => i));
    const phases = events.map((e) => e.event.phase);
    expect(phases.filter((p) => p === 'power-iteration')).toHaveLength(5);
    expect(phases[phases.length - 1]).toBe('finalize');
    const withSnap = events.filter((e) => e.event.snapshot !== undefined);
    expect(withSnap.length).toBeGreaterThanOrEqual(5);
    const snap = withSnap[0].event.snapshot;
    expect(snap?.components.kind).toBe('matrix');
    expect(snap?.components.rows).toBe(3);
    expect(snap?.components.cols).toBe(20);
    expect(snap?.scores?.rows).toBe(150);
    expect(events[events.length - 1].event.fraction).toBe(1);
  });

  it('coalesces with a huge minIntervalMs but always delivers phase boundaries', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 3, svdSolver: 'randomized', iteratedPower: 5, randomState: 0 },
    });
    const fitId = harness.id();
    harness.send({
      t: 'call',
      id: fitId,
      est: 'e1',
      method: 'fit',
      x: matrixToWire(X),
      progress: { minIntervalMs: 1e9 },
    });
    await harness.settled(fitId);
    const events = harness.progressOf(fitId);
    // First event, the coalesced last event of the power-iteration phase
    // (flushed at the boundary), and finalize.
    expect(events.map((e) => [e.event.phase, e.event.step])).toEqual([
      ['power-iteration', 1],
      ['power-iteration', 5],
      ['finalize', 1],
    ]);
    expect(events.map((e) => e.seq)).toEqual([0, 1, 2]);
  });

  it('aborts a running fit mid-flight (AbortError, estimator unfitted)', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 5, svdSolver: 'randomized', iteratedPower: 40, randomState: 0 },
    });
    const fitId = harness.id();
    harness.send({ t: 'call', id: fitId, est: 'e1', method: 'fit', x: matrixToWire(X) });
    harness.send({ t: 'abort', targetId: fitId });
    const settled = await harness.settled(fitId);
    expect(settled.t).toBe('error');
    expect((settled as ErrorResponse).error.name).toBe('AbortError');

    const tId = harness.id();
    harness.send({ t: 'call', id: tId, est: 'e1', method: 'transform', x: matrixToWire(X) });
    const t = await harness.settled(tId);
    expect(t.t).toBe('error');
    expect((t as ErrorResponse).error.name).toBe('NotFittedError');
  });

  it('aborts a queued request without executing it (FIFO preserved)', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 4, svdSolver: 'randomized', iteratedPower: 10, randomState: 0 },
    });
    const fit1 = harness.id();
    const fit2 = harness.id();
    harness.send({ t: 'call', id: fit1, est: 'e1', method: 'fit', x: matrixToWire(X) });
    harness.send({ t: 'call', id: fit2, est: 'e1', method: 'fit', x: matrixToWire(X) });
    harness.send({ t: 'abort', targetId: fit2 });
    const second = await harness.settled(fit2);
    expect(second.t).toBe('error');
    expect((second as ErrorResponse).error.name).toBe('AbortError');
    const first = await harness.settled(fit1);
    expect(first.t).toBe('result');
    // The aborted queued call was rejected before the running one settled.
    expect(harness.messages.indexOf(second)).toBeLessThan(harness.messages.indexOf(first));
  });

  it('marshals errors with names preserved and poisons failed creates', async () => {
    harness = new Harness();
    // Unknown estimator id.
    const unknownId = harness.id();
    harness.send({ t: 'call', id: unknownId, est: 'nope', method: 'info' });
    const unknown = (await harness.settled(unknownId)) as ErrorResponse;
    expect(unknown.error.message).toMatch(/unknown estimator/i);

    // NotFittedError crosses with its name intact.
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: {},
    });
    const nfId = harness.id();
    harness.send({ t: 'call', id: nfId, est: 'e1', method: 'transform', x: matrixToWire(X) });
    const nf = (await harness.settled(nfId)) as ErrorResponse;
    expect(nf.error.name).toBe('NotFittedError');
    expect(nf.error.message).toMatch(/not fitted/);

    // partialFit on a PCA estimator.
    const pfId = harness.id();
    harness.send({ t: 'call', id: pfId, est: 'e1', method: 'partialFit', x: matrixToWire(X) });
    expect(((await harness.settled(pfId)) as ErrorResponse).error.message).toMatch(/ipca/);

    // ipca + webgpu is rejected at create, and follow-ups report the root cause.
    const badCreate = harness.id();
    harness.send({
      t: 'create',
      id: badCreate,
      est: 'bad',
      estimator: 'ipca',
      backend: 'webgpu',
      options: {},
    });
    const followUp = harness.id();
    harness.send({ t: 'call', id: followUp, est: 'bad', method: 'info' });
    const createErr = (await harness.settled(badCreate)) as ErrorResponse;
    const followErr = (await harness.settled(followUp)) as ErrorResponse;
    expect(createErr.error.message).toMatch(/WebGPU backend/);
    expect(followErr.error.message).toBe(createErr.error.message);
  });

  it('never clones or detaches a subarray view parent (tight-slice rule)', async () => {
    const parent = new Float64Array(10_000);
    new RandomState(11).standardNormal(parent);
    const view = parent.subarray(100, 100 + 60 * 5);
    const viewMatrix = new Matrix(view, 60, 5);

    // matrixToWire tight-slices views…
    const wire = matrixToWire(viewMatrix);
    expect(wire.data.buffer).not.toBe(parent.buffer);
    expect(wire.data.length).toBe(300);
    // …and leaves tight arrays alone (zero-copy).
    const tight = new Float64Array(12);
    expect(tightArray(tight)).toBe(tight);

    // Transferring the wire buffer must not touch the parent.
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: { nComponents: 2, svdSolver: 'full' },
    });
    const fitId = harness.id();
    harness.send({ t: 'call', id: fitId, est: 'e1', method: 'fit', x: wire, returnModel: true }, [
      wire.data.buffer as ArrayBuffer,
    ]);
    const res = (await harness.settled(fitId)) as ResultResponse;
    expect(res.t).toBe('result');
    expect(parent.byteLength).toBe(10_000 * 8); // parent never detached
    expect(wire.data.byteLength).toBe(0); // the tight copy was transferred

    // The worker fitted the actual view contents.
    const direct = new PCA({ nComponents: 2, svdSolver: 'full' }).fit(
      new Matrix(parent.slice(100, 100 + 300), 60, 5),
    );
    const restored = PCA.fromModel(res.model as never);
    expect(restored.components.data).toEqual(direct.components.data);
  });

  it('supports IncrementalPCA partialFit streams with model piggyback', async () => {
    harness = new Harness();
    const Xi = gaussian(90, 6, 17);
    const rows = (a: number, b: number) => new Matrix(Xi.data.slice(a * 6, b * 6), b - a, 6);
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'ipca1',
      estimator: 'ipca',
      backend: 'cpu',
      options: { nComponents: 2 },
    });
    const pf1 = harness.id();
    const pf2 = harness.id();
    harness.send({
      t: 'call',
      id: pf1,
      est: 'ipca1',
      method: 'partialFit',
      x: matrixToWire(rows(0, 45)),
      returnModel: true,
    });
    harness.send({
      t: 'call',
      id: pf2,
      est: 'ipca1',
      method: 'partialFit',
      x: matrixToWire(rows(45, 90)),
      returnModel: true,
    });
    await harness.settled(pf1);
    const second = (await harness.settled(pf2)) as ResultResponse;

    const direct = new IncrementalPCA({ nComponents: 2 });
    direct.partialFit(rows(0, 45));
    direct.partialFit(rows(45, 90));
    const restored = IncrementalPCA.fromModel(second.model as never);
    expect(restored.nSamplesSeen).toBe(90);
    expect(restored.components.data).toEqual(direct.components.data);
    expect(restored.variance).toEqual(direct.variance);
  });

  it('importModel rehydrates and dispose forgets the estimator', async () => {
    harness = new Harness();
    const model = new PCA({ nComponents: 3, svdSolver: 'full' }).fit(X).toModel();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: {},
    });
    const impId = harness.id();
    harness.send({ t: 'call', id: impId, est: 'e1', method: 'importModel', model });
    expect((await harness.settled(impId)).t).toBe('result');

    const tId = harness.id();
    harness.send({ t: 'call', id: tId, est: 'e1', method: 'transform', x: matrixToWire(X) });
    const t = (await harness.settled(tId)) as ResultResponse;
    expect(wireToMatrix(t.value as WireMatrix).data).toEqual(
      PCA.fromModel(structuredClone(model)).transform(X).data,
    );

    const dispId = harness.id();
    harness.send({ t: 'dispose', id: dispId, est: 'e1' });
    await harness.settled(dispId);
    const afterId = harness.id();
    harness.send({ t: 'call', id: afterId, est: 'e1', method: 'info' });
    expect(((await harness.settled(afterId)) as ErrorResponse).error.message).toMatch(
      /unknown estimator/i,
    );
  });

  it('reports estimator info (cpu backend, no WebGPU in Node)', async () => {
    harness = new Harness();
    harness.send({
      t: 'create',
      id: harness.id(),
      est: 'e1',
      estimator: 'pca',
      backend: 'cpu',
      options: {},
    });
    const infoId = harness.id();
    harness.send({ t: 'call', id: infoId, est: 'e1', method: 'info' });
    const info = ((await harness.settled(infoId)) as ResultResponse).value as WorkerEstimatorInfo;
    expect(info).toMatchObject({
      kind: 'info',
      estimator: 'pca',
      requestedBackend: 'cpu',
      backend: 'cpu',
      gpuAdapterInfo: null,
      webgpuAvailable: false,
    });
  });
});
