/**
 * CPU-fallback contract of the WebGPU frontend, in an environment without
 * WebGPU (Node): WebGPUPCA must detect the absence, report backend 'cpu',
 * and produce BIT-IDENTICAL results to the plain PCA class — the fallback
 * literally runs the same code. (The GPU-executed counterpart of this suite
 * lives in tests/browser/, run by `npm run test:browser`.)
 */
import { describe, expect, it } from 'vitest';
import { Matrix, PCA, type PCAFitProgress, RandomState } from '../src/index.js';
import { isWebGPUSupported, WebGPUPCA } from '../src/webgpu/index.js';

function demoData(n: number, p: number, seed: number): Matrix {
  const rng = new RandomState(seed);
  const data = new Float64Array(n * p);
  rng.standardNormal(data);
  return new Matrix(data, n, p);
}

describe('WebGPUPCA in a WebGPU-less environment', () => {
  it('detects that WebGPU is unavailable in Node', () => {
    expect(isWebGPUSupported()).toBe(false);
    expect(WebGPUPCA.isSupported()).toBe(false);
  });

  it('falls back to the CPU with bit-identical results (randomized)', async () => {
    const X = demoData(600, 500, 42); // large enough to want the GPU
    const opts = { nComponents: 8, svdSolver: 'randomized' as const, randomState: 0 };
    const gpu = new WebGPUPCA(opts);
    await gpu.fit(X);
    expect(gpu.backend).toBe('cpu');
    expect(gpu.gpuAdapterInfo).toBeNull();

    const cpu = new PCA(opts).fit(X);
    expect(gpu.components.data).toEqual(cpu.components.data);
    expect(gpu.singularValues).toEqual(cpu.singularValues);
    expect(gpu.explainedVariance).toEqual(cpu.explainedVariance);
    expect(gpu.mean).toEqual(cpu.mean);
    expect(gpu.noiseVariance).toBe(cpu.noiseVariance);
    expect(gpu.resolvedSvdSolver).toBe('randomized');

    const gt = await gpu.transform(X);
    const ct = cpu.transform(X);
    expect(gt.data).toEqual(ct.data);
    expect(gpu.inverseTransform(gt).data).toEqual(cpu.inverseTransform(ct).data);
  });

  it('falls back with bit-identical fitTransform (covariance_eigh via auto)', async () => {
    const X = demoData(1200, 60, 7); // auto → covariance_eigh (n ≥ 10p)
    const gpu = new WebGPUPCA({ nComponents: 5 });
    const cpu = new PCA({ nComponents: 5 });
    const gt = await gpu.fitTransform(X);
    const ct = cpu.fitTransform(X);
    expect(gpu.backend).toBe('cpu');
    expect(gpu.resolvedSvdSolver).toBe('covariance_eigh');
    expect(gt.data).toEqual(ct.data);
    expect(gpu.getCovariance().data).toEqual(cpu.getCovariance().data);
    expect(gpu.scoreSamples(X)).toEqual(cpu.scoreSamples(X));
  });

  it('delegates full/arpack fits to the CPU path unchanged', async () => {
    const X = demoData(80, 20, 3);
    const gpu = new WebGPUPCA({ nComponents: 4, svdSolver: 'arpack', randomState: 42 });
    await gpu.fit(X);
    expect(gpu.backend).toBe('cpu');
    const cpu = new PCA({ nComponents: 4, svdSolver: 'arpack', randomState: 42 }).fit(X);
    expect(gpu.components.data).toEqual(cpu.components.data);
  });

  it('propagates validation errors before any fit work', async () => {
    const X = demoData(10, 5, 1);
    await expect(new WebGPUPCA({ nComponents: 9, svdSolver: 'full' }).fit(X)).rejects.toThrow(
      /between 0 and min/,
    );
    const bad = demoData(10, 5, 2);
    bad.data[3] = Number.NaN;
    await expect(new WebGPUPCA().fit(bad)).rejects.toThrow(/NaN/);
  });
});

describe('WebGPUPCA observer and abort on the CPU fallback', () => {
  it('threads progress events through fit (identical sequence to plain fitAsync)', async () => {
    const X = demoData(600, 500, 42);
    const opts = { nComponents: 8, svdSolver: 'randomized' as const, randomState: 0 };

    const gpuEvents: PCAFitProgress[] = [];
    const gpu = new WebGPUPCA(opts);
    await gpu.fit(X, {
      onProgress: (e) => gpuEvents.push(e),
      snapshot: { components: true, scores: true },
    });
    expect(gpu.backend).toBe('cpu');

    const cpuEvents: PCAFitProgress[] = [];
    const cpu = new PCA(opts);
    await cpu.fitAsync(X, {
      onProgress: (e) => cpuEvents.push(e),
      snapshot: { components: true, scores: true },
    });

    expect(gpuEvents.length).toBe(cpuEvents.length);
    gpuEvents.forEach((e, i) => {
      expect(e.phase).toBe(cpuEvents[i].phase);
      expect(e.step).toBe(cpuEvents[i].step);
      expect(e.fraction).toBe(cpuEvents[i].fraction);
      expect(e.snapshot?.components.data).toEqual(cpuEvents[i].snapshot?.components.data);
      expect(e.snapshot?.scores?.data).toEqual(cpuEvents[i].snapshot?.scores?.data);
    });
    expect(gpu.components.data).toEqual(cpu.components.data);
  });

  it('abort rejects with AbortError and does not fall through to a CPU refit', async () => {
    const X = demoData(600, 500, 7);
    const controller = new AbortController();
    const gpu = new WebGPUPCA({ nComponents: 8, svdSolver: 'randomized', randomState: 0 });
    let events = 0;
    await expect(
      gpu.fit(X, {
        signal: controller.signal,
        onProgress: () => {
          events++;
          controller.abort();
        },
      }),
    ).rejects.toMatchObject({ name: 'AbortError' });
    // Exactly one event: the fit stopped instead of restarting on the CPU.
    expect(events).toBe(1);
    expect(() => gpu.components).toThrow(/not fitted/);
  });

  it('pre-aborted fit rejects before any work', async () => {
    const X = demoData(100, 20, 3);
    const controller = new AbortController();
    controller.abort();
    const gpu = new WebGPUPCA({ nComponents: 4 });
    await expect(gpu.fit(X, { signal: controller.signal })).rejects.toMatchObject({
      name: 'AbortError',
    });
  });

  it('fitTransform threads options and stays bit-identical', async () => {
    const X = demoData(1200, 60, 7);
    const events: PCAFitProgress[] = [];
    const gpu = new WebGPUPCA({ nComponents: 5 });
    const gt = await gpu.fitTransform(X, { onProgress: (e) => events.push(e) });
    const ct = new PCA({ nComponents: 5 }).fitTransform(X);
    expect(gt.data).toEqual(ct.data);
    expect(events[events.length - 1].phase).toBe('finalize');
  });

  it('concurrent fits on one instance throw', async () => {
    const X = demoData(100, 20, 5);
    const gpu = new WebGPUPCA({ nComponents: 4 });
    const first = gpu.fit(X);
    await expect(gpu.fitTransform(X)).rejects.toThrow(/already fitting/);
    await first;
    expect(gpu.nComponents).toBe(4);
  });
});
