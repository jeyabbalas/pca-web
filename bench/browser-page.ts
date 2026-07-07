/**
 * In-browser CPU vs WebGPU benchmark page, driven by bench/browser.mjs.
 * Times PCA (CPU) against WebGPUPCA (GPU) on identical data and reports
 * medians; the GPU timing includes upload, dispatch, and readback.
 */
import { Matrix, PCA, RandomState } from '../src/index.js';
import { WebGPUPCA } from '../src/webgpu/index.js';

interface BenchRow {
  id: string;
  n: number;
  p: number;
  solver: string;
  cpuMs: number;
  gpuMs: number;
  backend: string;
  maxRelDiff: number;
}

interface BenchResults {
  adapter: string | null;
  rows: BenchRow[];
  error?: string;
}

declare global {
  interface Window {
    __PCA_BENCH__?: BenchResults;
  }
}

function data(n: number, p: number, seed: number, f32 = false): Matrix {
  const rng = new RandomState(seed);
  const x = new Float64Array(n * p);
  rng.standardNormal(x);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      x[i * p + j] = x[i * p + j] * (1 + (j % 7)) + (j % 5);
    }
  }
  return new Matrix(f32 ? Float32Array.from(x) : x, n, p);
}

async function medianAsync(fn: () => Promise<unknown>, reps: number): Promise<number> {
  const times: number[] = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    await fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

function relDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let scale = 0;
  for (let i = 0; i < b.length; i++) {
    scale = Math.max(scale, Math.abs(b[i]));
  }
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / (scale || 1));
  }
  return worst;
}

const CONFIGS = [
  {
    id: 'cov_50000x200',
    n: 50_000,
    p: 200,
    opts: { nComponents: 16, svdSolver: 'covariance_eigh' as const },
  },
  {
    id: 'cov_100000x100',
    n: 100_000,
    p: 100,
    opts: { nComponents: 16, svdSolver: 'covariance_eigh' as const },
  },
  {
    id: 'rand_20000x500_nc16',
    n: 20_000,
    p: 500,
    opts: { nComponents: 16, svdSolver: 'randomized' as const, randomState: 0 },
  },
  {
    id: 'rand_5000x2000_nc32',
    n: 5_000,
    p: 2_000,
    opts: { nComponents: 32, svdSolver: 'randomized' as const, randomState: 0 },
  },
  {
    id: 'cov_f32_50000x200',
    n: 50_000,
    p: 200,
    f32: true,
    opts: { nComponents: 16, svdSolver: 'covariance_eigh' as const },
  },
];

async function main(): Promise<BenchResults> {
  const rows: BenchRow[] = [];
  let adapter: string | null = null;
  for (const cfg of CONFIGS) {
    const X = data(cfg.n, cfg.p, 42, cfg.f32 === true);
    const reps = 3;

    const cpu = new PCA(cfg.opts);
    const cpuMs = await medianAsync(async () => cpu.fit(X), reps);

    const gpu = new WebGPUPCA({ ...cfg.opts, minGpuElements: 1 });
    const gpuMs = await medianAsync(() => gpu.fit(X), reps);
    adapter = gpu.gpuAdapterInfo ?? adapter;

    rows.push({
      id: cfg.id,
      n: cfg.n,
      p: cfg.p,
      solver: cfg.opts.svdSolver,
      cpuMs,
      gpuMs,
      backend: gpu.backend,
      maxRelDiff: relDiff(gpu.components.data, cpu.components.data),
    });
    gpu.dispose();
  }
  return { adapter, rows };
}

main()
  .then((r) => {
    window.__PCA_BENCH__ = r;
  })
  .catch((err) => {
    window.__PCA_BENCH__ = { adapter: null, rows: [], error: String(err) };
  });
