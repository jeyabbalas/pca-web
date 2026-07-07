/**
 * In-browser GPU test battery, driven by tests/browser/run.mjs (Playwright).
 *
 * Three sections:
 *  1. engine   — adapter acquisition + df64 precision self-check report
 *  2. equiv    — CPU↔GPU numerical equivalence on generated data
 *  3. fixtures — WebGPUPCA vs scikit-learn 1.9.0 fixture subset
 *               (covariance_eigh + randomized cases; the GPU-accelerated
 *               solvers), using the CPU parity suite's tolerance classes.
 *
 * Every case records which backend actually executed; a case that silently
 * ran on the CPU counts as NOT-EXECUTED for GPU purposes and fails the run.
 * Results land on window.__PCA_RESULTS__ for the runner to collect.
 */
import { Matrix, PCA, RandomState } from '../../src/index.js';
import type { FloatArray } from '../../src/types.js';
import { GpuEngine, isWebGPUSupported, WebGPUPCA } from '../../src/webgpu/index.js';

interface Tol {
  atol: number;
  rtol: number;
}

interface CaseResult {
  id: string;
  section: 'engine' | 'equiv' | 'fixtures';
  pass: boolean;
  backend: string;
  detail: string;
}

interface Results {
  supported: boolean;
  adapter: string | null;
  dsRelError: number | null;
  gpuExecuted: boolean;
  cases: CaseResult[];
  failures: number;
}

declare global {
  interface Window {
    __PCA_RESULTS__?: Results;
  }
}

// ---------------------------------------------------------------------------
// comparison helpers
// ---------------------------------------------------------------------------

function maxDiff(actual: ArrayLike<number>, expected: ArrayLike<number>, tol: Tol) {
  let maxAbs = 0;
  let maxRel = 0;
  let worstRatio = 0;
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(actual[i] - expected[i]);
    const bound = tol.atol + tol.rtol * Math.abs(expected[i]);
    maxAbs = Math.max(maxAbs, d);
    if (expected[i] !== 0) {
      maxRel = Math.max(maxRel, d / Math.abs(expected[i]));
    }
    worstRatio = Math.max(worstRatio, Number.isNaN(d) ? Number.POSITIVE_INFINITY : d / bound);
  }
  return { maxAbs, maxRel, ok: worstRatio <= 1, worstRatio };
}

function fmt(x: number): string {
  return Number.isFinite(x) ? x.toExponential(2) : String(x);
}

function gaussian(n: number, p: number, seed: number, dtype: 'float64' | 'float32'): Matrix {
  const rng = new RandomState(seed);
  const data = new Float64Array(n * p);
  rng.standardNormal(data);
  // Low-rank structure + offsets so PCA has something to find.
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      data[i * p + j] = data[i * p + j] * (1 + (j % 7)) + 2 * (j % 5);
    }
  }
  return new Matrix(dtype === 'float32' ? Float32Array.from(data) : data, n, p);
}

// ---------------------------------------------------------------------------
// fixture loading (fetch-based mirror of tests/helpers/fixtures.ts)
// ---------------------------------------------------------------------------

interface ArrayRef {
  dtype: 'float64' | 'float32';
  shape: number[];
  offset: number;
}

interface FixtureCase {
  id: string;
  params: Record<string, unknown>;
  arrays: Record<string, ArrayRef>;
  scalars: Record<string, unknown>;
  flags: Record<string, unknown>;
}

async function loadPcaSuite(): Promise<{ cases: FixtureCase[]; bin: ArrayBuffer }> {
  const manifest = await (await fetch('/fixtures/pca/manifest.json')).json();
  const bin = await (await fetch('/fixtures/pca/data.bin')).arrayBuffer();
  return { cases: manifest.cases, bin };
}

function getArray(bin: ArrayBuffer, ref: ArrayRef): FloatArray {
  const count = ref.shape.reduce((a, b) => a * b, 1);
  return ref.dtype === 'float64'
    ? new Float64Array(bin.slice(ref.offset, ref.offset + count * 8))
    : new Float32Array(bin.slice(ref.offset, ref.offset + count * 4));
}

function getMatrix(bin: ArrayBuffer, ref: ArrayRef): Matrix {
  return new Matrix(getArray(bin, ref), ref.shape[0], ref.shape[1]);
}

function fixtureRank(sv: FloatArray): number {
  if (sv.length === 0) {
    return 0;
  }
  const cut = sv[0] * 1e-7;
  let r = 0;
  while (r < sv.length && sv[r] > cut) {
    r++;
  }
  return r;
}

// Tolerance classes mirroring tests/pca-parity.test.ts (GPU df64 GEMM error
// is orders of magnitude below every class bound).
function fixtureTols(c: FixtureCase): Record<string, Tol> {
  if (c.flags.dtype === 'float32') {
    return {
      mean: { atol: 1e-5, rtol: 1e-5 },
      components: { atol: 2e-3, rtol: 2e-3 },
      explainedVariance: { atol: 1e-4, rtol: 2e-3 },
      ratio: { atol: 1e-5, rtol: 2e-3 },
      singularValues: { atol: 1e-4, rtol: 2e-3 },
      noiseVariance: { atol: 1e-6, rtol: 1e-3 },
      transform: { atol: 5e-3, rtol: 2e-3 },
    };
  }
  const solver = (c.flags.expected_solver ?? c.params.svd_solver) as string;
  if (solver === 'randomized') {
    return {
      mean: { atol: 1e-12, rtol: 1e-12 },
      components: { atol: 1e-7, rtol: 1e-5 },
      explainedVariance: { atol: 1e-9, rtol: 1e-7 },
      ratio: { atol: 1e-10, rtol: 1e-7 },
      singularValues: { atol: 1e-9, rtol: 1e-7 },
      noiseVariance: { atol: 1e-10, rtol: 1e-4 },
      transform: { atol: 1e-6, rtol: 1e-5 },
    };
  }
  return {
    mean: { atol: 1e-12, rtol: 1e-12 },
    // Near-degenerate trailing Gram eigenvectors rotate by (GEMM diff)/gap;
    // the GPU's ~1e-13 Gram differences can amplify to ~1e-8 in components
    // of full-spectrum covariance_eigh fits (CPU parity uses 1e-9 here).
    components: { atol: 1e-7, rtol: 1e-7 },
    explainedVariance: { atol: 1e-10, rtol: 1e-9 },
    ratio: { atol: 1e-12, rtol: 1e-9 },
    singularValues: { atol: 2e-6, rtol: 1e-9 },
    noiseVariance: { atol: 1e-11, rtol: 1e-9 },
    transform: { atol: 1e-7, rtol: 1e-6 },
  };
}

function toOptions(params: Record<string, unknown>) {
  return {
    ...('n_components' in params && { nComponents: params.n_components as number }),
    ...('svd_solver' in params && { svdSolver: params.svd_solver as never }),
    ...('whiten' in params && { whiten: params.whiten as boolean }),
    ...('iterated_power' in params && { iteratedPower: params.iterated_power as number }),
    ...('n_oversamples' in params && { nOversamples: params.n_oversamples as number }),
    ...('power_iteration_normalizer' in params && {
      powerIterationNormalizer: params.power_iteration_normalizer as never,
    }),
    ...('random_state' in params && { randomState: params.random_state as number }),
    ...('tol' in params && { tol: params.tol as number }),
  };
}

// ---------------------------------------------------------------------------
// the battery
// ---------------------------------------------------------------------------

async function main(): Promise<Results> {
  const results: Results = {
    supported: isWebGPUSupported(),
    adapter: null,
    dsRelError: null,
    gpuExecuted: false,
    cases: [],
    failures: 0,
  };
  const record = (r: CaseResult) => {
    results.cases.push(r);
    if (!r.pass) {
      results.failures++;
    }
  };

  // --- 1. engine ----------------------------------------------------------
  const engine = await GpuEngine.create({});
  if (engine === null) {
    record({
      id: 'engine.create',
      section: 'engine',
      pass: false,
      backend: 'none',
      detail: GpuEngine.lastCreateError ?? 'unknown failure',
    });
    return results;
  }
  results.adapter = engine.adapterInfo;
  results.dsRelError = engine.measuredRelError;
  record({
    id: 'engine.create',
    section: 'engine',
    pass: engine.measuredRelError < 1e-10,
    backend: 'webgpu',
    detail: `adapter="${engine.adapterInfo}" df64 self-check relErr=${fmt(engine.measuredRelError)}`,
  });
  engine.dispose();

  // --- 2. CPU↔GPU equivalence ---------------------------------------------
  // Documented equivalence bounds for the df64 GEMM path (f64 data).
  const EQ_F64: Tol = { atol: 1e-10, rtol: 1e-9 };
  const EQ_F64_TRANSFORM: Tol = { atol: 1e-9, rtol: 1e-8 };
  const EQ_F32: Tol = { atol: 1e-5, rtol: 1e-5 };

  interface EquivSpec {
    id: string;
    n: number;
    p: number;
    dtype: 'float64' | 'float32';
    options: Record<string, unknown>;
  }
  const specs: EquivSpec[] = [
    {
      id: 'cov_4000x120_nc10',
      n: 4000,
      p: 120,
      dtype: 'float64',
      options: { nComponents: 10, svdSolver: 'covariance_eigh' },
    },
    {
      id: 'cov_3000x80_whiten',
      n: 3000,
      p: 80,
      dtype: 'float64',
      options: { nComponents: 8, svdSolver: 'covariance_eigh', whiten: true },
    },
    {
      id: 'rand_2000x600_nc12',
      n: 2000,
      p: 600,
      dtype: 'float64',
      options: { nComponents: 12, svdSolver: 'randomized', randomState: 42 },
    },
    {
      id: 'rand_wide600x2000_nc10',
      n: 600,
      p: 2000,
      dtype: 'float64',
      options: { nComponents: 10, svdSolver: 'randomized', randomState: 0 },
    },
    {
      id: 'rand_2000x600_QR_whiten',
      n: 2000,
      p: 600,
      dtype: 'float64',
      options: {
        nComponents: 12,
        svdSolver: 'randomized',
        powerIterationNormalizer: 'QR',
        whiten: true,
        randomState: 7,
      },
    },
    {
      id: 'auto_5000x300_nc16',
      n: 5000,
      p: 300,
      dtype: 'float64',
      options: { nComponents: 16, randomState: 42 }, // auto → covariance_eigh
    },
    {
      id: 'cov_f32_4000x120_nc10',
      n: 4000,
      p: 120,
      dtype: 'float32',
      options: { nComponents: 10, svdSolver: 'covariance_eigh' },
    },
    {
      id: 'rand_f32_2000x600_nc12',
      n: 2000,
      p: 600,
      dtype: 'float32',
      options: { nComponents: 12, svdSolver: 'randomized', randomState: 42 },
    },
  ];

  for (const spec of specs) {
    const X = gaussian(spec.n, spec.p, 1000 + spec.n + spec.p, spec.dtype);
    const XTest = gaussian(64, spec.p, 2000 + spec.p, spec.dtype);
    const isF32 = spec.dtype === 'float32';
    const tolFit = isF32 ? EQ_F32 : EQ_F64;
    const tolTr = isF32 ? EQ_F32 : EQ_F64_TRANSFORM;
    try {
      const cpu = new PCA(spec.options as never);
      const cpuFt = cpu.fitTransform(X.copy());
      const gpu = new WebGPUPCA({ ...(spec.options as object), minGpuElements: 1 } as never);
      const gpuFt = await gpu.fitTransform(X.copy());

      if (gpu.backend !== 'webgpu') {
        record({
          id: `equiv.${spec.id}`,
          section: 'equiv',
          pass: false,
          backend: gpu.backend,
          detail: 'GPU path did not execute (fell back to CPU)',
        });
        continue;
      }
      results.gpuExecuted = true;

      const checks: [string, ReturnType<typeof maxDiff>][] = [
        ['components', maxDiff(gpu.components.data, cpu.components.data, tolFit)],
        ['singularValues', maxDiff(gpu.singularValues, cpu.singularValues, tolFit)],
        ['explainedVariance', maxDiff(gpu.explainedVariance, cpu.explainedVariance, tolFit)],
        ['mean', maxDiff(gpu.mean, cpu.mean, tolFit)],
        [
          'noiseVariance',
          maxDiff(
            [gpu.noiseVariance],
            [cpu.noiseVariance],
            isF32 ? EQ_F32 : { atol: 1e-10, rtol: 1e-6 },
          ),
        ],
        ['fitTransform', maxDiff(gpuFt.data, cpuFt.data, tolTr)],
        ['transform', maxDiff((await gpu.transform(XTest)).data, cpu.transform(XTest).data, tolTr)],
      ];
      const bad = checks.filter(([, d]) => !d.ok);
      const worst = checks.reduce((a, b) => (a[1].worstRatio > b[1].worstRatio ? a : b));
      record({
        id: `equiv.${spec.id}`,
        section: 'equiv',
        pass: bad.length === 0,
        backend: 'webgpu',
        detail:
          bad.length === 0
            ? `ok; worst=${worst[0]} maxAbs=${fmt(worst[1].maxAbs)} maxRel=${fmt(worst[1].maxRel)}`
            : bad
                .map(([name, d]) => `${name}: maxAbs=${fmt(d.maxAbs)} maxRel=${fmt(d.maxRel)}`)
                .join('; '),
      });
      gpu.dispose();
    } catch (err) {
      record({
        id: `equiv.${spec.id}`,
        section: 'equiv',
        pass: false,
        backend: 'error',
        detail: String(err),
      });
    }
  }

  // --- 3. sklearn fixtures on the GPU --------------------------------------
  // covariance_eigh and randomized cases only (the GPU-accelerated solvers).
  // powerIterationNormalizer='none' is excluded: sklearn documents it as
  // numerically unstable, and its trailing components amplify even 1-ulp
  // input differences by ~1e12 — no cross-backend comparison is meaningful
  // there (see docs/LESSONS.md).
  try {
    const { cases, bin } = await loadPcaSuite();
    const gpuCases = cases.filter((c) => {
      const solver = (c.flags.expected_solver ?? c.params.svd_solver) as string;
      const normalizer = c.params.power_iteration_normalizer as string | undefined;
      return (solver === 'covariance_eigh' || solver === 'randomized') && normalizer !== 'none';
    });
    for (const c of gpuCases) {
      try {
        const X = getMatrix(bin, c.arrays.X);
        const XTest = getMatrix(bin, c.arrays.X_test);
        const t = fixtureTols(c);
        const gpu = new WebGPUPCA({ ...toOptions(c.params), minGpuElements: 1 });
        await gpu.fit(X);
        if (gpu.backend !== 'webgpu') {
          record({
            id: `fixtures.${c.id}`,
            section: 'fixtures',
            pass: false,
            backend: gpu.backend,
            detail: 'GPU path did not execute',
          });
          continue;
        }
        results.gpuExecuted = true;

        const svFixture = getArray(bin, c.arrays.singular_values);
        const k = gpu.nComponents;
        const p = gpu.nFeaturesIn;
        const rank = Math.min(fixtureRank(svFixture), k);
        const expectedComp = getArray(bin, c.arrays.components);

        const checks: [string, ReturnType<typeof maxDiff>][] = [
          ['mean', maxDiff(gpu.mean, getArray(bin, c.arrays.mean), t.mean)],
          [
            'components',
            maxDiff(
              (gpu.components.data as FloatArray).subarray
                ? (gpu.components.data as Float64Array).subarray(0, rank * p)
                : gpu.components.data,
              (expectedComp as Float64Array).subarray(0, rank * p),
              t.components,
            ),
          ],
          [
            'explainedVariance',
            maxDiff(
              gpu.explainedVariance,
              getArray(bin, c.arrays.explained_variance),
              t.explainedVariance,
            ),
          ],
          [
            'ratio',
            maxDiff(
              gpu.explainedVarianceRatio,
              getArray(bin, c.arrays.explained_variance_ratio),
              t.ratio,
            ),
          ],
          ['singularValues', maxDiff(gpu.singularValues, svFixture, t.singularValues)],
          [
            'noiseVariance',
            maxDiff([gpu.noiseVariance], [Number(c.scalars.noise_variance_)], t.noiseVariance),
          ],
        ];
        // transform(test), rank-masked columns like the CPU parity suite.
        if (rank === k) {
          const got = await gpu.transform(XTest);
          checks.push([
            'transform',
            maxDiff(got.data, getArray(bin, c.arrays.transform_test), t.transform),
          ]);
        }
        const bad = checks.filter(([, d]) => !d.ok);
        record({
          id: `fixtures.${c.id}`,
          section: 'fixtures',
          pass: bad.length === 0,
          backend: 'webgpu',
          detail:
            bad.length === 0
              ? `ok (${checks.length} attrs)`
              : bad
                  .map(([name, d]) => `${name}: maxAbs=${fmt(d.maxAbs)} maxRel=${fmt(d.maxRel)}`)
                  .join('; '),
        });
        gpu.dispose();
      } catch (err) {
        record({
          id: `fixtures.${c.id}`,
          section: 'fixtures',
          pass: false,
          backend: 'error',
          detail: String(err),
        });
      }
    }
  } catch (err) {
    record({
      id: 'fixtures.load',
      section: 'fixtures',
      pass: false,
      backend: 'error',
      detail: `fixture fetch failed: ${String(err)}`,
    });
  }

  return results;
}

main()
  .then((results) => {
    window.__PCA_RESULTS__ = results;
  })
  .catch((err) => {
    window.__PCA_RESULTS__ = {
      supported: isWebGPUSupported(),
      adapter: null,
      dsRelError: null,
      gpuExecuted: false,
      cases: [
        { id: 'main', section: 'engine', pass: false, backend: 'error', detail: String(err) },
      ],
      failures: 1,
    };
  });
