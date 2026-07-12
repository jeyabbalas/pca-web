import { describe, expect, it } from 'vitest';
import { IncrementalPCA } from '../src/incremental-pca.js';
import { Matrix } from '../src/matrix.js';
import { RandomState } from '../src/numeric/rng.js';
import { PCA, type SvdSolver } from '../src/pca.js';
import type { Dtype } from '../src/types.js';

function gaussian(n: number, p: number, seed: number, dtype: Dtype = 'float64'): Matrix {
  const data = new Float64Array(n * p);
  new RandomState(seed).standardNormal(data);
  return new Matrix(dtype === 'float64' ? data : Float32Array.from(data), n, p);
}

interface FittedState {
  components: ArrayLike<number>;
  singularValues: ArrayLike<number>;
  explainedVariance: ArrayLike<number>;
  explainedVarianceRatio: ArrayLike<number>;
  mean: ArrayLike<number>;
  noiseVariance: number;
  nComponents: number;
}

function stateOf(pca: PCA): FittedState {
  return {
    components: pca.components.data,
    singularValues: pca.singularValues,
    explainedVariance: pca.explainedVariance,
    explainedVarianceRatio: pca.explainedVarianceRatio,
    mean: pca.mean,
    noiseVariance: pca.noiseVariance,
    nComponents: pca.nComponents,
  };
}

function expectSameModel(a: FittedState, b: FittedState): void {
  expect(a.nComponents).toBe(b.nComponents);
  expect(a.components).toEqual(b.components);
  expect(a.singularValues).toEqual(b.singularValues);
  expect(a.explainedVariance).toEqual(b.explainedVariance);
  expect(a.explainedVarianceRatio).toEqual(b.explainedVarianceRatio);
  expect(a.mean).toEqual(b.mean);
  expect(a.noiseVariance).toBe(b.noiseVariance);
}

const SOLVERS: Exclude<SvdSolver, 'auto'>[] = ['full', 'covariance_eigh', 'arpack', 'randomized'];
const DTYPES: Dtype[] = ['float64', 'float32'];

describe('fitAsync ≡ fit (bit-identical models)', () => {
  for (const solver of SOLVERS) {
    for (const dtype of DTYPES) {
      it(`${solver} / ${dtype}`, async () => {
        const X = gaussian(150, 30, 17, dtype);
        const opts = { nComponents: 5, svdSolver: solver, randomState: 0 } as const;
        const sync = new PCA(opts).fit(X);
        const async_ = await new PCA(opts).fitAsync(X, { budgetMs: 0 });
        expectSameModel(stateOf(async_), stateOf(sync));
      });
    }
  }

  it('fitTransformAsync ≡ fitTransform across solvers', async () => {
    for (const solver of SOLVERS) {
      const X = gaussian(120, 20, 23);
      const opts = { nComponents: 4, svdSolver: solver, randomState: 1 } as const;
      const sync = new PCA(opts).fitTransform(X);
      const async_ = await new PCA(opts).fitTransformAsync(X, { budgetMs: 0 });
      expect(async_.data, solver).toEqual(sync.data);
      expect(async_.rows).toBe(sync.rows);
      expect(async_.cols).toBe(sync.cols);
    }
  });

  it('default budget (12ms) also matches, not just budgetMs 0', async () => {
    const X = gaussian(200, 40, 29);
    const opts = { nComponents: 6, svdSolver: 'randomized', randomState: 2 } as const;
    const sync = new PCA(opts).fit(X);
    const async_ = await new PCA(opts).fitAsync(X);
    expectSameModel(stateOf(async_), stateOf(sync));
  });

  it('whiten and fraction-nComponents paths agree too', async () => {
    const X = gaussian(100, 15, 31);
    for (const opts of [
      { nComponents: 0.9, svdSolver: 'full' as const, whiten: true },
      { nComponents: 'mle' as const, svdSolver: 'full' as const },
      { whiten: true, svdSolver: 'covariance_eigh' as const },
    ]) {
      const sync = new PCA(opts).fit(X);
      const async_ = await new PCA(opts).fitAsync(X, { budgetMs: 0 });
      expectSameModel(stateOf(async_), stateOf(sync));
    }
  });
});

describe('observer neutrality', () => {
  it('full-snapshot observers do not change the model (all solvers, async)', async () => {
    for (const solver of SOLVERS) {
      const X = gaussian(150, 30, 37);
      const opts = { nComponents: 5, svdSolver: solver, randomState: 7 } as const;
      const plain = await new PCA(opts).fitAsync(X, { budgetMs: 0 });
      const observed = await new PCA(opts).fitAsync(X, {
        budgetMs: 0,
        onProgress: () => {},
        snapshot: { components: true, scores: true },
      });
      expectSameModel(stateOf(observed), stateOf(plain));
    }
  });

  it('snapshots draw no RNG: stream position identical with snapshots on vs off', () => {
    const X = gaussian(150, 30, 41);
    for (const solver of ['randomized', 'arpack'] as const) {
      const rngOn = new RandomState(0);
      const rngOff = new RandomState(0);
      new PCA({ nComponents: 5, svdSolver: solver, randomState: rngOn }).fit(X, {
        onProgress: () => {},
        snapshot: { components: true, scores: true },
      });
      new PCA({ nComponents: 5, svdSolver: solver, randomState: rngOff }).fit(X);
      // Same MT position and same Gaussian cache state afterwards.
      expect(rngOn.nextUint32(), solver).toBe(rngOff.nextUint32());
      const a = new Float64Array(3);
      const b = new Float64Array(3);
      rngOn.standardNormal(a);
      rngOff.standardNormal(b);
      expect(a, solver).toEqual(b);
    }
  });
});

describe('event-loop liveness', () => {
  it('macrotasks run while fitAsync is in flight', async () => {
    const X = gaussian(500, 100, 43);
    let ticks = 0;
    const interval = setInterval(() => {
      ticks++;
    }, 1);
    try {
      await new PCA({ nComponents: 10, svdSolver: 'randomized', randomState: 0 }).fitAsync(X, {
        budgetMs: 4,
      });
    } finally {
      clearInterval(interval);
    }
    expect(ticks).toBeGreaterThan(0);
  });
});

describe('IncrementalPCA fitAsync ≡ fit', () => {
  for (const dtype of DTYPES) {
    it(`bit-identical models (${dtype})`, async () => {
      const X = gaussian(120, 10, 47, dtype);
      const sync = new IncrementalPCA({ nComponents: 4, batchSize: 30 }).fit(X);
      const async_ = await new IncrementalPCA({ nComponents: 4, batchSize: 30 }).fitAsync(X, {
        budgetMs: 0,
      });
      expect(async_.components.data).toEqual(sync.components.data);
      expect(async_.singularValues).toEqual(sync.singularValues);
      expect(async_.explainedVariance).toEqual(sync.explainedVariance);
      expect(async_.explainedVarianceRatio).toEqual(sync.explainedVarianceRatio);
      expect(async_.mean).toEqual(sync.mean);
      expect(async_.variance).toEqual(sync.variance);
      expect(async_.nSamplesSeen).toBe(sync.nSamplesSeen);
      expect(async_.noiseVariance).toBe(sync.noiseVariance);
    });
  }

  it('fitTransformAsync ≡ fitTransform', async () => {
    const X = gaussian(90, 12, 53);
    const sync = new IncrementalPCA({ nComponents: 3, batchSize: 30 }).fitTransform(X);
    const async_ = await new IncrementalPCA({ nComponents: 3, batchSize: 30 }).fitTransformAsync(
      X,
      { budgetMs: 0 },
    );
    expect(async_.data).toEqual(sync.data);
  });
});
