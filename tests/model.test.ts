import { describe, expect, it } from 'vitest';
import { IncrementalPCA } from '../src/incremental-pca.js';
import { Matrix } from '../src/matrix.js';
import {
  type AnyPCAModel,
  assertValidModel,
  type IncrementalPCAModel,
  modelFromJSON,
  modelToJSON,
  PCA_MODEL_FORMAT_VERSION,
  type PCAModel,
} from '../src/model.js';
import { RandomState } from '../src/numeric/rng.js';
import { PCA, type SvdSolver } from '../src/pca.js';
import type { Dtype } from '../src/types.js';
import { WebGPUPCA } from '../src/webgpu/index.js';

function gaussian(n: number, p: number, seed: number, dtype: Dtype = 'float64'): Matrix {
  const data = new Float64Array(n * p);
  new RandomState(seed).standardNormal(data);
  return new Matrix(dtype === 'float64' ? data : Float32Array.from(data), n, p);
}

const SOLVERS: Exclude<SvdSolver, 'auto'>[] = ['full', 'covariance_eigh', 'arpack', 'randomized'];

describe('PCA model round-trip', () => {
  for (const solver of SOLVERS) {
    it(`bit-exact behavior after toModel/fromModel (${solver})`, () => {
      const X = gaussian(120, 20, 7);
      const Y = gaussian(15, 20, 8);
      const original = new PCA({
        nComponents: 5,
        svdSolver: solver,
        randomState: 0,
        whiten: solver === 'randomized',
      }).fit(X);
      const restored = PCA.fromModel(original.toModel());

      expect(restored.nComponents).toBe(original.nComponents);
      expect(restored.nFeaturesIn).toBe(original.nFeaturesIn);
      expect(restored.nSamples).toBe(original.nSamples);
      expect(restored.resolvedSvdSolver).toBe(original.resolvedSvdSolver);
      expect(restored.whiten).toBe(original.whiten);
      expect(restored.noiseVariance).toBe(original.noiseVariance);
      expect(restored.components.data).toEqual(original.components.data);
      expect(restored.mean).toEqual(original.mean);
      expect(restored.singularValues).toEqual(original.singularValues);
      expect(restored.explainedVariance).toEqual(original.explainedVariance);
      expect(restored.explainedVarianceRatio).toEqual(original.explainedVarianceRatio);

      expect(restored.transform(Y).data).toEqual(original.transform(Y).data);
      const t = original.transform(Y);
      expect(restored.inverseTransform(t).data).toEqual(original.inverseTransform(t).data);
      expect(restored.scoreSamples(Y)).toEqual(original.scoreSamples(Y));
      expect(restored.score(Y)).toBe(original.score(Y));
      expect(restored.getCovariance().data).toEqual(original.getCovariance().data);
      expect(restored.getPrecision().data).toEqual(original.getPrecision().data);
      expect(restored.getFeatureNamesOut()).toEqual(original.getFeatureNamesOut());
    });
  }

  it('float32 models keep their dtype and behavior', () => {
    const X = gaussian(100, 12, 11, 'float32');
    const original = new PCA({ nComponents: 4, svdSolver: 'full' }).fit(X);
    const model = original.toModel();
    expect(model.dtype).toBe('float32');
    expect(model.components).toBeInstanceOf(Float32Array);
    const restored = PCA.fromModel(model);
    expect(restored.transform(X).data).toEqual(original.transform(X).data);
    expect(restored.transform(X).data).toBeInstanceOf(Float32Array);
  });

  it('model arrays are snapshots decoupled from live and restored state', () => {
    const X = gaussian(60, 8, 13);
    const pca = new PCA({ nComponents: 3, svdSolver: 'full' }).fit(X);
    const model = pca.toModel();
    const before = pca.components.data.slice();
    model.components[0] = 999;
    expect(pca.components.data).toEqual(before);
    // fromModel adopts zero-copy by contract: restored sees the mutation.
    const restored = PCA.fromModel(model);
    expect(restored.components.data[0]).toBe(999);
  });

  it('a live RandomState in options serializes as null', () => {
    const X = gaussian(60, 8, 17);
    const pca = new PCA({
      nComponents: 2,
      svdSolver: 'randomized',
      randomState: new RandomState(5),
    }).fit(X);
    expect(pca.toModel().options.randomState).toBeNull();
    const seeded = new PCA({ nComponents: 2, svdSolver: 'randomized', randomState: 5 }).fit(X);
    expect(seeded.toModel().options.randomState).toBe(5);
  });
});

describe('IncrementalPCA model round-trip', () => {
  it('partialFit resumes bit-exactly after rehydration', () => {
    const X = gaussian(90, 10, 19);
    const rows = (a: number, b: number) => new Matrix(X.data.slice(a * 10, b * 10), b - a, 10);

    const uninterrupted = new IncrementalPCA({ nComponents: 3 });
    uninterrupted.partialFit(rows(0, 30));
    uninterrupted.partialFit(rows(30, 60));
    uninterrupted.partialFit(rows(60, 90));

    const first = new IncrementalPCA({ nComponents: 3 });
    first.partialFit(rows(0, 30));
    first.partialFit(rows(30, 60));
    const resumed = IncrementalPCA.fromModel(first.toModel());
    expect(resumed.nSamplesSeen).toBe(60);
    resumed.partialFit(rows(60, 90));

    expect(resumed.nSamplesSeen).toBe(uninterrupted.nSamplesSeen);
    expect(resumed.components.data).toEqual(uninterrupted.components.data);
    expect(resumed.singularValues).toEqual(uninterrupted.singularValues);
    expect(resumed.explainedVariance).toEqual(uninterrupted.explainedVariance);
    expect(resumed.mean).toEqual(uninterrupted.mean);
    expect(resumed.variance).toEqual(uninterrupted.variance);
    expect(resumed.noiseVariance).toBe(uninterrupted.noiseVariance);
    expect(resumed.transform(X).data).toEqual(uninterrupted.transform(X).data);
  });

  it('fit()-produced models carry batchSize and transform identically', () => {
    const X = gaussian(80, 8, 23);
    const original = new IncrementalPCA({ nComponents: 3, batchSize: 20, whiten: true }).fit(X);
    const model = original.toModel();
    expect(model.batchSize).toBe(20);
    expect(model.variance).toBeInstanceOf(Float64Array);
    const restored = IncrementalPCA.fromModel(model);
    expect(restored.batchSize).toBe(20);
    expect(restored.transform(X).data).toEqual(original.transform(X).data);
    expect(restored.getCovariance().data).toEqual(original.getCovariance().data);
  });
});

describe('structuredClone transport', () => {
  it('models survive structuredClone with independent buffers', () => {
    const X = gaussian(70, 9, 29);
    const pca = new PCA({ nComponents: 3, svdSolver: 'randomized', randomState: 1 }).fit(X);
    const model = pca.toModel();
    const clone = structuredClone(model);
    clone.components[0] = -12345;
    expect(model.components[0]).not.toBe(-12345);
    clone.components[0] = model.components[0];
    const restored = PCA.fromModel(clone);
    expect(restored.transform(X).data).toEqual(pca.transform(X).data);
  });
});

describe('JSON transport', () => {
  it('is bit-exact for float64 models', () => {
    const X = gaussian(100, 10, 31);
    const pca = new PCA({ nComponents: 4, svdSolver: 'randomized', randomState: 2 }).fit(X);
    const model = pca.toModel();
    const back = modelFromJSON(modelToJSON(model)) as PCAModel;
    expect(back.components).toEqual(model.components);
    expect(back.mean).toEqual(model.mean);
    expect(back.singularValues).toEqual(model.singularValues);
    expect(back.explainedVariance).toEqual(model.explainedVariance);
    expect(back.explainedVarianceRatio).toEqual(model.explainedVarianceRatio);
    expect(back.noiseVariance).toBe(model.noiseVariance);
    expect(back.options).toEqual(model.options);
    expect(PCA.fromModel(back).transform(X).data).toEqual(pca.transform(X).data);
  });

  it('is bit-exact for float32 models (arrays restored as Float32Array)', () => {
    const X = gaussian(80, 8, 37, 'float32');
    const model = new PCA({ nComponents: 3, svdSolver: 'full' }).fit(X).toModel();
    const back = modelFromJSON(modelToJSON(model)) as PCAModel;
    expect(back.dtype).toBe('float32');
    expect(back.components).toBeInstanceOf(Float32Array);
    expect(back.components).toEqual(model.components);
    expect(back.mean).toBeInstanceOf(Float32Array);
    expect(back.mean).toEqual(model.mean);
  });

  it('round-trips IncrementalPCA models with float64 statistics', () => {
    const X = gaussian(60, 6, 41, 'float32');
    const model = new IncrementalPCA({ nComponents: 2, batchSize: 30 }).fit(X).toModel();
    const back = modelFromJSON(modelToJSON(model)) as IncrementalPCAModel;
    expect(back.estimator).toBe('ipca');
    expect(back.mean).toBeInstanceOf(Float64Array);
    expect(back.variance).toBeInstanceOf(Float64Array);
    expect(back.mean).toEqual(model.mean);
    expect(back.variance).toEqual(model.variance);
    expect(back.nSamplesSeen).toBe(model.nSamplesSeen);
  });
});

describe('assertValidModel', () => {
  const valid = (): PCAModel =>
    new PCA({ nComponents: 2, svdSolver: 'full' }).fit(gaussian(30, 5, 43)).toModel();

  it('accepts a genuine model', () => {
    expect(() => assertValidModel(valid())).not.toThrow();
    expect(PCA_MODEL_FORMAT_VERSION).toBe(1);
  });

  it('rejects non-objects, wrong versions, and wrong estimators', () => {
    expect(() => assertValidModel(null)).toThrow(/not an object/);
    expect(() => assertValidModel({ ...valid(), formatVersion: 2 })).toThrow(/formatVersion/);
    expect(() => assertValidModel({ ...valid(), estimator: 'kmeans' })).toThrow(/estimator/);
    expect(() => assertValidModel(valid(), 'ipca')).toThrow(/expected a 'ipca' model/);
  });

  it('rejects shape and dtype mismatches', () => {
    const m = valid();
    expect(() =>
      assertValidModel({ ...m, components: m.components.slice(0, 3) } as AnyPCAModel),
    ).toThrow(/components must have length/);
    expect(() =>
      assertValidModel({ ...m, mean: Array.from(m.mean) } as unknown as AnyPCAModel),
    ).toThrow(/mean must be a Float64Array or Float32Array/);
    expect(() =>
      assertValidModel({ ...m, singularValues: Float32Array.from(m.singularValues) }),
    ).toThrow(/singularValues must be a Float64Array/);
    expect(() => assertValidModel({ ...m, noiseVariance: Number.NaN })).toThrow(/noiseVariance/);
    expect(() => assertValidModel({ ...m, svdSolver: 'auto' })).toThrow(/svdSolver/);
  });

  it('fromModel validates before adopting', () => {
    const m = valid();
    expect(() => PCA.fromModel({ ...m, nFeaturesIn: 4 })).toThrow(/length/);
  });
});

describe('WebGPUPCA model delegation', () => {
  it('toModel/fromModel round-trips through the CPU model (Node fallback)', async () => {
    const X = gaussian(100, 12, 47);
    const gpu = new WebGPUPCA({ nComponents: 3, svdSolver: 'randomized', randomState: 0 });
    await gpu.fit(X);
    const model = gpu.toModel();
    expect(model.estimator).toBe('pca');

    const restored = WebGPUPCA.fromModel(model, { minGpuElements: 1 });
    const rt = await restored.transform(X);
    const ot = await gpu.transform(X);
    expect(rt.data).toEqual(ot.data);
    expect(restored.nComponents).toBe(3);
    expect(restored.whiten).toBe(gpu.whiten);
    // Interchangeable with the plain PCA class.
    const cpuRestored = PCA.fromModel(model);
    expect(cpuRestored.transform(X).data).toEqual(rt.data);
  });
});
