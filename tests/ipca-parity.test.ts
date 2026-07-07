/**
 * IncrementalPCA parity vs scikit-learn 1.9.0 fixtures.
 *
 * Fit cases check the final state after batched fit(); partial cases replay
 * the exact partial_fit sequence and check the FULL fitted state after every
 * step (components, variances, mean, var, noise variance, sample counts),
 * so any divergence is pinned to the step that introduced it.
 *
 * Everything here is deterministic (no RNG in the incremental update), so
 * tolerances are LAPACK-vs-port roundoff, slightly amplified by the number
 * of stacked SVD steps.
 */
import { afterAll, describe, expect, it } from 'vitest';
import { IncrementalPCA, type IncrementalPCAOptions } from '../src/incremental-pca.js';
import type { Matrix } from '../src/matrix.js';
import type { FloatArray } from '../src/types.js';
import { assertClose, assertScalarClose, observedReport, type Tol } from './helpers/compare.js';
import { type FixtureCase, getArray, getMatrix, loadSuite, num } from './helpers/fixtures.js';

const suite = loadSuite('ipca');

function toOptions(params: Record<string, unknown>): IncrementalPCAOptions {
  const o: IncrementalPCAOptions = {};
  if ('n_components' in params) {
    o.nComponents = params.n_components as number | null;
  }
  if ('batch_size' in params) {
    o.batchSize = params.batch_size as number | null;
  }
  if ('whiten' in params) {
    o.whiten = params.whiten as boolean;
  }
  return o;
}

interface TolTable {
  meanVar: Tol;
  components: Tol;
  explainedVariance: Tol;
  ratio: Tol;
  singularValues: Tol;
  noiseVariance: Tol;
  transform: Tol;
  inverse: Tol;
  covariance: Tol;
  precision: Tol;
}

function tolerancesFor(c: FixtureCase): { t: TolTable; klass: string } {
  if (c.flags.dtype === 'float32') {
    return {
      klass: 'ipca/f32',
      t: {
        meanVar: { atol: 1e-5, rtol: 1e-5 },
        components: { atol: 2e-3, rtol: 2e-3 },
        explainedVariance: { atol: 1e-4, rtol: 2e-3 },
        ratio: { atol: 1e-5, rtol: 2e-3 },
        singularValues: { atol: 1e-4, rtol: 2e-3 },
        noiseVariance: { atol: 1e-6, rtol: 1e-3 },
        transform: { atol: 5e-3, rtol: 2e-3 },
        inverse: { atol: 5e-3, rtol: 2e-3 },
        covariance: { atol: 1e-3, rtol: 2e-3 },
        precision: { atol: 1e-2, rtol: 1e-2 },
      },
    };
  }
  return {
    klass: 'ipca/f64',
    t: {
      meanVar: { atol: 1e-12, rtol: 1e-12 },
      components: { atol: 1e-8, rtol: 1e-6 },
      explainedVariance: { atol: 1e-10, rtol: 1e-8 },
      ratio: { atol: 1e-12, rtol: 1e-8 },
      singularValues: { atol: 1e-9, rtol: 1e-8 },
      noiseVariance: { atol: 1e-11, rtol: 1e-8 },
      transform: { atol: 1e-7, rtol: 1e-6 },
      inverse: { atol: 1e-8, rtol: 1e-7 },
      covariance: { atol: 1e-9, rtol: 1e-7 },
      precision: { atol: 1e-7, rtol: 1e-5 },
    },
  };
}

/** Leading components whose fixture singular value is meaningfully nonzero. */
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

/**
 * Compares the complete fitted state against a captured sklearn snapshot.
 * `prefix` selects the fixture keys ('' for fit cases, 'stepN_' for
 * partial-fit sequences).
 */
function compareState(
  c: FixtureCase,
  ipca: IncrementalPCA,
  t: TolTable,
  g: (name: string) => string,
  prefix: string,
  stepScalars: Record<string, unknown>,
  where: string,
): void {
  expect(ipca.nComponents, `${where}: nComponents`).toBe(num(stepScalars.n_components_));
  expect(ipca.nSamplesSeen, `${where}: nSamplesSeen`).toBe(num(stepScalars.n_samples_seen_));
  assertScalarClose(
    ipca.noiseVariance,
    num(stepScalars.noise_variance_),
    t.noiseVariance,
    `${where}: noiseVariance`,
    g('noiseVariance'),
  );

  const svFixture = getArray(suite, c.arrays[`${prefix}singular_values`]);
  const k = ipca.nComponents;
  const p = ipca.nFeaturesIn;
  const rank = Math.min(fixtureRank(svFixture), k);

  assertClose(
    ipca.mean,
    getArray(suite, c.arrays[`${prefix}mean`]),
    t.meanVar,
    `${where}: mean`,
    g('mean'),
  );
  assertClose(
    ipca.variance,
    getArray(suite, c.arrays[`${prefix}var`]),
    t.meanVar,
    `${where}: var`,
    g('var'),
  );
  assertClose(
    ipca.components.data.subarray(0, rank * p),
    getArray(suite, c.arrays[`${prefix}components`]).subarray(0, rank * p),
    t.components,
    `${where}: components (first ${rank}/${k} rows)`,
    g('components'),
  );
  assertClose(
    ipca.explainedVariance,
    getArray(suite, c.arrays[`${prefix}explained_variance`]),
    t.explainedVariance,
    `${where}: explainedVariance`,
    g('explainedVariance'),
  );
  assertClose(
    ipca.explainedVarianceRatio,
    getArray(suite, c.arrays[`${prefix}explained_variance_ratio`]),
    t.ratio,
    `${where}: explainedVarianceRatio`,
    g('ratio'),
  );
  assertClose(
    ipca.singularValues,
    svFixture,
    t.singularValues,
    `${where}: singularValues`,
    g('singularValues'),
  );
}

/** Transform/inverse checks shared by fit and partial cases. */
function compareTransforms(
  c: FixtureCase,
  ipca: IncrementalPCA,
  XTest: Matrix,
  t: TolTable,
  g: (name: string) => string,
): void {
  const svFixture = getArray(suite, c.arrays.singular_values);
  const testTransformed = ipca.transform(XTest);
  assertClose(
    testTransformed.data,
    getArray(suite, c.arrays.transform_test),
    t.transform,
    `${c.id}: transform(test)`,
    g('transform'),
  );
  // Reconstruction through an arbitrary null-space basis is not comparable
  // (see pca-parity); all ipca fixtures are full-rank, asserted here.
  const rank = Math.min(fixtureRank(svFixture), ipca.nComponents);
  expect(rank, `${c.id}: fixture rank covers all components`).toBe(ipca.nComponents);
  assertClose(
    ipca.inverseTransform(testTransformed).data,
    getArray(suite, c.arrays.inverse_transform_test),
    t.inverse,
    `${c.id}: inverseTransform`,
    g('inverse'),
  );
}

describe('IncrementalPCA parity vs scikit-learn 1.9.0 fixtures', () => {
  const fitCases = suite.cases.filter((c) => c.arrays.X);
  const partialCases = suite.cases.filter((c) => !c.arrays.X);

  for (const c of fitCases) {
    it(`fit: ${c.id}`, () => {
      const X = getMatrix(suite, c.arrays.X);
      const XTest = getMatrix(suite, c.arrays.X_test);
      const { t, klass } = tolerancesFor(c);
      const g = (name: string) => `${klass}:${name}`;

      const ipca = new IncrementalPCA(toOptions(c.params));
      ipca.fit(X);

      expect(ipca.batchSize, 'batchSize').toBe(num(c.scalars.batch_size_));
      expect(ipca.nFeaturesIn, 'nFeaturesIn').toBe(num(c.scalars.n_features_in_));
      expect(ipca.getFeatureNamesOut()).toEqual(c.scalars.feature_names_out);
      compareState(c, ipca, t, g, '', c.scalars, c.id);
      compareTransforms(c, ipca, XTest, t, g);

      if (c.arrays.get_covariance) {
        assertClose(
          ipca.getCovariance().data,
          getArray(suite, c.arrays.get_covariance),
          t.covariance,
          `${c.id}: getCovariance`,
          g('covariance'),
        );
      }
      if (c.arrays.get_precision) {
        assertClose(
          ipca.getPrecision().data,
          getArray(suite, c.arrays.get_precision),
          t.precision,
          `${c.id}: getPrecision`,
          g('precision'),
        );
      }
    });
  }

  for (const c of partialCases) {
    it(`partial_fit: ${c.id}`, () => {
      const XTest = getMatrix(suite, c.arrays.X_test);
      const { t, klass } = tolerancesFor(c);
      const g = (name: string) => `${klass}:${name}`;
      const nBatches = num(c.scalars.n_batches);
      const steps = c.scalars.steps as Record<string, unknown>[];

      const ipca = new IncrementalPCA(toOptions(c.params));
      for (let i = 0; i < nBatches; i++) {
        ipca.partialFit(getMatrix(suite, c.arrays[`batch${i}`]));
        compareState(c, ipca, t, g, `step${i}_`, steps[i], `${c.id} step ${i}`);
      }
      expect(ipca.nFeaturesIn, 'nFeaturesIn').toBe(num(c.scalars.n_features_in_));

      const lastPrefix = `step${nBatches - 1}_`;
      const svFinal = getArray(suite, c.arrays[`${lastPrefix}singular_values`]);
      const testTransformed = ipca.transform(XTest);
      assertClose(
        testTransformed.data,
        getArray(suite, c.arrays.transform_test),
        t.transform,
        `${c.id}: transform(test)`,
        g('transform'),
      );
      const rank = Math.min(fixtureRank(svFinal), ipca.nComponents);
      expect(rank, `${c.id}: fixture rank covers all components`).toBe(ipca.nComponents);
      assertClose(
        ipca.inverseTransform(testTransformed).data,
        getArray(suite, c.arrays.inverse_transform_test),
        t.inverse,
        `${c.id}: inverseTransform`,
        g('inverse'),
      );
    });
  }

  afterAll(() => {
    console.log(`\nObserved parity diffs (IncrementalPCA suite):\n${observedReport()}\n`);
  });
});
