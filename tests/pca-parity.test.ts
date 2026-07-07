/**
 * The core parity suite: every fixture case fits pca-web's PCA on the exact
 * same input scikit-learn 1.9.0 saw, and compares every fitted attribute and
 * method output at documented tolerances. Sign conventions must match
 * exactly (svd_flip parity) — comparisons are elementwise, not up-to-sign.
 *
 * Rank-aware handling: components whose fixture singular value is below
 * s[0]·1e-7 span a (near-)null space where LAPACK's basis choice is
 * arbitrary; those rows/columns are excluded from elementwise vector
 * comparison (their variances are still compared, near zero).
 */
import { afterAll, describe, expect, it } from 'vitest';
import { Matrix } from '../src/matrix.js';
import { PCA, type PCAOptions, type PowerIterationNormalizer, type SvdSolver } from '../src/pca.js';
import type { FloatArray } from '../src/types.js';
import { assertClose, assertScalarClose, observedReport, type Tol } from './helpers/compare.js';
import { type FixtureCase, getArray, getMatrix, loadSuite, num } from './helpers/fixtures.js';

const suite = loadSuite('pca');

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

type TolTable = Record<
  | 'mean'
  | 'components'
  | 'explainedVariance'
  | 'ratio'
  | 'singularValues'
  | 'noiseVariance'
  | 'transform'
  | 'inverse'
  | 'covariance'
  | 'precision'
  | 'score',
  Tol
>;

function tolerancesFor(c: FixtureCase): { t: TolTable; klass: string } {
  const isF32 = c.flags.dtype === 'float32';
  // expected_solver records the solver sklearn actually dispatched to
  // (generator reads est._fit_svd_solver), so 'auto' cases classify correctly.
  const effSolver = (c.flags.expected_solver ?? c.params.svd_solver ?? 'auto') as string;
  const randomized = effSolver === 'randomized';
  const covEigh = effSolver === 'covariance_eigh';
  if (isF32) {
    return {
      klass: `f32/${randomized ? 'randomized' : 'deterministic'}`,
      t: {
        mean: { atol: 1e-5, rtol: 1e-5 },
        components: { atol: 2e-3, rtol: 2e-3 },
        explainedVariance: { atol: 1e-4, rtol: 2e-3 },
        ratio: { atol: 1e-5, rtol: 2e-3 },
        singularValues: { atol: 1e-4, rtol: 2e-3 },
        noiseVariance: { atol: 1e-6, rtol: 1e-3 },
        transform: { atol: 5e-3, rtol: 2e-3 },
        inverse: { atol: 5e-3, rtol: 2e-3 },
        covariance: { atol: 1e-3, rtol: 2e-3 },
        precision: { atol: 1e-2, rtol: 1e-2 },
        score: { atol: 1e-2, rtol: 1e-3 },
      },
    };
  }
  if (randomized && c.params.power_iteration_normalizer === 'none') {
    // Explicitly-requested unnormalized power iterations (sklearn documents
    // 'none' as numerically unstable): trailing directions carry
    // (σi/σ1)^(2·nIter+1) ≈ 1e-13 relative weight in the panel, so both
    // implementations sit at the f64 noise floor there. Measured: sklearn
    // itself moves its own trailing components by ~2e-4 abs under a 1-ulp
    // input perturbation for this configuration.
    return {
      klass: 'f64/randomized-none',
      t: {
        mean: { atol: 1e-12, rtol: 1e-12 },
        components: { atol: 5e-3, rtol: 1e-4 },
        explainedVariance: { atol: 1e-9, rtol: 1e-4 },
        ratio: { atol: 1e-10, rtol: 1e-4 },
        singularValues: { atol: 1e-9, rtol: 1e-4 },
        noiseVariance: { atol: 1e-10, rtol: 1e-3 },
        // Projections onto the rotated trailing direction differ by
        // ‖x−mean‖·Δθ — a few 1e-2 absolute in the last column.
        transform: { atol: 5e-2, rtol: 1e-3 },
        inverse: { atol: 5e-2, rtol: 1e-3 },
        covariance: { atol: 1e-3, rtol: 1e-3 },
        precision: { atol: 1e-3, rtol: 1e-2 },
        // log-likelihoods are large negatives; only relative error is
        // meaningful once the precision matrix carries ~1e-2 noise.
        score: { atol: 1e-2, rtol: 5e-4 },
      },
    };
  }
  if (randomized) {
    return {
      klass: 'f64/randomized',
      t: {
        mean: { atol: 1e-12, rtol: 1e-12 },
        components: { atol: 1e-7, rtol: 1e-5 },
        explainedVariance: { atol: 1e-9, rtol: 1e-7 },
        ratio: { atol: 1e-10, rtol: 1e-7 },
        singularValues: { atol: 1e-9, rtol: 1e-7 },
        // (totalVar - Σev) cancellation amplifies ev's relative error by
        // ~totalVar/noiseVar; worst observed with normalizer='none'.
        noiseVariance: { atol: 1e-10, rtol: 1e-4 },
        transform: { atol: 1e-6, rtol: 1e-5 },
        inverse: { atol: 1e-7, rtol: 1e-6 },
        covariance: { atol: 1e-8, rtol: 1e-6 },
        precision: { atol: 1e-6, rtol: 1e-5 },
        score: { atol: 1e-7, rtol: 1e-7 },
      },
    };
  }
  if (covEigh) {
    // The Gram-matrix route squares the condition number: eigenvalues of
    // XᵀX carry an ~eps·‖C‖ noise floor, so tail singular values and the
    // vectors they produce are noisier than a direct SVD of X.
    return {
      klass: 'f64/covariance_eigh',
      t: {
        mean: { atol: 1e-12, rtol: 1e-12 },
        components: { atol: 1e-9, rtol: 1e-7 },
        explainedVariance: { atol: 1e-10, rtol: 1e-9 },
        ratio: { atol: 1e-12, rtol: 1e-9 },
        singularValues: { atol: 2e-6, rtol: 1e-9 },
        noiseVariance: { atol: 1e-11, rtol: 1e-9 },
        transform: { atol: 1e-7, rtol: 1e-6 },
        inverse: { atol: 1e-8, rtol: 1e-7 },
        covariance: { atol: 1e-10, rtol: 1e-8 },
        precision: { atol: 1e-8, rtol: 1e-6 },
        score: { atol: 1e-8, rtol: 1e-8 },
      },
    };
  }
  return {
    klass: 'f64/deterministic',
    t: {
      mean: { atol: 1e-12, rtol: 1e-12 },
      components: { atol: 1e-9, rtol: 1e-7 },
      explainedVariance: { atol: 1e-10, rtol: 1e-9 },
      ratio: { atol: 1e-12, rtol: 1e-9 },
      singularValues: { atol: 1e-10, rtol: 1e-9 },
      // atol covers roundoff-scale zeros from (totalVar - Σev) cancellation.
      noiseVariance: { atol: 1e-11, rtol: 1e-9 },
      transform: { atol: 1e-8, rtol: 1e-7 },
      inverse: { atol: 1e-9, rtol: 1e-8 },
      covariance: { atol: 1e-10, rtol: 1e-8 },
      precision: { atol: 1e-8, rtol: 1e-6 },
      score: { atol: 1e-8, rtol: 1e-8 },
    },
  };
}

/** Number of leading components whose fixture singular value is meaningfully nonzero. */
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

/** Compares the first `rows` rows of two row-major matrices elementwise. */
function compareRows(
  actual: FloatArray,
  expected: FloatArray,
  rows: number,
  cols: number,
  tol: Tol,
  label: string,
  group: string,
): void {
  assertClose(
    actual.subarray(0, rows * cols),
    expected.subarray(0, rows * cols),
    tol,
    label,
    group,
  );
}

/** Compares the first `keepCols` columns of two row-major (n×k) matrices elementwise. */
function compareCols(
  actual: FloatArray,
  expected: FloatArray,
  n: number,
  k: number,
  keepCols: number,
  tol: Tol,
  label: string,
  group: string,
): void {
  if (keepCols === k) {
    assertClose(actual, expected, tol, label, group);
    return;
  }
  const a: number[] = [];
  const e: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < keepCols; c++) {
      a.push(actual[i * k + c]);
      e.push(expected[i * k + c]);
    }
  }
  assertClose(a, e, tol, label, group);
}

describe('PCA parity vs scikit-learn 1.9.0 fixtures', () => {
  for (const c of suite.cases) {
    it(c.id, () => {
      const X = getMatrix(suite, c.arrays.X);
      const XTest = getMatrix(suite, c.arrays.X_test);
      const { t, klass } = tolerancesFor(c);
      const g = (name: string) => `${klass}:${name}`;

      const pca = new PCA(toOptions(c.params));
      const fitTransformed = pca.fitTransform(X);

      // --- scalar attributes -------------------------------------------
      expect(pca.nComponents, 'nComponents').toBe(num(c.scalars.n_components_));
      expect(pca.nSamples, 'nSamples').toBe(num(c.scalars.n_samples_));
      expect(pca.nFeaturesIn, 'nFeaturesIn').toBe(num(c.scalars.n_features_in_));
      if (c.flags.expected_solver) {
        expect(pca.resolvedSvdSolver, 'auto solver selection').toBe(c.flags.expected_solver);
      }
      assertScalarClose(
        pca.noiseVariance,
        num(c.scalars.noise_variance_),
        t.noiseVariance,
        `${c.id}: noiseVariance`,
        g('noiseVariance'),
      );
      expect(pca.getFeatureNamesOut()).toEqual(c.scalars.feature_names_out);

      // --- array attributes --------------------------------------------
      const k = pca.nComponents;
      const p = pca.nFeaturesIn;
      const svFixture = getArray(suite, c.arrays.singular_values);
      const rank = Math.min(fixtureRank(svFixture), k);

      assertClose(pca.mean, getArray(suite, c.arrays.mean), t.mean, `${c.id}: mean`, g('mean'));
      compareRows(
        pca.components.data,
        getArray(suite, c.arrays.components),
        rank,
        p,
        t.components,
        `${c.id}: components (first ${rank}/${k} rows)`,
        g('components'),
      );
      assertClose(
        pca.explainedVariance,
        getArray(suite, c.arrays.explained_variance),
        t.explainedVariance,
        `${c.id}: explainedVariance`,
        g('explainedVariance'),
      );
      assertClose(
        pca.explainedVarianceRatio,
        getArray(suite, c.arrays.explained_variance_ratio),
        t.ratio,
        `${c.id}: explainedVarianceRatio`,
        g('ratio'),
      );
      assertClose(
        pca.singularValues,
        svFixture,
        t.singularValues,
        `${c.id}: singularValues`,
        g('singularValues'),
      );

      // --- transforms -----------------------------------------------------
      // Tail columns beyond the numerical rank live in an arbitrary basis
      // (and are eps-clip amplified under whitening) — excluded.
      const keepCols = rank;
      compareCols(
        fitTransformed.data,
        getArray(suite, c.arrays.fit_transform),
        X.rows,
        k,
        keepCols,
        t.transform,
        `${c.id}: fitTransform`,
        g('fitTransform'),
      );
      compareCols(
        pca.transform(X).data,
        getArray(suite, c.arrays.transform_train),
        X.rows,
        k,
        keepCols,
        t.transform,
        `${c.id}: transform(train)`,
        g('transform'),
      );
      const testTransformed = pca.transform(XTest);
      compareCols(
        testTransformed.data,
        getArray(suite, c.arrays.transform_test),
        XTest.rows,
        k,
        keepCols,
        t.transform,
        `${c.id}: transform(test)`,
        g('transform'),
      );
      // inverseTransform reconstructs through ALL k components. When
      // rank < k the tail components are an arbitrary null-space basis, and
      // out-of-span TEST rows have O(1) projections onto whichever basis the
      // implementation picked — reconstructions legitimately differ, so the
      // comparison is only meaningful at full rank.
      if (c.arrays.inverse_transform_test && rank === k) {
        assertClose(
          pca.inverseTransform(testTransformed).data,
          getArray(suite, c.arrays.inverse_transform_test),
          t.inverse,
          `${c.id}: inverseTransform`,
          g('inverse'),
        );
      }

      // --- generative model ------------------------------------------------
      if (!c.flags.skip_covariance) {
        assertClose(
          pca.getCovariance().data,
          getArray(suite, c.arrays.get_covariance),
          t.covariance,
          `${c.id}: getCovariance`,
          g('covariance'),
        );
      }
      if (!c.flags.skip_precision) {
        assertClose(
          pca.getPrecision().data,
          getArray(suite, c.arrays.get_precision),
          t.precision,
          `${c.id}: getPrecision`,
          g('precision'),
        );
        assertClose(
          pca.scoreSamples(XTest),
          getArray(suite, c.arrays.score_samples_test),
          t.score,
          `${c.id}: scoreSamples`,
          g('score'),
        );
        assertScalarClose(
          pca.score(XTest),
          num(c.scalars.score_test),
          t.score,
          `${c.id}: score`,
          g('score'),
        );
      }
    });
  }

  afterAll(() => {
    // eslint-disable-next-line no-console
    console.log(`\nObserved parity diffs (PCA suite):\n${observedReport()}\n`);
  });
});
