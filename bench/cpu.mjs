/**
 * CPU benchmark: PCA fit/transform timings across sizes and solvers in Node.
 * Run: npm run bench
 */
import { performance } from 'node:perf_hooks';
import { Matrix, PCA, RandomState } from '../dist/index.js';

function data(n, p, seed = 7) {
  const rng = new RandomState(seed);
  const x = new Float64Array(n * p);
  rng.standardNormal(x);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) {
      x[i * p + j] = x[i * p + j] * (1 + (j % 7)) + (j % 5);
    }
  }
  return new Matrix(x, n, p);
}

function time(fn, reps = 3) {
  const times = [];
  for (let i = 0; i < reps; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

const CONFIGS = [
  { n: 2000, p: 100, opts: { nComponents: 16, svdSolver: 'covariance_eigh' } },
  { n: 20000, p: 100, opts: { nComponents: 16, svdSolver: 'covariance_eigh' } },
  { n: 2000, p: 200, opts: { nComponents: 16, svdSolver: 'randomized', randomState: 0 } },
  { n: 10000, p: 300, opts: { nComponents: 16, svdSolver: 'randomized', randomState: 0 } },
  { n: 1000, p: 500, opts: { nComponents: 32, svdSolver: 'randomized', randomState: 0 } },
  { n: 500, p: 100, opts: { nComponents: 20, svdSolver: 'full' } },
  { n: 2000, p: 100, opts: { nComponents: 10, svdSolver: 'arpack', randomState: 0 } },
];

console.log('pca-web CPU benchmarks (Node, single-threaded)\n');
console.log('size            solver            fit(ms)   transform(ms)');
for (const { n, p, opts } of CONFIGS) {
  const X = data(n, p);
  const fitMs = time(() => new PCA(opts).fit(X));
  const pca = new PCA(opts).fit(X);
  const trMs = time(() => pca.transform(X));
  console.log(
    `${`${n}x${p}`.padEnd(15)} ${String(opts.svdSolver).padEnd(17)} ${fitMs.toFixed(1).padStart(8)} ${trMs.toFixed(1).padStart(14)}`,
  );
}
