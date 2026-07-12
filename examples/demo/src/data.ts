/**
 * Datasets for the demo: bundled sklearn digits, slider-parametrized synthetic
 * Gaussian clusters, and fixed-parameter visual presets (manifolds, hidden
 * circles, clusters with outliers).
 */
import { Matrix, RandomState } from 'pca-web';

export type DemoDtype = 'float64' | 'float32';

export type PresetKind = 'swissroll' | 'trefoil' | 'circles' | 'outliers';

export interface Dataset {
  /** Cache key of the configuration that produced this dataset. */
  key: string;
  kind: 'digits' | 'synthetic' | PresetKind;
  X: Matrix;
  /** Digit 0–9 or cluster id per row. */
  labels: Uint8Array;
  n: number;
  p: number;
  dtype: DemoDtype;
  /** Rows are side×side images (8 for digits); null when not image-shaped. */
  imageSide: number | null;
  /** Per-label colors overriding the default categorical palette. */
  palette?: string[];
}

let digitsRaw: Promise<{ pixels: Uint8Array; labels: Uint8Array }> | null = null;

async function fetchBytes(name: string): Promise<Uint8Array> {
  const res = await fetch(`${import.meta.env.BASE_URL}${name}`);
  if (!res.ok) {
    throw new Error(`failed to fetch ${name}: HTTP ${res.status}`);
  }
  return new Uint8Array(await res.arrayBuffer());
}

/** sklearn `load_digits()`: 1797×64 pixel intensities in 0..16, committed as uint8. */
export async function loadDigits(dtype: DemoDtype): Promise<Dataset> {
  digitsRaw ??= Promise.all([fetchBytes('digits.bin'), fetchBytes('digits-labels.bin')]).then(
    ([pixels, labels]) => {
      if (pixels.length !== labels.length * 64) {
        throw new Error('digits.bin does not match digits-labels.bin');
      }
      return { pixels, labels };
    },
  );
  const { pixels, labels } = await digitsRaw;
  const data = dtype === 'float64' ? Float64Array.from(pixels) : Float32Array.from(pixels);
  return {
    key: `digits|${dtype}`,
    kind: 'digits',
    X: new Matrix(data, labels.length, 64),
    labels,
    n: labels.length,
    p: 64,
    dtype,
    imageSide: 8,
  };
}

export interface SyntheticSpec {
  n: number;
  p: number;
  clusters: number;
  dtype: DemoDtype;
  seed?: number;
}

/**
 * Isotropic Gaussian blobs around RandomState-seeded centers (the same
 * MT19937 replica the randomized solver uses). Rows interleave clusters so
 * IncrementalPCA batches see every cluster from the first batch on.
 */
export function makeSynthetic(spec: SyntheticSpec): Dataset {
  const { n, p, clusters, dtype } = spec;
  const rng = new RandomState(spec.seed ?? 7);
  const centers = rng.standardNormal(new Float64Array(clusters * p));
  for (let i = 0; i < centers.length; i++) {
    centers[i] *= 4;
  }
  const data = dtype === 'float64' ? new Float64Array(n * p) : new Float32Array(n * p);
  const labels = new Uint8Array(n);
  const noise = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const c = i % clusters;
    labels[i] = c;
    rng.standardNormal(noise);
    const co = c * p;
    const ro = i * p;
    for (let j = 0; j < p; j++) {
      data[ro + j] = centers[co + j] + noise[j];
    }
  }
  return {
    key: `synthetic|${dtype}|${n}|${p}|${clusters}`,
    kind: 'synthetic',
    X: new Matrix(data, n, p),
    labels,
    n,
    p,
    dtype,
    imageSide: null,
  };
}

/** Turbo colormap sampled at 10 points — a smooth rainbow for labels that bin a curve parameter. */
const RAINBOW = [
  '#30123b',
  '#4458cb',
  '#3e9bfe',
  '#18d6cb',
  '#35f394',
  '#a2fc3c',
  '#e1dd37',
  '#fea331',
  '#e5460b',
  '#7a0403',
];

/**
 * Swiss roll (t·cos t, y, t·sin t): the classic rolled-up 2-D manifold. The
 * short y-extent keeps Var(y) ≈ 8 below the two spiral-plane variances ≈ 48,
 * so PC1×PC2 shows the roll face-on. Labels bin t into 10 rainbow arcs.
 */
function makeSwissRoll(dtype: DemoDtype): Dataset {
  const { n, p } = PRESET_INFO.swissroll;
  const rng = new RandomState(11);
  const data = dtype === 'float64' ? new Float64Array(n * p) : new Float32Array(n * p);
  const labels = new Uint8Array(n);
  const tLo = 1.5 * Math.PI;
  const tHi = 4.5 * Math.PI;
  const sigma = 0.3;
  for (let i = 0; i < n; i++) {
    const u = rng.randomSample();
    const t = tLo + (tHi - tLo) * u;
    const o = i * p;
    data[o] = t * Math.cos(t) + sigma * rng.nextGauss();
    data[o + 1] = 10 * rng.randomSample() + sigma * rng.nextGauss();
    data[o + 2] = t * Math.sin(t) + sigma * rng.nextGauss();
    labels[i] = Math.min(9, Math.floor(u * 10));
  }
  return {
    key: `swissroll|${dtype}`,
    kind: 'swissroll',
    X: new Matrix(data, n, p),
    labels,
    n,
    p,
    dtype,
    imageSide: null,
    palette: RAINBOW,
  };
}

/**
 * Trefoil knot (sin θ + 2 sin 2θ, cos θ − 2 cos 2θ, −sin 3θ) with tube noise.
 * The harmonics are uncorrelated over uniform θ, so the covariance is
 * ≈ diag(2.52, 2.52, 0.52) and PC1×PC2 is the classic planar knot projection.
 * Labels bin θ into 10 rainbow arcs around the loop.
 */
function makeTrefoil(dtype: DemoDtype): Dataset {
  const { n, p } = PRESET_INFO.trefoil;
  const rng = new RandomState(13);
  const data = dtype === 'float64' ? new Float64Array(n * p) : new Float32Array(n * p);
  const labels = new Uint8Array(n);
  const sigma = 0.15;
  for (let i = 0; i < n; i++) {
    const u = rng.randomSample();
    const th = 2 * Math.PI * u;
    const o = i * p;
    data[o] = Math.sin(th) + 2 * Math.sin(2 * th) + sigma * rng.nextGauss();
    data[o + 1] = Math.cos(th) - 2 * Math.cos(2 * th) + sigma * rng.nextGauss();
    data[o + 2] = -Math.sin(3 * th) + sigma * rng.nextGauss();
    labels[i] = Math.min(9, Math.floor(u * 10));
  }
  return {
    key: `trefoil|${dtype}`,
    kind: 'trefoil',
    X: new Matrix(data, n, p),
    labels,
    n,
    p,
    dtype,
    imageSide: null,
    palette: RAINBOW,
  };
}

/** Scales `w` to unit L2 norm in place. */
function normalize(w: Float64Array): void {
  let s = 0;
  for (const x of w) {
    s += x * x;
  }
  const inv = 1 / Math.sqrt(s);
  for (let j = 0; j < w.length; j++) {
    w[j] *= inv;
  }
}

/**
 * Three concentric rings living in a random 2-D subspace of R^p, plus
 * isotropic noise on every dimension. PCA recovers the plane — two dominant
 * eigenvalues ≈ 35 against a 0.16 noise floor — and the scatter shows the
 * rings. Rows interleave rings (like makeSynthetic's clusters).
 */
function makeHiddenCircles(dtype: DemoDtype): Dataset {
  const { n, p } = PRESET_INFO.circles;
  const rng = new RandomState(17);
  const u = rng.standardNormal(new Float64Array(p));
  normalize(u);
  const v = rng.standardNormal(new Float64Array(p));
  let dot = 0;
  for (let j = 0; j < p; j++) {
    dot += v[j] * u[j];
  }
  for (let j = 0; j < p; j++) {
    v[j] -= dot * u[j];
  }
  normalize(v);
  const radii = [1, 2.5, 4];
  const data = dtype === 'float64' ? new Float64Array(n * p) : new Float32Array(n * p);
  const labels = new Uint8Array(n);
  const noise = new Float64Array(p);
  for (let i = 0; i < n; i++) {
    const ring = i % 3;
    labels[i] = ring;
    const phi = 2 * Math.PI * rng.randomSample();
    const r = 3 * (radii[ring] + 0.08 * rng.nextGauss());
    const c1 = r * Math.cos(phi);
    const c2 = r * Math.sin(phi);
    rng.standardNormal(noise);
    const o = i * p;
    for (let j = 0; j < p; j++) {
      data[o + j] = c1 * u[j] + c2 * v[j] + 0.4 * noise[j];
    }
  }
  return {
    key: `circles|${dtype}`,
    kind: 'circles',
    X: new Matrix(data, n, p),
    labels,
    n,
    p,
    dtype,
    imageSide: null,
  };
}

/**
 * Three tight Gaussian blobs (interleaved rows, labels 0–2) plus 3%
 * uniform-box outliers appended at the end (label 3). Built for the
 * scoreSamples panel: the outliers own the histogram's low-likelihood tail.
 */
function makeClustersOutliers(dtype: DemoDtype): Dataset {
  const { n, p } = PRESET_INFO.outliers;
  const nOut = 90;
  const nIn = n - nOut;
  const rng = new RandomState(19);
  const centers = rng.standardNormal(new Float64Array(3 * p));
  for (let i = 0; i < centers.length; i++) {
    centers[i] *= 6;
  }
  const data = dtype === 'float64' ? new Float64Array(n * p) : new Float32Array(n * p);
  const labels = new Uint8Array(n);
  const scratch = new Float64Array(p);
  for (let i = 0; i < nIn; i++) {
    const c = i % 3;
    labels[i] = c;
    rng.standardNormal(scratch);
    const co = c * p;
    const ro = i * p;
    for (let j = 0; j < p; j++) {
      data[ro + j] = centers[co + j] + scratch[j];
    }
  }
  for (let i = nIn; i < n; i++) {
    labels[i] = 3;
    rng.uniform(-12, 12, scratch);
    data.set(scratch, i * p);
  }
  return {
    key: `outliers|${dtype}`,
    kind: 'outliers',
    X: new Matrix(data, n, p),
    labels,
    n,
    p,
    dtype,
    imageSide: null,
  };
}

export interface PresetInfo {
  n: number;
  p: number;
  blurb: string;
  make: (dtype: DemoDtype) => Dataset;
}

/** The fixed-parameter presets: dims, data-status blurb, and generator. */
export const PRESET_INFO: Record<PresetKind, PresetInfo> = {
  swissroll: {
    n: 3000,
    p: 3,
    blurb: 'swiss roll: t-colored 3-D spiral; PC1×PC2 shows it face-on',
    make: makeSwissRoll,
  },
  trefoil: {
    n: 3000,
    p: 3,
    blurb: 'trefoil knot: θ-colored; PC1×PC2 is the classic 2-D projection',
    make: makeTrefoil,
  },
  circles: {
    n: 3000,
    p: 100,
    blurb:
      '3 concentric rings hidden in a random 2-D subspace of R^100 + noise; scree shows 2 dominant PCs',
    make: makeHiddenCircles,
  },
  outliers: {
    n: 3090,
    p: 30,
    blurb: '3 tight Gaussian blobs + 3% uniform outliers (label 3); see the scoreSamples histogram',
    make: makeClustersOutliers,
  },
};

export function isPresetKind(v: string): v is PresetKind {
  return v in PRESET_INFO;
}
