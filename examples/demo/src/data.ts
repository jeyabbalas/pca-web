/** Datasets for the demo: bundled sklearn digits and synthetic Gaussian clusters. */
import { Matrix, RandomState } from 'pca-web';

export type DemoDtype = 'float64' | 'float32';

export interface Dataset {
  /** Cache key of the configuration that produced this dataset. */
  key: string;
  kind: 'digits' | 'synthetic';
  X: Matrix;
  /** Digit 0–9 or cluster id per row. */
  labels: Uint8Array;
  n: number;
  p: number;
  dtype: DemoDtype;
  /** Rows are side×side images (8 for digits); null when not image-shaped. */
  imageSide: number | null;
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
