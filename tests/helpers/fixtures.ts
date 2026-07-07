import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Matrix } from '../../src/matrix.js';
import type { FloatArray } from '../../src/types.js';

export interface ArrayRef {
  dtype: 'float64' | 'float32';
  shape: number[];
  offset: number;
}

export interface FixtureCase {
  id: string;
  params: Record<string, unknown>;
  arrays: Record<string, ArrayRef>;
  scalars: Record<string, unknown>;
  flags: Record<string, unknown>;
}

export interface Suite {
  meta: Record<string, unknown>;
  cases: FixtureCase[];
  bin: Uint8Array;
}

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

export function loadSuite(name: string): Suite {
  const dir = join(ROOT, 'fixtures', name);
  const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));
  const bin = new Uint8Array(readFileSync(join(dir, 'data.bin')));
  return { meta: manifest.meta, cases: manifest.cases, bin };
}

/** Materializes an array ref as a typed array (copied, so alignment is safe). */
export function getArray(suite: Suite, ref: ArrayRef): FloatArray {
  const count = ref.shape.reduce((a, b) => a * b, 1);
  const bytes = count * (ref.dtype === 'float64' ? 8 : 4);
  const slice = suite.bin.slice(ref.offset, ref.offset + bytes);
  return ref.dtype === 'float64'
    ? new Float64Array(slice.buffer, 0, count)
    : new Float32Array(slice.buffer, 0, count);
}

export function getMatrix(suite: Suite, ref: ArrayRef): Matrix {
  if (ref.shape.length !== 2) {
    throw new Error(`expected 2-d array, got shape ${ref.shape}`);
  }
  return new Matrix(getArray(suite, ref), ref.shape[0], ref.shape[1]);
}

/** Parses fixture scalars ("inf"/"-inf"/"nan" encode non-finite floats). */
export function num(v: unknown): number {
  if (typeof v === 'number') {
    return v;
  }
  if (v === 'inf') {
    return Number.POSITIVE_INFINITY;
  }
  if (v === '-inf') {
    return Number.NEGATIVE_INFINITY;
  }
  if (v === 'nan') {
    return Number.NaN;
  }
  throw new Error(`not a numeric scalar: ${JSON.stringify(v)}`);
}
