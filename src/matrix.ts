import { constructorFor, type Dtype, dtypeOf, type FloatArray } from './types.js';

/**
 * A dense row-major matrix backed by a Float64Array or Float32Array.
 *
 * This is the container returned by all pca-web methods. It is a thin,
 * zero-copy wrapper: `data[i * cols + j]` is the element at row `i`,
 * column `j`.
 */
export class Matrix {
  readonly data: FloatArray;
  readonly rows: number;
  readonly cols: number;

  /**
   * Wraps an existing typed array (no copy). `data.length` must equal
   * `rows * cols`.
   */
  constructor(data: FloatArray, rows: number, cols: number) {
    if (data.length !== rows * cols) {
      throw new Error(
        `Matrix: data length ${data.length} does not match shape ${rows}x${cols} (= ${rows * cols})`,
      );
    }
    this.data = data;
    this.rows = rows;
    this.cols = cols;
  }

  /** Storage dtype: 'float64' or 'float32'. */
  get dtype(): Dtype {
    return dtypeOf(this.data);
  }

  /** Allocates a zero-initialized matrix. */
  static zeros(rows: number, cols: number, dtype: Dtype = 'float64'): Matrix {
    const Ctor = constructorFor(dtype);
    return new Matrix(new Ctor(rows * cols), rows, cols);
  }

  /**
   * Builds a matrix from an array of rows, e.g. `Matrix.from2d([[1, 2], [3, 4]])`.
   * All rows must have equal length.
   */
  static from2d(rows2d: ArrayLike<number>[], dtype: Dtype = 'float64'): Matrix {
    const rows = rows2d.length;
    if (rows === 0) {
      throw new Error('Matrix.from2d: at least one row is required');
    }
    const cols = rows2d[0].length;
    const Ctor = constructorFor(dtype);
    const data = new Ctor(rows * cols);
    for (let i = 0; i < rows; i++) {
      const r = rows2d[i];
      if (r.length !== cols) {
        throw new Error(
          `Matrix.from2d: row ${i} has length ${r.length}, expected ${cols} (rows must be equal length)`,
        );
      }
      for (let j = 0; j < cols; j++) {
        data[i * cols + j] = r[j];
      }
    }
    return new Matrix(data, rows, cols);
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, v: number): void {
    this.data[i * this.cols + j] = v;
  }

  /** Returns a deep copy (same dtype). */
  copy(): Matrix {
    return new Matrix(this.data.slice(), this.rows, this.cols);
  }

  /** Converts to a plain nested array of numbers. */
  toArray(): number[][] {
    const out: number[][] = new Array(this.rows);
    for (let i = 0; i < this.rows; i++) {
      const row = new Array<number>(this.cols);
      for (let j = 0; j < this.cols; j++) {
        row[j] = this.data[i * this.cols + j];
      }
      out[i] = row;
    }
    return out;
  }

  /** Copy of row `i` as a plain typed array (same dtype). */
  row(i: number): FloatArray {
    return this.data.slice(i * this.cols, (i + 1) * this.cols);
  }
}

/** Accepted input type for training/transform data. */
export type MatrixInput = Matrix | ArrayLike<number>[];

/**
 * Coerces user input to a Matrix without copying when it is already one.
 * Nested arrays become float64 (mirrors sklearn coercing non-float dtypes
 * to float64).
 */
export function asMatrix(X: MatrixInput): Matrix {
  if (X instanceof Matrix) {
    return X;
  }
  if (Array.isArray(X)) {
    return Matrix.from2d(X);
  }
  throw new TypeError(
    'Expected a Matrix or an array of rows (number[][]); ' +
      'wrap typed-array data with new Matrix(data, rows, cols)',
  );
}
