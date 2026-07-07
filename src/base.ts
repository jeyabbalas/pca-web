/**
 * Shared machinery for PCA and IncrementalPCA — the counterpart of sklearn's
 * `_BasePCA`: transform / inverseTransform / getCovariance / getPrecision /
 * getFeatureNamesOut operating on the fitted state.
 */

import { asMatrix, Matrix, type MatrixInput } from './matrix.js';
import { matmul, matmulTransB } from './numeric/blas.js';
import { inverse } from './numeric/lu.js';
import { type Dtype, dtypeOf, epsFor, type FloatArray } from './types.js';
import { assertAllFinite, checkFeatureCount, NotFittedError } from './validation.js';

export function castTo(a: Float64Array, dtype: Dtype): FloatArray {
  return dtype === 'float64' ? a : Float32Array.from(a);
}

export function promoteDtype(a: Dtype, b: Dtype): Dtype {
  return a === 'float64' || b === 'float64' ? 'float64' : 'float32';
}

export abstract class BasePCA {
  protected readonly estimatorName: string = 'PCA';

  protected fitted = false;
  protected dtype: Dtype = 'float64';
  protected whitenOpt = false;

  protected components_: Matrix | null = null;
  protected explainedVariance_: FloatArray | null = null;
  protected explainedVarianceRatio_: FloatArray | null = null;
  protected singularValues_: FloatArray | null = null;
  protected mean_: FloatArray | null = null;
  protected nComponents_ = 0;
  protected nFeaturesIn_ = 0;
  protected noiseVariance_ = 0;

  protected assertFitted(): void {
    if (!this.fitted) {
      throw new NotFittedError(this.estimatorName);
    }
  }

  // ------------------------------------------------------------------
  // Fitted attributes (sklearn's trailing-underscore attributes)
  // ------------------------------------------------------------------

  /** Principal axes (nComponents × nFeatures) — sklearn's `components_`. */
  get components(): Matrix {
    this.assertFitted();
    return this.components_ as Matrix;
  }

  /** sklearn's `explained_variance_`. */
  get explainedVariance(): FloatArray {
    this.assertFitted();
    return this.explainedVariance_ as FloatArray;
  }

  /** sklearn's `explained_variance_ratio_`. */
  get explainedVarianceRatio(): FloatArray {
    this.assertFitted();
    return this.explainedVarianceRatio_ as FloatArray;
  }

  /** sklearn's `singular_values_`. */
  get singularValues(): FloatArray {
    this.assertFitted();
    return this.singularValues_ as FloatArray;
  }

  /** Per-feature training mean — sklearn's `mean_`. */
  get mean(): FloatArray {
    this.assertFitted();
    return this.mean_ as FloatArray;
  }

  /** Estimated number of components — sklearn's `n_components_`. */
  get nComponents(): number {
    this.assertFitted();
    return this.nComponents_;
  }

  /** sklearn's `n_features_in_`. */
  get nFeaturesIn(): number {
    this.assertFitted();
    return this.nFeaturesIn_;
  }

  /** Tipping–Bishop noise variance — sklearn's `noise_variance_`. */
  get noiseVariance(): number {
    this.assertFitted();
    return this.noiseVariance_;
  }

  /** Whether whitening is enabled (constructor option). */
  get whiten(): boolean {
    return this.whitenOpt;
  }

  // ------------------------------------------------------------------
  // Transforms
  // ------------------------------------------------------------------

  /** Project X onto the principal components — sklearn's `transform`. */
  transform(X: MatrixInput): Matrix {
    this.assertFitted();
    const xm = asMatrix(X);
    assertAllFinite(xm, `${this.estimatorName}.transform`);
    checkFeatureCount(xm, this.nFeaturesIn_, this.estimatorName);
    return this.transformCore(xm, false);
  }

  protected transformCore(X: Matrix, xIsCentered: boolean): Matrix {
    const comp = this.components_ as Matrix;
    // Project first, then remove the projected mean — sklearn's operation
    // order (avoids centering a copy of X).
    const xt = matmulTransB(X.data, comp.data, X.rows, X.cols, comp.rows);
    return this._transformFromProjection(xt, X.rows, X.dtype, xIsCentered);
  }

  /**
   * @internal Everything in transform downstream of the X·componentsᵀ
   * product (mean-projection removal, whitening, dtype cast). The WebGPU
   * frontend calls this with a device-computed projection; `xt` is consumed.
   */
  _transformFromProjection(
    xt: Float64Array,
    n: number,
    xDtype: Dtype,
    xIsCentered: boolean,
  ): Matrix {
    const comp = this.components_ as Matrix;
    const ev = this.explainedVariance_ as FloatArray;
    const meanArr = this.mean_ as FloatArray;
    const p = comp.cols;
    const k = comp.rows;
    if (!xIsCentered) {
      const meanProj = new Float64Array(k);
      for (let c = 0; c < k; c++) {
        let acc = 0;
        const off = c * p;
        for (let j = 0; j < p; j++) {
          acc += meanArr[j] * comp.data[off + j];
        }
        meanProj[c] = acc;
      }
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < k; c++) {
          xt[i * k + c] -= meanProj[c];
        }
      }
    }
    if (this.whitenOpt) {
      // Clip near-zero variances at dtype eps, as sklearn does, so
      // rank-deficient data cannot produce non-finite whitened output.
      const minScale = epsFor(dtypeOf(ev));
      for (let c = 0; c < k; c++) {
        let scale = Math.sqrt(ev[c]);
        if (scale < minScale) {
          scale = minScale;
        }
        for (let i = 0; i < n; i++) {
          xt[i * k + c] /= scale;
        }
      }
    }
    return new Matrix(castTo(xt, promoteDtype(xDtype, this.dtype)), n, k);
  }

  /** Map component-space data back to feature space — sklearn's `inverse_transform`. */
  inverseTransform(X: MatrixInput): Matrix {
    this.assertFitted();
    const xm = asMatrix(X);
    const comp = this.components_ as Matrix;
    const ev = this.explainedVariance_ as FloatArray;
    const meanArr = this.mean_ as FloatArray;
    const k = comp.rows;
    const p = comp.cols;
    if (xm.cols !== k) {
      throw new Error(
        `X has ${xm.cols} features, but inverseTransform expects ${k} (= nComponents)`,
      );
    }
    const n = xm.rows;
    let compData: FloatArray = comp.data;
    if (this.whitenOpt) {
      const scaled = new Float64Array(k * p);
      for (let c = 0; c < k; c++) {
        const f = Math.sqrt(ev[c]);
        const off = c * p;
        for (let j = 0; j < p; j++) {
          scaled[off + j] = f * comp.data[off + j];
        }
      }
      compData = scaled;
    }
    const out = matmul(xm.data, compData, n, k, p);
    for (let i = 0; i < n; i++) {
      const off = i * p;
      for (let j = 0; j < p; j++) {
        out[off + j] += meanArr[j];
      }
    }
    return new Matrix(castTo(out, promoteDtype(xm.dtype, this.dtype)), n, p);
  }

  // ------------------------------------------------------------------
  // Generative model: covariance and precision
  // ------------------------------------------------------------------

  /** Model covariance `componentsᵀ diag(ev − nv) components + nv·I` — sklearn's `get_covariance`. */
  getCovariance(): Matrix {
    this.assertFitted();
    const p = this.nFeaturesIn_;
    return new Matrix(castTo(this.covarianceF64(), this.dtype), p, p);
  }

  protected covarianceF64(): Float64Array {
    const comp = this.components_ as Matrix;
    const ev = this.explainedVariance_ as FloatArray;
    const nv = this.noiseVariance_;
    const k = comp.rows;
    const p = comp.cols;
    // w = components (rows scaled by sqrt(ev) when whitening, like sklearn);
    // scaled = diag(exp_var_diff) @ w, so cov = scaledᵀ @ w.
    const w = new Float64Array(k * p);
    const scaled = new Float64Array(k * p);
    for (let c = 0; c < k; c++) {
      const f = this.whitenOpt ? Math.sqrt(ev[c]) : 1;
      const evd = ev[c] > nv ? ev[c] - nv : 0;
      const off = c * p;
      for (let j = 0; j < p; j++) {
        const wv = comp.data[off + j] * f;
        w[off + j] = wv;
        scaled[off + j] = wv * evd;
      }
    }
    const cov = new Float64Array(p * p);
    for (let c = 0; c < k; c++) {
      const offW = c * p;
      for (let i = 0; i < p; i++) {
        const sv = scaled[offW + i];
        if (sv !== 0) {
          const covOff = i * p;
          for (let j = 0; j < p; j++) {
            cov[covOff + j] += sv * w[offW + j];
          }
        }
      }
    }
    for (let i = 0; i < p; i++) {
      cov[i * p + i] += nv;
    }
    return cov;
  }

  /** Model precision via the matrix inversion lemma — sklearn's `get_precision`. */
  getPrecision(): Matrix {
    this.assertFitted();
    const p = this.nFeaturesIn_;
    return new Matrix(castTo(this.precisionF64(), this.dtype), p, p);
  }

  protected precisionF64(): Float64Array {
    const p = this.nFeaturesIn_;
    const k = this.nComponents_;
    const nv = this.noiseVariance_;
    if (k === 0) {
      const eye = new Float64Array(p * p);
      for (let i = 0; i < p; i++) {
        eye[i * p + i] = 1 / nv;
      }
      return eye;
    }
    if (nv === 0) {
      return inverse(this.covarianceF64(), p);
    }
    const comp = this.components_ as Matrix;
    const ev = this.explainedVariance_ as FloatArray;
    const w = new Float64Array(k * p);
    for (let c = 0; c < k; c++) {
      const f = this.whitenOpt ? Math.sqrt(ev[c]) : 1;
      const off = c * p;
      for (let j = 0; j < p; j++) {
        w[off + j] = comp.data[off + j] * f;
      }
    }
    // inner = w wᵀ / nv + diag(1 / (ev − nv))
    const inner = new Float64Array(k * k);
    for (let a = 0; a < k; a++) {
      for (let b = 0; b < k; b++) {
        let acc = 0;
        const offA = a * p;
        const offB = b * p;
        for (let j = 0; j < p; j++) {
          acc += w[offA + j] * w[offB + j];
        }
        inner[a * k + b] = acc / nv;
      }
    }
    for (let c = 0; c < k; c++) {
      const evd = ev[c] > nv ? ev[c] - nv : 0;
      inner[c * k + c] += 1 / evd;
    }
    const innerInv = inverse(inner, k);
    // precision = −(wᵀ innerInv w) / nv² + I/nv
    const tmp = matmul(innerInv, w, k, k, p); // k×p
    const prec = new Float64Array(p * p);
    for (let c = 0; c < k; c++) {
      const offW = c * p;
      for (let i = 0; i < p; i++) {
        const wv = w[offW + i];
        if (wv !== 0) {
          const off = i * p;
          for (let j = 0; j < p; j++) {
            prec[off + j] += wv * tmp[offW + j];
          }
        }
      }
    }
    const nv2 = nv * nv;
    for (let i = 0; i < p * p; i++) {
      prec[i] /= -nv2;
    }
    for (let i = 0; i < p; i++) {
      prec[i * p + i] += 1 / nv;
    }
    return prec;
  }

  /** Output feature names, e.g. ['pca0', 'pca1', …] — sklearn's `get_feature_names_out`. */
  getFeatureNamesOut(): string[] {
    this.assertFitted();
    const prefix = this.estimatorName.toLowerCase();
    const names = new Array<string>(this.nComponents_);
    for (let i = 0; i < this.nComponents_; i++) {
      names[i] = `${prefix}${i}`;
    }
    return names;
  }
}
