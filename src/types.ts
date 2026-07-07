/** Floating-point storage used throughout the library. */
export type FloatArray = Float64Array | Float32Array;

/** Storage dtype. Mirrors sklearn's float64/float32 handling. */
export type Dtype = 'float64' | 'float32';

export type FloatArrayConstructor = Float64ArrayConstructor | Float32ArrayConstructor;

export function dtypeOf(a: FloatArray): Dtype {
  return a instanceof Float64Array ? 'float64' : 'float32';
}

export function constructorFor(dtype: Dtype): FloatArrayConstructor {
  return dtype === 'float64' ? Float64Array : Float32Array;
}

/** Machine epsilon for a dtype (numpy finfo(dtype).eps). */
export function epsFor(dtype: Dtype): number {
  return dtype === 'float64' ? 2.220446049250313e-16 : 1.1920929e-7;
}
