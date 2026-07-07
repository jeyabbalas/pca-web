/**
 * WGSL GEMM kernels with near-f64 accuracy on f32-only hardware.
 *
 * Design constraint discovered empirically (Chrome/Dawn on Metal): the MSL
 * compiler applies fast-math — classic Dekker/Knuth error-free
 * transformations (two_sum, split) are algebraically folded to zero, and
 * multiply-add pairs are contracted into fma, so double-single (df64)
 * accumulation silently degrades to f32. What DOES survive, verified by
 * on-device probes:
 *   - fma(a, b, -a*b) computes the exact f32 multiplication error
 *   - single float ops are correctly rounded (no cross-op guarantees)
 *   - i32 conversions and integer adds are exact and untouched by fast-math
 *
 * So products use fma-exact two_prod, and ACCUMULATION is exact integer
 * binning: every product term is decomposed — exactly, via power-of-two
 * scaling and fma residuals — into 5 levels of 13-bit integers on a fixed
 * per-dispatch grid (exponent E bounds every |a·b| term), accumulated in
 * i32 (no rounding at all), and recombined in float64 on the CPU. The only
 * error is the truncated tail below the last grid level:
 *   |error| ≤ K·7·2^(E-65) ≤ maxProduct·2^-45
 * comparable to true f64 accumulation. Inputs are (hi, lo) f32 pairs of the
 * f64 values; the three significant cross products (+ the tiny lo·lo term)
 * feed the binner, preserving ~48 significant bits end to end.
 *
 * One tiled kernel template covers all four op(A)·op(B) transpose variants;
 * only the injected index expressions differ.
 */

export const TILE = 16;

/**
 * Per-level bin capacity is 2^13; with ≤7 terms per MAC each contributing
 * |n| ≤ 2^13 per level, i32 overflows at K ≈ 2^31/(7·2^13) ≈ 37k. The
 * engine splits the K dimension at this bound and sums chunks in f64.
 */
export const MAX_K_PER_DISPATCH = 30000;

/** i32 bin levels per output element (13 bits each → 65 bits of grid). */
export const BIN_LEVELS = 5;

/** i32 slots per output element in the C buffer (BIN_LEVELS padded to 8). */
export const OUT_SLOTS = 8;

/**
 * Builds the GEMM shader for C(M×N) = op(A)·op(B) over K = [kStart, kStart+kCount).
 *
 * Physical shapes: A is M×K_total when !ta, K_total×M when ta; B is
 * K_total×N when !tb, N×K_total when tb. All row-major, ds-encoded
 * (array<vec2f> of hi/lo pairs). C is array<i32>, OUT_SLOTS per element.
 */
export function gemmShader(ta: boolean, tb: boolean): string {
  // Element (i, l) of op(A) and (l, j) of op(B) in physical storage
  // (kTotal is the stride of the K dimension in the untransposed layout).
  const aIndex = ta ? 'l * dims.m + i' : 'i * dims.kTotal + l';
  const bIndex = tb ? 'j * dims.kTotal + l' : 'l * dims.n + j';
  return /* wgsl */ `
struct Dims {
  m: u32,
  n: u32,
  kTotal: u32,
  kStart: u32,
  kCount: u32,
  e: i32,      // grid exponent: every |product term| < 2^e
  pad0: u32,
  pad1: u32,
}

@group(0) @binding(0) var<storage, read> matA: array<vec2f>;
@group(0) @binding(1) var<storage, read> matB: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> matC: array<i32>;
@group(0) @binding(3) var<uniform> dims: Dims;

var<workgroup> tileA: array<vec2f, ${TILE * TILE}>;
var<workgroup> tileB: array<vec2f, ${TILE * TILE}>;

fn loadA(i: u32, l: u32) -> vec2f {
  if (i < dims.m && l < dims.kStart + dims.kCount) {
    return matA[${aIndex}];
  }
  return vec2f(0.0, 0.0);
}

fn loadB(l: u32, j: u32) -> vec2f {
  if (l < dims.kStart + dims.kCount && j < dims.n) {
    return matB[${bIndex}];
  }
  return vec2f(0.0, 0.0);
}

// Decomposes x exactly into 13-bit integers on grids 2^(e-13)…2^(e-65) and
// adds them to the bins. Every step is exact: power-of-two scaling shifts
// the exponent only, i32() truncates exactly, and the fma residual is
// representable (bits of x strictly below the level's grid).
fn binAdd(x: f32, acc: ptr<function, array<i32, ${BIN_LEVELS}>>) {
  var r = x;
  for (var lvl = 0; lvl < ${BIN_LEVELS}; lvl++) {
    let g = ldexp(1.0, dims.e - 13 * (lvl + 1));
    let n = i32(r * ldexp(1.0, 13 * (lvl + 1) - dims.e));
    (*acc)[lvl] += n;
    r = fma(-f32(n), g, r);
  }
}

@compute @workgroup_size(${TILE}, ${TILE})
fn main(
  @builtin(workgroup_id) wg: vec3u,
  @builtin(local_invocation_id) lid: vec3u,
) {
  let row = wg.y * ${TILE}u + lid.y;
  let col = wg.x * ${TILE}u + lid.x;
  var acc: array<i32, ${BIN_LEVELS}>;
  for (var lvl = 0; lvl < ${BIN_LEVELS}; lvl++) {
    acc[lvl] = 0;
  }
  let numTiles = (dims.kCount + ${TILE}u - 1u) / ${TILE}u;
  for (var t = 0u; t < numTiles; t++) {
    let kA = dims.kStart + t * ${TILE}u + lid.x;
    let kB = dims.kStart + t * ${TILE}u + lid.y;
    tileA[lid.y * ${TILE}u + lid.x] = loadA(row, kA);
    tileB[lid.y * ${TILE}u + lid.x] = loadB(kB, col);
    workgroupBarrier();
    for (var m = 0u; m < ${TILE}u; m++) {
      let a = tileA[lid.y * ${TILE}u + m];
      let b = tileB[m * ${TILE}u + lid.x];
      // Exact ds cross products via fma (verified exact on-device).
      let p1 = a.x * b.x;
      let e1 = fma(a.x, b.x, -p1);
      let p2 = a.x * b.y;
      let e2 = fma(a.x, b.y, -p2);
      let p3 = a.y * b.x;
      let e3 = fma(a.y, b.x, -p3);
      let p4 = a.y * b.y; // ≤ 2^-48·|a·b| — plain product suffices
      binAdd(p1, &acc);
      binAdd(e1, &acc);
      binAdd(p2, &acc);
      binAdd(e2, &acc);
      binAdd(p3, &acc);
      binAdd(e3, &acc);
      binAdd(p4, &acc);
    }
    workgroupBarrier();
  }
  if (row < dims.m && col < dims.n) {
    let base = (row * dims.n + col) * ${OUT_SLOTS}u;
    for (var lvl = 0u; lvl < ${BIN_LEVELS}u; lvl++) {
      matC[base + lvl] = acc[lvl];
    }
  }
}
`;
}
